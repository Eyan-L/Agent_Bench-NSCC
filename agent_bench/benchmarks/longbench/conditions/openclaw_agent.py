"""LongBench OpenClaw agent condition.

A LongBench *instance* is a whole subtask, so this condition processes its
``instance_data["examples"]`` CONCURRENTLY (ThreadPoolExecutor, LONGBENCH_CONCURRENCY)
on the warm OC + sandbox containers. Each example is isolated in its OWN ClawRAG
collection (``lb_<run_id>_<i>``) and its own ``/workspace/ctx_<i>.md`` file, so they
don't collide and can run in parallel. Per example:

  1. ingest the example's ``context`` into its collection (docker exec curl into
     the task container; ClawRAG runs in-process there on localhost:8080).
  2. invoke the OpenClaw agent with the question + its collection name; the agent
     retrieves via ``/workspace/clawrag_search.py "<q>" 5 <collection>`` and answers.
  3. parse the agent's answer and record a prediction.

Predictions go to ``<workspace_host>/predictions.jsonl`` (== sandbox /workspace) so
the harness ``run_tests`` scores them with the LongBench metric. RAG-only: context
never goes into the prompt.
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition

logger = logging.getLogger(__name__)


class LongBenchOpenClawCondition(OpenClawAgentCondition):
    agent_id = "longbench_rag"

    # ---- ClawRAG ingest via `docker exec` into the task container ----------
    # ClawRAG runs INSIDE the task container (in-process chroma) on localhost:8080,
    # so ingest is curl executed inside that container, not host HTTP.
    def _exec_curl(self, args: List[str], timeout: int = 180) -> Dict:
        h = self._env_info["_harness"]
        task = self._env_info["task_container_name"]
        return h.exec_in_container(task, ["curl", "-s", "-m", "120", *args], timeout_seconds=timeout)

    def _cr_create(self, coll: str) -> None:
        self._exec_curl(["-X", "POST", "http://localhost:8080/api/v1/rag/collections",
                         "-F", f"collection_name={coll}"])

    def _cr_delete(self, coll: str) -> None:
        self._exec_curl(["-X", "DELETE", f"http://localhost:8080/api/v1/rag/collections/{coll}"])

    def _cr_ingest(self, coll: str, text: str, ws_host: Path, fname: str) -> int:
        # Per-example file (kept, NOT overwritten) so examples ingest in parallel;
        # uploaded into its own collection.
        (ws_host / fname).write_text(text, encoding="utf-8")
        r = self._exec_curl([
            "-X", "POST", "http://localhost:8080/api/v1/rag/documents/upload",
            "-F", f"files=@/workspace/{fname}",
            "-F", f"collection_name={coll}",
            "-F", "chunk_size=512", "-F", "chunk_overlap=128",
        ], timeout=600)
        try:
            return int(json.loads(r.get("stdout") or "{}").get("total_chunks", 0))
        except Exception:
            return 0

    # ---- per-example agent message ----------------------------------------
    @staticmethod
    def _example_message(ex: Dict, collection: str) -> str:
        return (
            "Use the `clawrag_retrieve` skill to answer this question from a "
            "knowledge base. You MUST complete it via the skill:\n"
            "  1. Load the skill: read /workspace/skills/clawrag_retrieve/SKILL.md\n"
            "  2. Follow it to retrieve passages for THIS collection by running "
            "(exactly, keeping this collection name):\n"
            f'       python3 /workspace/clawrag_search.py "<your search query>" 5 {collection}\n'
            "It prints the top passages. Then answer the question as concisely as "
            "possible using ONLY those passages. Output ONLY the final answer text — "
            "no explanation, no preamble.\n\n"
            f"Question: {ex['input']}"
        )

    # ---- per-agent invoke (distinct agent => distinct, isolated session) ----
    # We use one agent per example (longbench_rag_<i>) so concurrent sessions don't
    # collapse onto a single shared agent session. Lean docker-exec of the OC CLI
    # (always docker mode here); parsing reuses the base helpers.
    def _run_agent(self, container_id: str, agent_id: str, session_id: str,
                   message: str, timeout_seconds: int) -> Tuple[Optional[Dict], str]:
        inner = (
            "mkdir -p /root/.openclaw/logs && source /tmp/oc_gateway_env && "
            f"exec openclaw agent --agent {shlex.quote(agent_id)} "
            f"--session-id {shlex.quote(session_id)} "
            f"--message {shlex.quote(message)} --timeout {int(timeout_seconds)} --json"
        )
        try:
            proc = subprocess.run(
                ["docker", "exec", container_id, "bash", "-lc", inner],
                capture_output=True, text=True, timeout=int(timeout_seconds) + 120,
            )
        except subprocess.TimeoutExpired:
            return None, "openclaw agent wall-clock timeout"
        if proc.returncode != 0:
            return None, f"openclaw exit {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}"
        payload = self._extract_last_json(proc.stderr) or self._extract_last_json(proc.stdout)
        if payload is None:
            return None, f"no parseable JSON: {(proc.stderr or proc.stdout)[-200:]}"
        return payload, ""

    # ---- main loop ---------------------------------------------------------
    def run(
        self,
        instance_data: Dict,
        run_dir: Path,
        seed: int,
        workspace_root: str,
        container_id,
        env_info=None,
    ) -> Dict:
        self._env_info = env_info or {}
        instance_id = instance_data["instance_id"]
        examples: List[Dict] = instance_data.get("examples", [])
        collection = self._env_info["clawrag_collection"]
        ws_host = Path(self._env_info["workspace_host_path"])
        timeout_s = self.agent_config.max_time_seconds

        conc = max(1, min(len(examples) or 1, int(os.environ.get("LONGBENCH_CONCURRENCY", "5"))))
        logger.info(
            "LongBench %s: %d examples, concurrency=%d (per-example collections)",
            instance_id, len(examples), conc,
        )

        ex_meta: Dict[int, Dict] = {}

        def _do(item):
            i, ex = item
            coll = f"{collection}_{i}"
            eid = ex.get("_id") or f"ex_{i}"          # subfolder == dataset _id
            ex_dir = run_dir / "examples" / eid
            t0 = time.time()
            n_chunks = 0
            try:
                self._cr_create(coll)
                n_chunks = self._cr_ingest(coll, ex["context"], ws_host, f"ctx_{i}.md")
            except Exception as e:
                logger.warning("  ex%d ingest failed: %s", i, e)
            agent_id = f"longbench_rag_{i}"   # one agent per example -> isolated session
            step_start = time.time_ns()
            payload, err = self._run_agent(
                container_id, agent_id, f"{run_dir.name}-{i}",
                self._example_message(ex, coll), timeout_s,
            )
            step_end = time.time_ns()
            answer = self._parse_text(payload) if payload else ""
            pin, pout = self._parse_usage(payload)
            run = self._unwrap(payload)
            oc_sid = ((run.get("meta") or {}).get("agentMeta") or {}).get("sessionId")
            try:
                ex_dir.mkdir(parents=True, exist_ok=True)
                (ex_dir / "response.json").write_text(
                    json.dumps({"index": i, "_id": ex.get("_id"), "input": ex["input"],
                                "answers": ex.get("answers", []), "answer": answer,
                                "error": err, "oc_session_id": oc_sid, "payload": payload},
                               ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            ex_meta[i] = {"coll": coll, "eid": eid, "agent_id": agent_id, "sid": oc_sid,
                          "payload": payload, "start": step_start, "end": step_end}
            logger.info(
                "  ex%d done in %.1fs (id=%s coll=%s chunks=%d ans_len=%d err=%s)",
                i, time.time() - t0, eid[:12], coll, n_chunks, len(answer), err or "-",
            )
            return i, {
                "_id": ex.get("_id"),
                "input": ex["input"],
                "prediction": answer,
                "answers": ex.get("answers", []),
                "all_classes": ex.get("all_classes"),
            }, pin, pout

        preds: List[Dict] = [None] * len(examples)
        tot_in = tot_out = 0
        with ThreadPoolExecutor(max_workers=conc) as pool:
            for i, pred, pin, pout in pool.map(_do, list(enumerate(examples))):
                preds[i] = pred
                tot_in += pin
                tot_out += pout

        # Flush the gateway so session files are fully written before we archive
        # (OpenClaw flushes on gateway shutdown). All agent calls are done.
        try:
            subprocess.run(["docker", "exec", container_id, "pkill", "-TERM", "-f", "openclaw-gateway"],
                           capture_output=True, timeout=10)
            time.sleep(2)
        except Exception:
            pass

        # Per-example archive: each subfolder (named by the dataset _id) gets its
        # OWN session log + its OWN spans.jsonl (synthesized from that example's
        # session + payload — same shape as verified_full). /root/.openclaw is
        # bind-mounted to state_dir on the host; sessions are mapped via sessionId.
        state_dir = self._env_info.get("state_dir")
        for i, m in ex_meta.items():
            ex_dir = run_dir / "examples" / m["eid"]
            # Each example used its OWN agent (longbench_rag_<i>), so its session
            # is isolated under agents/<agent_id>/sessions/. Copy host-side; the
            # root-owned .trajectory.jsonl may be unreadable -> skip per-file.
            if state_dir:
                a_sess = Path(state_dir) / "agents" / m["agent_id"] / "sessions"
                if a_sess.is_dir():
                    sess_dst = ex_dir / "conversation" / "sessions"
                    sess_dst.mkdir(parents=True, exist_ok=True)
                    for src in sorted(a_sess.glob("*.jsonl")):
                        dst = ("session.trajectory.jsonl"
                               if src.name.endswith(".trajectory.jsonl") else "session.jsonl")
                        try:
                            shutil.copy2(src, sess_dst / dst)
                        except Exception as e:
                            logger.debug("  ex%d copy %s skipped: %s", i, src.name, e)
            try:
                spans_path = ex_dir / "spans.jsonl"
                self._maybe_synthesize_tool_call_spans(spans_path, ex_dir, run_dir.name)
                self._maybe_synthesize_model_usage_span(
                    spans_path, m["payload"], run_id=run_dir.name,
                    start_ns=m["start"], end_ns=m["end"],
                )
            except Exception as e:
                logger.warning("  ex%d spans synth failed: %s", i, e)

        # Clean up the per-example collections.
        for m in ex_meta.values():
            try:
                self._cr_delete(m["coll"])
            except Exception:
                pass

        # predictions.jsonl -> /workspace (so run_tests can read it) + run_dir archive.
        body = "\n".join(json.dumps(p, ensure_ascii=False) for p in preds) + "\n"
        (ws_host / "predictions.jsonl").write_text(body, encoding="utf-8")
        (run_dir / "predictions.jsonl").write_text(body, encoding="utf-8")

        step = {
            "trajectory_id": run_dir.name,
            "step_type": "LLM_INFER",
            "step_subtype": "LONGBENCH_RAG_SUBTASK_LOOP",
            "tool_name": "openclaw.agent",
            "instance_id": instance_id,
            "n_examples": len(preds),
            "prompt_tokens": tot_in,
            "completion_tokens": tot_out,
            "model_name": self.model_config.name,
        }
        step_trace_path = run_dir / "steps.jsonl"
        step_trace_path.write_text(json.dumps(step) + "\n")

        # Driver gates on candidate.patch existence; longbench is in skip_patch_gate
        # but write an empty one for uniformity.
        patch_path = run_dir / "candidate.patch"
        patch_path.write_text("")

        return {
            "step_trace_path": str(step_trace_path),
            "patch_path": str(patch_path),
            "total_prompt_tokens": tot_in,
            "total_completion_tokens": tot_out,
            "telemetry_path": None,
        }
