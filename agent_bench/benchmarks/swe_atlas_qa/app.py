from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SWEAtlasQAHarnessMixin:
    """Shared SWE-Atlas QA behavior layered onto existing agent harnesses."""

    agent_id = "swe_atlas_qa"
    task_workspace = "/app"
    requires_swebench_package = False
    task_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    def _resolve_container_image_for_instance(
        self,
        *,
        instance_id: str,
        container_image: str,
        instance_data: Optional[Dict] = None,
    ) -> str:
        if instance_data and instance_data.get("docker_image"):
            return instance_data["docker_image"]
        return container_image

    def run_tests(
        self,
        container_id: Optional[str],
        instance_id: str,
        workspace_root: str,
        timeout_seconds: int,
        patch: Optional[str] = None,
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
        run_dir: Optional[Path] = None,
        **_extra,
    ) -> Dict:
        if instance_data is None:
            return {
                "status": "ERROR",
                "test_pass_rate": 0.0,
                "passed": 0,
                "failed": 0,
                "error_message": "SWE-Atlas-QnA scoring requires instance_data",
            }

        run_id = run_id or instance_id

        # 1. Try to read /logs/agent/answer.txt from the task container
        from agent_bench.benchmarks.swebench.app import _oc_sandbox_name
        from agent_bench.deployments.containers import ContainerHandle

        rid_short = run_id[:12]
        sandbox_scope = "agent"
        scope_key = f"agent:{self.agent_id}"
        sandbox_prefix = f"oc-sbx-{rid_short}-"
        task_container_name = _oc_sandbox_name(sandbox_prefix, scope_key)

        h = ContainerHandle(
            id=task_container_name,
            name=task_container_name,
            backend=self.container.backend_id,
        )

        answer_text = ""
        try:
            res = self.container.exec(h, ["cat", "/logs/agent/answer.txt"], check=False)
            if res.returncode == 0 and res.stdout.strip():
                answer_text = res.stdout
        except Exception as e:
            logger.warning(f"Could not read /logs/agent/answer.txt from task container: {e}")

        # 2. Fall back to host workspace answer.md if task container file is not found/empty
        if not answer_text.strip():
            ws_host = self.workspace_base / run_id
            answer_path = ws_host / "answer.md"
            if answer_path.exists():
                try:
                    answer_text = answer_path.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Could not read fallback host answer.md: {e}")

        if not answer_text.strip():
            return {
                "status": "FAIL",
                "test_pass_rate": 0.0,
                "passed": 0,
                "failed": 1,
                "error_message": "answer.md or /logs/agent/answer.txt not found/empty",
            }

        # 3. Extract the answer wrapped in <<FINAL_ANSWER>> tags if present
        import re
        match = re.search(r"<<FINAL_ANSWER>>([\s\S]*?)<<FINAL_ANSWER>>", answer_text)
        if match:
            answer_text = match.group(1).strip()
        else:
            answer_text = answer_text.strip()

        if not answer_text:
            return {
                "status": "FAIL",
                "test_pass_rate": 0.0,
                "passed": 0,
                "failed": 1,
                "error_message": "Extracted answer is empty",
            }

        from agent_bench.benchmarks.swe_atlas_qa.scoring import score_answer

        try:
            verdict = score_answer(instance_data, answer_text)
            if run_dir is not None:
                run_dir.mkdir(parents=True, exist_ok=True)
                archived_answer = run_dir / "answer.md"
                archived_answer.write_text(answer_text, encoding="utf-8")
                verdict["answer_path"] = str(archived_answer)
            return verdict
        except Exception as e:
            logger.exception("LLM grading failed")
            return {
                "status": "ERROR",
                "test_pass_rate": 0.0,
                "passed": 0,
                "failed": 0,
                "error_message": f"LLM grading failed: {e}",
            }
