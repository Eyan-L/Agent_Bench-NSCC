from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_bench.core.profile_io import ProfileLoadError, load_profile, resolve_profile_ref
from agent_bench.core.spec import load_resolved_experiment


ROOT = Path(__file__).resolve().parents[1]


def test_load_profile_accepts_json_toml_yaml_and_yml(tmp_path: Path) -> None:
    files = {
        "profile.json": '{"name": "json", "nested": {"ok": true}}',
        "profile.toml": 'name = "toml"\n[nested]\nok = true\n',
        "profile.yaml": "name: yaml\nnested:\n  ok: true\n",
        "profile.yml": "name: yml\nnested:\n  ok: true\n",
    }
    for name, text in files.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        loaded = load_profile(path)
        assert loaded["nested"]["ok"] is True


def test_load_profile_rejects_missing_unsupported_and_non_mapping(tmp_path: Path) -> None:
    with pytest.raises(ProfileLoadError, match="missing profile file"):
        load_profile(tmp_path / "missing.yaml")

    unsupported = tmp_path / "profile.txt"
    unsupported.write_text("name: nope", encoding="utf-8")
    with pytest.raises(ProfileLoadError, match="unsupported profile file extension"):
        load_profile(unsupported)

    non_mapping = tmp_path / "profile.yaml"
    non_mapping.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ProfileLoadError, match="top-level mapping"):
        load_profile(non_mapping)


def test_yaml_profile_comments_and_nested_mappings_load(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(
        "# human-readable profile\n"
        "harness: openclaw\n"
        "openclaw:\n"
        "  diagnostics:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )

    assert load_profile(path)["openclaw"]["diagnostics"]["enabled"] is True


def test_resolve_profile_ref_prefers_yaml_before_json_and_allows_explicit_suffix(
    tmp_path: Path,
) -> None:
    base = tmp_path / "profiles" / "serving"
    (base / "vllm").mkdir(parents=True)
    yaml_path = base / "vllm" / "default.yaml"
    json_path = base / "vllm" / "default.json"
    yaml_path.write_text("serving:\n  backend: vllm\n", encoding="utf-8")
    json_path.write_text('{"serving": {"backend": "legacy"}}', encoding="utf-8")

    assert resolve_profile_ref("vllm/default", base, [".yaml", ".json"]) == yaml_path.resolve()
    assert resolve_profile_ref("vllm/default.json", base, [".yaml", ".json"]) == json_path.resolve()


def test_run_id_is_stable_across_equivalent_json_and_yaml_harness(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "sweep_name": "phase0_sanity",
                "plan_version": "v1.0",
                "application": "swebench",
                "instances": ["case"],
                "repo_mapping": {"case": {"repo_id": "repo/case", "base_commit": "abc"}},
                "container_image": "image:latest",
                "wall_clock_budget_seconds": 1800,
                "output_dir": "phase0",
            }
        ),
        encoding="utf-8",
    )
    harness_json = tmp_path / "harness.json"
    harness_json.write_text(
        json.dumps(
            {
                "harness": "openclaw",
                "conditions": [
                    {
                        "condition_id": "openclaw_agent",
                        "condition_type": "openclaw_agent",
                        "description": "agent",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    harness_yaml = tmp_path / "harness.yaml"
    harness_yaml.write_text(
        "harness: openclaw\n"
        "conditions:\n"
        "  - condition_id: openclaw_agent\n"
        "    condition_type: openclaw_agent\n"
        "    description: agent\n",
        encoding="utf-8",
    )

    def write_experiment(path: Path, harness_path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    'name = "stable"',
                    'output_dir = "runs/stable"',
                    "seeds = [42]",
                    "[benchmark]",
                    'name = "swebench"',
                    f'profile = "{benchmark}"',
                    "[harness]",
                    'name = "openclaw"',
                    f'profile = "{harness_path}"',
                    "[model]",
                    f'profile = "{ROOT / "profiles/models/qwen35_9b.json"}"',
                    "[serving]",
                    f'profile = "{ROOT / "profiles/serving/vllm/default.yaml"}"',
                    "[deployment]",
                    f'profile = "{ROOT / "profiles/deployments/local_docker.toml"}"',
                ]
            ),
            encoding="utf-8",
        )

    json_experiment = tmp_path / "json.toml"
    yaml_experiment = tmp_path / "yaml.toml"
    write_experiment(json_experiment, harness_json)
    write_experiment(yaml_experiment, harness_yaml)

    assert load_resolved_experiment(json_experiment, root=ROOT).run_id == load_resolved_experiment(
        yaml_experiment,
        root=ROOT,
    ).run_id
