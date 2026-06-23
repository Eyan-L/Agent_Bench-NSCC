from __future__ import annotations

import ast
from typing import Any, Mapping


DEFAULT_DATASET = "ScaleAI/SWE-bench_Pro"
DEFAULT_SPLIT = "test"
DEFAULT_DOCKERHUB_USERNAME = "jefzda"
BENCHMARK_NAME = "swebench_pro"
AGENT_ID = "swebench_pro"
TASK_WORKSPACE = "/app"


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [text]
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed]
        if parsed is None:
            return []
        return [str(parsed)]
    return [str(value)]


def selected_tests_arg(value: Any) -> str:
    return ",".join(as_list(value))


def get_dockerhub_image_uri(
    instance_id: str,
    dockerhub_username: str = DEFAULT_DOCKERHUB_USERNAME,
    repo_name: str = "",
) -> str:
    """Mirror SWE-bench Pro's public Docker Hub tag convention."""
    if not repo_name or "/" not in repo_name:
        hsh = instance_id.replace("instance_", "")
        return f"{dockerhub_username}/sweap-images:{hsh[:128]}"

    repo_base, repo_name_only = repo_name.lower().split("/", 1)
    hsh = instance_id.replace("instance_", "")

    if instance_id == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    return f"{dockerhub_username}/sweap-images:{tag[:128]}"


def normalize_instance(
    row: Mapping[str, Any],
    *,
    dockerhub_username: str = DEFAULT_DOCKERHUB_USERNAME,
) -> dict[str, Any]:
    data = dict(row)
    instance_id = str(data["instance_id"])
    repo = str(data.get("repo") or data.get("repo_id") or "")

    fail_to_pass = as_list(data.get("fail_to_pass", data.get("FAIL_TO_PASS")))
    pass_to_pass = as_list(data.get("pass_to_pass", data.get("PASS_TO_PASS")))
    selected = as_list(data.get("selected_test_files_to_run"))

    docker_image = (
        data.get("dockerhub_image")
        or data.get("container_image")
        or get_dockerhub_image_uri(instance_id, dockerhub_username, repo)
    )

    data.update({
        "benchmark": BENCHMARK_NAME,
        "instance_id": instance_id,
        "repo": repo,
        "repo_id": repo,
        "base_commit": str(data.get("base_commit") or ""),
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "FAIL_TO_PASS": fail_to_pass,
        "PASS_TO_PASS": pass_to_pass,
        "selected_test_files_to_run": selected,
        "dockerhub_image": str(docker_image),
        "container_image": str(docker_image),
        "workspace_root": TASK_WORKSPACE,
    })
    return data


def profile_repo_mapping(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        inst = normalize_instance(row)
        mapping[inst["instance_id"]] = {
            "repo_id": inst.get("repo") or "",
            "base_commit": inst.get("base_commit") or "",
        }
    return mapping

