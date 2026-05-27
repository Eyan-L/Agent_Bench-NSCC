from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys


def _q(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _model_entry(model_name: str, upstream_model: str, *, include_api_base: bool) -> list[str]:
    lines = [
        f"  - model_name: {_q(model_name)}",
        "    litellm_params:",
        f"      model: {_q(upstream_model)}",
    ]
    if include_api_base:
        lines.append('      api_base: "os.environ/LITELLM_UPSTREAM_BASE_URL"')
    lines.append('      api_key: "os.environ/LITELLM_UPSTREAM_API_KEY"')
    return lines


def render_config() -> str:
    alias = os.environ.get("LITELLM_MODEL_ALIAS") or os.environ.get("MODEL")
    upstream = os.environ.get("LITELLM_UPSTREAM_MODEL") or os.environ.get("LITELLM_MODEL") or alias
    master_key = os.environ.get("LITELLM_MASTER_KEY") or os.environ.get("LITELLM_PROXY_API_KEY")
    if not alias:
        raise ValueError("LITELLM_MODEL_ALIAS or MODEL must be set")
    if not upstream:
        raise ValueError("LITELLM_UPSTREAM_MODEL, LITELLM_MODEL, or MODEL must be set")
    if not master_key:
        raise ValueError("LITELLM_MASTER_KEY or LITELLM_PROXY_API_KEY must be set")

    include_api_base = bool(os.environ.get("LITELLM_UPSTREAM_BASE_URL"))
    lines = ["model_list:"]
    lines.extend(_model_entry(alias, upstream, include_api_base=include_api_base))
    if upstream != alias:
        lines.extend(_model_entry(upstream, upstream, include_api_base=include_api_base))
    lines.extend(
        [
            "litellm_settings:",
            "  drop_params: true",
            "  set_verbose: false",
            "general_settings:",
            f"  master_key: {_q(master_key)}",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a LiteLLM proxy config without upstream secrets.")
    parser.add_argument("--output", required=True, help="YAML path to write.")
    args = parser.parse_args(argv)

    try:
        text = render_config()
    except ValueError as exc:
        print(f"invalid LiteLLM environment: {exc}", file=sys.stderr)
        return 2

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
