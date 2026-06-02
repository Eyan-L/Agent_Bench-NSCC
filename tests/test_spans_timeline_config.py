from __future__ import annotations

from agent_bench.runtime import driver


def test_spans_html_uses_experiment_config(tmp_path, monkeypatch) -> None:
    config = tmp_path / "experiment.toml"
    config.write_text("[telemetry]\nspans_html = false\n", encoding="utf-8")

    monkeypatch.setenv("AB_EXPERIMENT_CONFIG", str(config))
    monkeypatch.delenv("AB_HARNESS_CONFIG", raising=False)
    monkeypatch.delenv("AB_APP_CONFIG", raising=False)

    assert driver._spans_html_enabled_from_config() is False


def test_spans_html_ignores_legacy_env_flag(monkeypatch) -> None:
    monkeypatch.delenv("AB_EXPERIMENT_CONFIG", raising=False)
    monkeypatch.delenv("AB_HARNESS_CONFIG", raising=False)
    monkeypatch.delenv("AB_APP_CONFIG", raising=False)
    monkeypatch.setenv("AGENT_BENCH_SPANS_HTML", "0")

    assert driver._spans_html_enabled_from_config() is True
