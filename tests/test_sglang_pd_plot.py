from __future__ import annotations

from scripts.plot_sglang_pd_breakdown import prepare_stacked_rows


def test_prepare_stacked_rows_uses_exact_pd_stages_when_available() -> None:
    report = {
        "router_http_chat_latency": {"sum_ms": 100.0},
        "pd_stage_latency_ms": {
            "prefill.forward": {"sum_ms": 40.0},
            "decode.transfer": {"sum_ms": 10.0},
            "decode.forward": {"sum_ms": 30.0},
        },
        "openclaw_spans": {
            "model_usage": {"sum_ms": 200.0},
            "runtime_tool": {"sum_ms": 50.0},
        },
    }

    rows = prepare_stacked_rows(report, include_agent_context=True)

    assert rows[0]["values"]["prefill.forward"] == 40.0
    assert rows[0]["values"]["decode.transfer"] == 10.0
    assert rows[0]["values"]["router.residual"] == 20.0
    assert rows[1]["values"]["agent.llm_serving"] == 200.0
    assert rows[1]["values"]["agent.runtime_tool"] == 50.0


def test_prepare_stacked_rows_falls_back_to_router_observed() -> None:
    report = {
        "router_http_chat_latency": {"sum_ms": 123.0},
        "pd_stage_latency_ms": {},
        "openclaw_spans": {},
    }

    rows = prepare_stacked_rows(report)

    assert rows == [
        {
            "label": "SGLang PD serving",
            "values": {"router.observed": 123.0},
            "total_ms": 123.0,
            "note": "Stage detail unavailable; bar shows router HTTP latency.",
        }
    ]


def test_prepare_stacked_rows_estimates_prefill_decode_without_time_stats() -> None:
    report = {
        "router_http_chat_latency": {"sum_ms": 100.0},
        "pd_stage_latency_ms": {},
        "batch_activity": {
            "prefill": {"batch_count": 1},
            "decode": {"batch_count": 3},
        },
        "openclaw_spans": {},
    }

    rows = prepare_stacked_rows(report)

    assert rows[0]["values"]["prefill.forward"] == 25.0
    assert rows[0]["values"]["decode.forward"] == 75.0
    assert "estimates Prefill/Decode" in rows[0]["note"]
