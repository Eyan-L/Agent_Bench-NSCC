from __future__ import annotations

from pathlib import Path

from scripts.breakdown_sglang_pd import build_report


def test_breakdown_parses_pd_request_time_stats(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "sglang.log"
    log.parent.mkdir()
    log.write_text(
        "\n".join(
            [
                "[sglang-pd] warming router with 1 chat request(s): http://127.0.0.1:8000/v1/chat/completions",
                "\x1b[2m2026-06-02 13:05:21\x1b[0m INFO http_request{method=POST uri=/v1/chat/completions version=HTTP/1.1 module=\"smg\" request_id=\"warm\"}: smg::request: started processing request",
                "\x1b[2m2026-06-02 13:05:21\x1b[0m INFO http_request{method=POST uri=/v1/chat/completions version=HTTP/1.1 module=\"smg\" request_id=\"warm\" status_code=200 latency=1000}: smg::response: finished processing request",
                "[sglang-pd] warmup 1/1 ok status=200 elapsed_ms=2.0",
                "\x1b[2m2026-06-02 13:05:22\x1b[0m INFO http_request{method=POST uri=/v1/chat/completions version=HTTP/1.1 module=\"smg\" request_id=\"real\"}: smg::request: started processing request",
                "\x1b[2m2026-06-02 13:05:23\x1b[0m INFO http_request{method=POST uri=/v1/chat/completions version=HTTP/1.1 module=\"smg\" request_id=\"real\" status_code=200 latency=250000}: smg::response: finished processing request",
                "[2026-06-02 13:05:22] Prefill batch, #new-seq: 1, #new-token: 128, #cached-token: 64, full token usage: 0.01, mamba usage: 0.01, #running-req: 0, #queue-req: 0, #pending-token: 0, #prealloc-req: 0, #inflight-req: 1, cuda graph: False, input throughput (token/s): 512.00",
                "[2026-06-02 13:05:23] Decode batch, #running-req: 1, #full token: 256, full token usage: 0.01, mamba num: 1, mamba usage: 0.01, pre-allocated usage: 0.00, #prealloc-req: 0, #transfer-req: 1, #retracted-req: 0, cuda graph: True, gen throughput (token/s): 128.00, #queue-req: 0",
                "Req Time Stats(rid=prefill-1, input len=128, output len=1, type=prefill): bootstrap_queue_duration(7.00ms) = alloc_wait(1.00ms) + bootstrap(2.00ms) + other(4.00ms); queue_duration=3.00ms, forward_duration=11.00ms, start=1.000, transfer_speed=42.00GB/s, transfer_total=16.00MB, #retries=0",
                "Req Time Stats(rid=decode-1, input len=128, output len=8, type=decode): prealloc_queue_duration(5.00ms) = alloc_wait(1.00ms) + bootstrap(1.50ms) + other(2.50ms); transfer_duration=6.00ms; queue_duration=2.00ms, forward_duration=20.00ms, start=1.000",
            ]
        )
        + "\n"
    )

    report = build_report(log, None, include_warmup=False)

    assert report["router_http_chat_latency"]["count"] == 1
    assert report["router_http_chat_latency"]["mean_ms"] == 250.0
    assert report["pd_stage_latency_ms"]["prefill.bootstrap_queue"]["mean_ms"] == 7.0
    assert report["pd_stage_latency_ms"]["decode.transfer"]["mean_ms"] == 6.0
    assert report["pd_transfer"]["transfer_speed_gb_s"]["mean_ms"] == 42.0
    assert report["batch_activity"]["decode"]["transfer_req_samples_sum"] == 1
