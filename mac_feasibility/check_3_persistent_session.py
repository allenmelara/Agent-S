"""Check 3: does one long-lived SDK session cut per-call overhead?

Runs N tiny text-only calls as fresh sessions (one CLI process each), then the
same N calls through one persistent ClaudeSDKClient. Also tries `/clear`
between turns, because Agent S keeps its own history and would not want the
session's history to grow. Makes 2*N model calls (default N=3).
"""

import argparse
import asyncio
import statistics
import tempfile
import time

from common import (
    FAIL,
    INFO,
    PASS,
    WARN,
    BillingGuardError,
    Report,
    TurnMetrics,
    ask_claude,
    check_turn_messages,
    describe_usage_warnings,
    ensure_python,
    sdk_options,
    subscription_guard,
)

SYSTEM_PROMPT = "You are a latency probe. Reply with exactly the word requested, nothing else."


async def context_tokens(client) -> int | None:
    try:
        usage = await client.get_context_usage()
        return usage.get("totalTokens")
    except Exception:
        return None


async def persistent_run(n: int, model: str | None, use_clear: bool):
    from claude_agent_sdk import ClaudeSDKClient

    turns: list[TurnMetrics] = []
    contexts: list[int | None] = []
    with tempfile.TemporaryDirectory(prefix="agent-s-check-") as workdir:
        t0 = time.perf_counter()
        async with ClaudeSDKClient(options=sdk_options(model, SYSTEM_PROMPT, workdir)) as client:
            connect_s = time.perf_counter() - t0
            for i in range(n):
                m = TurnMetrics()
                started = time.perf_counter()
                await client.query(f"Reply with: ok{i}")
                async for msg in client.receive_response():
                    check_turn_messages(msg, m, started)
                m.wall_s = time.perf_counter() - started
                turns.append(m)
                contexts.append(await context_tokens(client))
                if use_clear and i < n - 1:
                    await client.query("/clear")

                    async def drain():
                        async for _ in client.receive_response():
                            pass

                    try:
                        await asyncio.wait_for(drain(), timeout=30)
                    except asyncio.TimeoutError:
                        use_clear = False  # /clear gave no result; stop trying
                        contexts.append(None)
    return connect_s, turns, contexts


def main() -> int:
    ensure_python()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=3, help="calls per mode (default 3)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-clear", action="store_true", help="skip the /clear experiment")
    args = parser.parse_args()

    report = Report("check_3_persistent_session", "Fresh vs persistent SDK sessions")
    if not subscription_guard(report):
        report.add(FAIL, "aborted before any model call", "fix the billing-guard failures above")
        return report.finish()

    try:
        fresh: list[TurnMetrics] = []
        for i in range(args.n):
            _, m = asyncio.run(ask_claude(f"Reply with: ok{i}", system_prompt=SYSTEM_PROMPT, model=args.model))
            fresh.append(m)
        connect_s, persistent, contexts = asyncio.run(persistent_run(args.n, args.model, not args.no_clear))
    except BillingGuardError as exc:
        report.add(FAIL, "billing guard tripped", str(exc))
        return report.finish()
    except Exception as exc:
        report.add(FAIL, "Agent SDK call failed", f"{type(exc).__name__}: {exc}")
        return report.finish()

    fresh_s = [m.wall_s for m in fresh]
    pers_s = [m.wall_s for m in persistent]
    report.data.update(
        {
            "fresh_wall_s": fresh_s,
            "persistent_connect_s": connect_s,
            "persistent_turn_wall_s": pers_s,
            "persistent_context_tokens_after_each_turn": contexts,
            "fresh_metrics": [m.as_dict() for m in fresh],
            "persistent_metrics": [m.as_dict() for m in persistent],
        }
    )
    sources = {m.api_key_source for m in fresh + persistent}
    report.add(PASS if sources <= {None, "none"} else FAIL, "all sessions on subscription auth", f"apiKeySource values: {sources}")

    f_med, p_med = statistics.median(fresh_s), statistics.median(pers_s)
    report.add(INFO, "fresh session per call (median)", f"{f_med:.1f}s  {['%.1f' % s for s in fresh_s]}")
    report.add(INFO, "persistent session per call (median)", f"{p_med:.1f}s  {['%.1f' % s for s in pers_s]} (+{connect_s:.1f}s connect once)")
    saving = f_med - p_med
    if saving > 0.5:
        report.add(PASS, "persistent session is faster", f"saves ~{saving:.1f}s per call (measured, n={args.n})")
    else:
        report.add(WARN, "no meaningful saving measured", f"difference {saving:.1f}s; startup is not the bottleneck")

    if not args.no_clear:
        known = [c for c in contexts if c is not None]
        if len(known) >= 2 and max(known) - min(known) < 500:
            report.add(PASS, "/clear keeps context flat", f"context tokens per turn: {contexts}")
        elif known:
            report.add(WARN, "context grows between turns", f"context tokens per turn: {contexts}; history accumulates")
        else:
            report.add(INFO, "context size unavailable", "get_context_usage() not supported by this SDK/CLI")

    describe_usage_warnings(report, persistent[-1] if persistent else fresh[-1])
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
