"""Check 3: does one long-lived SDK session cut per-call overhead, and does
`/clear` really drop the session's history?

Agent S keeps its own history, so a persistent session must not also keep
one, or every step would carry old screenshots twice.

Each turn carries ~2,000 tokens of filler and a unique codeword, so leftover
history shows up clearly in the context size. A recall probe then asks for the
previous codeword: once without /clear (control, should remember) and once
after /clear (should not).

Model calls with the default N=3: 3 fresh + 3 persistent + 2 probes = 8.
"""

import argparse
import asyncio
import secrets
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

SYSTEM_PROMPT = (
    "You are a test probe. Ignore filler lines. Follow the final instruction exactly "
    "and reply with nothing else."
)
FILLER_LINES = 120  # roughly 2,000 tokens; the exact count is reported, not assumed
GROWTH_LIMIT = 1000  # tokens: a turn left in history would add ~2,000
RECALL_PROMPT = (
    "What was the codeword in my previous message? Reply with the codeword only, "
    "or exactly UNKNOWN if there is no previous message."
)


def padded_turn(i: int, codeword: str) -> str:
    filler = "\n".join(
        f"Filler line {k}: the quick brown fox jumps over the lazy dog, item {k}."
        for k in range(FILLER_LINES)
    )
    return f"{filler}\n\nCodeword for this message: {codeword}\nInstruction: reply with exactly ok{i}"


def new_codeword() -> str:
    return "CW" + secrets.token_hex(3).upper()


async def context_tokens(client) -> int | None:
    try:
        usage = await client.get_context_usage()
        return usage.get("totalTokens")
    except Exception:
        return None


async def send(client, text: str) -> tuple[TurnMetrics, str]:
    m = TurnMetrics()
    parts: list[str] = []
    started = time.perf_counter()
    await client.query(text)
    async for msg in client.receive_response():
        out = check_turn_messages(msg, m, started)
        if out:
            parts.append(out)
    m.wall_s = time.perf_counter() - started
    return m, "".join(parts)


async def clear(client) -> bool:
    """Send /clear; True if it completed. It is a local command, not a model call."""

    async def drain():
        await client.query("/clear")
        async for _ in client.receive_response():
            pass

    try:
        await asyncio.wait_for(drain(), timeout=30)
        return True
    except asyncio.TimeoutError:
        return False


async def persistent_run(n: int, model: str | None, use_clear: bool, codewords: list[str]):
    from claude_agent_sdk import ClaudeSDKClient

    result = {"turns": [], "contexts": [], "clear_ok": True}
    with tempfile.TemporaryDirectory(prefix="agent-s-check-") as workdir:
        t0 = time.perf_counter()
        async with ClaudeSDKClient(options=sdk_options(model, SYSTEM_PROMPT, workdir)) as client:
            result["connect_s"] = time.perf_counter() - t0
            for i in range(n):
                m, _ = await send(client, padded_turn(i, codewords[i]))
                result["turns"].append(m)
                result["contexts"].append(await context_tokens(client))
                if i == 0:
                    # Control: no /clear yet, so Claude should remember codeword 0.
                    pm, reply = await send(client, RECALL_PROMPT)
                    result["control"] = {"reply": reply.strip(), "metrics": pm}
                if use_clear and not await clear(client):
                    result["clear_ok"] = False
                    use_clear = False
            # After the last turn (and /clear, if used): Claude should not know any codeword.
            pm, reply = await send(client, RECALL_PROMPT)
            result["probe"] = {"reply": reply.strip(), "metrics": pm}
    return result


def main() -> int:
    ensure_python()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=3, help="padded calls per mode (default 3, minimum 2)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-clear", action="store_true", help="never send /clear (shows what retention looks like)")
    args = parser.parse_args()
    n = max(2, args.n)

    report = Report("check_3_persistent_session", "Fresh vs persistent SDK sessions, and /clear")
    if not subscription_guard(report):
        report.add(FAIL, "aborted before any model call", "fix the billing-guard failures above")
        return report.finish()

    codewords = [new_codeword() for _ in range(n)]
    try:
        fresh: list[TurnMetrics] = []
        for i in range(n):
            _, m = asyncio.run(
                ask_claude(padded_turn(i, new_codeword()), system_prompt=SYSTEM_PROMPT, model=args.model)
            )
            fresh.append(m)
        run = asyncio.run(persistent_run(n, args.model, not args.no_clear, codewords))
    except BillingGuardError as exc:
        report.add(FAIL, "billing guard tripped", str(exc))
        return report.finish()
    except Exception as exc:
        report.add(FAIL, "Agent SDK call failed", f"{type(exc).__name__}: {exc}")
        return report.finish()

    persistent = run["turns"]
    probes = [run["control"]["metrics"], run["probe"]["metrics"]]
    contexts = run["contexts"]
    fresh_s = [m.wall_s for m in fresh]
    pers_s = [m.wall_s for m in persistent]
    report.data.update(
        {
            "codewords": codewords,
            "fresh_wall_s": fresh_s,
            "persistent_connect_s": run["connect_s"],
            "persistent_turn_wall_s": pers_s,
            "persistent_context_tokens_after_each_turn": contexts,
            "control_probe_reply": run["control"]["reply"],
            "final_probe_reply": run["probe"]["reply"],
            "clear_completed": run["clear_ok"],
            "fresh_metrics": [m.as_dict() for m in fresh],
            "persistent_metrics": [m.as_dict() for m in persistent],
            "probe_metrics": [m.as_dict() for m in probes],
        }
    )

    sources = {m.api_key_source for m in fresh + persistent + probes}
    report.add(PASS if sources <= {None, "none"} else FAIL, "all sessions on subscription auth", f"apiKeySource values: {sources}")

    f_med, p_med = statistics.median(fresh_s), statistics.median(pers_s)
    report.add(INFO, "fresh session per call (median)", f"{f_med:.1f}s  {['%.1f' % s for s in fresh_s]}")
    report.add(INFO, "persistent session per call (median)", f"{p_med:.1f}s  {['%.1f' % s for s in pers_s]} (+{run['connect_s']:.1f}s connect once)")
    saving = f_med - p_med
    if saving > 0.5:
        report.add(PASS, "persistent session is faster", f"saves ~{saving:.1f}s per call (measured, n={n}, ~2k-token turns)")
    else:
        report.add(WARN, "no meaningful saving measured", f"difference {saving:.1f}s; startup is not the bottleneck")

    # The control shows the probe can detect history at all.
    control_ok = codewords[0] in run["control"]["reply"]
    report.add(
        PASS if control_ok else WARN,
        "control: history kept without /clear",
        f"asked for {codewords[0]}, got {run['control']['reply'][:60]!r}"
        + ("" if control_ok else "; probe inconclusive"),
    )

    if args.no_clear:
        report.add(INFO, "context per turn without /clear", f"{contexts}")
        return finish(report, persistent, probes)

    if not run["clear_ok"]:
        report.add(FAIL, "/clear completed", "/clear gave no result within 30s; Agent S would need a fresh session per call")
        return finish(report, persistent, probes)

    leaked = [c for c in codewords if c in run["probe"]["reply"]]
    report.add(
        FAIL if leaked else PASS,
        "/clear drops earlier turns (recall probe)",
        f"after /clear Claude replied {run['probe']['reply'][:60]!r}"
        + (f"; it still knew {leaked}" if leaked else ""),
    )

    known = [c for c in contexts if c is not None]
    if len(known) == len(contexts) and len(known) >= 2:
        growth = max(known[1:]) - known[0]
        report.add(
            PASS if growth < GROWTH_LIMIT else FAIL,
            "/clear keeps context flat (size)",
            f"context tokens after each turn: {contexts}; growth {growth} (one kept turn would add ~2,000)",
        )
    else:
        report.add(INFO, "context size unavailable", f"get_context_usage() returned {contexts}; relying on the recall probe")
    return finish(report, persistent, probes)


def finish(report: Report, persistent: list[TurnMetrics], probes: list[TurnMetrics]) -> int:
    # Rate-limit events arrive only when the status changes, so pool every turn's events.
    pooled = TurnMetrics()
    pooled.rate_limits = [rl for m in persistent + probes for rl in m.rate_limits]
    describe_usage_warnings(report, pooled)
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
