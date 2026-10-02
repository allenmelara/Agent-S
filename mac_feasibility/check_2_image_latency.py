"""Check 2: send one harmless synthetic image through the Agent SDK, tools disabled.

The image is generated here (coloured shapes and a number); no screenshot is
taken. Makes exactly one model call against your Max allowance.
"""

import argparse
import asyncio
import json
import re

from PIL import Image, ImageDraw

from common import (
    FAIL,
    INFO,
    PASS,
    WARN,
    BillingGuardError,
    Report,
    ask_claude,
    describe_usage_warnings,
    ensure_python,
    image_block,
    png_bytes,
    subscription_guard,
)

SECRET_NUMBER = 4719
SYSTEM_PROMPT = "You describe test images. Reply with JSON only, no prose."
QUESTION = (
    'Reply with JSON of the form {"number": <the number written in the image>, '
    '"left_shape": "<shape on the left>", "right_shape": "<shape on the right>"}.'
)


def make_test_image() -> bytes:
    img = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((60, 110, 200, 250), fill=(220, 40, 40))  # red circle, left
    draw.rectangle((440, 110, 580, 250), fill=(40, 90, 220))  # blue square, right
    # Draw the number large using the default bitmap font, scaled up.
    text_img = Image.new("RGB", (60, 16), "white")
    ImageDraw.Draw(text_img).text((2, 2), str(SECRET_NUMBER), fill="black")
    text_img = text_img.resize((240, 64), Image.NEAREST)
    img.paste(text_img, (200, 20))
    return png_bytes(img)


def main() -> int:
    ensure_python()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="model alias, e.g. sonnet or opus (default: your Claude Code default)")
    args = parser.parse_args()

    report = Report("check_2_image_latency", "One test image via Agent SDK, tools disabled")
    if not subscription_guard(report):
        report.add(FAIL, "aborted before any model call", "fix the billing-guard failures above")
        return report.finish()

    content = [image_block(make_test_image()), {"type": "text", "text": QUESTION}]
    try:
        text, m = asyncio.run(ask_claude(content, system_prompt=SYSTEM_PROMPT, model=args.model))
    except BillingGuardError as exc:
        report.add(FAIL, "billing guard tripped during the call", str(exc))
        return report.finish()
    except Exception as exc:  # surface SDK/CLI errors in the report
        report.add(FAIL, "Agent SDK call failed", f"{type(exc).__name__}: {exc}")
        return report.finish()

    report.data.update({"reply": text, "metrics": m.as_dict()})
    report.add(PASS if m.api_key_source in (None, "none") else FAIL, "session auth", f"apiKeySource={m.api_key_source!r}")
    report.add(PASS if not m.tools else FAIL, "tools disabled", f"tools offered to the model: {m.tools or 'none'}")
    report.add(PASS if m.tool_calls == 0 else FAIL, "no tool calls", f"{m.tool_calls} tool calls")
    report.add(INFO, "model", str(m.model))

    try:
        parsed = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except (AttributeError, json.JSONDecodeError):
        parsed = {}
    number_ok = str(parsed.get("number", "")).strip() == str(SECRET_NUMBER)
    shapes_ok = "circle" in str(parsed.get("left_shape", "")).lower() and any(
        w in str(parsed.get("right_shape", "")).lower() for w in ("square", "rect")
    )
    report.add(PASS if number_ok and shapes_ok else FAIL, "image was actually seen", f"reply: {text.strip()[:200]}")

    report.add(INFO, "wall time (process start to result)", f"{m.wall_s:.1f}s")
    if m.first_token_s is not None:
        report.add(INFO, "time to first reply", f"{m.first_token_s:.1f}s")
    if m.api_ms is not None:
        overhead = m.wall_s - m.api_ms / 1000
        report.add(INFO, "model time vs startup overhead", f"api {m.api_ms / 1000:.1f}s, other {overhead:.1f}s")
    report.add(
        INFO,
        "client-side cost estimate",
        f"${m.est_cost_usd} (API-price estimate computed by Claude Code; not an invoice)",
    )
    if m.wall_s > 30:
        report.add(WARN, "slow single call", "Agent S makes 3+ calls per step; expect long tasks")
    describe_usage_warnings(report, m)
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
