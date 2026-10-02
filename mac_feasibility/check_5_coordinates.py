"""Check 5: screen-coordinate accuracy and Retina scaling, without clicking or typing.

Part A (no model calls): reads display geometry and reproduces Agent S's
scaling math (gui_agents/s3/cli_app.py scale_screen_dimensions and
gui_agents/s3/agents/grounding.py resize_coordinates).

Part B (model calls): draws a fake screen with labelled buttons at known
positions and asks a grounder to locate each one, using Agent S's own
grounding prompt. Scores hits in several coordinate spaces so you can see
which grounding_width/height Agent S would need.

Part C (optional, --real-screen): asks the grounder for the Apple menu in a
real screenshot. With --grounder claude this sends your screen to Claude, so
it only runs when you pass the flag.

Nothing here moves the mouse, clicks, or types.
"""

import argparse
import asyncio
import base64
import json
import math
import random
import re
import urllib.request

from PIL import Image, ImageDraw

from common import (
    FAIL,
    INFO,
    PASS,
    REPORT_DIR,
    SKIP,
    WARN,
    BillingGuardError,
    Report,
    ask_claude,
    ensure_python,
    image_block,
    is_local_url,
    png_bytes,
    require_macos,
    subscription_guard,
)

LABELS = ["Save", "Cancel", "Search", "Settings", "Share", "Delete", "Export", "Help"]
# Agent S's grounding prompt (grounding.py generate_coords), verbatim.
GROUNDING_PROMPT = "Query:{ref}\nOutput only the coordinate of one point in your response.\n"
CLAUDE_SYSTEM = (
    "You locate UI elements in screenshots. Coordinates are pixels in the image as given, "
    "origin top-left. Output only two integers: x y."
)


# ---------------------------------------------------------------- Part A


def agent_s_scaled_dims(width: int, height: int, max_dim_size: int = 2400):
    # Same as gui_agents/s3/cli_app.py scale_screen_dimensions
    scale_factor = min(max_dim_size / width, max_dim_size / height, 1)
    return int(width * scale_factor), int(height * scale_factor)


def agent_s_resize(coords, grounding_w, grounding_h, screen_w, screen_h):
    # Same as gui_agents/s3/agents/grounding.py OSWorldACI.resize_coordinates
    return [round(coords[0] * screen_w / grounding_w), round(coords[1] * screen_h / grounding_h)]


def geometry(report: Report):
    import pyautogui  # imported only to read sizes and take a screenshot

    try:
        import Quartz

        has_capture = bool(Quartz.CGPreflightScreenCaptureAccess())
        _, ids, count = Quartz.CGGetActiveDisplayList(16, None, None)
    except Exception:
        has_capture, count = None, None

    if has_capture is False:
        report.add(FAIL, "Screen Recording permission", "grant it to your terminal in System Settings > Privacy & Security, then restart the terminal")
    elif has_capture:
        report.add(PASS, "Screen Recording permission", "granted")
    if count is not None:
        report.add(PASS if count == 1 else WARN, "displays", f"{count} active (Agent S is designed for one monitor)")

    pts_w, pts_h = pyautogui.size()
    shot = pyautogui.screenshot()
    px_w, px_h = shot.size
    scale = px_w / pts_w
    report.data.update({"points": [pts_w, pts_h], "screenshot_pixels": [px_w, px_h], "scale": scale})
    report.add(INFO, "screen size", f"{pts_w}x{pts_h} points; screenshot {px_w}x{px_h} pixels; scale {scale:.2f}x")
    if abs(scale - round(scale)) > 0.01:
        report.add(WARN, "non-integer scale", "unusual display scaling; watch coordinate errors in part B")

    sent_w, sent_h = agent_s_scaled_dims(pts_w, pts_h)
    report.data["agent_s_sent_image"] = [sent_w, sent_h]
    report.add(INFO, "image Agent S would send", f"{sent_w}x{sent_h} (screenshot resized to point size)")

    # Round-trip: a grounder that answers perfectly in the sent-image space must land on the right point.
    worst = 0.0
    for fx, fy in [(0.02, 0.02), (0.5, 0.5), (0.97, 0.95), (0.25, 0.8)]:
        truth = (fx * pts_w, fy * pts_h)
        answer = (truth[0] * sent_w / pts_w, truth[1] * sent_h / pts_h)
        back = agent_s_resize(answer, sent_w, sent_h, pts_w, pts_h)
        worst = max(worst, math.dist(back, truth))
    report.add(
        PASS if worst <= 1.5 else FAIL,
        "Agent S scaling math round-trip",
        f"worst error {worst:.1f} pt when grounding_width/height = sent image size; pyautogui clicks in points",
    )
    if scale > 1:
        wrong = agent_s_resize((pts_w / 2, pts_h / 2), pts_w, pts_h, px_w, px_h)
        report.add(
            INFO,
            "Retina pitfall avoided",
            f"if the screen size were taken in pixels, the centre would map to {wrong} instead of "
            f"({pts_w // 2}, {pts_h // 2}) — Agent S uses pyautogui.size() (points), which is correct",
        )
    return shot, (pts_w, pts_h), (sent_w, sent_h)


# ---------------------------------------------------------------- Part B


def make_fake_screen(w: int, h: int, n: int, seed: int):
    rng = random.Random(seed)
    img = Image.new("RGB", (w, h), (236, 236, 238))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, w, 28), fill=(250, 250, 250))  # fake menu bar
    targets, boxes = [], []
    for label in LABELS[:n]:
        for _ in range(200):
            bw, bh = 120, 44
            x0, y0 = rng.randint(20, w - bw - 20), rng.randint(60, h - bh - 20)
            box = (x0, y0, x0 + bw, y0 + bh)
            if all(box[2] < b[0] - 20 or box[0] > b[2] + 20 or box[3] < b[1] - 20 or box[1] > b[3] + 20 for b in boxes):
                break
        boxes.append(box)
        draw.rounded_rectangle(box, radius=8, fill=(255, 255, 255), outline=(120, 120, 130), width=2)
        text = Image.new("RGB", (60, 14), (255, 255, 255))
        ImageDraw.Draw(text).text((2, 1), label, fill=(20, 20, 20))
        text = text.resize((120, 28), Image.NEAREST)
        img.paste(text, (box[0] + 2, box[1] + 8))
        targets.append({"label": label, "box": box})
    return img, targets


def candidate_spaces(sent_w: int, sent_h: int) -> dict:
    spaces = {"image pixels": (sent_w, sent_h), "normalised 0-1000": (1000, 1000), "1920x1080 (README UI-TARS-1.5)": (1920, 1080)}
    # Estimate of server-side downscaling for large images (long edge 1568 / ~1.15 MP). Unverified.
    s = min(1.0, 1568 / max(sent_w, sent_h), math.sqrt(1_150_000 / (sent_w * sent_h)))
    if s < 1:
        spaces["estimated API downscale"] = (round(sent_w * s), round(sent_h * s))
    return spaces


def parse_xy(text: str):
    nums = re.findall(r"\d+(?:\.\d+)?", text or "")
    return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else None


async def ground_claude(png: bytes, ref: str, model):
    content = [image_block(png), {"type": "text", "text": GROUNDING_PROMPT.format(ref=ref)}]
    text, m = await ask_claude(content, system_prompt=CLAUDE_SYSTEM, model=model)
    if m.api_key_source not in (None, "none"):
        raise BillingGuardError(f"apiKeySource={m.api_key_source!r}")
    return text, m.wall_s


def ground_local(png: bytes, ref: str, url: str, model: str):
    import time

    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                    {"type": "text", "text": GROUNDING_PROMPT.format(ref=ref)},
                ],
            }
        ],
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read())
    return out["choices"][0]["message"]["content"], time.perf_counter() - t0


def run_grounder(name, fn, report, img, targets, sent):
    png = png_bytes(img)
    spaces = candidate_spaces(*sent)
    hits = {k: 0 for k in spaces}
    dists = {k: [] for k in spaces}
    rows, times = [], []
    for t in targets:
        try:
            raw, secs = fn(png, f"the button labelled '{t['label']}'")
        except BillingGuardError as exc:
            report.add(FAIL, f"{name}: billing guard tripped", str(exc))
            return None
        except Exception as exc:
            report.add(FAIL, f"{name}: grounder call failed", f"{type(exc).__name__}: {exc}")
            return None
        times.append(secs)
        xy = parse_xy(raw)
        row = {"label": t["label"], "box": t["box"], "raw": raw, "seconds": round(secs, 1)}
        if xy:
            for space, (gw, gh) in spaces.items():
                x, y = xy[0] * sent[0] / gw, xy[1] * sent[1] / gh
                b = t["box"]
                if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                    hits[space] += 1
                dists[space].append(math.dist((x, y), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)))
                row[space] = [round(x), round(y)]
        rows.append(row)
    # Most hits wins; ties go to the space with the smaller mean distance to button centres.
    best = max(hits, key=lambda k: (hits[k], -(sum(dists[k]) / len(dists[k]) if dists[k] else math.inf)))
    n = len(targets)
    report.data[f"{name}_results"] = rows
    report.data[f"{name}_hits_by_space"] = hits
    status = PASS if hits[best] == n else (WARN if hits[best] >= n / 2 else FAIL)
    report.add(status, f"{name}: synthetic accuracy", f"{hits[best]}/{n} hits using '{best}' space {spaces[best]}; all spaces: {hits}")
    report.add(INFO, f"{name}: seconds per grounding call", f"median {sorted(times)[len(times) // 2]:.1f}s over {len(times)} calls")

    annotated = img.copy()
    d = ImageDraw.Draw(annotated)
    for row in rows:
        if best in row:
            x, y = row[best]
            d.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(220, 0, 0), width=3)
    out = REPORT_DIR / f"check_5_{name}_synthetic.png"
    REPORT_DIR.mkdir(exist_ok=True)
    annotated.save(out)
    report.add(INFO, f"{name}: annotated image", str(out.name))
    return best, spaces[best]


def real_screen(name, fn, report, shot, pts, sent, space):
    img = shot.resize(sent, Image.LANCZOS)
    try:
        raw, _ = fn(png_bytes(img), "the Apple logo menu at the far left of the menu bar")
    except Exception as exc:
        report.add(FAIL, f"{name}: real-screen call failed", f"{type(exc).__name__}: {exc}")
        return
    xy = parse_xy(raw)
    if not xy:
        report.add(FAIL, f"{name}: real-screen answer unparseable", raw[:120])
        return
    gw, gh = space
    pt = agent_s_resize(xy, gw, gh, *pts)  # exactly what Agent S would click (it won't here)
    ok = pt[0] <= 60 and pt[1] <= 40
    report.data[f"{name}_real_screen"] = {"raw": raw, "would_click_points": pt}
    report.add(PASS if ok else FAIL, f"{name}: real-screen Apple menu", f"would click {pt} (points); expected x<=60, y<=40. Not clicked.")


def main() -> int:
    ensure_python()
    p = argparse.ArgumentParser()
    p.add_argument("--grounder", choices=["claude", "local"], action="append", default=[], help="repeatable")
    p.add_argument("--model", default=None, help="Claude model alias for --grounder claude")
    p.add_argument("--local-url", default="http://localhost:1234/v1", help="OpenAI-compatible endpoint on this Mac")
    p.add_argument("--local-model", default="ui-tars-1.5-7b")
    p.add_argument("--targets", type=int, default=4, help=f"buttons to locate (max {len(LABELS)})")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--real-screen", action="store_true", help="also test on a real screenshot (see docstring)")
    args = p.parse_args()

    report = Report("check_5_coordinates", "Coordinate accuracy and Retina scaling (no clicks)")
    if not require_macos(report):
        return report.finish()
    shot, pts, sent = geometry(report)

    if not args.grounder:
        report.add(SKIP, "accuracy test", "pass --grounder claude and/or --grounder local to run part B")
        return report.finish()

    grounders = {}
    if "claude" in args.grounder:
        if subscription_guard(report):
            grounders["claude"] = lambda png, ref: asyncio.run(ground_claude(png, ref, args.model))
        else:
            report.add(FAIL, "claude grounder skipped", "billing guard failed")
    if "local" in args.grounder:
        if not is_local_url(args.local_url):
            report.add(FAIL, "local grounder refused", f"{args.local_url} is not on this Mac; hosted endpoints are not allowed")
        else:
            grounders["local"] = lambda png, ref: ground_local(png, ref, args.local_url, args.local_model)

    img, targets = make_fake_screen(*sent, n=min(args.targets, len(LABELS)), seed=args.seed)
    for name, fn in grounders.items():
        found = run_grounder(name, fn, report, img, targets, sent)
        if found and args.real_screen:
            real_screen(name, fn, report, shot, pts, sent, found[1])
        if found:
            gw, gh = found[1]
            report.add(INFO, f"{name}: Agent S setting to use", f"--grounding_width {gw} --grounding_height {gh}")
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
