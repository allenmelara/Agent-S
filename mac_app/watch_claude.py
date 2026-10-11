"""Watch Claude: send Claude's finished replies to ChatGPT, checking every N minutes.

Each check is free: it screenshots only the Claude window (`screencapture -l`)
and compares small thumbnails locally. Agent S (and so your Max allowance) is
only used when the Claude window
  1. has changed since the last handoff, and
  2. has stopped changing (two captures SETTLE_SECONDS apart look the same).
Agent S then double-checks that Claude is really finished and that the reply
isn't already in ChatGPT before copying and sending it.

Runs in a Terminal window so it reuses Terminal's Screen Recording and
Accessibility permissions. The menu-bar icon starts and stops it; files in
~/.agent_s control it:
  watch.pid   this watcher is running     watch_stop  ask it to stop
  watch_now   check right away, ignoring "unchanged since last handoff"
  watch.log   what it did

Run:  .venv/bin/python mac_app/watch_claude.py --minutes 15
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gui_agents.s3.utils import run_control  # noqa: E402

STATE_DIR = run_control.STATE_DIR
PID_FILE = STATE_DIR / "watch.pid"
STOP_FILE = STATE_DIR / "watch_stop"
NOW_FILE = STATE_DIR / "watch_now"
STATE_FILE = STATE_DIR / "watch_state.json"
LOG_FILE = STATE_DIR / "watch.log"

THUMB = (96, 96)
SETTLE_SECONDS = 10
PIXEL_CHANGE = 12  # a thumbnail pixel differing by more than this (0-255) has changed
NEW_CONTENT_MIN = 0.01  # share of changed thumbnail pixels that counts as new content

RELAY_TASK = (
    "Look at the Claude app. If Claude is still working on its reply (a stop button, spinner or "
    "progress indicator is showing), call done without doing anything. Otherwise look at the ChatGPT "
    "app: if my most recent message there already contains the same text as Claude's latest reply, "
    "call done without doing anything. Otherwise copy Claude's latest reply using the copy button "
    "below it, click inside the message box at the bottom of the ChatGPT app, and if that box still "
    "holds leftovers from an earlier attempt (a stray character or a partial paste), clear them. Paste "
    "with Command+V, check that the box contains the copied text, and press Enter to send it. Then stop."
)


# ---------------------------------------------------------------- pure logic


def signature(image) -> bytes:
    """Small grayscale thumbnail of the conversation area (no title bar, no input box)."""
    w, h = image.size
    area = image.crop((0, int(h * 0.08), w, int(h * 0.85)))
    return area.convert("L").resize(THUMB).tobytes()


def changed_share(a: bytes, b: bytes) -> float:
    """Share of thumbnail pixels that differ noticeably (1.0 if sizes differ)."""
    if len(a) != len(b):
        return 1.0
    return sum(abs(x - y) > PIXEL_CHANGE for x, y in zip(a, b)) / len(a)


def decide(baseline, first, second) -> str:
    """'no-window', 'working', 'unchanged' or 'relay'. baseline None means 'don't compare'.

    Any changed pixel between the two captures means Claude is still working:
    a small spinner or timer touches only a few thumbnail pixels.
    """
    if first is None or second is None:
        return "no-window"
    if changed_share(first, second) > 0:
        return "working"
    if baseline is not None and changed_share(baseline, second) < NEW_CONTENT_MIN:
        return "unchanged"
    return "relay"


def wait_for(seconds: float, poll: float = 2.0, sleep=time.sleep) -> str:
    """Sleep up to `seconds`; return 'stop', 'now' or 'timeout'."""
    waited = 0.0
    while True:
        if STOP_FILE.exists():
            return "stop"
        if NOW_FILE.exists():
            NOW_FILE.unlink(missing_ok=True)
            return "now"
        if waited >= seconds:
            return "timeout"
        sleep(poll)
        waited += poll


def relay_command(approval: str) -> list:
    return [
        sys.executable, "-m", "gui_agents.s3.cli_app",
        "--provider", "claude_subscription", "--ground_provider", "claude_subscription",
        "--approval", approval, "--max_steps", "12", "--max_model_calls", "40", "--quiet_done",
        "--task", RELAY_TASK,
    ]


def watcher_pid():
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


# ---------------------------------------------------------------- macOS helpers


def claude_window_id():
    """Window number of the largest on-screen Claude window, or None."""
    import Quartz

    infos = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    best = None
    for w in infos or []:
        if w.get("kCGWindowOwnerName") != "Claude" or w.get("kCGWindowLayer", 0) != 0:
            continue
        bounds = w.get("kCGWindowBounds", {})
        area = bounds.get("Width", 0) * bounds.get("Height", 0)
        if area > 200 * 200 and (best is None or area > best[0]):
            best = (area, int(w["kCGWindowNumber"]))
    return best[1] if best else None


def capture_signature(window_id):
    """Signature of that window, even if other windows cover it; None on failure."""
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "claude.png"
        result = subprocess.run(["screencapture", "-x", "-o", f"-l{window_id}", str(path)], capture_output=True)
        if result.returncode != 0 or not path.exists():
            return None
        with Image.open(io.BytesIO(path.read_bytes())) as img:
            return signature(img)


def clipboard_hash():
    try:
        text = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
    except OSError:
        return None
    return hashlib.sha256(text.encode()).hexdigest() if text.strip() else None


def notify(message: str) -> None:
    safe = message.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e", f'display notification "{safe}" with title "Agent S"'], capture_output=True)


def log(message: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {message}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------- main loop


def check_once(state: dict, forced: bool, approval: str) -> str:
    if run_control.running_pid():
        log("Another Agent S task is running; skipping this check.")
        return "busy"
    window = claude_window_id()
    first = capture_signature(window) if window else None
    if first is not None:
        time.sleep(SETTLE_SECONDS)
        second = capture_signature(window)
    else:
        second = None
    baseline = None if forced else (bytes.fromhex(state["baseline"]) if state.get("baseline") else None)
    verdict = decide(baseline, first, second)
    messages = {
        "no-window": "Claude window not visible (minimized, closed or on another desktop); skipping.",
        "working": "Claude's window is still changing; checking again later.",
        "unchanged": "Nothing new from Claude since the last handoff.",
        "relay": "New finished reply from Claude; asking Agent S to send it to ChatGPT.",
    }
    log(messages[verdict])
    if verdict != "relay":
        return verdict

    before = clipboard_hash()
    finished = subprocess.run(relay_command(approval), cwd=REPO).returncode == 0
    after = clipboard_hash()
    copied = bool(after) and after != before
    if finished and copied:
        if after == state.get("last_sent"):
            log("Warning: the copied text matches the previous handoff; it may have been sent twice.")
        state.update(last_sent=after, baseline=second.hex())
        save_state(state)
        log("Handoff done.")
        notify("Sent Claude's latest reply to ChatGPT.")
        return "sent"
    if copied:
        # Copied but Agent S didn't finish (paste or send failed): retry next check.
        log("Handoff NOT finished: the reply was copied but not sent. Will try again next check.")
        notify("Couldn't finish sending Claude's reply to ChatGPT; will retry. Check ChatGPT's message box.")
        return "failed"
    log("Agent S didn't copy anything (Claude may still be working, or it was already sent).")
    return "nothing-copied"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=15)
    parser.add_argument("--approval", default="risky", choices=("always", "risky", "never"))
    args = parser.parse_args()

    STATE_DIR.mkdir(exist_ok=True)
    if watcher_pid() not in (None, os.getpid()):
        print("Watch Claude is already running.")
        return 0
    PID_FILE.write_text(str(os.getpid()))
    STOP_FILE.unlink(missing_ok=True)
    NOW_FILE.unlink(missing_ok=True)

    state = load_state()
    window = claude_window_id()
    current = capture_signature(window) if window else None
    if current is not None:
        state["baseline"] = current.hex()  # only replies that arrive from now on are sent
        save_state(state)
    log(
        f"Watching Claude every {args.minutes:g} min. Stop it from the Agent S menu "
        "(Watch Claude), or press Ctrl+C here."
    )
    if current is None:
        log("Claude window not found yet; open the Claude app and keep it on this desktop.")
    try:
        while True:
            outcome = wait_for(args.minutes * 60)
            if outcome == "stop":
                log("Stopped.")
                break
            check_once(state, forced=outcome == "now", approval=args.approval)
    except KeyboardInterrupt:
        log("Stopped (Ctrl+C).")
    finally:
        STOP_FILE.unlink(missing_ok=True)
        if watcher_pid() in (None, os.getpid()):
            PID_FILE.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
