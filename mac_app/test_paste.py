"""Check that Agent S can paste on this Mac. No Claude calls; about 10 seconds.

Opens a new TextEdit document, puts a test phrase on the clipboard, pastes it
with each method Agent S can use, and reads the document back:

  menu      Edit > Paste clicked via System Events (Agent S's first choice)
  quartz    Command+V as key events carrying the Command flag (fallback)
  pyautogui pyautogui.hotkey('command', 'v') (the old way that typed "v")
  agent     what Agent S actually does now: press_combo(['command', 'v'])

The first time, macOS may ask to let Terminal control "System Events" and
"TextEdit": click OK. The test document is closed without saving.

Run:  .venv/bin/python mac_app/test_paste.py
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_agents.s3.utils import mac_keys  # noqa: E402


def osa(script: str) -> str:
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def set_clipboard(text: str) -> None:
    subprocess.run(["pbcopy"], input=text.encode(), check=True)


def paste_pyautogui() -> bool:
    import pyautogui

    pyautogui.hotkey("command", "v")
    return True


METHODS = {
    "menu": lambda: mac_keys.click_edit_menu("Paste"),
    "quartz": lambda: mac_keys.quartz_combo(["command", "v"]),
    "pyautogui": paste_pyautogui,
    "agent": lambda: mac_keys.press_combo(["command", "v"]) or True,
}


def try_method(name: str, method) -> tuple:
    phrase = f"Agent S paste test {name} {int(time.time())}"
    osa('tell application "TextEdit" to make new document')
    osa('tell application "TextEdit" to activate')
    time.sleep(0.8)
    set_clipboard(phrase)
    try:
        ran = method()
        time.sleep(0.8)
        text = osa('tell application "TextEdit" to get text of front document')
    finally:
        try:
            osa('tell application "TextEdit" to close front document saving no')
        except RuntimeError:
            pass
    if not ran:
        return False, "method reported failure"
    if text == phrase:
        return True, "pasted correctly"
    shown = text if len(text) < 60 else text[:57] + "..."
    return False, f"document contains {shown!r}"


def main() -> int:
    if sys.platform != "darwin":
        print("Run this on your Mac.")
        return 1
    print("Testing paste methods in a new TextEdit document. Don't touch the keyboard or mouse.\n")
    results = {}
    for name, method in METHODS.items():
        try:
            ok, detail = try_method(name, method)
        except Exception as exc:  # permission prompts, TextEdit missing, etc.
            ok, detail = False, f"error: {exc}"
        results[name] = ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:9} {detail}")
    osa('tell application "Terminal" to activate')
    print()
    if results.get("agent"):
        print("Agent S can paste reliably on this Mac.")
        return 0
    print("Agent S's paste failed. Send these lines to Claude.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
