"""Agent S menu-bar icon: start a task, see whether one is running, stop it.

Tasks run in a Terminal window, exactly like the Dock app, so Agent S keeps
using the Accessibility and Screen Recording permissions Terminal already
has, and the log stays visible. Stopping uses the same ~/.agent_s/stop file
as the Dock app (see gui_agents/s3/utils/run_control.py).

Run with the repo's .venv Python:  .venv/bin/python mac_app/menubar.py
(mac_app/build_menubar_app.sh wraps this in "Agent S Menu Bar.app").
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gui_agents.s3.utils import run_control  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import watch_claude  # noqa: E402

WATCH_MINUTES = (5, 10, 15, 30)

ICON = REPO / "mac_app" / "menubar_icon.png"
SETTINGS_FILE = run_control.STATE_DIR / "menubar.json"
LOCK_FILE = run_control.STATE_DIR / "menubar.pid"

APPROVAL_LABELS = {
    "risky": "Ask only before risky actions",
    "always": "Ask before every action",
    "never": "Never ask",
}


def task_command(task: str, approval: str) -> str:
    """The shell command Terminal runs; same settings as the Dock app."""
    return (
        f"cd {shlex.quote(str(REPO))}"
        " && { conda deactivate 2>/dev/null || true; }"
        " && source .venv/bin/activate"
        " && python -m gui_agents.s3.cli_app"
        " --provider claude_subscription --ground_provider claude_subscription"
        f" --max_steps 15 --max_model_calls 60 --approval {shlex.quote(approval)}"
        f" --task {shlex.quote(task)}"
    )


def watch_command(minutes, approval: str) -> str:
    """Shell command that runs the Watch Claude loop in Terminal."""
    return (
        f"cd {shlex.quote(str(REPO))}"
        " && { conda deactivate 2>/dev/null || true; }"
        " && source .venv/bin/activate"
        f" && python mac_app/watch_claude.py --minutes {float(minutes):g} --approval {shlex.quote(approval)}"
    )


def run_in_terminal(command: str) -> None:
    # Pass the command as an argument so no AppleScript quoting is needed.
    subprocess.run(
        [
            "osascript",
            "-e", "on run argv",
            "-e", 'tell application "Terminal" to activate',
            "-e", 'tell application "Terminal" to do script (item 1 of argv)',
            "-e", "end run",
            command,
        ],
        check=False,
    )


def load_settings() -> dict:
    try:
        settings = json.loads(SETTINGS_FILE.read_text())
    except (OSError, ValueError):
        settings = {}
    if settings.get("approval") not in APPROVAL_LABELS:
        settings["approval"] = "risky"
    if settings.get("watch_minutes") not in WATCH_MINUTES:
        settings["watch_minutes"] = 15
    return settings


def save_settings(settings: dict) -> None:
    run_control.STATE_DIR.mkdir(exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings))


def already_open() -> bool:
    """True if another copy of this menu-bar app is running; else claim the lock."""
    try:
        pid = int(LOCK_FILE.read_text().strip())
        if pid != os.getpid():
            os.kill(pid, 0)
            return True
    except (OSError, ValueError):
        pass
    run_control.STATE_DIR.mkdir(exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))
    return False


def main() -> None:
    import rumps

    if already_open():
        return

    settings = load_settings()

    class AgentSMenu(rumps.App):
        def __init__(self):
            super().__init__("Agent S", icon=str(ICON) if ICON.exists() else None, quit_button=None)
            self.status = rumps.MenuItem("Idle")
            self.new_task = rumps.MenuItem("New task…", callback=self.on_new_task, key="n")
            self.stop_task = rumps.MenuItem("Stop task", callback=self.on_stop)
            self.approval_items = {
                mode: rumps.MenuItem(label, callback=self.on_approval)
                for mode, label in APPROVAL_LABELS.items()
            }
            approval_menu = rumps.MenuItem("Approvals")
            for item in self.approval_items.values():
                approval_menu.add(item)
            self.watch = rumps.MenuItem("Watch Claude → ChatGPT", callback=self.on_watch)
            self.check_now = rumps.MenuItem("Check now", callback=self.on_check_now)
            self.minute_items = {
                m: rumps.MenuItem(f"Every {m} minutes", callback=self.on_minutes) for m in WATCH_MINUTES
            }
            minutes_menu = rumps.MenuItem("Watch interval")
            for item in self.minute_items.values():
                minutes_menu.add(item)
            self.menu = [
                self.status,
                None,
                self.new_task,
                self.stop_task,
                None,
                self.watch,
                self.check_now,
                minutes_menu,
                rumps.MenuItem("Open watch log", callback=self.on_watch_log),
                None,
                approval_menu,
                rumps.MenuItem("Open guide", callback=self.on_guide),
                None,
                rumps.MenuItem("Quit menu-bar icon", callback=self.on_quit),
            ]
            self.mark_approval()
            self.mark_minutes()
            self.refresh(None)
            self.timer = rumps.Timer(self.refresh, 1.5)
            self.timer.start()

        def mark_approval(self):
            for mode, item in self.approval_items.items():
                item.state = mode == settings["approval"]

        def mark_minutes(self):
            for minutes, item in self.minute_items.items():
                item.state = minutes == settings["watch_minutes"]

        def refresh(self, _):
            running = run_control.running_pid() is not None
            stopping = running and run_control.STOP_FILE.exists()
            watching = watch_claude.watcher_pid() is not None
            watch_stopping = watching and watch_claude.STOP_FILE.exists()
            self.title = "●" if running else "◎" if watching else None
            if stopping:
                status = "Stopping…"
            elif running:
                status = "Working on a task"
            elif watch_stopping:
                status = "Stopping the watch…"
            elif watching:
                status = "Watching Claude"
            else:
                status = "Idle"
            self.status.title = status
            self.watch.state = watching and not watch_stopping
            self.check_now.set_callback(self.on_check_now if watching and not watch_stopping else None)
            self.new_task.set_callback(None if running else self.on_new_task)
            self.stop_task.set_callback(self.on_stop if running and not stopping else None)

        def on_new_task(self, _):
            window = rumps.Window(
                message=(
                    "It uses your Claude Max login (no API key). "
                    f"Approvals: {APPROVAL_LABELS[settings['approval']].lower()}.\n"
                    "To stop it: Stop task in this menu, or move the mouse to the top-left corner."
                ),
                title="What should Agent S do on your Mac?",
                default_text="",
                ok="Start",
                cancel="Cancel",
                dimensions=(360, 60),
            )
            response = window.run()
            task = response.text.strip()
            if response.clicked and task:
                run_in_terminal(task_command(task, settings["approval"]))

        def on_stop(self, _):
            run_control.request_stop()
            # rumps.notification needs an app bundle Info.plist, which a venv Python lacks.
            subprocess.run(
                ["osascript", "-e", 'display notification "Stopping before its next action." with title "Agent S"'],
                check=False,
            )
            self.refresh(None)

        def on_watch(self, _):
            if watch_claude.watcher_pid() is not None:
                watch_claude.STATE_DIR.mkdir(exist_ok=True)
                watch_claude.STOP_FILE.touch()  # stops after any check in progress
            else:
                run_in_terminal(watch_command(settings["watch_minutes"], settings["approval"]))
            self.refresh(None)

        def on_check_now(self, _):
            watch_claude.NOW_FILE.touch()

        def on_minutes(self, sender):
            for minutes, item in self.minute_items.items():
                if item is sender:
                    settings["watch_minutes"] = minutes
            save_settings(settings)
            self.mark_minutes()
            if watch_claude.watcher_pid() is not None:
                rumps.alert("Agent S", "The new interval applies the next time you turn on Watch Claude.")

        def on_watch_log(self, _):
            watch_claude.STATE_DIR.mkdir(exist_ok=True)
            watch_claude.LOG_FILE.touch()
            subprocess.run(["open", "-a", "Console", str(watch_claude.LOG_FILE)], check=False)

        def on_approval(self, sender):
            for mode, item in self.approval_items.items():
                if item is sender:
                    settings["approval"] = mode
            save_settings(settings)
            self.mark_approval()

        def on_guide(self, _):
            subprocess.run(["open", str(REPO / "mac_feasibility" / "RUN_AGENT_S.md")], check=False)

        def on_quit(self, _):
            LOCK_FILE.unlink(missing_ok=True)
            rumps.quit_application()

    AgentSMenu().run()


if __name__ == "__main__":
    main()
