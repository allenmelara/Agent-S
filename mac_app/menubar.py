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
            self.menu = [
                self.status,
                None,
                self.new_task,
                self.stop_task,
                None,
                approval_menu,
                rumps.MenuItem("Open guide", callback=self.on_guide),
                None,
                rumps.MenuItem("Quit menu-bar icon", callback=self.on_quit),
            ]
            self.mark_approval()
            self.refresh(None)
            self.timer = rumps.Timer(self.refresh, 1.5)
            self.timer.start()

        def mark_approval(self):
            for mode, item in self.approval_items.items():
                item.state = mode == settings["approval"]

        def refresh(self, _):
            running = run_control.running_pid() is not None
            stopping = running and run_control.STOP_FILE.exists()
            self.title = "●" if running else None
            self.status.title = (
                "Stopping…" if stopping else "Working on a task" if running else "Idle"
            )
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
