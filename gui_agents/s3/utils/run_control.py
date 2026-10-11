"""Ways to stop an Agent S run that is working on its own (cli_app.py).

- Move the mouse into the top-left corner of the screen.
- Click the Agent S app in the Dock again and choose Stop, or choose Stop
  task in the menu-bar icon. Both create ~/.agent_s/stop, which the run
  checks before every step and every action.
- Ctrl+C in the Terminal window, as before.
"""

import os
from pathlib import Path

STATE_DIR = Path.home() / ".agent_s"
RUNNING_FILE = STATE_DIR / "running.pid"  # read by mac_app/build_agent_s_app.sh
STOP_FILE = STATE_DIR / "stop"
CORNER_PX = 3  # points from the top-left corner that count as "in the corner"


def start_run() -> None:
    """Record that a run is active and clear any stale stop request."""
    STATE_DIR.mkdir(exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)
    RUNNING_FILE.write_text(str(os.getpid()))


def finish_run() -> None:
    STOP_FILE.unlink(missing_ok=True)
    try:
        if RUNNING_FILE.read_text().strip() == str(os.getpid()):
            RUNNING_FILE.unlink()
    except OSError:
        pass


def running_pid() -> int | None:
    """PID of the active run, or None if no run is active (stale files ignored)."""
    try:
        pid = int(RUNNING_FILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def request_stop() -> None:
    """Ask the active run to stop before its next step or action."""
    STATE_DIR.mkdir(exist_ok=True)
    STOP_FILE.touch()


def stop_reason(mouse_position) -> str | None:
    """Why the run should stop now, or None. mouse_position() -> (x, y)."""
    if STOP_FILE.exists():
        return "stop requested from the Agent S app"
    try:
        x, y = mouse_position()
    except Exception:
        return None
    if x <= CORNER_PX and y <= CORNER_PX:
        return "mouse moved to the top-left corner"
    return None
