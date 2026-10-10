"""Plain-English text for the action confirmation dialog in cli_app.py."""

import re

# Code from grounding._macos_open_code starts with this; it only runs `open`.
_MACOS_OPEN_PREFIX = "import os, subprocess, time; _n = "


def _between(text: str, start: str, end: str) -> str:
    match = re.search(re.escape(start) + r"\s*(.*?)\s*(?:" + re.escape(end) + r"|$)", text, re.S)
    return match.group(1).strip() if match else ""


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def action_warnings(code: str) -> list:
    """Things worth a second look before approving this code."""
    warnings = []
    if code.startswith(_MACOS_OPEN_PREFIX):
        return warnings  # macOS's own app launcher; nothing else runs
    if "subprocess" in code or "os.system" in code or "os.popen" in code:
        warnings.append("Runs a shell command on your Mac.")
    if re.search(r"\b(rm|rmdir|unlink|shutil\.rmtree|os\.remove)\b", code):
        warnings.append("May delete files.")
    if re.search(r"hotkey\([^)]*'command'[^)]*'q'", code):
        warnings.append("Quits an app (Command+Q).")
    return warnings


def describe_action(info, code: str) -> str:
    """Dialog text: what Agent S plans, the action it chose, then the code."""
    info = info if isinstance(info, dict) else {}
    plan = info.get("plan") or ""
    next_action = _between(plan, "(Next Action)", "(Grounded Action)")
    agent_call = (info.get("plan_code") or "").strip()

    parts = []
    if next_action:
        parts.append("Next action: " + _clip(next_action, 300))
    if agent_call:
        parts.append("Agent S call: " + _clip(agent_call, 200))
    for warning in action_warnings(code):
        parts.append("⚠️ " + warning)
    parts.append("Code that will run:\n" + _clip(code, 600))
    parts.append("Run it?")
    return "\n\n".join(parts)
