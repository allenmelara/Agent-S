"""Shared helpers for the Mac feasibility checks.

These checks never touch gui_agents/ and never click, type, or move the mouse.
Every check that talks to Claude goes through `ask_claude`, which refuses to
run unless the billing guard in `subscription_guard` passes.
"""

from __future__ import annotations

import base64
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
REPORT_DIR = HERE / "reports"

PASS, FAIL, WARN, INFO, SKIP = "PASS", "FAIL", "WARN", "INFO", "SKIP"

# Claude Code variables that route requests away from a claude.ai Max login.
FORBIDDEN_CLAUDE_ENV = {
    "ANTHROPIC_API_KEY": "bills the Claude API (pay-per-token) instead of your Max plan",
    "ANTHROPIC_AUTH_TOKEN": "sends a bearer token to the API or a gateway instead of your Max login",
    "ANTHROPIC_BASE_URL": "routes requests to a different endpoint (proxy or gateway)",
    "CLAUDE_CODE_USE_BEDROCK": "uses Amazon Bedrock (billed by AWS)",
    "CLAUDE_CODE_USE_VERTEX": "uses Google Cloud Vertex AI (billed by Google)",
    "CLAUDE_CODE_USE_FOUNDRY": "uses Microsoft Foundry (billed by Microsoft)",
    "CLAUDE_CODE_OAUTH_TOKEN": (
        "a long-lived token whose plan these checks cannot verify; unset it and "
        "sign in with `claude auth login` instead"
    ),
}

# Variables Agent S reads (gui_agents/s3/core/engine.py, mllm.py). Keys for
# hosted providers are refused; endpoint URLs are allowed only on this Mac.
AGENT_S_KEY_ENV = [
    "OPENAI_API_KEY",
    "OPENAI_ORG_ID",
    "AZURE_OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "PARASAIL_API_KEY",
    "DEEPSEEK_API_KEY",
    "QWEN_API_KEY",
    "HF_TOKEN",
    "vLLM_API_KEY",
]
AGENT_S_URL_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "GEMINI_ENDPOINT_URL",
    "OPEN_ROUTER_ENDPOINT_URL",
    "DEEPSEEK_ENDPOINT_URL",
    "QWEN_ENDPOINT_URL",
    "HF_ENDPOINT_URL",
    "vLLM_ENDPOINT_URL",
    "OLLAMA_HOST",
]

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def is_local_url(url: str) -> bool:
    if "://" not in url:
        url = "http://" + url
    host = (urlparse(url).hostname or "").lower()
    return host in LOCAL_HOSTS


class Report:
    """Collects PASS/FAIL/WARN/INFO/SKIP lines and writes reports/<id>.json."""

    def __init__(self, check_id: str, title: str):
        self.check_id = check_id
        self.title = title
        self.results: list[dict] = []
        self.data: dict = {}
        print(f"\n=== {check_id}: {title} ===")

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.results.append({"status": status, "name": name, "detail": detail})
        line = f"[{status}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)

    def overall(self) -> str:
        statuses = {r["status"] for r in self.results}
        if FAIL in statuses:
            return FAIL
        if WARN in statuses:
            return WARN
        if PASS in statuses:
            return PASS
        return SKIP

    def finish(self) -> int:
        REPORT_DIR.mkdir(exist_ok=True)
        overall = self.overall()
        payload = {
            "check": self.check_id,
            "title": self.title,
            "overall": overall,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "machine": {
                "system": platform.system(),
                "release": platform.mac_ver()[0] or platform.release(),
                "arch": platform.machine(),
                "python": platform.python_version(),
            },
            "results": self.results,
            "data": self.data,
        }
        path = REPORT_DIR / f"{self.check_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"--- {self.check_id} overall: {overall}  (report: {path.relative_to(REPO_ROOT)})")
        return 1 if overall == FAIL else 0


def require_macos(report: Report) -> bool:
    if platform.system() != "Darwin":
        report.add(SKIP, "macOS required", f"running on {platform.system()}; run this on your Mac")
        return False
    return True


# --------------------------------------------------------------------------
# Billing guard
# --------------------------------------------------------------------------


def _settings_files() -> list[Path]:
    home = Path.home()
    return [
        home / ".claude" / "settings.json",
        REPO_ROOT / ".claude" / "settings.json",
        REPO_ROOT / ".claude" / "settings.local.json",
        Path("/Library/Application Support/ClaudeCode/managed-settings.json"),
    ]


def _dotenv_keys(path: Path) -> list[str]:
    keys = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].replace("export ", "").strip()
        value = line.split("=", 1)[1].strip().strip("'\"")
        if value:
            keys.append(key)
    return keys


def bundled_cli_path() -> str | None:
    """The Claude Code binary the Agent SDK actually launches."""
    try:
        import claude_agent_sdk
    except ImportError:
        return None
    path = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    return str(path) if path.is_file() else shutil.which("claude")


def _auth_status(cli: str) -> dict | None:
    try:
        proc = subprocess.run(
            [cli, "auth", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def subscription_guard(report: Report) -> bool:
    """Return True only if Claude calls would go to a claude.ai Max login.

    Checks environment variables, Claude Code settings files, Agent S's .env,
    and `claude auth status` for both the installed CLI and the SDK's bundled
    CLI. Prints only variable names, never their values.
    """
    ok = True

    for var, why in FORBIDDEN_CLAUDE_ENV.items():
        if os.environ.get(var):
            report.add(FAIL, f"env {var} is set", f"refusing: it {why}. Run `unset {var}`.")
            ok = False

    for var in AGENT_S_KEY_ENV:
        if os.environ.get(var):
            report.add(FAIL, f"env {var} is set", "hosted-provider key for Agent S; refusing. Unset it.")
            ok = False
    for var in AGENT_S_URL_ENV:
        value = os.environ.get(var)
        if value and not is_local_url(value):
            report.add(FAIL, f"env {var} points off this Mac", "only localhost endpoints are allowed")
            ok = False

    dotenv = REPO_ROOT / ".env"
    if dotenv.is_file():
        bad = [
            k
            for k in _dotenv_keys(dotenv)
            if k in FORBIDDEN_CLAUDE_ENV or k in AGENT_S_KEY_ENV
        ]
        if bad:
            report.add(FAIL, ".env defines provider keys", ", ".join(bad) + " (Agent S loads this file)")
            ok = False

    for path in _settings_files():
        if not path.is_file():
            continue
        try:
            settings = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            report.add(WARN, f"could not parse {path}")
            continue
        if settings.get("apiKeyHelper"):
            report.add(FAIL, f"{path} sets apiKeyHelper", "that supplies an API key; remove it")
            ok = False
        if settings.get("forceLoginMethod") == "console":
            report.add(FAIL, f"{path} sets forceLoginMethod=console", "that forces API (Console) billing")
            ok = False
        env_block = settings.get("env") or {}
        bad = [k for k in env_block if k in FORBIDDEN_CLAUDE_ENV]
        if bad:
            report.add(FAIL, f"{path} env block", "sets " + ", ".join(bad))
            ok = False

    clis = {}
    installed = shutil.which("claude")
    if installed:
        clis["installed CLI"] = installed
    bundled = bundled_cli_path()
    if bundled and bundled != installed:
        clis["Agent SDK bundled CLI"] = bundled
    if not clis:
        report.add(FAIL, "Claude Code CLI not found", "install Claude Code and/or claude-agent-sdk")
        return False

    for label, cli in clis.items():
        status = _auth_status(cli)
        if status is None:
            report.add(FAIL, f"{label}: auth status unreadable", cli)
            ok = False
            continue
        report.data[f"auth_status[{label}]"] = {
            k: status.get(k)
            for k in ("loggedIn", "authMethod", "apiProvider", "apiKeySource", "subscriptionType")
        }
        problems = []
        if not status.get("loggedIn"):
            problems.append("not logged in (run `claude auth login`)")
        if status.get("authMethod") != "claude.ai":
            problems.append(f"authMethod={status.get('authMethod')!r}, expected 'claude.ai'")
        if status.get("apiProvider") != "firstParty":
            problems.append(f"apiProvider={status.get('apiProvider')!r}, expected 'firstParty'")
        if status.get("apiKeySource"):
            problems.append(f"an API key is configured (source: {status['apiKeySource']})")
        sub = str(status.get("subscriptionType") or "")
        if not sub.lower().startswith("max"):
            problems.append(f"subscriptionType={sub or None!r}, expected a Max plan")
        if problems:
            report.add(FAIL, f"{label}: subscription login", "; ".join(problems))
            ok = False
        else:
            report.add(PASS, f"{label}: subscription login", f"claude.ai login, plan={sub}")

    if ok:
        ok = _sdk_account_check(report)
    return ok


def _sdk_account_check(report: Report) -> bool:
    """Connect the SDK exactly as the checks do and read the account it reports.

    The connect handshake makes no model request.
    """
    try:
        import asyncio

        from claude_agent_sdk import ClaudeSDKClient
    except ImportError:
        report.add(FAIL, "Agent SDK account check", "claude-agent-sdk not installed")
        return False

    async def probe():
        with tempfile.TemporaryDirectory(prefix="agent-s-check-") as workdir:
            async with ClaudeSDKClient(options=sdk_options(None, "probe", workdir)) as client:
                info = await client.get_server_info() or {}
                return info.get("account") or {}

    try:
        account = asyncio.run(probe())
    except Exception as exc:
        report.add(FAIL, "Agent SDK account check", f"could not start the SDK: {type(exc).__name__}: {exc}")
        return False
    sub = str(account.get("subscriptionType") or "")
    provider = account.get("apiProvider")
    report.data["sdk_account"] = {"subscriptionType": sub, "apiProvider": provider}
    if provider == "firstParty" and "max" in sub.lower():
        report.add(PASS, "Agent SDK session account", f"{sub} via {provider} (live handshake, no model call)")
        return True
    report.add(FAIL, "Agent SDK session account", f"subscriptionType={sub!r}, apiProvider={provider!r}; expected a Max plan")
    return False


# --------------------------------------------------------------------------
# Agent SDK wrapper (tools disabled, no settings, no MCP, no transcripts)
# --------------------------------------------------------------------------


class BillingGuardError(RuntimeError):
    pass


def sdk_options(model: str | None, system_prompt: str, workdir: str):
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(
        tools=[],  # no built-in tools at all
        allowed_tools=[],
        system_prompt=system_prompt,
        setting_sources=[],  # ignore user/project hooks, CLAUDE.md, plugins
        strict_mcp_config=True,  # no MCP servers
        max_turns=1,
        model=model,
        cwd=workdir,  # empty temp dir, so no project files are visible
        extra_args={"no-session-persistence": None},  # don't save screenshots to transcripts
    )


def image_block(png_bytes: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(png_bytes).decode(),
        },
    }


def user_message(content: list[dict] | str) -> dict:
    return {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
    }


class TurnMetrics:
    def __init__(self):
        self.wall_s = 0.0
        self.first_token_s: float | None = None
        self.api_ms: int | None = None
        self.est_cost_usd: float | None = None
        self.usage: dict | None = None
        self.model: str | None = None
        self.api_key_source: str | None = None
        self.rate_limits: list[dict] = []
        self.tools: list[str] = []
        self.tool_calls = 0
        self.is_error = False

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def check_turn_messages(msg, metrics: TurnMetrics, started: float) -> str | None:
    """Update metrics from one SDK message; return text if it carries any.

    Raises BillingGuardError when the stream shows API-key or overage billing.
    """
    from claude_agent_sdk import (
        AssistantMessage,
        RateLimitEvent,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ToolUseBlock,
    )

    if isinstance(msg, SystemMessage) and msg.subtype == "init":
        source = msg.data.get("apiKeySource")
        metrics.api_key_source = source
        metrics.model = msg.data.get("model")
        if source not in (None, "none"):
            raise BillingGuardError(f"session started with apiKeySource={source!r}; aborting")
        metrics.tools = list(msg.data.get("tools") or [])
    elif isinstance(msg, RateLimitEvent):
        info = msg.rate_limit_info
        metrics.rate_limits.append(
            {
                "status": info.status,
                "type": info.rate_limit_type,
                "utilization": info.utilization,
                "overage_status": info.overage_status,
                "resets_at": info.resets_at,
            }
        )
        if info.rate_limit_type == "overage":
            raise BillingGuardError("this request would draw on extra (billed) usage; aborting")
        if info.status == "rejected":
            raise BillingGuardError(f"Max usage limit reached ({info.rate_limit_type}); try after reset")
    elif isinstance(msg, AssistantMessage):
        if metrics.first_token_s is None:
            metrics.first_token_s = time.perf_counter() - started
        texts = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                texts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                metrics.tool_calls += 1
        return "".join(texts)
    elif isinstance(msg, ResultMessage):
        metrics.api_ms = msg.duration_api_ms
        metrics.est_cost_usd = msg.total_cost_usd
        metrics.usage = msg.usage
        metrics.is_error = msg.is_error
    return None


async def ask_claude(
    content: list[dict] | str,
    *,
    system_prompt: str,
    model: str | None = None,
) -> tuple[str, TurnMetrics]:
    """One fresh Agent SDK session, one turn, tools disabled."""
    from claude_agent_sdk import query

    metrics = TurnMetrics()
    text_parts: list[str] = []

    async def prompt_stream():
        yield user_message(content)

    with tempfile.TemporaryDirectory(prefix="agent-s-check-") as workdir:
        started = time.perf_counter()
        async for msg in query(
            prompt=prompt_stream(),
            options=sdk_options(model, system_prompt, workdir),
        ):
            text = check_turn_messages(msg, metrics, started)
            if text:
                text_parts.append(text)
        metrics.wall_s = time.perf_counter() - started
    return "".join(text_parts), metrics


def describe_usage_warnings(report: Report, metrics: TurnMetrics) -> None:
    for rl in metrics.rate_limits:
        if rl.get("overage_status") == "allowed":
            report.add(
                WARN,
                "extra usage is enabled on your account",
                "once Max limits run out, further usage could be billed; turn it off in "
                "claude.ai Settings > Usage if you want a hard stop",
            )
            return


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def ensure_python() -> None:
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10+ is required (claude-agent-sdk needs it). See mac_feasibility/README.md.")
