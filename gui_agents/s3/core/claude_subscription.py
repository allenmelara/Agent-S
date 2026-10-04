"""Claude via a claude.ai Max login, through the Claude Agent SDK.

engine_type "claude_subscription" runs each model call as a fresh, tool-less
Claude Code session. It never uses an API key: before the first call it
checks that Claude Code is signed in to claude.ai with a Max plan and that no
API key, gateway or third-party provider is configured, and it stops the
agent if a call reports an API key source, extra (billed) usage, or a reached
usage limit.

Stops are raised as SubscriptionStop, a BaseException, so Agent S's retry
helpers (which catch Exception) cannot swallow them and carry on.

Personal use only: Anthropic does not allow offering claude.ai login in
products for other people.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("desktopenv.agent")

# Variables that would send Claude Code requests somewhere other than a
# claude.ai login, or that hold a key for a hosted Agent S provider.
FORBIDDEN_ENV = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "PARASAIL_API_KEY",
    "DEEPSEEK_API_KEY",
    "QWEN_API_KEY",
    "HF_TOKEN",
]

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

GROUNDING_SYSTEM_PROMPT = (
    "You locate UI elements in screenshots. Coordinates are pixels in the image as "
    "given, origin top-left. Output only two integers: x y."
)


class SubscriptionStop(BaseException):
    """Stop the whole agent run; deliberately not an Exception subclass."""


def is_local_url(url: str) -> bool:
    if not url:
        return False
    if "://" not in url:
        url = "http://" + url
    return (urlparse(url).hostname or "").lower() in LOCAL_HOSTS


# --------------------------------------------------------------------------
# Call budget, shared by every engine instance in the process
# --------------------------------------------------------------------------

_budget_lock = threading.Lock()
_budget = {"max_calls": None, "calls": 0, "est_cost_usd": 0.0}


def set_call_budget(max_calls: int | None) -> None:
    with _budget_lock:
        _budget["max_calls"] = max_calls
        _budget["calls"] = 0
        _budget["est_cost_usd"] = 0.0


def usage_summary() -> dict:
    with _budget_lock:
        return dict(_budget)


def _count_call() -> None:
    with _budget_lock:
        limit = _budget["max_calls"]
        if limit is not None and _budget["calls"] >= limit:
            raise SubscriptionStop(
                f"model-call budget of {limit} reached; raise --max_model_calls to allow more"
            )
        _budget["calls"] += 1


# --------------------------------------------------------------------------
# Billing guard (runs once per process)
# --------------------------------------------------------------------------

_guard_lock = threading.Lock()
_guard_passed = False


def _bundled_cli() -> str | None:
    try:
        import claude_agent_sdk
    except ImportError:
        return None
    path = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    return str(path) if path.is_file() else shutil.which("claude")


def _auth_problems(cli: str) -> list[str]:
    try:
        proc = subprocess.run(
            [cli, "auth", "status", "--json"], capture_output=True, text=True, timeout=60
        )
        status = json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return [f"could not read `claude auth status`: {exc}"]
    problems = []
    if not status.get("loggedIn"):
        problems.append("Claude Code is not logged in (run `claude auth login`)")
    if status.get("authMethod") != "claude.ai":
        problems.append(f"authMethod is {status.get('authMethod')!r}, not 'claude.ai'")
    if status.get("apiProvider") != "firstParty":
        problems.append(f"apiProvider is {status.get('apiProvider')!r}, not 'firstParty'")
    if status.get("apiKeySource"):
        problems.append(f"an API key is configured ({status['apiKeySource']})")
    if not str(status.get("subscriptionType") or "").lower().startswith("max"):
        problems.append(f"plan is {status.get('subscriptionType')!r}, not Max")
    return problems


def _settings_problems() -> list[str]:
    problems = []
    for path in (
        Path.home() / ".claude" / "settings.json",
        Path.cwd() / ".claude" / "settings.json",
        Path.cwd() / ".claude" / "settings.local.json",
        Path("/Library/Application Support/ClaudeCode/managed-settings.json"),
    ):
        if not path.is_file():
            continue
        try:
            settings = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if settings.get("apiKeyHelper"):
            problems.append(f"{path} sets apiKeyHelper")
        if settings.get("forceLoginMethod") == "console":
            problems.append(f"{path} forces Console (API) login")
        bad = [k for k in (settings.get("env") or {}) if k in FORBIDDEN_ENV]
        if bad:
            problems.append(f"{path} env sets {', '.join(bad)}")
    return problems


def ensure_subscription_only() -> None:
    """Raise SubscriptionStop unless calls would go to a claude.ai Max login."""
    global _guard_passed
    with _guard_lock:
        if _guard_passed:
            return
        problems = [f"environment variable {v} is set" for v in FORBIDDEN_ENV if os.environ.get(v)]
        for var in ("OLLAMA_HOST", "vLLM_ENDPOINT_URL", "HF_ENDPOINT_URL"):
            value = os.environ.get(var)
            if value and not is_local_url(value):
                problems.append(f"{var} points off this machine")
        problems += _settings_problems()
        cli = _bundled_cli()
        if cli is None:
            problems.append("claude-agent-sdk is not installed")
        else:
            problems += _auth_problems(cli)
        if problems:
            raise SubscriptionStop(
                "Refusing to call Claude, because it might not use your Max login:\n  - "
                + "\n  - ".join(problems)
            )
        _guard_passed = True


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


def to_sdk_prompt(messages: list[dict]) -> tuple[str, list[dict]]:
    """Turn Agent S's Anthropic-format history into (system prompt, user blocks).

    The SDK takes one user message per turn, so earlier assistant replies are
    folded in as labelled text, keeping every image in order.
    """
    system = _text_of(messages[0]["content"]) if messages and messages[0]["role"] == "system" else ""
    rest = messages[1:] if system or (messages and messages[0]["role"] == "system") else messages
    if len(rest) == 1 and rest[0]["role"] == "user":
        content = rest[0]["content"]
        return system, [{"type": "text", "text": content}] if isinstance(content, str) else list(content)

    blocks: list[dict] = [
        {"type": "text", "text": "Conversation so far (your earlier replies are marked):"}
    ]
    for i, msg in enumerate(rest):
        last = i == len(rest) - 1
        if msg["role"] == "assistant":
            blocks.append({"type": "text", "text": f"[Your earlier reply]\n{_text_of(msg['content'])}"})
        else:
            label = "[Current message]" if last else "[Earlier user message]"
            blocks.append({"type": "text", "text": label})
            content = msg["content"]
            blocks.extend([{"type": "text", "text": content}] if isinstance(content, str) else content)
    return system, blocks


def _run(coro):
    """Run a coroutine from sync code, even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box = {}

    def worker():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised in the caller's thread
            box["error"] = exc

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


class LMMEngineClaudeSubscription:
    """One fresh, tool-less Claude Code session per call, on your Max login."""

    def __init__(
        self,
        model=None,
        api_key=None,
        base_url=None,
        temperature=None,
        system_prompt_override=None,
        **kwargs,
    ):
        if api_key:
            raise SubscriptionStop("claude_subscription never takes an API key; remove --model_api_key/--ground_api_key")
        if base_url:
            raise SubscriptionStop("claude_subscription never takes a base URL; remove --model_url/--ground_url")
        self.model = model or None  # None = your Claude Code default model
        self.system_prompt_override = system_prompt_override
        if temperature not in (None, 0, 0.0):
            logger.info("claude_subscription ignores temperature=%s", temperature)

    def generate(self, messages, temperature=None, max_new_tokens=None, **kwargs):
        ensure_subscription_only()
        _count_call()
        system, blocks = to_sdk_prompt(messages)
        if self.system_prompt_override:
            system = self.system_prompt_override
        return _run(self._ask(system, blocks))

    # Claude Code decides on its own whether to think; Agent S's prompts ask for
    # <thoughts>/<answer> tags in the text, which works the same way.
    generate_with_thinking = generate

    async def _ask(self, system: str, blocks: list[dict]) -> str:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            RateLimitEvent,
            ResultMessage,
            SystemMessage,
            TextBlock,
            query,
        )

        async def prompt():
            yield {
                "type": "user",
                "message": {"role": "user", "content": blocks},
                "parent_tool_use_id": None,
            }

        parts: list[str] = []
        with tempfile.TemporaryDirectory(prefix="agent-s-claude-") as workdir:
            options = ClaudeAgentOptions(
                tools=[],  # no built-in tools; Agent S acts, Claude only answers
                allowed_tools=[],
                system_prompt=system or "You are a helpful assistant.",
                setting_sources=[],  # ignore user/project hooks, CLAUDE.md, plugins
                strict_mcp_config=True,
                max_turns=1,
                model=self.model,
                cwd=workdir,
                extra_args={"no-session-persistence": None},  # keep screenshots out of transcripts
            )
            async for msg in query(prompt=prompt(), options=options):
                if isinstance(msg, SystemMessage) and msg.subtype == "init":
                    source = msg.data.get("apiKeySource")
                    if source not in (None, "none"):
                        raise SubscriptionStop(f"session started with apiKeySource={source!r}")
                    if msg.data.get("tools"):
                        raise SubscriptionStop(f"tools were not disabled: {msg.data['tools']}")
                elif isinstance(msg, RateLimitEvent):
                    info = msg.rate_limit_info
                    if info.rate_limit_type == "overage":
                        raise SubscriptionStop("this call drew on extra (billed) usage; stopping")
                    if info.status == "rejected":
                        raise SubscriptionStop(f"Max usage limit reached ({info.rate_limit_type})")
                elif isinstance(msg, AssistantMessage):
                    parts.extend(b.text for b in msg.content if isinstance(b, TextBlock))
                elif isinstance(msg, ResultMessage):
                    if msg.total_cost_usd:
                        with _budget_lock:
                            _budget["est_cost_usd"] += msg.total_cost_usd
                    if msg.is_error:
                        raise RuntimeError(f"Claude call failed: {msg.result or msg.errors}")
        return "".join(parts)
