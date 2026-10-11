"""Tests for the claude_subscription engine. A fake Agent SDK stands in for
Claude Code, so no model calls are made."""

import asyncio
import os
import unittest
from unittest.mock import patch

import claude_agent_sdk
from claude_agent_sdk import (
    AssistantMessage,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    SystemMessage,
    TextBlock,
)

from gui_agents.s3.core import claude_subscription as cs
from gui_agents.s3.core.mllm import LMMAgent
from gui_agents.s3.utils.common_utils import call_llm_safe


def fake_query(*stream, seen=None):
    """Build a stand-in for claude_agent_sdk.query that yields `stream`."""

    async def query(prompt, options):
        if seen is not None:
            async for msg in prompt:
                seen.append((msg, options))
        for msg in stream:
            yield msg

    return query


INIT_OK = SystemMessage("init", {"apiKeySource": "none", "tools": []})
RESULT_OK = ResultMessage("success", 10, 5, False, 1, "s", total_cost_usd=0.01)


class GuardBypass(unittest.TestCase):
    def setUp(self):
        cs._guard_passed = True
        cs.set_call_budget(None)

    def tearDown(self):
        cs._guard_passed = False


class TestPromptConversion(unittest.TestCase):
    def test_single_user_message_passes_blocks_through(self):
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}}
        system, blocks = cs.to_sdk_prompt(
            [
                {"role": "system", "content": [{"type": "text", "text": "SYS"}]},
                {"role": "user", "content": [{"type": "text", "text": "hi"}, image]},
            ]
        )
        self.assertEqual(system, "SYS")
        self.assertEqual(blocks, [{"type": "text", "text": "hi"}, image])

    def test_history_is_folded_in_order(self):
        img1 = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "1"}}
        img2 = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "2"}}
        _, blocks = cs.to_sdk_prompt(
            [
                {"role": "system", "content": [{"type": "text", "text": "SYS"}]},
                {"role": "user", "content": [{"type": "text", "text": "step 1"}, img1]},
                {"role": "assistant", "content": [{"type": "text", "text": "did A"}]},
                {"role": "user", "content": [{"type": "text", "text": "step 2"}, img2]},
            ]
        )
        texts = [b.get("text") for b in blocks]
        self.assertIn("[Your earlier reply]\ndid A", texts)
        self.assertEqual(texts[-3], "[Current message]")
        self.assertLess(blocks.index(img1), blocks.index(img2))


class TestEngine(GuardBypass):
    def make_agent(self, **params):
        params.setdefault("persistent", False)  # these tests cover fresh sessions
        return LMMAgent({"engine_type": "claude_subscription", "model": None, **params})

    def test_returns_text_and_disables_tools(self):
        seen = []
        stream = (INIT_OK, AssistantMessage([TextBlock("hello")], "m"), RESULT_OK)
        with patch.object(claude_agent_sdk, "query", fake_query(*stream, seen=seen)):
            agent = self.make_agent()
            agent.add_message("hi", image_content=b"\x89PNG", role="user")
            self.assertEqual(agent.get_response(), "hello")
        msg, options = seen[0]
        self.assertEqual(options.tools, [])
        self.assertEqual(options.setting_sources, [])
        self.assertIn("no-session-persistence", options.extra_args)
        self.assertEqual(msg["message"]["content"][1]["type"], "image")  # Anthropic image format

    def test_grounding_system_prompt_override(self):
        seen = []
        stream = (INIT_OK, AssistantMessage([TextBlock("10 20")], "m"), RESULT_OK)
        with patch.object(claude_agent_sdk, "query", fake_query(*stream, seen=seen)):
            agent = self.make_agent(system_prompt_override=cs.GROUNDING_SYSTEM_PROMPT)
            agent.add_message("Query:the Save button", role="user")
            agent.get_response()
        self.assertEqual(seen[0][1].system_prompt, cs.GROUNDING_SYSTEM_PROMPT)

    def test_refuses_api_key_and_url(self):
        with self.assertRaises(cs.SubscriptionStop):
            self.make_agent(api_key="sk-x")
        with self.assertRaises(cs.SubscriptionStop):
            self.make_agent(base_url="http://example.com")

    def test_stops_on_api_key_source(self):
        stream = (SystemMessage("init", {"apiKeySource": "ANTHROPIC_API_KEY"}),)
        with patch.object(claude_agent_sdk, "query", fake_query(*stream)):
            agent = self.make_agent()
            agent.add_message("hi", role="user")
            with self.assertRaises(cs.SubscriptionStop):
                agent.get_response()

    def test_stops_on_overage_and_limit(self):
        for info in (
            RateLimitInfo(status="allowed", rate_limit_type="overage"),
            RateLimitInfo(status="rejected", rate_limit_type="five_hour"),
        ):
            stream = (INIT_OK, RateLimitEvent(info, "u", "s"))
            with patch.object(claude_agent_sdk, "query", fake_query(*stream)):
                agent = self.make_agent()
                agent.add_message("hi", role="user")
                with self.assertRaises(cs.SubscriptionStop):
                    agent.get_response()

    def test_stop_is_not_swallowed_by_retry_helper(self):
        stream = (SystemMessage("init", {"apiKeySource": "ANTHROPIC_API_KEY"}),)
        with patch.object(claude_agent_sdk, "query", fake_query(*stream)):
            agent = self.make_agent()
            agent.add_message("hi", role="user")
            with self.assertRaises(cs.SubscriptionStop):
                call_llm_safe(agent)

    def test_call_budget(self):
        cs.set_call_budget(2)
        stream = (INIT_OK, AssistantMessage([TextBlock("ok")], "m"), RESULT_OK)
        with patch.object(claude_agent_sdk, "query", fake_query(*stream)):
            agent = self.make_agent()
            agent.add_message("hi", role="user")
            agent.get_response()
            agent.get_response()
            with self.assertRaises(cs.SubscriptionStop):
                agent.get_response()
        self.assertAlmostEqual(cs.usage_summary()["est_cost_usd"], 0.02)

    def test_works_inside_running_event_loop(self):
        stream = (INIT_OK, AssistantMessage([TextBlock("ok")], "m"), RESULT_OK)

        async def inside():
            agent = self.make_agent()
            agent.add_message("hi", role="user")
            return agent.get_response()

        with patch.object(claude_agent_sdk, "query", fake_query(*stream)):
            self.assertEqual(asyncio.run(inside()), "ok")


class TestGroundingCache(GuardBypass):
    def test_same_screenshot_and_description_grounds_once(self):
        from gui_agents.s3.agents.grounding import OSWorldACI

        calls = []

        async def query(prompt, options):
            async for _ in prompt:
                pass
            calls.append(options.system_prompt)
            yield INIT_OK
            yield AssistantMessage([TextBlock("10 20")], "m")
            yield RESULT_OK

        params = {"engine_type": "claude_subscription", "model": None, "persistent": False}
        ground = dict(params, grounding_width=100, grounding_height=100,
                      system_prompt_override=cs.GROUNDING_SYSTEM_PROMPT)
        aci = OSWorldACI(env=None, platform="darwin", engine_params_for_generation=params,
                         engine_params_for_grounding=ground, width=100, height=100)
        with patch.object(claude_agent_sdk, "query", query):
            a = aci.generate_coords("Save", {"screenshot": b"shot-1"})
            b = aci.generate_coords("Save", {"screenshot": b"shot-1"})
            aci.generate_coords("Save", {"screenshot": b"shot-2"})
        self.assertEqual(a, [10, 20])
        self.assertEqual(a, b)
        self.assertEqual(len(calls), 2)  # one per distinct screenshot


class FakeClient:
    """Stands in for ClaudeSDKClient: keeps history until /clear."""

    instances = []
    clear_works = True
    clear_hangs = False
    api_key_source = "none"

    def __init__(self, options):
        self.options = options
        self.history = []
        self.sent = []
        self.connected = self.disconnected = False
        self._pending = []
        FakeClient.instances.append(self)

    async def connect(self, prompt=None):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

    async def query(self, prompt, session_id="default"):
        if prompt == "/clear":
            self.sent.append("/clear")
            if FakeClient.clear_works:
                self.history = []
            self._pending = [] if FakeClient.clear_hangs else [RESULT_OK]
            return
        async for msg in prompt:
            text = "".join(b.get("text", "") for b in msg["message"]["content"])
            self.sent.append(text)
            self.history.append(text)
        self._pending = [
            SystemMessage("init", {"apiKeySource": FakeClient.api_key_source, "tools": []}),
            AssistantMessage([TextBlock(f"reply to {text}")], "m"),
            RESULT_OK,
        ]

    async def receive_response(self):
        if not self._pending:  # a /clear that never answers
            await asyncio.sleep(3600)
        for msg in self._pending:
            yield msg

    async def get_context_usage(self):
        return {"totalTokens": 1000 + 3000 * len(self.history)}


class TestPersistentSessions(GuardBypass):
    def setUp(self):
        super().setUp()
        FakeClient.instances = []
        FakeClient.clear_works, FakeClient.clear_hangs, FakeClient.api_key_source = True, False, "none"

    def agent(self, system="SYS"):
        agent = LMMAgent({"engine_type": "claude_subscription", "model": None}, system_prompt=system)
        return agent

    def ask(self, agent, text):
        agent.messages = agent.messages[:1]
        agent.add_message(text, role="user")
        return agent.get_response()

    def test_one_session_cleared_between_calls(self):
        with patch.object(claude_agent_sdk, "ClaudeSDKClient", FakeClient):
            agent = self.agent()
            self.assertEqual(self.ask(agent, "a"), "reply to a")
            self.assertEqual(self.ask(agent, "b"), "reply to b")
        self.assertEqual(len(FakeClient.instances), 1)
        self.assertEqual(FakeClient.instances[0].sent, ["a", "/clear", "b"])
        self.assertTrue(agent.engine.persistent)

    def test_new_system_prompt_opens_new_session(self):
        with patch.object(claude_agent_sdk, "ClaudeSDKClient", FakeClient):
            agent = self.agent("one")
            self.ask(agent, "a")
            agent.add_system_prompt("two")
            self.ask(agent, "b")
        self.assertEqual(len(FakeClient.instances), 2)
        self.assertTrue(FakeClient.instances[0].disconnected)
        self.assertEqual(FakeClient.instances[1].options.system_prompt, "two")

    def test_clear_that_keeps_history_falls_back_to_fresh(self):
        FakeClient.clear_works = False
        fresh = (INIT_OK, AssistantMessage([TextBlock("fresh reply")], "m"), RESULT_OK)
        with patch.object(claude_agent_sdk, "ClaudeSDKClient", FakeClient), patch.object(
            claude_agent_sdk, "query", fake_query(*fresh)
        ):
            agent = self.agent()
            self.ask(agent, "a")
            self.assertEqual(self.ask(agent, "b"), "fresh reply")
        self.assertFalse(agent.engine.persistent)
        self.assertTrue(FakeClient.instances[0].disconnected)

    def test_clear_that_hangs_falls_back_to_fresh(self):
        FakeClient.clear_hangs = True
        fresh = (INIT_OK, AssistantMessage([TextBlock("fresh reply")], "m"), RESULT_OK)
        with patch.object(claude_agent_sdk, "ClaudeSDKClient", FakeClient), patch.object(
            claude_agent_sdk, "query", fake_query(*fresh)
        ), patch.object(cs.LMMEngineClaudeSubscription, "CLEAR_TIMEOUT_S", 0.2):
            agent = self.agent()
            self.ask(agent, "a")
            self.assertEqual(self.ask(agent, "b"), "fresh reply")
        self.assertFalse(agent.engine.persistent)

    def test_billing_stop_propagates_and_closes_session(self):
        FakeClient.api_key_source = "ANTHROPIC_API_KEY"
        with patch.object(claude_agent_sdk, "ClaudeSDKClient", FakeClient):
            agent = self.agent()
            with self.assertRaises(cs.SubscriptionStop):
                call_llm_safe(agent)
        self.assertTrue(FakeClient.instances[0].disconnected)
        self.assertIsNone(agent.engine._client)


class TestMacOpen(unittest.TestCase):
    def run_code(self, name, exists):
        from gui_agents.s3.agents.grounding import _macos_open_code

        code = _macos_open_code(name)
        for word in ("done", "fail", "next", "wait", "typewrite", "hotkey"):
            self.assertNotIn(word, code.lower())  # control words / blind typing
        calls = []

        class Result:
            returncode, stderr = 0, b""

        with patch("subprocess.run", lambda argv, **kw: calls.append(argv) or Result()), patch(
            "os.path.exists", return_value=exists
        ), patch("time.sleep"):
            exec(code, {})
        return calls

    def test_app_opens_with_open_a(self):
        self.assertEqual(self.run_code("Calculator", exists=False), [["open", "-a", "Calculator"]])

    def test_existing_path_opens_as_file(self):
        calls = self.run_code("~/Desktop/report.pdf", exists=True)
        self.assertEqual(calls[0][0], "open")
        self.assertTrue(calls[0][1].endswith("Desktop/report.pdf"))

    def test_quotes_in_name_are_safe(self):
        self.assertEqual(self.run_code("Bob's App", exists=False), [["open", "-a", "Bob's App"]])


class TestActionSummary(unittest.TestCase):
    PLAN = (
        "(Previous action verification)\nNo action has been taken yet.\n\n"
        "(Screenshot Analysis)\nThe Calculator app is not open yet.\n\n"
        "(Next Action)\nOpen the Calculator application using the open action.\n\n"
        "(Grounded Action)\n```python\nagent.open(\"Calculator\")\n```"
    )

    def test_shows_plan_call_and_code(self):
        from gui_agents.s3.agents.grounding import _macos_open_code
        from gui_agents.s3.utils.action_summary import describe_action

        code = _macos_open_code("Calculator")
        text = describe_action({"plan": self.PLAN, "plan_code": 'agent.open("Calculator")'}, code)
        self.assertTrue(text.startswith("Next action: Open the Calculator application using the open action."))
        self.assertIn('Agent S call: agent.open("Calculator")', text)
        self.assertIn("Code that will run:\n" + code[:40], text)
        self.assertNotIn("⚠️", text)  # macOS's own launcher is not flagged
        self.assertNotIn("Screenshot Analysis", text)

    def test_flags_shell_and_delete(self):
        from gui_agents.s3.utils.action_summary import describe_action

        text = describe_action({}, "import subprocess; subprocess.run(['rm', '-rf', '/tmp/x'])")
        self.assertIn("⚠️ Runs a shell command on your Mac.", text)
        self.assertIn("⚠️ May delete files.", text)
        self.assertTrue(text.startswith("⚠️"))  # no plan available: warnings come first

    def test_flags_quit_and_handles_missing_info(self):
        from gui_agents.s3.utils.action_summary import describe_action

        text = describe_action(None, "import pyautogui; pyautogui.hotkey('command', 'q')")
        self.assertIn("⚠️ Quits an app (Command+Q).", text)
        self.assertTrue(text.endswith("Run it?"))

    def test_long_plan_is_clipped(self):
        from gui_agents.s3.utils.action_summary import describe_action

        plan = "(Next Action)\n" + "x" * 1000 + "\n(Grounded Action)\n"
        line = describe_action({"plan": plan}, "pass").split("\n\n")[0]
        self.assertLessEqual(len(line), len("Next action: ") + 300)
        self.assertTrue(line.endswith("…"))


class TestApprovalAndStop(unittest.TestCase):
    def test_approval_modes(self):
        from gui_agents.s3.agents.grounding import _macos_open_code
        from gui_agents.s3.utils.action_summary import needs_approval

        click = "import pyautogui; pyautogui.click(427, 715, clicks=1, button='left');"
        shell = "import subprocess; subprocess.run(['ls'])"
        for code, risky in ((click, False), (_macos_open_code("Calculator"), False), (shell, True)):
            self.assertTrue(needs_approval("always", code))
            self.assertEqual(needs_approval("risky", code), risky)
            self.assertFalse(needs_approval("never", code))

    def test_stop_file_and_corner(self):
        import tempfile
        from pathlib import Path

        from gui_agents.s3.utils import run_control as rc

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / ".agent_s"
            with patch.object(rc, "STATE_DIR", state), patch.object(
                rc, "RUNNING_FILE", state / "running.pid"
            ), patch.object(rc, "STOP_FILE", state / "stop"):
                (state).mkdir()
                (state / "stop").touch()  # stale request from an earlier run
                rc.start_run()
                self.assertFalse((state / "stop").exists())
                self.assertEqual((state / "running.pid").read_text(), str(os.getpid()))

                self.assertIsNone(rc.stop_reason(lambda: (700, 400)))
                self.assertIn("top-left corner", rc.stop_reason(lambda: (0, 1)))
                (state / "stop").touch()  # what the Dock app does
                self.assertIn("Agent S app", rc.stop_reason(lambda: (700, 400)))

                rc.finish_run()
                self.assertFalse((state / "running.pid").exists())
                self.assertFalse((state / "stop").exists())

    def test_unreadable_mouse_does_not_stop(self):
        from gui_agents.s3.utils import run_control as rc

        def broken():
            raise RuntimeError("no display")

        with patch.object(rc, "STOP_FILE", rc.STATE_DIR / "definitely-not-there"):
            self.assertIsNone(rc.stop_reason(broken))


class TestMacTyping(GuardBypass):
    def make_aci(self, platform):
        from gui_agents.s3.agents.grounding import OSWorldACI

        params = {"engine_type": "claude_subscription", "model": None, "persistent": False}
        ground = dict(params, grounding_width=100, grounding_height=100)
        return OSWorldACI(env=None, platform=platform, engine_params_for_generation=params,
                          engine_params_for_grounding=ground, width=100, height=100)

    def test_mac_typing_has_no_shell_or_sudo(self):
        from gui_agents.s3.utils.action_summary import needs_approval

        aci = self.make_aci("darwin")
        for text in ("12*7=", "café ☕"):
            code = aci.type(text=text)
            self.assertNotIn("subprocess", code)
            self.assertNotIn("sudo", code)
            self.assertFalse(needs_approval("risky", code))
        self.assertNotIn("pyperclip", aci.type(text="12*7="))
        self.assertIn("import pyperclip; ", aci.type(text="café"))

    def test_linux_unicode_keeps_upstream_fallback(self):
        self.assertIn("apt-get", self.make_aci("linux").type(text="café"))

    def test_mac_prompt_drops_benchmark_password(self):
        from gui_agents.s3.agents.worker import Worker

        params = {"engine_type": "claude_subscription", "model": None, "persistent": False}
        prompt = Worker(params, self.make_aci("darwin"), platform="darwin").generator_agent.system_prompt
        self.assertNotIn("osworld-public-evaluation", prompt)
        self.assertIn("Never use sudo", prompt)


class TestTerminalKeyGuard(unittest.TestCase):
    def test_keyboard_without_click(self):
        from gui_agents.s3.agents.grounding import _macos_open_code
        from gui_agents.s3.utils.action_summary import keyboard_without_click

        # From the failed ChatGPT -> Claude run: these went to whatever app was in front.
        self.assertTrue(keyboard_without_click("import pyautogui; pyautogui.hotkey('cmd', 'v')"))
        self.assertTrue(keyboard_without_click("import pyautogui; pyautogui.hotkey('backspace')"))
        self.assertTrue(keyboard_without_click("import pyautogui; pyautogui.write('12*7=')"))
        # A click first selects the target window, so these are fine.
        self.assertFalse(keyboard_without_click(
            "import pyautogui; pyautogui.click(1297, 977); pyautogui.hotkey('command', 'a'); "
            "pyautogui.press('backspace'); pyautogui.write(''); "
        ))
        self.assertFalse(keyboard_without_click("import pyautogui; pyautogui.click(708, 185, clicks=1, button='left'); "))
        self.assertFalse(keyboard_without_click(_macos_open_code("Claude")))

    def test_is_terminal_app(self):
        from gui_agents.s3.utils.action_summary import is_terminal_app

        self.assertTrue(is_terminal_app("/System/Applications/Utilities/Terminal.app"))
        self.assertTrue(is_terminal_app("/Applications/iTerm.app/"))
        self.assertFalse(is_terminal_app("/Applications/Claude.app"))
        self.assertFalse(is_terminal_app(None))


class TestMacKeys(unittest.TestCase):
    def fake_quartz(self, events):
        import types

        q = types.SimpleNamespace(
            kCGEventSourceStateHIDSystemState=1,
            kCGHIDEventTap=0,
            kCGEventFlagMaskCommand=1 << 20,
            kCGEventFlagMaskShift=1 << 17,
            kCGEventFlagMaskAlternate=1 << 19,
            kCGEventFlagMaskControl=1 << 18,
        )
        q.CGEventSourceCreate = lambda state: "src"
        q.CGEventCreateKeyboardEvent = lambda src, code, down: {"code": code, "down": down}
        q.CGEventSetFlags = lambda event, flags: event.update(flags=flags)
        q.CGEventPost = lambda tap, event: events.append(event)
        return q

    def test_command_v_carries_the_command_flag(self):
        from gui_agents.s3.utils import mac_keys

        events = []
        with patch.dict("sys.modules", {"Quartz": self.fake_quartz(events)}), patch.object(mac_keys.time, "sleep"):
            mac_keys.press_combo(["cmd", "v"])
        cmd = 1 << 20
        self.assertEqual(
            events,
            [
                {"code": 55, "down": True, "flags": cmd},  # Command down
                {"code": 9, "down": True, "flags": cmd},  # V down WITH Command flag
                {"code": 9, "down": False, "flags": cmd},
                {"code": 55, "down": False, "flags": 0},  # Command up
            ],
        )

    def test_unknown_key_falls_back_to_pyautogui(self):
        from gui_agents.s3.utils import mac_keys

        calls = []
        fake_pyautogui = type("P", (), {"hotkey": staticmethod(lambda *k, **kw: calls.append(k))})
        with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
            mac_keys.press_combo(["command", "f13"])
        self.assertEqual(calls, [("command", "f13")])

    def test_mac_actions_use_press_combo(self):
        from gui_agents.s3.agents.grounding import OSWorldACI
        from gui_agents.s3.utils.action_summary import action_warnings, keyboard_without_click

        params = {"engine_type": "claude_subscription", "model": None, "persistent": False}
        ground = dict(params, grounding_width=100, grounding_height=100)
        mac = OSWorldACI(env=None, platform="darwin", engine_params_for_generation=params,
                         engine_params_for_grounding=ground, width=100, height=100)
        code = mac.hotkey(["cmd", "v"])
        self.assertIn("press_combo(['cmd', 'v'])", code)
        self.assertNotIn("pyautogui.hotkey", code)
        self.assertTrue(keyboard_without_click(code))  # still blocked while Terminal is in front
        self.assertIn("Quits an app (Command+Q).", action_warnings(mac.hotkey(["command", "q"])))
        typed = mac.type(text="café", overwrite=True)
        self.assertIn("press_combo(['command', 'a'])", typed)
        self.assertIn("press_combo(['command', 'v'])", typed)
        self.assertNotIn("pyautogui.hotkey", typed)
        linux = OSWorldACI(env=None, platform="linux", engine_params_for_generation=params,
                           engine_params_for_grounding=ground, width=100, height=100)
        self.assertEqual(linux.hotkey(["ctrl", "v"]), "import pyautogui; pyautogui.hotkey('ctrl', 'v')")


class TestWatchClaude(unittest.TestCase):
    def setUp(self):
        import importlib.util
        import tempfile
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "mac_app" / "watch_claude.py"
        spec = importlib.util.spec_from_file_location("agent_s_watch", path)
        self.wc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.wc)
        self.tmp = tempfile.TemporaryDirectory()
        state = Path(self.tmp.name)
        self.patches = [
            patch.object(self.wc, name, state / filename)
            for name, filename in (
                ("STATE_DIR", ""), ("PID_FILE", "watch.pid"), ("STOP_FILE", "watch_stop"),
                ("NOW_FILE", "watch_now"), ("STATE_FILE", "watch_state.json"), ("LOG_FILE", "watch.log"),
            )
        ]
        for p in self.patches:
            p.start()
        self.state = state

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def window(self, lines, spinner=0):
        """A fake Claude window: `lines` reply lines, and a spinner frame."""
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (900, 700), (30, 30, 30))
        d = ImageDraw.Draw(img)
        for i in range(lines):
            d.rectangle((40, 80 + i * 30, 800, 95 + i * 30), fill=(220, 220, 220))
        if spinner:  # a 16 px spinner, like Claude's "working" indicator
            d.ellipse((40, 560, 56, 576), fill=(200, 120, 40) if spinner % 2 else (40, 40, 40))
        return self.wc.signature(img)

    def test_decide(self):
        base = self.window(5)
        self.assertEqual(self.wc.decide(base, None, None), "no-window")
        self.assertEqual(self.wc.decide(base, self.window(5), self.window(5)), "unchanged")
        self.assertEqual(self.wc.decide(base, self.window(9, spinner=1), self.window(9, spinner=2)), "working")
        self.assertEqual(self.wc.decide(base, self.window(12), self.window(12)), "relay")
        self.assertEqual(self.wc.decide(None, self.window(5), self.window(5)), "relay")  # "Check now"

    def test_ignores_typing_in_the_input_box(self):
        from PIL import Image, ImageDraw

        def with_draft(text_width):
            img = Image.new("RGB", (900, 700), (30, 30, 30))
            d = ImageDraw.Draw(img)
            for i in range(5):
                d.rectangle((40, 80 + i * 30, 800, 95 + i * 30), fill=(220, 220, 220))
            d.rectangle((40, 640, 40 + text_width, 660), fill=(255, 255, 255))  # bottom 15%: input box
            return self.wc.signature(img)

        self.assertEqual(self.wc.decide(with_draft(0), with_draft(600), with_draft(600)), "unchanged")

    def test_wait_for_stop_now_timeout(self):
        sleeps = []
        self.assertEqual(self.wc.wait_for(6, sleep=sleeps.append), "timeout")
        self.assertEqual(sum(sleeps), 6)
        self.wc.NOW_FILE.touch()
        self.assertEqual(self.wc.wait_for(600, sleep=lambda s: None), "now")
        self.assertFalse(self.wc.NOW_FILE.exists())
        self.wc.STOP_FILE.touch()
        self.assertEqual(self.wc.wait_for(600, sleep=lambda s: None), "stop")

    def test_relay_command_and_watcher_pid(self):
        cmd = self.wc.relay_command("risky")
        self.assertEqual(cmd[cmd.index("--task") + 1], self.wc.RELAY_TASK)
        self.assertEqual(cmd[cmd.index("--approval") + 1], "risky")
        self.assertIsNone(self.wc.watcher_pid())
        self.wc.PID_FILE.write_text(str(os.getpid()))
        self.assertEqual(self.wc.watcher_pid(), os.getpid())

    def test_check_once_relays_only_new_settled_reply(self):
        base, new = self.window(5), self.window(12)
        state = {"baseline": base.hex()}
        runs = []

        def check(returncode, clip_before, clip_after):
            captures = iter([new, new])
            clip = iter([clip_before, clip_after])

            class Done:
                pass

            def run(cmd, **kw):
                runs.append(cmd)
                result = Done()
                result.returncode = returncode
                return result

            with patch.object(self.wc, "claude_window_id", return_value=7), patch.object(
                self.wc, "capture_signature", lambda w: next(captures)
            ), patch.object(self.wc, "clipboard_hash", lambda: next(clip)), patch.object(
                self.wc.subprocess, "run", run
            ), patch.object(self.wc, "SETTLE_SECONDS", 0), patch.object(self.wc, "notify"), patch.object(
                self.wc.run_control, "running_pid", return_value=None
            ):
                return self.wc.check_once(state, forced=False, approval="risky")

        # Copied but the paste failed (the run that logged a false "Handoff done"): retry later.
        self.assertEqual(check(1, "old", "new-reply"), "failed")
        self.assertEqual(state["baseline"], base.hex())
        self.assertNotIn("last_sent", state)
        # Finished and copied: sent.
        self.assertEqual(check(0, "old", "new-reply"), "sent")
        self.assertEqual(state["baseline"], new.hex())
        self.assertEqual(state["last_sent"], "new-reply")
        self.assertIn("--quiet_done", runs[0])
        self.assertEqual(len(runs), 2)

    def test_check_once_skips_when_busy_or_unchanged(self):
        base = self.window(5)
        runs = []
        with patch.object(self.wc.subprocess, "run", lambda cmd, **kw: runs.append(cmd)), patch.object(
            self.wc, "SETTLE_SECONDS", 0
        ):
            with patch.object(self.wc.run_control, "running_pid", return_value=123):
                self.assertEqual(self.wc.check_once({"baseline": base.hex()}, False, "risky"), "busy")
            with patch.object(self.wc.run_control, "running_pid", return_value=None), patch.object(
                self.wc, "claude_window_id", return_value=7
            ), patch.object(self.wc, "capture_signature", lambda w: base):
                self.assertEqual(self.wc.check_once({"baseline": base.hex()}, False, "risky"), "unchanged")
        self.assertEqual(runs, [])


class TestMenuBar(unittest.TestCase):
    def setUp(self):
        import importlib.util
        import tempfile
        from pathlib import Path

        from gui_agents.s3.utils import run_control as rc

        path = Path(__file__).resolve().parent.parent / "mac_app" / "menubar.py"
        spec = importlib.util.spec_from_file_location("agent_s_menubar", path)
        self.mb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mb)

        self.tmp = tempfile.TemporaryDirectory()
        state = Path(self.tmp.name) / ".agent_s"
        self.rc = rc
        self.patches = [
            patch.object(rc, "STATE_DIR", state),
            patch.object(rc, "RUNNING_FILE", state / "running.pid"),
            patch.object(rc, "STOP_FILE", state / "stop"),
            patch.object(self.mb, "SETTINGS_FILE", state / "menubar.json"),
            patch.object(self.mb, "LOCK_FILE", state / "menubar.pid"),
        ]
        for p in self.patches:
            p.start()
        self.state = state

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_task_command_quotes_the_task(self):
        import shlex

        task = """Write "hi" in Notes; rm -rf ~ && echo $(whoami) `x` it's"""
        cmd = self.mb.task_command(task, "risky")
        words = shlex.split(cmd)
        self.assertEqual(words[words.index("--task") + 1], task)  # one literal argument
        self.assertEqual(words[words.index("--approval") + 1], "risky")
        self.assertIn("--provider", words)
        self.assertIn("claude_subscription", words)

    def test_running_and_stop(self):
        self.assertIsNone(self.rc.running_pid())
        self.rc.start_run()
        self.assertEqual(self.rc.running_pid(), os.getpid())
        self.rc.request_stop()
        self.assertIn("Agent S app", self.rc.stop_reason(lambda: (500, 500)))
        self.rc.finish_run()
        self.assertIsNone(self.rc.running_pid())
        self.state.mkdir(exist_ok=True)
        (self.state / "running.pid").write_text("999999999")  # stale: process is gone
        self.assertIsNone(self.rc.running_pid())

    def test_settings_default_and_save(self):
        self.assertEqual(self.mb.load_settings()["approval"], "risky")
        self.mb.save_settings({"approval": "always"})
        self.assertEqual(self.mb.load_settings()["approval"], "always")
        self.mb.save_settings({"approval": "bogus"})
        self.assertEqual(self.mb.load_settings()["approval"], "risky")

    def test_single_copy_lock(self):
        self.assertFalse(self.mb.already_open())  # claims the lock for this process
        self.assertFalse(self.mb.already_open())  # our own pid is not "another copy"
        (self.state / "menubar.pid").write_text(str(os.getppid()))  # a live, different process
        self.assertTrue(self.mb.already_open())


class TestGuard(unittest.TestCase):
    def setUp(self):
        cs._guard_passed = False

    def test_refuses_forbidden_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-x"}), patch.object(
            cs, "_auth_problems", return_value=[]
        ), patch.object(cs, "_bundled_cli", return_value="/bin/true"):
            with self.assertRaises(cs.SubscriptionStop) as ctx:
                cs.ensure_subscription_only()
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_refuses_non_max_login(self):
        with patch.object(cs, "_auth_problems", return_value=["plan is 'pro', not Max"]), patch.object(
            cs, "_bundled_cli", return_value="/bin/true"
        ), patch.object(cs, "_settings_problems", return_value=[]), patch.dict(
            os.environ, {k: "" for k in cs.FORBIDDEN_ENV}
        ):
            with self.assertRaises(cs.SubscriptionStop):
                cs.ensure_subscription_only()

    def test_passes_clean_max_login(self):
        with patch.object(cs, "_auth_problems", return_value=[]), patch.object(
            cs, "_bundled_cli", return_value="/bin/true"
        ), patch.object(cs, "_settings_problems", return_value=[]), patch.dict(
            os.environ, {k: "" for k in cs.FORBIDDEN_ENV}
        ):
            cs.ensure_subscription_only()
        self.assertTrue(cs._guard_passed)
        cs._guard_passed = False


if __name__ == "__main__":
    unittest.main()
