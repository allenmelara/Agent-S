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

        params = {"engine_type": "claude_subscription", "model": None}
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
