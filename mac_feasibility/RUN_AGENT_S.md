# Run Agent S on your Mac with your Claude Max login

`--provider claude_subscription` sends every Agent S model call through the
Claude Agent SDK on your claude.ai Max login. That covers planning,
reflection, the code agent and screen grounding. No API key or hosted model
is involved. Personal use only.

**Status:** unit tests and a simulated end-to-end step pass in the cloud
session that wrote this. It has **not yet run on a real Mac.** Start with
the small first task below.

## What changed in Agent S

| File | Change |
|---|---|
| `gui_agents/s3/core/claude_subscription.py` | New engine with:<br>• one fresh, tool-less Claude Code session per call<br>• a billing guard that runs before the first call<br>• stops on an API key source, extra usage or a reached limit<br>• a per-task call budget |
| `gui_agents/s3/core/mllm.py` | Registers `claude_subscription`. It uses the Anthropic image format. |
| `gui_agents/s3/cli_app.py` | Subscription mode:<br>• no URLs or keys accepted<br>• grounding size defaults to the screenshot size<br>• confirmation dialog before every action (default button: Cancel)<br>• `--max_steps`, `--max_model_calls` and `--no_confirm`<br>• usage summary per task<br>Also fixes the macOS dialog, which broke on any quote in the action code. |
| `gui_agents/s3/agents/grounding.py` | Reuses a grounding answer for the same screenshot and description. Upstream asked twice per action: once to validate the code, once to run it. |
| `tests/test_claude_subscription.py` | 14 tests with a fake SDK |

Other providers behave as before. The only difference is that
`--ground_model` and `--grounding_width/height` are now checked in code
instead of by argparse.

## How each step uses your allowance (estimate, not measured)

Each step makes these Claude calls:
- 1 planner call, with up to `--max_trajectory_length` (default 8) earlier
  screenshots;
- 1 reflection call, from step 2 onward;
- 1 grounding call per action that targets something on screen.

That's about 3 calls per step. `--max_model_calls` (default 60) stops a task
before it runs away. Run `/usage` in an interactive `claude` session before
and after a task to see what it really costs.

## Setup (in the `~/Agent-S-checks` worktree you already have)

1. **Update the branch.**
   ```bash
   cd ~/Agent-S-checks
   git pull https://github.com/allenmelara/Agent-S.git claude/vigilant-franklin-5fs60d
   source .venv/bin/activate
   pip install -r mac_feasibility/requirements-agent-s.txt
   brew install tesseract
   ```
2. **Finish check 3 first.**
   `cd mac_feasibility && python check_3_persistent_session.py && cd ..`
3. **Give your terminal Accessibility permission** (System Settings → Privacy
   & Security → Accessibility). Agent S needs it to click and type. Screen
   Recording is already granted.
4. **Close anything private on screen.** Each step sends a screenshot to
   Claude.
5. **Run a small first task** from the worktree root:
   ```bash
   python -m gui_agents.s3.cli_app \
     --provider claude_subscription --ground_provider claude_subscription \
     --max_steps 5 --max_model_calls 20 \
     --task "Open the Calculator app"
   ```
   It prints `✅ Using your Claude Max login (no API key).` before doing
   anything. A dialog shows each action's code; click **OK** to run it or
   **Cancel** to stop. Ctrl+C pauses the agent.
6. **Send back:**
   - the terminal output from `RAW GROUNDING MODEL RESPONSE` to the end,
     including the `Claude calls this task:` line;
   - your `/usage` before and after;
   - whether each action did what it said.

## Options

- `--model sonnet --ground_model sonnet`: a smaller model that may stretch
  your allowance. Its accuracy here is untested; check 5 measured Opus 5.5.
- `--max_trajectory_length 4`: sends fewer old screenshots per planner
  call. The effect on quality is untested.
- `--no_confirm`: skips the dialogs. Only use it after several supervised
  runs.
- Leave `--enable_local_env` off. It lets the agent run arbitrary code
  without the action dialog.

## Known limits

- **Temperature is ignored.** The Agent SDK has no temperature setting.
- **A new session per call.** The ~1.3 s/call saving from a persistent
  session is not used yet. It will be added only if check 3 shows `/clear`
  really drops history.
- **The dialog takes focus.** It briefly takes focus from the target app.
  Agent S usually clicks a field before typing; watch for typing that goes
  to the wrong window.
- **The billing guard checks two places.** It looks at `.claude/settings`
  in the folder you run from and in your home folder.
