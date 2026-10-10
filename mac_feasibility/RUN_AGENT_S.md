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

## Setup (in your repo at `~/Projects/Agent-S`)

1. **Update the branch.**
   ```bash
   cd ~/Projects/Agent-S
   conda deactivate
   source .venv/bin/activate
   git pull --ff-only https://github.com/allenmelara/Agent-S.git claude/vigilant-franklin-5fs60d
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
   anything. A dialog shows Agent S's next action in plain English, the Agent S
   call, any warnings (shell commands, deleting files, quitting apps) and
   the code; click **OK** to run it or
   **Cancel** to stop. Ctrl+C pauses the agent.
6. **Send back:**
   - the terminal output from `RAW GROUNDING MODEL RESPONSE` to the end,
     including the `Claude calls this task:` line;
   - your `/usage` before and after;
   - whether each action did what it said.

## Start Agent S from the Dock

`mac_app/build_agent_s_app.sh` builds **Agent S.app** in `~/Applications`.
Opening it asks what Agent S should do. It then runs Agent S in a Terminal
window with the settings above (Max login, a dialog before every action,
15 steps, 60 calls). Running inside Terminal reuses the permissions you
already gave Terminal, and you can watch the log and stop it with Ctrl+C.

```bash
cd ~/Projects/Agent-S
bash mac_app/build_agent_s_app.sh
```

Then open the app once, right-click its Dock icon, and choose **Options →
Keep in Dock**. Running the builder with `--dock` adds it to the Dock for
you (this restarts the Dock). The first time, macOS asks whether "Agent S"
may control Terminal; click **OK**. Re-run the builder if you move the
repo folder.

## Options

- `--model sonnet --ground_model sonnet`: a smaller model that may stretch
  your allowance. Its accuracy here is untested; check 5 measured Opus 5.5.
- `--max_trajectory_length 4`: sends fewer old screenshots per planner
  call. The effect on quality is untested.
- `--fresh_sessions`: start a new Claude session for every call (the
  slower, original behaviour).
- `--no_confirm`: skips the dialogs. Only use it after several supervised
  runs.
- Leave `--enable_local_env` off. It lets the agent run arbitrary code
  without the action dialog.

## Known limits

- **Temperature is ignored.** The Agent SDK has no temperature setting.
- **Kept-open sessions.** Each role (planner, reflection, grounding, and
  the code agent if used) keeps one Claude Code session open and sends
  `/clear` before each call. Check 3 measured about 1.5 s per call instead
  of 2.8 s. If `/clear` doesn't finish, or the context doesn't shrink back,
  that role switches to a new session per call for the rest of the run.
  Each open session is a Claude Code process using memory; use
  `--fresh_sessions` to turn this off.
- **The dialog takes focus.** After you click OK, the app that was in front
  before the dialog is brought back. Still watch for typing that lands in
  the wrong window.
- **Opening apps.** On macOS, "open app" and "switch app" now use
  `subprocess.run(['open', '-a', '<App>'])`. That's macOS's own launcher,
  and it never types. The old Spotlight sequence typed the app name blindly;
  in the first real run it typed into the Claude app.
- **The billing guard checks two places.** It looks at `.claude/settings`
  in the folder you run from and in your home folder.
