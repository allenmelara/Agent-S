# Mac feasibility checks for a Max-subscription Agent S

These scripts decide whether Agent S can run on your Mac using only your
Claude Max plan, with no API key and no hosted model. They don't change
`gui_agents/`. They never click, type or move the mouse.

| Check | What it checks | Model calls |
|---|---|---|
| 1 `check_1_auth.py` | Claude would use your claude.ai Max login. API keys, gateways, Bedrock/Vertex/Foundry and hosted Agent S provider keys are refused. | 0 |
| 2 `check_2_image_latency.py` | One generated test image goes through the Agent SDK with all tools disabled. Measures response time. | 1 |
| 3 `check_3_persistent_session.py` | Fresh session per call vs one long-lived session, using ~2,000-token turns. Tests whether `/clear` really drops history, by context size and by asking Claude to recall an earlier codeword. | 8 (default `-n 3`) |
| 4 `check_4_hardware.py` | Chip, memory, disk and installed local runtimes, then a UI-TARS size suggestion | 0 |
| 5 `check_5_coordinates.py` | Retina scaling math (no model), then accuracy on a generated fake screen | 0, or 4 per grounder |

Every script that calls Claude runs the check 1 guard first and stops before
any call if the guard fails. During a call it also stops if the session
reports an API key source, if a request would draw on extra (billed) usage,
or if your Max limit is reached.

## Setup (on your Mac)

1. **Sign in to Claude Code with your Max account.**
   `claude auth login`, then `claude auth status --text`. It should show your
   claude.ai account and a Max plan.
2. **Remove API keys from your shell.** The Agent S README tells you to put
   keys in `~/.zshrc`, so look there. Run
   `env | grep -E 'ANTHROPIC|OPENAI|HF_|GEMINI|OPENROUTER|CLAUDE_CODE_USE|CLAUDE_CODE_OAUTH' | cut -d= -f1`
   (prints names only, never the secret values).
   For each name it prints, delete its `export` line from `~/.zshrc` and
   open a new terminal. Do the same for a `.env` file in the repo root.
3. **Optional hard stop on spending.** If "extra usage" is turned on for
   your claude.ai account, usage beyond your Max limits can be billed. Check 2
   warns when it's on. Turn it off in claude.ai settings; the exact menu
   name isn't verified here.
4. **Python 3.10 or newer** (the Agent SDK needs it; macOS's built-in Python
   is 3.9): `brew install python@3.12`.
5. **Get the branch:**
   ```bash
   git clone https://github.com/allenmelara/Agent-S.git
   cd Agent-S
   git checkout claude/vigilant-franklin-5fs60d
   ```
6. **Create a separate environment for the checks:**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r mac_feasibility/requirements.txt
   ```
7. **Screen Recording permission (check 5 only).** In System Settings >
   Privacy & Security > Screen Recording, allow the terminal app you use, then
   restart it. Accessibility permission is **not** needed, because nothing
   clicks.
8. **Run the checks that make no model calls:**
   ```bash
   cd mac_feasibility
   python run_all.py
   ```
9. **If check 1 passes, run the Claude checks** (about 13 small calls on your
   Max allowance):
   ```bash
   python run_all.py --with-claude            # add --model sonnet to pick a model
   ```
10. **Optional local grounder** (free, runs on your Mac). Install LM Studio
    or `mlx-vlm` and load a UI-TARS-1.5-7B build sized per check 4. Start
    its OpenAI-compatible server, then run:
    ```bash
    python run_all.py --with-local http://localhost:1234/v1 --local-model <model id from the server>
    ```
    Whether a working UI-TARS-1.5-7B build is available for each runtime is
    **not verified**; check 5 is the test. Only `localhost` URLs are
    accepted.
11. **Optional real-screen test.** `python check_5_coordinates.py --grounder local --real-screen`
    asks for the Apple menu's position on your actual screen, and reports
    where Agent S *would* click. Nothing is clicked. With `--grounder claude`
    this sends a screenshot of your screen to Claude, so close anything
    private first.
12. **Share the results.** Send back `mac_feasibility/reports/*.json` and
    the annotated `check_5_*.png`. They record plan type and timings, not
    keys or your email.

## Expected results and pass/fail

| Check | PASS means | FAIL means | Expected if all is well |
|---|---|---|---|
| 1 | Installed CLI and SDK-bundled CLI both report `authMethod=claude.ai`, `apiProvider=firstParty`, a Max plan and no API key. A live SDK handshake also reports a Max plan. | Any API key, `apiKeyHelper`, gateway URL, third-party provider, `CLAUDE_CODE_OAUTH_TOKEN`, hosted Agent S key, or a non-Max plan | PASS |
| 2 | Session started with no API key source, no tools were offered or used, and Claude read the number `4719` and both shapes | Guard tripped, SDK error, or the image wasn't understood | PASS. Timing is informational. |
| 3 | Claude recalls the codeword without `/clear` (control) but not after it, and context grows by less than 1,000 tokens per turn. Timing is a separate PASS/WARN line. | After `/clear`, Claude still knows an earlier codeword, or context grows ~2,000 tokens per turn (history is kept), or `/clear` doesn't complete | PASS, unknown until measured. WARN on the control means the probe was inconclusive. |
| 4 | Apple Silicon. A suggested UI-TARS size fits your memory estimate. | Intel Mac (use Claude for grounding instead) | PASS on Apple Silicon with 16 GB or more. The estimate is unverified. |
| 5A | Screen Recording granted, one display, and Agent S's point/pixel round-trip error at most 1.5 pt | Missing permission or wrong scaling math | PASS |
| 5B | Grounder hits every fake button in some coordinate space. The report gives the matching `--grounding_width/height`. | Fewer than half hit | Unknown until measured |

Overall result per check: any FAIL gives FAIL; otherwise any WARN gives WARN;
otherwise PASS. SKIP means the check didn't apply, for example on a
non-Mac. `run_all.py` prints a one-line summary per check.

`total_cost_usd` in the reports is Claude Code's client-side API-price
estimate. It is not an invoice. On a Max login, the docs say these calls
draw on your plan's usage limits.

## Verified in the cloud vs still to test on your Mac

Verified in the cloud session that wrote these scripts (Linux, Claude Code
2.1.287, claude-agent-sdk 0.2.163):

- All scripts compile. On Linux, checks 4 and 5 report SKIP.
- The guard refuses `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`,
  `apiKeyHelper` and provider env in a settings file, `OPENAI_API_KEY`, a
  non-local `OLLAMA_HOST`, and an OAuth-token login. It allows a localhost
  `vLLM_ENDPOINT_URL`. Model-call checks stop before any call when the guard
  fails.
- `claude auth status --json` has the fields `loggedIn`, `authMethod`
  (`claude.ai` for subscription login), `apiProvider` and `apiKeySource`,
  read from the CLI source.
- The SDK accepts the options used here
  (`--tools ''`, `--setting-sources=`, `--strict-mcp-config`,
  `--no-session-persistence`, `--max-turns 1`). A connect-only handshake
  works and reports an `account.subscriptionType`, with no model call made.
- Stream handling stops on an API-key source, on overage usage and on a
  rejected rate limit, tested with simulated messages.
- Fake-screen scoring picks the right coordinate space for simulated
  pixel-space and 0–1000 grounders.

Not verified. These need your Mac, and all usage and hardware numbers
stay estimates until measured:

- The exact `subscriptionType` string your Max login reports (the check
  accepts anything containing "max").
- That image input works through the SDK on your account, and its response
  time.
- Whether a persistent session saves time, and whether `/clear` really drops history.
- Real memory headroom and the speed of UI-TARS-1.5-7B on your chip; the
  quantization suggestions are rough estimates.
- Grounding accuracy of Claude or local UI-TARS, which coordinate space
  each uses, and Retina behavior on your display.
- How much of your Max allowance these calls use.
