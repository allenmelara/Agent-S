import argparse
import datetime
import io
import logging
import os
import platform
import pyautogui
import signal
import subprocess
import sys
import time

from PIL import Image

from gui_agents.s3.agents.grounding import OSWorldACI
from gui_agents.s3.agents.agent_s import AgentS3
from gui_agents.s3.core.claude_subscription import (
    GROUNDING_SYSTEM_PROMPT,
    SubscriptionStop,
    ensure_subscription_only,
    is_local_url,
    set_call_budget,
    usage_summary,
)
from gui_agents.s3.utils.local_env import LocalEnv
from gui_agents.s3.utils.action_summary import describe_action

current_platform = platform.system().lower()

# Global flag to track pause state for debugging
paused = False


def get_char():
    """Get a single character from stdin without pressing Enter"""
    try:
        # Import termios and tty on Unix-like systems
        if platform.system() in ["Darwin", "Linux"]:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        else:
            # Windows fallback
            import msvcrt

            return msvcrt.getch().decode("utf-8", errors="ignore")
    except:
        return input()  # Fallback for non-terminal environments


def signal_handler(signum, frame):
    """Handle Ctrl+C signal for debugging during agent execution"""
    global paused

    if not paused:
        print("\n\n🔸 Agent-S Workflow Paused 🔸")
        print("=" * 50)
        print("Options:")
        print("  • Press Ctrl+C again to quit")
        print("  • Press Esc to resume workflow")
        print("=" * 50)

        paused = True

        while paused:
            try:
                print("\n[PAUSED] Waiting for input... ", end="", flush=True)
                char = get_char()

                if ord(char) == 3:  # Ctrl+C
                    print("\n\n🛑 Exiting Agent-S...")
                    sys.exit(0)
                elif ord(char) == 27:  # Esc
                    print("\n\n▶️  Resuming Agent-S workflow...")
                    paused = False
                    break
                else:
                    print(f"\n   Unknown command: '{char}' (ord: {ord(char)})")

            except KeyboardInterrupt:
                print("\n\n🛑 Exiting Agent-S...")
                sys.exit(0)
    else:
        # Already paused, second Ctrl+C means quit
        print("\n\n🛑 Exiting Agent-S...")
        sys.exit(0)


# Set up signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)

platform_os = platform.system()


def _frontmost_app_path():
    """Bundle path of the app in front on macOS, or None if unavailable."""
    try:
        from AppKit import NSWorkspace

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        return app.bundleURL().path() if app and app.bundleURL() else None
    except Exception:
        return None


def show_permission_dialog(message: str):
    """Show a platform-specific permission dialog and return True if approved."""
    if platform.system() == "Darwin":
        front = _frontmost_app_path()
        text = message[:1500].replace("\\", "\\\\").replace('"', '\\"')
        script = (
            f'display dialog "{text}" with title "Agent S — approve this action?" '
            'buttons {"Cancel", "OK"} default button "Cancel" cancel button "Cancel"'
        )
        approved = subprocess.run(["osascript", "-e", script], capture_output=True).returncode == 0
        if approved and front:
            # The dialog took focus; give it back so typing and hotkeys reach
            # the app that was in front when the screenshot was taken.
            subprocess.run(["open", "-a", front], capture_output=True)
            time.sleep(0.7)
        return approved
    elif platform.system() == "Linux":
        result = subprocess.run(
            ["zenity", "--question", "--title=Agent S — approve this action?", f"--text={message[:1500]}", "--width=500"],
            capture_output=True,
        )
        return result.returncode == 0
    return False


def scale_screen_dimensions(width: int, height: int, max_dim_size: int):
    scale_factor = min(max_dim_size / width, max_dim_size / height, 1)
    safe_width = int(width * scale_factor)
    safe_height = int(height * scale_factor)
    return safe_width, safe_height


def run_agent(
    agent,
    instruction: str,
    scaled_width: int,
    scaled_height: int,
    max_steps: int = 15,
    confirm_actions: bool = False,
):
    global paused
    obs = {}
    traj = "Task:\n" + instruction
    subtask_traj = ""
    for step in range(max_steps):
        # Check if we're in paused state and wait
        while paused:
            time.sleep(0.1)
        # Get screen shot using pyautogui
        screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((scaled_width, scaled_height), Image.LANCZOS)

        # Save the screenshot to a BytesIO object
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")

        # Get the byte value of the screenshot
        screenshot_bytes = buffered.getvalue()
        # Convert to base64 string.
        obs["screenshot"] = screenshot_bytes

        # Check again for pause state before prediction
        while paused:
            time.sleep(0.1)

        print(f"\n🔄 Step {step + 1}/{max_steps}: Getting next action from agent...")

        # Get next action code from the agent
        try:
            info, code = agent.predict(instruction=instruction, observation=obs)
        except SubscriptionStop as stop:
            print(f"\n⛔ Stopped: {stop}")
            break

        if "done" in code[0].lower() or "fail" in code[0].lower():
            if platform.system() == "Darwin":
                os.system(
                    f'osascript -e \'display dialog "Task Completed" with title "OpenACI Agent" buttons "OK" default button "OK"\''
                )
            elif platform.system() == "Linux":
                os.system(
                    f'zenity --info --title="OpenACI Agent" --text="Task Completed" --width=200 --height=100'
                )

            break

        if "next" in code[0].lower():
            continue

        if "wait" in code[0].lower():
            print("⏳ Agent requested wait...")
            time.sleep(5)
            continue

        else:
            time.sleep(1.0)
            print("EXECUTING CODE:", code[0])

            # Check for pause state before execution
            while paused:
                time.sleep(0.1)

            # Ask for permission before executing
            if confirm_actions and not show_permission_dialog(describe_action(info, code[0])):
                print("🚫 Action declined; stopping this task.")
                break
            try:
                exec(code[0])
            except SubscriptionStop as stop:  # grounding calls happen inside exec'd code
                print(f"\n⛔ Stopped: {stop}")
                break
            time.sleep(1.0)

            # Update task and subtask trajectories
            if "reflection" in info and "executor_plan" in info:
                traj += (
                    "\n\nReflection:\n"
                    + str(info["reflection"])
                    + "\n\n----------------------\n\nPlan:\n"
                    + info["executor_plan"]
                )


def main():
    parser = argparse.ArgumentParser(description="Run AgentS3 with specified model.")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Specify the provider to use (e.g., openai, anthropic, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-2025-08-07",
        help="Specify the model to use (e.g., gpt-5-2025-08-07)",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default="",
        help="The URL of the main generation model API.",
    )
    parser.add_argument(
        "--model_api_key",
        type=str,
        default="",
        help="The API key of the main generation model.",
    )
    parser.add_argument(
        "--model_temperature",
        type=float,
        default=None,
        help="Temperature to fix the generation model at (e.g. o3 can only be run with 1.0)",
    )

    # Grounding model config: Self-hosted endpoint based (required)
    parser.add_argument(
        "--ground_provider",
        type=str,
        required=True,
        help="The provider for the grounding model",
    )
    parser.add_argument(
        "--ground_url",
        type=str,
        default="",
        help="The URL of the grounding model (not used with claude_subscription)",
    )
    parser.add_argument(
        "--ground_api_key",
        type=str,
        default="",
        help="The API key of the grounding model.",
    )
    parser.add_argument(
        "--ground_model",
        type=str,
        default=None,
        help="The model name for the grounding model (claude_subscription: optional alias, e.g. sonnet)",
    )
    parser.add_argument(
        "--grounding_width",
        type=int,
        default=None,
        help="Width of screenshot image after processor rescaling (claude_subscription: defaults to the screenshot width)",
    )
    parser.add_argument(
        "--grounding_height",
        type=int,
        default=None,
        help="Height of screenshot image after processor rescaling (claude_subscription: defaults to the screenshot height)",
    )

    # AgentS3 specific arguments
    parser.add_argument(
        "--max_trajectory_length",
        type=int,
        default=8,
        help="Maximum number of image turns to keep in trajectory",
    )
    parser.add_argument(
        "--enable_reflection",
        action="store_true",
        default=True,
        help="Enable reflection agent to assist the worker agent",
    )
    parser.add_argument(
        "--enable_local_env",
        action="store_true",
        default=False,
        help="Enable local coding environment for code execution (WARNING: Executes arbitrary code locally)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=15,
        help="Maximum agent steps per task",
    )
    parser.add_argument(
        "--max_model_calls",
        type=int,
        default=60,
        help="claude_subscription only: stop a task after this many Claude calls",
    )
    parser.add_argument(
        "--fresh_sessions",
        action="store_true",
        help="claude_subscription only: start a new Claude session for every call instead of reusing one per role (slower)",
    )
    parser.add_argument(
        "--no_confirm",
        action="store_true",
        help="claude_subscription only: run actions without a confirmation dialog",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="The task instruction for Agent-S3 to perform.",
    )

    args = parser.parse_args()

    use_subscription = "claude_subscription" in (args.provider, args.ground_provider)
    if use_subscription:
        if args.provider == "claude_subscription" and args.model == parser.get_default("model"):
            args.model = None  # your Claude Code default model
        if args.ground_provider != "claude_subscription" and not is_local_url(args.ground_url):
            parser.error("with claude_subscription, the grounding model must be claude_subscription or a localhost URL")
        if args.provider != "claude_subscription" or args.model_url or args.model_api_key:
            parser.error("with claude_subscription, the main model must be claude_subscription, with no --model_url or --model_api_key")
        if args.enable_local_env:
            print("⚠️  Local code execution is enabled; generated code runs without the action dialog.")
    elif args.ground_model is None or args.grounding_width is None or args.grounding_height is None:
        parser.error("--ground_model, --grounding_width and --grounding_height are required for this provider")

    # Re-scales screenshot size to ensure it fits in UI-TARS context limit
    screen_width, screen_height = pyautogui.size()
    scaled_width, scaled_height = scale_screen_dimensions(
        screen_width, screen_height, max_dim_size=2400
    )

    # Load the general engine params
    engine_params = {
        "engine_type": args.provider,
        "model": args.model,
        "base_url": args.model_url,
        "api_key": args.model_api_key,
        "temperature": getattr(args, "model_temperature", None),
    }

    # Load the grounding engine from a custom endpoint
    engine_params_for_grounding = {
        "engine_type": args.ground_provider,
        "model": args.ground_model,
        "base_url": args.ground_url,
        "api_key": args.ground_api_key,
        "grounding_width": args.grounding_width or scaled_width,
        "grounding_height": args.grounding_height or scaled_height,
    }
    if args.ground_provider == "claude_subscription":
        engine_params_for_grounding["system_prompt_override"] = GROUNDING_SYSTEM_PROMPT
        engine_params_for_grounding["persistent"] = not args.fresh_sessions
    if args.provider == "claude_subscription":
        engine_params["persistent"] = not args.fresh_sessions

    confirm_actions = use_subscription and not args.no_confirm
    if use_subscription:
        try:
            ensure_subscription_only()
        except SubscriptionStop as stop:
            sys.exit(f"⛔ {stop}")
        print("✅ Using your Claude Max login (no API key).")

    def run_task(instruction):
        set_call_budget(args.max_model_calls if use_subscription else None)
        run_agent(agent, instruction, scaled_width, scaled_height, args.max_steps, confirm_actions)
        if use_subscription:
            used = usage_summary()
            print(
                f"Claude calls this task: {used['calls']} of {used['max_calls']}; "
                f"client-side API-price estimate ${used['est_cost_usd']:.2f} (not a bill; usage counts against your Max limits)"
            )

    # Initialize environment based on user preference
    local_env = None
    if args.enable_local_env:
        print(
            "⚠️  WARNING: Local coding environment enabled. This will execute arbitrary code locally!"
        )
        local_env = LocalEnv()

    grounding_agent = OSWorldACI(
        env=local_env,
        platform=current_platform,
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        width=screen_width,
        height=screen_height,
    )

    agent = AgentS3(
        engine_params,
        grounding_agent,
        platform=current_platform,
        max_trajectory_length=args.max_trajectory_length,
        enable_reflection=args.enable_reflection,
    )

    task = args.task

    # handle query from command line
    if isinstance(task, str) and task.strip():
        agent.reset()
        run_task(task)
        return

    while True:
        query = input("Query: ")

        agent.reset()

        # Run the agent on your own device
        run_task(query)

        response = input("Would you like to provide another query? (y/n): ")
        if response.lower() != "y":
            break


if __name__ == "__main__":
    main()
