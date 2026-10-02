"""Check 1: Claude calls would use your claude.ai Max login, nothing billed separately.

Makes no model calls. Reads environment variable names, Claude Code settings
files, the repo's .env, and `claude auth status --json`.
"""

from common import FAIL, PASS, Report, ensure_python, require_macos, subscription_guard


def main() -> int:
    ensure_python()
    report = Report("check_1_auth", "Subscription authentication only")
    require_macos(report)
    try:
        import claude_agent_sdk

        report.add(PASS, "claude-agent-sdk installed", f"version {claude_agent_sdk.__version__}")
    except ImportError:
        report.add(FAIL, "claude-agent-sdk not installed", "pip install -r mac_feasibility/requirements.txt")
    if subscription_guard(report):
        report.add(PASS, "billing guard", "no API keys, gateways or third-party providers found")
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
