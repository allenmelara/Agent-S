"""Run the feasibility checks in order and print one pass/fail table.

Default: checks 1, 4 and 5 part A only (no model calls).
--with-claude: also checks 2, 3 and 5 with Claude as grounder (about 15 calls).
--with-local URL: also check 5 with a local grounder at URL (e.g. http://localhost:1234/v1).
Model-call checks run only if check 1 passed.
"""

import argparse
import json
import subprocess
import sys

from common import REPORT_DIR, ensure_python


def run(script: str, *extra: str) -> None:
    print(f"\n$ python {script} {' '.join(extra)}".rstrip())
    subprocess.run([sys.executable, script, *extra], cwd=REPORT_DIR.parent)


def overall(check_id: str) -> str:
    path = REPORT_DIR / f"{check_id}.json"
    return json.loads(path.read_text())["overall"] if path.exists() else "NOT RUN"


def main() -> int:
    ensure_python()
    p = argparse.ArgumentParser()
    p.add_argument("--with-claude", action="store_true")
    p.add_argument("--with-local", metavar="URL")
    p.add_argument("--local-model", default="ui-tars-1.5-7b")
    p.add_argument("--model", default=None, help="Claude model alias for checks 2, 3, 5")
    args = p.parse_args()

    REPORT_DIR.mkdir(exist_ok=True)
    for old in REPORT_DIR.glob("*.json"):
        old.unlink()

    model = ["--model", args.model] if args.model else []
    run("check_1_auth.py")
    run("check_4_hardware.py")
    auth_ok = overall("check_1_auth") == "PASS"

    grounders = []
    if args.with_claude:
        if auth_ok:
            run("check_2_image_latency.py", *model)
            run("check_3_persistent_session.py", *model)
            grounders += ["--grounder", "claude", *model]
        else:
            print("\nSkipping Claude calls: check 1 did not pass.")
    if args.with_local:
        grounders += ["--grounder", "local", "--local-url", args.with_local, "--local-model", args.local_model]
    run("check_5_coordinates.py", *grounders)

    print("\n================ SUMMARY ================")
    ids = ["check_1_auth", "check_2_image_latency", "check_3_persistent_session", "check_4_hardware", "check_5_coordinates"]
    for cid in ids:
        print(f"{overall(cid):8} {cid}")
    print(f"Details: {REPORT_DIR}/*.json")
    return 1 if any(overall(c) == "FAIL" for c in ids) else 0


if __name__ == "__main__":
    raise SystemExit(main())
