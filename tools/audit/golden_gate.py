#!/usr/bin/env python3
"""CI gate for #620 step 9: a decision-path diff with no golden change needs the `goldens-equal` label.

  golden_gate.py --base-ref master --event "$GITHUB_EVENT_PATH"

1. The files changed since the merge-base with origin/<base-ref> are matched against
   tools/audit/decision_paths.txt. No match: pass, nothing more to do.
2. Otherwise tools/audit/golden_diff.py --range origin/<base-ref>...HEAD summarizes the goldens.
3. Goldens changed: print the summary, pass (the reviewer reads it).
   Goldens equal and the PR carries the label `goldens-equal`: pass, print the author's claim.
   Goldens equal and no label: fail.
The labels come from the pull_request event payload, so the job needs no token.
Exit: 0 pass, 1 gate failed, 2 the tools themselves failed.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

LABEL = "goldens-equal"
HERE = Path(__file__).resolve().parent
PATHS_FILE = HERE / "decision_paths.txt"
GOLDEN_DIFF = HERE / "golden_diff.py"


def glob_to_regex(pattern):
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


def load_patterns(text):
    pats = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pats.append((line, glob_to_regex(line)))
    return pats


def decision_files(changed, patterns):
    return sorted(f for f in changed if any(rx.match(f) for _, rx in patterns))


def labels_of(event):
    pr = event.get("pull_request") or {}
    return {label.get("name") for label in pr.get("labels") or []}


def decide(touched, golden_rc, labels):
    """(exit code, message). golden_rc: 0 goldens equal, 1 changed (golden_diff --fail-on-change)."""
    if not touched:
        return 0, "no decision-path file changed: nothing to check"
    if golden_rc == 1:
        return 0, "decision path changed and the goldens moved: the summary above is for the reviewer"
    if LABEL in labels:
        return 0, f"label `{LABEL}`: the author claims this change is golden-equal; the goldens are equal"
    return 1, (
        "the decision path changed but no selector golden did.\n"
        "If the change is meant to move decisions, regenerate the goldens (test_selector_golden --update).\n"
        f"If it is a refactor that leaves every decision as it was, add the label `{LABEL}` to the PR."
    )


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def main(argv=None, run_diff=None, changed_files=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ref", required=True, help="the PR's base branch (compared as origin/<base-ref>)")
    ap.add_argument("--event", help="path of the pull_request event payload (GITHUB_EVENT_PATH)")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--paths", default=str(PATHS_FILE))
    args = ap.parse_args(argv)
    event = json.loads(Path(args.event).read_text(encoding="utf-8")) if args.event else {}
    try:
        patterns = load_patterns(Path(args.paths).read_text(encoding="utf-8"))
        rng = f"origin/{args.base_ref}...HEAD"
        if changed_files is None:
            changed = git(["diff", "--name-only", rng], args.repo).split()
        else:
            changed = changed_files
        touched = decision_files(changed, patterns)
        golden_rc = 0
        if touched:
            print("decision-path files changed:\n  " + "\n  ".join(touched))
            if run_diff is None:
                proc = subprocess.run(
                    [sys.executable, str(GOLDEN_DIFF), "--repo", args.repo, "--range", rng, "--fail-on-change"],
                    capture_output=True, text=True)
                golden_rc, out = proc.returncode, proc.stdout + proc.stderr
            else:
                golden_rc, out = run_diff(rng)
            print(out.rstrip())
            if golden_rc not in (0, 1):
                print("golden_gate: golden_diff failed", file=sys.stderr)
                return 2
        code, message = decide(touched, golden_rc, labels_of(event))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"golden_gate: {getattr(error, 'stderr', None) or error}", file=sys.stderr)
        return 2
    print(("golden_gate: " if code == 0 else "golden_gate FAILED: ") + message)
    return code


if __name__ == "__main__":
    sys.exit(main())
