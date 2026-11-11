#!/usr/bin/env python3
"""CI gate for #620 step 9: a decision-path diff needs a golden change or the `goldens-equal` label.

  golden_gate.py --base-ref master --event "$GITHUB_EVENT_PATH"

1. The files changed since the merge-base with origin/<base-ref> are matched against the union of
   tools/audit/decision_paths.txt at HEAD and at origin/<base-ref> (a PR cannot drop its own
   line). No match: pass, nothing more to do.
2. Otherwise tools/audit/golden_diff.py --range origin/<base-ref>...HEAD summarizes the goldens.
3. Goldens changed: print the summary, pass (the reviewer reads it).
   Goldens equal and the PR carries the label `goldens-equal`: pass, print the author's claim.
   Goldens equal and no label: fail.
The labels come from the pull_request event payload, so the job needs no token.
Only the selector goldens (tests/ops/linear/selector_golden) are read. The MoE plan goldens
(tests/ops/moe_plan_golden) are not compared: a change there is counted and named, nothing more.
The label is not tied to a commit: it stays on after later pushes, and the author's claim then
covers them all.
Exit: 0 pass, 1 gate failed, 2 the tools themselves failed.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

LABEL = "goldens-equal"
HERE = Path(__file__).resolve().parent
PATHS_FILE = HERE / "decision_paths.txt"
GOLDEN_DIFF = HERE / "golden_diff.py"
SELECTOR_GOLDENS = "tests/ops/linear/selector_golden/"
MOE_GOLDENS = "tests/ops/moe_plan_golden/"
# the gate's own files: a PR that edits them changes what the gate checks
GATE_FILES = ("tools/audit/decision_paths.txt", "tools/audit/golden_gate.py", "tools/audit/golden_diff.py",
              "tests/test_golden_diff.py", "tests/test_golden_gate.py", ".github/workflows/golden-gate.yml")
# the line golden_diff prints whenever it exits 1 on purpose (--fail-on-change)
TOTAL_LINE = re.compile(r"^total: \d+ decisions changed", re.M)
# the line it prints when it exits 0 with --fail-on-change: nothing changed
NO_CHANGE_LINE = re.compile(r"^no golden changes$", re.M)
# golden_diff reads two trees of text files; a run that takes longer than this is hung, not slow
# (chosen, not measured: a whole golden set diffs in seconds)
DIFF_TIMEOUT_S = 600.0


def diff_timeout_s():
    """NINFERNO_GOLDEN_DIFF_TIMEOUT in seconds, else DIFF_TIMEOUT_S. Read when the gate runs, so a
    bad value is a clear error of the gate and not an exception at import."""
    raw = os.environ.get("NINFERNO_GOLDEN_DIFF_TIMEOUT")
    if raw is None:
        return DIFF_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not value > 0 or value == float("inf"):
        raise ValueError(f"NINFERNO_GOLDEN_DIFF_TIMEOUT={raw!r} is not a positive number of seconds")
    return value


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
    """(exit code, message). golden_rc: 0 goldens equal, 1 changed (--fail-on-change)."""
    if not touched:
        return 0, "no decision-path file changed: nothing to check"
    if golden_rc == 1:
        return 0, ("decision path changed and the goldens moved: "
                   "the summary above is for the reviewer")
    if LABEL in labels:
        return 0, (f"label `{LABEL}`: the author claims every selector decision is as it was; "
                   "the selector goldens are equal (MoE plan goldens are not compared)")
    return 1, (
        "the decision path changed but no selector golden did.\n"
        "If the change is meant to move decisions, regenerate the goldens "
        "(test_selector_golden --update).\n"
        "If it is a refactor that leaves every decision as it was, "
        f"add the label `{LABEL}` to the PR;\n"
        "the label event starts a new run. Re-running the old run replays the old payload "
        "and keeps failing: add the label, do not re-run.\n"
        f"Only the selector goldens ({SELECTOR_GOLDENS}) are compared; the MoE plan goldens are not read.\n"
        "The label stays on after later pushes: it is the author's claim for the whole PR."
    )


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout


def base_patterns(base_ref, repo, text=None):
    """The base branch's list: a PR cannot drop its own line to escape the gate. Missing = empty."""
    if text is None:
        try:
            text = git(["show", f"origin/{base_ref}:tools/audit/decision_paths.txt"], repo)
        except subprocess.CalledProcessError:
            text = ""
    return load_patterns(text)


def main(argv=None, run_diff=None, changed_files=None, base_list=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ref", required=True,
                    help="the PR's base branch (compared as origin/<base-ref>)")
    ap.add_argument("--event", help="path of the pull_request event payload (GITHUB_EVENT_PATH)")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--paths", default=str(PATHS_FILE))
    args = ap.parse_args(argv)  # bad arguments: argparse exits 2
    try:
        event = json.loads(Path(args.event).read_text(encoding="utf-8")) if args.event else {}
        patterns = load_patterns(Path(args.paths).read_text(encoding="utf-8"))
        # the union of the head and base lists
        patterns += base_patterns(args.base_ref, args.repo, base_list)
        rng = f"origin/{args.base_ref}...HEAD"
        if changed_files is None:
            listing = git(["diff", "--name-only", "--no-renames", "-z", rng], args.repo)
            changed = [f for f in listing.split("\0") if f]
        else:
            changed = changed_files
        touched = decision_files(changed, patterns)
        golden_rc = 0
        gate_edits = [f for f in changed if f in GATE_FILES]
        if gate_edits:
            print("this PR edits the gate itself, and the workflow that runs here is the PR's own copy: "
                  "review those files by hand:\n  " + "\n  ".join(gate_edits))
        moe_goldens = [f for f in changed if f.startswith(MOE_GOLDENS)]
        if moe_goldens:
            print(f"{len(moe_goldens)} MoE plan golden file(s) changed; the gate does not read them")
        if not touched and any(f.startswith(SELECTOR_GOLDENS) for f in changed):
            print("selector goldens changed, no decision-path file did: the reviewer reads the golden diff")
        if touched:
            print("decision-path files changed:\n  " + "\n  ".join(touched))
            if run_diff is None:
                timeout = diff_timeout_s()  # a bad value: ValueError, exit 2 below
                try:
                    proc = subprocess.run(
                        [sys.executable, str(GOLDEN_DIFF), "--repo", args.repo, "--range", rng,
                         "--fail-on-change"],
                        capture_output=True, text=True, timeout=timeout)
                except subprocess.TimeoutExpired:
                    print(f"golden_gate: golden_diff ran longer than {timeout:g} s", file=sys.stderr)
                    return 2
                golden_rc, out = proc.returncode, proc.stdout + proc.stderr
            else:
                golden_rc, out = run_diff(rng)
            print(out.rstrip())
            # Exit 1 is "goldens changed" only if golden_diff reached its summary line. A crash (a
            # traceback, a syntax error, sys.exit("message")) also exits 1 and prints no such line.
            if golden_rc == 1 and not TOTAL_LINE.search(out):
                golden_rc = 2
            # and exit 0 is "goldens equal" only with the line golden_diff prints for it: an exit 0
            # with nothing or something else printed compared nothing we can trust
            if golden_rc == 0 and not NO_CHANGE_LINE.search(out):
                golden_rc = 2
            if golden_rc not in (0, 1):
                print("golden_gate: golden_diff failed", file=sys.stderr)
                return 2
        code, message = decide(touched, golden_rc, labels_of(event))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"golden_gate: {getattr(error, 'stderr', None) or error}", file=sys.stderr)
        return 2
    print(("golden_gate: " if code == 0 else "golden_gate FAILED: ") + message)
    return code


def run():
    """Entry point: every tool failure, caught or not, exits 2; 1 is only "the gate refused"."""
    try:
        return main()
    except Exception as error:  # noqa: BLE001
        print(f"golden_gate: unexpected {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(run())
