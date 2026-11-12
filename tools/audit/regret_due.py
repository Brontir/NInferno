#!/usr/bin/env python3
"""Is the regret job (tools/bench/regret_job.sh, #620 step 10) due after a range of commits, and on
which lab boards?

  regret_due.py BASE..HEAD [--root DIR] [--json]
  (BASE...HEAD too: the diff is always from the merge base, as affected_arches.py reads it)

Due when the range changes a file that can move a board's route regret:
  - a decision-path file: tools/audit/decision_paths.txt, the list at either end of the range and
    in this checkout (the golden gate's list: the route selector, its cost model, the route tables,
    the facts);
  - the sweep's grid, tools/bench/data/route_regret/shapes.txt, or the job that runs it,
    tools/bench/regret_job.sh (its passes, samples, qualification and route-tune flags);
  - a route kernel: a .cu or .cuh under src/ops/generic, or any file a .cu there includes,
    transitively (src/ops/common/mma.cuh, ...), at either end of the range: the routes' times
    are the regret's other half.
Only the scoring tools (regret_collect.py, selection_regret.py, ...) changed: not due, re-score
the stands' archived CSVs instead.

The boards are those of the regret reports in this checkout (tools/bench/data/route_regret/
<board>.json), each with its build set from its chip's compute capability (gpu_facts.json). A route table
(routes/<class>.json) moves the boards of its own compute capability: the table lookup never
crosses an architecture (src/ops/route_table.cpp class_chain). A sibling minor with the same
features (sm_80 and sm_86) may lend its table (src/core/device_class_host.cpp gpu_similar_chips),
so a table of an architecture no lab board has moves the boards of its major. The lab boards'
own architectures differ in features from the other minors of their majors (sm_70 / sm_75,
sm_86 / sm_89). The grid moves every board. Every other trigger is classified by
affected_arches.py, each path on its own: an arch-guarded answer names its boards; "all", and host
code (the selector runs on the host for every board), name every board.
Exit: 0 due, 1 not due, 2 error.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import affected_arches as aa  # noqa: E402
import golden_gate as gate  # noqa: E402

PATHS_FILE = "tools/audit/decision_paths.txt"
REPORTS = "tools/bench/data/route_regret"
GPU_FACTS = "src/core/data/gpu_facts.json"
# What moves the regret besides the decision paths.
GRID = "tools/bench/data/route_regret/shapes.txt"
JOB = "tools/bench/regret_job.sh"  # defines the sweep: passes, samples, qualification, flags
SWEEP = (GRID, JOB)
EXTRA_PATTERNS = SWEEP
KERNEL_DIR = "src/ops/generic/"  # the route kernels the sweep times (kernel_files)
ROUTE_TABLE = re.compile(r"routes/[^/]+\.json\Z")
# affected_arches' build sets by compute capability.
ARCH_OF_CC = {70: "sm_70", 75: "sm_75", 86: "sm_86", 89: "sm_89", 120: "sm_120a"}


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def git_show(root, ref, path):
    """The file at ref, or None where it does not exist."""
    cp = subprocess.run(["git", "-C", str(root), "show", f"{ref}:{path}"], capture_output=True,
                        text=True, encoding="utf-8")
    return cp.stdout if cp.returncode == 0 else None


def chip_cc(gpu_facts):
    """{chip: compute capability} of gpu_facts.json's rows."""
    return {row["chip"]: int(row["cc"]) for row in gpu_facts["gpus"]}


def lab_boards(reports, chips):
    """{board: {card, device, cc, arch}} of the committed reports ({path: document})."""
    out = {}
    for path, doc in sorted(reports.items()):
        device = (doc.get("regret_report") or {}).get("device") or doc.get("device")
        board = doc.get("board")
        if not board or not device:
            continue  # not a board's regret report (policy_evidence_moves.json)
        cc = chips.get(device.split(":")[0])
        if cc is None:
            raise ValueError(f"{path}: the chip of {device} is not in {GPU_FACTS}")
        out[board] = {"card": doc.get("card", board), "device": device, "cc": cc,
                      "arch": ARCH_OF_CC.get(cc)}
    return out


def table_cc(text, chips):
    """The compute capability of a route table's text, or None when it cannot be read."""
    try:
        cls = json.loads(text)["gpu"]["class"]
    except (TypeError, ValueError, KeyError):
        return None
    chip = cls.split(":")[0]
    found = re.fullmatch(r"sm_?(\d+)", chip)
    return int(found[1]) if found else chips.get(chip)


def table_boards(cc, boards):
    """(boards, why) a route table of compute capability cc moves: those of its own cc; when no
    lab board has that cc, those of its major, any of which may be the sibling that borrows it."""
    own = sorted(b for b, v in boards.items() if v["cc"] == cc)
    if own:
        return own, f"route table of sm_{cc}: " + ", ".join(own)
    major = sorted(b for b, v in boards.items() if v["cc"] // 10 == cc // 10)
    return major, (f"route table of sm_{cc}, no lab board of its own; a sibling minor may borrow "
                   f"it: " + (", ".join(major) or "no lab board of its major"))


def kernel_files(repo):
    """The generic route kernels and every file they include, transitively: the .cu and .cuh
    files under src/ops/generic and the include closure of each .cu there (src/ops/common/mma.cuh,
    ...). A change to any of them can change a route's time."""
    out = set()
    for f in repo.files:
        if f.startswith(KERNEL_DIR) and f.endswith((".cu", ".cuh")):
            out.add(f)
            if f.endswith(".cu"):
                out |= repo.include_closure(f)
    return out


def boards_of_classification(res, boards):
    """(boards, why) of one path's affected_arches.classify result. "all", and host code (the
    selector runs on the host for every board), name every board; only an arch-guarded answer
    narrows them."""
    if res["all"] or not res["arches"]:
        why = "affected_arches.py: all" if res["all"] else \
            "affected_arches.py: host code, which every board's selector runs"
        return sorted(boards), why
    arches = set(res["arches"])
    hit = sorted(b for b, v in boards.items() if v["arch"] in arches or v["arch"] is None)
    return hit, "affected_arches.py: " + ", ".join(a for a in aa.ARCHES if a in arches)


def answer(changed, patterns, boards, table_text, classify_one, chips, kernels=()):
    """The verdict. changed: the paths of the range; patterns: load_patterns() of the trigger
    globs; table_text(path): a route table's text at the head or, deleted, at the base (None if
    neither); classify_one(path): affected_arches.classify's result for that path alone (one call
    a path: a host file beside an arch-guarded one keeps its own answer); kernels: kernel_files()
    at either end of the range."""
    kernels = set(kernels)
    triggers = sorted(p for p in changed
                      if p in kernels or any(rx.match(p) for _, rx in patterns))
    why = {}
    other = []
    for p in triggers:
        cc = table_cc(table_text(p), chips) if ROUTE_TABLE.match(p) else None
        if cc is None:
            other.append(p)
            continue
        hit, why[p] = table_boards(cc, boards)
        for b in hit:
            boards[b].setdefault("_by", []).append(p)
    grid = [p for p in other if p in SWEEP]
    other = [p for p in other if p not in SWEEP]
    for p in grid:
        why[p] = ("the sweep's grid: every board" if p == GRID
                  else "the job that runs the sweep: every board")
        for v in boards.values():
            v.setdefault("_by", []).append(p)
    for p in other:
        hit, why[p] = boards_of_classification(classify_one(p), boards)
        for b in hit:
            boards[b].setdefault("_by", []).append(p)
    due = {b: sorted(set(v.pop("_by"))) for b, v in sorted(boards.items()) if "_by" in v}
    return {"due": bool(due), "triggers": triggers, "boards": due, "reasons": why,
            "cards": {b: boards[b]["card"] for b in due}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("range", help="BASE..HEAD or BASE...HEAD")
    ap.add_argument("--root", default=str(HERE.parents[1]), help="the repository")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        found = re.fullmatch(r"(.+?)\.\.\.?(.+)", args.range)
        if not found:
            raise ValueError(f"{args.range}: expected BASE..HEAD")
        base, head = found[1], found[2]
        root = args.root
        changed = aa.git_diff_paths(base, head, root)
        merge_base = git(root, "merge-base", base, head).strip()
        # The list at both ends of the range and in this checkout (an older range is judged by
        # today's list too); the lab boards and their facts are this checkout's.
        here = Path(root)
        lists = [git_show(root, head, PATHS_FILE), git_show(root, merge_base, PATHS_FILE),
                 (here / PATHS_FILE).read_text(encoding="utf-8")]
        patterns = gate.load_patterns("\n".join([t or "" for t in lists] + list(EXTRA_PATTERNS)))
        chips = chip_cc(json.loads((here / GPU_FACTS).read_text(encoding="utf-8")))
        reports = {p.as_posix(): json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted((here / REPORTS).glob("*.json"))}
        boards = lab_boards(reports, chips)
        if not boards:  # "not due" would be a silent wrong answer
            raise ValueError(f"no lab board: no board's regret report in {REPORTS}")

        def table_text(p):
            text = git_show(root, head, p)
            return text if text is not None else git_show(root, merge_base, p)

        head_repo = aa.Repo(root, ref=head)
        base_repo = aa.Repo(root, ref=merge_base)

        def classify_one(p):
            return aa.classify([p], head_repo, base=base_repo)

        kernels = kernel_files(head_repo) | kernel_files(base_repo)
        res = answer(changed, patterns, boards, table_text, classify_one, chips, kernels)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"regret_due: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(res, indent=1))
    elif not res["due"]:
        print("regret job: not due (no decision path, grid or kernel source changed)")
    else:
        print("regret job: due on " + ", ".join(f"{b} ({res['cards'][b]})" for b in res["boards"]))
        for p in res["triggers"]:
            print(f"  {p}: {res['reasons'][p]}")
    return 0 if res["due"] else 1


if __name__ == "__main__":
    sys.exit(main())
