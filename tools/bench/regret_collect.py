#!/usr/bin/env python3
"""Lab collector for selection_regret.py (#301): route sweeps and the production selection.

Combines ninferno_route_tune's CSVs of one board into one ninferno-selection-regret/1 document:
  --selected  each pass's --selected-out CSV: per (problem, T), the route the production path runs;
  --routes    one --routes-out CSV per measurement pass (interleave whole runs, same build and board).
Every (problem, T) of the selected CSV becomes a case whose candidates are the routes timed for it.
A route timed in some passes and not others is `unmeasured`; nothing timed is dropped silently.

route_tune does not check results. A candidate is `ok` only with --qualified, which states that this
build's route correctness tests (ninferno_linear_generic_test, ninferno_route_classes_test) passed
on this board; the statement is copied into the document. Without it every candidate is
`unmeasured` and the evaluator reports the cases incomplete.

A --selected CSV may also carry `authority` and `evidence` columns (the route authority that
answered and the kind of evidence it rests on, #614 route authorities); each case then records
them, and selection_regret.py groups regret by board x regime x authority. Without the columns
(written before route_tune named them) both read "unknown": the collector still works on old data.
--pci-id adds the board's compute capability, SMs, DRAM rate, boost clock and `mma_slow` (no tensor
cores) from gpu_facts.json to the document, from which selection_regret.py computes the card's
ridge for its regimes; the rows' device must be that board's class (chip or architecture, and SMs).
Rows that carry a `pci_device_id` column (route_tune writes it when the driver reports the id, #684)
name the board itself: two SKUs of one class (an RTX 4070 Ti and an RTX 4500 Ada are both
ad104:60sm) are told apart, --pci-id must then be that id, and without --pci-id the board's facts
are read by the recorded id. Older rows keep the class check alone.

Each row names its run, board and timing protocol (#518): a pass's --selected-out and --routes-out
must come from one run, every file from one board, and a case's passes from one protocol ("copies":
consecutive calls over weight copies, a decode width under --rotate 1; "flushed": one call after
the L2 flush), or the files are refused. A CSV without these columns (written before route_tune
named them) is refused: its passes could not be told apart.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

KEY = ("format", "compute", "epilogue", "n", "k", "T")

# Route families by the activations they compute with: a case compares only routes of its compute
# class. route_tune also times ggml's matrix kernel under an a16 key (the default policy runs it at
# verify widths), which computes int8 activations.
INT8_FAMILIES = ("ggml_mmq", "ggml_vec", "r8.", "w4a8", "i8.")
FP4_FAMILIES = ("w4a4",)


def route_class(route):
    if route.startswith(INT8_FAMILIES):
        return "a8"
    if route.startswith(FP4_FAMILIES):
        return "a4"
    return "a16"


def read_rows(path, columns):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        return list(reader)


def key_of(row):
    return tuple(row[c].strip() for c in KEY)


def case_id(key):
    fmt, compute, epilogue, n, k, t = key
    return f"{fmt}.{compute}.{epilogue}.n{n}.k{k}.T{t}"


# The policy route_tune selects with for a compute class (its problems' policy), and the route
# profile it runs: the speed profile (prefill_routes inactive), so a T >= 17 case is a batched
# decode call, not a prefill chunk. The default Fast mode runs a published model's non-NVFP4
# projections AllowA8 (the vector kernel, R8, W4A8 ahead of the a16 routes): an a16 case is the
# A16Only path, the exact mode's linears.
POLICY = {"a16": "A16Only", "a8": "AllowA8", "a4": "AllowA4"}


ROUTE_COLUMNS = KEY + ("route", "median_us", "run", "device", "protocol", "calls", "samples")
SELECTED_COLUMNS = KEY + ("selected", "run", "device")
UNKNOWN = "unknown"


def identity(path, rows, column):
    """The one value a file's rows give `column` (its run or board)."""
    values = {(row.get(column) or "").strip() for row in rows}
    if not values:
        raise ValueError(f"{path}: no rows")
    if len(values) != 1 or "" in values:
        raise ValueError(f"{path}: rows from more than one {column} ({', '.join(sorted(values))})")
    return values.pop()


def protocol_of(row):
    """How a row's time was taken: (protocol, calls per sample, samples)."""
    protocol = row["protocol"].strip()
    if protocol not in ("copies", "flushed", "warm"):
        raise ValueError(f"unknown timing protocol {protocol!r}")
    return (protocol, int(row["calls"]), int(row["samples"]))


def describe(protocol):
    kind, calls, samples = protocol
    if kind == "copies":
        return (f"{calls} consecutive calls over weight copies (2x L2 or 128 MiB) between one event "
                f"pair, median of {samples} samples")
    if kind == "warm":
        return f"one call with a warm L2 (the timer had no flush buffer), median of {samples} samples"
    return f"one call after an L2 flush, median of {samples} samples"


def read_selections(paths, authorities=None):
    """{(problem, T): the selected route} of the passes, which must agree. `authorities`, a dict,
    is filled with {(problem, T): (authority, evidence)} from the optional columns of the same
    names ("unknown" where a file has none or a row leaves one empty)."""
    selected = None
    for path in paths:
        run = {}
        meta = {}
        for row in read_rows(path, SELECTED_COLUMNS):
            key = key_of(row)
            route = row["selected"].strip()
            if key in run and run[key] != route:
                raise ValueError(f"{path}: {case_id(key)} selected twice ({run[key]}, {route})")
            run[key] = route
            named = ((row.get("authority") or "").strip() or UNKNOWN,
                     (row.get("evidence") or "").strip() or UNKNOWN)
            if meta.setdefault(key, named) != named:
                raise ValueError(f"{path}: {case_id(key)} answered twice with different "
                                 f"authority or evidence ({meta[key]}, {named})")
        if selected is not None and run != selected:
            raise ValueError(f"{path}: the selection differs from the first pass's")
        if authorities is not None:
            for key, value in meta.items():
                first = authorities.setdefault(key, value)
                for what, a, b in zip(("authority", "evidence"), first, value):
                    if a != b:
                        raise ValueError(f"{path}: {case_id(key)} names another {what} than the "
                                         f"first pass's ({a}, {b})")
        selected = run
    return selected


def board_facts(gpu_facts, pci_id):
    """The row of core/data/gpu_facts.json for a PCI device id (hex string), as the numbers the
    ridge needs: compute capability, SMs, DRAM GB/s (bus bits x Gbps / 8), boost clock, and
    whether the board lacks tensor cores (`mma_slow`: the generation's tensor rate is not its).
    With them the row's name and chip, by which collect() tells it is the board of the CSVs."""
    wanted = int(str(pci_id), 16)
    data = json.loads(Path(gpu_facts).read_text(encoding="utf-8"))
    for row in data["gpus"]:
        if int(row["id"], 16) == wanted:
            slow = row.get("mma_slow", False)
            if type(slow) is not bool:
                raise ValueError(f"{gpu_facts}: {row['id']}: mma_slow is not true or false")
            return {"id": f"0x{wanted:04x}", "name": row["name"], "chip": row["chip"],
                    "cc": int(row["cc"]),
                    "sms": int(row["sms"]), "dram_gbps": row["bus_bits"] * row["gbps"] / 8.0,
                    "boost_mhz": row["boost_mhz"], "mma_slow": slow}
    raise ValueError(f"{gpu_facts}: no row for PCI id {pci_id}")


def recorded_pci_id(path, rows):
    """The PCI device id a file's rows record (an int), or None: a file written before route_tune
    recorded it, or a board whose driver did not report one (the column empty)."""
    if not rows or "pci_device_id" not in rows[0]:
        return None
    values = {(row.get("pci_device_id") or "").strip().lower() for row in rows}
    if len(values) != 1:
        raise ValueError(f"{path}: rows from more than one PCI device id ({', '.join(sorted(values))})")
    value = values.pop()
    if not value:
        return None
    try:
        return int(value, 16)
    except ValueError:
        raise ValueError(f"{path}: pci_device_id {value!r} is not a hex id") from None


def collect(selected_csvs, pass_csvs, board, model, revision, build_id, holdout, fit_boards,
            fit_models, qualified, facts=None, gpu_facts=None):
    if not pass_csvs:
        raise ValueError("at least one --routes pass")
    if isinstance(selected_csvs, (str, Path)):
        selected_csvs = [selected_csvs]
    qualified = (qualified or "").strip()  # a blank statement qualifies nothing
    if len(selected_csvs) != len(pass_csvs):
        raise ValueError(f"{len(selected_csvs)} --selected files for {len(pass_csvs)} passes: "
                         "one per pass")
    authorities = {}
    selected = read_selections(selected_csvs, authorities)
    passes = {}
    protocols = {}  # (problem, T) -> {protocol: the passes that timed it so}
    boards = set()
    pci_ids = set()  # the PCI device ids the rows record (newer route_tune files)
    runs = set()
    for i, (sel_path, path) in enumerate(zip(selected_csvs, pass_csvs), 1):
        rows = read_rows(path, ROUTE_COLUMNS)
        sel_rows = read_rows(sel_path, SELECTED_COLUMNS)
        run, sel_run = identity(path, rows, "run"), identity(sel_path, sel_rows, "run")
        # A pass's selection and its sweep are one route_tune run.
        if run != sel_run:
            raise ValueError(f"{sel_path} and {path} come from different runs ({sel_run}, {run})")
        # The same run given twice is one pass, not two that agree.
        if run in runs:
            raise ValueError(f"{path}: run {run} is given more than once")
        runs.add(run)
        boards |= {identity(path, rows, "device"), identity(sel_path, sel_rows, "device")}
        pci_ids |= {recorded_pci_id(path, rows), recorded_pci_id(sel_path, sel_rows)} - {None}
        timed = {}
        for row in rows:
            us = float(row["median_us"])
            at = (key_of(row), row["route"].strip())
            # route_tune times a (route, T) once per run: a repeat is two runs in one file.
            if at in timed:
                raise ValueError(f"{path}: {case_id(at[0])} {at[1]} timed twice in one pass")
            timed[at] = us
            protocols.setdefault(at[0], {}).setdefault(protocol_of(row), set()).add(f"p{i}")
        passes[f"p{i}"] = timed
    if len(boards) > 1:
        raise ValueError("passes from more than one board: " + ", ".join(sorted(boards)))
    if len(pci_ids) > 1:
        raise ValueError("passes from more than one board: PCI ids "
                         + ", ".join(f"0x{p:04x}" for p in sorted(pci_ids)))
    recorded = next(iter(pci_ids), None)
    if recorded is not None:
        if facts and int(facts["id"], 16) != recorded:
            # Within a class only the id tells the SKU, whose DRAM rate and clock the ridge reads.
            raise ValueError(f"--pci-id names {facts['name']} ({facts['id']}); the rows were "
                             f"recorded on PCI id 0x{recorded:04x}")
        if not facts and gpu_facts is not None:
            facts = board_facts(gpu_facts, f"0x{recorded:04x}")
    for key, seen in protocols.items():
        if len(seen) > 1:
            kinds = "; ".join(f"{describe(p)} in {', '.join(sorted(ps))}" for p, ps in seen.items())
            raise ValueError(f"{case_id(key)} was timed with different protocols: {kinds}")
    cases = []
    for key in sorted(selected, key=lambda k: (k[:5], int(k[5]))):
        routes = sorted({r for timed in passes.values() for (k, r) in timed if k == key})
        candidates = []
        for route in routes:
            samples = {p: timed[(key, route)] / 1000.0 for p, timed in passes.items()
                       if (key, route) in timed}
            if key[1] == "a16" and route_class(route) != "a16":
                candidates.append({"id": route, "status": "ineligible",
                                   "reason": f"{route_class(route)} activations, not this case's a16"})
            elif len(samples) != len(passes):
                missing = sorted(set(passes) - set(samples))
                candidates.append({"id": route, "status": "unmeasured",
                                   "reason": "not timed in pass " + ", ".join(missing)})
            elif not qualified:
                candidates.append({"id": route, "status": "unmeasured",
                                   "reason": "no correctness qualification stated (--qualified)"})
            else:
                candidates.append({"id": route, "status": "ok", "samples_ms": samples})
        if selected[key] not in routes:
            candidates.append({"id": selected[key], "status": "unmeasured",
                               "reason": "the selected route was not timed"})
        fmt, compute, epilogue, n, k, t = key
        cases.append({
            "id": case_id(key), "board": board, "model": model,
            "workload": f"route_tune {fmt} {compute} {epilogue} n={n} k={k} T={t}, policy "
                        f"{POLICY.get(compute, compute)}, speed route profile, " +
                        (describe(next(iter(protocols[key]))) if key in protocols else "no route timed"),
            "numerical_contract": f"{compute} activations under {POLICY.get(compute, compute)}",
            "selected": selected[key], "candidates": candidates, "t": int(t),
            "authority": authorities[key][0], "evidence": authorities[key][1],
        })
    if not cases:
        raise ValueError("no selections")
    document = {
        "schema": "ninferno-selection-regret/1", "selector_revision": revision,
        "build_id": build_id, "holdout": holdout, "fit_boards": sorted(set(fit_boards)),
        "fit_models": sorted(set(fit_models)), "cases": cases,
    }
    if qualified:
        document["qualification"] = qualified
    device = boards.pop()
    if facts:
        # The registry row must be the board these CSVs came from. The device's class id names
        # its chip and its SMs ("tu116:24sm"); the SMs alone do not tell a GTX 1660 Ti from an
        # RTX 4060 (ad107, 24 SMs too), whose ridge the first does not have. A board whose chip
        # the class tables do not hold is named by its architecture ("sm90:132sm", an H100).
        named = re.match(r"([^:\s]+):(\d+)sm\b", device)
        if not named:
            raise ValueError(f"--pci-id: the rows' device {device} is not a class id "
                             "(<chip>:<SMs>sm), so the board cannot be told; leave --pci-id out")
        if (named[1] not in (facts["chip"], f"sm{facts['cc']}")
                or int(named[2]) != facts["sms"]):
            raise ValueError(f"--pci-id names {facts['name']} ({facts['chip']}, sm{facts['cc']}, "
                             f"{facts['sms']} SMs); the rows' device is {device}")
        document["board_facts"] = {board: facts}
    document["device"] = device  # route_tune's device class id, as every row named it
    if recorded is not None:
        document["pci_device_id"] = f"0x{recorded:04x}"
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selected", type=Path, nargs="+", required=True,
                        help="each pass's --selected-out; they must agree")
    parser.add_argument("--routes", type=Path, nargs="+", required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--model", required=True, help="the checkpoint the shapes come from")
    parser.add_argument("--selector-revision", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--holdout", choices=("board", "model", "both"), required=True)
    parser.add_argument("--fit-boards", default="", help="comma-separated")
    parser.add_argument("--fit-models", default="", help="comma-separated")
    parser.add_argument("--qualified", default="",
                        help="the correctness tests that passed on this build and board")
    parser.add_argument("--pci-id", default="",
                        help="the board's PCI device id (hex), for the ridge of the report's regimes")
    parser.add_argument("--gpu-facts", type=Path,
                        default=Path(__file__).resolve().parents[2] / "src/core/data/gpu_facts.json")
    args = parser.parse_args(argv)
    split = lambda s: [x for x in (v.strip() for v in s.split(",")) if x]
    try:
        facts = board_facts(args.gpu_facts, args.pci_id) if args.pci_id else None
        document = collect(args.selected, args.routes, args.board, args.model,
                           args.selector_revision, args.build_id, args.holdout,
                           split(args.fit_boards), split(args.fit_models), args.qualified, facts,
                           gpu_facts=args.gpu_facts)
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as error:  # a short row
        print(f"regret_collect: {error}", file=sys.stderr)
        return 2
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
