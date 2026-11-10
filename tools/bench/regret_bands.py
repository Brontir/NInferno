#!/usr/bin/env python3
"""Band-weighted selection regret (#613 PR-0). See README.md.

Reads recorded route choices and reports the extra time they cost, per T band, grouped by board,
format and route family. Offline and descriptive: it selects nothing, fits nothing, runs no GPU.

Inputs (--doc files and --routes groups are repeatable, but not mixed in one run: legacy CSVs
carry no provenance to report next to the documents'):
  --doc FILE.json     a ninferno-selection-regret/1 document (what regret_collect.py writes);
  --routes P1.csv ... --selected SEL.csv --board NAME
                      route_tune --routes-out passes of one board and its --selected-out file. These
                      are accepted without the run/device columns regret_collect.py demands, because
                      the existing sweep archive predates them. Without those columns nothing tells
                      two passes of different builds or boards apart: give passes of one run. route_tune checks no
                      results, so --qualified TEXT must state what qualified the candidates.
                      --doc inputs of different selector_revision, build_id, holdout or fit
                      boards / models are refused, and so is a case (board, model, id) that two
                      inputs both hold.
  --band-weights J    a mean of the band regrets with these weights (band -> weight); a group that
                      has no case in a weighted band gets no mean
  --pick model        with --routes only (not --doc): score the pick of the model price (lowest
                      model_us; T >= 17, where the model ranks; not NVFP4, which it does not rank,
                      nor an int8 q4 / q5 / q6 case whose I8 tiles route_tune does not sweep)
                      instead of the recorded production selection. --selected is still needed.
With every case skipped the report is still written (it lists them) and the exit status is 2.

--routes-out also holds the prefill profile's rows (its widths from chunk/2 and its bands from 32)
with no profile column. The speed loop writes one selection per problem and width and the prefill
loop none, so only the widths --selected lists are cases, and at those widths a route the speed
loop does not time there is left out (`speed_loop_times`); the report counts the pairs it left out
as prefill_profile_rows_excluded. This tool says nothing about prefill.
"""
import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regret_collect as collect  # noqa: E402
import selection_regret as regret  # noqa: E402

# Upper edge (inclusive) of each band and what it is in the product. The decode regime is T <= 8
# (R8 and the vector kernel), 9..16 is a speculative verify, 17..32 the narrow batched widths where
# the cost model misranks most, 33..128 wide verify/batched decode, and 129+ the widest batched
# calls route_tune times (192..512, speed profile; not the prefill profile's chunks).
BANDS = (("decode", 1, 8), ("verify", 9, 16), ("narrow", 17, 32), ("wide", 33, 128),
         ("batch129", 129, None))
BAND_NAMES = tuple(b[0] for b in BANDS)


MODEL_MIN_T = 17  # the cost model ranks routes only from this many tokens

# What route_tune's speed loop times at a width (route_tune_tool.cpp, the `for (const auto t :
# points)` loop): every admitted route, except R8 above kR8MaxTokens (generic_gemm.h: 4) and the
# w4a4, staged, NVFP4 tiled and tiled families below 17 tokens, plus the executed selection. Its
# prefill loop runs after it and writes a (route, T) pair the speed loop did not; the routes CSV
# has no profile column, so a row outside this is the prefill profile's, never a speed candidate.
R8_MAX_TOKENS = 4
WIDE_ONLY_FAMILIES = ("w4a4", "staged", "nvfp4t", "tiled")
WIDE_FROM_T = 17
# The I8 tiles: cost_route offers them (generic_cost.cpp) where allows_i8 holds (generic_gemm.cpp:
# an int8 policy, a qx_g64 format of format_traits.h, q4 / q5 / q6 by their route names, K a
# multiple of 512, aligned planes), but route_tune's route list has no I8 family.
I8_FORMATS = ("q4", "q5", "q6")
I8_K_MULTIPLE = 512


def speed_loop_times(route, t, executed):
    """Whether route_tune's speed loop writes `route` at width `t` (`executed`: the selection)."""
    if route == executed:
        return True
    family = family_of(route)
    if family == "r8":
        return t <= R8_MAX_TOKENS
    return not (family in WIDE_ONLY_FAMILIES and t < WIDE_FROM_T)


def band_of(t):
    for name, lo, hi in BANDS:
        if t >= lo and (hi is None or t <= hi):
            return name
    raise ValueError(f"T={t}: below the first band")


def family_of(route):
    """The route family: the part of the id before its first dot (rmma, ggml_mmq, r8, w4a8 ...)."""
    return route.split(".", 1)[0]


def offer_family(route):
    """The family the cost model offers a route in: `family_of`, but w4a8ws is a W4A8 schedule.

    route_name (generic_gemm.cpp) spells the producer-warp W4A8 schedule `w4a8ws.128x64.p4`; it is
    Family::W4a8Mma schedule index 5, offered after the five `w4a8.` schedules, so the two
    prefixes are one family to the price's first-offered rule.
    """
    family = family_of(route)
    return "w4a8" if family == "w4a8ws" else family


def significant_digits(text):
    """Digits of a number's mantissa as written, without leading and trailing zeros."""
    mantissa = text.strip().lstrip("+-").lower().split("e")[0].replace(".", "")
    return len(mantissa.strip("0"))


# route_tune wrote model_us with the stream's default six significant digits until it printed 17.
LEGACY_MODEL_DIGITS = 6


def offer_order_votes(sequences, family):
    """Pairwise votes (a, b) -> how many sequences write route a before route b, of one family."""
    votes = {}
    for seq in sequences:
        own = [r for r in seq if offer_family(r) == family]
        for i, a in enumerate(own):
            for b in own[i + 1:]:
                votes[(a, b)] = votes.get((a, b), 0) + 1
    return votes


def first_offered(tied, votes):
    """The tied route that precedes every other tied route by a strict majority, else None."""
    for a in tied:
        if all(votes.get((a, b), 0) > votes.get((b, a), 0) for b in tied if b != a):
            return a
    return None


def tokens_of(case):
    for text, pattern in ((case["id"], r"\.T(\d+)$"), (case.get("workload", ""), r"\bT=(\d+)")):
        found = re.search(pattern, text)
        if found:
            return int(found.group(1))
    raise ValueError(f"{case['id']}: no T in id or workload")


def format_of(case):
    return case["id"].split(".", 1)[0]


def skip_row(case_id, board, fmt, t, pick, reason):
    """A case that could not be scored: kept as a row so every band says what it lost.

    `pick` is the route the selector chose, or None where there is none (the model priced nothing).
    """
    return {"id": case_id, "board": board, "format": fmt, "t": t, "selected": pick,
            "reason": reason}


def rows_from_document(document):
    """Complete cases of a regret document as rows; the others come back as skipped rows."""
    report = regret.evaluate(document)
    by_id = {c["id"]: c for c in document["cases"]}
    rows, skipped = [], []
    for case in report["cases"]:
        src = by_id[case["id"]]
        if case["status"] != "complete":
            skipped.append(dict(skip_row(case["id"], case["board"], format_of(case), tokens_of(src),
                                         case["selected"], f"document case {case['status']}"),
                                model=src["model"]))
            continue
        medians = case["median_ms"]
        rows.append({
            "id": case["id"], "board": case["board"], "model": src["model"],
            "format": format_of(case), "t": tokens_of(src), "selected": case["selected"],
            "best": case["best_observed"], "selected_ms": medians[case["selected"]],
            "best_ms": medians[case["best_observed"]]})
    return rows, skipped


def refuse_repeated_cases(inputs):
    """Refuse a case that two inputs both hold: merged, it would count and weigh twice.

    `inputs` is [(input name, rows + skip rows)]. A case is its (board, model, id); route CSV rows
    have no model. A case scored in one input and skipped in another is a repeat too.
    """
    seen = {}
    for name, cases in inputs:
        for r in cases:
            identity = (r["board"], r.get("model"), r["id"])
            if identity in seen:
                raise ValueError(f"{name} and {seen[identity]} both hold case {r['id']} of board "
                                 f"{r['board']}" + (f", model {r['model']}" if r.get("model")
                                                    else "") + ": a case is counted once")
            seen[identity] = name


def provenance_of(document):
    """The selector revision and build a regret document was measured with, and its evaluation
    scope: what is held out and the boards and models the selector was fit on (as sets). One
    frozen selector has one fit inventory, so documents of one selector_revision that name
    different ones, or hold out another identity, are not one evaluation."""
    prov = {k: document.get(k) for k in ("selector_revision", "build_id", "holdout")}
    for k in ("fit_boards", "fit_models"):
        prov[k] = sorted(document.get(k) or [])
    return prov


def merged_board_devices(named):
    """Board label -> device class of the merged --doc inputs; one label on two devices is refused.

    `named` is (file name, device or None, board labels). A document that names no device says
    nothing about its boards. Two documents naming one label on different devices (their case ids
    may differ) would pool two GPUs into one by_board cell with no trace, so the merge is rejected.
    """
    seen = {}  # board -> (device, file)
    for name, device, boards in named:
        if not device:
            continue
        for board in sorted(boards):
            if board in seen and seen[board][0] != device:
                raise ValueError(f"board {board} is on device {seen[board][0]} in {seen[board][1]} "
                                 f"but on {device} in {name}: one board label is one device class")
            seen.setdefault(board, (device, name))
    return {board: device for board, (device, _) in sorted(seen.items())}


def merged_provenance(named):
    """Provenance of the merged --doc inputs; inputs of different provenance are refused.

    Rows of different selectors or builds averaged into one band metric could not be attributed or
    reproduced, so the merge is rejected and the error says which field differs between which files.
    """
    first_name, first = named[0]
    for name, prov in named[1:]:
        diff = [k for k in first if prov[k] != first[k]]
        if diff:
            raise ValueError(
                f"{name} and {first_name} differ in {', '.join(diff)} "
                f"({', '.join(f'{k}: {prov[k]!r} vs {first[k]!r}' for k in diff)}); "
                "report them one at a time, not merged")
    return dict(first)


def read_selected(selected_path):
    """The selection CSV: key -> the executed route, one row per key."""
    selected = {}
    for row in collect.read_rows(selected_path, collect.KEY + ("selected",)):
        key = collect.key_of(row)
        if key in selected:
            raise ValueError(f"{selected_path}: {collect.case_id(key)} selected twice")
        selected[key] = row["selected"].strip()
    return selected


def rows_from_csv(route_paths, selected_path, board, pick="selected", excluded=None):
    """Rows from legacy route_tune CSVs: per case, medians over passes of the routes of its class.

    The selection CSV is needed in both modes: route_tune writes one selection per problem and
    width of its speed loop and none in its prefill loop, so its widths are the speed profile's.
    The speed widths are per problem: a case is a (problem, T) the selection lists. A routes row
    at a width no problem's selection lists, or of a route the speed loop does not time at its
    width (`speed_loop_times`), is a prefill-profile timing: it is not a case or a candidate, and
    its (case id, route) is appended to `excluded` when a list is given. A row at a width the
    selection lists only for other problems is a case without its selection: skipped in both picks.
    """
    if pick not in ("selected", "model"):
        raise ValueError("pick must be selected or model")
    if not selected_path:
        raise ValueError("--selected is needed: its widths are the speed profile's (route_tune "
                         "also writes prefill-profile rows into --routes-out, with no profile "
                         "column)")
    selected = read_selected(selected_path)
    widths = {int(k[5]) for k in selected}
    timed, model = {}, {}  # (key, route) -> [us per pass]
    written = {}  # (pass, key) -> speed-loop routes in the order the pass wrote them
    digits = {}  # pass -> most significant digits of any model_us it wrote
    excluded_pairs = set()  # (key, route) of prefill-profile rows, over all passes
    unselected = set()  # keys at a selected width of another problem, without their own selection
    for number, path in enumerate(route_paths):
        columns = collect.KEY + ("route", "median_us") + (("model_us",) if pick == "model" else ())
        seen = set()
        for row in collect.read_rows(path, columns):
            at = (collect.key_of(row), row["route"].strip())
            if at in seen:
                raise ValueError(f"{path}: {collect.case_id(at[0])} {at[1]} timed twice in one pass")
            seen.add(at)
            if pick == "model":
                # Of every row: the pass's writer printed all its prices with one precision.
                digits[number] = max(digits.get(number, 0),
                                     significant_digits(row.get("model_us") or ""))
            t = int(at[0][5])
            if at[0] not in selected and t in widths:
                # Another problem's selection lists this width, this problem's none: a selection
                # cut short or of another run. Nothing proves the rows are the speed loop's, so
                # the case is no candidate set in either pick; it is a skip, not left out unseen.
                unselected.add(at[0])
                continue
            if not (at[0] in selected and speed_loop_times(at[1], t, selected[at[0]])):
                if excluded is not None and (at[0], at[1]) not in excluded_pairs:
                    excluded.append((collect.case_id(at[0]), at[1]))
                excluded_pairs.add((at[0], at[1]))
                continue
            written.setdefault((number, at[0]), []).append(at[1])
            us = float(row["median_us"])
            if not (math.isfinite(us) and us > 0):
                raise ValueError(f"{path}: {collect.case_id(at[0])} {at[1]}: median_us {us}")
            timed.setdefault(at, []).append(us)
            if pick == "model":
                # Empty where the model prices no route (route_tune leaves the cell blank).
                text = (row.get("model_us") or "").strip()
                price = float(text) if text else None
                model.setdefault(at, []).append(
                    price if price is not None and math.isfinite(price) and price > 0 else None)
    passes = len(route_paths)
    routes = {}
    for key, route in timed:
        routes.setdefault(key, []).append(route)
    rows, skipped = [], []
    for key in sorted(unselected, key=lambda k: (k[:5], int(k[5]))):
        skipped.append(skip_row(collect.case_id(key), board, key[0], int(key[5]), None,
                                "no selection recorded for this case"))
    # A selection the sweep has no row for at all (a pass cut short) is skipped, not dropped, in
    # either pick: the selection file proves the case exists.
    for key in sorted(set(selected) - set(routes), key=lambda k: (k[:5], int(k[5]))):
        skipped.append(skip_row(collect.case_id(key), board, key[0], int(key[5]), selected[key],
                                "pick has no sweep rows"))
    for key in sorted(routes, key=lambda k: (k[:5], int(k[5]))):
        eligible = [r for r in routes[key]
                    if not (key[1] == "a16" and collect.route_class(r) != "a16")]
        # A route a pass did not time leaves the case without a comparable best: the case is
        # skipped, never scored without that route (selection_regret.score_case alike).
        def skip(pick, reason):
            skipped.append(skip_row(collect.case_id(key), board, key[0], int(key[5]), pick, reason))

        if any(len(timed[(key, r)]) != passes for r in eligible):
            skip(selected.get(key) if pick == "selected" else None,
                 "a candidate was not timed in every pass")
            continue
        cands = {r: statistics.median(timed[(key, r)]) for r in eligible}
        if pick == "model":
            if int(key[5]) < MODEL_MIN_T:  # the model ranks only from 17 tokens
                skip(None, "the model ranks only from T 17")
                continue
            if key[0] == "nvfp4":
                # route_authorities.cpp cost_model declines NVFP4: route_tune writes model_us for
                # it as a diagnostic, but no production pick follows that price.
                skip(None, "the model does not rank NVFP4 routes")
                continue
            if key[0] in I8_FORMATS and key[1] != "a16" and int(key[4]) % I8_K_MULTIPLE == 0:
                # cost_route also prices the I8 schedules here (generic_cost.cpp, allows_i8), but
                # route_tune's sweep holds none of them (only an executed one is timed): the
                # model's pick among all its offers is not in the file.
                skip(None, "the model also prices the I8 schedules, which route_tune does not "
                           "sweep")
                continue
            priced = {r: statistics.median(model[(key, r)]) for r in cands
                      if all(x is not None for x in model[(key, r)])}
            if not priced:
                skip(None, "the model prices none of the candidates")
                continue
            low = min(priced.values())
            tied = [r for r in routes[key] if r in priced and priced[r] == low]
            if len({offer_family(r) for r in tied}) > 1:
                # Production (generic_cost.cpp cost_route, a strict `<`) keeps the first route
                # offered on an equal price. Across families the offer order is not in the file
                # (tiled and staged are offered in other orders), so the pick is unknown and the
                # case is skipped, not guessed.
                skip(None, "the model prices the best candidates of two families equally")
                continue
            if len(tied) > 1 and any(digits.get(n, 0) <= LEGACY_MODEL_DIGITS
                                       for n in range(passes)):
                # A pass written with six digits (before route_tune printed 17) cannot tell a
                # tie from two prices that differ in the seventh digit, where production takes
                # the cheaper: no value of its model_us is as long as a 17-digit one.
                skip(None, "model_us has six significant digits: an equal price may be a rounding")
                continue
            # Within one family the offer order is the schedule index. The order is read from the
            # other speed-loop widths of the same problem (`written` holds no prefill-profile row:
            # the prefill loop writes the prefill route first); a pair they do not settle is a skip.
            votes = offer_order_votes(
                [seq for (number, other), seq in written.items()
                 if other[:5] == key[:5] and other != key], offer_family(tied[0]))
            chosen = first_offered(tied, votes)
            if chosen is None:
                skip(None, "the pass wrote the tied rows out of offer order")
                continue
        else:
            chosen = selected.get(key)
        if chosen is None:
            skip(None, "no selection recorded for this case")
            continue
        if chosen not in cands:
            # Timed, but of another compute class than the case's key (an a8 route under an a16
            # key) is left out of the candidates above; say so rather than "not timed".
            skip(chosen, "the pick is not of the case's compute class" if chosen in routes[key]
                 else "the pick was not timed among the candidates")
            continue
        if len(cands) < 2:
            # One candidate of the case's class: nothing to compare, and 0 % would read as a win
            # (the document evaluator calls this no_comparison).
            skip(chosen, "fewer than two candidates of the case's compute class")
            continue
        best = min(cands, key=lambda r: (cands[r], r))
        rows.append({"id": collect.case_id(key), "board": board, "format": key[0],
                     "t": int(key[5]), "selected": chosen, "best": best,
                     "selected_ms": cands[chosen] / 1000.0, "best_ms": cands[best] / 1000.0})
    return rows, skipped


def stats(rows, skipped=0):
    """Time-weighted regret: extra time of the picks over the best, as a share of the best time.

    `skipped` is the number of cases of the group that could not be scored; a group with none scored
    reads None, not zero.
    """
    if not rows:
        return {"cases": 0, "skipped": skipped, "tokens": 0, "selected_ms": 0.0, "best_ms": 0.0,
                "weighted_regret": None, "geomean_regret": None, "worst_regret": None,
                "cases_over_5pct": 0}
    sel = math.fsum(r["selected_ms"] for r in rows)
    best = math.fsum(r["best_ms"] for r in rows)
    ratios = [r["selected_ms"] / r["best_ms"] for r in rows]
    return {
        "cases": len(rows), "skipped": skipped, "tokens": sum(r["t"] for r in rows), "selected_ms": sel, "best_ms": best,
        "weighted_regret": sel / best - 1,
        "geomean_regret": math.exp(math.fsum(map(math.log, ratios)) / len(ratios)) - 1,
        "worst_regret": max(ratios) - 1,
        "cases_over_5pct": sum(1 for x in ratios if x > 1.05),
    }


def by_band(rows, skipped=()):
    """Per band stats; a band with only skipped cases is present, with cases 0."""
    return {b: stats([r for r in rows if band_of(r["t"]) == b],
                     sum(1 for r in skipped if band_of(r["t"]) == b))
            for b in BAND_NAMES if any(band_of(r["t"]) == b for r in list(rows) + list(skipped))}


def report(rows, skipped=(), band_weights=None):
    """Per band overall, then per board / format / route family, each by band.

    With `band_weights` (band -> weight) every group also gets "weighted", the mean of its band
    regrets by those weights, or None where the group has no case in a band of nonzero weight (a
    mean over the bands it has would stand for a mix it does not cover). It is also None where a
    weighted band has skipped cases, and "weighted_incomplete" then lists those bands: the mean of
    the scored cases would read lower than the truth, and a printed skip share would still leave a
    number that looks like a regret.

    `skipped` is the list of skip rows the readers return; each keeps its T, board, format and
    reason, counts in its group and band, and is listed under "skipped_cases". The family of a skip
    row is the family of the pick, or "(no pick)" where the model priced none.
    """
    for r in list(rows) + list(skipped):
        r["family"] = family_of(r["selected"]) if r["selected"] else "(no pick)"
    out = {"schema": "ninferno-regret-bands/1", "bands": {n: [lo, hi] for n, lo, hi in BANDS},
           "metric": "sum(selected time) / sum(best observed time) - 1 per group and band",
           "cases": len(rows), "cases_skipped": len(skipped), "overall": by_band(rows, skipped),
           "skipped_cases": [{k: r[k] for k in ("id", "board", "format", "t", "selected", "family",
                                                 "reason")}
                             for r in skipped]}
    for dim in ("board", "format", "family"):
        out["by_" + dim] = {
            k: by_band([r for r in rows if r[dim] == k], [r for r in skipped if r[dim] == k])
            for k in sorted({r[dim] for r in list(rows) + list(skipped)})}
    if band_weights is not None:
        if not isinstance(band_weights, dict):
            raise ValueError("band weights must be a JSON object of band -> weight")
        unknown = set(band_weights) - set(BAND_NAMES)
        if (unknown
                or not all(isinstance(w, (int, float)) and not isinstance(w, bool)
                           and math.isfinite(w) and w >= 0
                           for w in band_weights.values())
                or not any(band_weights.values())):
            raise ValueError(f"band weights must name {BAND_NAMES} and not all be zero")
        out["band_weights"] = band_weights
    if band_weights is None:
        return out
    total = sum(band_weights.values())
    groups = [out["overall"]] + [g for k in out if k.startswith("by_") for g in out[k].values()]
    for group in groups:
        used = [b for b, v in band_weights.items() if v > 0]
        covered = all(b in group and group[b]["cases"] for b in used)
        incomplete = [b for b in used if b in group and group[b]["skipped"]]
        group["weighted"] = (sum(group[b]["weighted_regret"] * band_weights[b] for b in used)
                             / total if covered and not incomplete else None)
        if incomplete:
            group["weighted_incomplete"] = incomplete
    return out


def pct(x):
    return "-" if x is None else f"{100 * x:+.1f}%"


def markdown(rep):
    lines = ["# Band-weighted selection regret", "",
             f"{rep['cases']} cases scored, {rep['cases_skipped']} skipped. Each cell is the extra time "
             "of the picks over the best observed route, summed over the band's scored cases "
             "(time-weighted), then `(n / skipped)`: the scored and the skipped cases of the band; a "
             "cell with skipped cases does not cover them."
             + (" `weighted` averages the bands by the given weights (`-` where a weighted band "
                "has no scored case or has skipped cases)." if "band_weights" in rep else ""),
             "", "Bands (T): " + ", ".join(f"{n} {lo}-{hi}" if hi else f"{n} {lo}+"
                                           for n, (lo, hi) in rep["bands"].items()), ""]
    weighted = "band_weights" in rep

    def table(title, groups):
        head = ["| " + title + " | " + " | ".join(f"{b} (n / skipped)" for b in BAND_NAMES)
                + (" | weighted |" if weighted else " |"),
                "|---|" + "---:|" * (len(BAND_NAMES) + weighted)]
        for name, bands in groups.items():
            cells = [f"{pct(bands[b]['weighted_regret'])} ({bands[b]['cases']} / "
                     f"{bands[b]['skipped']})" if b in bands else "" for b in BAND_NAMES]
            if weighted:
                cells.append(pct(bands.get("weighted")))
            head.append(f"| {name} | " + " | ".join(cells) + " |")
        return head + [""]

    lines += table("all", {"all": rep["overall"]})
    for dim in ("board", "format", "family"):
        lines += table(dim, rep["by_" + dim])
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--doc", type=Path, action="append", default=[])
    p.add_argument("--routes", type=Path, nargs="+", action="append", default=[],
                   help="route_tune passes of one board (one group per --routes)")
    p.add_argument("--selected", type=Path, action="append", default=[],
                   help="selection CSV for the matching --routes group")
    p.add_argument("--board", action="append", default=[], help="board name for the matching --routes group")
    p.add_argument("--qualified", default="",
                   help="what qualified the --routes candidates for correctness (route_tune does "
                        "not check results); required with --routes")
    p.add_argument("--pick", choices=("selected", "model"), default="selected")
    p.add_argument("--band-weights", help='JSON, e.g. {"decode":0.4,"verify":0.2}')
    p.add_argument("--json", type=Path, help="write the JSON report here")
    p.add_argument("--md", type=Path, help="write the markdown report here")
    args = p.parse_args(argv)
    try:
        if args.doc and args.routes:
            raise ValueError("--doc and --routes cannot be merged: documents carry the selector "
                             "revision and build, legacy route CSVs carry none, so the report's "
                             "provenance would label rows of an unknown build; run them separately")
        if args.pick == "model" and args.doc:
            raise ValueError("--pick model scores the model price's own pick from --routes; it "
                             "cannot be combined with --doc")
        rows, skipped, named, inputs, excluded, devices = [], [], [], [], [], []
        for doc in args.doc:
            loaded = json.loads(doc.read_bytes(), object_pairs_hook=regret.unique_object)
            r, s = rows_from_document(loaded)
            named.append((doc.name, provenance_of(loaded)))
            devices.append((doc.name, loaded.get("device"), {x["board"] for x in r + s}))
            inputs.append((str(doc), r + s))
            rows += r
            skipped += s
        provenance = merged_provenance(named) if named else None
        board_devices = merged_board_devices(devices)
        if len(args.board) != len(args.routes):
            raise ValueError(f"one --board per --routes group ({len(args.board)} boards, "
                             f"{len(args.routes)} groups)")
        if len(args.selected) > len(args.routes):
            raise ValueError(f"at most one --selected per --routes group ({len(args.selected)} "
                             f"selected, {len(args.routes)} groups)")
        if args.routes and not args.qualified.strip():
            raise ValueError("--routes needs --qualified: route_tune does not check results, so "
                             "state the correctness tests that qualified these candidates")
        if args.routes and len(args.selected) != len(args.routes):
            raise ValueError(f"one --selected per --routes group, with either --pick: its widths "
                             f"tell the speed profile's rows from the prefill profile's "
                             f"({len(args.selected)} selected, {len(args.routes)} groups)")
        for i, passes in enumerate(args.routes):
            r, s = rows_from_csv(passes, args.selected[i], args.board[i], args.pick, excluded)
            inputs.append((f"--routes group {i + 1} ({args.board[i]})", r + s))
            rows += r
            skipped += s
        refuse_repeated_cases(inputs)
        if not rows and not skipped:
            raise ValueError("no cases in the inputs")
        weights = json.loads(args.band_weights) if args.band_weights else None
        rep = report(rows, skipped, weights)
        if provenance is not None:
            rep["provenance"] = provenance
        if board_devices:
            rep["board_devices"] = board_devices
        if args.routes:
            rep["legacy_qualification"] = args.qualified.strip()
            # Prefill-profile (case, route) pairs of the route CSVs, left out (rows_from_csv).
            rep["prefill_profile_rows_excluded"] = len(excluded)
        # The report is written also when every case was skipped: it is where the skipped cases
        # and their reasons are listed. The exit status then says nothing was scored.
        text = json.dumps(rep, indent=2, sort_keys=True, allow_nan=False)
        if args.json:
            args.json.write_text(text + "\n", encoding="utf-8")
        if args.md:
            args.md.write_text(markdown(rep) + "\n", encoding="utf-8")
        if not args.json and not args.md:
            print(markdown(rep))
    except (OSError, UnicodeError, ValueError, KeyError, OverflowError) as error:
        print(f"regret_bands: {error}", file=sys.stderr)
        return 2
    if not rows:
        print(f"regret_bands: no scored cases ({len(skipped)} skipped; the report lists them)",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
