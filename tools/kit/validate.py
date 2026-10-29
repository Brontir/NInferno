#!/usr/bin/env python3
"""Validate a user measurement kit result (docs/maintainer/user-measurement-kit.md): the stated
rejection rules, then predicted against measured per accepted metric and what it would confirm or
contradict.

    python3 tools/kit/validate.py RUN.json [--json] [--tolerance 0.10]

A submission is evidence for a maintainer, never a change: a contradiction names the estimate it
disagrees with, and the model is changed only after it is reproduced, from facts, never by fitting.
"""
import argparse
import json
import statistics
import sys

SCHEMA_MAJOR = "ninferno-user-run/1"
# The stated rules' numbers (the design note's table; change them there first).
IDLE_UTIL_MAX = 5.0            # % median utilization in the idle window before a pass (R2)
IDLE_MEM_GROWTH_MIB = 256.0    # memory.used growth across the idle window (R2)
REPEAT_MEDIANS = 0.05          # two passes' medians apart (R5)
REPEAT_SPREAD = 0.10           # one pass's min..max over its median (R5)
DEPTH_MATCH = 0.05             # measured prompt tokens against the plan's depth (R7)
TOLERANCE = 0.10               # predicted against measured: confirm within, contradict outside
REJECT_REASONS = ("hw_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown",
                  "sw_thermal_slowdown")
RECORD_REASONS = ("sw_power_cap",)


def active(value):
    return isinstance(value, str) and value.strip().lower() == "active"


def window_indices(tel, name_prefix):
    """The sample indices inside the marks whose name starts with `name_prefix`."""
    t = tel.get("t") or []
    out = []
    for mark in tel.get("marks") or []:
        if mark["name"].startswith(name_prefix):
            out += [i for i, x in enumerate(t) if mark["start"] <= x < mark["end"]]
    return sorted(set(out))


def reasons_in(tel, indices):
    """The event reasons active at any of the samples."""
    prefix = tel.get("reason_fields")
    if not prefix:
        return None  # unknown: never a pass by itself
    found = set()
    for reason in REJECT_REASONS + RECORD_REASONS + ("applications_clocks_setting",):
        column = tel["columns"].get(f"{prefix}.{reason}") or []
        if any(active(column[i]) for i in indices if i < len(column)):
            found.add(reason)
    return found


def pass_rules(p):
    """R2, R3 and R6 for one pass: (rejected rule or None, notes, reasons per metric)."""
    if p.get("error"):
        return "R6 a pass with an error", [], {}
    tel = p.get("telemetry") or {}
    notes = []
    idle = window_indices(tel, "idle")
    util = [tel["columns"].get("utilization.gpu", [None])[i] for i in idle] if idle else []
    util = [u for u in util if isinstance(u, float)]
    mem = [tel["columns"].get("memory.used", [None])[i] for i in idle] if idle else []
    mem = [m for m in mem if isinstance(m, float)]
    if p.get("other_processes"):
        return "R2 another compute process on the card", notes, {}
    if util and statistics.median(util) > IDLE_UTIL_MAX:
        return f"R2 idle utilization {statistics.median(util):.0f} % > {IDLE_UTIL_MAX:.0f} %", notes, {}
    if len(mem) >= 2 and mem[-1] - mem[0] > IDLE_MEM_GROWTH_MIB:
        return f"R2 memory grew {mem[-1] - mem[0]:.0f} MiB in the idle window", notes, {}
    if not idle:
        notes.append("no idle window recorded: R2 by the process list only")
    per_metric = {}
    for metric in ("prefill", "decode"):
        found = reasons_in(tel, window_indices(tel, metric))
        if found is None:
            notes.append("the driver reports no clock event reasons: R3 unknown")
        per_metric[metric] = found
    return None, notes, per_metric


def card_rules(doc):
    """R1 and R4 for the file: (rejected or None, capped marks)."""
    card = doc.get("card") or {}
    smi = card.get("smi") or {}
    plan_device = card.get("plan_device") or {}
    if plan_device.get("name") and smi.get("name") and plan_device["name"] != smi["name"]:
        return f"R1 the plan priced {plan_device['name']}, nvidia-smi names {smi['name']}", []
    capped = []
    limit, default = smi.get("enforced.power.limit") or smi.get("power.limit"), smi.get("power.default_limit")
    if isinstance(limit, float) and isinstance(default, float) and limit < default - 1:
        capped.append(f"power limit {limit:.0f} W below the default {default:.0f} W")
    app, app_default = smi.get("clocks.applications.graphics"), smi.get("clocks.default_applications.graphics")
    if isinstance(app, float) and isinstance(app_default, float) and app < app_default:
        capped.append(f"application clocks {app:.0f} below the default {app_default:.0f} MHz")
    if active((smi.get("reasons_now") or {}).get(f"{smi.get('reason_fields')}.applications_clocks_setting")):
        capped.append("a locked clock (applications clocks setting)")
    if doc.get("host", {}).get("on_battery"):
        capped.append("on battery")
    return None, capped


def metric_values(p, metric):
    m = (p.get("measured") or {}).get(metric) or {}
    return m.get("median"), m.get("min"), m.get("max"), m.get("prompt_tokens")


def validate(doc, tolerance=TOLERANCE):
    report = {"schema": doc.get("schema"), "file": "accepted", "capped": [], "runs": []}
    if not str(doc.get("schema", "")).startswith(SCHEMA_MAJOR) or "card" not in doc or "runs" not in doc:
        report["file"] = "rejected: R0 schema"
        return report
    rejected, capped = card_rules(doc)
    if rejected:
        report["file"] = "rejected: " + rejected
        return report
    report["capped"] = capped
    for run in doc["runs"]:
        out = {"id": run.get("id"), "mode": run.get("mode"), "metrics": [], "notes": []}
        report["runs"].append(out)
        if run.get("error"):
            out["notes"].append("R6 " + str(run["error"]))
            continue
        if run.get("mode") != "plain":
            out["notes"].append("recorded, not validated in this version (" + str(run.get("mode")) + ")")
            continue
        plan = run.get("plan") or {}
        passes = run.get("passes") or []
        # R1: every start ran what the plan priced (its storages, as the start's `choice |` line
        # says them) and the same decisions (one fingerprint across the passes).
        chosen = plan.get("chosen") or {}
        mismatch = None
        for p in passes:
            line = (p.get("served") or {}).get("choice") or ""
            for key, word in (("kv", "KV"), ("activations", "activations")):
                if line and chosen.get(key) and f"{word} {chosen[key]}" not in line:
                    mismatch = f"R1 the start chose '{line.split('|')[0].strip()}', the plan {key} {chosen[key]}"
        prints = {(p.get("served") or {}).get("fingerprint") for p in passes if p.get("served")}
        prints.discard(None)
        if len(prints) > 1:
            mismatch = "R1 the passes started with different decision fingerprints"
        if mismatch:
            out["notes"].append(mismatch)
            continue
        kept, rule_of = [], {}
        for i, p in enumerate(passes):
            rule, notes, per_metric = pass_rules(p)
            out["notes"] += [f"pass {i + 1}: {n}" for n in notes]
            if rule:
                out["notes"].append(f"pass {i + 1} rejected: {rule}")
                continue
            kept.append((p, per_metric))
        predicted = {"decode": chosen.get("decode_tok_s_at_depth"), "prefill": chosen.get("prefill_tok_s")}
        depth = chosen.get("depth")
        for metric in ("decode", "prefill"):
            row = {"metric": metric, "predicted": predicted[metric]}
            usable = []
            for p, per_metric in kept:
                found = per_metric.get(metric) or set()
                bad = sorted(found & set(REJECT_REASONS))
                if bad:
                    row.setdefault("notes", []).append("a pass rejected: R3 " + ",".join(bad))
                    continue
                if found & set(RECORD_REASONS):
                    row.setdefault("notes", []).append("power cap events (recorded, stock behaviour)")
                usable.append(p)
            medians = [metric_values(p, metric)[0] for p in usable]
            medians = [m for m in medians if isinstance(m, (int, float)) and m > 0]
            if len(medians) < 2:
                row["status"] = "rejected: R6 fewer than 2 usable passes"
                out["metrics"].append(row)
                continue
            spreads = []
            for p in usable:
                med, lo, hi, _ = metric_values(p, metric)
                if med and lo is not None and hi is not None:
                    spreads.append((hi - lo) / med)
            apart = (max(medians) - min(medians)) / min(medians)
            if apart > REPEAT_MEDIANS:
                row["status"] = f"rejected: R5 passes {apart * 100:.1f} % apart"
            elif spreads and max(spreads) > REPEAT_SPREAD:
                row["status"] = f"rejected: R5 a pass spans {max(spreads) * 100:.1f} % of its median"
            else:
                tokens = [metric_values(p, metric)[3] for p in usable]
                tokens = [t for t in tokens if isinstance(t, (int, float))]
                if metric == "decode" and depth and tokens and abs(statistics.median(tokens) - depth) / depth > DEPTH_MATCH:
                    row["status"] = (f"rejected: R7 measured at {statistics.median(tokens):.0f} tokens, "
                                     f"the plan's depth is {depth}")
                else:
                    measured = statistics.median(medians)
                    row["measured"] = round(measured, 2)
                    row["status"] = "accepted"
                    if isinstance(row["predicted"], (int, float)) and row["predicted"] > 0:
                        ratio = measured / row["predicted"]
                        row["ratio"] = round(ratio, 3)
                        entry = (f"the speed model's estimate for {doc['card'].get('smi', {}).get('name')}"
                                 f" ({run.get('repo')} {run.get('quant')}, {metric}, C=1"
                                 + (f", depth {depth}" if metric == "decode" else "") + ")")
                        if chosen.get("speed_transferred"):
                            entry += ", transferred from another card: tests the transfer rule"
                        if capped:
                            row["verdict"] = "not used: the card is capped"
                        elif abs(ratio - 1) <= tolerance:
                            row["verdict"] = "confirms " + entry
                        else:
                            row["verdict"] = "contradicts " + entry
            out["metrics"].append(row)
    return report


def render(report):
    lines = [f"schema {report['schema']}: {report['file']}"]
    if report["capped"]:
        lines.append("capped (kept for the record, not used to confirm): " + "; ".join(report["capped"]))
    for run in report["runs"]:
        lines.append(f"- {run['id']} ({run['mode']})")
        for note in run["notes"]:
            lines.append(f"    note: {note}")
        for m in run["metrics"]:
            text = f"    {m['metric']}: {m['status']}"
            if "measured" in m:
                text += f"; predicted {m['predicted']}, measured {m['measured']}"
            if "ratio" in m:
                text += f" (x{m['ratio']})"
            lines.append(text)
            if m.get("verdict"):
                lines.append(f"      {m['verdict']}")
            for note in m.get("notes", []):
                lines.append(f"      note: {note}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run", help="a ninferno-user-run JSON file")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()
    with open(args.run, encoding="utf-8") as f:
        doc = json.load(f)
    report = validate(doc, args.tolerance)
    print(json.dumps(report, indent=1) if args.json else render(report))
    return 0 if not report["file"].startswith("rejected") else 1


if __name__ == "__main__":
    sys.exit(main())
