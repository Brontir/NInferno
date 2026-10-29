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
    """The event reasons active at any of the samples; None when unknown (no reason fields, or no
    sample inside the window): never a pass by itself."""
    prefix = tel.get("reason_fields")
    if not prefix or not indices:
        return None
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
    columns = tel.get("columns") or {}

    def column(name):
        values = columns.get(name) or []
        return [values[i] for i in idle if i < len(values) and isinstance(values[i], float)]
    util, mem = column("utilization.gpu"), column("memory.used")
    # The process list counts under TCC/Linux; under WDDM it lists the desktop's processes too
    # (informative false): there R2 reads the idle window only.
    for when in ("processes_before", "processes_after"):
        apps = p.get(when) or {}
        if apps.get("informative") and apps.get("others"):
            return f"R2 {apps['others']} other compute process(es) on the card", notes, {}
    if util and statistics.median(util) > IDLE_UTIL_MAX:
        return f"R2 idle utilization {statistics.median(util):.0f} % > {IDLE_UTIL_MAX:.0f} %", notes, {}
    if len(mem) >= 2 and mem[-1] - mem[0] > IDLE_MEM_GROWTH_MIB:
        return f"R2 memory grew {mem[-1] - mem[0]:.0f} MiB in the idle window", notes, {}
    if not util:
        notes.append("no idle samples: R2 by the process list only")
    per_metric = {}
    for metric in ("prefill", "decode"):
        found = reasons_in(tel, window_indices(tel, metric))
        if found is None:
            notes.append(f"{metric}: no clock event reasons in its window: R3 unknown")
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


def card_unknowns(smi):
    """What R4 could not check (the fields N/A on this card or driver)."""
    out = []
    if not isinstance(smi.get("power.default_limit"), float):
        out.append("the power limit's default")
    if not isinstance(smi.get("clocks.default_applications.graphics"), float):
        out.append("the application clocks")
    return out


def metric_values(p, metric):
    m = (p.get("measured") or {}).get(metric) or {}
    return m.get("median"), m.get("min"), m.get("max"), m.get("prompt_tokens")


def validate(doc, tolerance=TOLERANCE):
    report = {"schema": doc.get("schema"), "file": "accepted", "capped": [], "runs": []}
    schema = str(doc.get("schema", ""))
    major = schema.split(".")[0]
    if (major != SCHEMA_MAJOR or not isinstance(doc.get("card"), dict)
            or not isinstance(doc.get("runs"), list)
            or not all(isinstance(r, dict) for r in doc["runs"])):
        report["file"] = "rejected: R0 schema"
        return report
    rejected, capped = card_rules(doc)
    if rejected:
        report["file"] = "rejected: " + rejected
        return report
    report["capped"] = capped
    report["unknown"] = card_unknowns(doc["card"].get("smi") or {})
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
        mismatch, identity_unknown = None, False
        for p in passes:
            if p.get("error"):
                continue
            served = p.get("served") or {}
            line = served.get("choice") or ""
            # A start that does not say what it chose cannot be held to the plan: no verdict.
            if not line or not served.get("fingerprint"):
                identity_unknown = True
            for key, word in (("kv", "KV"), ("activations", "activations")):
                if line and chosen.get(key) and f"{word} {chosen[key]}" not in line:
                    mismatch = f"R1 the start chose '{line.split('|')[0].strip()}', the plan {key} {chosen[key]}"
        prints = {(p.get("served") or {}).get("fingerprint") for p in passes if not p.get("error")}
        prints.discard(None)
        if len(prints) > 1:
            mismatch = "R1 the passes started with different decision fingerprints"
        if identity_unknown:
            out["notes"].append("R1 unknown: a start did not report its choice or fingerprint")
        if mismatch:
            out["notes"].append(mismatch)
            continue
        kept = []
        for i, p in enumerate(passes):
            rule, notes, per_metric = pass_rules(p)
            out["notes"] += [f"pass {i + 1}: {n}" for n in notes]
            if rule:
                out["notes"].append(f"pass {i + 1} rejected: {rule}")
                continue
            kept.append((p, per_metric))
        predicted = {"decode": chosen.get("decode_tok_s_at_depth"), "prefill": chosen.get("prefill_tok_s")}
        depth = chosen.get("depth")
        # The lengths the plan priced: decode at the choice's depth, prefill at its typical prompt.
        at = {"decode": depth, "prefill": plan.get("prompt")}
        for metric in ("decode", "prefill"):
            row = {"metric": metric, "predicted": predicted[metric]}
            usable, unknown = [], False
            for p, per_metric in kept:
                found = per_metric.get(metric)
                unknown = unknown or found is None
                found = found or set()
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
                want = at[metric]
                # Every contributing pass at the length the plan priced (each request's range).
                off = None
                for p in usable:
                    m = (p.get("measured") or {}).get(metric) or {}
                    span = m.get("prompt_tokens_range") or [m.get("prompt_tokens")] * 2
                    if not want or not all(isinstance(x, (int, float)) for x in span):
                        off = "no token count"
                        break
                    worst = max(abs(x - want) for x in span)
                    if worst / want > DEPTH_MATCH:
                        off = f"a request at {span[0] if abs(span[0] - want) == worst else span[1]:.0f} tokens"
                        break
                if off:
                    row["status"] = f"rejected: R7 {off}, the plan priced {want}"
                else:
                    measured = statistics.median(medians)
                    row["measured"] = round(measured, 2)
                    row["status"] = "accepted"
                    if isinstance(row["predicted"], (int, float)) and row["predicted"] > 0:
                        ratio = measured / row["predicted"]
                        row["ratio"] = round(ratio, 3)
                        entry = (f"the speed model's estimate for this {doc['card'].get('smi', {}).get('name')}"
                                 f" ({run.get('repo')} {run.get('quant')}, {metric}, C=1, "
                                 f"{'depth' if metric == 'decode' else 'prompt'} {want})")
                        if chosen.get("speed_transferred"):
                            entry += ", transferred from another card: tests the transfer rule"
                        if capped:
                            row["verdict"] = "not used: the card is capped"
                        elif unknown:
                            row["verdict"] = "not used: R3 unknown (no clock event reasons in a window)"
                        elif identity_unknown:
                            row["verdict"] = "not used: R1 unknown (a start did not report its choice)"
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
    if report.get("unknown"):
        lines.append("R4 could not check: " + "; ".join(report["unknown"]))
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
