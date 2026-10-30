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
from pathlib import Path

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


def window_load(passes, metric):
    """The median SM clock (MHz) and power draw (W) over the metric's windows of `passes`, each None
    when no sample fell inside one: the clock a speed was measured at (a long prefill runs into the
    power limit, which a short one finishes before the limiter acts)."""
    clock, power = [], []
    for p in passes:
        tel = p.get("telemetry") or {}
        columns = tel.get("columns") or {}
        for i in window_indices(tel, metric):
            for name, out in (("clocks.sm", clock), ("power.draw", power)):
                values = columns.get(name) or []
                if i < len(values) and isinstance(values[i], float):
                    out.append(values[i])
    return (round(statistics.median(clock)) if clock else None,
            round(statistics.median(power)) if power else None)


def reasons_in(tel, indices):
    """The event reasons active at any of the samples; None when unknown (no reason fields, or no
    sample inside the window): never a pass by itself."""
    prefix = tel.get("reason_fields")
    if not prefix or not indices:
        return None
    found, seen = set(), False
    for reason in REJECT_REASONS + RECORD_REASONS + ("applications_clocks_setting",):
        column = (tel.get("columns") or {}).get(f"{prefix}.{reason}") or []
        values = [column[i] for i in indices if i < len(column)]
        seen = seen or any(isinstance(v, str) for v in values)
        if any(active(v) for v in values):
            found.add(reason)
    # Every sample N/A: the reasons were not read in this window.
    return found if seen else None


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
    if not plan_device.get("name") or not smi.get("name"):
        return "R1 unknown: the plan's or nvidia-smi's name of the card is missing", []
    if plan_device["name"] != smi["name"]:
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
    if (doc.get("host") or {}).get("on_battery"):
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


def power_unknown(doc):
    """Whether the power source could not be told (a battery whose state is unknown)."""
    host = doc.get("host") or {}
    return "on_battery" not in host or host.get("on_battery") is None


def metric_values(p, metric):
    m = (p.get("measured") or {}).get(metric) or {}
    return m.get("median"), m.get("min"), m.get("max"), m.get("prompt_tokens")


# --- Speculation (S1-S6) and hard50 (H1-H3): the design note's tables; change them there first. ---

HERE = Path(__file__).resolve().parent
REGISTRY = HERE.parents[1] / "src/spec/data/drafter_registry.json"
HARD_REFERENCE = HERE / "hard50_reference.json"
ACCEPT_TOLERANCE = 0.10   # pooled tokens a round against the lab cards' median (S5)
ACCEPT_MIN_CARDS = 2      # lab cards a configuration's reference needs (S5)
ROUND_TOLERANCE = 0.10    # measured R(k) against the plan's round_cost (S6)
PLAIN_STEADY = 0.05       # the tune's plain decode across its sections (S4)
HARD_SDS = 2.0            # binomial standard deviations below the lab's lowest card (H3)
HARD_MIN_CARDS = 3        # lab cards the item-by-item comparison needs (H3)
FACTS_WORDS = "priced from the weights' bytes"


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def settings_items(text):
    """A tune's settings ("max context 4096, prefill chunk 1024, KV bf16, greedy, wait spin") as a
    set of items; `acceptance` drops the wait mode (it moves the clock, never a token)."""
    return {s.strip() for s in str(text or "").split(",") if s.strip()}


def acceptance_settings(text):
    return {s for s in settings_items(text) if not s.startswith("wait")}


def sections_of(measurements):
    """The corpus sections plain decode measured, with their prompt counts."""
    return {m.get("family"): m.get("prompts") for m in measurements
            if m.get("config") == "plain" and (m.get("tokens") or 0) > 0}


def pooled_tokens_per_round(measurements, label, sections):
    """Tokens a round of `label` over `sections`, pooled ((accepted + rounds) / rounds, as
    spec::aggregate); None when a section is missing or has no round."""
    accepted = rounds = 0
    for family in sections:
        rows = [m for m in measurements if m.get("config") == label and m.get("family") == family]
        if not rows or not sum(m.get("rounds") or 0 for m in rows):
            return None
        accepted += sum(m.get("accepted") or 0 for m in rows)
        rounds += sum(m.get("rounds") or 0 for m in rows)
    return (accepted + rounds) / rounds if rounds else None


def section_tokens_per_round(measurements, label, family):
    return pooled_tokens_per_round(measurements, label, [family])


def round_costs(measurements):
    """What a round costs by configuration label, in plain decode steps: per section a round's
    seconds over plain decode's seconds a token, the mean over the sections plain decode measured
    too (spec::profile_round_costs, src/spec/pairs.cpp)."""
    plain, rounds = {}, {}
    for m in measurements:
        if m.get("config") == "plain":
            if (m.get("tokens") or 0) > 0:
                s = plain.setdefault(m.get("family"), [0.0, 0.0])
                s[0] += m.get("seconds") or 0
                s[1] += m.get("tokens") or 0
        elif (m.get("rounds") or 0) > 0:
            r = rounds.setdefault((m.get("config"), m.get("family")), [0.0, 0.0])
            r[0] += m.get("seconds") or 0
            r[1] += m.get("rounds") or 0
    sums = {}
    for (label, family), (seconds, n) in rounds.items():
        p = plain.get(family)
        if not p or p[0] <= 0:
            continue
        s = sums.setdefault(label, [0.0, 0])
        s[0] += (seconds / n) / (p[0] / p[1])
        s[1] += 1
    return {label: s[0] / s[1] for label, s in sums.items()}


def chip_of(gpu_class):
    return str(gpu_class or "").split(":")[0].lower()


# The drafter kinds a configuration's mode drafts with (spec::drafter_kind_mode, pairs.cpp): an
# "mtp:K" label runs a Gemma 4 assistant or a NemotronH sidecar where the target has one.
MODE_KINDS = {"mtp": ("mtp", "gemma4_assistant", "nemotron_h_mtp")}


def kinds_of(label):
    mode = str(label).split(":")[0]
    return MODE_KINDS.get(mode, (mode,))


def drafter_matches(pair, manifest, label):
    """The pair drafted `label` with the drafter the user's manifest attached: each separate
    drafter of the label's mode by the same repo@revision (an embedded MTP head is the target's
    own)."""
    for kind in kinds_of(label):
        wanted = (pair.get("drafters") or {}).get(kind)
        if not wanted:
            continue
        mine = [d for d in (manifest or {}).get("drafters", []) if d.get("kind") == kind
                and d.get("repo") not in (None, "", "other")]
        if not any(f"{d['repo']}@{d.get('revision', '')}" == wanted for d in mine):
            return False
    return True


def unrecordable_drafters(manifest, labels):
    """The measured modes' separate drafters the manifest could not name (a repository the
    registry does not declare: recorded as "other")."""
    kinds = {k for label in labels for k in kinds_of(label)}
    return sorted({d.get("kind") for d in (manifest or {}).get("drafters", [])
                   if d.get("kind") in kinds and d.get("repo") == "other"})


def lab_pairs(registry, run):
    """The lab's pairs of the run's target (by its weights' SHA-256), of the tune's engine key."""
    weights = ((run.get("target") or {}).get("weights")) or []
    key = ",".join(sorted(w.get("sha256", "") for w in weights))
    engine = ((run.get("profile") or {}).get("key") or {}).get("engine")
    return [p for p in (registry or {}).get("pairs", [])
            if key and p.get("target") == key and not p.get("user_kit")
            and ((p.get("tune") or {}).get("key") or {}).get("engine") == engine]


def prediction_kind(speculation, label, pairs, user_class):
    """What the plan's round cost for `label` is: the facts, the facts raised by a lab card's
    measurement, or a lab measurement (which card's, the source says: this class's own pair, a
    generation's agreeing cards, a sibling quantization or a generation ratio)."""
    source = str((speculation or {}).get("source") or "")
    if FACTS_WORDS in source:
        raised = source.split("bytes:", 1)[1] if "bytes:" in source else ""
        if any(part.strip().split(" ")[0] == label for part in raised.split(",")):
            return "the facts' R(k) raised to a lab card's measurement (" + raised.strip() + ")"
        return "the facts' R(k)"
    return "a lab measurement, not the facts (" + (source or "no source") + ")"


def spec_speed_gate(doc, run, capped):
    """S2-S4 for the round cost: None when it may be judged, else why not (rejected or not used)."""
    profile = run.get("profile") or {}
    plain = [m.get("tok_s") for m in profile.get("measurements", [])
             if m.get("config") == "plain" and isinstance(m.get("tok_s"), (int, float)) and m["tok_s"] > 0]
    if len(plain) >= 2 and (max(plain) - min(plain)) / min(plain) > PLAIN_STEADY:
        return f"rejected: S4 the tune's plain decode {(max(plain) - min(plain)) / min(plain) * 100:.1f} % apart across its sections"
    for when in ("processes_before", "processes_after"):
        apps = run.get(when) or {}
        if apps.get("informative") and apps.get("others"):
            return f"rejected: S3 {apps['others']} other compute process(es) on the card"
    tel = run.get("telemetry") or {}
    found = reasons_in(tel, window_indices(tel, "spec-tune"))
    if found and found & set(REJECT_REASONS):
        return "rejected: S3 " + ",".join(sorted(found & set(REJECT_REASONS))) + " during the tune"
    if capped:
        return "not used: the card is capped"
    if found is None:
        return "not used: S3 unknown (no clock event reasons in the tune's window)"
    if power_unknown(doc):
        return "not used: the power source (AC or battery) is unknown"
    return None


def judge_spec(doc, run, out, capped, registry, tolerance):
    profile, publish = run.get("profile"), run.get("publish")
    if not isinstance(profile, dict) or not isinstance(publish, dict):
        out["notes"].append("rejected: S1 no tune (spec-tune did not publish)")
        out["candidate"] = "no: no tune"
        return
    measurements = profile.get("measurements") or []
    sections = sections_of(measurements)
    labels = sorted({m.get("config") for m in measurements if m.get("config") not in (None, "plain")})
    partial = [d.get("name") for d in profile.get("drafters", []) if d.get("status") == "untested"]
    skipped = profile.get("skipped") or []
    if partial or skipped:
        out["notes"].append("S1 partial tune (judged on what it measured, no registry candidate): "
                            + "; ".join(([f"untested {', '.join(partial)}"] if partial else [])
                                        + ([f"skipped {', '.join(skipped)}"] if skipped else [])))
    pairs = lab_pairs(registry, run)
    target = run.get("target") or {}
    if not target.get("weights"):
        out["notes"].append("S2 the target's weights were not recorded: no lab pair can be named")
    elif not pairs:
        out["notes"].append("S5 no lab pair of this target (its weights) and engine key: no reference")
    if any(w.get("blob_matches") is False for w in target.get("weights", [])):
        out["notes"].append("S2 a weight file's bytes differ from its hub blob name (a modified or "
                            "corrupted download): judged by its own bytes")
    same_settings = [p for p in pairs if acceptance_settings((p.get("tune") or {}).get("settings"))
                     == acceptance_settings(profile.get("settings"))]
    if pairs and not same_settings:
        out["notes"].append("S2 the tune's settings differ from the lab pairs': no reference")
    gate = spec_speed_gate(doc, run, capped)
    speculation = (run.get("plan") or {}).get("speculation") or {}
    predicted_costs = speculation.get("round_cost") or {}
    measured_costs = round_costs(measurements)
    user_class = (profile.get("key") or {}).get("gpu")
    contradicted = unmeasured = False
    for label in labels:
        row = {"metric": f"acceptance {label}", "status": "accepted"}
        user_t = pooled_tokens_per_round(measurements, label, sections)
        measured_it = [p for p in same_settings
                       if pooled_tokens_per_round(p["tune"]["measurements"], label, sections_of(p["tune"]["measurements"]))]
        same_drafter = [p for p in measured_it if drafter_matches(p, run.get("manifest"), label)]
        refs = [p for p in same_drafter if sections_of(p["tune"]["measurements"]) == sections
                and pooled_tokens_per_round(p["tune"]["measurements"], label, sections)]
        lab_t = [pooled_tokens_per_round(p["tune"]["measurements"], label, sections) for p in refs]
        if user_t is None:
            row["status"] = "rejected: S1 a section without a round"
            unmeasured = True
        elif len(lab_t) < ACCEPT_MIN_CARDS:
            if not pairs or not same_settings:
                why = "no lab pair to compare with"
            elif not measured_it:
                why = "the lab measured no " + label + (" (it measures no proposal head)" if "+head" in label else "")
            elif not same_drafter:
                why = "the lab measured " + label + " with another drafter (repo@revision)"
            elif len(lab_t) < len(same_drafter):
                why = "the lab measured other sections or prompt counts"
            else:
                why = f"fewer than {ACCEPT_MIN_CARDS} lab cards"
            row["status"] = "accepted"
            row["verdict"] = f"no reference: {why}"
            row["measured"] = round(user_t, 3)
        else:
            ref = statistics.median(lab_t)
            ratio = user_t / ref
            row.update({"predicted": round(ref, 3), "measured": round(user_t, 3), "ratio": round(ratio, 3)})
            worst = max(abs(section_tokens_per_round(measurements, label, f)
                            / statistics.median([section_tokens_per_round(p["tune"]["measurements"], label, f)
                                                 for p in refs]) - 1) for f in sections)
            row.setdefault("notes", []).append(f"lab cards {len(lab_t)}, {min(lab_t):.3f}..{max(lab_t):.3f}; "
                                               f"worst section {worst * 100:.1f} % off")
            entry = f"the registry pairs' acceptance ({run.get('repo')} {run.get('quant')}, {label}, tokens a round)"
            if abs(ratio - 1) <= ACCEPT_TOLERANCE:
                row["verdict"] = "confirms " + entry
            else:
                row["verdict"] = "contradicts " + entry
                contradicted = True
        out["metrics"].append(row)
    for label in labels:
        row = {"metric": f"round cost {label}", "predicted": predicted_costs.get(label)}
        measured = measured_costs.get(label)
        if gate and gate.startswith("rejected"):
            row["status"] = gate
        elif measured is None:
            row["status"] = "rejected: S6 no section with both plain decode and rounds"
        else:
            row["status"] = "accepted"
            row["measured"] = round(measured, 3)
            if not isinstance(row["predicted"], (int, float)) or row["predicted"] <= 0:
                row["verdict"] = ("no prediction: the plan's resolved profile does not price " + label
                                  + " (deeper than the portable depth, or another drafter's profile)")
            else:
                ratio = measured / row["predicted"]
                row["ratio"] = round(ratio, 3)
                entry = (f"{prediction_kind(speculation, label, pairs, user_class)} for this "
                         f"{doc['card'].get('smi', {}).get('name')} ({run.get('repo')} {run.get('quant')}, {label})")
                if gate:
                    row["verdict"] = gate
                elif abs(ratio - 1) <= tolerance:
                    row["verdict"] = "confirms " + entry
                else:
                    row["verdict"] = "contradicts " + entry
        out["metrics"].append(row)
    if partial or skipped:
        out["candidate"] = "no: partial tune"
    elif not target.get("weights") or not run.get("manifest"):
        out["candidate"] = "no: the target's identity or the manifest was not recorded"
    elif unmeasured:
        out["candidate"] = "no: a configuration without a round in a section"
    elif unrecordable_drafters(run.get("manifest"), labels):
        out["candidate"] = ("no: a measured drafter the registry does not declare ("
                            + ", ".join(unrecordable_drafters(run.get("manifest"), labels)) + ")")
    elif contradicted:
        out["candidate"] = "no: an acceptance contradicts the lab pairs"
    elif pairs and not same_settings:
        out["candidate"] = "no: the tune's settings differ from the lab pairs'"
    else:
        out["candidate"] = "yes"


def judge_hard50(run, out, reference):
    h = run.get("hard50") or {}
    if h.get("questions") != 50 or (h.get("errors") or 0) > 0 or not isinstance(h.get("correct"), int):
        out["metrics"].append({"metric": "hard50", "status": "rejected: H1 not 50 answered questions "
                               f"({h.get('questions')} questions, {h.get('errors')} errors)"})
        return
    row = {"metric": "hard50", "status": "accepted", "measured": h["correct"]}
    out["metrics"].append(row)
    if not reference or run.get("set_sha256") != reference.get("set_sha256"):
        row["verdict"] = "no reference: H2 the question set differs from the reference's (or was not recorded)"
        return
    lab = ((reference.get("models") or {}).get(f"{run.get('repo')} {run.get('quant')}")) or []
    if not lab:
        row["verdict"] = "no reference: the lab has no hard50 run of this model"
        return
    counts = [r["correct"] for r in lab]
    p = statistics.median(counts) / 50
    sd = (50 * p * (1 - p)) ** 0.5
    low = min(counts) - HARD_SDS * sd
    row["predicted"] = statistics.median(counts)
    row.setdefault("notes", []).append(f"lab {len(counts)} card(s), {min(counts)}..{max(counts)} of 50; "
                                       f"below the lab under {low:.1f}")
    below = h["correct"] < low
    wrong = h.get("wrong")
    if isinstance(wrong, list) and len(lab) >= HARD_MIN_CARDS and all("wrong" in r for r in lab):
        union = set().union(*(set(r["wrong"]) for r in lab))
        flips = sorted(set(wrong) - union)
        lab_flips = [len(set(r["wrong"]) - set().union(*(set(o["wrong"]) for o in lab if o is not r)))
                     for r in lab]
        limit = 2 * max(lab_flips) + 2
        row["notes"].append(f"{len(flips)} question(s) every lab card answered right answered wrong "
                            f"here (a lab card against the others: at most {max(lab_flips)}; below the lab "
                            f"over {limit})")
        below = below or len(flips) > limit
    entry = f"the lab's hard50 of {run.get('repo')} {run.get('quant')}"
    row["verdict"] = ("below " + entry + ": a gross failure (a broken kernel, template or download)"
                      if below else "consistent with " + entry)


def validate(doc, tolerance=TOLERANCE, registry=None):
    """The report of `doc`; `registry` the drafter registry its spec runs are held to (the
    in-tree one when None)."""
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
    reference = None
    for run in doc["runs"]:
        out = {"id": run.get("id"), "mode": run.get("mode"), "metrics": [], "notes": []}
        report["runs"].append(out)
        if run.get("error"):
            out["notes"].append("R6 " + str(run["error"]))
            if run.get("mode") == "spec":
                out["candidate"] = "no: the run failed"
            continue
        if run.get("mode") == "spec":
            registry = registry if registry is not None else (load_json(REGISTRY) or {})
            judge_spec(doc, run, out, capped, registry, tolerance)
            continue
        if run.get("mode") == "hard50":
            reference = reference if reference is not None else (load_json(HARD_REFERENCE) or {})
            judge_hard50(run, out, reference)
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
            clock, power = window_load(usable, metric)
            if clock is not None:
                row["sm_clock_mhz"] = clock
            if power is not None:
                row["power_w"] = power
            medians = [metric_values(p, metric)[0] for p in usable]
            medians = [m for m in medians if isinstance(m, (int, float)) and m > 0]
            if len(medians) < 2:
                row["status"] = "rejected: R6 fewer than 2 usable passes"
                out["metrics"].append(row)
                continue
            spreads, no_spread = [], False
            for p in usable:
                med, lo, hi, _ = metric_values(p, metric)
                if all(isinstance(x, (int, float)) for x in (med, lo, hi)) and med > 0:
                    spreads.append((hi - lo) / med)
                else:
                    no_spread = True
            apart = (max(medians) - min(medians)) / min(medians)
            if no_spread:
                row["status"] = "rejected: R5 a pass without its min..max"
            elif apart > REPEAT_MEDIANS:
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
                        elif power_unknown(doc):
                            row["verdict"] = "not used: the power source (AC or battery) is unknown"
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
        if run.get("candidate"):
            lines.append(f"    registry candidate: {run['candidate']}")
        for m in run["metrics"]:
            text = f"    {m['metric']}: {m['status']}"
            if "measured" in m:
                text += (f"; predicted {m['predicted']}, measured {m['measured']}"
                         if m.get("predicted") is not None else f"; measured {m['measured']}")
            if "ratio" in m:
                text += f" (x{m['ratio']})"
            load = [f"{m['sm_clock_mhz']} MHz" if m.get("sm_clock_mhz") else None,
                    f"{m['power_w']} W" if m.get("power_w") else None]
            if any(load):
                text += " at " + ", ".join(x for x in load if x)
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
