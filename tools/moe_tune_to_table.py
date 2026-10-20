#!/usr/bin/env python3
"""Turn the MoE tuner's cache into route-table `moe` entries.

    tools/moe_tune_to_table.py --cache moe_tune.json --table routes/C.json [--class C]
                               [--gpu-name NAME] [--tuned-on MODEL] [--replace]

The engine's startup MoE tuner (src/ops/moe/moe_tune.cpp, NINFERNO_MOE_TUNE) times every compiled
gate/up and down schedule per decode width T = 1..16 and keeps a plan only where it beats the
default by 3 %; its cache (`ninferno-moe-tune/2`, default moe_tune.json in NInferno's cache directory) holds,
per layer, the kept plan per T (null: the default stays) and the default and best times. This tool
writes those plans as the decode `bands` of a `ninferno-route-table/1` `moe` entry: consecutive T
with one plan form one band; a T whose default stayed is left out (the default runs there, as it
did under the tuner). A release that ships the table runs the tuned plans with no startup timing.

Only cache layers of the table's GPU class are taken (the class is the key's first field; a chip
table takes its SM variants' layers; --class overrides the table's own). The table file is created when missing. A layer the table already has
is kept unless --replace. Layers that differ only in activation, expert policy or bank layout share
one table key; the first is taken and the rest are reported.
"""
import argparse
import json
import sys
from pathlib import Path

# QType ids (src/core/weight.h) -> route-table format names (src/ops/route_table.cpp parse_qtype).
QTYPES = {
    0: "q4", 1: "q5", 2: "q6", 3: "q8", 4: "bf16", 7: "nvfp4", 8: "fp8", 9: "mxfp4", 10: "fp8r32",
    11: "fp8b128", 12: "u4zp", 40: "u4zp32", 41: "u4zp64", 42: "q3g128", 43: "q3g64", 13: "q4k",
    14: "q5k", 15: "q6k", 16: "q3k", 17: "q2k", 18: "iq4xs", 19: "iq3xxs", 20: "iq3s", 21: "q40",
    22: "q41", 23: "q50", 24: "q51", 25: "iq4nl", 26: "iq2xxs", 27: "iq2xs", 28: "iq2s", 29: "iq1s",
}


def parse_key(key):
    """'ad106:34sm|20/3|E128|k8|H2816|F704|act1|a8|s9.4gd' -> (class, entry key fields)."""
    f = key.split("|")
    if len(f) < 7 or not f[2].startswith("E") or not f[3].startswith("k"):
        raise ValueError(f"unknown moe tune key {key!r}")
    gu, dn = (int(b.rstrip("f")) for b in f[1].split("/"))
    if gu not in QTYPES or dn not in QTYPES:
        raise ValueError(f"moe tune key {key!r}: a format the route table cannot name")
    fields = {"format": QTYPES[gu]}
    if dn != gu:
        fields["down_format"] = QTYPES[dn]
    fields.update(experts=int(f[2][1:]), top_k=int(f[3][1:]), hidden=int(f[4][1:]), inter=int(f[5][1:]))
    return f[0], fields


def bands_of(plans):
    """{"1": plan|None, ...} -> [[t0, t1, plan], ...] over the T with a kept plan."""
    out = []
    for t in sorted(int(k) for k in plans):
        p = plans[str(t)]
        if p is None:
            continue
        if out and out[-1][1] == t - 1 and out[-1][2] == p:
            out[-1][1] = t
        else:
            out.append([t, t, p])
    return out


def ident(e):
    return (e["format"], e.get("down_format", e["format"]), e["experts"], e["top_k"], e["hidden"], e["inter"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--class", dest="gpu_class")
    ap.add_argument("--gpu-name")
    ap.add_argument("--tuned-on", help="the model the layers were tuned on, recorded per entry")
    ap.add_argument("--replace", action="store_true", help="replace layers the table already has")
    a = ap.parse_args()

    cache = json.loads(a.cache.read_text(encoding="utf-8"))
    if cache.get("schema") != "ninferno-moe-tune/2":
        sys.exit(f"{a.cache}: not ninferno-moe-tune/2")
    if a.table.exists():
        table = json.loads(a.table.read_text(encoding="utf-8"))
        if table.get("schema") != "ninferno-route-table/1":
            sys.exit(f"{a.table}: not ninferno-route-table/1")
    else:
        if not a.gpu_class:
            sys.exit(f"{a.table} does not exist: --class names the GPU class of a new table")
        table = {"schema": "ninferno-route-table/1",
                 "gpu": {"class": a.gpu_class, "name": a.gpu_name or a.gpu_class}, "entries": []}
    gpu_class = a.gpu_class or table["gpu"]["class"]
    moe = table.setdefault("moe", [])
    have = {ident(e): i for i, e in enumerate(moe)}
    taken, added, replaced, kept, skipped = set(), 0, 0, 0, []
    for key, layer in cache.get("layers", {}).items():
        cls, fields = parse_key(key)
        # A chip table ("ad106") takes its SM variants' layers ("ad106:34sm"); a variant table
        # takes only its own.
        if cls != gpu_class and (":" in gpu_class or cls.split(":")[0] != gpu_class):
            continue
        bands = bands_of(layer.get("plans", {}))
        if not bands:
            skipped.append(f"{key}: the default plan stays at every T")
            continue
        entry = dict(fields)
        if a.tuned_on:
            entry["tuned_on"] = a.tuned_on
        entry["bands"] = bands
        us = layer.get("us", {})
        entry["evidence_decode_us"] = [[b[0], *us[str(b[0])]] for b in bands if str(b[0]) in us]
        k = ident(entry)
        if k in taken:
            skipped.append(f"{key}: a second layer with the table key of an earlier one")
            continue
        taken.add(k)
        if k in have:
            if not a.replace:
                kept += 1
                continue
            old = moe[have[k]]
            if "prefill" in old:  # the tuner times decode only: a measured prefill plan stays
                entry["prefill"] = old["prefill"]
            moe[have[k]] = entry
            replaced += 1
        else:
            moe.append(entry)
            added += 1
    a.table.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{a.table}: {added} moe layers added, {replaced} replaced, {kept} kept")
    for s in skipped:
        print("  skipped " + s)


if __name__ == "__main__":
    main()
