#!/usr/bin/env python3
"""Offline audit of how the MMQ choice transfers between same-generation cards (#373).

Two layers, both from data in the repository; it never selects a route, fits a constant or runs a GPU.

1. Route tables (routes/*.json): for every same-generation card pair and every problem both tables
   hold, does the route the small-T `bands` pick at width T agree (ggml_mmq or not, and the exact
   route)? Split by workload regime. A table records only the winner per band, so this layer
   reports AGREEMENT and COVERAGE, never regret (no timing of the route the other card chose).
2. Gate labels (core/measured_boards.h kMmqPrefillPays, as the production rule carries them) against
   paired prefill timings (tools/bench/data/mmq_paired_prefill.json, copied from issue #373):
   regret of running card A's label on card B = how much slower B's prefill runs on the arm A's
   label picks than on the faster of {mmq, w4a8}, per format and prompt length.

Regimes come from architecture facts only: decode T<=4 (the vec/R8 row range), verify 5..16 (below
the 17-token prefill the MMQ gate is defined for: generic_gemm.cpp mmq_prefill_wins), and prefill
T>=17 split by the card's own ridge, T* = int8 ops/s * bytes_per_value / (2 * DRAM bytes/s), with
the generation's int8 tensor rate (core/generation.h s8), SMs and the reference boost clock
(core/data/gpu_facts.json). A pair straddles when the two cards' ridges differ: widths between
them are memory-bound on one card and compute-bound on the other.
"""
import argparse
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Q4_K: 4.5 bits a value. The ridge is shown for this one representative format.
BYTES_PER_VALUE = 4.5 / 8
DECODE_MAX, VERIFY_MAX = 4, 16
PREFILL_MIN = 17  # route_table.cpp kPrefillMinTokens
T_GRID = list(range(1, 17)) + [17, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
                               2048, 4096]
MMQ = "ggml_mmq"
REGIMES = ["decode T<=4", "verify T5-16", "prefill memory-bound (both)",
           "prefill straddles ridge", "prefill compute-bound (both)"]
# routes file -> card id used in the paired timings
PAIRED_CARDS = {"tu102": "rtx2080ti", "ga106": "rtx3060", "ad106": "rtx4060ti",
                "ad104-60sm": "rtx4070ti", "ad102": "rtx4090", "gb203-70sm": "rtx5070ti",
                "gb202": "rtx5090"}


def generations(path):
    """kGenerations rows of core/generation.h: (major, minor_lo, minor_hi, name, s8)."""
    rows = []
    text = Path(path).read_text(encoding="utf-8")
    for m in re.finditer(r'\{(\d+), (\d+), (\d+), "(\w+)", ([\d.]+), ([\d.]+),', text):
        rows.append((int(m[1]), int(m[2]), int(m[3]), m[4], float(m[6])))
    if not rows:
        raise ValueError("no generation rows found")
    return rows


def generation_of(rows, cc):
    major, minor = divmod(cc, 10)
    for r in rows:
        if r[0] == major and r[1] <= minor <= r[2]:
            return r
    raise ValueError(f"no generation row for cc {cc}")


def measured_boards(path):
    """kMmqPrefillPays: [(name, cc, sms, l2, dram, pays)] from core/measured_boards.h."""
    text = Path(path).read_text(encoding="utf-8")
    consts = {}
    for m in re.finditer(r"constexpr Board (k\w+)\{(\d+), (\d+), (\d+), ([\d.]+), ([\d.]+)"
                         r"(?:, (true|false))?\}", text):
        if m[7] == "true":
            raise ValueError(f"{m[1]}: mma_slow boards are not mirrored")
        consts[m[1]] = (int(m[2]) * 10 + int(m[3]), int(m[4]), float(m[5]), float(m[6]))
    table = re.search(r"kMmqPrefillPays\[\] = \{(.*?)\};", text, re.S)
    out = []
    for m in re.finditer(r"\{(k\w+), (true|false)\}", table[1]):
        cc, sms, l2, dram = consts[m[1]]
        out.append((m[1], cc, sms, l2, dram, m[2] == "true"))
    return out


def distance(a, b):
    """Mirror of core/device_class.cpp board_distance for two boards of one compute capability.

    The C++ also returns infinity for other pairs (mma_slow or instruction-set mismatch,
    sms <= 0, cross-cc boards of different FP32-accumulate kind); those are not mirrored, so they
    raise instead of being filtered silently."""
    if a.get("cc") is not None and b.get("cc") is not None and a["cc"] != b["cc"]:
        raise ValueError("board_distance mirror: different compute capabilities")
    if a.get("mma_slow", False) != b.get("mma_slow", False):
        raise ValueError("board_distance mirror: mma_slow differs (infinite in C++)")
    if a["sms"] <= 0 or b["sms"] <= 0:
        raise ValueError("board_distance mirror: sms <= 0 (infinite in C++)")
    def apart(x, y):
        return abs(math.log2(x) - math.log2(y)) if x > 0 and y > 0 else 0.0
    return apart(a["sms"], b["sms"]) + apart(a["dram_gbps"], b["dram_gbps"]) \
        + 0.5 * apart(a["l2_mib"], b["l2_mib"])


def boost_mhz(facts, board):
    for g in facts["gpus"]:
        if (g["cc"] == board["cc"] and g["sms"] == board["sms"] and "Laptop" not in g["name"]
                and abs(g["bus_bits"] * g["gbps"] / 8 - board["dram_gbps"]) < 2
                and g.get("l2_mib") == board["l2_mib"]):
            return g["boost_mhz"]
    return None


def load_cards(routes_dir, gen_rows, facts):
    cards = {}
    for path in sorted(Path(routes_dir).glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "board" not in doc["gpu"]:
            print(f"warning: {path.name}: no gpu.board, card skipped", file=sys.stderr)
            continue
        board = dict(doc["gpu"]["board"])
        gen = generation_of(gen_rows, board["cc"])
        clock = boost_mhz(facts, board)
        if clock is None:
            print(f"warning: {path.name}: no gpu_facts.json row matches board {board}; "
                  "its ridge is unknown", file=sys.stderr)
        ridge = None
        if clock and gen[4] > 0:
            ops = gen[4] * board["sms"] * clock * 1e6
            ridge = ops * BYTES_PER_VALUE / (2 * board["dram_gbps"] * 1e9)
        entries = {(e["format"], e["compute"], e.get("epilogue"), e["n"], e["k"]): e
                   for e in doc["entries"]}
        cards[path.stem] = {"name": doc["gpu"].get("name", path.stem), "board": board,
                            "generation": gen[3], "ridge_t": ridge, "entries": entries}
    return cards


def route_at(entry, t):
    """The route the table gives at t under the profile that serves t: prefill calls (t >= 17)
    run the prefill profile, which reads prefill_bands and then prefill; decode and verify calls
    read bands (route_table.cpp table_route)."""
    if t >= PREFILL_MIN:
        for lo, hi, route in entry.get("prefill_bands", []):
            if lo <= t <= hi:
                return route
        return entry.get("prefill")
    for lo, hi, route in entry.get("bands", []):
        if lo <= t <= hi:
            return route
    return None


def regime(t, ridges):
    if t <= DECODE_MAX:
        return REGIMES[0]
    if t <= VERIFY_MAX:
        return REGIMES[1]
    if t < min(ridges):
        return REGIMES[2]
    if t < max(ridges):
        return REGIMES[3]
    return REGIMES[4]


def table_audit(cards):
    by_gen = defaultdict(list)
    for key, card in cards.items():
        by_gen[card["generation"]].append(key)
    pairs = []
    for gen, keys in sorted(by_gen.items()):
        for a, b in itertools.combinations(sorted(keys), 2):
            ca, cb = cards[a], cards[b]
            ridges = [r for r in (ca["ridge_t"], cb["ridge_t"]) if r]
            if len(ridges) < 2:
                print(f"warning: pair {a}/{b} skipped: a ridge is unknown", file=sys.stderr)
                continue
            shared = sorted(set(ca["entries"]) & set(cb["entries"]),
                            key=lambda k: tuple(str(x) for x in k))
            stats = {r: {"points": 0, "mmq_a": 0, "mmq_b": 0, "mmq_agree": 0, "route_agree": 0}
                     for r in REGIMES}
            for key in shared:
                for t in T_GRID:
                    ra, rb = route_at(ca["entries"][key], t), route_at(cb["entries"][key], t)
                    if ra is None or rb is None:
                        continue
                    s = stats[regime(t, ridges)]
                    s["points"] += 1
                    s["mmq_a"] += ra == MMQ
                    s["mmq_b"] += rb == MMQ
                    s["mmq_agree"] += (ra == MMQ) == (rb == MMQ)
                    s["route_agree"] += ra == rb
            same_cc = ca["board"]["cc"] == cb["board"]["cc"]
            pairs.append({"generation": gen, "a": a, "b": b,
                          "distance": distance(ca["board"], cb["board"]) if same_cc else None,
                          "ridge_t": [ca["ridge_t"], cb["ridge_t"]],
                          "problems": [len(ca["entries"]), len(cb["entries"]), len(shared)],
                          "regimes": stats})
    return pairs


def gate_labels(measured, lab_cards):
    """The production gate's label per lab card: its own row, else the nearest row of the same
    compute capability (core/measured_boards.h mmq_prefill_gate, at any distance)."""
    labels = {}
    for card, c in lab_cards.items():
        board = c["board"]
        same = [m for m in measured if m[1] == board["cc"]]
        if not same:
            labels[card] = None
            continue

        def far(m):
            return distance(board, {"sms": m[2], "dram_gbps": m[4], "l2_mib": m[3]})
        best = min(same, key=far)
        labels[card] = {"pays": best[5], "donor": best[0], "distance": far(best)}
    return labels


def gate_audit(paired, labels, card_gen):
    """Regret of card A's gate label on card B, from B's paired timings. Same generation only."""
    rows = defaultdict(dict)
    for r in paired["rows"]:
        rows[r["card"]][(r["set"], r["model"], r["format"])] = r
    out = []
    for a, b in itertools.permutations(sorted(rows), 2):
        if card_gen.get(a) != card_gen.get(b) or labels.get(a) is None:
            continue
        regrets, wrong = [], 0
        for key, r in sorted(rows[b].items()):
            ratio = r["mmq"] / r["w4a8"]
            if labels[a]["pays"]:
                slow = 1 / ratio - 1 if ratio < 1 else 0.0
            else:
                slow = ratio - 1 if ratio > 1 else 0.0
            wrong += slow > 0
            regrets.append((slow, list(key), ratio))
        out.append({"generation": card_gen[a], "label_of": a, "applied_on": b,
                    "label_pays": labels[a]["pays"], "rows": len(regrets), "wrong": wrong,
                    "max_regret": max(r[0] for r in regrets),
                    "mean_regret": sum(r[0] for r in regrets) / len(regrets),
                    "worst": max(regrets, key=lambda r: r[0])[1:] if wrong else None})
    return out


def report(pairs, gates, cards):
    lines = ["# MMQ transfer audit (#373), offline", "",
             f"## Ridge T* per card (Q4_K, {BYTES_PER_VALUE} B/value)", "",
             "| card | generation | SMs | DRAM GB/s | T* |", "|---|---|---|---|---|"]
    for k, c in sorted(cards.items()):
        ridge = "-" if c["ridge_t"] is None else f"{c['ridge_t']:.0f}"
        lines.append(f"| {k} | {c['generation']} | {c['board']['sms']} | "
                     f"{c['board']['dram_gbps']:g} | {ridge} |")
    lines += ["", "## Route tables: MMQ-or-not DISAGREEMENTS by regime (disagree/points; MMQ points on A+B; exact-route disagreements)", "",
              "| generation | pair | dist | shared | " + " | ".join(REGIMES) + " |",
              "|---|---|---|---|" + "---|" * len(REGIMES)]
    for p in pairs:
        cells = []
        for r in REGIMES:
            s = p["regimes"][r]
            cells.append("-" if not s["points"] else
                         f"{s['points'] - s['mmq_agree']}/{s['points']}; "
                         f"mmq {s['mmq_a']}+{s['mmq_b']}; route {s['points'] - s['route_agree']}")
        d = "-" if p["distance"] is None else f"{p['distance']:.2f}"
        lines.append(f"| {p['generation']} | {p['a']} / {p['b']} | {d} | {p['problems'][2]} | "
                     + " | ".join(cells) + " |")
    lines += ["", "## Gate label regret on paired prefill timings", "",
              "| generation | label of | applied on | label | wrong/rows | max | mean |",
              "|---|---|---|---|---|---|---|"]
    for g in gates:
        lines.append(f"| {g['generation']} | {g['label_of']} | {g['applied_on']} | "
                     f"{'mmq' if g['label_pays'] else 'w4a8'} | {g['wrong']}/{g['rows']} | "
                     f"{100 * g['max_regret']:.1f} % | {100 * g['mean_regret']:.1f} % |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json", action="store_true", help="print the data as JSON, not markdown")
    args = ap.parse_args(argv)
    root = Path(args.root)
    gen_rows = generations(root / "src/core/generation.h")
    facts = json.loads((root / "src/core/data/gpu_facts.json").read_text(encoding="utf-8"))
    cards = load_cards(root / "routes", gen_rows, facts)
    pairs = table_audit(cards)
    paired = json.loads((root / "tools/bench/data/mmq_paired_prefill.json").read_text(encoding="utf-8"))
    measured = measured_boards(root / "src/core/measured_boards.h")
    lab = {v: cards[k] for k, v in PAIRED_CARDS.items() if k in cards}
    labels = gate_labels(measured, lab)
    gates = gate_audit(paired, labels, {v: cards[k]["generation"] for k, v in PAIRED_CARDS.items()
                                        if k in cards})
    if args.json:
        json.dump({"pairs": pairs, "gates": gates, "labels": labels}, sys.stdout, indent=1)
    else:
        sys.stdout.write(report(pairs, gates, cards))
    return 0


if __name__ == "__main__":
    sys.exit(main())
