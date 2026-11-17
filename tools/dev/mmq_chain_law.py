#!/usr/bin/env python3
"""The MMQ unit from counts and chip facts: the load phase plus the warp's chain (#613 PR-5).

  mmq_chain_law.py --card "RTX 4090" --arch 89 [--formats q4_K q5_K q6_K] [--out FILE]

The unit is the launch law's (tools/bench/mmq_launch_law.py derive): one 256-K iteration of a
128 x J tile on the `law` grid, S CTAs (one a SM, S the board's SMs) over floor(S / 2) weight tiles,
so that every weight tile is read from DRAM once and once more through the L2. ggml's matrix
kernel loads the iteration's tiles, synchronises, multiplies, synchronises: the load phase and the
compute phase do not overlap. So

  unit = max(B_dram / BW_dram, B_l2 / BW_l2) + L_dram / f + (stall + w_shared x L_shared) / f

  B_dram    floor(S / 2) x 128 rows x the format's bytes per 256 values (the weight tiles);
  B_l2      S x (128 x the format's bytes per 256 + J x 2 x 144): every CTA's weight tile and its
            activation tile (two block_q8_1_mmq of 144 bytes per column and 256 values);
  BW_dram   bus_bits / 8 x gbps (gpu_facts.json);
  BW_l2     the L2's bytes per SM clock per SM (the card's global-load microbench floor) x S x f;
  L_dram    isa_probe `dram_latency_w1` of the arch (clk), the load phase's last return;
  stall, w_shared   the counted iteration's stall sum and its scoreboard waits on LDS / LDSM
            (tests/fixtures/mmq_law/waits_sm_<arch>.csv, tools/dev/mmq_waits.py over the SASS);
  L_shared  isa_probe `shared_latency` of the arch (clk);
  f         the board's boost clock (gpu_facts.json), the clock the launch law states units at.

Written per format and J: the unit in microseconds and in clocks at f, and its two phases. No GPU;
no time is read. The predictions are frozen before the cards' launch-law runs are read.
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BYTES_PER_256 = {"q4_K": 144, "q5_K": 176, "q6_K": 210}
TYPE = {"q4_K": 12, "q5_K": 13, "q6_K": 14}
WIDTHS = (16, 32, 64, 80, 96, 128)


def isa_facts(arch):
    out = {}
    for line in (ROOT / f"src/core/data/isa_probe/isa_probe_{arch}.txt").read_text().splitlines():
        c = line.split()
        if len(c) >= 3 and c[0] == "fact":
            out[c[1]] = float(c[2])
    return out


def board(name):
    gpus = json.loads((ROOT / "src/core/data/gpu_facts.json").read_text())["gpus"]
    hits = [g for g in gpus if g["name"].endswith(name) and g.get("form", "desktop") == "desktop"]
    if len(hits) != 1:
        raise SystemExit(f"{name}: {len(hits)} desktop boards in gpu_facts.json")
    return hits[0]


def waits(arch):
    rows = {}
    with (ROOT / f"tests/fixtures/mmq_law/waits_sm_{arch}.csv").open() as f:
        for r in csv.DictReader(f):
            rows[(int(r["type"]), int(r["J"]))] = r
    return rows


def unit(g, facts, w, l2_bpc, fmt, J):
    S, f = g["sms"], g["boost_mhz"] * 1e6
    bw = g["bus_bits"] / 8 * g["gbps"] * 1e9
    b_dram = (S // 2) * 128 * BYTES_PER_256[fmt]
    b_l2 = S * (128 * BYTES_PER_256[fmt] + J * 2 * 144)
    load = max(b_dram / bw, b_l2 / (l2_bpc * S * f)) + facts["dram_latency_w1"] / f
    r = w[(TYPE[fmt], J)]
    chain = (int(r["stall"]) + int(r["w_shared"]) * facts["shared_latency"]) / f
    return load + chain, load, chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--l2-bytes-per-clk-per-sm", type=float, required=True)
    ap.add_argument("--formats", nargs="+", default=["q4_K", "q5_K", "q6_K"])
    ap.add_argument("--out")
    a = ap.parse_args()
    g, facts, w = board(a.card), isa_facts(a.arch), waits(a.arch)
    lines = [["card", "arch", "format", "J", "unit_us", "unit_clk", "load_us", "chain_us"]]
    for fmt in a.formats:
        for J in WIDTHS:
            u, load, chain = unit(g, facts, w, a.l2_bytes_per_clk_per_sm, fmt, J)
            lines.append([g["name"], a.arch, fmt, J, f"{u * 1e6:.3f}", f"{u * g['boost_mhz'] * 1e6:.0f}",
                          f"{load * 1e6:.3f}", f"{chain * 1e6:.3f}"])
    out = open(a.out, "w", newline="") if a.out else None
    wr = csv.writer(out or __import__("sys").stdout, lineterminator="\n")
    wr.writerows(lines)
    if out:
        out.close()


if __name__ == "__main__":
    main()
