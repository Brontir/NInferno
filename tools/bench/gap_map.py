#!/usr/bin/env python3
"""Gap map (#619 step 1): where measured kernel time sits against a counted law.

Host only, standard library only, no GPU. For every measured time the repository already commits
it computes the time two counted laws allow and prints the ratio (the gap):

  DRAM law    bytes the call must move / board bandwidth. Bytes: weight elements x bits per weight
              (docs/maintainer/tensor-formats.md) plus the activations in and out. Bandwidth and
              SM count: src/core/data/gpu_facts.json (the board's published facts).
  tensor law  2 x n x k x T flops / (SMs x boost clock x flops per SM-clock). The per-SM rate is
              the generation row of src/core/generation.h, which the cost model prices with
              (a lent, analytic rate; the issue #619 records that the gb20x bf16 value is about
              twice what mma.sync reaches, so a gb20x tensor gap is a lower bound on the truth).

The law time is the larger of the two; "binds" names it. A row whose law needs a number the
repository does not carry is kept and marked "no law" with the reason; no constant is invented.

Measured sources (all committed, see docs/maintainer/gap-map.md):
  routes/*.json           entries with evidence_us: [T, default us, shipped-plan us] per linear.
  tests/fixtures/choice   decode_measured.json (tokens a second per stream, whole decode step)
                          and prefill_measured.json (prompt tokens a second, whole prefill).

Usage: python tools/bench/gap_map.py [--repo ROOT] [--md out.md] [--json out.json] [--top N]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "ninferno-gap-map/1"

# Route-table spelling (src/ops/route_table.cpp parse_qtype) -> canonical format name of
# docs/maintainer/tensor-formats.md. Names only; every number comes from the doc.
ROUTE_FORMATS = {
    "bf16": "bf16", "fp8": "fp8_e4m3fn_row_bf16", "nvfp4": "nvfp4", "mxfp4": "mxfp4",
    "fp8r32": "fp8_e4m3fn_row_f32", "fp8b128": "fp8_e4m3fn_b128_f32",
    "q4": "q4_g64_fp16", "q5": "q5_g64_fp16", "q6": "q6_g64_fp16", "q8": "q8_g32_fp16",
    "u4zp": "u4_g128_zp_fp16",
    "q4k": "ggml_q4_k", "q5k": "ggml_q5_k", "q6k": "ggml_q6_k", "q3k": "ggml_q3_k",
    "q2k": "ggml_q2_k", "iq4xs": "ggml_iq4_xs", "iq3xxs": "ggml_iq3_xxs", "iq3s": "ggml_iq3_s",
    "q40": "ggml_q4_0", "q41": "ggml_q4_1", "q50": "ggml_q5_0", "q51": "ggml_q5_1",
    "iq4nl": "ggml_iq4_nl", "iq2xxs": "ggml_iq2_xxs", "iq2xs": "ggml_iq2_xs",
    "iq2s": "ggml_iq2_s", "iq1s": "ggml_iq1_s", "iq1m": "ggml_iq1_m", "tq10": "ggml_tq1_0",
    "tq20": "ggml_tq2_0", "q10": "ggml_q1_0", "q20": "ggml_q2_0",
}

# Activation-in bytes per element by compute mode (a16: bf16, a8: int8; per-token scales ignored);
# the output is bf16.
# a4 (e2m1 plus block scales) is left out on purpose: its bytes are not counted here.
ACT_BYTES = {"a16": 2.0, "a8": 1.0}

# T bands of the issue's regimes.
BANDS = ((1, 1), (2, 4), (5, 8), (9, 16), (17, 64), (65, 256), (257, 1 << 30))


def band_of(t: int) -> str:
    for lo, hi in BANDS:
        if lo <= t <= hi:
            return f"T{lo}" if lo == hi else (f"T{lo}+" if hi >= 1 << 30 else f"T{lo}-{hi}")
    return "T?"


# --------------------------------------------------------------------------- counted facts

def load_format_bits(doc_text: str) -> dict[str, float]:
    """Bits per weight by canonical format name, read from the tables of tensor-formats.md."""
    bits: dict[str, float] = {"bf16": 16.0}
    for m in re.finditer(r"\|\s*`(ggml_\w+)`\s*\|\s*(\d+) K values, (\d+) bytes", doc_text):
        bits[m.group(1)] = int(m.group(3)) * 8.0 / int(m.group(2))
    for m in re.finditer(r"\|\s*`(q\d_g\d+_fp16)`\s*\|[^\n]*?\|\s*([\d.]+)\s*\|\s*$", doc_text,
                         re.M):
        bits[m.group(1)] = float(m.group(2))
    # E2M1 codes with one scale word per group: 4 + scale bits / group.
    for m in re.finditer(r"\|\s*`(nvfp4|mxfp4)`\s*\|\s*E2M1, 4 bits/weight\s*\|\s*(\d+)\s*\|"
                         r"\s*one (E4M3FN word|E8M0 exponent)/group", doc_text):
        bits[m.group(1)] = 4.0 + 8.0 / int(m.group(2))
    # Row- and block-scaled FP8: 8 bits plus a scale per row / block (the scale is not counted:
    # one word per row of K values or per 16K elements, under 0.4 % of the bytes).
    for m in re.finditer(r"\|\s*`(fp8_e4m3fn_\w+)`\s*\|\s*E4M3FN, 8 bits/weight", doc_text):
        bits[m.group(1)] = 8.0
    return bits


def load_generations(header_text: str) -> list[dict]:
    """The generation rows of src/core/generation.h: {major, lo, hi, name, bf16, s8}."""
    rows = []
    for m in re.finditer(r"\{(\d+),\s*(\d+),\s*(\d+),\s*\"(\w+)\",\s*([\d.]+),\s*([\d.]+),",
                         header_text):
        rows.append({"major": int(m.group(1)), "lo": int(m.group(2)), "hi": int(m.group(3)),
                     "name": m.group(4), "bf16": float(m.group(5)), "s8": float(m.group(6))})
    return rows


def generation_of(rows: list[dict], cc: int) -> dict | None:
    major, minor = divmod(cc, 10)
    for r in rows:
        if r["major"] == major and r["lo"] <= minor <= r["hi"]:
            return r
    return None


def board_from_facts(facts: dict, chip: str, sms: int, dram_gbps: float | None) -> dict | None:
    """The gpu_facts row of a chip with that SM count; ties go to the closest bandwidth."""
    best, best_err = None, None
    for g in facts["gpus"]:
        if g["chip"] != chip or g["sms"] != sms or g.get("target") is False:
            continue
        bw = g["gbps"] * g["bus_bits"] / 8.0
        err = 0.0 if dram_gbps is None else abs(bw - dram_gbps) / dram_gbps
        if err <= 0.02 and (best_err is None or err < best_err):
            best, best_err = g, err
    if best is None:
        return None
    return {"name": best["name"], "cc": best["cc"], "sms": best["sms"],
            "boost_mhz": best["boost_mhz"], "dram_gbs": best["gbps"] * best["bus_bits"] / 8.0}


# --------------------------------------------------------------------------- laws

def tensor_rate(gen: dict | None, board: dict, compute: str) -> tuple[float | None, str]:
    """Dense flops a second for the mode, or (None, why not)."""
    if gen is None:
        return None, "no generation row"
    per_sm = gen["bf16"] if compute == "a16" else gen["s8"] if compute == "a8" else 0.0
    if per_sm <= 0:
        return None, f"no {compute} tensor rate for {gen['name']}"
    return board["sms"] * board["boost_mhz"] * 1e6 * per_sm, ""


def law_linear(board: dict, gen: dict | None, bits: float | None, compute: str, n: int, k: int,
               t: int, sets: int = 1) -> dict:
    """Law times (us) of one n x k linear on t tokens. n = output rows; a SwiGLU call reads `sets`
    = 2 weight rows per output row (gate and up, generic_tune.cpp: plane_rows = 2 x rows)."""
    if bits is None:
        return {"law": None, "why": "format bits per weight not in the format tables"}
    if compute not in ACT_BYTES:
        return {"law": None, "why": f"activation bytes of {compute} not counted"}
    nbytes = sets * n * k * bits / 8.0 + t * k * ACT_BYTES[compute] + t * n * 2.0
    dram_us = nbytes / (board["dram_gbs"] * 1e9) * 1e6
    rate, why = tensor_rate(gen, board, compute)
    tensor_us = None if rate is None else 2.0 * sets * n * k * t / rate * 1e6
    return finish(dram_us, tensor_us, nbytes, why)


def finish(dram_us: float, tensor_us: float | None, nbytes: float, why: str = "") -> dict:
    law = dram_us if tensor_us is None else max(dram_us, tensor_us)
    binds = "dram" if tensor_us is None or dram_us >= tensor_us else "tensor"
    out = {"law": law, "dram_us": dram_us, "tensor_us": tensor_us, "binds": binds,
           "bytes": nbytes}
    if tensor_us is None:
        out["note"] = f"tensor law missing ({why}); DRAM only"
    return out


def make_row(source: str, card: str, fmt: str, shape: str, t: int, measured_us: float,
             law: dict, **extra) -> dict:
    row = {"source": source, "card": card, "format": fmt, "shape": shape, "t": t,
           "band": band_of(t), "measured_us": measured_us}
    row.update(extra)
    if law.get("law") is None:
        row.update(law_us=None, binds="no law", gap=None, why=law["why"])
    else:
        row.update(law_us=law["law"], dram_us=law["dram_us"], tensor_us=law["tensor_us"],
                   binds=law["binds"], gap=measured_us / law["law"], bytes=law["bytes"])
        if "note" in law:
            row["note"] = law["note"]
    return row


# --------------------------------------------------------------------------- measured sources

def rows_from_route_table(doc: dict, name: str, facts: dict, gens: list[dict],
                          bits_by_format: dict[str, float], seen: set | None = None) -> list[dict]:
    """Rows of one route table. An entry's evidence belongs to the board it was measured on:
    tuned_on "class:NNsm" names it (a table that borrows another board's entries keeps their
    evidence), otherwise the table's own board. Identical evidence copied between tables (a
    board that borrows another's entries) is counted once, for the first table the caller passes
    (`seen`, shared by the caller; main() goes shortest table name first)."""
    gpu = doc.get("gpu", {})
    board_in = gpu.get("board", {})
    chip = str(gpu.get("class", "")).split(":")[0]
    own = board_from_facts(facts, chip, board_in.get("sms", 0), board_in.get("dram_gbps"))
    own_name = gpu.get("name") or name
    seen = set() if seen is None else seen
    rows: list[dict] = []
    for e in doc.get("entries", []):
        ev = e.get("evidence_us")
        if not ev:
            continue
        board, card, why = own, own_name, f"no gpu_facts row for {chip} with {board_in.get('sms')} SMs"
        m = re.fullmatch(r"(\w+):(\d+)sm", str(e.get("tuned_on", "")))
        if m:
            board = board_from_facts(facts, m.group(1), int(m.group(2)), None)
            card = board["name"] if board else e["tuned_on"]
            why = f"no gpu_facts row for {e['tuned_on']}"
        fmt = e.get("format", "?")
        n, k, compute = e.get("n"), e.get("k"), e.get("compute", "a16")
        key = (fmt, n, k, compute, e.get("epilogue"), json.dumps(ev))
        if key in seen:
            continue
        seen.add(key)
        shape = f"{n}x{k} {compute} {e.get('epilogue', '')}".strip()
        for point in ev:
            t, shipped = int(point[0]), float(point[-1])
            if board is None:
                law = {"law": None, "why": why}
            else:
                law = law_linear(board, generation_of(gens, board["cc"]),
                                 bits_by_format.get(ROUTE_FORMATS.get(fmt)), compute, int(n), int(k), t,
                                 2 if e.get("epilogue") == "swiglu" else 1)
            rows.append(make_row(f"routes/{name}", card, fmt, shape, t, shipped, law, kind="linear"))
    moe = len(doc.get("moe", []))
    if moe:
        # An MoE layer needs the expert gating, the distinct experts a batch touches and the down
        # format; the route table carries none of them as a count, so it gets no law (step 11).
        rows.append({"source": f"routes/{name}", "card": own_name, "kind": "linear", "format": "moe",
                     "shape": f"{moe} layers", "t": 0, "band": "moe", "measured_us": None,
                     "law_us": None, "binds": "no law", "gap": None,
                     "why": "MoE expert-call law not counted yet (#619 step 11)"})
    return rows


def card_board(c: dict) -> dict:
    return {"sms": c["sms"], "boost_mhz": c["clock_hz"] / 1e6, "dram_gbs": c["dram_bytes_per_s"] / 1e9,
            "cc": c["major"] * 10 + c["minor"]}


def rows_from_decode(doc: dict, gens: list[dict], short_ctx: int = 600) -> list[dict]:
    rows = []
    for r in doc["rows"]:
        m, c = doc["models"][r["model"]], doc["cards"][r["card"]]
        q = m["quants"].get(r["quant"])
        streams, ctx = r["concurrency"], r["context"]
        step_us = 1e6 / r["decode"]  # per-stream rate: one step serves every stream
        label = f"{r['model']} {r['quant']} ctx{ctx}"
        base = dict(kind="decode", model=r["model"], quant=r["quant"])
        if q is None or m["experts"]:
            why = "MoE: distinct experts a step touches not counted" if m["experts"] else "quant not in the fixture"
            rows.append(dict(make_row("decode_measured", r["card"], r["quant"], label, streams, step_us,
                                      {"law": None, "why": why}), **base))
            continue
        if m["sliding_attention_width"] and ctx > short_ctx:
            rows.append(dict(make_row("decode_measured", r["card"], r["quant"], label, streams, step_us,
                                      {"law": None, "why": "sliding-window KV bytes not counted at long context"}), **base))
            continue
        kv = m["kv_bytes_per_token"].get(r["kv"])
        state = m["state_bytes"].get(r["state"], 0) if m.get("recurrent") else 0
        if kv is None:
            rows.append(dict(make_row("decode_measured", r["card"], r["quant"], label, streams, step_us,
                                      {"law": None, "why": f"KV dtype {r['kv']} bytes not in the fixture"}), **base))
            continue
        # Weights are read once a step whatever the streams; the embedding table is a lookup.
        nbytes = (q["weight_bytes"] - q["input_table_bytes"]) + streams * (kv * ctx + 2 * state)
        dram_us = nbytes / c["dram_bytes_per_s"] * 1e6
        rows.append(dict(make_row("decode_measured", r["card"], r["quant"], label, streams, step_us,
                                  finish(dram_us, None, nbytes, "decode step is bytes-bound")), **base))
    return rows


def rows_from_prefill(doc: dict, gens: list[dict]) -> list[dict]:
    rows = []
    for r in doc["rows"]:
        m, c = doc["models"][r["model"]], doc["cards"][r["card"]]
        board = card_board(c)
        gen = generation_of(gens, board["cc"])
        us = r["prompt"] / r["prefill"] * 1e6
        label = f"{r['model']} {r['quant']} prompt{r['prompt']}"
        base = dict(kind="prefill", model=r["model"], quant=r["quant"])
        q = m["quants"].get(r["quant"])
        if q is None or m["experts"]:
            why = "MoE: experts a prompt touches not counted" if m["experts"] else "quant not in the fixture"
            rows.append(dict(make_row("prefill_measured", r["card"], r["quant"], label, r["prompt"], us,
                                      {"law": None, "why": why}), **base))
            continue
        flops = 2.0 * (m["active_params"] - m["input_table_params"]) * r["prompt"]
        # Best rate any activation mode can reach on the generation: int8 where it has one.
        mode = "a8" if gen and gen["s8"] > 0 else "a16"
        rate, why = tensor_rate(gen, board, mode)
        nbytes = q["weight_bytes"] - q["input_table_bytes"]
        dram_us = nbytes / c["dram_bytes_per_s"] * 1e6
        tensor_us = None if rate is None else flops / rate * 1e6
        law = finish(dram_us, tensor_us, nbytes, why)
        law["note"] = f"tensor law at the {mode} rate; attention and norms not counted"
        rows.append(dict(make_row("prefill_measured", r["card"], r["quant"], label, r["prompt"], us, law),
                         **base))
    return rows


# --------------------------------------------------------------------------- aggregation

def aggregate(rows: list[dict], min_law_us: float) -> list[dict]:
    """Per card x format x T band (decode/prefill whole-model rows by card x quant x kind)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["card"], r["kind"], r["format"], r["band"] if r["kind"] == "linear" else "")
        groups[key].append(r)
    out = []
    for (card, kind, fmt, band), g in sorted(groups.items()):
        scored = [r for r in g if r["gap"] is not None and r["law_us"] >= min_law_us]
        small = [r for r in g if r["gap"] is not None and r["law_us"] < min_law_us]
        nolaw = [r for r in g if r["gap"] is None]
        entry = {"card": card, "kind": kind, "format": fmt, "band": band or "-",
                 "rows": len(g), "scored": len(scored), "below_floor": len(small),
                 "no_law": len(nolaw)}
        if scored:
            gaps = [r["gap"] for r in scored]
            entry.update(median_gap=statistics.median(gaps), worst_gap=max(gaps),
                         binds=Counter(r["binds"] for r in scored).most_common(1)[0][0],
                         median_measured_us=statistics.median(r["measured_us"] for r in scored),
                         median_law_us=statistics.median(r["law_us"] for r in scored))
        if nolaw:
            entry["no_law_why"] = Counter(r["why"] for r in nolaw).most_common(1)[0][0]
        out.append(entry)
    return out


def top_gaps(rows: list[dict], n: int, min_law_us: float) -> dict[str, list[dict]]:
    """The n largest gaps of each kind (linear / decode / prefill): their laws differ."""
    out = {}
    for kind in ("linear", "decode", "prefill"):
        scored = [r for r in rows if r["kind"] == kind and r["gap"] is not None
                  and r["law_us"] >= min_law_us]
        out[kind] = sorted(scored, key=lambda r: -r["gap"])[:n]
    return out


# --------------------------------------------------------------------------- rendering

def fmt_us(x) -> str:
    return "-" if x is None else f"{x:.1f}" if x < 1000 else f"{x:.0f}"


def render_markdown(rep: dict) -> str:
    L = ["# Gap map", "",
         f"Measured against counted laws (`{SCHEMA}`). Gap = measured time / law time; law = max(DRAM, tensor). "
         f"Rows whose law time is under {rep['min_law_us']} us are counted but not ranked "
         "(a call that short is dominated by its launch, not a law).", ""]
    titles = {"linear": "one linear (route-table evidence)", "decode": "whole decode step",
              "prefill": "whole prefill"}
    for kind, rows in rep["top"].items():
        L += [f"## Largest gaps: {titles[kind]}", "",
              "| card | source | format | shape | T | measured us | law us | binds | gap |",
              "|---|---|---|---|---:|---:|---:|---|---:|"]
        for r in rows:
            L.append(f"| {r['card']} | {r['source']} | {r['format']} | {r['shape']} | {r['t']} | "
                     f"{fmt_us(r['measured_us'])} | {fmt_us(r['law_us'])} | {r['binds']} | {r['gap']:.2f} |")
        L.append("")
    L += ["## Per card, format and T band", "",
          "| card | kind | format | band | rows | scored | no law | median gap | worst gap | binds |",
          "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for a in rep["groups"]:
        if a["scored"]:
            L.append(f"| {a['card']} | {a['kind']} | {a['format']} | {a['band']} | {a['rows']} | {a['scored']} | "
                     f"{a['no_law']} | {a['median_gap']:.2f} | {a['worst_gap']:.2f} | {a['binds']} |")
        else:
            why = a.get("no_law_why", f"{a['below_floor']} rows under the floor")
            L.append(f"| {a['card']} | {a['kind']} | {a['format']} | {a['band']} | {a['rows']} | 0 | {a['no_law']} | "
                     f"- | - | no law: {why} |" if a["no_law"] else
                     f"| {a['card']} | {a['kind']} | {a['format']} | {a['band']} | {a['rows']} | 0 | 0 | - | - | "
                     f"under floor |")
    L += ["", "## No law", ""]
    for why, cnt in rep["no_law"]:
        L.append(f"- {cnt} rows: {why}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- driver

def build_report(repo: Path, min_law_us: float, top: int) -> dict:
    facts = json.loads((repo / "src/core/data/gpu_facts.json").read_text(encoding="utf-8"))
    gens = load_generations((repo / "src/core/generation.h").read_text(encoding="utf-8"))
    bits = load_format_bits((repo / "docs/maintainer/tensor-formats.md").read_text(encoding="utf-8"))
    rows: list[dict] = []
    seen: set = set()
    for p in sorted((repo / "routes").glob("*.json"), key=lambda q: (len(q.stem), q.stem)):
        rows += rows_from_route_table(json.loads(p.read_text(encoding="utf-8")), p.stem, facts, gens,
                                      bits, seen)
    fx = repo / "tests/fixtures/choice"
    if (fx / "decode_measured.json").exists():
        rows += rows_from_decode(json.loads((fx / "decode_measured.json").read_text(encoding="utf-8")), gens)
    if (fx / "prefill_measured.json").exists():
        rows += rows_from_prefill(json.loads((fx / "prefill_measured.json").read_text(encoding="utf-8")), gens)
    no_law = Counter(r["why"] for r in rows if r["gap"] is None).most_common()
    return {"schema": SCHEMA, "min_law_us": min_law_us, "rows": rows,
            "groups": aggregate(rows, min_law_us), "top": top_gaps(rows, top, min_law_us),
            "no_law": no_law}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--md", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-law-us", type=float, default=5.0,
                    help="do not rank rows whose law time is shorter (launch-dominated)")
    a = ap.parse_args(argv)
    rep = build_report(a.repo, a.min_law_us, a.top)
    md = render_markdown(rep)
    if a.md:
        a.md.write_text(md, encoding="utf-8")
    else:
        sys.stdout.write(md)
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
