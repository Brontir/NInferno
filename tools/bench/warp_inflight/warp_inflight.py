#!/usr/bin/env python3
"""Build and run warp_inflight on this card, or read saved runs, and give the per-SM DRAM rate as a
function of resident warps with the bytes a warp keeps in flight (b_w) and the saturated per-SM
rate (P) read from it: the facts the decode price's per-SM bound min(P, W x b_w / L0) needs
(#690, #735).

    warp_inflight.py run  --arch 89 [--nvcc nvcc] [--out warp_inflight_89.txt] [--idle-unchecked]
    warp_inflight.py rows --input warp_inflight_89.txt [--input ...] [--isa-table ...] [--out F]
    warp_inflight.py show --input warp_inflight_89.txt [--isa-table ...]

`run` compiles tools/bench/warp_inflight/warp_inflight.cu for sm_<arch> into a temporary directory
(nvcc -O3 -std=c++17 -arch=sm_<arch>), counts the 16-byte loads of each variant's kernel in the
build's SASS (cuobjdump), runs it (5 to 15 s, the GPU otherwise idle) and saves its output, headed
by the source's sha256, the build's -arch and those load counts, and followed by notes of the
compute processes on the probe's GPU before, during and after the run and of the clocks
nvidia-smi saw (isa_probe's notes, read by its own code): a run with another process on the GPU is
no row, and its output goes to <out>.refused. `rows` turns runs into JSON rows with their
provenance (source file, card, probe version, nvcc, driver, runtime, source hash) and the derived
b_w and P; `show` prints the summary. No fact is fitted here: b_w = rate(W) x L0 / W is Little's
law on the measured rate and the ISA table's measured L0, P is the largest measured rate.
"""

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SOURCE = os.path.join(HERE, "warp_inflight.cu")
DEFAULT_ISA_TABLE = os.path.join(ROOT, "src", "core", "data", "isa_facts.json")
SCHEMA = 1


def _isa_probe():
    """tools/bench/isa_probe/isa_probe.py: its source hash, idle check and nvidia-smi notes."""
    path = os.path.join(ROOT, "tools", "bench", "isa_probe", "isa_probe.py")
    spec = importlib.util.spec_from_file_location("isa_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IP = _isa_probe()

WARP_LOAD = 512  # one warp-wide 16-byte load: 32 lanes x 16 B
# The probe's variants (warp_inflight.cu kVariants): name -> (loads a fetch, ring). A chain keeps
# F loads a lane in flight, a ring 2F; the SASS of a chain's loop holds F 16-byte loads, a ring's
# 3F (the first fetch before the loop, two in it).
VARIANTS = {"chain1": (1, False), "chain2": (2, False), "chain4": (4, False),
            "ring1": (1, True), "ring2": (2, True), "ring4": (4, True)}
W_GRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
K_GRID = (1, 2, 4, 8)
MODES = ("free", "placed")
COLUMNS = ("mode", "k", "variant", "inflight", "w", "gbps", "gbps_lo", "gbps_hi", "ns_fetch",
           "sm_mhz", "mib")
HEADER = ("probe", "card", "cc", "sms", "l2_bytes", "mem_clock_khz", "max_warps_per_sm",
          "smem_per_sm", "driver", "runtime", "nvcc")
RUN_HEADER = ("source_sha256", "arch", "sass_ldg128")
COUNTS = ("cc", "sms", "l2_bytes", "max_warps_per_sm", "smem_per_sm", "driver", "runtime")
PROBE_LINE = re.compile(r'printf\("probe (warp_inflight \d+)\\n"\)')
PROBE_VALUE = re.compile(r"warp_inflight (\d+)")
NO_ROW = re.compile(r"placed (\w+) w(\d+): \d+ of \d+ CTAs fit an SM \(registers\), no row")
REFUSED_LINE = "refused"
REFUSED = ".refused"
# b_w's linear region: the rows from W = 1 up whose b_w stays within this share of b_w at W = 1
# (a reading of the measured table, not a fitted constant: it decides only the summary's
# `linear_to_w`, never a value).
LINEAR_SHARE = 0.9


def inflight_of(variant):
    f, ring = VARIANTS[variant]
    return 2 * f if ring else f


def sass_loads_of(variant):
    f, ring = VARIANTS[variant]
    return 3 * f if ring else f


def checked_in_probe():
    """The `probe` header value the checked-in warp_inflight.cu prints."""
    with open(SOURCE) as f:
        m = PROBE_LINE.search(f.read())
    if not m:
        raise ValueError("warp_inflight.cu prints no probe version")
    return m.group(1)


def source_sha256():
    return IP.source_sha256(SOURCE)


def expected_points(head, no_row):
    """Every (mode, k, variant, w) the probe measures on this card: the free rows up to 32 warps,
    the placed rows up to the SM's resident warps (two CTAs above 32), less the placed pairs the
    run noted the registers do not fit."""
    sms, max_w = int(head["sms"]), int(head["max_warps_per_sm"])
    points = set()
    for mode in MODES:
        for k in K_GRID:
            if k > sms:
                continue
            for v in VARIANTS:
                for w in W_GRID:
                    if w > max_w or (mode == "free" and w > 32):
                        continue
                    if mode == "placed" and (w > 32 and w % 2 or (v, w) in no_row):
                        continue
                    points.add((mode, k, v, w))
    return points


def _positive(text, what):
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{what} is no number: {text!r}") from None
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{what} is not a positive finite number: {text!r}")
    return value


def parse_run(text):
    """The header fields and the rows of one warp_inflight output (as `run` saves it); ValueError
    when a line is odd or unknown, the columns are not the parser's, a row is malformed, out of
    the grid, given twice or missing, the probe version or source hash is not the checked-in one,
    the build's -arch is not the card's, the SASS load counts are not the variants', the
    self-check failed, the run did not end, or the process notes do not show the GPU idle."""
    head, rows, notes, ended, checked, failures = {}, {}, [], False, False, []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        word = parts[0]
        if word == "row":
            if len(parts) != 1 + len(COLUMNS):
                raise ValueError(f"odd row line: {line!r}")
            r = dict(zip(COLUMNS, parts[1:]))
            if r["mode"] not in MODES:
                raise ValueError(f"unknown mode: {line!r}")
            if r["variant"] not in VARIANTS:
                raise ValueError(f"unknown variant: {line!r}")
            if not (r["k"].isdigit() and r["w"].isdigit() and r["inflight"].isdigit()):
                raise ValueError(f"odd row line: {line!r}")
            key = (r["mode"], int(r["k"]), r["variant"], int(r["w"]))
            if key in rows:
                raise ValueError(f"row given twice: {key}")
            if int(r["inflight"]) != inflight_of(r["variant"]):
                raise ValueError(f"{r['variant']} keeps {inflight_of(r['variant'])} loads a lane "
                                 f"in flight, not {r['inflight']}: {line!r}")
            row = {"mode": key[0], "k": key[1], "variant": key[2],
                   "inflight_loads": int(r["inflight"]),
                   "warp_bytes_in_flight": int(r["inflight"]) * WARP_LOAD, "w": key[3]}
            for c in COLUMNS[5:]:
                row[c] = _positive(r[c], f"{c} of {key}")
            if not row["gbps_lo"] <= row["gbps"] <= row["gbps_hi"]:
                raise ValueError(f"gbps is outside its SMs' range: {line!r}")
            rows[key] = row
        elif word == "columns":
            if tuple(parts[1:]) != COLUMNS:
                raise ValueError(f"the probe's columns are not the parser's: {line!r}")
            head["columns"] = True
        elif word == "note":
            notes.append(" ".join(parts[1:]))
        elif word == "selfcheck":
            checked = True
            if parts[1:2] != ["ok"]:
                failures.append(" ".join(parts[2:]))
        elif word == "end":
            ended = True
        elif word == REFUSED_LINE:
            raise ValueError(f"`run` refused this run: {' '.join(parts[1:])}")
        elif word in HEADER or word in RUN_HEADER:
            if word in head:
                raise ValueError(f"{word} line given twice")
            head[word] = " ".join(parts[1:])
        else:
            raise ValueError(f"unknown line: {line!r}")
    if not ended or not checked:
        raise ValueError(f"incomplete run: ended={ended}, selfcheck={checked}")
    if failures:
        raise ValueError("the probe's self-check failed: " + "; ".join(failures))
    for k in HEADER + RUN_HEADER + ("columns",):
        if not head.get(k):
            raise ValueError(f"no {k} line")
    for k in COUNTS:
        if not (head[k].isdigit() and int(head[k]) > 0):
            raise ValueError(f"{k} is no positive count: {head[k]!r}")
    if not head["mem_clock_khz"].isdigit():
        raise ValueError(f"mem_clock_khz is no count: {head['mem_clock_khz']!r}")
    if not PROBE_VALUE.fullmatch(head["probe"]):
        raise ValueError(f"odd probe line: {head['probe']!r}")
    # A run is of the checked-in probe: its version and its source's hash.
    if head["probe"] != checked_in_probe():
        raise ValueError(f"{head['probe']} is not the checked-in probe ({checked_in_probe()})")
    if head["source_sha256"] != source_sha256():
        raise ValueError(f"source_sha256 {head['source_sha256']} is not the checked-in "
                         f"warp_inflight.cu's ({source_sha256()}): a changed probe bumps its "
                         "version")
    own = "sm_" + IP.arch_key(int(head["cc"]), "")
    if head["arch"] != own:
        raise ValueError(f"built for {head['arch']}, not {own} (cc {head['cc']})")
    want = " ".join(f"{v}:{sass_loads_of(v)}" for v in VARIANTS)
    if head["sass_ldg128"] != want:
        raise ValueError(f"the build's 16-byte loads per kernel are {head['sass_ldg128']!r}, not "
                         f"the variants' {want!r}")
    max_w = int(head["max_warps_per_sm"])
    no_row = set()
    for n in notes:
        m = NO_ROW.fullmatch(n)
        if m:
            if m.group(1) not in VARIANTS:
                raise ValueError(f"odd note: {n!r}")
            no_row.add((m.group(1), int(m.group(2))))
    want_points = expected_points(head, no_row)
    extra = sorted(set(rows) - want_points)
    if extra:
        raise ValueError(f"rows outside the grid (max {max_w} warps, {head['sms']} SMs): {extra}")
    missing = sorted(want_points - set(rows))
    if missing:
        raise ValueError(f"rows missing: {missing}")
    refusal = IP.idle_refusal(notes)
    if refusal:
        raise ValueError(refusal)
    head["notes"] = notes
    return head, [rows[k] for k in sorted(rows, key=lambda k: (MODES.index(k[0]), k[1],
                                                              list(VARIANTS).index(k[2]), k[3]))]


def isa_l0(cc, table_path):
    """(L0 in ns, its source) from the ISA facts table: the architecture's
    dram_tlbwarm_latency_ns_w1, the L0 of #735's law; (None, why) without one."""
    try:
        with open(table_path) as f:
            table = json.load(f)
    except OSError as e:
        return None, f"no ISA table ({type(e).__name__})"
    key = IP.arch_key(cc, "")
    row = table.get("arches", {}).get(key)
    fact = (row or {}).get("facts", {}).get("dram_tlbwarm_latency_ns_w1")
    if not fact:
        return None, f"the ISA table has no sm_{key} row"
    return fact["value"], (f"isa_facts sm_{key} ({row['card']}, {row['source']}) "
                           "dram_tlbwarm_latency_ns_w1")


def derive(rows, l0_ns):
    """Each row's b_w (bytes a warp keeps in flight, by Little's law on the row's rate and L0)
    and the summary: per mode and variant at k = 1, b_w at W = 1 and the largest W up to which
    every row keeps LINEAR_SHARE of it; per mode at k = 1, P (the largest rate) in GB/s and in
    bytes per SM clock at that row's clock."""
    for r in rows:
        r["b_w_bytes"] = round(r["gbps"] * l0_ns / r["w"], 1) if l0_ns else None
    summary = {"variants": {}, "p_gbps": {}, "p_bytes_per_clk": {}}
    for mode in MODES:
        one = [r for r in rows if r["mode"] == mode and r["k"] == 1]
        if not one:
            continue
        top = max(one, key=lambda r: r["gbps"])
        summary["p_gbps"][mode] = top["gbps"]
        summary["p_bytes_per_clk"][mode] = round(top["gbps"] * 1e3 / top["sm_mhz"], 2)
        per = summary["variants"].setdefault(mode, {})
        for v in VARIANTS:
            vr = sorted((r for r in one if r["variant"] == v), key=lambda r: r["w"])
            if not vr or vr[0]["w"] != 1 or not l0_ns:
                continue
            b1, linear = vr[0]["b_w_bytes"], 1
            for r in vr:
                if r["b_w_bytes"] < LINEAR_SHARE * b1:
                    break
                linear = r["w"]
            per[v] = {"warp_bytes_in_flight": vr[0]["warp_bytes_in_flight"], "b_w_w1": b1,
                      "linear_to_w": linear}
    return summary


NOTE = ("Per-SM DRAM rate against resident warps W, written by `tools/bench/warp_inflight/"
        "warp_inflight.py rows` from `warp_inflight.py run` outputs (one per card): every lane "
        "streams 16-byte ld.global.cs loads from a range of its own, the warp keeping a known "
        "number of them in flight (chainF: F a lane; ringF: the decode kernels' two-slot ring, "
        "up to 2F); `free` is k CTAs of W <= 32 warps on k distinct SMs with the L1-largest "
        "carve-out, `placed` 1 or 2 CTAs on each of k chosen SMs held there by shared memory "
        "(shared-largest carve-out). gbps is bytes per ns of one SM (median over the k SMs, "
        "median of three launches); b_w_bytes = gbps x L0 / W with L0 the ISA table's "
        "dram_tlbwarm_latency_ns_w1 of the architecture; the summary's P is the largest rate at "
        "k = 1. Provenance Measured for every run, with its source file, card, probe version, "
        "nvcc, driver, runtime and the sha256 of the warp_inflight.cu it was built from.")


def run_of(head, rows, source, isa_table):
    l0, why = isa_l0(int(head["cc"]), isa_table)
    summary = derive(rows, l0)
    return {
        "provenance": "Measured", "source": source, "card": head["card"],
        "cc": int(head["cc"]), "sms": int(head["sms"]), "l2_bytes": int(head["l2_bytes"]),
        "max_warps_per_sm": int(head["max_warps_per_sm"]),
        "smem_per_sm": int(head["smem_per_sm"]), "mem_clock_khz": int(head["mem_clock_khz"]),
        "tool": {"probe": head["probe"], "nvcc": head["nvcc"], "driver": head["driver"],
                 "runtime": head["runtime"], "arch": head["arch"],
                 "source_sha256": head["source_sha256"], "sass_ldg128": head["sass_ldg128"]},
        "notes": head["notes"],
        "l0": {"value_ns": l0, "source": why},
        "rows": rows,
        "summary": summary,
    }


def table_of(paths, isa_table):
    runs = []
    for p in paths:
        with open(p) as f:
            text = f.read()
        try:
            head, rows = parse_run(text)
        except ValueError as e:
            raise ValueError(f"{os.path.basename(p)}: {e}") from None
        runs.append(run_of(head, rows, os.path.basename(p), isa_table))
    return {"schema": SCHEMA, "note": NOTE, "runs": runs}


def sass_loads(sass):
    """{variant: 16-byte global loads in its kernel} of `cuobjdump -sass` text, read with
    sass_counts' parser: LDG instructions whose opcode carries .128."""
    sc = IP._sass_counts()
    tag = {f"streamILi{f}ELb{int(ring)}E": v for v, (f, ring) in VARIANTS.items()}
    out = {}
    for name, body in sc.iter_functions(sass.splitlines(),
                                        lambda n: any(t in n for t in tag)):
        variant = next(v for t, v in tag.items() if t in name)
        if variant in out:
            raise ValueError(f"two kernels of {variant} in the build's SASS")
        out[variant] = sum(1 for row in body
                           if sc.base_opcode(row[2]) == "LDG" and ".128" in row[2])
    missing = [v for v in VARIANTS if v not in out]
    if missing:
        raise ValueError(f"no kernel of {missing} in the build's SASS")
    return out


def cmd_run(a):
    head = [f"source_sha256 {source_sha256()}", f"arch sm_{a.arch}"]
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "warp_inflight.exe" if os.name == "nt" else "warp_inflight")
        cmd = [a.nvcc, "-O3", "-std=c++17", f"-arch=sm_{a.arch}", "-o", exe, SOURCE]
        print("+", " ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True)
        cuobjdump = a.cuobjdump
        if cuobjdump is None:
            import shutil
            nvcc = shutil.which(a.nvcc) or a.nvcc
            near = os.path.join(os.path.dirname(nvcc),
                                "cuobjdump" + (".exe" if os.name == "nt" else ""))
            cuobjdump = near if os.path.exists(near) else "cuobjdump"
        sass = subprocess.run([cuobjdump, "-sass", exe], stdout=subprocess.PIPE, text=True,
                              check=True).stdout
        try:
            loads = sass_loads(sass)
        except ValueError as e:
            print(f"warp_inflight: {e}; no run", file=sys.stderr)
            return 3
        head.append("sass_ldg128 " + " ".join(f"{v}:{loads[v]}" for v in VARIANTS))
        before = IP._compute_apps()

        def sampler(query):
            try:
                return subprocess.Popen(
                    ["nvidia-smi", query, "--format=csv,noheader,nounits", "-lms", "200"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            except OSError:
                return None

        def rows_of(proc, fields):
            if proc is None:
                return None
            proc.terminate()
            out, _ = proc.communicate()
            return [[x.strip() for x in line.split(",")] for line in out.splitlines()
                    if line.count(",") == fields - 1]

        clocks = sampler("--query-gpu=pci.bus_id,pstate,clocks.sm,clocks.mem")
        apps = sampler("--query-compute-apps=timestamp,gpu_bus_id,pid")
        probe = subprocess.Popen([exe], stdout=subprocess.PIPE, text=True)
        stdout, _ = probe.communicate()
        samples = rows_of(clocks, 4) or []
        during = rows_of(apps, 3)
        after = IP._compute_apps()
    m = re.search(r"^note device 0 is the GPU at PCI (\S+)$", stdout, re.M)
    bus = IP._bus(m.group(1)) if m else None
    tail = [IP._apps_note("before", before, bus), IP._apps_note("after", after, bus),
            IP._during_note(during, bus, str(probe.pid))]
    mine = [s for s in samples if bus is not None and IP._bus(s[0]) == bus]
    if mine:
        def span(col):
            vals = sorted(int(s[col]) for s in mine if s[col].isdigit())
            return f"{vals[0]}..{vals[-1]}" if vals else "not read"
        tail.append(f"note nvidia-smi over the run ({len(mine)} samples): memory clock "
                    f"{span(3)} MHz, SM clock {span(2)} MHz, P-state "
                    f"{'/'.join(sorted({s[1] for s in mine}))}")
    else:
        tail.append("note nvidia-smi over the run: no clock sample of the probe's GPU")
    if a.idle_unchecked and any(
            n and (n.group(3).startswith("not read") or
                   (n.group(1).startswith("any GPU") and n.group(3).startswith("0")))
            for n in (IP.PROCESS_NOTE.fullmatch(t[5:]) for t in tail)):
        tail.append("note " + IP.WAIVED)
    body = stdout if stdout.endswith("\n") or not stdout else stdout + "\n"
    out = "\n".join(head) + "\n" + body + "\n".join(tail) + "\n"
    refusal = None
    if probe.returncode != 0:
        refusal = f"the probe exited {probe.returncode}"
    else:
        try:
            parse_run(out)
        except ValueError as e:
            refusal = str(e)
    path = a.out or f"warp_inflight_{a.arch}.txt"
    if refusal:
        path += REFUSED
        out += f"{REFUSED_LINE} {' '.join(refusal.split())}\n"
    with open(path, "w", newline="\n") as f:
        f.write(out)
    sys.stdout.write(out)
    if refusal:
        print(f"warp_inflight: {refusal}; wrote {path}, which is no row", file=sys.stderr)
        return probe.returncode or 3
    print(f"warp_inflight: wrote {path}")
    return 0


def cmd_rows(a):
    try:
        table = table_of(a.input, a.isa_table)
    except (OSError, ValueError) as e:
        print(f"warp_inflight: {e}", file=sys.stderr)
        return 1
    text = json.dumps(table, indent=2, allow_nan=False) + "\n"
    if a.out:
        with open(a.out, "w", newline="\n") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


def cmd_show(a):
    try:
        table = table_of(a.input, a.isa_table)
    except (OSError, ValueError) as e:
        print(f"warp_inflight: {e}", file=sys.stderr)
        return 1
    for run in table["runs"]:
        s = run["summary"]
        print(f"{run['card']} ({run['tool']['arch']}, {run['source']}); L0 "
              f"{run['l0']['value_ns']} ns from {run['l0']['source']}")
        for mode, p in s["p_gbps"].items():
            print(f"  {mode}: P {p:.2f} GB/s an SM ({s['p_bytes_per_clk'][mode]} B/clk)")
            for v, d in s["variants"].get(mode, {}).items():
                print(f"    {v:7s} in flight {d['warp_bytes_in_flight']:5d} B a warp: b_w at W=1 "
                      f"{d['b_w_w1']:7.1f} B, linear to W={d['linear_to_w']}")
        for r in run["rows"]:
            if r["k"] == 1:
                print(f"  {r['mode']:6s} {r['variant']:7s} W={r['w']:2d} {r['gbps']:8.3f} GB/s "
                      f"ns_fetch {r['ns_fetch']:7.1f} b_w {r['b_w_bytes']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arch", required=True, help="sm_<arch> to compile for (70, 75, 86, 89, 120a)")
    r.add_argument("--nvcc", default="nvcc")
    r.add_argument("--cuobjdump", help="default: the one beside --nvcc, else from PATH")
    r.add_argument("--out", help="the run's output; a refused run's goes to <out>" + REFUSED)
    r.add_argument("--idle-unchecked", action="store_true",
                   help="accept a run whose compute-process list nvidia-smi could not give")
    r.set_defaults(fn=cmd_run)
    for name, fn in (("rows", cmd_rows), ("show", cmd_show)):
        s = sub.add_parser(name)
        s.add_argument("--input", action="append", required=True)
        s.add_argument("--isa-table", default=DEFAULT_ISA_TABLE)
        if name == "rows":
            s.add_argument("--out")
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
