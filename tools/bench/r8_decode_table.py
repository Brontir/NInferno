#!/usr/bin/env python3
"""Write src/ops/generic/r8_decode_table.h from ninferno_r8_schedule_bench sweeps.

Each argument is GPU=FILE[,FILE...]: the bench output of repeated sweeps on one GPU, GPU its class id
as the bench's "device" line prints it (chip and SM count, "gb202:170sm"). Every file must be a
complete sweep (its "done" line) on that GPU. For every problem the median time of each R8 schedule
over the sweeps decides. A problem gets an entry when the fastest schedule is not the shape-class
rules' choice (the bench's "rules" mark: r8_rules_route, without any table entry), beats it by at
least --min-gain in the median and is faster in every sweep, and every schedule gave the same bytes
(same=1) in every sweep. A sweep from a bench before the "rules" mark (5655273d and earlier, builds
without a table) is read by its "picked" mark, which then is the rules' choice.

The add epilogue has no entries of its own: the engine looks an add call up as a store call. An add
row stands for its store problem only where the sweep has no store row of that problem.

--merge HEADER keeps HEADER's entries (another format family's sweeps, another GPU's) for every
problem the given sweeps of that GPU did not measure; a measured problem takes the new verdict.

Each table names the board its sweeps ran on by PCI id, which the bench's "board" line prints
(`pci 0x2684`), and the header reads its facts from the registry
(`measured::board_of_pci(0x2684)`, core/measured_boards.h): no board's facts are written here
(#612 step 9). A sweep from a bench that printed no id takes the --merge header's for its GPU; an
id src/core/data/gpu_facts.json lacks stops the tool (add the board's row first).

    python3 tools/bench/r8_decode_table.py --out src/ops/generic/r8_decode_table.h \
        ad102:128sm=a1.txt,a2.txt,a3.txt gb202:170sm=b1.txt,b2.txt,b3.txt
    python3 tools/bench/r8_decode_table.py --merge src/ops/generic/r8_decode_table.h \
        --out src/ops/generic/r8_decode_table.h gb202:170sm=k1.txt,k2.txt,k3.txt
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROW = re.compile(
    r"^r8 (\S+) (\S+) n=(\d+) k=(\d+) r8\.w(\d+)r(\d+) ([\d.]+) us [\d.]+ GB/s same=(\d)"
    r"( picked)?( rules)?\s*$"
)
DEVICE = re.compile(r"^device .* class (\S+), ")
QTYPES = {
    "iq3xxs": "GGML_IQ3_XXS",
    "iq3s": "GGML_IQ3_S",
    "iq2xxs": "GGML_IQ2_XXS",
    "iq2xs": "GGML_IQ2_XS",
    "iq2s": "GGML_IQ2_S",
    "iq4xs": "GGML_IQ4_XS",
    "iq4nl": "GGML_IQ4_NL",
    "q4k": "GGML_Q4_K",
    "q5k": "GGML_Q5_K",
    "q6k": "GGML_Q6_K",
    "q3k": "GGML_Q3_K",
    "q40": "GGML_Q4_0",
    "q41": "GGML_Q4_1",
    "q50": "GGML_Q5_0",
    "q51": "GGML_Q5_1",
    "mxfp4": "MXFP4",
    "q8": "Q8_G32_FP16",
}
# clang-format aligns the comments: any spacing before "//".
ENTRY = re.compile(r"^\s*\{QType::(\w+), Epilogue::(\w+), (\d+), (\d+), (\d+), (\d+)\},\s*// (.*)$")
TABLE = re.compile(r'^\s*\{"(\S+)", (?:measured::board_of_pci\((0x[0-9A-Fa-f]+)\), )?(\w+)\},\s*$')
BOARD = re.compile(r"^board cc (\d+) sms (\d+) l2_mib ([\d.]+) dram_gbps ([\d.]+)"
                   r"(?: pci (0x[0-9A-Fa-f]+))?")
GPU_FACTS = Path(__file__).resolve().parents[2] / "src" / "core" / "data" / "gpu_facts.json"
EPILOGUES = {"store": "Store", "swiglu": "SwiGlu"}


# The facts every sweep's board line printed, by PCI id (each one: a --merge of several sweeps
# under one id may mix the stock board with a variant on other memory): each is checked against
# the id's registry row.
printed = {}


def pci_text(pci_id):
    """A PCI product id as the header spells it: "0x" and four upper-case hex digits."""
    if not re.fullmatch(r"0[xX][0-9A-Fa-f]{4}", pci_id):
        sys.exit(f"{pci_id!r} is not a PCI product id (0x and four hex digits)")
    return "0x%04X" % int(pci_id, 16)


def registry_rows():
    """{PCI id: (cc, SMs, L2 MiB, DRAM GB/s)} of src/core/data/gpu_facts.json, the rows
    measured::board_of_pci reads."""
    rows = json.loads(GPU_FACTS.read_text(encoding="utf-8"))["gpus"]
    return {pci_text(r["id"]): (r["cc"], r["sms"], float(r.get("l2_mib", 0)),
                                r["bus_bits"] * r["gbps"] / 8.0) for r in rows}


def read_run(path, gpu):
    times, picked, rules = {}, {}, {}
    device, done, board = None, False, None
    for line in open(path, errors="replace", encoding="utf-8"):
        m = BOARD.match(line)
        if m:
            # The board's PCI id; a bench before it printed one gives None. The facts it printed
            # are not copied (the header reads them from the registry by id), only checked
            # against the id's row (main).
            if m.group(5):
                cc, sms, l2, bw = m.groups()[:4]
                printed.setdefault(pci_text(m.group(5)), set()).add(
                    (int(cc), int(sms), float(l2), float(bw)))
            board = pci_text(m.group(5)) if m.group(5) else None
            continue
        if line.strip() == "done":
            done = True
            continue
        m = DEVICE.match(line)
        if m:
            device = m.group(1)
            continue
        m = ROW.match(line)
        if not m:
            continue
        fmt, epi, n, k, warps, rows, us, same, pick, rule = m.groups()
        if same != "1":
            sys.exit(f"{path}: schedules differ in bytes: {line.strip()}")
        key = (fmt, epi, int(n), int(k))
        schedule = (int(warps), int(rows))
        times.setdefault(key, {})[schedule] = float(us)
        if pick:
            picked[key] = schedule
        if rule:
            rules[key] = schedule
    if device != gpu:
        sys.exit(f"{path}: device class {device}, expected {gpu}")
    if not done:
        sys.exit(f"{path}: no 'done' line (an incomplete sweep)")
    base = rules if rules else picked
    # An add row stands for its store problem where the sweep has no store row of it.
    run, run_base = {}, {}
    for key in sorted(times, key=lambda k: k[1] == "add"):
        fmt, epi, n, k = key
        target = (fmt, "store" if epi == "add" else epi, n, k)
        if target in run or key not in base:
            continue
        run[target], run_base[target] = times[key], base[key]
    return run, run_base, board


def read_header(path):
    """The entries of an r8_decode_table.h per GPU: {gpu: {(qtype, epilogue, n, k): line}}."""
    by_name, gpus, name = {}, {}, None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^inline constexpr R8DecodeEntry (\w+)\[\] = \{", line)
        if m:
            name = m.group(1)
            by_name[name] = {}
            continue
        if line.startswith("};"):
            name = None
            continue
        m = ENTRY.match(line)
        if m and name:
            key = (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))
            by_name[name][key] = line.strip()
            continue
        m = TABLE.match(line)
        if m:
            gpus[m.group(1)] = (m.group(3), pci_text(m.group(2)) if m.group(2) else None)
    return {gpu: (by_name.get(name, {}), board) for gpu, (name, board) in gpus.items()}


def measured_keys(paths, gpu):
    """Every (qtype, epilogue, n, k) the sweeps timed, in the header's spelling."""
    keys = set()
    for path in paths:
        run, _, _ = read_run(path, gpu)
        for fmt, epi, n, k in run:
            if fmt in QTYPES:
                keys.add((QTYPES[fmt], EPILOGUES[epi], n, k))
    return keys


def entries_for(gpu, paths, min_gain):
    runs, base = [], {}
    for path in paths:
        run, run_base, _ = read_run(path, gpu)
        runs.append(run)
        for key, value in run_base.items():
            if base.setdefault(key, value) != value:
                sys.exit(f"{path}: another rules choice for {key}")
    out = []
    for key in sorted(base, key=lambda k: (k[0], k[1], k[3], k[2])):
        if key[0] not in QTYPES or any(key not in r for r in runs):
            continue
        schedules = runs[0][key].keys()
        median = {s: statistics.median(r[key][s] for r in runs) for s in schedules}
        best, rule = min(median, key=median.get), base[key]
        if best == rule or median[best] > median[rule] * (1.0 - min_gain):
            continue
        if not all(r[key][best] < r[key][rule] for r in runs):
            continue
        out.append((key, best, median[best], rule, median[rule]))
    return out


def table_name(gpu):
    chip, _, variant = gpu.partition(":")
    return "kR8Decode" + chip[:1].upper() + chip[1:] + ("_" + variant if variant else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-gain", type=float, default=0.005)
    parser.add_argument("--merge", help="keep this header's entries for unmeasured problems")
    parser.add_argument("sweeps", nargs="+", help="GPU=FILE[,FILE...], GPU as 'gb202:170sm'")
    args = parser.parse_args()

    old = read_header(args.merge) if args.merge else {}
    tables = []
    for spec in args.sweeps:
        gpu, _, files = spec.partition("=")
        paths = files.split(",")
        rows = []
        for (fmt, epi, n, k), best, t_best, rule, t_rule in entries_for(gpu, paths, args.min_gain):
            rows.append((
                (QTYPES[fmt], EPILOGUES[epi], n, k),
                f"{{QType::{QTYPES[fmt]}, Epilogue::{EPILOGUES[epi]}, {n}, {k}, {best[0]}, {best[1]}}}, "
                f"// w{rule[0]}r{rule[1]} {t_rule:.2f} -> {t_best:.2f}",
            ))
        board = read_run(paths[0], gpu)[2]
        if gpu in old:
            measured = measured_keys(paths, gpu)
            entries, old_board = old.pop(gpu)
            rows += [(key, line) for key, line in entries.items() if key not in measured]
            board = board or old_board
        if board is None:
            sys.exit(f"{gpu}: no PCI id in its sweeps' 'board' line and none in --merge")
        tables.append((gpu, len(paths), rows, board))
    for gpu, (entries, board) in old.items():  # a GPU without new sweeps keeps its table
        if board is None:
            sys.exit(f"{gpu}: the --merge header records no board")
        tables.append((gpu, 0, list(entries.items()), board))
    known = registry_rows()
    for gpu, _, _, board in tables:
        if board not in known:
            sys.exit(f"{gpu}: PCI id {board} has no row in {GPU_FACTS}; add the board's row first")
        # A board that shares its id with a variant on other memory (core/machine_facts.cpp: the
        # 8 GB RTX 3060 on 128 bits, the RTX 4070 with GDDR6) is not its id's row: its id would
        # read back as the row's facts.
        cc, sms, l2, bw = known[board]
        # The rule of measured::row_board_facts (core/measured_boards.h): cc and SMs exact, L2 and
        # bandwidth to a part in 10^4.
        def close(x, y):
            return abs(x - y) <= 1e-4 * max(x, y)
        for seen in sorted(printed.get(board, ())):
            if seen[0] == cc and seen[1] == sms and close(seen[2], l2) and close(seen[3], bw):
                continue
            sys.exit(f"{gpu}: the sweeps' board (cc {seen[0]}, {seen[1]} SMs, L2 {seen[2]:g} MiB, "
                     f"{seen[3]:g} GB/s) is not the row of its PCI id {board} (cc {cc}, {sms} SMs, "
                     f"L2 {l2:g} MiB, {bw:g} GB/s): a variant under the id, which the table "
                     "cannot name")
    tables.sort(key=lambda t: t[0])
    order = {"Store": 0, "SwiGlu": 1}
    for _, _, rows, _ in tables:
        rows.sort(key=lambda r: (r[0][0], order.get(r[0][1], 9), r[0][3], r[0][2]))

    gain = f"{args.min_gain * 100:g} %"
    lines = [
        "#pragma once",
        "",
        "// R8 decode (T = 1) schedules for exact problems (codebook i-quants, K-quants), per GPU",
        "// (class id: chip and SM count): the fastest compiled schedule where it beats the rules",
        f"// (r8_rules_route) by at least {gain} in the median of repeated ninferno_r8_schedule_bench",
        "// sweeps and in every sweep. Every R8 schedule sums a row in one order, so an entry changes",
        "// no output byte. Written by tools/bench/r8_decode_table.py; do not edit by hand. An add",
        "// call takes the store entry (the same kernel up to the final read-modify-write).",
        "",
        '#include "core/device_class.h"',
        '#include "core/measured_boards.h"',
        '#include "core/weight.h"',
        '#include "ops/generic/generic_gemm.h"',
        "",
        "#include <array>",
        "#include <cstdint>",
        "#include <span>",
        "#include <string_view>",
        "",
        "namespace ninferno::ops::detail::generic {",
        "",
        "struct R8DecodeEntry {",
        "    QType qtype;",
        "    Epilogue epilogue; // Store or SwiGlu",
        "    std::int32_t n;    // the op's output rows (SwiGLU: the gate rows)",
        "    std::int32_t k;",
        "    std::int8_t warps;",
        "    std::int8_t rows;",
        "};",
        "",
        "struct R8DecodeTable {",
        '    std::string_view gpu; // DeviceClass::id() of the board, "gb202:170sm"',
        "    Board board;          // the board the sweeps ran on: the device takes the nearest",
        "    std::span<const R8DecodeEntry> entries;",
        "};",
        "",
    ]
    names = []
    for gpu, runs, entries, board in tables:
        if not entries:
            continue
        name = table_name(gpu)
        names.append((gpu, name, board))
        lines.append(f"// {gpu}: {len(entries)} entries (median us: rules -> entry).")
        lines.append(f"inline constexpr R8DecodeEntry {name}[] = {{")
        for _, line in entries:
            lines.append("    " + line)
        lines.append("};")
        lines.append("")
    lines.append(f"inline constexpr std::array<R8DecodeTable, {len(names)}> kR8DecodeTables{{{{")
    for gpu, name, board in names:
        lines.append(f'    {{"{gpu}", measured::board_of_pci({board}), {name}}},')
    lines.append("}};")
    # nearest_known_measured skips a board without its L2 or bandwidth: the build refuses one.
    lines.append("static_assert(measured_boards_known(std::span(kR8DecodeTables)));")
    lines.append("")
    lines.append("} // namespace ninferno::ops::detail::generic")
    with open(args.out, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for gpu, runs, entries, _ in tables:
        print(f"{gpu}: {len(entries)} entries ({runs} new sweeps)")


if __name__ == "__main__":
    main()
