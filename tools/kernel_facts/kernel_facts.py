#!/usr/bin/env python3
"""The kernel-facts snapshot of one build architecture (#612, #613): what ptxas made of the
generic path's register-MMA kernels, read off the linked binary with `cuobjdump
--dump-resource-usage` (registers, stack, local and shared memory) and `cuobjdump -sass` (the
instructions of the K loop, counted per iteration: sass_counts.py), and written as committed data
(src/ops/generic/data/kernel_facts/sm_<arch>.txt). No GPU is involved: the selector's price reads
these files (ops/generic/kernel_facts.h), never the host GPU's kernels, so a decision is the same
on every machine, whatever nvcc or driver.

    kernel_facts.py extract --binary BIN --arch 89 --nvcc 13.4.92 [--flags TEXT]
                            [--source-root DIR] [-o FILE]
    kernel_facts.py check   --binary BIN --expected FILE [--nvcc VER]
    kernel_facts.py hash    --source-root DIR

`extract` writes the snapshot (stdout without -o). `check` extracts the same from BIN and compares
it with FILE row by row: exit 0 where they agree, 1 where they differ (every differing kernel is
named, with the line to regenerate), 77 where BIN was compiled by another nvcc than FILE records
and holds the same kernels (ptxas then allocates other registers and schedules other instructions
by construction; the price still reads FILE, so the decisions stay the release's), 3 on a usage or
tool error. A kernel added or removed fails under any nvcc.

The binary is the linked executable or shared library (any program that links ninferno_ops):
under relocatable device code the per-unit objects report no shared memory and a smaller stack
for a kernel, the device link decides them. `cuobjdump` and `cu++filt` come from the CUDA
toolkit's bin directory (--cuda-bin, else the directory of `nvcc` on PATH, else CUDA_PATH). The
SASS pass disassembles the register-MMA kernels in batches (--sass-batch; about ten minutes for
an engine's 2400 kernels, 1.5 GB of memory) and keeps only their loops; --sass-dump reads a saved
`cuobjdump -sass` instead, --no-sass leaves the loop columns out (a resources-only check).

The loop's critical path (the cp_* columns) is chosen with the architecture's measured dependent
latencies, its row of src/core/data/isa_facts.json (--isa-facts for another file), and the
snapshot's `chain` line records them; where the row or one of its latencies is missing the nominal
clocks of sass_counts.py choose and the line says so. `check` with the SASS pass fails (1) on a
snapshot whose `chain` line is not what the ISA facts give today: extract it again after an ISA
row lands or moves.

A kernel's key: the decoder type, the schedule's template arguments (warps_m, warps_k, m_tiles,
n_tiles, cluster: the entries of NINFERNO_GENERIC_RMMA_SCHEDULES, generic_schedules.h), the
epilogue, and the latency and add-sums variants, as they stand in the demangled symbol; the
engine builds the same key from the template arguments it launches (generic_gemm_launch.cuh),
so nothing is retyped. Names are written in the canonical form of both sides: the namespace
`ninferno::ops::detail::generic::` dropped, no spaces, integer template arguments bare and bools
as true/false.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sass_counts  # noqa: E402

SCHEMA = 3
NAMESPACE = "ninferno::ops::detail::generic::"
FAMILY = "rmma"
RESOURCE_COLUMNS = ("registers", "stack", "local", "shared")
KEY_COLUMNS = ("decoder", "warps_m", "warps_k", "m_tiles", "n_tiles", "cluster", "epilogue",
               "latency", "add_sums")
COLUMNS = " ".join(("family",) + KEY_COLUMNS + RESOURCE_COLUMNS + sass_counts.COLUMNS)
EPILOGUE_SWIGLU = 2  # Epilogue::SwiGlu (generic_gemm.h): two row sets per warp
# The sources whose text decides the device code: the CUDA units and headers of the ops tree.
SOURCE_GLOBS = (".cu", ".cuh")
SOURCE_DIRS = ("src/ops",)
# The measured ISA facts the critical path's latencies come from (sass_counts.chain_weights).
ISA_FACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "core",
                         "data", "isa_facts.json")


def fail(message, code=3):
    print(f"kernel_facts: {message}", file=sys.stderr)
    sys.exit(code)


def cuda_tool(name, cuda_bin):
    """cuobjdump / cu++filt: from --cuda-bin, beside nvcc on PATH, or under CUDA_PATH."""
    candidates = []
    if cuda_bin:
        candidates.append(os.path.join(cuda_bin, name))
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(os.path.join(os.path.dirname(os.path.realpath(nvcc)), name))
    for var in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(var):
            candidates.append(os.path.join(os.environ[var], "bin", name))
    candidates.append(name)
    for c in candidates:
        found = shutil.which(c) or shutil.which(c + ".exe")
        if found:
            return found
    fail(f"{name} not found (pass --cuda-bin <cuda>/bin)")


RESERVED_SHARED_PER_CTA = 1024


def sm_major(arch):
    """The major compute capability of "sm_90", "sm_120a", "sm_86"."""
    digits = re.match(r"sm_(\d+)", arch).group(1)
    return int(digits[:-1])


def resource_usage(binary, cuda_bin, dump=None):
    """[(arch, mangled name, (REG, STACK, LOCAL, SHARED))] over every fatbin image of the binary
    (or of a saved `cuobjdump --dump-resource-usage` output, `dump`)."""
    if dump:
        with open(dump) as f:
            text = f.read()
    else:
        out = subprocess.run([cuda_tool("cuobjdump", cuda_bin), "--dump-resource-usage", binary],
                             capture_output=True, text=True)
        if out.returncode != 0:
            fail(f"cuobjdump failed on {binary}: {out.stderr.strip()[:400]}")
        text = out.stdout
    facts = []
    arch = None
    function = None
    for line in text.splitlines():
        m = re.match(r"\s*arch = (sm_\w+)", line)
        if m:
            arch = m.group(1)
            continue
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            function = m.group(1)
            continue
        m = re.match(r"\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", line)
        if m and function is not None:
            reg, stack, shared, local = (int(v) for v in m.groups())
            # From sm_90 on, cuobjdump's SHARED counts the 1 KiB the driver reserves per CTA
            # (sm_90: on every kernel; sm_100 and sm_120: on a kernel with static shared memory).
            # The runtime's static size (cudaFuncAttributes::sharedSizeBytes) leaves it out, and the
            # occupancy rule adds it itself (reserved_shared_bytes_per_cta), so the snapshot keeps
            # the runtime's figure. Checked on sm_120a: 714 of 1152 launched kernels, each +1024.
            if arch is not None and sm_major(arch) >= 9:
                shared = max(0, shared - RESERVED_SHARED_PER_CTA)
            facts.append((arch, function, (reg, stack, local, shared)))
            function = None
    return facts


def demangle(names, cuda_bin):
    """cu++filt over many names at once (one per input line)."""
    if not names:
        return {}
    out = subprocess.run([cuda_tool("cu++filt", cuda_bin)], input="\n".join(names) + "\n",
                         capture_output=True, text=True)
    if out.returncode != 0:
        fail(f"cu++filt failed: {out.stderr.strip()[:400]}")
    lines = out.stdout.splitlines()
    if len(lines) != len(names):
        fail(f"cu++filt returned {len(lines)} lines for {len(names)} names")
    return dict(zip(names, lines))


def split_top_level(text):
    """Comma-separated items outside any <> or () nesting."""
    items, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append(text[start:i])
            start = i + 1
    items.append(text[start:])
    return [s.strip() for s in items]


def canonical(arg):
    """One template argument in the canonical spelling (module docstring)."""
    arg = re.sub(r"\((?:unsigned )?(?:int|long|short|char|bool|long long)\)", "", arg)
    arg = arg.replace("(bool)", "")
    arg = re.sub(r"\(" + re.escape(NAMESPACE) + r"\w+\)", "", arg)  # (Epilogue)2 -> 2
    arg = arg.replace(NAMESPACE, "")
    arg = re.sub(r"\b(struct|class|enum)\s+", "", arg)
    arg = re.sub(r"\s+", "", arg)
    return arg


def as_bool(arg):
    if arg in ("true", "1"):
        return 1
    if arg in ("false", "0"):
        return 0
    raise ValueError(arg)


def rmma_key(demangled):
    """The key of an rmma_kernel instance from its demangled name, or None for other symbols."""
    head = "void " + NAMESPACE + "rmma_kernel<"
    if not demangled.startswith(head):
        return None
    depth, end = 1, len(head)
    while end < len(demangled) and depth > 0:
        if demangled[end] == "<":
            depth += 1
        elif demangled[end] == ">":
            depth -= 1
        end += 1
    args = [canonical(a) for a in split_top_level(demangled[len(head):end - 1])]
    if len(args) != 5:
        fail(f"unexpected rmma_kernel arguments in {demangled}")
    decoder, schedule, epilogue, latency, add_sums = args
    m = re.fullmatch(r"RmmaSchedule<(\d+),(\d+),(\d+),(\d+),(\d+)>", schedule)
    if not m or not re.fullmatch(r"\d+", epilogue):
        fail(f"unexpected rmma_kernel schedule or epilogue in {demangled}")
    return (decoder, *(int(v) for v in m.groups()), int(epilogue), as_bool(latency),
            as_bool(add_sums))


def key_tiles(key):
    """The m16n8 tiles one warp of the kernel accumulates: m_tiles * n_tiles, twice under SwiGLU."""
    _, _, _, m_tiles, n_tiles, _, epilogue, _, _ = key
    return m_tiles * n_tiles * (2 if epilogue == EPILOGUE_SWIGLU else 1)


def extract_resources(binary, cuda_bin, dump=None):
    """({arch: {key: (registers, stack, local, shared)}}, {arch: {mangled: key}}) of the binary's
    rmma kernels. A kernel that appears in two images of one arch (a template instantiated in two
    units) must agree."""
    usage = resource_usage(binary, cuda_bin, dump)
    rmma = [u for u in usage if "rmma_kernel" in u[1]]
    names = demangle(sorted({name for _, name, _ in rmma}), cuda_bin)
    rows, keys = {}, {}
    for arch, name, value in rmma:
        key = rmma_key(names[name])
        if key is None:
            continue
        per_arch = rows.setdefault(arch, {})
        if key in per_arch and per_arch[key] != value:
            fail(f"{name} appears twice in {arch} of {binary or dump} with different resources "
                 f"{per_arch[key]} and {value}")
        per_arch[key] = value
        keys.setdefault(arch, {})[name] = key
    return rows, keys


SASS_BATCH = 256
# The longest --function list handed to cuobjdump: a Windows command line holds 32 767 characters
# in all, and 256 mangled rmma names run to 38 KB.
SASS_BATCH_CHARS = 24000


def name_batches(names, batch=SASS_BATCH, max_chars=SASS_BATCH_CHARS):
    """`names` cut into lists of at most `batch` names whose comma-joined length stays within
    `max_chars` (a single longer name is a batch of its own); batch 0: one batch of everything."""
    if batch <= 0:
        return [list(names)]
    out, current, length = [], [], 0
    for name in names:
        if current and (len(current) >= batch or length + 1 + len(name) > max_chars):
            out.append(current)
            current, length = [], 0
        current.append(name)
        length += len(name) + (1 if length else 0)
    if current:
        out.append(current)
    return out


def sass_lines(binary, cuda_bin, arch, sass_dump=None, names=None, batch=SASS_BATCH):
    """The lines of `cuobjdump -sass` for one image of the binary, streamed, or of a saved dump.
    With `names` (the mangled functions wanted) and a batch size, cuobjdump is run per batch with
    --function: disassembling a whole linked image at once holds the image's worth of state
    (nvdisasm took 26 GB on a 370 MB relocatable sm_75 binary and was killed on a 30 GB host),
    per batch it stays near 1.5 GB at about a quarter of a second per function; batch 0 streams
    the whole image in one run (faster where the memory is there)."""
    if sass_dump:
        with open(sass_dump, errors="replace") as f:
            yield from f
        return
    tool = cuda_tool("cuobjdump", cuda_bin)
    batches = [None] if not names or batch <= 0 else name_batches(names, batch)
    for chunk in batches:
        command = [tool, "-sass", "-arch", arch]
        if chunk is not None:
            command += ["-fun", ",".join(chunk)]
        command.append(binary)
        # stderr goes to a file, not a pipe: a pipe nobody drains while the dump streams would
        # stall cuobjdump once its warnings (a function absent from one image) fill it.
        with tempfile.TemporaryFile(mode="w+", errors="replace") as errors:
            try:
                proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True,
                                        errors="replace", bufsize=1 << 20)
            except OSError as error:
                fail(f"cannot run cuobjdump ({error}); the command line was "
                     f"{len(' '.join(command))} characters")
            try:
                yield from proc.stdout
            finally:
                if proc.poll() is None:  # the consumer stopped early: do not leave the dump running
                    proc.kill()
                code = proc.wait()
            errors.seek(0)
            if code != 0:
                fail(f"cuobjdump -sass failed on {binary}: {errors.read().strip()[:400]}")


def chain_weight_sets(arch, isa_facts=ISA_FACTS):
    """(weights, basis) the critical path of the `arch` kernels is chosen with: the measured
    latencies of the architecture's row in the ISA facts table at each resident warp count
    (sass_counts.CP_WARPS; the first, one warp, chooses the chain) under "measured"; where the
    table has no row for it, or the row lacks one of the latencies, the nominal ones alone under
    "nominal" (no facts: never another architecture's row)."""
    with open(isa_facts) as f:
        row = json.load(f)["arches"].get(arch)
    facts = {name: v["value"] for name, v in row["facts"].items()} if row else {}
    sets = [sass_counts.chain_weights(facts, w) for w in sass_counts.CP_WARPS]
    if None in sets:
        return (sass_counts.CP_NOMINAL,), "nominal"
    return tuple(sets), "measured"


def chain_line(weights, basis):
    """The header's `chain` value: the basis and the clocks per class that chose the path."""
    return basis + " " + " ".join(f"{name[3:]}={weights[0][name]:g}"
                                  for name in sass_counts.CP_FIELDS)


def extract_counts(binary, cuda_bin, arch, keys, sass_dump=None, batch=SASS_BATCH,
                   weights=(sass_counts.CP_NOMINAL,)):
    """{key: LoopCounts} of the rmma kernels `keys` ({mangled: key}) of one image, from its SASS;
    `weights` choose the critical path (sass_counts.critical_path). A kernel disassembled twice in
    the image (two units) must agree."""
    counts = {}
    lines = sass_lines(binary, cuda_bin, arch, sass_dump, sorted(keys), batch)
    for name, body in sass_counts.iter_functions(lines, lambda n: n in keys):
        key = keys[name]
        c = sass_counts.loop_counts(body, key_tiles(key), weights)
        if key in counts and counts[key] != c:
            fail(f"{name} is disassembled twice in {arch} with different loops {counts[key]} "
                 f"and {c}")
        counts[key] = c
    return counts


def extract_rows(binary, cuda_bin, arch, dump=None, sass_dump=None, sass=True,
                 batch=SASS_BATCH, weights=(sass_counts.CP_NOMINAL,)):
    """{key: (registers, stack, local, shared, LoopCounts)} of one image of the binary (`arch`:
    the label, "89"); the LoopCounts is None under sass=False; `weights` as extract_counts'."""
    rows, keys = extract_resources(binary, cuda_bin, dump)
    image = image_arch(rows, arch)
    resources = rows[image]
    if not sass:
        return {k: (*v, None) for k, v in resources.items()}
    counts = extract_counts(binary, cuda_bin, image, keys[image], sass_dump, batch, weights)
    missing = sorted(k for k in resources if k not in counts)
    if missing:
        fail(f"{len(missing)} rmma kernels of {image} have resources but no SASS, first "
             f"{row_key(missing[0])}")
    return {k: (*v, counts[k]) for k, v in resources.items()}


def loop_audit(rows):
    """Lines naming what the loop finder could not settle: kernels without a loop, with several
    candidate loops, and decoders whose rows disagree on k_iter (one decoder has one kRun, so one
    loop iteration covers one K count in every schedule, epilogue and variant)."""
    out = []
    none = [k for k, v in rows.items() if v[4] is not None and v[4].loops == 0]
    several = [k for k, v in rows.items() if v[4] is not None and v[4].loops > 1]
    if none:
        out.append(f"{len(none)} kernels without a K loop, first {row_key(none[0])}")
    if several:
        out.append(f"{len(several)} kernels with several candidate loops, first "
                   f"{row_key(several[0])}")
    by_decoder = {}
    for k, v in rows.items():
        if v[4] is not None:
            by_decoder.setdefault(k[0], set()).add(v[4].k_iter)
    for decoder, k_iters in sorted(by_decoder.items()):
        if len(k_iters) > 1:
            out.append(f"{decoder}: k_iter differs across its kernels: "
                       f"{', '.join(str(k) for k in sorted(k_iters))}")
    return out


def source_hash(root):
    """sha256 over the device sources (SOURCE_DIRS, SOURCE_GLOBS), path and text, sorted."""
    h = hashlib.sha256()
    for d in SOURCE_DIRS:
        base = os.path.join(root, d)
        paths = []
        for dirpath, _, files in os.walk(base):
            for f in files:
                if f.endswith(SOURCE_GLOBS):
                    paths.append(os.path.join(dirpath, f))
        for p in sorted(paths):
            h.update(os.path.relpath(p, root).replace(os.sep, "/").encode())
            h.update(b"\0")
            with open(p, "rb") as f:
                h.update(f.read().replace(b"\r\n", b"\n"))
            h.update(b"\0")
    return h.hexdigest()


def row_key(key):
    return " ".join(str(v) for v in key)


def row_line(key, value):
    fields = [FAMILY, *key, *value[:4]]
    if value[4] is not None:
        fields += value[4].row()
    return " ".join(str(v) for v in fields)


def render(arch, rows, nvcc, flags, source, chain=None):
    """The snapshot's text; `chain` (chain_line) names the latencies the critical paths of
    `rows` were chosen with."""
    with_loops = any(v[4] is not None for v in rows.values())
    columns = COLUMNS if with_loops else COLUMNS.split(" loops ")[0]
    chain_lines = []
    if with_loops and chain:
        short = [v[4].cp_short for v in rows.values() if v[4].cp_short > 0]
        chain_lines = [f"chain {chain}",
                       f"chain_short {len(short)} {max(short, default=0.0):.3f}"]
    lines = [
        f"# NInferno kernel facts, schema {SCHEMA}: what ptxas made of the register-MMA kernels of",
        f"# the sm_{arch} build, read off the linked binary with cuobjdump. Per kernel: the registers",
        "# a thread keeps, its stack frame and local bytes, the CTA's static shared memory",
        "# (--dump-resource-usage); then the K loop's instructions per iteration (-sass; one",
        "# iteration is two K steps, k_iter K values per warp): the candidate loops found, the",
        "# MMAs, of them the min-term ones (A = 1.0 pairs) and the MMA shape, the loads and stores",
        "# by space with the bytes a warp requests (32 lanes x width; LDG global, LD generic, LDGSTS",
        "# async copy, LDS/STS shared, LDL/STL local), SHFL, integer and float ALU, uniform-datapath,",
        "# the rest, the total; then the loop's critical path, the instructions per latency class",
        "# (cp_*, one per measured latency: the MMA, global / local and shared loads, SHFL, IMAD,",
        "# IADD3, LOP3, PRMT, SHF, HFMA2, FFMA, the I2F / F2I converts, the F32 / F16 ones, the",
        "# rest) on the longest chain of register dependences of one iteration, chosen with the",
        "# clocks per class of the `chain` line (measured: this architecture's row of",
        "# src/core/data/isa_facts.json at one resident warp; nominal: that row lacks a latency);",
        "# `chain_short` counts the kernels where another chain is longer at 2, 4 or 8 resident",
        "# warps, and by how many clocks at most. Static counts of one full iteration; how the loop",
        "# and the path are found is in",
        "# tools/kernel_facts/sass_counts.py. Generated by tools/kernel_facts/kernel_facts.py; never",
        "# edited by hand. The selector's price reads this file, not the host GPU's kernels",
        "# (ops/generic/kernel_facts.h, #612), and ninferno_kernel_facts_test asserts that a build's",
        "# kernels equal it. After a kernel or toolchain change regenerate it with the build's own",
        "# binary:",
        f"#   python3 tools/kernel_facts/kernel_facts.py extract --binary <linked exe> --arch {arch}",
        f"#     --nvcc <version> -o src/ops/generic/data/kernel_facts/sm_{arch}.txt",
        f"arch {arch}",
        f"nvcc {nvcc}",
        f"flags {flags}",
        f"source {source}",
        *chain_lines,
        f"kernels {len(rows)}",
        f"columns {columns}",
    ]
    lines += [row_line(k, rows[k]) for k in sorted(rows)]
    return "\n".join(lines) + "\n"


def parse_snapshot(text):
    """(header, {key: (registers, stack, local, shared, LoopCounts or None)}) of a snapshot; a
    schema 1 file (resources only) parses with None for the loop."""
    header, rows = {}, {}
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    for line in lines:
        fields = line.split()
        if fields[0] != FAMILY:
            header[fields[0]] = line[len(fields[0]) + 1:] if len(fields) > 1 else ""
    for want in ("arch", "nvcc", "kernels", "columns"):
        if want not in header:
            fail(f"snapshot has no '{want}' line")
    # Every row spells the header's columns: 14 fields in a resources-only file, 14 plus the
    # loop's in a schema 2 one, plus the critical path's in a schema 3 one, never a mixture.
    columns = header["columns"].split()
    n_cp = len(sass_counts.CP_FIELDS)
    if columns not in (COLUMNS.split()[:14], COLUMNS.split()[:-n_cp], COLUMNS.split()):
        fail(f"unexpected columns: {header['columns']}")
    width = len(columns)
    for line in lines:
        fields = line.split()
        if fields[0] != FAMILY:
            continue
        if len(fields) != width:
            fail(f"a row of {len(fields)} fields under {width} columns: {line}")
        try:
            key = (fields[1], *(int(v) for v in fields[2:10]))
            counts = sass_counts.LoopCounts.from_row(fields[14:]) if width > 14 else None
            rows[key] = (*(int(v) for v in fields[10:14]), counts)
        except ValueError as error:
            fail(f"malformed row ({error}): {line}")
    if int(header["kernels"]) != len(rows):
        fail(f"snapshot says {header['kernels']} kernels, holds {len(rows)}")
    return header, rows


def image_arch(rows, arch):
    """The binary's image for the arch label: sm_<label>; a family target's image is sm_120f."""
    name = "sm_" + arch
    if name in rows:
        return name
    fail(f"the binary holds no sm_{arch} image, only {', '.join(sorted(rows)) or 'none'}")


def cmd_extract(args):
    weights, basis = chain_weight_sets(args.arch, args.isa_facts)
    rows = extract_rows(args.binary, args.cuda_bin, args.arch, args.dump, args.sass_dump,
                        not args.no_sass, args.sass_batch, weights)
    if not rows:
        fail("no rmma_kernel in the binary")
    audit = loop_audit(rows)
    for line in audit:
        print(f"kernel_facts: {line}", file=sys.stderr)
    if basis == "nominal" and not args.no_sass:
        print(f"kernel_facts: {args.isa_facts} has no dependent latencies for sm_{args.arch}: "
              "the critical paths are chosen with nominal ones; extract again once its row is "
              "measured", file=sys.stderr)
    source = args.source or (source_hash(args.source_root) if args.source_root else "unknown")
    text = render(args.arch, rows, args.nvcc or "unknown", args.flags or "unknown", source,
                  chain_line(weights, basis))
    if args.output:
        with open(args.output, "w", newline="\n") as f:
            f.write(text)
        print(f"kernel_facts: wrote {len(rows)} kernels of sm_{args.arch} to {args.output}")
    else:
        sys.stdout.write(text)
    if audit:
        # Written for inspection, but not a snapshot to commit: the loop finder did not settle.
        print(f"kernel_facts: {len(audit)} loop audit findings (above): the counts are not "
              "settled, do not commit them", file=sys.stderr)
        return 1
    return 0


def cmd_check(args):
    if not os.path.exists(args.expected):
        print(f"kernel_facts: no snapshot {args.expected} for this build architecture: generate "
              "it with\n  python3 tools/kernel_facts/kernel_facts.py extract --binary "
              f"{args.binary or args.dump} --arch <label> --nvcc {args.nvcc or '<version>'} "
              f"-o {args.expected}")
        return 1
    with open(args.expected) as f:
        header, expected = parse_snapshot(f.read())
    arch = header["arch"]
    other_nvcc = bool(args.nvcc) and args.nvcc != header["nvcc"]
    # A snapshot without loop columns is checked on its resources alone; under another nvcc the
    # kernel set is all that is compared, so the SASS pass (minutes) is not run.
    with_sass = not args.no_sass and not other_nvcc and \
        any(v[4] is not None for v in expected.values())
    weights, basis = chain_weight_sets(arch, args.isa_facts)
    if with_sass and "chain" in header and header["chain"] != chain_line(weights, basis):
        # The critical paths were chosen with other latencies than the ISA facts give today (a
        # row measured or measured again since): the cp_* columns are then another chain's.
        print(f"kernel_facts: {args.expected} chose its critical paths with\n  {header['chain']}\n"
              f"and {args.isa_facts} gives sm_{arch} today\n  {chain_line(weights, basis)}\n"
              "Regenerate the snapshot with\n"
              f"  python3 tools/kernel_facts/kernel_facts.py extract --binary "
              f"{args.binary or args.dump} --arch {arch} --nvcc <version> -o {args.expected}")
        return 1
    got = extract_rows(args.binary, args.cuda_bin, arch, args.dump, args.sass_dump, with_sass,
                       args.sass_batch, weights)
    if not with_sass:
        expected = {k: (*v[:4], None) for k, v in expected.items()}
    missing = sorted(k for k in expected if k not in got)
    extra = sorted(k for k in got if k not in expected)
    if other_nvcc and not (missing or extra):
        # The set of kernels does not depend on the compiler; their registers and schedules do.
        print(f"kernel_facts: SKIPPED the resources and loop counts: this build's nvcc "
              f"{args.nvcc} is not the snapshot's {header['nvcc']} ({args.expected}); ptxas then "
              "allocates other registers and schedules other instructions by construction. The "
              "kernel set agrees. The price reads the snapshot either way; regenerate it only on "
              "the release toolchain.")
        return 77
    differ = [] if other_nvcc else sorted(
        k for k in expected if k in got and expected[k] != got[k])
    if not (missing or extra or differ):
        print(f"kernel_facts: {len(got)} kernels of sm_{arch} equal {args.expected}"
              + ("" if with_sass else " (resources only)"))
        return 0
    shown = 0
    for k in differ:
        if shown < args.show:
            print(f"  {row_line(k, expected[k])}\n    build: {row_line(k, got[k])}")
        shown += 1
    for k in missing:
        if shown < args.show:
            print(f"  missing in the build: {row_line(k, expected[k])}")
        shown += 1
    for k in extra:
        if shown < args.show:
            print(f"  not in the snapshot: {row_line(k, got[k])}")
        shown += 1
    if shown > args.show:
        print(f"  ... {shown - args.show} more")
    binary = args.binary or args.dump
    print(f"kernel_facts: the sm_{arch} kernels of {binary} differ from {args.expected}: "
          f"{len(differ)} changed, {len(missing)} missing, {len(extra)} new (nvcc "
          f"{args.nvcc or 'unknown'}, snapshot {header['nvcc']}). A kernel, flag or compiler "
          "change moved them: regenerate the snapshot with\n"
          f"  python3 tools/kernel_facts/kernel_facts.py extract --binary {binary} "
          f"--arch {arch} --nvcc <version> -o {args.expected}\n"
          "and review the selector goldens that move with it.")
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "check"):
        p = sub.add_parser(name)
        p.add_argument("--binary", help="the linked executable or shared library")
        p.add_argument("--dump", help="a saved `cuobjdump --dump-resource-usage` output instead")
        p.add_argument("--sass-dump", help="a saved `cuobjdump -sass` output of the image instead")
        p.add_argument("--no-sass", action="store_true",
                       help="resources only: no SASS pass, no loop columns")
        p.add_argument("--sass-batch", type=int, default=SASS_BATCH,
                       help="kernels per cuobjdump run of the SASS pass (0: the whole image at "
                            f"once; default {SASS_BATCH}, bounded memory)")
        p.add_argument("--cuda-bin", help="directory of cuobjdump and cu++filt")
        p.add_argument("--nvcc", help="the nvcc version the binary was built with")
        p.add_argument("--isa-facts", default=ISA_FACTS,
                       help="the measured ISA facts table whose latencies choose the critical "
                            "path (default: the source tree's src/core/data/isa_facts.json)")
    e = sub.choices["extract"]
    e.add_argument("--arch", required=True, help="the build architecture label (89, 120a, ...)")
    e.add_argument("--flags", help="the CUDA flags of the build, recorded in the header")
    e.add_argument("--source-root", help="the source tree, for the header's source hash")
    e.add_argument("--source", help="the source hash itself (from `hash`), in place of a tree")
    e.add_argument("-o", "--output")
    c = sub.choices["check"]
    c.add_argument("--expected", required=True, help="the committed snapshot")
    c.add_argument("--show", type=int, default=40, help="differing kernels to list")
    h = sub.add_parser("hash", help="print the source hash of a tree (the header's `source`)")
    h.add_argument("--source-root", required=True)
    args = parser.parse_args(argv)
    if args.command != "hash" and not (args.binary or args.dump):
        parser.error("--binary or --dump is required")
    if args.command != "hash" and not (args.binary or args.sass_dump or args.no_sass):
        parser.error("the SASS pass needs --binary or --sass-dump (or --no-sass)")
    if args.command == "hash":
        print(source_hash(args.source_root))
        return 0
    return cmd_extract(args) if args.command == "extract" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
