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
    kernel_facts.py check   --binary BIN --expected FILE [--nvcc VER] [--stamp FILE]
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

A kernel's key: the decoder type, the schedule's template arguments (warps_m, warps_k, m_tiles,
n_tiles, cluster: the entries of NINFERNO_GENERIC_RMMA_SCHEDULES, generic_schedules.h), the
epilogue, the latency variant and where the min-term sums come from (`sums`: rmma_kernel's
RmmaSums, 0 the MMA with ones, 1 FP32 adds, 2 the pre-summed plane; a binary from before it
spells the adds as a bool, read as 0 / 1), as they stand in the demangled symbol; the
engine builds the same key from the template arguments it launches (generic_gemm_launch.cuh),
so nothing is retyped. Names are written in the canonical form of both sides: the namespace
`ninferno::ops::detail::generic::` dropped, no spaces, integer template arguments bare and bools
as true/false.

After the register-MMA rows a snapshot holds one section per further kernel family (FAMILIES,
#785): ggml's matrix kernel `mul_mat_q<type, J, fallback, prec>` (mmq) and the W4A8 tiles
(w4a8), each with its own nvcc, kernel count and columns (`family_nvcc`, `family_kernels`,
`family_columns` after its `family <name>` line), and the pipe counts of the loop
(sass_counts.PIPE_FIELDS) besides schema 4's. `extract --families mmq,w4a8 --update FILE`
rewrites those sections alone and keeps the rest of FILE byte for byte. `check` compares every
section the snapshot has: in full where the build's nvcc is the one the section records, its
kernel set alone otherwise (the exit code then follows the register-MMA rows); a family the
snapshot has no section for is named and not checked.
"""

import argparse
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sass_counts  # noqa: E402

SCHEMA = 4
NAMESPACE = "ninferno::ops::detail::generic::"
FAMILY = "rmma"
RESOURCE_COLUMNS = ("registers", "stack", "local", "shared")
KEY_COLUMNS = ("decoder", "warps_m", "warps_k", "m_tiles", "n_tiles", "cluster", "epilogue",
               "latency", "sums")
ROW_HEAD = ("family",) + KEY_COLUMNS + RESOURCE_COLUMNS
COLUMNS = " ".join(ROW_HEAD + sass_counts.COLUMNS)
# The columns line of every snapshot `check` reads, and the loop fields its rows carry: schema 1
# the resources alone (`extract --no-sass`), 4 the counts and the static schedule. Older formats
# (the counts without a schedule, the critical-path columns) are not read: regenerate the file.
SCHEMA_COLUMNS = {
    1: (ROW_HEAD, 0),
    4: (ROW_HEAD + sass_counts.COLUMNS, len(sass_counts.COLUMNS)),
}
EPILOGUE_SWIGLU = 2  # Epilogue::SwiGlu (generic_gemm.h): two row sets per warp
# The sources whose text decides the device code: the CUDA units and headers of the ops tree.
SOURCE_GLOBS = (".cu", ".cuh")
SOURCE_DIRS = ("src/ops",)


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


def as_sums(arg):
    """rmma_kernel's RmmaSums argument as the key's integer (a bool before #619: the adds)."""
    if arg in ("true", "false"):
        return as_bool(arg)
    if re.fullmatch(r"[0-2]", arg):
        return int(arg)
    raise ValueError(arg)


def template_args(demangled, head):
    """The canonical template arguments of a demangled `head<...>` symbol, or None where the
    symbol does not start with `head` (its `<` included)."""
    if not demangled.startswith(head):
        return None
    depth, end = 1, len(head)
    while end < len(demangled) and depth > 0:
        if demangled[end] == "<":
            depth += 1
        elif demangled[end] == ">":
            depth -= 1
        end += 1
    return [canonical(a) for a in split_top_level(demangled[len(head):end - 1])]


def rmma_key(demangled):
    """The key of an rmma_kernel instance from its demangled name, or None for other symbols."""
    args = template_args(demangled, "void " + NAMESPACE + "rmma_kernel<")
    if args is None:
        return None
    if len(args) != 5:
        fail(f"unexpected rmma_kernel arguments in {demangled}")
    decoder, schedule, epilogue, latency, sums = args
    m = re.fullmatch(r"RmmaSchedule<(\d+),(\d+),(\d+),(\d+),(\d+)>", schedule)
    if not m or not re.fullmatch(r"\d+", epilogue):
        fail(f"unexpected rmma_kernel schedule or epilogue in {demangled}")
    return (decoder, *(int(v) for v in m.groups()), int(epilogue), as_bool(latency),
            as_sums(sums))


def key_tiles(key):
    """The m16n8 tiles one warp of the kernel accumulates: m_tiles * n_tiles, twice under SwiGLU."""
    _, _, _, m_tiles, n_tiles, _, epilogue, _, _ = key
    return m_tiles * n_tiles * (2 if epilogue == EPILOGUE_SWIGLU else 1)


GGML_ENUM = re.compile(r"^\((?:enum\s+)?(?:ggml_type|ggml_prec)\)(-?\d+)$")


def ggml_int(arg):
    """A template argument of ggml's kernels as an integer: `(ggml_type)12`, `32`, `false`."""
    m = GGML_ENUM.match(arg)
    if m:
        return int(m.group(1))
    if arg in ("true", "false"):
        return as_bool(arg)
    return int(arg)


def mmq_key(demangled):
    """The key of a ggml matrix kernel instance, `mul_mat_q<type, J, fallback, prec>` (mmq.cuh;
    a static function, so no namespace): the ggml_type and ggml_prec values as cu++filt spells
    them, the column tile J, the fallback flag. None for other symbols (the stream-k fixup)."""
    args = template_args(demangled, "void mul_mat_q<")
    if args is None:
        return None
    if len(args) != 4:
        fail(f"unexpected mul_mat_q arguments in {demangled}")
    try:
        return tuple(ggml_int(a) for a in args)
    except ValueError:
        fail(f"unexpected mul_mat_q arguments in {demangled}")


W4A8_SCHEDULE = re.compile(r"W4a8Schedule<(\d+),(\d+),(\d+),(\d+)>")


def w4a8_key(demangled):
    """The key of a W4A8 tile (generic_w4a8.cuh): the codec, the schedule's block_m, block_n,
    warps_m, warps_n, the epilogue, and the variant: w4a8_kernel<C, S, E, Flavor> (flavor, no
    stages or producers) or w4a8_ws_kernel<C, S, E, Stages, Producers> (flavor 0). None for other
    symbols (the activation quantizer)."""
    for name, count in (("w4a8_kernel", 4), ("w4a8_ws_kernel", 5)):
        args = template_args(demangled, "void " + NAMESPACE + name + "<")
        if args is None:
            continue
        m = W4A8_SCHEDULE.fullmatch(args[1]) if len(args) == count else None
        if not m or not all(re.fullmatch(r"\d+", a) for a in args[2:]):
            fail(f"unexpected {name} arguments in {demangled}")
        rest = [int(a) for a in args[2:]]
        variant = (rest[1], 0, 0) if count == 4 else (0, rest[1], rest[2])
        return (args[0], *(int(v) for v in m.groups()), rest[0], *variant)
    return None


GEMMA4_NAMESPACE = "ninferno::ops::"
GEMMA4_GEOMETRY = re.compile(r"(?:ninferno::ops::)?Gemma4Geometry<(\d+),(-?\d+),(\d+)>")


def gemma4_key(demangled):
    """The key of a Gemma 4 attention kernel (src/ops/softmax_attention/gemma4, #830): the
    kernel ("attention", "volta", "volta_tiled"), the head width D, the geometry's group G (0: a
    runtime group) and slice width, the rows (the tile's warp rows WR; the Volta decode instance's
    MaxRows), the key tile Bc, and the attention kernel's Partial, Fp8 and Chain flags (0 for the
    Volta kernels). None for other symbols (the combine and copy kernels)."""
    args = template_args(demangled, "void " + GEMMA4_NAMESPACE + "gemma4_attention_kernel<")
    if args is not None:
        m = GEMMA4_GEOMETRY.fullmatch(args[0]) if len(args) == 6 else None
        if not m or not all(re.fullmatch(r"\d+|true|false", a) for a in args[1:]):
            fail(f"unexpected gemma4_attention_kernel arguments in {demangled}")
        wr, bc = int(args[1]), int(args[2])
        return ("attention", *(int(v) for v in m.groups()), wr, bc,
                *(as_bool(a) for a in args[3:]))
    for name, kind in (("gemma4_volta_decode_tiled_kernel", "volta_tiled"),
                       ("gemma4_volta_decode_kernel", "volta")):
        head = "void " + GEMMA4_NAMESPACE + "detail::(anonymousnamespace)::" + name + "<"
        # cu++filt spells the anonymous namespace "<unnamed>"; other demanglers
        # "(anonymous namespace)".
        unnamed = demangled.replace("(anonymous namespace)", "(anonymousnamespace)")
        args = template_args(unnamed.replace("<unnamed>", "(anonymousnamespace)"), head)
        if args is None:
            continue
        if len(args) != 2 or not all(re.fullmatch(r"\d+", a) for a in args):
            fail(f"unexpected {name} arguments in {demangled}")
        return (kind, int(args[0]), 0, 0, int(args[1]), 0, 0, 0, 0)
    return None


class Family:
    """One kernel family of the snapshot: how its symbols are told apart and keyed, the columns
    of its key (the first spelled as text where `text_key`, the others integers), the m16n8 tiles
    k_iter divides a warp's MMA work by, and whether its rows carry the pipe counts."""

    def __init__(self, name, key_columns, key_of, needle, tiles, pipes, text_key, loop=True):
        self.name = name
        # Whether the SASS pass counts its loop: a family whose price reads its resources alone
        # (the CTAs an SM holds) is a resources-only section, schema 1's columns.
        self.loop = loop
        self.key_columns = key_columns
        self.key_of = key_of
        self.needle = needle
        self.tiles = tiles
        self.pipes = pipes
        self.text_key = text_key
        self.row_head = ("family",) + key_columns + RESOURCE_COLUMNS

    def schema_columns(self):
        """{schema: (columns, loop fields)}: the resources alone (1), and the loop (4; with the
        pipe counts for a family that carries them)."""
        loop = sass_counts.COLUMNS + (sass_counts.PIPE_FIELDS if self.pipes else ())
        return {1: (self.row_head, 0), 4: (self.row_head + loop, len(loop))}

    def parse_key(self, fields):
        return tuple(v if i == 0 and self.text_key else int(v) for i, v in enumerate(fields))


# The families in the order a snapshot holds them: the register-MMA kernels first (the file's
# header and rows, as schema 4 has them), then one section each (`family <name>`, render_section,
# #785): ggml's matrix kernel and the W4A8 tiles, with the pipe counts. Their k_iter is the warp's
# MMA work per iteration over one m16n8 tile (m x n x k / 128): 32 J for an MMQ unit (one 128-row x
# J-column tile over 256 K on 8 warps), (block_m / warps_m) x (block_n / warps_n) for a W4A8 K
# tile of 128.
FAMILIES = {
    FAMILY: Family(FAMILY, KEY_COLUMNS, lambda d: rmma_key(d), "rmma_kernel", key_tiles, False,
                   True),
    "mmq": Family("mmq", ("type", "J", "fallback", "prec"), lambda d: mmq_key(d), "mul_mat_q",
                  lambda key: 1, True, False),
    "w4a8": Family("w4a8", ("codec", "block_m", "block_n", "warps_m", "warps_n", "epilogue",
                            "flavor", "stages", "producers"), lambda d: w4a8_key(d), "w4a8_",
                   lambda key: 1, True, True),
    "gemma4": Family("gemma4", ("kernel", "D", "G", "slice", "rows", "Bc", "partial", "fp8",
                                "chain"), lambda d: gemma4_key(d), "gemma4_", lambda key: 1,
                     False, True, loop=False),
}
SECTION_FAMILIES = tuple(f for f in FAMILIES if f != FAMILY)
GGML_Q2_K = 10  # the ggml_type value of Q2_K (third_party/ggml-quants/include/ggml.h)
MMQ_UNIT_K = 256  # MMQ_ITER_K (mmq.cuh): the K one iteration of the unit's loop covers
MMQ_WARPS = 8
MMQ_ROWS = 128


def extract_resources(binary, cuda_bin, dump=None, families=(FAMILY,)):
    """({family: {arch: {key: (registers, stack, local, shared)}}}, {arch: {mangled: (family,
    key)}}) of the binary's kernels of `families`. A kernel that appears in two images of one arch
    (a template instantiated in two units) must agree."""
    usage = resource_usage(binary, cuda_bin, dump)
    wanted = [FAMILIES[f] for f in families]
    picked = [u for u in usage if any(f.needle in u[1] for f in wanted)]
    names = demangle(sorted({name for _, name, _ in picked}), cuda_bin)
    rows, keys = {f: {} for f in families}, {}
    for arch, name, value in picked:
        for family in wanted:
            key = family.key_of(names[name]) if family.needle in name else None
            if key is None:
                continue
            per_arch = rows[family.name].setdefault(arch, {})
            if key in per_arch and per_arch[key] != value:
                fail(f"{name} appears twice in {arch} of {binary or dump} with different "
                     f"resources {per_arch[key]} and {value}")
            per_arch[key] = value
            keys.setdefault(arch, {})[name] = (family.name, key)
            break
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


def extract_counts(binary, cuda_bin, arch, keys, sass_dump=None, batch=SASS_BATCH):
    """{(family, key): LoopCounts} of the kernels `keys` ({mangled: (family, key)}) of one image,
    from its SASS. A kernel disassembled twice in the image (two units) must agree."""
    counts = {}
    lines = sass_lines(binary, cuda_bin, arch, sass_dump, sorted(keys), batch)
    for name, body in sass_counts.iter_functions(lines, lambda n: n in keys):
        fk = keys[name]
        family = FAMILIES[fk[0]]
        c = sass_counts.loop_counts(body, family.tiles(fk[1])).with_pipes(family.pipes)
        if fk in counts and counts[fk] != c:
            fail(f"{name} is disassembled twice in {arch} with different loops {counts[fk]} "
                 f"and {c}")
        counts[fk] = c
    return counts


def extract_families(binary, cuda_bin, arch, dump=None, sass_dump=None, families=(FAMILY,),
                     sass_families=(FAMILY,), batch=SASS_BATCH):
    """{family: {key: (registers, stack, local, shared, LoopCounts)}} of one image of the binary
    (`arch`: the label, "89") for `families`; the LoopCounts is None outside `sass_families`."""
    rows, keys = extract_resources(binary, cuda_bin, dump, families)
    image = image_arch({a: None for per in rows.values() for a in per}, arch)
    resources = {f: rows[f].get(image, {}) for f in families}
    wanted = {n: fk for n, fk in keys.get(image, {}).items() if fk[0] in sass_families}
    counts = extract_counts(binary, cuda_bin, image, wanted, sass_dump, batch) if wanted else {}
    out = {}
    for f in families:
        if f not in sass_families:
            out[f] = {k: (*v, None) for k, v in resources[f].items()}
            continue
        missing = sorted(k for k in resources[f] if (f, k) not in counts)
        if missing:
            fail(f"{len(missing)} {f} kernels of {image} have resources but no SASS, first "
                 f"{row_key(missing[0])}")
        out[f] = {k: (*v, counts[(f, k)]) for k, v in resources[f].items()}
    return out


def extract_rows(binary, cuda_bin, arch, dump=None, sass_dump=None, sass=True,
                 batch=SASS_BATCH):
    """{key: (registers, stack, local, shared, LoopCounts)} of one image's register-MMA kernels
    (`arch`: the label, "89"); the LoopCounts is None under sass=False."""
    return extract_families(binary, cuda_bin, arch, dump, sass_dump, (FAMILY,),
                            (FAMILY,) if sass else (), batch)[FAMILY]


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
    loops = [v[4] for v in rows.values() if v[4] is not None]
    unscheduled = sum(1 for c in loops if not c.scheduled)
    if unscheduled and unscheduled < len(loops):
        out.append(f"{unscheduled} of {len(loops)} kernels have no control codes")
    for k, v in sorted(rows.items()):
        if v[4] is not None and v[4].schedule_problems():
            out.append(f"{row_key(k)}: {'; '.join(v[4].schedule_problems())}")
            break
    return out


# The most instructions an MMQ kernel ggml_mmq_internal.cuh compiles empty may hold (family_audit).
MMQ_EMPTY_INSTRUCTIONS = 64


def family_audit(family, rows):
    """loop_audit for a section family's rows (#785): lines naming the kernels whose counted
    iteration is not the unit the family's price reads. ggml's matrix kernel: one or two
    candidate loops (its stream-k body holds the unit's loop twice, with and without the fixup
    write; the heavier is counted), four CTA barriers per iteration (two y-tile halves, each
    loaded, synchronised, multiplied, synchronised), and k_iter 32 J (MMQ_ROWS x J outputs on
    MMQ_WARPS warps over MMQ_UNIT_K values, per m16n8 tile); a kernel without a loop only at the
    80- and 96-column widths, which ggml_mmq_internal.cuh compiles empty where the generation
    does not take them. The W4A8 tiles: one loop, k_iter (block_m / warps_m) x (block_n /
    warps_n), and two CTA barriers per K tile (the classic tile's __syncthreads pair; the
    producer-warp tile's consumers wait on a full stage and mark it empty). Q2_K's k_iter is
    above its unit's in both families (5 / 4 of it: its min term, a scale and a min per 16
    values, takes MMAs of its own that advance no K, which the ones-register search of
    sass_counts.py does not tell apart)."""
    out = []
    for k, v in sorted(rows.items()):
        c = v[4]
        if c is None:
            continue
        problems = []
        if family == "mmq":
            J = k[1]
            if c.loops == 0:
                # Only the widths compiled empty may lack the loop; any other is a failed audit.
                # An empty kernel is its guard and exit: no CTA barrier and a few dozen
                # instructions (25-31 on sm_80, sm_90, sm_100), where a matrix kernel whose loop
                # went unfound holds thousands and its barriers outside the loop (#827). None
                # counted at all is a flow sass_counts could not follow, not an empty kernel.
                if J not in (80, 96):
                    out.append(f"{family} {row_key(k)}: no K loop")
                elif c.scheduled and (c.outside_bar or
                                      not 0 < c.outside_total <= MMQ_EMPTY_INSTRUCTIONS):
                    out.append(f"{family} {row_key(k)}: no K loop, and not empty "
                               f"({c.outside_total} instructions, {c.outside_bar} barriers)")
                continue
            if c.loops > 2:
                problems.append(f"{c.loops} candidate loops")
            if c.bar != 4:
                problems.append(f"{c.bar} barriers in the iteration")
            want = MMQ_ROWS * J * MMQ_UNIT_K // (MMQ_WARPS * 128)
            if c.k_iter < want or (k[0] != GGML_Q2_K and c.k_iter != want):
                problems.append(f"k_iter {c.k_iter}, not {want}")
        elif family == "w4a8":
            _, bm, bn, wm, wn = k[:5]
            want = (bm // wm) * (bn // wn)
            if c.loops != 1:
                problems.append(f"{c.loops} candidate loops")
            if c.k_iter < want or (k[0] != "W4a8Q2K" and c.k_iter != want):
                problems.append(f"k_iter {c.k_iter}, not {want}")
            if c.bar != 2:
                problems.append(f"{c.bar} barriers in the iteration")
        problems += c.schedule_problems()
        if problems:
            out.append(f"{family} {row_key(k)}: {'; '.join(problems)}")
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


def render(arch, rows, nvcc, flags, source):
    """The snapshot's text: schema 4 where every row carries its loop and schedule; the resources
    alone where there is no loop. A row with a loop but no schedule is an error."""
    loops = [v[4] for v in rows.values() if v[4] is not None]
    if not loops:
        schema = 1
    elif all(c.scheduled for c in loops):
        schema = SCHEMA
    else:
        fail("a row has no control codes: only schema 4 (loop and schedule) is written")
    columns = SCHEMA_COLUMNS[schema][0]
    lines = [
        f"# NInferno kernel facts, schema {schema}: what ptxas made of the register-MMA kernels of",
        f"# the sm_{arch} build, read off the linked binary with cuobjdump. Per kernel: the registers",
        "# a thread keeps, its stack frame and local bytes, the CTA's static shared memory",
        "# (--dump-resource-usage); then the K loop's instructions per iteration (-sass; one",
        "# iteration is two K steps, k_iter K values per warp): the candidate loops found, the",
        "# MMAs, of them the min-term ones (A = 1.0 pairs) and the MMA shape, the loads and stores",
        "# by space with the bytes a warp requests (32 lanes x width; LDG global, LD generic, LDGSTS",
        "# async copy, LDS/STS shared, LDL/STL local), SHFL, integer and float ALU, uniform-datapath,",
        "# the rest, the total; then the static schedule from the control codes ptxas wrote: the",
        "# stall counts summed over the iteration (the clocks one warp alone takes to issue it), its",
        "# instructions that wait on a scoreboard and of them those waiting on a local load, and the",
        "# code outside the K loop (the instructions the entry reaches beyond the loop, their stall",
        "# sum, scoreboard waits and CTA barriers). Static counts of one full iteration; how the",
        "# loop is found and the control code read is in tools/kernel_facts/sass_counts.py.",
        "# Generated by tools/kernel_facts/kernel_facts.py; never edited by hand. The selector's",
        "# price reads this file, not the host GPU's kernels (ops/generic/kernel_facts.h, #612), and",
        "# ninferno_kernel_facts_test asserts that a build's kernels equal it. After a kernel or",
        "# toolchain change regenerate it with the build's own binary:",
        f"#   python3 tools/kernel_facts/kernel_facts.py extract --binary <linked exe> --arch {arch}",
        f"#     --nvcc <version> -o src/ops/generic/data/kernel_facts/sm_{arch}.txt",
        f"arch {arch}",
        f"nvcc {nvcc}",
        f"flags {flags}",
        f"source {source}",
        f"kernels {len(rows)}",
        f"columns {' '.join(columns)}",
    ]
    lines += [row_line(k, rows[k]) for k in sorted(rows)]
    return "\n".join(lines) + "\n"


# What each section family's comment says it holds (render_section).
SECTION_NOTES = {
    "mmq": (
        "# ggml's matrix kernel, mul_mat_q<type, J, fallback, prec> (third_party/ggml-quants",
        "# mmq.cuh, built by src/ops/generic/ggml_mmq; key: the ggml_type and ggml_prec values, the",
        "# column tile J, the fallback flag). One iteration of the counted loop is one unit: a",
        "# 128-row x J-column tile over 256 K values (the unit core/generation.h MmqLaunchLaw prices);",
        "# the stream-k body holds it twice (with and without the fixup write), the heavier is",
        "# counted. The 80- and 96-column kernels are empty (loops 0) where the generation does not",
        "# take them.",
    ),
    "w4a8": (
        "# The W4A8 tiles (src/ops/generic/generic_w4a8.cuh): w4a8_kernel<codec, schedule, epilogue,",
        "# flavor> (stages and producers 0) and the producer-warp tile w4a8_ws_kernel<codec,",
        "# schedule, epilogue, stages, producers> (flavor 0; its MMA warps' loop is counted, the",
        "# producers' decode loop is not). One iteration is one 128-value K tile.",
    ),
    "gemma4": (
        "# The Gemma 4 attention kernels (src/ops/softmax_attention/gemma4, #830):",
        "# gemma4_attention_kernel<Gemma4Geometry<D, G, slice>, WR, Bc, Partial, Fp8, Chain> (kernel",
        "# `attention`, rows = WR) and the Volta decode kernels gemma4_volta_decode_kernel<D, MaxRows>",
        "# and gemma4_volta_decode_tiled_kernel<D, MaxRows> (`volta`, `volta_tiled`, rows = MaxRows,",
        "# the other columns 0). Resources alone: the decode plan reads the CTAs an SM holds",
        "# (core/kernel_occupancy.h), never the host's kernels.",
    ),
}


def render_section(family, rows, nvcc):
    """The text of one section family's rows (#785), appended after the register-MMA rows: a
    `family <name>` line, its own nvcc, kernel count and columns (family_nvcc, family_kernels,
    family_columns: so the file's own nvcc, kernels and columns lines stay the register-MMA ones
    every reader and script takes), the rows. The loop columns are schema 4's and the pipe counts
    (sass_counts.PIPE_FIELDS), or the resources alone where no row has a loop."""
    f = FAMILIES[family]
    loops = [v[4] for v in rows.values() if v[4] is not None]
    if loops and not all(c.scheduled for c in loops):
        fail(f"a {family} row has no control codes: only the loop and its schedule are written")
    columns = f.schema_columns()[4 if loops else 1][0]
    loop_notes = (
        "# The columns: the key, the resources, the loop's counts and schedule as above, then the",
        "# pipe counts of the same iteration: I2F, I2FP, FFMA, FMUL, FADD, IMAD, BAR (sass_counts.py).",
        "# k_iter is the warp's MMA work per iteration over one m16n8 tile (m x n x k / 128).",
    ) if f.loop else ("# The columns: the key, then the resources as above.",)
    lines = [
        f"family {family}",
        *SECTION_NOTES[family],
        *loop_notes,
        f"family_nvcc {nvcc}",
        f"family_kernels {len(rows)}",
        f"family_columns {' '.join(columns)}",
    ]
    lines += [" ".join(str(v) for v in [family, *k, *rows[k][:4]] +
                       (rows[k][4].row() if rows[k][4] is not None else [])) for k in sorted(rows)]
    return "\n".join(lines) + "\n"


def split_sections(text):
    """(the register-MMA part, {family: its section's text}) of a snapshot's text: each section
    runs from its `family` line to the next one or the end."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, l in enumerate(lines) if l.startswith("family ")]
    top = "".join(lines[:starts[0]] if starts else lines)
    sections = {}
    for n, s in enumerate(starts):
        name = lines[s].split()[1] if len(lines[s].split()) > 1 else ""
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        if name not in SECTION_FAMILIES or name in sections:
            fail(f"unexpected or repeated section: {lines[s].strip()}")
        sections[name] = "".join(lines[s:end])
    return top, sections


def parse_rows(family, lines, columns, what):
    """{key: (registers, stack, local, shared, LoopCounts or None)} of one family's row lines
    under its columns line; fails on any other columns, a row of another width, a malformed field
    or a schedule no kernel has (LoopCounts.schedule_problems)."""
    f = FAMILIES[family]
    columns = tuple(columns.split())
    schemas = f.schema_columns() if family != FAMILY else SCHEMA_COLUMNS
    known = [s for s, (c, _) in schemas.items() if c == columns]
    if not known:
        fail(f"unexpected {what}: {' '.join(columns)}")
    width = len(columns)
    keys = len(f.key_columns)
    head = len(f.row_head)
    loop_width = schemas[known[0]][1]
    rows = {}
    for line in lines:
        fields = line.split()
        if len(fields) != width:
            fail(f"a row of {len(fields)} fields under {width} columns: {line}")
        try:
            key = f.parse_key(fields[1:1 + keys])
            loop = fields[head:head + loop_width]
            counts = sass_counts.LoopCounts.from_row(loop) if loop_width else None
            if counts is not None and counts.pipes != f.pipes:
                raise ValueError("pipe columns of another family")
            rows[key] = (*(int(v) for v in fields[1 + keys:head]), counts)
        except ValueError as error:
            fail(f"malformed row ({error}): {line}")
        if counts is not None and counts.schedule_problems():
            fail(f"a schedule no kernel has ({'; '.join(counts.schedule_problems())}): {line}")
    return rows


def parse_sections(text):
    """(header, {family: {"nvcc", "kernels", "rows"}}) of a snapshot: the register-MMA part (the
    file's header lines and `rmma` rows, schema 4 or the resources-only schema 1) under FAMILY,
    then every section (render_section). Any other columns line fails: regenerate the file."""
    top, section_texts = split_sections(text)
    header, rmma_lines = {}, []
    for line in top.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == FAMILY:
            rmma_lines.append(line)
        else:
            header[fields[0]] = line[len(fields[0]) + 1:] if len(fields) > 1 else ""
    for want in ("arch", "nvcc", "kernels", "columns"):
        if want not in header:
            fail(f"snapshot has no '{want}' line")
    # Every row spells the header's columns, those of one schema, never a mixture.
    rows = parse_rows(FAMILY, rmma_lines, header["columns"], "columns")
    if int(header["kernels"]) != len(rows):
        fail(f"snapshot says {header['kernels']} kernels, holds {len(rows)}")
    sections = {FAMILY: {"nvcc": header["nvcc"], "kernels": len(rows), "rows": rows}}
    for family, section in section_texts.items():
        own, row_lines = {}, []
        for line in section.splitlines()[1:]:
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == family:
                row_lines.append(line)
            elif fields[0] in ("family_nvcc", "family_kernels", "family_columns"):
                own[fields[0]] = line[len(fields[0]) + 1:] if len(fields) > 1 else ""
            else:
                fail(f"unexpected line in the {family} section: {line}")
        for want in ("family_nvcc", "family_kernels", "family_columns"):
            if want not in own:
                fail(f"the {family} section has no '{want}' line")
        family_rows = parse_rows(family, row_lines, own["family_columns"], "family_columns")
        if int(own["family_kernels"]) != len(family_rows):
            fail(f"the {family} section says {own['family_kernels']} kernels, holds "
                 f"{len(family_rows)}")
        sections[family] = {"nvcc": own["family_nvcc"], "kernels": len(family_rows),
                            "rows": family_rows}
    return header, sections


def parse_snapshot(text):
    """(header, {key: (registers, stack, local, shared, LoopCounts or None)}) of a snapshot's
    register-MMA kernels, schema 4, or a schema 1 file (resources only), which parses with None
    for the loop; its sections (parse_sections) are parsed and checked too. Any other columns
    line fails: regenerate the file. A schedule no kernel can have
    (LoopCounts.schedule_problems) fails."""
    header, sections = parse_sections(text)
    return header, sections[FAMILY]["rows"]


def image_arch(rows, arch):
    """The binary's image for the arch label: sm_<label>; a family target's image is sm_120f."""
    name = "sm_" + arch
    if name in rows:
        return name
    fail(f"the binary holds no sm_{arch} image, only {', '.join(sorted(rows)) or 'none'}")


def family_list(text):
    """The --families argument: FAMILIES names, comma-separated, in the snapshot's order."""
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in FAMILIES]
    if unknown or not names:
        fail(f"unknown families {', '.join(unknown) or '(none)'}: {', '.join(FAMILIES)}")
    return tuple(f for f in FAMILIES if f in names)


def cmd_extract(args):
    families = family_list(args.families)
    sass = () if args.no_sass else tuple(f for f in families if FAMILIES[f].loop)
    extracted = extract_families(args.binary, args.cuda_bin, args.arch, args.dump, args.sass_dump,
                                 families, sass, args.sass_batch)
    if FAMILY in families and not extracted[FAMILY]:
        fail("no rmma_kernel in the binary")
    for f in families:
        if f != FAMILY and not extracted[f]:
            print(f"kernel_facts: no {f} kernel in the binary: no {f} section", file=sys.stderr)
    families = tuple(f for f in families if f == FAMILY or extracted[f])
    audit = loop_audit(extracted[FAMILY]) if FAMILY in families else []
    for f in families:
        if f != FAMILY:
            audit += family_audit(f, extracted[f])
    for line in audit:
        print(f"kernel_facts: {line}", file=sys.stderr)
    nvcc = args.nvcc or "unknown"
    # --update: the families extracted replace their part of the file, the rest is kept byte for
    # byte (a section family added to a snapshot leaves its register-MMA rows as they were).
    top, sections = "", {}
    if args.update:
        with open(args.update) as f:
            old = f.read()
        parse_sections(old)
        top, sections = split_sections(old)
    if FAMILY in families:
        source = args.source or (source_hash(args.source_root) if args.source_root else "unknown")
        top = render(args.arch, extracted[FAMILY], nvcc, args.flags or "unknown", source)
    elif not args.update:
        fail("without the rmma family, extract writes into a snapshot: pass --update FILE")
    for f in families:
        if f != FAMILY:
            sections[f] = render_section(f, extracted[f], nvcc)
    text = top + "".join(sections[f] for f in SECTION_FAMILIES if f in sections)
    if args.update:
        header, _ = parse_sections(text)
        if header["arch"] != args.arch:
            fail(f"{args.update} is the snapshot of sm_{header['arch']}, not sm_{args.arch}")
    output = args.output or args.update
    counts = ", ".join(f"{len(extracted[f])} {f}" for f in families)
    if output:
        with open(output, "w", newline="\n") as f:
            f.write(text)
        print(f"kernel_facts: wrote {counts} kernels of sm_{args.arch} to {output}")
    else:
        sys.stdout.write(text)
    if audit:
        # Written for inspection, but not a snapshot to commit: the loop finder did not settle.
        print(f"kernel_facts: {len(audit)} loop audit findings (above): the counts are not "
              "settled, do not commit them", file=sys.stderr)
        return 1
    return 0


def elf_section(path, name):
    """The bytes of an ELF64 file's section `name`, or None (not ELF64, or no such section)."""
    found = elf_sections(path, lambda n: n == name.encode())
    return found[0][1] if found else None


def device_code(path):
    """Every fatbin section of an ELF64 binary, name and bytes in file order: .nv_fatbin (the
    linked image) and, under separable compilation, any __nv_relfatbin the objects left; not
    .nvFatBinSegment, which holds the images' addresses. None when there is none."""
    found = elf_sections(path, lambda n: b"fatbin" in n.lower() and n != b".nvFatBinSegment")
    return b"".join(n + b"\0" + d for n, d in found) if found else None


def elf_sections(path, wanted):
    """[(name, bytes)] of an ELF64 file's sections whose name `wanted` takes, in file order; []
    when the file is not little-endian ELF64."""
    out = []
    with open(path, "rb") as f:
        head = f.read(64)
        if len(head) < 64 or head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
            return out
        shoff, = struct.unpack_from("<Q", head, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", head, 0x3A)
        if shentsize < 64 or shnum == 0 or shstrndx >= shnum:
            return out

        def header(i):
            f.seek(shoff + i * shentsize)
            raw = f.read(64)
            if len(raw) < 64:
                return None
            sh_name, _, _, _, offset, size = struct.unpack_from("<IIQQQQ", raw)
            return sh_name, offset, size

        names = header(shstrndx)
        if names is None:
            return out
        f.seek(names[1])
        table = f.read(names[2])
        for i in range(shnum):
            h = header(i)
            if h is None:
                return []
            end = table.find(b"\0", h[0])
            if end >= 0 and wanted(table[h[0]:end]):
                f.seek(h[1])
                out.append((table[h[0]:end], f.read(h[2])))
    return out


def check_key(args):
    """What a passed check rests on: the binary's device code (its fatbin sections, device_code:
    the part cuobjdump reads, which stripping the binary or setting its run path leaves as it
    was; the whole file where there is none) or the saved dumps, the snapshot, the nvcc named, the options
    and this checker's own source. A stamp holding it says the check passed on exactly these.
    None when an input is missing (the check then says what is)."""
    h = hashlib.sha256()
    try:
        for path in (args.binary, args.dump, args.sass_dump, args.expected, __file__,
                     sass_counts.__file__):
            h.update(b"\0" + str(bool(path)).encode())
            if not path:
                continue
            device = device_code(path) if path == args.binary else None
            if device is not None:
                h.update(b"device code\0" + device)
                continue
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
    except OSError:
        return None
    h.update(repr((args.nvcc, args.no_sass, args.cuda_bin, args.sass_batch)).encode())
    return h.hexdigest()


def cmd_check(args):
    # --stamp: a resumed release run re-checks every engine it already packed (minutes each);
    # one checked before on exactly these inputs passes again without the SASS pass.
    key = check_key(args) if args.stamp else None
    if key:
        try:
            with open(args.stamp) as f:
                if f.read().strip() == key:
                    print(f"kernel_facts: {args.binary or args.dump} unchanged since it equalled "
                          f"{args.expected} ({args.stamp})")
                    return 0
        except OSError:
            pass
    status = check_once(args)
    if key and status == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.stamp)), exist_ok=True)
        with open(args.stamp, "w") as f:
            f.write(key + "\n")
    return status


def check_once(args):
    if not os.path.exists(args.expected):
        print(f"kernel_facts: no snapshot {args.expected} for this build architecture: generate "
              "it with\n  python3 tools/kernel_facts/kernel_facts.py extract --binary "
              f"{args.binary or args.dump} --arch <label> --nvcc {args.nvcc or '<version>'} "
              f"-o {args.expected}")
        return 1
    with open(args.expected) as f:
        header, sections = parse_sections(f.read())
    arch = header["arch"]
    # Per family (the register-MMA rows, then each section the snapshot has): a section without
    # loop columns is checked on its resources alone; under another nvcc than the one it records
    # the kernel set is all that is compared, so the SASS pass (minutes) is not run for it.
    families = tuple(f for f in FAMILIES if f in sections)
    other = {f: bool(args.nvcc) and args.nvcc != sections[f]["nvcc"] for f in families}
    with_sass = tuple(f for f in families if not args.no_sass and not other[f] and
                      any(v[4] is not None for v in sections[f]["rows"].values()))
    # The families the snapshot has no section for are read for their count alone.
    absent = tuple(f for f in FAMILIES if f not in sections)
    extracted = extract_families(args.binary, args.cuda_bin, arch, args.dump, args.sass_dump,
                                 families + absent, with_sass, args.sass_batch)
    for f in absent:
        if extracted[f]:
            print(f"kernel_facts: {args.expected} has no {f} section: the build's "
                  f"{len(extracted[f])} {f} kernels are not checked")
    status = 0
    for f in families[1:]:
        status = max(status, check_section(args, f, sections[f], extracted[f], other[f],
                                           f in with_sass, arch))
    expected = sections[FAMILY]["rows"]
    got = extracted[FAMILY]
    with_sass = FAMILY in with_sass
    other_nvcc = other[FAMILY]
    if not with_sass:
        expected = {k: (*v[:4], None) for k, v in expected.items()}
    missing = sorted(k for k in expected if k not in got)
    extra = sorted(k for k in got if k not in expected)
    rmma = compare_rmma(args, header, arch, expected, got, missing, extra, other_nvcc,
                        with_sass)
    return 1 if status or rmma == 1 else rmma


def compare_rmma(args, header, arch, expected, got, missing, extra, other_nvcc, with_sass):
    """check's verdict on the register-MMA rows: 0 equal, 77 another nvcc with the same kernels,
    1 different (the kernels listed)."""
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
    list_differences(args, differ, missing, extra, expected, got, row_line)
    binary = args.binary or args.dump
    print(f"kernel_facts: the sm_{arch} kernels of {binary} differ from {args.expected}: "
          f"{len(differ)} changed, {len(missing)} missing, {len(extra)} new (nvcc "
          f"{args.nvcc or 'unknown'}, snapshot {header['nvcc']}). A kernel, flag or compiler "
          "change moved them: regenerate the snapshot with\n"
          f"  python3 tools/kernel_facts/kernel_facts.py extract --binary {binary} "
          f"--arch {arch} --nvcc <version> -o {args.expected}\n"
          "and review the selector goldens that move with it.")
    return 1


def list_differences(args, differ, missing, extra, expected, got, line):
    """The differing, missing and new rows, at most args.show of them."""
    shown = 0
    for k in differ:
        if shown < args.show:
            print(f"  {line(k, expected[k])}\n    build: {line(k, got[k])}")
        shown += 1
    for k in missing:
        if shown < args.show:
            print(f"  missing in the build: {line(k, expected[k])}")
        shown += 1
    for k in extra:
        if shown < args.show:
            print(f"  not in the snapshot: {line(k, got[k])}")
        shown += 1
    if shown > args.show:
        print(f"  ... {shown - args.show} more")


def section_line(family):
    """A section row's text, as render_section writes it."""
    return lambda k, v: " ".join(str(x) for x in [family, *k, *v[:4]] +
                                 (v[4].row() if v[4] is not None else []))


def check_section(args, family, section, got, other_nvcc, with_sass, arch):
    """check's verdict on one section family (#785): 0 where its kernels equal the section (their
    set alone under another nvcc than the section records, with a line saying so), 1 where they
    differ (listed, with the command that regenerates the section)."""
    expected = section["rows"]
    if not with_sass:
        expected = {k: (*v[:4], None) for k, v in expected.items()}
        got = {k: (*v[:4], None) for k, v in got.items()}
    missing = sorted(k for k in expected if k not in got)
    extra = sorted(k for k in got if k not in expected)
    differ = [] if other_nvcc else sorted(
        k for k in expected if k in got and expected[k] != got[k])
    if not (missing or extra or differ):
        if other_nvcc:
            print(f"kernel_facts: {len(got)} {family} kernels of sm_{arch}: the set equals "
                  f"{args.expected}; SKIPPED their resources and loop counts (this build's nvcc "
                  f"{args.nvcc}, the section's {section['nvcc']})")
        else:
            print(f"kernel_facts: {len(got)} {family} kernels of sm_{arch} equal {args.expected}"
                  + ("" if with_sass else " (resources only)"))
        return 0
    list_differences(args, differ, missing, extra, expected, got, section_line(family))
    binary = args.binary or args.dump
    print(f"kernel_facts: the sm_{arch} {family} kernels of {binary} differ from "
          f"{args.expected}: {len(differ)} changed, {len(missing)} missing, {len(extra)} new "
          f"(nvcc {args.nvcc or 'unknown'}, section {section['nvcc']}). Regenerate the section "
          f"with\n  python3 tools/kernel_facts/kernel_facts.py extract --binary {binary} "
          f"--arch {arch} --nvcc <version> --families {family} --update {args.expected}")
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
    e = sub.choices["extract"]
    e.add_argument("--arch", required=True, help="the build architecture label (89, 120a, ...)")
    e.add_argument("--flags", help="the CUDA flags of the build, recorded in the header")
    e.add_argument("--source-root", help="the source tree, for the header's source hash")
    e.add_argument("--source", help="the source hash itself (from `hash`), in place of a tree")
    e.add_argument("--families", default=",".join(FAMILIES),
                   help=f"the kernel families to extract (default: all, {','.join(FAMILIES)})")
    e.add_argument("--update", help="a committed snapshot: the families extracted replace their "
                                    "part of it, the rest is kept byte for byte (written back "
                                    "unless -o names another file)")
    e.add_argument("-o", "--output")
    c = sub.choices["check"]
    c.add_argument("--expected", required=True, help="the committed snapshot")
    c.add_argument("--show", type=int, default=40, help="differing kernels to list")
    c.add_argument("--stamp", help="a file recording a passed check: a check on the same inputs "
                                   "passes again without extracting (written on a pass)")
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
