#!/usr/bin/env python3
"""The kernel-facts snapshot of one build architecture (#612): what ptxas made of the generic
path's register-MMA kernels, read off the linked binary with `cuobjdump --dump-resource-usage`
and written as committed data (src/ops/generic/data/kernel_facts/sm_<arch>.txt). No GPU is
involved: the selector's price reads these files (ops/generic/kernel_facts.h), never the host
GPU's kernels, so a decision is the same on every machine, whatever nvcc or driver.

    kernel_facts.py extract --binary BIN --arch 89 --nvcc 13.4.92 [--flags TEXT]
                            [--source-root DIR] [-o FILE]
    kernel_facts.py check   --binary BIN --expected FILE [--nvcc VER]
    kernel_facts.py hash    --source-root DIR

`extract` writes the snapshot (stdout without -o). `check` extracts the same from BIN and compares
it with FILE row by row: exit 0 where they agree, 1 where they differ (every differing kernel is
named, with the line to regenerate), 77 where BIN was compiled by another nvcc than FILE records
and holds the same kernels (ptxas then allocates other registers by construction; the price still
reads FILE, so the decisions stay the release's), 3 on a usage or tool error. A kernel added or
removed fails under any nvcc.

The binary is the linked executable or shared library (any program that links ninferno_ops):
under relocatable device code the per-unit objects report no shared memory and a smaller stack
for a kernel, the device link decides them. `cuobjdump` and `cu++filt` come from the CUDA
toolkit's bin directory (--cuda-bin, else the directory of `nvcc` on PATH, else CUDA_PATH).

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
import os
import re
import shutil
import subprocess
import sys

SCHEMA = 1
NAMESPACE = "ninferno::ops::detail::generic::"
FAMILY = "rmma"
COLUMNS = ("family decoder warps_m warps_k m_tiles n_tiles cluster epilogue latency add_sums "
           "registers stack local shared")
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


def extract_rows(binary, cuda_bin, dump=None):
    """{arch: {key: (registers, stack, local, shared)}} of the binary's rmma kernels. A kernel
    that appears in two images of one arch (a template instantiated in two units) must agree."""
    usage = resource_usage(binary, cuda_bin, dump)
    rmma = [u for u in usage if "rmma_kernel" in u[1]]
    names = demangle(sorted({name for _, name, _ in rmma}), cuda_bin)
    rows = {}
    for arch, name, value in rmma:
        key = rmma_key(names[name])
        if key is None:
            continue
        per_arch = rows.setdefault(arch, {})
        if key in per_arch and per_arch[key] != value:
            fail(f"{name} appears twice in {arch} of {binary or dump} with different resources "
                 f"{per_arch[key]} and {value}")
        per_arch[key] = value
    return rows


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


def row_line(key, value):
    return " ".join(str(v) for v in (FAMILY, *key, *value))


def render(arch, rows, nvcc, flags, source):
    lines = [
        f"# NInferno kernel facts, schema {SCHEMA}: what ptxas made of the register-MMA kernels of",
        f"# the sm_{arch} build (registers a thread keeps, its stack frame and local bytes, the CTA's",
        "# static shared memory), read off the linked binary with cuobjdump --dump-resource-usage.",
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
        f"columns {COLUMNS}",
    ]
    lines += [row_line(k, rows[k]) for k in sorted(rows)]
    return "\n".join(lines) + "\n"


def parse_snapshot(text):
    header, rows = {}, {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == FAMILY:
            if len(fields) != 14:
                fail(f"malformed row: {line}")
            key = (fields[1], *(int(v) for v in fields[2:10]))
            rows[key] = tuple(int(v) for v in fields[10:14])
        else:
            header[fields[0]] = line[len(fields[0]) + 1:] if len(fields) > 1 else ""
    for want in ("arch", "nvcc", "kernels"):
        if want not in header:
            fail(f"snapshot has no '{want}' line")
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
    rows = extract_rows(args.binary, args.cuda_bin, args.dump)
    per_arch = rows[image_arch(rows, args.arch)]
    if not per_arch:
        fail("no rmma_kernel in the binary")
    source = args.source or (source_hash(args.source_root) if args.source_root else "unknown")
    text = render(args.arch, per_arch, args.nvcc or "unknown", args.flags or "unknown", source)
    if args.output:
        with open(args.output, "w", newline="\n") as f:
            f.write(text)
        print(f"kernel_facts: wrote {len(per_arch)} kernels of sm_{args.arch} to {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_check(args):
    if not os.path.exists(args.expected):
        print(f"kernel_facts: no snapshot {args.expected} for this build architecture: generate "
              "it with\n  python3 tools/kernel_facts/kernel_facts.py extract --binary "
              f"{args.binary} --arch <label> --nvcc {args.nvcc or '<version>'} -o {args.expected}")
        return 1
    with open(args.expected) as f:
        header, expected = parse_snapshot(f.read())
    arch = header["arch"]
    rows = extract_rows(args.binary, args.cuda_bin, args.dump)
    got = rows[image_arch(rows, arch)]
    missing = sorted(k for k in expected if k not in got)
    extra = sorted(k for k in got if k not in expected)
    other_nvcc = args.nvcc and args.nvcc != header["nvcc"]
    if other_nvcc and not (missing or extra):
        # The set of kernels does not depend on the compiler; their registers do.
        print(f"kernel_facts: SKIPPED the resources: this build's nvcc {args.nvcc} is not the "
              f"snapshot's {header['nvcc']} ({args.expected}); ptxas then allocates other "
              "registers by construction. The kernel set agrees. The price reads the snapshot "
              "either way; regenerate it only on the release toolchain.")
        return 77
    differ = [] if other_nvcc else sorted(
        k for k in expected if k in got and expected[k] != got[k])
    if not (missing or extra or differ):
        print(f"kernel_facts: {len(got)} kernels of sm_{arch} equal {args.expected}")
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
    print(f"kernel_facts: the sm_{arch} kernels of {args.binary} differ from {args.expected}: "
          f"{len(differ)} changed, {len(missing)} missing, {len(extra)} new (nvcc "
          f"{args.nvcc or 'unknown'}, snapshot {header['nvcc']}). A kernel, flag or compiler "
          "change moved them: regenerate the snapshot with\n"
          f"  python3 tools/kernel_facts/kernel_facts.py extract --binary {args.binary} "
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
        p.add_argument("--cuda-bin", help="directory of cuobjdump and cu++filt")
        p.add_argument("--nvcc", help="the nvcc version the binary was built with")
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
    if args.command == "hash":
        print(source_hash(args.source_root))
        return 0
    return cmd_extract(args) if args.command == "extract" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
