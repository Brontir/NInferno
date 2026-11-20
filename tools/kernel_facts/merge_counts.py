#!/usr/bin/env python3
"""The natural merge's SASS counts against the constants its chunk law prices with (#872).

    merge_counts.py check --binary BIN --arch 89 [--cuda-bin DIR] [--header FILE]
    merge_counts.py count --sass-dump FILE

src/ops/softmax_attention/common/causal_merge_shape.h prices the merge's column chunk with two counts
read off `cuobjdump -sass` by hand (#867): the fixed work a warp of a CTA runs whatever its chunk
(kCausalMergeFixedWarpInstructions) and the work of each further stage
(kCausalMergeStageWarpInstructions). The header names the instances and the rule; this tool applies
the rule to the build's own binary, so a compiler bump or a kernel edit that moves the counts past
what the header prices fails a test instead of passing unseen. Only speed is at stake, never bits.

The instances: causal_natural_merge_kernel<CausalGeometry<256, 16, 4>, D, false, false, false,
false> at D = 64 and 256 (the decode-token form). In each, with the 128-bit global loads (LDG...128)
numbered from one:
- fixed: the instructions from the entry to the twelfth load (the end of the first stage's
  refetch), plus those from the last barrier to the final EXIT;
- stage: from the first barrier after the twelfth load to the sixteenth load.

`check` passes where every instance's fixed count is at most the header's (the header takes the
largest architecture's count, so a board is never priced as narrowing more cheaply than its SASS
says) and its stage count lies within STAGE_TOLERANCE of the header's (a rounded figure): exit 0,
1 where a count leaves its bound, an instance is missing or its SASS no longer has the shape the
rule counts, 3 on a tool error. `count` prints the
counts of a saved dump.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_facts  # noqa: E402
import sass_counts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
HEADER = os.path.join(ROOT, "src", "ops", "softmax_attention", "common", "causal_merge_shape.h")
# The header rounds the per-arch stage counts (65-74) to 70: a count may lie this far either side.
STAGE_TOLERANCE = 0.10
# The instance by its mangled name (the demanglers spell template arguments differently: cu++filt
# writes (int)256 and (bool)0): causal_natural_merge_kernel<CausalGeometry<256, 16, 4>, D, false,
# false, false, false>.
INSTANCE = re.compile(r"causal_natural_merge_kernelI\w*?CausalGeometryILi256ELi16ELi4EEELi(\d+)"
                      r"ELb0ELb0ELb0ELb0EE")
WIDTHS = (64, 256)


def header_constant(text, name):
    m = re.search(r"inline constexpr std::int64_t " + name + r"\s*=\s*(\d+)\s*;", text)
    if not m:
        kernel_facts.fail(f"{HEADER}: no {name}")
    return int(m.group(1))


def counts(body):
    """{fixed, stage} of one function body (sass_counts.iter_functions), None where its shape is
    not the merge's (fewer than 16 wide loads, no barrier, no EXIT after the last barrier)."""
    ops = [sass_counts.base_opcode(line[2]) for line in body]
    full = [line[2] for line in body]
    # LDG alone: LDGSTS (cp.async) is not a global load into registers.
    loads = [i for i, op in enumerate(ops) if op == "LDG" and ".128" in full[i]]
    bars = [i for i, op in enumerate(ops) if op == "BAR"]
    exits = [i for i, op in enumerate(ops) if op == "EXIT"]
    if len(loads) < 16 or not bars or not exits or exits[-1] < bars[-1]:
        return None
    fixed = (loads[11] + 1) + (exits[-1] - bars[-1] + 1)
    after = [b for b in bars if b > loads[11]]
    if not after or after[0] >= loads[15]:
        return None
    return {"fixed": fixed, "stage": loads[15] - after[0] + 1}


def instance_counts(lines, wanted):
    """{D: counts} of the functions `wanted` ({mangled: D}) in SASS text."""
    out = {}
    for name, body in sass_counts.iter_functions(lines, lambda n: n in wanted):
        c = counts(body)
        if c is None:
            kernel_facts.fail(f"{name}: not the merge's shape (16 wide loads, a barrier, an EXIT "
                              "after it): recount by hand", code=1)
        d = wanted[name]
        if d in out and out[d] != c:
            kernel_facts.fail(f"the D = {d} merge is disassembled twice with counts {out[d]} and {c}",
                              code=1)
        out[d] = c
    return out


def check(a):
    with open(a.header, encoding="utf-8") as f:
        text = f.read()
    fixed_max = header_constant(text, "kCausalMergeFixedWarpInstructions")
    stage = header_constant(text, "kCausalMergeStageWarpInstructions")
    usage = kernel_facts.resource_usage(a.binary, a.cuda_bin)
    image = kernel_facts.image_arch({arch: None for arch, _, _ in usage}, a.arch)
    names = sorted({n for arch, n, _ in usage if arch == image and "causal_natural_merge_kernel" in n})
    wanted = {}
    for n in names:
        m = INSTANCE.search(n)
        if m and int(m.group(1)) in WIDTHS:
            wanted[n] = int(m.group(1))
    # None found: no disassembly (an empty --function list would disassemble the whole image).
    found = instance_counts(kernel_facts.sass_lines(a.binary, a.cuda_bin, image, None, sorted(wanted)),
                            wanted) if wanted else {}
    bad = []
    for d in WIDTHS:
        c = found.get(d)
        if c is None:
            bad.append(f"D = {d}: no causal_natural_merge_kernel<CausalGeometry<256, 16, 4>, {d}, "
                       f"false, false, false, false> in the {image} image")
            continue
        print(f"{image} merge D = {d}: fixed {c['fixed']} (header {fixed_max}), stage {c['stage']} "
              f"(header {stage})")
        if c["fixed"] > fixed_max:
            bad.append(f"D = {d}: fixed {c['fixed']} above kCausalMergeFixedWarpInstructions "
                       f"{fixed_max}")
        if abs(c["stage"] - stage) > STAGE_TOLERANCE * stage:
            bad.append(f"D = {d}: stage {c['stage']} more than {STAGE_TOLERANCE:.0%} from "
                       f"kCausalMergeStageWarpInstructions {stage}")
    for line in bad:
        print("FAIL: " + line)
    if bad:
        print(f"recount the instances and update {os.path.relpath(a.header, ROOT)} (its comment "
              "names the rule), then check the chunk decisions it moves")
        return 1
    print("merge counts: within the header's constants")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--binary", required=True)
    c.add_argument("--arch", required=True)
    c.add_argument("--cuda-bin")
    c.add_argument("--header", default=HEADER)
    n = sub.add_parser("count")
    n.add_argument("--sass-dump", required=True)
    a = ap.parse_args()
    if a.command == "check":
        return check(a)
    with open(a.sass_dump, errors="replace") as f:
        for name, body in sass_counts.iter_functions(f):
            if "causal_natural_merge_kernel" in name:
                print(name, counts(body))
    return 0


if __name__ == "__main__":
    sys.exit(main())
