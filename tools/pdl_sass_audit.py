"""SASS audit of programmatic dependent launch: what a consumer kernel reads before it waits.

    python tools/pdl_sass_audit.py --build BUILD --source SOURCE [--allow FILE] [--list]
    python tools/pdl_sass_audit.py --objects OBJ... --allow FILE

A kernel launched with programmatic stream serialization (core/pdl.cuh) starts while its
producer still runs; griddepcontrol.wait (SASS ACQBULK) is where the producer's writes become
visible. Every global load placed before the first ACQBULK therefore reads memory the producer
may not have written yet. That is correct only for what was complete before the producer ran:
the call's inputs and weights. The compiler is free to hoist a load it believes invariant (an
LDG.E.CONSTANT through `const T* __restrict__`) above the wait, and once did: a
source that reads producer output after the wait is not enough, the SASS has to be checked.

For every kernel in the audited objects that contains ACQBULK, this finds the global loads
(LDG*, generic LD, LDGSTS, TMA loads) that some control-flow path reaches from the entry without
executing ACQBULK, and the kernel parameters each load's address depends on (a backward slice
over those instructions to the constant-bank loads of parameters; '?' when the address depends on
a loaded value). Each (kernel, parameter) pair must be listed in the allow file
(tools/pdl_prewait_loads.txt) with the reason it is complete at launch; an unlisted pair fails.

The audited objects are the ninferno_ops translation units whose include closure contains a source
that calls pdl::wait_for_dependencies or griddepcontrol.wait. Needs cuobjdump (CUDA toolkit);
exits 77 (skipped) without it or without built objects.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

PARAM_BASE = 0x380  # sm_90+: kernel parameters start at c[0x0][0x380]
LOAD = re.compile(r"^(LDG|LD\b|LD\.|LDGSTS|UTMALDG|UBLKCP)")
INSTRUCTION = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);")
FUNCTION = re.compile(r"Function : (\S+)")
CONSTANT = re.compile(r"c\[0x0\]\[(0x[0-9a-f]+)\]")
ADDRESS = re.compile(r"\[(U?R\d+)(?:\.64)?(?:\+[^\]]*)?\]")


def pdl_sources(source: Path) -> set[Path]:
    """Every .cu under src/ops whose quoted-include closure reaches a PDL wait."""
    root = source / "src"
    marker = re.compile(r"wait_for_dependencies|griddepcontrol\.wait")
    include = re.compile(r'#include\s+"([^"]+)"')
    cache: dict[Path, bool] = {}

    def resolve(name: str, here: Path) -> Path | None:
        for base in (here.parent, root, source / "include"):
            candidate = (base / name).resolve()
            if candidate.is_file():
                return candidate
        return None

    def reaches(path: Path, stack: set[Path]) -> bool:
        if path in cache:
            return cache[path]
        if path in stack:
            return False
        text = path.read_text(errors="replace", encoding="utf-8")
        found = bool(marker.search(text)) and path.name != "pdl.cuh"
        if not found:
            for name in include.findall(text):
                target = resolve(name, path)
                if target and reaches(target, stack | {path}):
                    found = True
                    break
        cache[path] = found
        return found

    return {path for path in (root / "ops").rglob("*.cu") if reaches(path.resolve(), set())}


def objects_for(build: Path, source: Path, sources: set[Path]) -> list[Path]:
    directory = build / "src" / "ops" / "CMakeFiles"
    out = []
    for path in sorted(sources):
        relative = path.relative_to((source / "src" / "ops").resolve())
        matches = list(directory.glob(f"*/{relative}.o"))
        out.extend(matches)
    return out


CUOBJDUMP = "cuobjdump"


def disassemble(obj: Path) -> str:
    result = subprocess.run([CUOBJDUMP, "-sass", str(obj)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"cuobjdump -sass {obj}: {result.stderr[-500:]}")
    return result.stdout


def unstatic(mangled: str) -> str:
    """The C++ mangled name inside a static function's __nv_static_ prefix."""
    if mangled.startswith("__nv_static_") and "__Z" in mangled:
        return mangled[mangled.index("__Z") + 1:]
    return mangled


def demangle_all(names: list[str]) -> dict[str, str]:
    """One c++filt run for every name (one process per kernel took tens of seconds)."""
    plain = sorted({unstatic(n) for n in names})
    if not plain or not shutil.which("c++filt"):
        return {}
    out = subprocess.run(["c++filt"], input="\n".join(plain) + "\n", capture_output=True,
                         text=True).stdout.splitlines()
    return {m: d.strip() for m, d in zip(plain, out) if d.strip()} if len(out) == len(plain) else {}


def kernel_key(mangled: str, demangled: dict[str, str]) -> str:
    """A build-stable kernel name: demangled, without the anonymous-namespace hashes of static
    functions and without the parameter list."""
    name = unstatic(mangled)
    name = demangled.get(name, name)
    name = re.sub(r"\(anonymous namespace\)::", "", name)
    name = re.sub(r"_GLOBAL__N__\w+?::", "", name)
    depth, cut = 0, len(name)
    for i, ch in enumerate(name):  # drop the parameter list (the last top-level parenthesis)
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            cut = i
    return name[:cut]


Instruction = tuple  # (address, opcode, operands, guarded by a predicate)


def parse(sass: str) -> dict[str, list[Instruction]]:
    functions: dict[str, list[Instruction]] = {}
    current = None
    for line in sass.splitlines():
        match = FUNCTION.search(line)
        if match:
            current = match.group(1)
            functions[current] = []
            continue
        if current is None:
            continue
        match = INSTRUCTION.search(line)
        if match:
            guard = (match.group(2) or "").strip()
            functions[current].append((int(match.group(1), 16), match.group(3),
                                       match.group(4).strip(), guard not in ("", "@PT")))
    return functions


def registers(text: str, wide: bool = False) -> set[str]:
    """Registers an operand list names; a .64 operand, or any operand of a 64-bit opcode
    (`wide`), names its pair."""
    out = set()
    for match in re.finditer(r"\b(U?R)(\d+)(\.64)?\b", text):
        out.add(match.group(1) + match.group(2))
        if match.group(3) or wide:
            out.add(match.group(1) + str(int(match.group(2)) + 1))
    return out - {"RZ", "URZ"}


def sources(opcode: str, rest: str) -> set[str]:
    """Registers the source operands `rest` read: a 64-bit opcode (IADD.64, ...) reads every
    register operand as a pair, a widening one (IMAD.WIDE) only its addend, the last operand."""
    operands = [o for o in rest.split(",") if o.strip()]
    if ".WIDE" in opcode and operands:
        return registers(",".join(operands[:-1])) | registers(operands[-1], True)
    return registers(rest, ".64" in opcode)


def written(opcode: str, operands: str) -> set[str]:
    first = operands.split(",")[0].strip()
    match = re.fullmatch(r"(U?R)(\d+)(\.64)?", first)
    if not match or opcode.startswith(("ST", "RED", "ATOM", "BRA", "ISETP", "FSETP")):
        return set()
    width = 4 if ".128" in opcode else 2 if match.group(3) or ".64" in opcode or ".WIDE" in opcode else 1
    return {match.group(1) + str(int(match.group(2)) + k) for k in range(width)}


def successors(body: list[Instruction]) -> list[list[int]]:
    """Control-flow successors of every instruction: a branch target, and the next instruction
    unless the instruction is an unguarded branch or exit."""
    at = {address: i for i, (address, _, _, _) in enumerate(body)}
    out = []
    for i, (_, opcode, operands, guarded) in enumerate(body):
        nexts = []
        if opcode.startswith(("BRA", "JMP")):
            target = re.search(r"0x([0-9a-f]+)\s*$", operands)
            if target and int(target.group(1), 16) in at:
                nexts.append(at[int(target.group(1), 16)])
            if guarded or opcode.startswith("BRA.DIV") or "P" in operands.split("0x")[0]:
                nexts.append(i + 1)
        elif not (opcode.startswith(("EXIT", "RET", "KILL")) and not guarded):
            nexts.append(i + 1)
        out.append([n for n in nexts if n < len(body)])
    return out


def cone(body: list[Instruction], index: int, address: set[str], region: set[int],
         preds: list[list[int]]) -> set[str]:
    """Kernel parameters (byte offsets from the parameter base) the address of the load at
    `index` depends on: a backward slice along the control flow, through the predecessors that
    lie in `region` (reachable before the wait), to the definitions that reach the load; '?'
    when the address depends on a value loaded from memory. A guarded definition does not end
    the search for its register. (Walking in address order instead credits a definition on a
    sibling branch, which the register reuse of a whole-program build makes common.)"""
    params: set[str] = set()
    seen: set[tuple[int, str]] = set()
    work = [(index, r) for r in address]
    while work:
        j, reg = work.pop()
        for p in preds[j]:
            if p not in region or (p, reg) in seen:
                continue
            seen.add((p, reg))
            _, opcode, operands, guarded = body[p]
            if reg not in written(opcode, operands):
                work.append((p, reg))
                continue
            _, _, rest = operands.partition(",")
            for constant in CONSTANT.findall(rest):
                offset = int(constant, 16) - PARAM_BASE
                if offset >= 0:
                    params.add(str(offset))
            if opcode.startswith(("LDG", "LD.", "LDS", "LDL", "LD ")):
                params.add("?")
            work.extend((p, r) for r in sources(opcode, rest))
            if guarded:
                work.append((p, reg))
    # Registers never written before the load hold thread/CTA ids (S2R) or nothing: no parameter.
    return params


def cone_in_address_order(body: list[Instruction], index: int, address: set[str],
                          region: set[int]) -> set[str]:
    """The address-order slice: from the load backwards in address order over `region`, taking
    the nearest earlier definition of each register. It follows a call's register arguments but
    may credit a definition on a sibling branch."""
    wanted, params = set(address), set()
    for i in range(index - 1, -1, -1):
        if not wanted:
            break
        if i not in region:
            continue
        _, opcode, operands, _ = body[i]
        defined = written(opcode, operands) & wanted
        if not defined:
            continue
        wanted -= defined
        _, _, rest = operands.partition(",")
        for constant in CONSTANT.findall(rest):
            offset = int(constant, 16) - PARAM_BASE
            if offset >= 0:
                params.add(str(offset))
        if opcode.startswith(("LDG", "LD.", "LDS", "LDL", "LD ")):
            params.add("?")
        wanted |= registers(rest)
    # Registers never written before the load hold thread/CTA ids (S2R) or nothing: no parameter.
    return params


def before_wait(body: list[Instruction]) -> list[int]:
    """Instructions some path from the entry reaches without executing ACQBULK: a depth-first
    walk over the control flow (branch targets; fall-through after predicated branches and
    exits) that stops at every ACQBULK. An indirect branch makes every instruction reachable."""
    at = {address: i for i, (address, _, _, _) in enumerate(body)}
    seen, stack = set(), [0]
    while stack:
        i = stack.pop()
        if i in seen or i >= len(body):
            continue
        seen.add(i)
        _, opcode, operands, guarded = body[i]
        if opcode == "ACQBULK":
            continue
        if opcode.startswith(("BRX", "JMX")):
            return list(range(len(body)))
        if opcode.startswith(("BRA", "JMP")):
            target = re.search(r"0x([0-9a-f]+)\s*$", operands)
            if target and int(target.group(1), 16) in at:
                stack.append(at[int(target.group(1), 16)])
            if guarded or opcode.startswith("BRA.DIV") or "P" in operands.split("0x")[0]:
                stack.append(i + 1)
            continue
        if opcode.startswith(("EXIT", "RET", "KILL")) and not guarded:
            continue
        stack.append(i + 1)
    return sorted(seen)


def prewait_loads(functions, allowed=None) -> dict[str, list[tuple[int, str, str]]]:
    """(address, opcode, parameter) of every global load some path reaches before the wait.
    `allowed(mangled)` gives a kernel's allowed parameters: a load whose address-order slice
    holds only allowed ones skips the control-flow slice, whose result is within it anyway."""
    out = {}
    for mangled, body in functions.items():
        if not any(opcode == "ACQBULK" for _, opcode, _, _ in body):
            continue
        loads = []
        region = before_wait(body)
        members = set(region)
        permitted = allowed(mangled) if allowed else None
        preds: list[list[int]] | None = None
        for i in region:
            address, opcode, operands, _ = body[i]
            if not LOAD.match(opcode):
                continue
            match = ADDRESS.search(re.sub(r"desc\[[^\]]*\]", "", operands))
            if not match:
                loads.append((address, opcode, "?"))
                continue
            address_registers = registers(match.group(0))
            # A parameter counts when both slices find it: the control-flow slice drops the sibling
            # branches a register-reusing (whole-program) build puts between definition and load,
            # the address-order one bounds what the control-flow slice adds through call ABIs and
            # guarded definitions, so the report never grows past the address-order audit's.
            order = cone_in_address_order(body, i, address_registers, members)
            found = order
            if permitted is None or not order <= permitted:
                if preds is None:
                    preds = [[] for _ in body]
                    for k, nexts in enumerate(successors(body)):
                        for n in nexts:
                            preds[n].append(k)
                flow = cone(body, i, address_registers, members, preds)
                found = (flow & order) or order
            for parameter in sorted(found) or ["?"]:
                loads.append((address, opcode, parameter))
        out[mangled] = loads
    return out


def read_allow(path: Path) -> list[tuple[re.Pattern, set[str]]]:
    rules = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        pattern, _, params = line.rpartition(" ")
        rules.append((re.compile(pattern.strip()), {p for p in params.split(",") if p}))
    return rules


def analyze(job: tuple) -> dict[str, list[tuple[int, str, str]]]:
    """One object's pre-wait loads, in a worker process (the SASS parse is the audit's cost).
    With the allow file and no listing, the control-flow slice runs only where it can matter."""
    global CUOBJDUMP
    CUOBJDUMP, obj, allow, listing = job
    functions = parse(disassemble(obj))
    if allow is None or listing:
        return prewait_loads(functions)
    rules = read_allow(allow)
    waiting = [m for m, body in functions.items() if any(op == "ACQBULK" for _, op, _, _ in body)]
    demangled = demangle_all(waiting)

    def allowed(mangled: str) -> set[str]:
        key = kernel_key(mangled, demangled)
        return set().union(*(names for pattern, names in rules if pattern.search(key)))

    return prewait_loads(functions, allowed)


def audit(objects: list[Path], allow: Path | None, listing: bool) -> tuple[int, set[str]]:
    """(kernels with a wait, the set of 'kernel:parameter' pre-wait loads not allowed)."""
    rules = read_allow(allow) if allow else []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        found = list(pool.map(analyze, [(CUOBJDUMP, obj, allow, listing) for obj in objects]))
    demangled = demangle_all([m for loads in found for m in loads])
    kernels, violations, count = 0, set(), 0
    for obj, per_kernel in zip(objects, found):
        for mangled, loads in sorted(per_kernel.items()):
            kernels += 1
            key = kernel_key(mangled, demangled)
            params = sorted({p for _, _, p in loads}, key=lambda p: (p == "?", p))
            allowed = set()
            for pattern, names in rules:
                if pattern.search(key):
                    allowed |= names
            if listing:
                print(f"{key}  {','.join(params) or '-'}   [{obj.name}]")
            reported = set()
            for address, opcode, parameter in loads:
                if parameter in allowed or parameter in reported:
                    continue
                reported.add(parameter)
                count += 1
                violations.add(f"{key}:{parameter}")
                print(f"VIOLATION {obj.name}: {key}: {opcode} at 0x{address:04x} before "
                      f"griddepcontrol.wait reads parameter {parameter} (not allowed)")
    print(f"pdl audit: {len(objects)} objects, {kernels} kernels with griddepcontrol.wait, "
          f"{count} (kernel, parameter) pairs read before the wait and not allowed")
    return kernels, violations


def pdl_arches(objects) -> bool:
    """Whether any object carries sm_90+ code: below it griddepcontrol compiles out (no PDL), so
    there is nothing to audit."""
    for obj in objects:
        listing = subprocess.run([CUOBJDUMP, "--list-elf", str(obj)], capture_output=True,
                                 text=True).stdout
        if any(int(arch) >= 90 for arch in re.findall(r"sm_(\d+)", listing)):
            return True
    return False


def find_cuobjdump(requested: str) -> str | None:
    """The requested cuobjdump, else one on PATH, else the newest /usr/local/cuda-*/bin/cuobjdump.

    Some toolkits ship without it (CUDA 13.2), and a build configured with one of them would
    otherwise skip the audit for good while the SASS still has to be checked."""
    for candidate in (requested, "cuobjdump"):
        found = shutil.which(candidate)
        if found:
            return found
    toolkits = sorted(Path("/usr/local").glob("cuda-*/bin/cuobjdump"),
                      key=lambda p: [int(v) for v in re.findall(r"\d+", p.parent.parent.name)])
    return str(toolkits[-1]) if toolkits else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--objects", type=Path, nargs="*")
    parser.add_argument("--allow", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--expect", nargs="*", default=None,
                        help="self-test: exactly these 'kernel-regex:parameter' violations")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()
    global CUOBJDUMP
    CUOBJDUMP = find_cuobjdump(args.cuobjdump)
    if CUOBJDUMP is None:
        print("pdl audit: cuobjdump not found; skipped")
        return 77
    if args.objects:
        # A tree built target by target may not hold them (the sabotage object is built by `all`).
        objects = [o for o in args.objects if o.exists()]
    else:
        objects = objects_for(args.build, args.source, pdl_sources(args.source.resolve()))
    if not objects:
        print("pdl audit: no built objects to audit; skipped")
        return 77
    kernels, violations = audit(objects, args.allow, args.list)
    if kernels == 0 and not pdl_arches(objects):
        print("pdl audit: no sm_90+ code (griddepcontrol compiles out below it); skipped")
        return 77
    if args.expect is not None:
        missing = [e for e in args.expect
                   if not any(re.search(e.rsplit(":", 1)[0], v.rsplit(":", 1)[0]) and
                              v.rsplit(":", 1)[1] == e.rsplit(":", 1)[1] for v in violations)]
        extra = [v for v in violations
                 if not any(re.search(e.rsplit(":", 1)[0], v.rsplit(":", 1)[0]) and
                            v.rsplit(":", 1)[1] == e.rsplit(":", 1)[1] for e in args.expect)]
        print(f"self-test: expected violations missing {missing}, unexpected {extra}")
        return 0 if kernels and not missing and not extra else 1
    return 1 if violations or kernels == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
