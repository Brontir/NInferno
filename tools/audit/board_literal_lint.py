#!/usr/bin/env python3
"""Fail on a hand-written board identity: a board's facts written as a literal.

A board is named by its PCI id and read from the registry (core/board_registry.h, generated from
core/data/gpu_facts.json): `board_registry::board(0x2B85)`. A literal that spells a board's compute
capability, SMs, L2 and bandwidth is a second copy of the registry row, the class of bug behind
#502, #507, #556 and #594 (roadmap #612, step 9).

The text of every tracked C++ file is read as one token stream (comments and the inside of string
and character literals blanked, line breaks kept as white space), so a literal may span lines, sit
in a macro or follow any declarator. Flagged:

  1. `Board` (or a `using`/`typedef` alias of it; `ninferno::Board`, `::ninferno::Board`, or
     `Board` as a template argument: `std::vector<Board>`, `std::array<Board, N>`) initialised with
     `{...}` or `(...)` that has at least two arguments and whose first is a constant (digits,
     `kName` constants, parentheses, operators, or a designated `.major = 12`); any declarator
     between the type and the bracket (`Board k[] = {`, `const Board& b = {`, `Board const b(`).
     When the first argument starts with `{` the group is a table: every row is tested, however
     few numbers it has (`{8, 6, 82}`), however it is written (`{.major = 8, ...}`, `{8, 0, kSms,
     ...}`);
     A function returning `Board` has its braced `return {...}` tested the same way.
  2. a `.board = {...}` designated initialiser, tested like 1;
  3. any brace tuple of five numbers that reads as a board (major 5..15, minor 0..9, SMs 1..400,
     L2 up to 512 MiB, bandwidth of 100 GB/s or more), optionally a bool after it, optionally a
     trailing comma: `{12, 0, 170, 96, 1792}`, `{7, 5, 68, 5.5, 616, true}`, wherever it stands
     (`return {...}`, a call argument, `b = {...}`, a table row). Lists of sizes, token ids and
     shapes fall outside the ranges.

Not flagged: conversions from fields (`Board{cc / 10, cc % 10, sms}`, `Board{row->major, ...}`),
`struct Board {` and other declarations, a same-named type of another namespace (`chess::Board`),
words that only start with Board (`BoardRow`, `BoardCount`).

A qualified constant as the first field (`facts::kMajor`) counts only when a plain number follows
in the group: without one it is a list of registry boards.

The braced `return` of a lambda or local class nested in a Board function is not that function's
return and is skipped; a lambda with template parameters (`[]<class T>(T t) {`) is not recognised
as one, and a braced default nested two deep in a parameter list hides the function.

Not seen (known limits): a row that starts with a name or an id (`{"rtx3090", 8, 6, 82, ...}`,
`{0x2204, 8, 6, ...}`), a helper call (`make_board(8, 6, 82, 6, 936)`, `emplace_back(...)`), a
board written field by field (`b.sms = 82;`), a typed literal whose first field is a lower-case
name, a designated group without a type (`{.major = 8, .minor = 6, ...}`), JSON and Python
files (routes/*.json, tests/*.py), an alias declared in another file, a table whose first row
is a name rather than a braced row (`Board k[] = {measured::kRtx5090, {8, 9, 128}}`), the second
declarator of a statement (`Board a = measured::kX, b{...};`; five bare numbers there are still
caught by the tuple rule). Code under `#if 0` is read like any other.

Flagged although it is no board: an unqualified `Board` of an unrelated type in a file that defines
its own (`namespace chess { struct Board {...}; Board start{8, 8}; }`), and a five-number list that
happens to fall inside the ranges (a tensor shape). Name such a file in ALLOW with the reason.

Exceptions are files named in ALLOW, each with the sorted normalised texts of its literals and a
reason. The pin is the literals themselves: a new, replaced or edited literal fails until the pin is
reviewed, and so does a pinned literal that is gone. The pin holds a literal's text, not its place:
moving a pinned literal inside its file, or swapping two pinned rows, passes.
"""
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".inc", ".ipp", ".tpp",
            ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh"}
# Vendored code keeps its own style: these prefixes are skipped, no other directory.
THIRD_PARTY = ("third_party/", "src/ui/third_party/")
SKIP_DIRS = {".git", "build"}  # only the walk used without a git checkout

# path -> (sorted normalised literals, reason). Every entry is a reviewed exception, pinned by the
# literals themselves (whitespace and comments dropped), as `literals()` prints them.
ALLOW = {
    "src/core/measured_boards.h": (
        (
            "Board{12,0,170,96,1792}",
            "Board{12,0,36,32,448}",
            "Board{12,0,84,64,960}",
            "Board{7,0,80,6,900}",
            "Board{7,5,68,5.5,616}",
            "Board{8,0,108,40,1555.2}",
            "Board{8,6,72,6,600}",
            "Board{8,6,82,6,936}",
            "Board{8,9,128,72,1008}",
            "Board{8,9,34,32,288}",
            "Board{8,9,58,48,300}",
            "Board{8,9,80,64,736}",
        ),
        "pins: static_asserts that each registry row equals the literal it replaced"),
    "src/ops/generic/r8_decode_table.h": (
        (
            "{12,0,170,96,1792}",
            "{12,0,36,32,448}",
            "{7,5,68,5.5,616}",
            "{8,0,108,40,1555}",
            "{8,6,72,6,600}",
            "{8,6,82,6,936}",
            "{8,9,128,72,1008}",
            "{8,9,34,32,288}",
            "{8,9,80,64,736}",
        ),
        "generated by tools/bench/r8_decode_table.py, which still writes each table's board as "
        "numbers (one row has drifted from the registry); to emit registry ids (#612 step 9)"),
    "tests/test_device.cpp": (
        (
            "ninferno::Board a30{8,0,56,24,933}",
            "ninferno::Board known{12,0,170,8,1792}",
            "ninferno::Board n1{13,0,100,64,1200}",
            "ninferno::Board ti5070{12,0,70,48,896}",
            "ninferno::Board ti{12,0,70,48,896}",
            "{12,0,170,0,1792}",
            "{12,0,170,96,1792}",
            "{12,0,36,32,448}",
            "{12,0,84,64,960}",
            "{13,0,120,64,1400}",
            "{8,0,108,40,1555}",
            "{8,6,72,6,600}",
            "{8,9,58,48,300}",
            "{8,9,60,48,504}",
        ),
        "boards written by hand next to the registry rows they approximate, plus hypothetical "
        "ones (cc 13.0, synthetic SM counts) probing board_distance; to move to the registry"),
    "tests/test_gate_atlas.cpp": (
        (
            "ninferno::Board small_l2{12,0,170,8,1792}",
            "ninferno::Board unknown_dram{12,0,170,96,0}",
            "ninferno::Board unknown_l2{12,0,170,0,1792}",
            "ninferno::Board{12,0,170,0,1792}",
        ),
        "hypothetical boards with a quantity zeroed or altered, probing the lookup rules"),
    "tests/test_context_cost.cpp": (
        (
            ".board={12,0,188,128,1792}",
            "{8,9,114,72,1008}",
        ),
        "real boards written by hand as fixtures of the context-cost resolver; to move to the "
        "registry"),
}

NUM = r"[-+]?(?:0[xX][0-9a-fA-F']+|\d[\d']*(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+)[fFuUlL]*"
TUPLE = re.compile(r"\{\s*(%s)\s*,\s*(%s)\s*,\s*(%s)\s*,\s*(%s)\s*,\s*(%s)\s*"
                   r"(?:,\s*(?:true|false)\s*)?,?\s*\}" % ((NUM,) * 5))
DOT_BOARD = re.compile(r"\.board\s*=\s*\{")
ALIAS = re.compile(r"\busing\s+(\w+)\s*=\s*(?:const\b\s*)?(?:::\s*)?"
                   r"(?:ninferno\s*::\s*)?Board\b\s*(?:const\b\s*)?;"
                   r"|\btypedef\s+(?:const\b\s*)?(?:::\s*)?"
                   r"(?:ninferno\s*::\s*)?Board\b\s*(?:const\b\s*)?(\w+)\s*;")
NUMTOK = re.compile(
    r"0[xX][0-9a-fA-F']+[uUlL]*|\d[\d']*(?:\.\d*)?(?:[eE][-+]?\d+)?[fFuUlL]*|\.\d+[fF]?")
RAW = re.compile(r'R"([^()\\ ]{0,16})\(')


CHAR_PREFIXES = {"L", "u", "U", "u8"}


def in_word(text, i):
    """True when the quote at text[i] follows a word and is therefore a digit separator
    (`1'000`), not the start of a character literal: the word before it is not an encoding
    prefix (`L'/'`, `u'x'`, `U'x'`, `u8'x'`)."""
    j = i
    while j > 0 and (text[j - 1].isalnum() or text[j - 1] == "_"):
        j -= 1
    return j < i and text[j:i] not in CHAR_PREFIXES


def blank(text):
    """`text` with comments and the inside of string/char literals replaced by spaces; line breaks
    stay, so offsets and line numbers are unchanged."""
    out, i, n = [], 0, len(text)

    def fill(a, b):
        out.append("".join(c if c == "\n" else " " for c in text[a:b]))

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = i
            while j < n and text[j] != "\n":
                j = j + 2 if text[j] == "\\" and j + 1 < n else j + 1
            fill(i, j)
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            fill(i, j)
            i = j
        elif c == '"':
            m = RAW.match(text, i - 1) if i and text[i - 1] == "R" else None
            if m:
                end = text.find(")" + m.group(1) + '"', m.end())
                j = n if end < 0 else end + len(m.group(1)) + 2
                out.append('"')
                fill(i + 1, max(i + 1, j - 1))
                out.append('"')
                i = j
                continue
            j = i + 1
            while j < n and text[j] != '"' and text[j] != "\n":
                j = j + 2 if text[j] == "\\" and j + 1 < n else j + 1
            out.append('"')
            fill(i + 1, j)
            if j < n and text[j] == '"':
                out.append('"')
                j += 1
            i = j
        elif c == "'" and not in_word(text, i):
            j = i + 1
            while j < n and text[j] != "'" and text[j] != "\n":
                j = j + 2 if text[j] == "\\" and j + 1 < n else j + 1
            out.append("'")
            fill(i + 1, j)
            if j < n and text[j] == "'":
                out.append("'")
                j += 1
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def close(code, i):
    """Index of the bracket that closes the one at code[i], or None when unbalanced."""
    depth = 0
    for j in range(i, len(code)):
        if code[j] in "({[":
            depth += 1
        elif code[j] in ")}]":
            depth -= 1
            if depth == 0:
                return j
    return None


def split(code, a, b):
    """(start, end) of the top-level comma separated parts of code[a:b], white space trimmed,
    empty parts left out."""
    parts, depth, cur = [], 0, a
    for j in range(a, b + 1):
        c = code[j] if j < b else ","
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
        if c == "," and depth == 0:
            s, e = cur, j
            while s < e and code[s].isspace():
                s += 1
            while e > s and code[e - 1].isspace():
                e -= 1
            if s < e:
                parts.append((s, e))
            cur = j + 1
    return parts


def constant(arg):
    """True when `arg` is made of numbers, `kName` constants, parentheses and operators only."""
    arg = re.sub(r"^\.\w+\s*=\s*", "", arg)
    # A qualified constant (`facts::kMajor`, `::ns::kSms`) is one constant too.
    arg = re.sub(r"(?:(?:\b\w+)?\s*::\s*)*\bk[A-Z]\w*\b", "0", arg)
    arg = NUMTOK.sub("0", arg)
    return bool(arg.strip()) and re.fullmatch(r"[\s0()+\-*/%<>|&~^]*", arg) is not None


def is_fields(code, parts):
    """True when the arguments `parts` (spans) read as the fields of a board: at least two, the
    first a constant. A first argument that is a qualified name (`measured::kRtx5090`) is also the
    spelling of a registry board in a list of boards, so it counts only when a plain number
    follows among the arguments."""
    if len(parts) < 2:
        return False
    first = code[parts[0][0]:parts[0][1]]
    if not constant(first):
        return False
    if "::" not in first:
        return True
    return any(NUMTOK.fullmatch(code[s:e].strip()) for s, e in parts[1:])


def number(tok):
    """The value of a number token; integer suffixes (u, l, ul, ll ...) are dropped first. A hex
    token keeps its trailing f/F, which is a digit there."""
    tok = tok.replace("'", "")
    if tok.lower().lstrip("+-").startswith("0x"):
        return float(int(tok.rstrip("uUlL"), 16))
    return float(tok.rstrip("fFuUlL"))


def board_shaped(m):
    """The five numbers of a tuple read as (major, minor, SMs, L2 MiB, GB/s) of a board: a major
    5..15, a minor 0..9, SMs 1..400, an L2 of at most 512, a bandwidth of 100 or more. Lists of
    sizes, token ids and shapes fall outside."""
    try:
        major, minor, sms, l2, bw = (number(g) for g in m.groups())
    except ValueError:
        return False
    return (5 <= major <= 15 and major == int(major) and 0 <= minor <= 9 and minor == int(minor)
            and 1 <= sms <= 400 and sms == int(sms) and 0 <= l2 <= 512 and bw >= 100)


def rows(code, a, b):
    """Spans of the board-literal rows of the table code[a:b]: each `{...}` part with at least two
    arguments and a constant first one; a part whose first argument is `{` is a table again."""
    out = []
    for s, e in split(code, a, b):
        if code[s] != "{":
            continue
        inner = split(code, s + 1, e - 1)
        if not inner:
            continue
        if code[inner[0][0]] == "{":
            out += rows(code, s + 1, e - 1)
        elif is_fields(code, inner):
            out.append((s, e))
    return out


def initialiser(code, start, bracket):
    """Spans of the board literals of the initialiser whose bracket is code[bracket]; `start` is
    where the literal begins (the type or the designator)."""
    end = close(code, bracket)
    if end is None:
        return []
    parts = split(code, bracket + 1, end)
    if not parts:
        return []
    if code[parts[0][0]] == "{":
        return rows(code, bracket + 1, end)
    if is_fields(code, parts):
        return [(start, end + 1)]
    return []


def type_name(names):
    """The spellings of the Board type: the bare name (`Board`, `::ninferno::Board`) only where
    nothing qualifies it (`chess::Board` is another type), a discovered alias also through any
    namespace (`facts::GPUBoard`, `::a::b::GPUBoard`: the alias was declared as Board)."""
    bare = r"(?<![\w.:])(?:::\s*)?(?:ninferno\s*::\s*)?Board\b"
    aliases = sorted(set(names) - {"Board"})
    if not aliases:
        return bare
    alt = "|".join(map(re.escape, aliases))
    return r"(?:%s|(?<![\w.])(?:::\s*)?(?:\w+\s*::\s*)*(?:%s)\b)" % (bare, alt)


def typed_pattern(names):
    ty = type_name(names)
    tail = r"(?:\w+\s*)?(?:\[[^\]]*\]\s*)?=?\s*([{(])"
    plain = ty + r"(?!\s*::)\s*(?:const\b\s*)?[&*]*\s*(?:const\b\s*)?" + tail
    templ = r"<\s*" + ty + r"\s*(?:,[^<>;{}]*)?>\s*(?:const\b\s*)?[&*]*\s*" + tail
    return re.compile(plain + "|" + templ)


def returns(names):
    """Matches the head of a function whose body is searched for braced returns, and gives the `{`
    that opens the body: `Board f(...) [const] [&|&&] [noexcept(...)] [override] {` (the name may
    be qualified, `Device::board`; a
    parameter may have a braced default, `Board b = {}`) and a trailing return `-> Board {`."""
    ty = type_name(names)
    # After the parameters: cv and virt-specifiers, `noexcept(...)`, `throw(...)`, a ref-qualifier
    # (`&`, `&&`) and an attribute (`[[...]]`), in any order.
    paren = r"\s*\((?:[^()]|\([^()]*\))*\)"
    quals = (r"(?:(?:(?:const|volatile|override|final|mutable|constexpr)\b|noexcept\b(?:%s)?|"
             r"throw%s|&&?|\[\[[^\]]*\]\])\s*)*" % (paren, paren))
    head = (ty + r"(?!\s*::)\s*(?:const\b\s*)?[&*]*\s*(?:\w+\s*::\s*)*\w+\s*"
            r"\((?:[^;{}]|\{[^;{}]*\})*\)\s*" + quals + r"\{")
    trailing = (r"->\s*(?:const\b\s*)?" + ty + r"(?!\s*::)\s*(?:const\b\s*)?[&*]*\s*"
                r"(?:const\b\s*)?\{")
    return re.compile(head + "|" + trailing)


RETURN_BRACE = re.compile(r"\breturn\s*\{")
LOCAL_TYPE = re.compile(r"\b(?:struct|class|union)\b[^;{}()]*$")
TRAILING_TYPE = re.compile(r"->\s*[^;{}()\[\]]*$")
CALLABLE_QUALS = re.compile(r"(?:\b(?:mutable|noexcept|constexpr|const)\b\s*)+$")


def opens_other_callable(code, brace):
    """True when the `{` at code[brace] opens the body of a lambda or the members of a local class:
    a `return {...}` inside is not the return of the function around it."""
    pre = code[:brace].rstrip()
    if LOCAL_TYPE.search(pre):
        return True
    pre = TRAILING_TYPE.sub("", pre).rstrip()
    pre = CALLABLE_QUALS.sub("", pre).rstrip()
    if pre.endswith("]"):
        return True
    if pre.endswith(")"):
        depth = 0
        for j in range(len(pre) - 1, -1, -1):
            if pre[j] == ")":
                depth += 1
            elif pre[j] == "(":
                depth -= 1
                if depth == 0:
                    return pre[:j].rstrip().endswith("]")
    return False


def own_returns(code, body, end):
    """Matches of `return {` in the body code[body:end] that return from the function itself, not
    from a lambda or local class nested in it."""
    out = []
    for r in RETURN_BRACE.finditer(code, body + 1, end):
        stack = []
        for j in range(body + 1, r.start()):
            if code[j] == "{":
                stack.append(j)
            elif code[j] == "}" and stack:
                stack.pop()
        if not any(opens_other_callable(code, j) for j in stack):
            out.append(r)
    return out


def normal(text):
    """`text` with white space dropped around punctuation and collapsed elsewhere."""
    return re.sub(r"\s*([{}(),=])\s*", r"\1", re.sub(r"\s+", " ", text)).strip()


def literals(text):
    """[(line, normalised literal, source line)] of the board literals in `text`, in file order.
    A literal nested in another counts once, as the outer one."""
    code = blank(text)
    names = {"Board"}
    for m in ALIAS.finditer(code):
        names.add(m.group(1) or m.group(2))
    spans = []
    for m in typed_pattern(names).finditer(code):
        spans += initialiser(code, m.start(), m.end() - 1)
    for m in returns(names).finditer(code):
        end = close(code, m.end() - 1)
        for r in own_returns(code, m.end() - 1, end if end is not None else len(code)):
            spans += initialiser(code, r.end() - 1, r.end() - 1)
    for m in DOT_BOARD.finditer(code):
        spans += initialiser(code, m.start(), m.end() - 1)
    for m in TUPLE.finditer(code):
        if board_shaped(m):
            spans.append((m.start(), m.end()))
    kept, last = [], -1
    for s, e in sorted(set(spans), key=lambda x: (x[0], -x[1])):
        if s >= last:
            kept.append((s, e))
            last = e
    lines = text.splitlines()
    out = []
    for s, e in kept:
        n = code.count("\n", 0, s) + 1
        out.append((n, normal(code[s:e]), lines[n - 1].strip() if n <= len(lines) else ""))
    return out


def violations(text):
    """[(line number, source line)] of the board literals in `text`, one entry per line."""
    found = {}
    for n, _lit, line in literals(text):
        found.setdefault(n, line)
    return sorted(found.items())


def is_cpp(path):
    """True for a C++ source or header, and for a template of one (`embedded_gpu_facts.h.in`: the
    suffix before a trailing `.in` counts)."""
    name = path.name.lower()
    if name.endswith(".in"):
        name = name[:-3]
    return Path(name).suffix in SUFFIXES


def files(root):
    """C++ sources to scan: the tracked files outside the third-party prefixes, whatever the
    directory (apps/, bench/ and include/ too); without a git checkout, a walk of the same."""
    paths = None
    if (root / ".git").exists():
        try:
            r = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True)
        except FileNotFoundError:
            raise SystemExit("board_literal_lint: %s is a git checkout but the git command is "
                             "not installed or not on PATH; install git to list tracked files"
                             % root)
        if r.returncode == 0:
            paths = [root / p for p in r.stdout.decode("utf-8", "replace").split("\0") if p]
    if paths is None:
        paths = [p for p in root.rglob("*") if not SKIP_DIRS & set(p.relative_to(root).parts)]
    return sorted(p for p in paths
                  if is_cpp(p) and p.is_file()
                  and not p.relative_to(root).as_posix().startswith(THIRD_PARTY))


def scan(root, allow=ALLOW):
    """[(path, line, text)] of what to fix. A literal outside `allow`, or one a pinned file does not
    list, carries its line number; a pinned literal that is not in its file (changed, removed,
    file gone) has line 0 and the text `pinned: <literal>`."""
    bad, seen = [], {}
    for p in files(root):
        rel = p.relative_to(root).as_posix()
        hits = literals(p.read_text(encoding="utf-8", errors="replace"))
        if rel not in allow:
            bad.extend((rel, n, line) for n, _lit, line in hits)
            continue
        seen[rel] = Counter(allow[rel][0])
        for n, lit, line in hits:
            if seen[rel][lit] > 0:
                seen[rel][lit] -= 1
            else:
                bad.append((rel, n, line))
    for rel, (pinned, _reason) in sorted(allow.items()):
        left = seen.get(rel, Counter(pinned))
        bad.extend((rel, 0, "pinned: " + lit) for lit in sorted(left.elements()))
    return bad


def main(root=ROOT, allow=ALLOW):
    bad = scan(root, allow)
    for rel, n, line in bad:
        if n:
            print(f"{rel}:{n}: board literal outside the registry: {line}")
        else:
            print(f"{rel}: {line} (no longer in the file)")
    if any(n for _, n, _ in bad):
        print("A new literal: name the board by PCI id, board_registry::board(0x....) "
              "(core/board_registry.h).")
    if any(not n for _, n, _ in bad):
        print("A pin is stale: lower or update the entry of that file in ALLOW "
              "(tools/audit/board_literal_lint.py) to the literals it now holds, after reviewing "
              "that none is a new copy of a registry row.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
