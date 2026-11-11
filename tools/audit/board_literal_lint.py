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
in the group of a list or a table (`Board k[] = {...}`, `std::vector<Board>`): without one it is a
list of registry boards. After `Board b{`, in a `return {` of a Board function and in
`.board = {`, the group is one board and all-qualified fields are flagged. Numbers are decimal,
hex and binary (`0b1100`), with digit separators and suffixes.

A function name may carry a template-id (`make_board<int>()`, `Foo<T>::make()`) or be an
operator (`operator+`, `operator()`, `operator[]`). A discovered alias is read bare, or qualified
as it was declared: `namespace facts { using GPUBoard = Board; }` is `facts::GPUBoard`, not
`chess::GPUBoard`; an alias declared at global scope is not seen through a qualifier.

The braced `return` of a lambda or local class nested in a Board function is not that function's
return and is skipped; a lambda with template parameters (`[]<class T>(T t) {`) is not recognised
as one, and a braced default nested two deep in a parameter list hides the function.

A requires-clause (C++20) after the parameters is walked for at most 300 tokens and a
requires-expression skipped up to 4000 characters; beyond that the function is not seen. A braced
initialiser at paren depth 0 inside the clause (`requires C<T, int{3}>`) is taken for the body.

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
    "tests/test_device.cpp": (
        (
            "ninferno::Board known{12,0,170,8,1792}",
            "ninferno::Board n1{13,0,100,64,1200}",
            "{12,0,170,0,1792}",
            "{13,0,120,64,1400}",
        ),
        "hypothetical boards probing board_distance and the gate: a generation the table does "
        "not know (cc 13.0), and the RTX 5090 with its L2 altered or withdrawn"),
    "tests/test_gate_atlas.cpp": (
        (
            "ninferno::Board small_l2{12,0,170,8,1792}",
            "ninferno::Board unknown_dram{12,0,170,96,0}",
            "ninferno::Board unknown_l2{12,0,170,0,1792}",
            "ninferno::Board{12,0,170,0,1792}",
        ),
        "hypothetical boards with a quantity zeroed or altered, probing the lookup rules"),
}

NUM = r"[-+]?(?:0[xX][0-9a-fA-F']+|0[bB][01']+|\d[\d']*(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+)[fFuUlL]*"
TUPLE = re.compile(r"\{\s*(%s)\s*,\s*(%s)\s*,\s*(%s)\s*,\s*(%s)\s*,\s*(%s)\s*"
                   r"(?:,\s*(?:true|false)\s*)?,?\s*\}" % ((NUM,) * 5))
DOT_BOARD = re.compile(r"\.board\s*=\s*\{")
# An alias and what it names: `using A = [const] T;` and `typedef [const] T [const] A;`, T a plain
# or qualified name. Whether T is Board, or an alias of it, is decided by alias_names().
ALIAS = re.compile(r"\busing\s+(\w+)\s*=\s*(?:const\b\s*)?(?:::\s*)?((?:\w+\s*::\s*)*\w+)\s*"
                   r"(?:const\b\s*)?;"
                   r"|\btypedef\s+(?:const\b\s*)?(?:::\s*)?((?:\w+\s*::\s*)*\w+)\s*"
                   r"(?:const\b\s*)?(\w+)\s*;")
NUMTOK = re.compile(
    r"0[xX][0-9a-fA-F']+[uUlL]*|0[bB][01']+[uUlL]*|\d[\d']*(?:\.\d*)?(?:[eE][-+]?\d+)?[fFuUlL]*|\.\d+[fF]?")
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


BRACKET = re.compile(r"[(){}\[\]]")


def close(code, i, limit=None):
    """Index of the bracket that closes the one at code[i], or None when unbalanced (or, with a
    `limit`, not closed within that many characters)."""
    depth = 0
    stop = len(code) if limit is None else min(len(code), i + limit)
    for m in BRACKET.finditer(code, i, stop):
        if m.group() in "({[":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return m.start()
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


def is_fields(code, parts, direct=False):
    """True when the arguments `parts` (spans) read as the fields of a board: at least two, the
    first a constant. A first argument that is a qualified name (`measured::kRtx5090`) is also the
    spelling of a registry board in a list of boards, so it counts only when a plain number
    follows among the arguments; for a `direct` literal (the group initialises one Board, not a
    list of them: `Board b{...}`, `return {...}` of a Board function, `.board = {...}`) the fields
    may all be qualified constants."""
    if len(parts) < 2:
        return False
    first = code[parts[0][0]:parts[0][1]]
    if not constant(first):
        return False
    if direct or "::" not in first:
        return True
    return any(NUMTOK.fullmatch(code[s:e].strip()) for s, e in parts[1:])


def number(tok):
    """The value of a number token (decimal, hex, binary); integer suffixes (u, l, ul, ll ...) are
    dropped first. A hex
    token keeps its trailing f/F, which is a digit there."""
    tok = tok.replace("'", "")
    if tok.lower().lstrip("+-").startswith("0b"):
        return float(int(tok.rstrip("uUlL"), 2))
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


def initialiser(code, start, bracket, direct=False):
    """Spans of the board literals of the initialiser whose bracket is code[bracket]; `start` is
    where the literal begins (the type or the designator); `direct` when the group initialises a
    single Board (see is_fields)."""
    end = close(code, bracket)
    if end is None:
        return []
    parts = split(code, bracket + 1, end)
    if not parts:
        return []
    if code[parts[0][0]] == "{":
        return rows(code, bracket + 1, end)
    if is_fields(code, parts, direct):
        return [(start, end + 1)]
    return []


def type_name(table):
    """The spellings of the Board type: the bare name (`Board`, `::ninferno::Board`) only where
    nothing qualifies it (`chess::Board` is another type); a discovered alias bare, or qualified
    only as it was declared: `table` maps each alias to the namespaces (tuples of names) that
    enclose its declaration, and a qualifier must be the whole path or its tail
    (`namespace facts { using GPUBoard = Board; }` is `facts::GPUBoard` here and in
    `ninferno::facts::GPUBoard`, never `chess::GPUBoard`)."""
    bare = r"(?<![\w.:])(?:::\s*)?(?:ninferno\s*::\s*)?Board\b"
    if not table:
        return bare
    alts = set()
    for alias, paths in table.items():
        alts.add(re.escape(alias))
        for path in paths:
            for k in range(len(path)):
                alts.add(r"\s*::\s*".join(map(re.escape, path[k:])) + r"\s*::\s*" + re.escape(alias))
    return r"(?:%s|(?<![\w.:])(?:::\s*)?(?:%s)\b)" % (bare, "|".join(sorted(alts)))


def typed_pattern(table):
    ty = type_name(table)
    tail = r"(?:\w+\s*)?(?:\[[^\]]*\]\s*)?=?\s*([{(])"
    plain = ty + r"(?!\s*::)\s*(?:const\b\s*)?[&*]*\s*(?:const\b\s*)?" + tail
    templ = r"<\s*" + ty + r"\s*(?:,[^<>;{}]*)?>\s*(?:const\b\s*)?[&*]*\s*" + tail
    return re.compile(plain + "|" + templ)


# `<...>` of a template-id, one level of nesting inside (`make<std::array<int, 2>>`).
TARGS = r"<[^<>;{}()]*(?:<[^<>;{}()]*>[^<>;{}()]*)*>"
# A function's name: an operator (`operator+`, `operator()`, `operator[]`, `operator bool`), or a
# word with an optional template-id.
FNAME = (r"(?:operator\s*(?:\(\s*\)|\[\s*\]|[-+*/%^&|~!=<>,]+|\w+)|\w+(?:\s*" + TARGS + r")?)")


def returns(table):
    """Matches the head of a function whose body is searched for braced returns, and gives the `{`
    that opens the body: `Board f(...) [const] [&|&&] [noexcept(...)] [override] {` (the name may
    be qualified, `Device::board`; a
    parameter may have a braced default, `Board b = {}`) and a trailing return `-> Board {`."""
    ty = type_name(table)
    # After the parameters: cv and virt-specifiers, `noexcept(...)`, `throw(...)`, a ref-qualifier
    # (`&`, `&&`) and an attribute (`[[...]]`), in any order. `&&|&(?!&)` reads a run of `&` one
    # way only: `&&?` could split it many ways, and a run with no `{` after it took exponential time.
    paren = r"\s*\((?:[^()]|\([^()]*\))*\)"
    quals = (r"(?:(?:(?:const|volatile|override|final|mutable|constexpr)\b|noexcept\b(?:%s)?|"
             r"throw%s|&&|&(?!&)|\[\[[^\]]*\]\])\s*)*" % (paren, paren))
    head = (ty + r"(?!\s*::)\s*(?:const\b\s*)?[&*]*\s*(?:\w+\s*(?:%s)?\s*::\s*)*" % TARGS + FNAME +
            r"\s*\((?:(?!(?<!\w)requires\b)[^;{}]|\{[^;{}]*\})*\)\s*" + quals +
            r"(?:\{|requires\b(?P<req1>))")
    trailing = (r"->\s*(?:const\b\s*)?" + ty + r"(?!\s*::)\s*(?:const\b\s*)?[&*]*\s*"
                r"(?:const\b\s*)?(?:\{|requires\b(?P<req2>))")
    return re.compile(head + "|" + trailing)


TOKEN = re.compile(r"\w+|\S")
# The walk after `requires` reads at most this many tokens and skips a requires-expression of at
# most this many characters; a clause longer than that is not followed (the function is not seen).
MAX_CLAUSE_TOKENS = 300
MAX_REQUIRES_EXPRESSION = 4000


def body_brace(code, m):
    """The index of the `{` that opens the body of the function head matched by returns(). After a
    requires-clause (C++20) the head ends at the word `requires`: the body is the first `{` at
    paren depth 0 that is not the braces of a requires-expression (`requires { ... }` or
    `requires(T t) { ... }`); None when a `;` or a `}` comes first, or when the clause is longer
    than MAX_CLAUSE_TOKENS tokens. The walk keeps the word before the brace as it goes (past a
    closed paren group, the word before that group), so it costs a bounded number of steps per
    head. Known limit: a braced initialiser at paren depth 0 inside the clause
    (`requires C<T, int{3}>`) is taken for the body."""
    if m.group("req1") is None and m.group("req2") is None:
        return m.end() - 1
    stack = []     # the word before each open paren
    before = None  # the word a `{` here would follow: the last token, or before a closed group
    j = m.end()
    for _ in range(MAX_CLAUSE_TOKENS):
        t = TOKEN.search(code, j)
        if t is None:
            return None
        tok, j = t.group(), t.end()
        if tok == "(":
            stack.append(before)
            before = None
        elif tok == ")":
            if not stack:
                return None
            before = stack.pop()
        elif stack:
            before = None
        elif tok in ";}":
            return None
        elif tok == "{":
            if before != "requires":
                return t.start()
            end = close(code, t.start(), MAX_REQUIRES_EXPRESSION)
            if end is None:
                return None
            j, before = end + 1, None
        else:
            before = tok if tok[0].isalnum() or tok[0] == "_" else None
    return None


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


SCOPE_HEAD = re.compile(r"\b(?:namespace|struct|class|union)\s+(?:\[\[[^\]]*\]\]\s*)?"
                        r"((?:\w+\s*::\s*)*\w+)\s*(?:final\b\s*)?(?::(?!:)[^;{}]*)?$")
BRACE = re.compile(r"[{}]")


def scopes_at(code, positions):
    """{position: tuple of the named scopes (namespace, struct, class, union) around it}."""
    out, stack = {}, []
    todo = sorted(set(positions))
    k = 0
    for b in BRACE.finditer(code):
        while k < len(todo) and todo[k] < b.start():
            out[todo[k]] = tuple(n for entry in stack for n in entry)
            k += 1
        if b.group() == "{":
            h = SCOPE_HEAD.search(code, max(0, b.start() - 300), b.start())
            stack.append(tuple(re.sub(r"\s+", "", h.group(1)).split("::")) if h else ())
        elif stack:
            stack.pop()
    for pos in todo[k:]:
        out[pos] = tuple(n for entry in stack for n in entry)
    return out


def alias_table(code):
    """{alias: set of the scopes it is declared in} for every alias of `Board` declared in `code`,
    chains followed to a fixed point (`using B = Board; using C = B;`, in any order): an alias of
    `Board` or `ninferno::Board` spelled so, or of an alias found so, through any namespace."""
    decls = []
    for m in ALIAS.finditer(code):
        alias, target = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        decls.append((alias, re.sub(r"\s+", "", target), m.start()))
    names = {"Board"}
    grew = True
    while grew:
        grew = False
        for alias, target, _pos in decls:
            if alias in names:
                continue
            if target in ("Board", "ninferno::Board") or target.split("::")[-1] in names - {"Board"}:
                names.add(alias)
                grew = True
    where = scopes_at(code, [pos for _a, _t, pos in decls]) if len(names) > 1 else {}
    table = {}
    for alias, _target, pos in decls:
        if alias in names and alias != "Board":
            table.setdefault(alias, set()).add(where[pos])
    return table


def literals(text):
    """[(line, normalised literal, source line)] of the board literals in `text`, in file order.
    A literal nested in another counts once, as the outer one."""
    code = blank(text)
    table = alias_table(code)
    spans = []
    for m in typed_pattern(table).finditer(code):
        # A group right after the type names one Board; after `<Board>` or `Board k[]` it may be
        # a list of boards, where `{measured::kA, measured::kB}` is registry names.
        spans += initialiser(code, m.start(), m.end() - 1,
                             direct=not m.group().startswith("<") and "[" not in m.group())
    for m in returns(table).finditer(code):
        body = body_brace(code, m)
        if body is None:
            continue
        end = close(code, body)
        for r in own_returns(code, body, end if end is not None else len(code)):
            spans += initialiser(code, r.end() - 1, r.end() - 1, direct=True)
    for m in DOT_BOARD.finditer(code):
        spans += initialiser(code, m.start(), m.end() - 1, direct=True)
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
