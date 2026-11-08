#!/usr/bin/env python3
r"""Generate src/text/onig_unicode_tables.h: the code points whose general category in the
Unicode version of Hugging Face `tokenizers` differs from the bundled utf8proc.

`tokenizers` 0.23 matches pre-tokenizer regexes with Oniguruma (the `onig` crate bundles
Oniguruma 6.9.x), whose \p{..} tables are Unicode 16.0.0. NInferno takes categories from
utf8proc, which is Unicode 17.0.0 (utf8proc 2.11). A code point that Unicode 17 assigned (or
re-categorised) is therefore classed differently by the two, and the split around it changes.
The table lists every code point whose category in the --old UnicodeData.txt differs from the
--new one, with its category in --old, so the pre-tokenizer can reproduce the old tables
exactly. No code point is listed by hand.

    python tools/text/generate_onig_unicode_tables.py \
        --old 16.0.0/UnicodeData.txt --new 17.0.0/UnicodeData.txt \
        --output src/text/onig_unicode_tables.h

UnicodeData.txt comes from https://www.unicode.org/Public/<version>/ucd/UnicodeData.txt (the
version is also read from the directory name when it is an x.y.z). Code points absent from a
file are Cn (unassigned). Surrogates are skipped: UTF-8 text cannot contain them.
To confirm which Unicode version tokenizers uses, run `--probe-tokenizers` with the Python
`tokenizers` package installed: it classifies every code point with Split(Regex(\p{XX})) for
all 30 categories and reports the UnicodeData.txt version (--old, plus any --also versions)
that matches with zero differences.
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

CATEGORIES = ("Cn Lu Ll Lt Lm Lo Mn Mc Me Nd Nl No Pc Pd Ps Pe Pi Pf Po Sm Sc Sk So "
              "Zs Zl Zp Cc Cf Cs Co").split()  # order = utf8proc_category_t
MAX_CP = 0x10FFFF


def load(path: Path) -> dict:
    """code point -> general category, unassigned code points omitted."""
    categories = {}
    first = None
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(";")
        if len(fields) < 3:
            continue
        cp = int(fields[0], 16)
        if fields[1].endswith(", First>"):
            first = cp
        elif fields[1].endswith(", Last>"):
            for c in range(first, cp + 1):
                categories[c] = fields[2]
        else:
            categories[cp] = fields[2]
    return categories


def load_case(path: Path) -> dict:
    """code point -> (simple upper, simple lower); code points that map to themselves omitted."""
    maps = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(";")
        if len(fields) < 14 or fields[1].endswith((", First>", ", Last>")):
            continue
        cp = int(fields[0], 16)
        upper = int(fields[12], 16) if fields[12] else cp
        lower = int(fields[13], 16) if fields[13] else cp
        if (upper, lower) != (cp, cp):
            maps[cp] = (upper, lower)
    return maps


def differing_case(old: dict, new: dict, skip) -> list:
    """Sorted (cp, lower, upper) in the old version for code points whose simple case maps
    differ, except the ones `skip` already treats as unchanged-by-identity (new code points)."""
    out = []
    for cp in sorted(set(old) | set(new)):
        if skip(cp):
            continue
        before, after = old.get(cp, (cp, cp)), new.get(cp, (cp, cp))
        if before != after:
            out.append((cp, before[1], before[0]))
    return out


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def version_of(path: Path) -> str:
    for part in reversed(path.resolve().parts):
        if re.fullmatch(r"\d+\.\d+\.\d+", part):
            return part
    return "unknown"


def differing(old: dict, new: dict):
    """Sorted (first, last, old category) ranges of code points whose category differs."""
    ranges = []
    for cp in range(MAX_CP + 1):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        before, after = old.get(cp, "Cn"), new.get(cp, "Cn")
        if before == after:
            continue
        if ranges and ranges[-1][1] == cp - 1 and ranges[-1][2] == before:
            ranges[-1][1] = cp
        else:
            ranges.append([cp, cp, before])
    return ranges


def probe_tokenizers(candidates: dict) -> None:
    from tokenizers import Regex
    from tokenizers.pre_tokenizers import Split
    import tokenizers
    cps = [c for c in range(MAX_CP + 1) if not 0xD800 <= c <= 0xDFFF and c != 0x20]
    actual = {}
    for category in CATEGORIES:
        if category == "Cs":
            continue
        split = Split(Regex(r"\p{%s}" % category), "removed")
        for i in range(0, len(cps), 20000):
            chunk = cps[i:i + 20000]
            unmatched = set()
            for piece, _ in split.pre_tokenize_str(" ".join(map(chr, chunk))):
                unmatched.update(ord(t) for t in piece.split(" ") if t)
            for c in chunk:
                if c not in unmatched:
                    actual[c] = category
    print(f"tokenizers {tokenizers.__version__}")
    for version, categories in candidates.items():
        diff = sum(1 for c in cps if categories.get(c, "Cn") != actual.get(c))
        print(f"  Unicode {version}: {diff} code points differ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", type=Path, required=True, help="UnicodeData.txt of tokenizers")
    parser.add_argument("--new", type=Path, required=True, help="UnicodeData.txt of utf8proc")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--also", type=Path, nargs="*", default=[],
                        help="more UnicodeData.txt files for --probe-tokenizers")
    parser.add_argument("--probe-tokenizers", action="store_true")
    args = parser.parse_args()

    old, new = load(args.old), load(args.new)
    old_version, new_version = version_of(args.old), version_of(args.new)
    if args.probe_tokenizers:
        candidates = {version_of(p): load(p) for p in [args.old, args.new, *args.also]}
        probe_tokenizers(candidates)
    if args.output is None:
        return 0

    ranges = differing(old, new)
    in_table = set()
    for first, last, _ in ranges:
        in_table.update(range(first, last + 1))
    case_maps = differing_case(load_case(args.old), load_case(args.new), in_table.__contains__)
    count = sum(last - first + 1 for first, last, _ in ranges)
    lines = [
        f"// Generated by tools/text/generate_onig_unicode_tables.py; do not edit.",
        f"// Code points whose general category in Unicode {old_version} (the Oniguruma tables of",
        f"// tokenizers 0.23) differs from Unicode {new_version} (bundled utf8proc 2.11), with the",
        f"// category they have in {old_version}. Sources: UnicodeData.txt of both versions from",
        f"// https://www.unicode.org/Public/<version>/ucd/.",
        f"//   UnicodeData.txt {old_version} sha256 {sha256_of(args.old)}",
        f"//   UnicodeData.txt {new_version} sha256 {sha256_of(args.new)}",
        "// Also the code points that existed in the old version and whose simple case maps",
        "// changed (kOnigCaseMaps, old values; new code points keep no case maps).",
        "// clang-format off",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace ninferno::text::onig_unicode {",
        "",
        f'inline constexpr const char* kOnigUnicodeVersion = "{old_version}";',
        f'inline constexpr const char* kUtf8procUnicodeVersion = "{new_version}";',
        f"inline constexpr std::size_t kDifferingCodePoints = {count};",
        "",
        "struct CategoryRange {",
        "    std::uint32_t first;",
        "    std::uint32_t last;",
        "    std::uint8_t category;  // utf8proc_category_t value",
        "};",
        "",
        "struct CaseMap {",
        "    std::uint32_t cp;",
        "    std::uint32_t lower;",
        "    std::uint32_t upper;",
        "};",
        "",
        "inline constexpr CategoryRange kOnigCategoryRanges[] = {",
    ]
    for first, last, category in ranges:
        lines.append(f"    {{0x{first:X}, 0x{last:X}, {CATEGORIES.index(category)}}},  // {category}")
    lines += ["};", "", "inline constexpr CaseMap kOnigCaseMaps[] = {"]
    for cp, lower, upper in case_maps:
        lines.append(f"    {{0x{cp:X}, 0x{lower:X}, 0x{upper:X}}},")
    lines += ["};", "", "} // namespace ninferno::text::onig_unicode", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"{len(ranges)} ranges, {count} code points, {len(case_maps)} case maps -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
