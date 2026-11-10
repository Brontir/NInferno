"""tools/audit/fusion_sites.json against the code, no GPU (#617 step A).

The registry lists the fusion sites found so far (the markers enumerate it; a site nobody has marked
is not in it): places where ops can run as one fused kernel or as separate launches. The code side is a marker line `// FUSION-SITE: <id>` directly above the signature of the
function that decides the site today. The marker is the enumeration point on purpose: a grep over
a list of function names would pass when someone adds a new decision function beside them, and a
marker moves with its function when the function is renamed or moved. A new site then needs two
edits that this test ties together: a marker in the code and an entry in the registry (and an
entry whose marker or function is gone fails the other way round).
"""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "tools" / "audit" / "fusion_sites.json"
SOURCE_DIRS = ("src", "include")
SUFFIXES = {".cpp", ".cc", ".cu", ".h", ".cuh", ".hpp", ".inl"}
MARKER = re.compile(r"^\s*// FUSION-SITE: ([a-z0-9_]+)$")
PRICED = {"priced", "partial", "fact", "literal", "none"}
PROMISES = {"Equal", "OwnOrder", "NotApplicable", "Unreviewed"}
REQUIRED = ("id", "name", "layer", "fuses", "decided_in", "rule_today", "alternatives", "priced",
            "byte_promise", "invariant_pin", "parity_test", "oracle_test", "one_card_literal",
            "evidence")
# Any slash-separated path with a source or doc extension, in any directory (not only src/).
PATH_IN_TEXT = re.compile(
    r"(?<![\w/.-])((?:[\w+.-]+/)+[\w+.-]+\.(?:cpp|cc|cu|cuh|h|hpp|inl|md|json|py|cmake|sh|txt))\b")


def load():
    with REGISTRY.open(encoding="utf-8") as f:
        return json.load(f)


GPU_FACTS = ROOT / "src" / "core" / "data" / "gpu_facts.json"
CARD_UNRECORDED = "card: not recorded"
METHOD_UNRECORDED = "method: not recorded"
# A method word, not negated: "not measured", "no measurement", "never profiled" do not count.
METHOD = re.compile(
    r"(?<!not )(?<!no )(?<!never )(?<!un)\b(measur\w*|a/b|nsys|ncu|census|sweep\w*|bench\w*|profil\w*)\b",
    re.I)


def board_names():
    """{alias: canonical lower-case name} from the board registry (no hand list). Aliases: the full
    product name, without a leading NVIDIA/GeForce, and for RTX/GTX boards the bare model number."""
    with GPU_FACTS.open(encoding="utf-8") as f:
        gpus = json.load(f)["gpus"]
    out = {}
    for g in gpus:
        full = re.sub(r"^(NVIDIA )?(GeForce )?", "", g["name"]).strip()
        canon = full.lower()
        out[canon] = canon
        m = re.match(r"^(RTX|GTX) (\d{4}.*)$", full)
        if m:
            out[m.group(2).lower()] = canon
    return out


def cards_named(text, names):
    """Canonical names of the boards the text names, longest alias first so `RTX 3090 Ti` is not
    also `RTX 3090`."""
    low = text.lower()
    found = set()
    for alias in sorted(names, key=len, reverse=True):
        pat = r"(?<![\w.$])" + re.escape(alias) + r"(?!\w|\.\d)"
        if re.search(pat, low):
            found.add(names[alias])
            low = re.sub(pat, " ", low)
    return found


BAD = []  # lines that mention FUSION-SITE but are not a well-formed marker


def markers():
    """[(id, file relative to the root, line index, lines of that file)] for every marker."""
    found = []
    for top in SOURCE_DIRS:
        for path in sorted((ROOT / top).rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "FUSION-SITE" not in text:
                continue
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if "FUSION-SITE" in line and not MARKER.match(line):
                    BAD.append(f"{path.relative_to(ROOT).as_posix()}:{i + 1}: {line.strip()}")
                m = MARKER.match(line)
                if m:
                    found.append((m.group(1), path.relative_to(ROOT).as_posix(), i, lines))
    return found


def signature_start(lines, i):
    """Index of the first code line after the marker at i: blank lines, comments, other markers
    and `template <...>` headers are skipped, so a marker may sit above a template."""
    j = i + 1
    while j < len(lines):
        t = lines[j].strip()
        if not t or t.startswith("//") or t.startswith("/*") or t.startswith("*") or (
                t.startswith("template") and t.endswith(">")):
            j += 1
            continue
        break
    return j


KEYWORDS = {"if", "while", "for", "switch", "return", "else", "do", "case", "throw", "new", "co_return"}


def names_function(line, name):
    """The line begins a declaration of `name`: only a return type, qualifiers and template
    arguments before it (no `(`, `=`, `[` or control keyword), so a call in a condition or a lambda
    body does not count."""
    line = re.sub(r"\[\[[^\]]*\]\]", "", line)  # [[nodiscard]] and other attributes
    m = re.match(r"^([\w:<>,*&\s~]*)\b" + re.escape(name) + r"\s*\(", line)
    return bool(m) and not (set(re.findall(r"\w+", m.group(1))) & KEYWORDS)


# What may stand between a definition's parameter list and its `{`: qualifiers, attributes, a trailing
# return type, a requires clause. Anything else (`==`, `,`, `<`, a lambda introducer) is an expression.
TAIL = re.compile(
    r"^(?:\s|const\b|volatile\b|noexcept(?:\([^)]*\))?|override\b|final\b|&&?|\[\[[^\]]*\]\]"
    r"|->[^;{=]*|requires\b[^;{]*)*$")


def starts_definition(lines, j):
    """True when the statement that starts at line j is a function definition: after the closing
    parenthesis of the first parameter list only qualifier tokens stand before the `{`, and the
    `{` comes before any `;`. A call whose arguments hold braces or lambdas, or an expression
    statement such as `select(a) == T{1};`, is not."""
    depth, closed, tail = 0, False, ""
    for k in range(j, min(j + 12, len(lines))):
        for ch in re.sub(r"//.*", "", lines[k]) + "\n":
            if depth == 0 and closed:
                if ch == "{":
                    return bool(TAIL.match(tail))
                if ch == ";":
                    return False
                tail += ch
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    closed = True
            elif depth == 0 and ch == ";":
                return False
    return False


class FusionSites(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = load()
        cls.sites = cls.reg["sites"]
        cls.by_id = {s["id"]: s for s in cls.sites}

    def test_entries_are_complete_and_unique(self):
        self.assertEqual(len(self.by_id), len(self.sites), "duplicate site id")
        for s in self.sites:
            for key in REQUIRED:
                self.assertIn(key, s, f"{s.get('id')}: missing {key}")
            self.assertRegex(s["id"], r"^[a-z0-9_]+$")
            self.assertTrue(s["fuses"] and s["alternatives"] and s["evidence"], s["id"])
            self.assertIn(s["priced"]["status"], PRICED, s["id"])
            self.assertTrue(s["priced"].get("by"), f"{s['id']}: priced.by says what prices it")
            self.assertIn(s["byte_promise"], PROMISES, s["id"])
            self.assertIsInstance(s["one_card_literal"], bool, s["id"])
            self.assertEqual(sorted(s["decided_in"]), ["file", "function"], s["id"])

    def test_no_malformed_marker(self):
        markers()
        self.assertEqual(BAD, [], "a FUSION-SITE line that is not `// FUSION-SITE: <id>`")

    def test_every_marker_has_an_entry_and_every_entry_a_marker(self):
        in_code = {}
        for site_id, file, _, _ in markers():
            in_code.setdefault(site_id, []).append(file)
        self.assertEqual(sorted(set(in_code) - set(self.by_id)), [],
                         "FUSION-SITE marker without a registry entry (tools/audit/fusion_sites.json)")
        self.assertEqual(sorted(set(self.by_id) - set(in_code)), [],
                         "registry entry whose marker is gone from the code")
        for site_id, files in in_code.items():
            self.assertEqual(len(files), 1, f"{site_id}: marked more than once: {files}")

    def test_marker_sits_on_the_registered_function(self):
        for site_id, file, i, lines in markers():
            site = self.by_id.get(site_id)
            if site is None:
                continue  # reported by the test above
            self.assertEqual(file, site["decided_in"]["file"], f"{site_id}: marker is in another file")
            j = signature_start(lines, i)
            name = site["decided_in"]["function"]
            self.assertTrue(j < len(lines) and names_function(lines[j], name),
                            f"{site_id}: the code line after the marker is not the signature of {name}")
            self.assertTrue(starts_definition(lines, j),
                            f"{site_id}: the marker is above a call or a declaration, not a definition")

    def test_cited_files_exist(self):
        def texts(node):
            if isinstance(node, str):
                yield node
            elif isinstance(node, dict):
                for v in node.values():
                    yield from texts(v)
            elif isinstance(node, list):
                for v in node:
                    yield from texts(v)

        for s in self.sites:
            self.assertTrue((ROOT / s["decided_in"]["file"]).is_file(), s["id"])
        # Every string of the registry, in sites, excluded and pending alike.
        not_yet = {p["file"] for p in self.reg["pending"]}  # a pending PR's file is not in the tree
        for text in texts(self.reg):
            for rel in PATH_IN_TEXT.findall(text):
                if rel in not_yet:
                    continue
                self.assertTrue((ROOT / rel).exists(), f"{rel} does not exist (in: {text[:60]}...)")

    def test_path_regex_accepts_dotted_basenames(self):
        self.assertEqual(PATH_IN_TEXT.findall("see docs/v0.1.0.md and a/b/c.cu."),
                         ["docs/v0.1.0.md", "a/b/c.cu"])
        self.assertEqual(PATH_IN_TEXT.findall("docs/qwen3.5/perf.md and tests/CMakeLists.txt"),
                         ["docs/qwen3.5/perf.md", "tests/CMakeLists.txt"])

    def test_every_literal_names_its_cards_and_method(self):
        """Every number carries its source: a literal site names the card(s) it was measured on (a
        name the board registry knows) and the method, or says explicitly that the record is
        missing. one_card_literal is true iff exactly one card is named."""
        names = board_names()
        unrecorded = []
        for s in self.sites:
            literal = s["priced"]["status"] == "literal"
            if not literal:
                self.assertFalse(s["one_card_literal"], f"{s['id']}: one_card_literal on a non-literal")
                continue
            text = " ".join([s["priced"]["by"]] + s["evidence"])
            cards = cards_named(text, names)
            if CARD_UNRECORDED in text:
                self.assertEqual(cards, set(), f"{s['id']}: says the card is not recorded and names one")
                unrecorded.append(f"{s['id']}: card")
            else:
                self.assertTrue(cards, f"{s['id']}: names no card the board registry knows "
                                f"(or write '{CARD_UNRECORDED}')")
            if METHOD_UNRECORDED in text:
                self.assertIsNone(METHOD.search(text), f"{s['id']}: says the method is not recorded "
                                  "but names a method word")
                unrecorded.append(f"{s['id']}: method")
            else:
                self.assertRegex(text, METHOD, f"{s['id']}: names no method (measured, A/B, nsys, "
                                 f"sweep, census, bench, profile, or write '{METHOD_UNRECORDED}')")
            self.assertEqual(s["one_card_literal"], len(cards) == 1,
                             f"{s['id']}: one_card_literal is {s['one_card_literal']}, cards named: "
                             f"{sorted(cards)}")
        print("literal sources explicitly not recorded:", unrecorded, file=sys.stderr)

    def test_card_and_method_checks_reject_what_they_should(self):
        names = board_names()
        self.assertEqual(cards_named("measured on an RTX 5090 alone", names), {"rtx 5090"})
        self.assertEqual(cards_named("RTX 3090 gains, RTX 5090 loses", names), {"rtx 3090", "rtx 5090"})
        self.assertEqual(cards_named("RTX 4060 Ti and 4090", names), {"rtx 4060 ti", "rtx 4090"})
        self.assertEqual(cards_named("one card, unnamed; 5000 tokens", names), set())
        self.assertEqual(cards_named("took 5090.5 us, cost $5090, id 15090", names), set())
        self.assertIsNone(re.search(METHOD, "a column table"))
        for negated in ("not measured", "no measurement", "never profiled", "unmeasured"):
            self.assertIsNone(METHOD.search(negated), negated)
        self.assertIsNotNone(METHOD.search("measured on one card"))

    def test_invariant_pin_and_parity_test(self):
        for s in self.sites:
            self.assertTrue(isinstance(s["invariant_pin"], str) and s["invariant_pin"], s["id"])
            for key in ("parity_test", "oracle_test"):
                if s[key] is not None:
                    self.assertTrue((ROOT / s[key]).is_file(), f"{s['id']}: {key} {s[key]} does not exist")

    def test_pending_prs_name_known_sites_and_are_not_landed(self):
        for p in self.reg["pending"]:
            self.assertIsInstance(p["pr"], int)
            for priced in p["prices"]:
                self.assertIn(priced["site"], self.by_id, f"PR {p['pr']}: unknown site")
            landed = ROOT / p["file"]
            self.assertFalse(
                landed.exists(),
                f"PR {p['pr']} landed ({p['file']} exists): move its entry from pending to priced.by of the "
                "sites it prices (this failure is intended)")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
