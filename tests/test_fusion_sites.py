"""tools/audit/fusion_sites.json against the code, no GPU (#617 step A).

The registry lists every fusion site: a place where ops can run as one fused kernel or as separate
launches. The code side is a marker line `// FUSION-SITE: <id>` directly above the signature of the
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
SUFFIXES = {".cpp", ".cu", ".h", ".cuh", ".hpp"}
MARKER = re.compile(r"^\s*//\s*FUSION-SITE:\s*([a-z0-9_]+)\s*$")
PRICED = {"priced", "partial", "fact", "literal", "none"}
PROMISES = {"Equal", "OwnOrder", "NotApplicable", "Unreviewed"}
REQUIRED = ("id", "name", "layer", "fuses", "decided_in", "rule_today", "alternatives", "priced",
            "byte_promise", "one_card_literal", "evidence")
PATH_IN_TEXT = re.compile(r"\b((?:src|include|tests|docs|tools)/[\w./+-]+\.\w+)")


def load():
    with REGISTRY.open(encoding="utf-8") as f:
        return json.load(f)


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
                m = MARKER.match(line)
                if m:
                    found.append((m.group(1), path.relative_to(ROOT).as_posix(), i, lines))
    return found


def next_code_line(lines, i):
    """The first line after i that is neither blank, another marker nor a comment."""
    for j in range(i + 1, len(lines)):
        s = lines[j].strip()
        if not s or s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            continue
        return s
    return ""


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
            # A site that has a literal from one card says so, and the reverse.
            if s["priced"]["status"] == "literal" and s["one_card_literal"]:
                self.assertTrue(any("5090" in e or "3090" in e or "card" in e for e in
                                    [s["priced"]["by"]] + s["evidence"]), s["id"])

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
            signature = next_code_line(lines, i)
            name = site["decided_in"]["function"]
            self.assertRegex(signature, r"\b" + re.escape(name) + r"\s*\(",
                             f"{site_id}: the line after the marker is not the signature of {name}")

    def test_cited_files_exist(self):
        for s in self.sites:
            self.assertTrue((ROOT / s["decided_in"]["file"]).is_file(), s["id"])
            for text in s["evidence"] + [s["priced"]["by"], s["rule_today"]]:
                for rel in PATH_IN_TEXT.findall(text):
                    self.assertTrue((ROOT / rel).exists(), f"{s['id']}: {rel} does not exist")

    def test_pending_prs_name_known_sites_and_are_not_landed(self):
        for p in self.reg["pending"]:
            self.assertIsInstance(p["pr"], int)
            for priced in p["prices"]:
                self.assertIn(priced["site"], self.by_id, f"PR {p['pr']}: unknown site")
            landed = ROOT / p["file"]
            self.assertFalse(
                landed.exists(),
                f"PR {p['pr']} landed ({p['file']} exists): move its sites to priced.by in the registry")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
