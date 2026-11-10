"""Host lint over routes/*.json: provenance of evidence_us and tuned_on (#634), no GPU.

The loader reads neither key; tools (gap maps, regret tooling) do. An evidence_us row is
[T, table us, chosen-route us], so the timings are only about this card if the route the entry
selects at T is the route that was timed:

 (a) a route_as entry's evidence widths lie in its route_as bands. A generic entry's evidence may
     sit at widths its bands leave to the shape-owned table (a gap is a decision too), so for
     those entries the route check is (b).
 (b) tuned_on names the board the numbers were measured on. When it names another table's board
     key (class or class:NNsm) the entries are borrowed, and are valid only while the borrowed
     entry is still the donor's: the donor table has the same entry with the same evidence_us,
     and this entry selects the donor's route at every evidence width. A free-text label (a
     model name, a tool and card) is not a board key and is not checked here. route_as entries
     are selected by the shape-owned tables, whose routes this lint cannot see, so a borrowed
     route_as entry is refused outright: its evidence must be re-measured or dropped.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_tables():
    tables = {}
    for path in sorted((ROOT / "routes").glob("*.json")):
        tables[path.name] = json.loads(path.read_text(encoding="utf-8"))
    return tables


def board_keys(table):
    gpu = table["gpu"]
    keys = {gpu["class"]}
    sms = (gpu.get("board") or {}).get("sms")
    if sms and ":" not in gpu["class"]:
        keys.add("%s:%dsm" % (gpu["class"], sms))
    return keys


def entry_key(e):
    return (e["format"], e["compute"], e["epilogue"], e["n"], e["k"])


def selected_route(e, t):
    """The route the generic path selects at T tokens, or None when no field covers T."""
    for lo, hi, route in e.get("bands", []):
        if lo <= t <= hi:
            return route
    for lo, hi, route in e.get("prefill_bands", []):
        if lo <= t <= hi:
            return route
    if t >= 17 and "prefill" in e:
        return e["prefill"]
    return None


def violations(tables):
    out = []
    owner = {}
    for name, table in tables.items():
        for key in board_keys(table):
            owner.setdefault(key, name)
    for name, table in tables.items():
        own = board_keys(table)
        for e in table["entries"]:
            ev = e.get("evidence_us")
            tuned = e.get("tuned_on")
            if ev is None and tuned is None:
                continue
            where = "%s %s" % (name, entry_key(e))
            if ev is None:
                continue  # a bare tuned_on is a label (what the entry was tuned on), no timings
            widths = [row[0] for row in ev]
            if "route_as" in e:
                for t in widths:
                    if not any(lo <= t <= hi for lo, hi, _ in e["route_as"]):
                        out.append("%s: evidence width %d is in no route_as band" % (where, t))
            if tuned is None or tuned in own or tuned not in owner:
                continue
            donor = tables[owner[tuned]]
            twin = next((d for d in donor["entries"] if entry_key(d) == entry_key(e)), None)
            if twin is None or twin.get("evidence_us") != ev:
                out.append("%s: tuned_on %s but %s has no entry with this evidence" %
                           (where, tuned, owner[tuned]))
            elif "route_as" in e:
                out.append("%s: route_as evidence borrowed from %s" % (where, tuned))
            else:
                moved = [t for t in widths if selected_route(e, t) != selected_route(twin, t)]
                if moved:
                    out.append("%s: evidence copied from %s, but the bands pick a different "
                               "route at T=%s" % (where, tuned, moved))
    return out


class RouteEvidenceLint(unittest.TestCase):
    def test_all_tables(self):
        found = violations(load_tables())
        self.assertEqual(found, [], "\n" + "\n".join(found))

    def test_lint_catches_a_borrowed_entry_whose_bands_moved(self):
        donor = {"gpu": {"class": "aa", "board": {"sms": 8}}, "entries": [
            {"format": "q8", "compute": "a16", "epilogue": "store", "n": 1, "k": 1,
             "bands": [[1, 16, "r.x"]], "evidence_us": [[4, 2.0, 1.0]], "tuned_on": "aa:8sm"}]}
        copy = {"gpu": {"class": "bb", "board": {"sms": 4}}, "entries": [
            {"format": "q8", "compute": "a16", "epilogue": "store", "n": 1, "k": 1,
             "bands": [[1, 16, "r.y"]], "evidence_us": [[4, 2.0, 1.0]], "tuned_on": "aa:8sm"}]}
        self.assertEqual(violations({"a.json": donor}), [])
        self.assertEqual(len(violations({"a.json": donor, "b.json": copy})), 1)
        copy["entries"][0]["bands"] = [[1, 16, "r.x"]]
        self.assertEqual(violations({"a.json": donor, "b.json": copy}), [])
        copy["entries"][0]["bands"] = [[1, 2, "r.x"]]
        self.assertEqual(len(violations({"a.json": donor, "b.json": copy})), 1)


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])
