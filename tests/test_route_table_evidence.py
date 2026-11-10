"""Host lint over routes/*.json: provenance of evidence_us and tuned_on (#634), no GPU.

The loader reads neither key; tools (gap maps, regret tooling) do. An evidence_us row is
[T, table us, chosen-route us], so the timings are only about this card if the route the entry
selects at T is the route that was timed. Rows carry no route names, so a borrowed copy is
found by comparing with the table it was copied from.

 (a) a route_as entry's evidence widths lie in its route_as bands. A generic entry's evidence may
     sit at widths its bands leave to the shape-owned table (a gap is a decision too).
 (b) no tuned_on, or a tuned_on naming this board: the evidence is this board's own, so it must
     not be byte-identical to the same entry's evidence in another table (a silent copy).
 (c) tuned_on naming another table's board key (class or class:NNsm): an honest borrow, valid
     only while the donor has the same entry with the same evidence_us and this entry takes the
     donor's route at every evidence width: the generic route under both profiles (bands for the
     decode profile, prefill_bands / prefill from 17 tokens for the prefill profile, as
     route_table.cpp table_route reads them) and, for route_as, the donor's T'.
 A free-text tuned_on (a model name, a tool and card) is not a board key and is not checked; a
 tuned_on without evidence_us is a label and is not checked.
Rule (c) assumes the shape-owned table picks the same kernel at T' on both boards; a shape
that branches on the SM count (src/ops/linear/nvfp4/shapes/n2048_k5376.cu, select_a4 by
grid waves) can break that across a wave threshold, which this lint cannot see.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFILL_MIN_TOKENS = 17  # kPrefillMinTokens in src/ops/route_table.cpp


def load_tables():
    return {p.name: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((ROOT / "routes").glob("*.json"))}


def board_keys(table):
    gpu = table["gpu"]
    keys = {gpu["class"]}
    sms = (gpu.get("board") or {}).get("sms")
    if sms and ":" not in gpu["class"]:
        keys.add("%s:%dsm" % (gpu["class"], sms))
    return keys


def entry_key(e):
    return (e["format"], e["compute"], e["epilogue"], e["n"], e["k"])


def generic_route(e, t, prefill):
    """The route table_route() returns at T tokens under the decode or the prefill profile."""
    if prefill and t >= PREFILL_MIN_TOKENS:
        for lo, hi, route in e.get("prefill_bands", []):
            if lo <= t <= hi:
                return route
        return e.get("prefill")
    for lo, hi, route in e.get("bands", []):
        if lo <= t <= hi:
            return route
    return None


def route_as_at(e, t):
    for lo, hi, as_t in e.get("route_as", []):
        if lo <= t <= hi:
            return as_t
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
            if ev is None:
                continue
            tuned = e.get("tuned_on")
            where = "%s %s" % (name, entry_key(e))
            widths = [row[0] for row in ev]
            if "route_as" in e:
                for t in widths:
                    if route_as_at(e, t) is None:
                        out.append("%s: evidence width %d is in no route_as band" % (where, t))
            if tuned is None or tuned in own:
                for other_name, other in tables.items():
                    if other_name == name:
                        continue
                    # An entry of the other table that names this board in tuned_on is the
                    # honest borrow of this evidence, not a silent copy.
                    if any(entry_key(d) == entry_key(e) and d.get("evidence_us") == ev
                           and d.get("tuned_on") not in own for d in other["entries"]):
                        out.append("%s: evidence_us is byte-identical to %s's; label it "
                                   "tuned_on with that board's key or drop it"
                                   % (where, other_name))
                continue
            if tuned not in owner:
                continue
            twin = next((d for d in tables[owner[tuned]]["entries"]
                         if entry_key(d) == entry_key(e)), None)
            if twin is None or twin.get("evidence_us") != ev:
                out.append("%s: tuned_on %s but %s has no entry with this evidence" %
                           (where, tuned, owner[tuned]))
                continue
            moved = [t for t in widths
                     if any(generic_route(e, t, p) != generic_route(twin, t, p)
                            for p in (False, True))
                     or route_as_at(e, t) != route_as_at(twin, t)]
            if moved:
                out.append("%s: evidence borrowed from %s, but the entry takes a different "
                           "route at T=%s" % (where, tuned, moved))
    return out


def table(cls, sms, entries):
    return {"gpu": {"class": cls, "board": {"sms": sms}}, "entries": entries}


def generic(bands, evidence=None, tuned=None, prefill=None, prefill_bands=None):
    e = {"format": "q8", "compute": "a16", "epilogue": "store", "n": 1, "k": 1, "bands": bands}
    if prefill:
        e["prefill"] = prefill
    if prefill_bands:
        e["prefill_bands"] = prefill_bands
    if evidence:
        e["evidence_us"] = evidence
    if tuned:
        e["tuned_on"] = tuned
    return e


def route_as(bands, evidence, tuned=None):
    e = {"format": "q8", "compute": "a16", "epilogue": "store", "n": 1, "k": 1,
         "route_as": bands, "evidence_us": evidence}
    if tuned:
        e["tuned_on"] = tuned
    return e


EV = [[4, 2.0, 1.0]]


class RouteEvidenceLint(unittest.TestCase):
    def test_all_tables(self):
        found = violations(load_tables())
        self.assertEqual(found, [], "\n" + "\n".join(found))

    def check(self, donor_entry, entry, count=0):
        """Donor table is aa (8 SMs); the entry under test lives in bb (4 SMs)."""
        tables = {"a.json": table("aa", 8, [donor_entry]), "b.json": table("bb", 4, [entry])}
        self.assertEqual(len(violations(tables)), count, violations(tables))

    def test_borrow_valid_and_moved_routes(self):
        donor = generic([[1, 16, "r.x"]], EV, "aa:8sm")
        self.check(donor, generic([[1, 16, "r.x"]], EV, "aa:8sm"))
        self.check(donor, generic([[1, 16, "r.y"]], EV, "aa:8sm"), count=1)
        self.check(donor, generic([[1, 2, "r.x"]], EV, "aa:8sm"), count=1)  # T=4 now a gap

    def test_class_only_key_and_free_text(self):
        donor = generic([[1, 16, "r.x"]], EV, "aa")
        self.check(donor, generic([[1, 16, "r.y"]], EV, "aa"), count=1)
        self.check(donor, generic([[1, 16, "r.y"]], [[4, 3.0, 1.0]], "qwen3_5_9b on the RTX 9999"))

    def test_donor_mismatch(self):
        donor = generic([[1, 16, "r.x"]], EV, "aa:8sm")
        self.check(donor, generic([[1, 16, "r.x"]], [[4, 3.0, 1.0]], "aa:8sm"), count=1)
        alone = {"b.json": table("bb", 4, [generic([[1, 16, "r.x"]], EV, "zz:1sm")])}
        self.assertEqual(violations(alone), [])  # an unknown board key is a label

    def test_prefill_profile_compared(self):
        ev = [[20, 2.0, 1.0]]
        donor = generic([[1, 16, "r.x"]], ev, "aa:8sm", prefill="p.1")
        self.check(donor, generic([[1, 16, "r.x"]], ev, "aa:8sm", prefill="p.1"))
        self.check(donor, generic([[1, 16, "r.x"]], ev, "aa:8sm", prefill="p.2"), count=1)
        self.check(donor, generic([[1, 16, "r.x"]], ev, "aa:8sm", prefill="p.1",
                                  prefill_bands=[[17, 64, "p.3"]]), count=1)

    def test_silent_copy_without_tuned_on(self):
        donor = generic([[1, 16, "r.x"]], EV)
        # the copy's direction is unknown, so both tables are named
        self.check(donor, generic([[1, 16, "r.y"]], EV), count=2)
        self.check(donor, generic([[1, 16, "r.y"]], EV, "bb:4sm"), count=2)  # names itself
        self.check(donor, generic([[1, 16, "r.y"]], [[4, 3.0, 1.0]]))

    def test_route_as_rules(self):
        donor = route_as([[2, 8, 12]], EV)
        self.check(donor, route_as([[2, 8, 12]], EV, "aa:8sm"))           # honest borrow
        self.check(donor, route_as([[2, 8, 16]], EV, "aa:8sm"), count=1)  # other T'
        self.check(donor, route_as([[6, 8, 12]], EV, "aa:8sm"), count=2)  # width outside bands
        self.check(donor, route_as([[2, 8, 12]], EV), count=2)            # silent copy


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])
