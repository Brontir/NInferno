"""The small-T eta rows of src/spec/round_cost.h (kSmallTEta) against tools/fits/round_cost/fit_eta.py.

    python3 tests/test_round_cost_eta_fit.py

The constants are fitted: fit_eta.py reads the shipped registry's pairs and prints one eta per
generation, and the header's rows and comment must be what its fit() returns, at 3 decimals. A card
is counted as a chip and its SM count; the tool's printout counts board names, which for one chip
under two names (#692) gives a larger number. A registry refresh that
moves the fit fails here until the rows are re-read from the tool (and their hold-out re-checked).
No GPU.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "fits" / "round_cost"))
import fit_eta  # noqa: E402

HEADER = (ROOT / "src" / "spec" / "round_cost.h").read_text(encoding="utf-8")
ROW = re.compile(r'^\s*\{"(\w+)",\s*([0-9.]+),\s*(\d+)\},?\s*$', re.M)
COMMENT = re.compile(r"^//   (\w+)\s+([0-9.]+)\s+RTX", re.M)


def header_rows() -> dict[str, tuple[float, int]]:
    start = HEADER.index("inline constexpr SmallTEta kSmallTEta[]")
    end = HEADER.index("};", start)
    table = HEADER[start:end]
    rows = ROW.findall(table)
    # Every entry of the table must parse: a row of another shape would otherwise drop out unseen.
    entries = len(re.findall(r'\{"', table))
    if entries != len(rows):
        raise AssertionError(f"kSmallTEta has {entries} entries, {len(rows)} of them in the shape this test reads")
    return {g: (float(eta), int(cards)) for g, eta, cards in rows}


def fitted_rows() -> dict[str, tuple[float, int]]:
    """{generation: (eta, lab cards)} as fit_eta.py fits them: the rows a generation's cards share."""
    out = {}
    rows = [r for r in fit_eta.rows_of_registry() if fit_eta.fitted(r)]
    for gen in sorted({r["board"]["gen"] for r in rows}):
        sel = [r for r in rows if r["board"]["gen"] == gen]
        # A card is a chip and its SM count (the registry's tune key), whatever board name it carries.
        cards = {(fit_eta.BOARDS[r["p"]["lab_board"].upper()]["chip"], fit_eta.BOARDS[r["p"]["lab_board"].upper()]["sms"])
                 for r in sel}
        out[gen] = (fit_eta.fit(sel), len(cards))
    return out


class EtaFitAgreement(unittest.TestCase):
    def test_header_rows_are_what_the_tool_prints(self):
        header, fitted = header_rows(), fitted_rows()
        self.assertTrue(header, "no kSmallTEta rows found in src/spec/round_cost.h")
        for gen, (eta, cards) in header.items():
            self.assertIn(gen, fitted, f"{gen}: kSmallTEta has a row, fit_eta.py has no rounds for it")
            tool_eta, tool_cards = fitted[gen]
            self.assertEqual(f"{eta:.3f}", f"{tool_eta:.3f}",
                             f"{gen}: kSmallTEta {eta:.3f}, fit_eta.py prints {tool_eta:.3f} on the shipped registry")
            self.assertEqual(cards, tool_cards, f"{gen}: kSmallTEta rests on {cards} cards, the registry has {tool_cards}")

    def test_a_row_rests_on_two_cards_or_more(self):
        for gen, (_, cards) in header_rows().items():
            self.assertGreaterEqual(cards, 2, f"{gen}: a generation constant rests on two lab cards or more")

    def test_comment_lists_the_same_etas(self):
        header = header_rows()
        commented = {g: float(e) for g, e in COMMENT.findall(HEADER)}
        for gen, (eta, _) in header.items():
            self.assertIn(gen, commented, f"{gen}: the comment above kSmallTEta has no line for it")
            self.assertAlmostEqual(commented[gen], eta, places=3, msg=f"{gen}: comment and row differ")


if __name__ == "__main__":
    unittest.main()
