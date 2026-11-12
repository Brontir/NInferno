"""tools/bench/policy_moves_predict.py (#614 F1): the A5 ratio of a moved cell from routes CSVs."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import policy_moves_predict as predict  # noqa: E402

HEAD = "format,compute,epilogue,n,k,T,route,median_us,model_us\n"


def move(t, profile="speed"):
    return {"format": "q4k", "policy": "a8", "epilogue": "store", "n": 4096, "k": 4096, "T": t,
            "profile": profile, "legacy": {"route": "rmma.a"}, "evidence": {"route": "rmma.b"},
            "model_ratio": 0.9, "a5_ratio": None}


class Predict(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.doc = {"boards": [{"board": "rtx4090", "tables": "own",
                                "moves": [move(5), move(12), move(1024, "prefill")]},
                               {"board": "rtx4090", "tables": "without_own", "moves": [move(5)]}]}

    def tearDown(self):
        self.dir.cleanup()

    def csv(self, name, rows):
        path = self.root / name
        path.write_text(HEAD + "".join(r + "\n" for r in rows), encoding="utf-8")
        return path

    def test_median_ratio_of_the_two_routes(self):
        passes = [self.csv(f"p{i}.csv", [f"q4k,a8,store,4096,4096,5,rmma.a,{a},",
                                         f"q4k,a8,store,4096,4096,5,rmma.b,{b},",
                                         "q4k,a8,store,4096,4096,12,rmma.a,10,"])
                  for i, (a, b) in enumerate(((10, 8), (12, 9), (11, 30)), 1)]
        pairs = predict.predict(self.doc, "rtx4090", "own", predict.read_passes(passes))
        ratios = {m["T"]: r for m, r in pairs}
        self.assertAlmostEqual(ratios[5], 9 / 11)   # medians: legacy 11, evidence 9
        self.assertIsNone(ratios[12])               # the evidence route is not timed
        self.assertIsNone(ratios[1024])             # the prefill chunk is left out
        self.assertIsNone(self.doc["boards"][1]["moves"][0]["a5_ratio"])  # other tables kept

    def test_a_pass_without_the_route_leaves_it_null(self):
        one = self.csv("p1.csv", ["q4k,a8,store,4096,4096,5,rmma.a,10,",
                                  "q4k,a8,store,4096,4096,5,rmma.b,8,"])
        two = self.csv("p2.csv", ["q4k,a8,store,4096,4096,5,rmma.a,10,"])
        pairs = predict.predict(self.doc, "rtx4090", "own", predict.read_passes([one, two]))
        self.assertIsNone(dict((m["T"], r) for m, r in pairs)[5])

    def test_write(self):
        moves = self.root / "moves.json"
        moves.write_text(json.dumps(self.doc), encoding="utf-8")
        routes = self.csv("p1.csv", ["q4k,a8,store,4096,4096,5,rmma.a,10,",
                                     "q4k,a8,store,4096,4096,5,rmma.b,5,"])
        self.assertEqual(predict.main(["--moves", str(moves), "--board", "rtx4090", "--routes",
                                       str(routes), "--write"]), 0)
        written = json.loads(moves.read_text(encoding="utf-8"))
        self.assertAlmostEqual(written["boards"][0]["moves"][0]["a5_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
