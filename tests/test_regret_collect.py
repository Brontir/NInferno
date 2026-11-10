import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import regret_collect as collect
import selection_regret as regret

HEAD = "format,compute,epilogue,n,k,T,route,median_us,run,device,protocol,calls,samples\n"
SELECTED_HEAD = "format,compute,epilogue,n,k,T,asked,selected,run,device\n"


def stamp(text, run, sweep):
    """Rows as route_tune writes them: each names its run and board, a sweep row its protocol."""
    tail = f",{run},ga106:28sm" + (",flushed,1,15" if sweep else "")
    return "".join(line + tail + "\n" for line in text.splitlines() if line)


class Collect(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def run_collect(self, passes, selected, qualified="generic linear and route class tests"):
        sel = [self.write(f"sel{i}.csv", SELECTED_HEAD + stamp(selected, f"r{i}", False))
               for i in range(1, len(passes) + 1)]
        routes = [self.write(f"p{i}.csv", HEAD + stamp(text, f"r{i}", True))
                  for i, text in enumerate(passes, 1)]
        return collect.collect(sel, routes, "held-board", "held-model", "rev", "build", "both",
                               ["fit-board"], ["fit-model"], qualified)

    def test_regret_of_the_selected_route(self):
        rows = "q8,a16,store,4096,4096,4,rmma.a,12\nq8,a16,store,4096,4096,4,rmma.b,10\n"
        doc = self.run_collect([rows, rows], "q8,a16,store,4096,4096,4,rmma.a,rmma.a\n")
        report = regret.evaluate(json.loads(json.dumps(doc)))
        case = report["cases"][0]
        self.assertEqual(case["status"], "complete")
        self.assertEqual(case["best_observed"], "rmma.b")
        self.assertAlmostEqual(case["regret"], 0.2)
        self.assertEqual(case["passes"], 2)
        self.assertEqual(doc["qualification"], "generic linear and route class tests")

    def test_route_missing_from_a_pass_is_unmeasured(self):
        full = "q8,a16,store,64,64,1,rmma.a,5\nq8,a16,store,64,64,1,rmma.b,6\n"
        part = "q8,a16,store,64,64,1,rmma.a,5\n"
        doc = self.run_collect([full, part], "q8,a16,store,64,64,1,rmma.a,rmma.a\n")
        b = [c for c in doc["cases"][0]["candidates"] if c["id"] == "rmma.b"][0]
        self.assertEqual(b["status"], "unmeasured")
        self.assertEqual(regret.evaluate(doc)["cases"][0]["status"], "incomplete")

    def test_untimed_selection_is_listed(self):
        rows = "q8,a16,store,64,64,1,rmma.a,5\nq8,a16,store,64,64,1,rmma.b,6\n"
        doc = self.run_collect([rows], "q8,a16,store,64,64,1,ggml_vec,ggml_vec\n")
        ids = {c["id"]: c["status"] for c in doc["cases"][0]["candidates"]}
        self.assertEqual(ids["ggml_vec"], "unmeasured")
        self.assertEqual(regret.evaluate(doc)["cases"][0]["status"], "incomplete")

    def test_without_qualification_nothing_is_ok(self):
        rows = "q8,a16,store,64,64,1,rmma.a,5\nq8,a16,store,64,64,1,rmma.b,6\n"
        doc = self.run_collect([rows], "q8,a16,store,64,64,1,rmma.a,rmma.a\n", qualified="")
        self.assertTrue(all(c["status"] == "unmeasured" for c in doc["cases"][0]["candidates"]))
        self.assertNotIn("qualification", doc)
        doc = self.run_collect([rows], "q8,a16,store,64,64,1,rmma.a,rmma.a\n", qualified="  ")
        self.assertTrue(all(c["status"] == "unmeasured" for c in doc["cases"][0]["candidates"]))

    def test_other_compute_class_is_ineligible(self):
        rows = ("q4k,a16,store,64,64,12,rmma.a,5\nq4k,a16,store,64,64,12,rmma.b,6\n"
                "q4k,a16,store,64,64,12,ggml_mmq,3\n")
        doc = self.run_collect([rows], "q4k,a16,store,64,64,12,rmma.a,rmma.a\n")
        mmq = [c for c in doc["cases"][0]["candidates"] if c["id"] == "ggml_mmq"][0]
        self.assertEqual(mmq["status"], "ineligible")
        case = regret.evaluate(doc)["cases"][0]
        self.assertEqual(case["status"], "complete")
        self.assertEqual(case["best_observed"], "rmma.a")

    def test_conflicting_selections_refused(self):
        rows = "q8,a16,store,64,64,1,rmma.a,5\n"
        with self.assertRaises(ValueError):
            self.run_collect([rows], "q8,a16,store,64,64,1,rmma.a,rmma.a\n"
                                     "q8,a16,store,64,64,1,rmma.b,rmma.b\n")

    def test_route_timed_twice_in_a_pass_refused(self):
        rows = "q8,a16,store,64,64,1,rmma.a,5\nq8,a16,store,64,64,1,rmma.a,4\n"
        with self.assertRaises(ValueError):
            self.run_collect([rows], "q8,a16,store,64,64,1,rmma.a,rmma.a\n")

    def test_passes_with_different_selections_refused(self):
        rows = "q8,a16,store,64,64,1,rmma.a,5\nq8,a16,store,64,64,1,rmma.b,6\n"
        a = self.write("sa.csv", SELECTED_HEAD + stamp("q8,a16,store,64,64,1,rmma.a,rmma.a", "r1",
                                                       False))
        b = self.write("sb.csv", SELECTED_HEAD + stamp("q8,a16,store,64,64,1,rmma.a,rmma.b", "r2",
                                                       False))
        a2 = self.write("sa2.csv", SELECTED_HEAD + stamp("q8,a16,store,64,64,1,rmma.a,rmma.a", "r2",
                                                         False))
        p1 = self.write("p1.csv", HEAD + stamp(rows, "r1", True))
        p2 = self.write("p2.csv", HEAD + stamp(rows, "r2", True))
        with self.assertRaises(ValueError):
            collect.collect([a, b], [p1, p2], "b", "m", "r", "i", "both", [], [], "q")
        with self.assertRaises(ValueError):  # one selection for two passes
            collect.collect([a], [p1, p2], "b", "m", "r", "i", "both", [], [], "q")
        doc = collect.collect([a, a2], [p1, p2], "b", "m", "r", "i", "both", [], [], "q")
        self.assertIn("A16Only", doc["cases"][0]["numerical_contract"])
        self.assertIn("speed route profile", doc["cases"][0]["workload"])

    # #518: route_tune names each row's run, board and timing protocol.
    NEW_HEAD = "format,compute,epilogue,n,k,T,route,median_us,run,device,protocol,calls,samples\n"
    NEW_SELECTED_HEAD = "format,compute,epilogue,n,k,T,asked,selected,run,device\n"

    def new_pass(self, i, run, rows, device="ga106:28sm"):
        routes = "".join(f"q8,a16,store,64,64,{t},{r},{us},{run},{device},{p},{c},15\n"
                         for t, r, us, p, c in rows)
        sel = "".join(f"q8,a16,store,64,64,{t},rmma.a,rmma.a,{run},{device}\n"
                      for t in sorted({row[0] for row in rows}))
        return (self.write(f"nsel{i}.csv", self.NEW_SELECTED_HEAD + sel),
                self.write(f"np{i}.csv", self.NEW_HEAD + routes))

    def collect_new(self, pairs):
        return collect.collect([s for s, _ in pairs], [r for _, r in pairs], "b", "m", "r", "i",
                               "both", [], [], "q")

    def test_protocol_is_reported_per_case(self):
        rows = [(4, "rmma.a", 5, "copies", 64), (4, "rmma.b", 6, "copies", 64),
                (24, "rmma.a", 50, "flushed", 1), (24, "rmma.b", 40, "flushed", 1)]
        doc = self.collect_new([self.new_pass(1, "x-1", rows), self.new_pass(2, "x-2", rows)])
        work = {c["id"]: c["workload"] for c in doc["cases"]}
        self.assertIn("64 consecutive calls over weight copies", work["q8.a16.store.n64.k64.T4"])
        self.assertIn("one call after an L2 flush", work["q8.a16.store.n64.k64.T24"])
        self.assertEqual(doc["device"], "ga106:28sm")

    def test_passes_with_different_protocols_refused(self):
        rotated = [(4, "rmma.a", 5, "copies", 64)]
        flushed = [(4, "rmma.a", 7, "flushed", 1)]
        with self.assertRaisesRegex(ValueError, "different protocols"):
            self.collect_new([self.new_pass(1, "x-1", rotated), self.new_pass(2, "x-2", flushed)])

    def test_warm_samples_are_their_own_protocol(self):
        # #574: a timer without a flush buffer times a warm L2; its rows say so, and a pass of
        # them is not one of flushed rows.
        warm = [(24, "rmma.a", 30, "warm", 1), (24, "rmma.b", 25, "warm", 1)]
        doc = self.collect_new([self.new_pass(1, "x-1", warm), self.new_pass(2, "x-2", warm)])
        work = {c["id"]: c["workload"] for c in doc["cases"]}
        self.assertIn("warm L2", work["q8.a16.store.n64.k64.T24"])
        flushed = [(24, "rmma.a", 50, "flushed", 1), (24, "rmma.b", 40, "flushed", 1)]
        with self.assertRaisesRegex(ValueError, "different protocols"):
            self.collect_new([self.new_pass(1, "x-1", warm), self.new_pass(2, "x-2", flushed)])

    def test_files_of_different_runs_refused(self):
        rows = [(4, "rmma.a", 5, "copies", 64)]
        sel, _ = self.new_pass(1, "x-1", rows)
        _, routes = self.new_pass(2, "x-2", rows)
        with self.assertRaisesRegex(ValueError, "different runs"):
            self.collect_new([(sel, routes)])

    def test_one_run_given_twice_refused(self):
        pair = self.new_pass(1, "x-1", [(4, "rmma.a", 5, "copies", 64)])
        with self.assertRaisesRegex(ValueError, "more than once"):
            self.collect_new([pair, pair])

    def test_passes_of_different_boards_refused(self):
        rows = [(4, "rmma.a", 5, "copies", 64)]
        with self.assertRaisesRegex(ValueError, "more than one board"):
            self.collect_new([self.new_pass(1, "x-1", rows),
                              self.new_pass(2, "x-2", rows, device="ad106:34sm")])

    def test_files_without_provenance_refused(self):
        old_sel = self.write("old_sel.csv", "format,compute,epilogue,n,k,T,asked,selected\n"
                                            "q8,a16,store,64,64,4,rmma.a,rmma.a\n")
        old_routes = self.write("old_p.csv", "format,compute,epilogue,n,k,T,route,median_us\n"
                                             "q8,a16,store,64,64,4,rmma.a,5\n")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            collect.collect([old_sel], [old_routes], "b", "m", "r", "i", "both", [], [], "q")

    def test_short_row_exits_2(self):
        sel = self.write("sel.csv", SELECTED_HEAD + "q8,a16\n")
        routes = self.write("p1.csv", HEAD + "q8,a16,store,64,64,1,rmma.a,5\n")
        args = ["--selected", str(sel), "--routes", str(routes), "--board", "b", "--model", "m",
                "--selector-revision", "r", "--build-id", "i", "--holdout", "both"]
        self.assertEqual(collect.main(args), 2)

    def test_missing_column_refused(self):
        sel = self.write("sel.csv", "format,compute\nq8,a16\n")
        with self.assertRaises(ValueError):
            collect.collect(sel, [self.write("p1.csv", HEAD)], "b", "m", "r", "i", "both", [], [],
                            "q")

    def collect_with_authority(self, columns, extras, facts=None, rows=1):
        """One pass per element of `extras`, whose selected CSV carries `columns` after `device`
        with that element's values on its one row."""
        sweep = "q8,a16,store,64,64,1,rmma.a,10\nq8,a16,store,64,64,1,rmma.b,12\n"
        sel = []
        for i, extra in enumerate(extras, 1):
            head = SELECTED_HEAD.rstrip("\n") + "".join("," + c for c in columns) + "\n"
            row = f"q8,a16,store,64,64,1,rmma.a,rmma.a,r{i},ga106:28sm{extra}\n"
            sel.append(self.write(f"sel{i}.csv", head + row * rows))
        routes = [self.write(f"p{i}.csv", HEAD + stamp(sweep, f"r{i}", True))
                  for i in range(1, len(extras) + 1)]
        return collect.collect(sel, routes, "held-board", "held-model", "rev", "build", "both",
                               [], [], "tests", facts)

    def test_authority_and_evidence_columns_reach_the_cases(self):
        doc = self.collect_with_authority(["authority", "evidence"],
                                          [",route-table,measured", ",route-table,measured"])
        (case,) = doc["cases"]
        self.assertEqual((case["authority"], case["evidence"], case["t"]),
                         ("route-table", "measured", 1))

    def test_old_selected_files_read_unknown(self):
        doc = self.collect_with_authority([], ["", ""])
        (case,) = doc["cases"]
        self.assertEqual((case["authority"], case["evidence"]), ("unknown", "unknown"))
        empty = self.collect_with_authority(["authority", "evidence"], [",,", ",,"])
        self.assertEqual(empty["cases"][0]["authority"], "unknown")
        # and the document evaluates, grouped under "unknown"
        report = regret.evaluate(json.loads(json.dumps(doc)))
        self.assertEqual(report["by_board_regime_authority"][0]["authority"], "unknown")

    def test_passes_naming_another_authority_are_refused(self):
        with self.assertRaisesRegex(ValueError, "another authority"):
            self.collect_with_authority(["authority", "evidence"],
                                        [",route-table,measured", ",cost-model,ranker"])

    def test_board_facts_from_the_registry_for_the_ridge(self):
        root = Path(__file__).resolve().parents[1]
        facts = collect.board_facts(root / "src/core/data/gpu_facts.json", "0x2B85")
        self.assertEqual((facts["cc"], facts["sms"], facts["boost_mhz"]), (120, 170, 2407))
        self.assertAlmostEqual(facts["dram_gbps"], 512 * 28.0 / 8)
        self.assertIs(facts["mma_slow"], False)
        # A GTX 1660 Ti has no tensor cores: the fact travels with the board.
        self.assertIs(collect.board_facts(root / "src/core/data/gpu_facts.json", "0x2182")["mma_slow"],
                      True)
        with self.assertRaises(ValueError):
            collect.board_facts(root / "src/core/data/gpu_facts.json", "0xFFFF")
        doc = self.run_collect(["q8,a16,store,64,64,1,rmma.a,10\n"],
                               "q8,a16,store,64,64,1,rmma.a,rmma.a\n")
        self.assertNotIn("board_facts", doc)

    def test_a_pass_answering_twice_with_different_authority_is_refused(self):
        with self.assertRaisesRegex(ValueError, "answered twice"):
            sel = self.write("twice.csv", SELECTED_HEAD.rstrip("\n") + ",authority,evidence\n"
                             + "q8,a16,store,64,64,1,rmma.a,rmma.a,r1,ga106:28sm,a,measured\n"
                             + "q8,a16,store,64,64,1,rmma.a,rmma.a,r1,ga106:28sm,b,measured\n")
            collect.read_selections([sel], {})

    def test_the_message_names_which_column_differs(self):
        with self.assertRaisesRegex(ValueError, "another evidence"):
            self.collect_with_authority(["authority", "evidence"],
                                        [",route-table,measured", ",route-table,borrowed"])
        with self.assertRaisesRegex(ValueError, "another authority"):
            self.collect_with_authority(["authority", "evidence"],
                                        [",route-table,measured", ",cost-model,measured"])

    def test_a_wrong_pci_id_is_refused_by_the_device_class(self):
        facts_json = Path(__file__).resolve().parents[1] / "src/core/data/gpu_facts.json"
        rtx5090 = collect.board_facts(facts_json, "0x2B85")  # gb202, 170 SMs
        self.assertEqual(rtx5090["chip"], "gb202")
        with self.assertRaisesRegex(ValueError, "170 SMs"):
            self.collect_with_authority([], [""], facts=rtx5090)  # the rows say ga106:28sm
        # The SMs alone do not name the board: another chip of 28 SMs is refused too.
        with self.assertRaisesRegex(ValueError, r"\(gb202, sm120, 28 SMs\)"):
            self.collect_with_authority([], [""], facts=dict(rtx5090, sms=28))
        right = dict(rtx5090, chip="ga106", sms=28)
        doc = self.collect_with_authority([], [""], facts=right)
        self.assertEqual(doc["board_facts"]["held-board"]["sms"], 28)

    def test_the_ranker_column_reaches_the_case(self):
        doc = self.collect_with_authority(["authority", "evidence", "ranker"],
                                          [",route-table,measured,rmma.b"] * 2)
        self.assertEqual(doc["cases"][0]["ranker"], "rmma.b")
        report = regret.evaluate(json.loads(json.dumps(doc)))
        # selected rmma.a 10 ms against the ranker's rmma.b 12 ms: 20 % slower than the selected
        self.assertAlmostEqual(report["cases"][0]["ranker_vs_selected"], 0.2)
        none = self.collect_with_authority(["authority", "evidence"], [",route-table,measured"] * 2)
        self.assertNotIn("ranker", none["cases"][0])
        with self.assertRaisesRegex(ValueError, "another ranker"):
            self.collect_with_authority(["authority", "evidence", "ranker"],
                                        [",route-table,measured,rmma.a",
                                         ",route-table,measured,rmma.b"])

    def collect_on(self, device, pci_id):
        """One case measured on `device`, collected with the registry row of `pci_id`."""
        facts_json = Path(__file__).resolve().parents[1] / "src/core/data/gpu_facts.json"
        sel = self.write("s.csv", SELECTED_HEAD
                         + f"q8,a16,store,64,64,1,rmma.a,rmma.a,r1,{device}\n")
        routes = self.write("p.csv", HEAD
                            + f"q8,a16,store,64,64,1,rmma.a,10,r1,{device},flushed,1,15\n")
        return collect.collect([sel], [routes], "held-board", "held-model", "rev", "build",
                               "both", [], [], "tests", collect.board_facts(facts_json, pci_id))

    def collect_recorded(self, sel_pci, route_pci, pci_id=None, facts_json=None):
        """One case of an ad104:60sm board whose rows record a PCI id (#684)."""
        facts_json = facts_json or Path(__file__).resolve().parents[1] / "src/core/data/gpu_facts.json"
        device = "ad104:60sm GPU-0a1b2c3d"
        sel = self.write("s.csv", SELECTED_HEAD.rstrip("\n") + ",pci_device_id\n"
                         + f"q8,a16,store,64,64,1,rmma.a,rmma.a,r1,{device},{sel_pci}\n")
        routes = self.write("p.csv", HEAD.rstrip("\n") + ",pci_device_id\n"
                            + f"q8,a16,store,64,64,1,rmma.a,10,r1,{device},flushed,1,15,{route_pci}\n")
        facts = collect.board_facts(facts_json, pci_id) if pci_id else None
        return collect.collect([sel], [routes], "held-board", "held-model", "rev", "build",
                               "both", [], [], "tests", facts, gpu_facts=facts_json)

    def test_a_recorded_pci_id_tells_two_skus_of_one_class_apart(self):
        # An RTX 4070 Ti (0x2782) and an RTX 4500 Ada (0x27b1) are both ad104:60sm: the class
        # check accepts either, the recorded id only the board the rows ran on.
        with self.assertRaisesRegex(ValueError, r"RTX 4500 Ada Generation \(0x27b1\).*0x2782"):
            self.collect_recorded("0x2782", "0x2782", pci_id="0x27B1")
        doc = self.collect_recorded("0x2782", "0x2782", pci_id="0x2782")
        self.assertEqual(doc["pci_device_id"], "0x2782")
        self.assertEqual(doc["board_facts"]["held-board"]["dram_gbps"], 504.0)

    def test_without_pci_id_the_recorded_id_reads_the_board_facts(self):
        doc = self.collect_recorded("0x27B1", "0x27b1")
        self.assertEqual(doc["board_facts"]["held-board"]["name"], "RTX 4500 Ada Generation")
        self.assertEqual(doc["board_facts"]["held-board"]["dram_gbps"], 432.0)

    def test_an_id_the_registry_does_not_list_still_collects(self):
        # A new SKU the engine runs by its class: the document is written, without a ridge.
        doc = self.collect_recorded("0x2fff", "0x2fff")
        self.assertEqual(doc["pci_device_id"], "0x2fff")
        self.assertNotIn("board_facts", doc)

    def test_a_malformed_registry_row_is_still_an_error(self):
        row = {"id": "0x2782", "name": "GeForce RTX 4070 Ti", "chip": "ad104", "cc": 89, "sms": 60,
               "bus_bits": 192, "gbps": 21, "boost_mhz": 2610, "mma_slow": "no"}
        bad = self.write("facts.json", json.dumps({"gpus": [row]}))
        with self.assertRaisesRegex(ValueError, "mma_slow is not true or false"):
            self.collect_recorded("0x2782", "0x2782", facts_json=bad)

    def test_an_empty_or_absent_pci_id_keeps_the_class_check(self):
        doc = self.collect_recorded("", "")
        self.assertNotIn("pci_device_id", doc)
        self.assertNotIn("board_facts", doc)
        doc = self.collect_recorded("", "", pci_id="0x27B1")  # the class check alone accepts it
        self.assertEqual(doc["board_facts"]["held-board"]["dram_gbps"], 432.0)

    def test_files_of_two_pci_ids_are_refused(self):
        with self.assertRaisesRegex(ValueError, "PCI ids 0x2782, 0x27b1"):
            self.collect_recorded("0x2782", "0x27b1")
        with self.assertRaisesRegex(ValueError, "not a hex id"):
            self.collect_recorded("0x2782", "rtx")

    def test_a_board_of_the_same_sms_and_another_chip_is_refused(self):
        # A GTX 1660 Ti's rows (tu116:24sm, with the device's UUID after the class id) and the
        # PCI id of an RTX 4060 (ad107, 24 SMs): the Ada ridge is not this board's.
        device = "tu116:24sm GPU-0a1b2c3d"
        with self.assertRaisesRegex(ValueError, r"GeForce RTX 4060 \(ad107, sm89, 24 SMs\)"):
            self.collect_on(device, "0x2882")
        self.assertIs(self.collect_on(device, "0x2182")["board_facts"]["held-board"]["mma_slow"],
                      True)

    def test_a_board_the_class_names_by_its_architecture_is_accepted(self):
        # The class tables hold no chip for an H100 or a B200: their class id is the architecture
        # and the SMs, and the registry row of that architecture and SM count is the board.
        for device, pci_id, sms in (("sm90:132sm GPU-0a1b2c3d", "0x2330", 132),
                                    ("sm90:114sm", "0x2331", 114), ("sm100:148sm", "0x2901", 148)):
            with self.subTest(device=device):
                doc = self.collect_on(device, pci_id)
                self.assertEqual(doc["board_facts"]["held-board"]["sms"], sms)
        # Another architecture, or the other H100's SMs, is still another board.
        with self.assertRaisesRegex(ValueError, r"\(gh100, sm90, 114 SMs\)"):
            self.collect_on("sm90:132sm", "0x2331")
        with self.assertRaisesRegex(ValueError, r"\(gb100, sm100, 148 SMs\)"):
            self.collect_on("sm90:148sm", "0x2901")

    def test_a_device_that_is_not_a_class_id_cannot_take_board_facts(self):
        # A hand-made CSV with another device name: nothing says the PCI id is its board.
        for device in ("board-x", "unknown", "24sm", "tu116:24"):
            with self.subTest(device=device), \
                    self.assertRaisesRegex(ValueError, "not a class id"):
                self.collect_on(device, "0x2182")

    def test_a_registry_flag_that_is_not_a_boolean_is_refused(self):
        row = {"id": "0x2182", "name": "a board", "chip": "tu116", "cc": 75, "sms": 24,
               "bus_bits": 192, "gbps": 12.0, "boost_mhz": 1770, "mma_slow": "false"}
        path = self.write("facts.json", json.dumps({"gpus": [row]}))
        with self.assertRaisesRegex(ValueError, "mma_slow"):
            collect.board_facts(path, "0x2182")


if __name__ == "__main__":
    unittest.main()
