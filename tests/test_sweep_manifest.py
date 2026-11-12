import contextlib
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import sweep_manifest as sm

ROUTES_NEW = ("format,compute,epilogue,n,k,T,route,median_us,model_us,registers,local_bytes,"
              "weight_bytes,activation_bytes,output_bytes,run,device,protocol,calls,samples\n"
              "q4_K,a16,store,4096,4096,1,r8.w4r1,8.25,9.12345678901234567,,,1,1,1,r1,ad102:128sm,copies,8,25\n"
              "q4_K,a16,store,4096,4096,32,tiled.64x128,30.5,28.1234567890123456,,,1,1,1,r1,ad102:128sm,flushed,1,25\n"
              "q6_K,a8,store,4096,12288,32,ggml_mmq.x64,40,,,,1,1,1,r1,ad102:128sm,flushed,1,25\n")
ROUTES_OLD = ("format,compute,epilogue,n,k,T,route,median_us,model_us\n"
              "q4_K,a16,store,4096,4096,17,tiled.64x128,20.1,19.1234\n")
SELECTED = ("format,compute,epilogue,n,k,T,asked,selected,select_ns,run,device\n"
            "q4_K,a16,store,4096,4096,1,r8,r8.w4r1,100,r1,ad102:128sm\n")


class Scan(unittest.TestCase):
    def tree(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / "rw").mkdir()
        (root / "rw" / "routes_p1.csv").write_text(ROUTES_NEW)
        (root / "rw" / "sel_p1.csv").write_text(SELECTED)
        (root / "old").mkdir()
        (root / "old" / "r.csv").write_text(ROUTES_OLD)
        (root / "other.csv").write_text("a,b\n1,2\n")  # no route_tune header: skipped
        (root / "build_x_89").mkdir()  # build trees are not walked
        (root / "build_x_89" / "r.csv").write_text(ROUTES_NEW)
        (root / "bad").mkdir()
        (root / "bad" / "short.csv").write_text("format,compute,epilogue,n,k,T,route,median_us\n"
                                                "q4_K,a16,store,4096,4096,notanumber,r,1\n")
        return tmp, root

    def test_records(self):
        tmp, root = self.tree()
        with tmp:
            doc = sm.scan(root, "rtx4090")
        self.assertEqual(doc["schema"], sm.SCHEMA)
        self.assertEqual(doc["card"], "rtx4090")
        by = {f["path"]: f for f in doc["files"]}
        self.assertEqual(sorted(by), ["bad/short.csv", "old/r.csv", "rw/routes_p1.csv",
                                      "rw/sel_p1.csv"])
        new = by["rw/routes_p1.csv"]
        self.assertEqual(new["kind"], "routes")
        self.assertEqual((new["rows"], new["problems"], new["widths"], new["t_min"], new["t_max"]),
                         (3, 2, 2, 1, 32))
        self.assertEqual(new["formats"], ["q4_K", "q6_K"])
        self.assertEqual(new["computes"], ["a16", "a8"])
        self.assertEqual(new["runs"], ["r1"])
        self.assertEqual(new["devices"], ["ad102:128sm"])
        self.assertEqual(new["protocols"], ["copies", "flushed"])
        self.assertEqual((new["model_us_rows"], new["model_us_digits"]), (2, 18))
        self.assertEqual(new["selected_in_dir"], ["rw/sel_p1.csv"])
        self.assertEqual(len(new["sha256"]), 64)
        old = by["old/r.csv"]
        # A sweep from before route_tune named its run, board and protocol says so.
        self.assertEqual((old["runs"], old["devices"], old["protocols"]),
                         ([sm.UNNAMED], [sm.UNNAMED], [sm.UNNAMED]))
        self.assertEqual(old["model_us_digits"], 6)  # too few to tell a tie
        self.assertEqual(old["selected_in_dir"], [])
        sel = by["rw/sel_p1.csv"]
        self.assertEqual((sel["kind"], sel["rows"], sel["authority_column"]), ("selected", 1, False))
        self.assertIn("error", by["bad/short.csv"])  # listed with its error, not dropped

    def test_a_cut_row_a_bom_and_a_vanished_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # A sweep killed mid-write: its last row stops after T. Recorded with its error.
            (root / "cut.csv").write_text("format,compute,epilogue,n,k,T,route,median_us\n"
                                          "q4k,a16,store,4096,4096,16,r8,1.5\n"
                                          "q4k,a16,store,4096,4096,32\n")
            # A byte-order mark before the header is still route_tune's header.
            (root / "bom.csv").write_bytes(b"\xef\xbb\xbf" + ROUTES_OLD.encode())
            by = {f["path"]: f for f in sm.scan(root, "rtx4090")["files"]}
            # The cut row is left out and named; the rest is described.
            self.assertNotIn("error", by["cut.csv"])
            self.assertEqual((by["cut.csv"]["rows"], by["cut.csv"]["cut_last_row"]), (1, 3))
            # A short row that is not the last makes the file unreadable.
            (root / "mid.csv").write_text("format,compute,epilogue,n,k,T,route,median_us\n"
                                          "q4k,a16,store,4096,4096,32\n"
                                          "q4k,a16,store,4096,4096,16,r8,1.5\n")
            by = {f["path"]: f for f in sm.scan(root, "rtx4090")["files"]}
            self.assertIn("line 2 has too few columns", by["mid.csv"]["error"])
            text = sm.summary(sm.merge([("s", {"schema": sm.SCHEMA, "card": "rtx4090",
                                              "files": list(by.values())})]))
            self.assertIn("(1 unreadable, 1 cut)", text)
            self.assertEqual((by["bom.csv"]["kind"], by["bom.csv"]["rows"]), ("routes", 1))
            # A file removed between the read and the stat is listed with its error.
            real = Path.stat

            def stat(self, *a, **k):
                if self.name == "bom.csv" and not k and not a:
                    raise FileNotFoundError(2, "No such file or directory")
                return real(self, *a, **k)

            with unittest.mock.patch.object(Path, "stat", stat):
                by = {f["path"]: f for f in sm.scan(root, "rtx4090")["files"]}
            self.assertIn("No such file", by["bom.csv"]["error"])

    def test_odd_files_do_not_stop_the_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "good.csv").write_text(ROUTES_OLD)
            # An unrelated CSV with bare CR line ends: no header csv reads, skipped.
            (root / "cr.csv").write_bytes(b"a,b\rc,d\r")
            # A field past csv's limit: listed with its error.
            (root / "big.csv").write_text("format,compute,epilogue,n,k,T,route,median_us\n"
                                          "q4k,a16,store,4096,4096,16,r" + "x" * 200000 + ",1\n")
            # A row cut inside its last field still has every column: the missing final newline
            # (route_tune ends every row with one) names it cut.
            (root / "nolf.csv").write_text("format,compute,epilogue,n,k,T,route,median_us\n"
                                           "q4k,a16,store,4096,4096,16,r8,1.5\n"
                                           "q4k,a16,store,4096,4096,32,r8,2.")
            # A blank line does not shift the line a short row is named by.
            (root / "blank.csv").write_text("format,compute,epilogue,n,k,T,route,median_us\n"
                                            "\n"
                                            "q4k,a16,store,4096,4096,32\n"
                                            "q4k,a16,store,4096,4096,16,r8,1.5\n")
            by = {f["path"]: f for f in sm.scan(root, "rtx4090")["files"]}
            self.assertEqual(sorted(by), ["big.csv", "blank.csv", "good.csv", "nolf.csv"])
            self.assertIn("field larger than field limit", by["big.csv"]["error"])
            self.assertEqual((by["nolf.csv"]["rows"], by["nolf.csv"]["cut_last_row"]), (1, 3))
            self.assertIn("line 3 has too few columns", by["blank.csv"]["error"])

    def test_build_trees_are_not_walked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for d in ("build", "build-release", "cmake-build-debug", "build_mst1_89", "wt_x", "builds"):
                (root / d).mkdir()
                (root / d / "r.csv").write_text(ROUTES_OLD)
            # "builds" is no build tree's name: it is walked.
            self.assertEqual([f["path"] for f in sm.scan(root, "rtx4090")["files"]], ["builds/r.csv"])

    def test_an_unlisted_directory_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "locked").mkdir()
            (root / "r.csv").write_text(ROUTES_OLD)
            real = Path.iterdir

            def iterdir(self):
                if self.name == "locked":
                    raise PermissionError(13, "Permission denied")
                return real(self)

            with unittest.mock.patch.object(Path, "iterdir", iterdir):
                files = sm.scan(root, "rtx4090")["files"]
            by = {f["path"]: f for f in files}
            self.assertEqual(sorted(by), ["locked", "r.csv"])
            self.assertIn("Permission denied", by["locked"]["error"])
            text = sm.summary(sm.merge([("s", {"schema": sm.SCHEMA, "card": "rtx4090", "files": files})]))
            self.assertIn("1 unreadable", text)

    def test_digits(self):
        self.assertEqual(sm.significant_digits("12.3456"), 6)
        self.assertEqual(sm.significant_digits("0.000125"), 3)
        self.assertEqual(sm.significant_digits("1.5e-05"), 2)
        self.assertEqual(sm.significant_digits("-7"), 1)

    def test_refusals(self):
        tmp, root = self.tree()
        with tmp:
            with self.assertRaises(ValueError):
                sm.scan(root, "RTX 4090")  # a card name, lowercase, no spaces
            with self.assertRaises(ValueError):
                sm.scan(root / "missing", "rtx4090")


class Merge(unittest.TestCase):
    def test_merge_and_twice(self):
        a = {"schema": sm.SCHEMA, "card": "rtx4090",
             "files": [{"path": "x.csv", "kind": "routes", "bytes": 10}]}
        b = {"schema": sm.SCHEMA, "card": "rtx3060",
             "files": [{"path": "y.csv", "kind": "selected", "bytes": 5},
                       {"path": "z.csv", "error": "unreadable"}]}
        doc = sm.merge([("a", a), ("b", b)])
        self.assertEqual(doc["cards"], ["rtx3060", "rtx4090"])
        self.assertEqual([(f["card"], f["path"]) for f in doc["files"]],
                         [("rtx3060", "y.csv"), ("rtx3060", "z.csv"), ("rtx4090", "x.csv")])
        self.assertEqual(doc["totals"]["rtx3060"], {"routes": 0, "selected": 1, "errors": 1, "bytes": 5})
        with self.assertRaises(ValueError):
            sm.merge([("a", a), ("a2", a)])
        with self.assertRaises(ValueError):
            sm.merge([("c", {"schema": "other", "card": "x", "files": []})])

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rw").mkdir()
            (root / "rw" / "routes_p1.csv").write_text(ROUTES_NEW.replace(
                "ad102:128sm,", "ad102:128sm GPU-0000aaaa-1111,"))
            (root / "rw" / "sel_p1.csv").write_text(SELECTED)
            (root / "r.csv").write_text(ROUTES_OLD)
            scan = sm.scan(root, "rtx4090")
        # The GPU UUID route_tune writes after the class id is not recorded.
        self.assertEqual(scan["files"][1]["devices"], ["ad102:128sm"])
        self.assertNotIn("GPU-", json.dumps(scan))
        text = sm.summary(sm.merge([("s", scan)]))
        self.assertIn("### rtx4090", text)
        self.assertIn("2 sweeps, 1 selections", text)
        rows = [l for l in text.splitlines() if l.startswith("| ") and not l.startswith("| directory")]
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0].startswith("| . | 1 | 0 | 1 | q4_K | 17..17 | unnamed | 6 | 0/1 |"))
        self.assertTrue(rows[1].startswith("| rw | 1 | 1 | 4 | q4_K q6_K | 1..32 | copies flushed | 18 | 2/2 |"))
        with self.assertRaises(ValueError):
            sm.summary(scan)  # a scan, not a merged manifest

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "r.csv").write_text(ROUTES_OLD)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(sm.main(["scan", str(root), "--card", "rtx3060"]), 0)
            scan = Path(tmp) / "scan.json"
            scan.write_text(out.getvalue())
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(sm.main(["merge", str(scan)]), 0)
            self.assertEqual(json.loads(out.getvalue())["cards"], ["rtx3060"])
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(sm.main(["scan", str(root / "nope"), "--card", "rtx3060"]), 2)
            self.assertIn("sweep_manifest: ", err.getvalue())


if __name__ == "__main__":
    unittest.main()
