"""tools/audit/facts_compare.py and facts_vs_driver.py on canned driver output, no GPU (#612 step 2).

The canned device is an RTX 4090 as ninferno-facts-probe prints it: the driver's values in
"queried", synthetic_device_class's architecture limits in "synthetic", the raw attributes the
query does not keep. Each test changes one thing and checks the field that must fail.
"""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "audit"))
import facts_compare  # noqa: E402
import facts_vs_driver  # noqa: E402

MIB = 1024 * 1024
FEATURES = {"mma_bf16": True, "mma_bf16_native": True, "mma_tensor": True, "mma_fp8": True,
            "mma_f8f6f4": False, "mma_mxf4nvf4": False, "tma": False, "pdl": False,
            "clusters": False, "mma_slow": False}


def device_4090():
    queried = {"name": "NVIDIA GeForce RTX 4090", "pci_device_id": 0x2684, "chip": "ad102",
               "major": 8, "minor": 9, "sms": 128, "l2_bytes": 72 * MIB,
               "global_bytes": 25_757_220_864, "smem_per_block_optin": 101376,
               "smem_per_sm": 102400, "max_threads_per_sm": 1536, "max_blocks_per_sm": 24,
               "registers_per_sm": 65536, "persisting_l2_max_bytes": 54 * MIB,
               "memory_bus_bits": 384, "memory_clock_khz": 10_501_000, "sm_clock_khz": 2_520_000,
               "features": dict(FEATURES)}
    synthetic = dict(copy.deepcopy(queried), name="synthetic sm89")
    return {"index": 0, "queried": queried, "synthetic": synthetic,
            "raw": {"reserved_shared_per_block": 1024, "shared_per_block_default": 49152,
                    "registers_per_block": 65536, "max_threads_per_block": 1024, "warp_size": 32},
            "engine_table_facts": {"source": 1, "sku": "GeForce RTX 4090",
                                   "reference_clock_khz": 2_520_000, "memory_bus_bits": 384,
                                   "memory_clock_khz": 10_500_000, "l2_mib": 72.0}}


ROWS = facts_vs_driver.load_rows(facts_vs_driver.FACTS)


def statuses(device):
    row = ROWS.get(device["queried"]["pci_device_id"])
    return {(r["source"], r["field"]): r["status"] for r in facts_compare.compare_device(device, row)}


class Compare(unittest.TestCase):
    def test_a_matching_board_has_no_mismatch(self):
        s = statuses(device_4090())
        self.assertNotIn("mismatch", s.values(), [k for k, v in s.items() if v == "mismatch"])
        self.assertEqual(s[("table", "sms")], "match")
        self.assertEqual(s[("table", "memory_clock_khz")], "match")  # 10 501 000 within 0.5 % of 10.5 M
        self.assertEqual(s[("arch", "smem_optin_plus_reserved_is_per_sm")], "match")

    def test_each_table_field_fails_alone(self):
        for field, change in (
                ("sms", lambda q: q.update(sms=130)),
                ("l2_mib", lambda q: q.update(l2_bytes=64 * MIB)),
                ("bus_bits", lambda q: q.update(memory_bus_bits=300)),  # no whole channel count
                ("memory_clock_khz", lambda q: q.update(memory_clock_khz=10_600_000)),  # +0.95 %
                ("boost_clock_khz", lambda q: q.update(sm_clock_khz=2_200_000)),
                ("vram_gib", lambda q: q.update(global_bytes=12 * 1024 * MIB)),
                ("cc", lambda q: q.update(minor=6))):
            d = device_4090()
            change(d["queried"])
            self.assertEqual(statuses(d)[("table", field)], "mismatch", field)

    def test_memory_clock_tolerance_edge(self):
        for khz, want in ((10_500_000 * 1.0049, "match"), (10_500_000 * 0.9951, "match"),
                          (10_500_000 * 1.0051, "mismatch")):
            d = device_4090()
            d["queried"]["memory_clock_khz"] = int(khz)
            self.assertEqual(statuses(d)[("table", "memory_clock_khz")], want, khz)

    def test_a_driver_that_does_not_say_is_unavailable_not_a_failure(self):
        d = device_4090()
        d["queried"].update(memory_bus_bits=0, memory_clock_khz=-1, sm_clock_khz=0)
        s = statuses(d)
        for field in ("bus_bits", "memory_clock_khz", "boost_clock_khz"):
            self.assertEqual(s[("table", field)], "unavailable", field)
        self.assertNotIn("mismatch", [v for k, v in s.items() if k[0] == "table" and not k[1].startswith("engine_")])

    def test_unknown_pci_id_has_no_row(self):
        d = device_4090()
        d["queried"]["pci_device_id"] = 0xDEAD
        self.assertEqual(statuses(d)[("table", "row")], "mismatch")

    def test_synthetic_limits_against_the_queried_ones(self):
        # The Volta case: the driver says 2048 threads an SM where the synthetic class wrote 1536.
        d = device_4090()
        d["queried"]["max_threads_per_sm"] = 2048
        d["queried"]["max_blocks_per_sm"] = 32
        s = statuses(d)
        self.assertEqual(s[("arch", "max_threads_per_sm")], "mismatch")
        self.assertEqual(s[("arch", "max_blocks_per_sm")], "mismatch")
        self.assertEqual(s[("arch", "registers_per_sm")], "match")
        d = device_4090()
        d["raw"]["reserved_shared_per_block"] = 0  # opt-in + reserved no longer the per-SM figure
        self.assertEqual(statuses(d)[("arch", "smem_optin_plus_reserved_is_per_sm")], "mismatch")

    def test_features_and_persisting_l2_are_information(self):
        d = device_4090()
        d["queried"]["features"]["pdl"] = True
        d["queried"]["persisting_l2_max_bytes"] = 40 * MIB
        s = statuses(d)
        self.assertEqual(s[("arch", "feature_pdl")], "info")
        self.assertEqual(s[("arch", "persisting_l2_max_bytes")], "info")

    def test_vendor_boost_is_information(self):
        # A partner board's VBIOS boost (+5.8 %, a 2080 Ti read 1635 against 1545) or an OEM
        # board's lower one is not a failure; more than 10 % off is.
        for khz, want in ((2_520_000 * 1.058, "info"), (2_520_000 * 1.014, "match"),
                          (2_520_000 * 0.98, "info"), (2_520_000 * 1.12, "mismatch"),
                          (2_520_000 * 0.88, "mismatch")):
            d = device_4090()
            d["queried"]["sm_clock_khz"] = int(khz)
            self.assertEqual(statuses(d)[("table", "boost_clock_khz")], want, khz)

    def test_ecc_memory_is_not_a_failure(self):
        # GDDR ECC keeps 1/16: 24 GiB reads about 22.3 GiB (0.928).
        d = device_4090()
        d["queried"]["global_bytes"] = int(24 * 1024 * MIB * 0.928)
        self.assertEqual(statuses(d)[("table", "vram_gib")], "match")
        d["queried"]["global_bytes"] = int(24 * 1024 * MIB * 0.91)
        self.assertEqual(statuses(d)[("table", "vram_gib")], "mismatch")

    def assert_no_mismatch(self, d):
        s = statuses(d)
        self.assertNotIn("mismatch", s.values(), [k for k, v in s.items() if v == "mismatch"])
        return s

    def test_memory_variant_is_read_from_the_driver(self):
        # A board on a narrower bus and a slower memory under the row's id (an 8 GB 3060, a GDDR6
        # 4070): the driver's bus and clock are information, and the engine prices them.
        d = device_4090()
        d["queried"].update(memory_bus_bits=256, memory_clock_khz=9_000_000)
        d["engine_table_facts"].update(source=2, memory_bus_bits=256,
                                       memory_clock_khz=round(facts_compare.snap_to_fact_grid(9_000_000)))
        s = self.assert_no_mismatch(d)
        self.assertEqual(s[("table", "bus_bits")], "info")
        self.assertEqual(s[("table", "memory_clock_khz")], "info")
        self.assertEqual(s[("table", "engine_source")], "match")
        d["engine_table_facts"]["memory_bus_bits"] = 384  # the engine kept the row: wrong
        self.assertEqual(statuses(d)[("table", "engine_memory_bus_bits")], "mismatch")

    @staticmethod
    def device_of_row(pci_id, **driver):
        """A whole board as its row states it, with the driver's values given."""
        row = ROWS[pci_id]
        d = device_4090()
        d["queried"].update(pci_device_id=pci_id, name="NVIDIA " + row["name"],
                            major=row["cc"] // 10, minor=row["cc"] % 10, sms=row["sms"],
                            l2_bytes=row["l2_mib"] * MIB,
                            global_bytes=int(row["vram_gib"] * 1024 * MIB * 0.98),
                            memory_bus_bits=row["bus_bits"],
                            memory_clock_khz=round(facts_compare.row_memory_khz(row)),
                            sm_clock_khz=row["boost_mhz"] * 1000)
        d["queried"].update(driver)
        d["engine_table_facts"].update(sku=row["name"], reference_clock_khz=row["boost_mhz"] * 1000,
                                       memory_bus_bits=row["bus_bits"],
                                       memory_clock_khz=round(facts_compare.row_memory_khz(row)),
                                       l2_mib=float(row["l2_mib"]))
        return d

    def test_a_recorded_memory_size_under_the_id_is_information(self):
        # RTX 5060 Ti (0x2D04): the row is the 16 GB board, and the 8 GB one shares the id.
        self.assertEqual(self.assert_no_mismatch(self.device_of_row(0x2D04))[("table", "vram_gib")],
                         "match")
        eight = self.device_of_row(0x2D04, global_bytes=int(8 * 1024 * MIB * 0.98))
        self.assertEqual(self.assert_no_mismatch(eight)[("table", "vram_gib")], "info")
        # A size the row does not record stays a failure, and so does 8 GB under another id.
        twelve = self.device_of_row(0x2D04, global_bytes=int(12 * 1024 * MIB * 0.98))
        self.assertEqual(statuses(twelve)[("table", "vram_gib")], "mismatch")
        d = device_4090()
        d["queried"]["global_bytes"] = int(8 * 1024 * MIB * 0.98)
        self.assertEqual(statuses(d)[("table", "vram_gib")], "mismatch")

    def test_a_recorded_faster_memory_under_the_id_is_information(self):
        # GTX 1650 (0x1F82): the row is the GDDR5 board (8 Gbps, 1665 MHz), and a GDDR6 one
        # (12 Gbps, 1590 MHz) shares the id. The engine keeps the row's rate: a faster clock is
        # never read from the driver.
        gddr6 = self.device_of_row(0x1F82, memory_clock_khz=6_000_000, sm_clock_khz=1_590_000)
        s = self.assert_no_mismatch(gddr6)
        self.assertEqual(s[("table", "memory_clock_khz")], "info")
        self.assertEqual(s[("table", "boost_clock_khz")], "info")
        self.assertEqual(s[("table", "engine_memory_clock_khz")], "match")
        self.assertEqual(s[("table", "engine_source")], "match")
        # An overclock of the row's own memory is not that board: still a failure.
        for khz in (5_000_000, 6_100_000, 7_000_000):
            d = self.device_of_row(0x1F82, memory_clock_khz=khz)
            self.assertEqual(statuses(d)[("table", "memory_clock_khz")], "mismatch", khz)

    def test_variants_are_boards_that_differ_from_the_row(self):
        recorded = {i: r["variants"] for i, r in ROWS.items() if "variants" in r}
        self.assertEqual(recorded, {0x2D04: [{"vram_gib": 8}],
                                    0x1F82: [{"gbps": 12.0, "boost_mhz": 1590}]})
        row = ROWS[0x2D04]
        for bad in ([], {"vram_gib": 8}, [{}], [{"vram_gib": 16}], [{"vram_gib": 0}],
                    [{"vram_gib": "8"}], [{"sms": 30}], [{"vram_gib": True}]):
            with self.assertRaises(ValueError, msg=bad):
                facts_compare.check_variants(dict(row, variants=bad))

    def test_laptop_is_priced_at_the_bottom_of_its_range(self):
        # RTX 5090 Laptop (0x2C18): boost 1597 to 2160 MHz by the OEM's power limit.
        d = device_4090()
        d["queried"].update(pci_device_id=0x2C18, name="NVIDIA GeForce RTX 5090 Laptop GPU",
                            major=12, minor=0, sms=82, l2_bytes=64 * MIB,
                            global_bytes=int(24 * 1024 * MIB * 0.98), memory_bus_bits=256,
                            memory_clock_khz=14_001_000, sm_clock_khz=1_800_000)
        d["engine_table_facts"].update(source=2, sku="GeForce RTX 5090 Laptop GPU",
                                       reference_clock_khz=1_597_000, memory_bus_bits=256,
                                       memory_clock_khz=14_000_000, l2_mib=64.0)
        s = self.assert_no_mismatch(d)
        self.assertEqual(s[("table", "boost_clock_khz")], "info")
        d["engine_table_facts"]["reference_clock_khz"] = 2_160_000  # the top of the range: wrong
        self.assertEqual(statuses(d)[("table", "engine_reference_clock_khz")], "mismatch")
        d = device_4090()
        d["engine_table_facts"]["source"] = 2  # a desktop board read as extrapolated: wrong
        self.assertEqual(statuses(d)[("table", "engine_source")], "mismatch")

    @staticmethod
    def device_a100():
        # A100 SXM4 80GB (0x20B2): a GA100 board, which MIG partitions.
        d = device_4090()
        d["queried"].update(pci_device_id=0x20B2, name="NVIDIA A100-SXM4-80GB", chip="ga100",
                            major=8, minor=0, sms=108, l2_bytes=40 * MIB, global_bytes=int(80 * 1024 * MIB * 0.99),
                            memory_bus_bits=5120, memory_clock_khz=1_593_000,
                            sm_clock_khz=1_410_000)
        d["engine_table_facts"].update(sku="A100 SXM4 80GB", reference_clock_khz=1_410_000,
                                       memory_bus_bits=5120, memory_clock_khz=1_593_000,
                                       l2_mib=40.0)
        return d

    def test_whole_a100_matches(self):
        self.assert_no_mismatch(self.device_a100())

    def test_slice_keeps_the_rows_values(self):
        # A MIG slice reports fewer SMs; the engine prices its share and keeps the row's values.
        d = self.device_a100()
        d["queried"]["sms"] = 42
        d["engine_table_facts"]["source"] = 2
        s = self.assert_no_mismatch(d)
        self.assertEqual(s[("table", "sms")], "info")
        self.assertEqual(s[("table", "engine_reference_clock_khz")], "match")
        self.assertEqual(s[("table", "engine_l2_mib")], "match")

    def test_mig_slice_reports_its_part(self):
        # A 3g.40gb slice: 42 SMs, half the L2, memory and bus. Information, not failures; the
        # whole board reporting the same smaller values fails, and so does a GeForce board with
        # fewer SMs than its row (no GeForce board partitions).
        d = self.device_a100()
        d["queried"].update(sms=42, l2_bytes=20 * MIB, global_bytes=40 * 1024 * MIB,
                            memory_bus_bits=2560)
        d["engine_table_facts"]["source"] = 2
        s = self.assert_no_mismatch(d)
        for field in ("sms", "l2_mib", "vram_gib", "bus_bits"):
            self.assertEqual(s[("table", field)], "info", field)
        d["queried"]["sms"] = 108
        d["engine_table_facts"]["source"] = 1
        s = statuses(d)
        self.assertEqual(s[("table", "l2_mib")], "mismatch")
        self.assertEqual(s[("table", "vram_gib")], "mismatch")
        d = device_4090()
        d["queried"].update(sms=64, l2_bytes=36 * MIB)
        s = statuses(d)
        self.assertEqual(s[("table", "sms")], "mismatch")
        self.assertEqual(s[("table", "l2_mib")], "mismatch")

    def test_mig_by_chip_and_segment(self):
        # GB202 partitions on its professional boards, not on the GeForce 5090; L40 (AD102) and
        # V100 (GV100) server boards do not partition at all.
        for pci, want in ((0x2BB1, True), (0x2BB5, True), (0x2B85, False), (0x26B5, False),
                          (0x1DB4, False), (0x20B7, True), (0x2330, True), (0x2901, True)):
            row = ROWS[pci]
            self.assertEqual(facts_compare.mig_capable(row), want, row["name"])

    def test_engine_table_reading_against_the_json(self):
        d = device_4090()
        d["engine_table_facts"]["l2_mib"] = 64.0
        self.assertEqual(statuses(d)[("table", "engine_l2_mib")], "mismatch")
        d = device_4090()
        d["engine_table_facts"]["sku"] = ""
        self.assertEqual(statuses(d)[("table", "engine_sku")], "mismatch")


class Cli(unittest.TestCase):
    def run_cli(self, probe):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.json"
            path.write_text(json.dumps(probe), encoding="utf-8")
            out = Path(tmp) / "summary.json"
            done = subprocess.run([sys.executable, "-B", str(ROOT / "tools" / "audit" / "facts_vs_driver.py"),
                                   "--probe-json", str(path), "--json-out", str(out)],
                                  capture_output=True, text=True)
            return done, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None)

    def test_exit_codes_and_summary(self):
        good, summary = self.run_cli({"devices": [device_4090()]})
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.assertTrue(summary["ok"])
        self.assertIn("device 0: NVIDIA GeForce RTX 4090", good.stdout)
        bad_device = copy.deepcopy(device_4090())
        bad_device["queried"]["l2_bytes"] = 64 * MIB
        bad, summary = self.run_cli({"devices": [device_4090(), bad_device]})
        self.assertEqual(bad.returncode, 1)
        self.assertFalse(summary["ok"])
        self.assertEqual([m["field"] for m in summary["devices"][1]["mismatches"]], ["l2_mib"])
        self.assertEqual(summary["devices"][0]["mismatches"], [])

    def test_a_missing_probe_is_exit_2(self):
        done = subprocess.run([sys.executable, "-B", str(ROOT / "tools" / "audit" / "facts_vs_driver.py"),
                               "--probe", str(ROOT / "no_such_probe")], capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)


if __name__ == "__main__":
    unittest.main()
