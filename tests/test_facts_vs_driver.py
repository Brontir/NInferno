"""tools/audit/facts_compare.py and facts_vs_driver.py on canned driver output, no GPU (#612 step 2).

The canned device is an RTX 4090 as ninferno-facts-probe prints it: the driver's values in
"queried", synthetic_device_class's architecture limits in "synthetic", the raw attributes the
query does not keep, the engine's readings of the table. Every value is written out here, none is
computed by the tool under test. Each test changes one thing and checks the field that must fail.
A real recorded probe output goes to tests/data/facts_probe_recorded.json (see RecordedProbe).
"""
import copy
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "audit"))
import facts_compare  # noqa: E402
import facts_vs_driver  # noqa: E402

MIB = 1024 * 1024
GIB = 1024 * MIB
TOOL = str(ROOT / "tools" / "audit" / "facts_vs_driver.py")
FEATURES = {"mma_bf16": True, "mma_bf16_native": True, "mma_tensor": True, "mma_fp8": True,
            "mma_f8f6f4": False, "mma_mxf4nvf4": False, "tma": False, "pdl": False,
            "clusters": False, "mma_slow": False}


def facts_of_4090():
    return {"source": 1, "sku": "GeForce RTX 4090", "reference_clock_khz": 2_520_000,
            "memory_bus_bits": 384, "memory_clock_khz": 10_500_000, "l2_mib": 72.0,
            "mma_slow": False}


def device_4090():
    queried = {"name": "NVIDIA GeForce RTX 4090", "pci_device_id": 0x2684, "chip": "ad102",
               "major": 8, "minor": 9, "sms": 128, "l2_bytes": 72 * MIB,
               "global_bytes": 25_757_220_864, "smem_per_block_optin": 101376,
               "smem_per_sm": 102400, "max_threads_per_sm": 1536, "max_blocks_per_sm": 24,
               "registers_per_sm": 65536, "persisting_l2_max_bytes": 54 * MIB,
               "memory_bus_bits": 384, "memory_clock_khz": 10_501_000, "sm_clock_khz": 2_520_000,
               "features": dict(FEATURES)}
    synthetic = {"name": "synthetic sm89", "pci_device_id": 0x2684, "chip": "ad102",
                 "major": 8, "minor": 9, "sms": 128, "l2_bytes": 72 * MIB,
                 "global_bytes": 25_757_220_864, "smem_per_block_optin": 101376,
                 "smem_per_sm": 102400, "max_threads_per_sm": 1536, "max_blocks_per_sm": 24,
                 "registers_per_sm": 65536, "persisting_l2_max_bytes": 54 * MIB,
                 "memory_bus_bits": 384, "memory_clock_khz": 10_501_000, "sm_clock_khz": 2_520_000,
                 "features": dict(FEATURES)}
    return {"index": 0, "pci_bus_id": "0000:01:00.0", "mig_mode": 0, "queried": queried,
            "synthetic": synthetic,
            "raw": {"reserved_shared_per_block": 1024, "shared_per_block_default": 49152,
                    "registers_per_block": 65536, "max_threads_per_block": 1024, "warp_size": 32},
            "engine_occupancy_limits": {"reserved_shared_per_block": 1024,
                                        "shared_per_block_default": 49152,
                                        "registers_per_block": 65536,
                                        "max_threads_per_block": 1024, "warp_size": 32},
            "engine_table_facts": facts_of_4090(), "engine_row_facts": facts_of_4090()}


ROWS = facts_vs_driver.load_rows(facts_vs_driver.FACTS)


def row_of(pci_id, **change):
    """A copy of a row of gpu_facts.json with the given keys changed (a wrong row)."""
    return dict(copy.deepcopy(ROWS[pci_id]), **change)


def statuses(device, row="table"):
    if row == "table":
        row = ROWS.get(device["queried"]["pci_device_id"])
    return {(r["source"], r["field"]): r["status"] for r in facts_compare.compare_device(device, row)}


def set_engine(device, **change):
    """Both engine readings (the stored one and the row alone) changed the same way."""
    device["engine_table_facts"].update(change)
    device["engine_row_facts"].update(change)


def device_a100():
    # A100 SXM4 80GB (0x20B2): a GA100 board, which MIG partitions.
    d = device_4090()
    d["queried"].update(pci_device_id=0x20B2, name="NVIDIA A100-SXM4-80GB", chip="ga100",
                        major=8, minor=0, sms=108, l2_bytes=40 * MIB,
                        global_bytes=85_057_388_544, memory_bus_bits=5120,
                        memory_clock_khz=1_593_000, sm_clock_khz=1_410_000)
    d["synthetic"].update(pci_device_id=0x20B2, chip="ga100", minor=0)
    set_engine(d, sku="A100 SXM4 80GB", reference_clock_khz=1_410_000, memory_bus_bits=5120,
               memory_clock_khz=1_593_000, l2_mib=40.0)
    return d


def device_1650():
    # GTX 1650 (0x1F82), TU117: the row is the GDDR5 board; mma_slow on both classes.
    d = device_4090()
    d["queried"].update(pci_device_id=0x1F82, name="NVIDIA GeForce GTX 1650", chip="tu117",
                        major=7, minor=5, sms=14, l2_bytes=1 * MIB, global_bytes=4_179_000_000,
                        memory_bus_bits=128, memory_clock_khz=4_001_000, sm_clock_khz=1_665_000)
    d["queried"]["features"]["mma_slow"] = True
    d["synthetic"]["features"]["mma_slow"] = True
    set_engine(d, sku="GeForce GTX 1650", reference_clock_khz=1_665_000, memory_bus_bits=128,
               memory_clock_khz=4_000_000, l2_mib=1.0, mma_slow=True)
    return d


def device_laptop():
    # RTX 5090 Laptop (0x2C18): boost 1597 to 2160 MHz by the OEM's power limit.
    d = device_4090()
    d["queried"].update(pci_device_id=0x2C18, name="NVIDIA GeForce RTX 5090 Laptop GPU",
                        chip="gb203", major=12, minor=0, sms=82, l2_bytes=64 * MIB,
                        global_bytes=25_265_000_000, memory_bus_bits=256,
                        memory_clock_khz=14_001_000, sm_clock_khz=1_800_000)
    set_engine(d, sku="GeForce RTX 5090 Laptop GPU", source=2, memory_bus_bits=256,
               memory_clock_khz=14_000_000, l2_mib=64.0, reference_clock_khz=1_597_000)
    return d


def no_mismatch(test, statuses_of):
    test.assertNotIn("mismatch", statuses_of.values(),
                     [k for k, v in statuses_of.items() if v == "mismatch"])
    return statuses_of


def mismatches(statuses_of):
    return sorted(k for k, v in statuses_of.items() if v == "mismatch")


class Compare(unittest.TestCase):
    def test_a_matching_board_has_no_mismatch(self):
        s = no_mismatch(self, statuses(device_4090()))
        self.assertEqual(s[("table", "sms")], "match")
        self.assertEqual(s[("table", "chip")], "match")
        self.assertEqual(s[("table", "memory_clock_khz")], "match")  # 10 501 000 within 0.5 %
        self.assertEqual(s[("arch", "smem_optin_plus_reserved_is_per_sm")], "match")
        self.assertEqual(s[("table", "engine_row_mma_slow")], "match")

    def test_each_table_field_fails_alone(self):
        for field, change in (
                ("sms", lambda q: q.update(sms=130)),
                ("l2_mib", lambda q: q.update(l2_bytes=64 * MIB)),
                ("bus_bits", lambda q: q.update(memory_bus_bits=300)),
                ("memory_clock_khz", lambda q: q.update(memory_clock_khz=10_600_000)),  # +0.95 %
                ("boost_clock_khz", lambda q: q.update(sm_clock_khz=2_200_000)),
                ("vram_gib", lambda q: q.update(global_bytes=12 * 1024 * MIB)),
                ("chip", lambda q: q.update(chip="gk104")),
                ("cc", lambda q: q.update(minor=6))):
            d = device_4090()
            change(d["queried"])
            self.assertEqual(statuses(d)[("table", field)], "mismatch", field)

    def test_wrong_rows_are_mismatches_not_information(self):
        # The row is wrong, the driver and the engine agree with each other: a whole board.
        d = device_4090()
        wrong_bus = row_of(0x2684, bus_bits=256)  # the driver's 384 is not a recorded variant
        self.assertEqual(statuses(d, wrong_bus)[("table", "bus_bits")], "mismatch")
        wrong_rate = row_of(0x2684, gbps=24.0)  # the driver's 10 501 000 kHz is 21 Gbps
        self.assertEqual(statuses(d, wrong_rate)[("table", "memory_clock_khz")], "mismatch")
        wrong_boost = row_of(0x2684, boost_mhz=2700)  # the driver is 6.7 % under it
        self.assertEqual(statuses(d, wrong_boost)[("table", "boost_clock_khz")], "mismatch")
        wrong_vram = row_of(0x2684, vram_gib=26)  # 24 GiB of a GeForce board is not ECC
        self.assertEqual(statuses(d, wrong_vram)[("table", "vram_gib")], "mismatch")
        wrong_sms = row_of(0x20B2, sms=124)  # an A100 reporting its whole memory is not a slice
        self.assertEqual(statuses(device_a100(), wrong_sms)[("table", "sms")], "mismatch")

    def test_a_recorded_bus_and_rate_are_information(self):
        d = device_4090()
        d["queried"].update(memory_bus_bits=256, memory_clock_khz=9_000_000)
        row = row_of(0x2684, variants=[{"bus_bits": 256, "gbps": 18.0}])
        s = statuses(d, row)
        self.assertEqual(s[("table", "bus_bits")], "info")
        self.assertEqual(s[("table", "memory_clock_khz")], "info")
        other = device_4090()
        other["queried"]["memory_bus_bits"] = 192  # not the recorded board's bus
        self.assertEqual(statuses(other, row)[("table", "bus_bits")], "mismatch")
        self.assertEqual(statuses(d)[("table", "bus_bits")], "mismatch")  # unrecorded: a failure
        self.assertEqual(statuses(d)[("table", "memory_clock_khz")], "mismatch")

    def test_memory_clock_tolerance_edge(self):
        for khz, want in ((10_500_000 * 1.0049, "match"), (10_500_000 * 0.9951, "match"),
                          (10_500_000 * 1.0051, "mismatch"), (10_500_000 * 0.9949, "mismatch")):
            d = device_4090()
            d["queried"]["memory_clock_khz"] = int(khz)
            self.assertEqual(statuses(d)[("table", "memory_clock_khz")], want, khz)

    def test_l2_tolerance_edge(self):
        for ratio, want in ((1.009, "match"), (0.991, "match"), (1.011, "mismatch"),
                            (0.989, "mismatch")):
            d = device_4090()
            d["queried"]["l2_bytes"] = int(72 * MIB * ratio)
            self.assertEqual(statuses(d)[("table", "l2_mib")], want, ratio)

    def test_boost_windows(self):
        # Within 1.5 % a match; a partner board up to 7 % over and an OEM one up to 3 % under are
        # information; anything farther is a wrong row.
        for ratio, want in ((1.014, "match"), (0.986, "match"), (1.016, "info"), (0.984, "info"),
                            (1.069, "info"), (1.071, "mismatch"), (0.971, "info"),
                            (0.969, "mismatch")):
            d = device_4090()
            d["queried"]["sm_clock_khz"] = int(2_520_000 * ratio)
            self.assertEqual(statuses(d)[("table", "boost_clock_khz")], want, ratio)

    def test_vram_windows(self):
        # GDDR ECC keeps 1/16 on a professional board; a GeForce board reports nearly all.
        pro = row_of(0x2684, segment="pro")
        for gib_share, row, want in ((0.928, pro, "match"), (0.91, pro, "mismatch"),
                                     (0.928, None, "mismatch"), (0.98, None, "match"),
                                     (0.96, None, "mismatch")):
            d = device_4090()
            d["queried"]["global_bytes"] = int(24 * GIB * gib_share)
            self.assertEqual(statuses(d, row or "table")[("table", "vram_gib")], want, gib_share)

    def test_a_driver_that_does_not_say_is_unavailable_not_a_failure(self):
        d = device_4090()
        d["queried"].update(memory_bus_bits=0, memory_clock_khz=-1, sm_clock_khz=0)
        s = statuses(d)
        for field in ("bus_bits", "memory_clock_khz", "boost_clock_khz"):
            self.assertEqual(s[("table", field)], "unavailable", field)
        no_mismatch(self, s)

    def test_unknown_pci_id_has_no_row(self):
        d = device_4090()
        d["queried"]["pci_device_id"] = 0xDEAD
        self.assertEqual(statuses(d)[("table", "row")], "mismatch")

    def test_each_architecture_limit_fails_alone(self):
        for field in ("smem_per_block_optin", "smem_per_sm", "max_threads_per_sm",
                      "max_blocks_per_sm", "registers_per_sm"):
            d = device_4090()
            d["queried"][field] += 1024  # the driver says more than the synthetic class
            self.assertIn(("arch", field), mismatches(statuses(d)), field)
        d = device_4090()
        d["queried"].update(max_threads_per_sm=2048, max_blocks_per_sm=32)  # the Volta case
        s = statuses(d)
        self.assertEqual(s[("arch", "max_threads_per_sm")], "mismatch")
        self.assertEqual(s[("arch", "registers_per_sm")], "match")

    def test_the_reserved_shared_invariants(self):
        d = device_4090()
        d["raw"]["reserved_shared_per_block"] = 0  # opt-in + reserved is no longer the per-SM figure
        self.assertIn(("arch", "smem_optin_plus_reserved_is_per_sm"), mismatches(statuses(d)))
        d = device_4090()
        d["synthetic"]["smem_per_block_optin"] -= 1024  # the synthetic class's own pair breaks
        self.assertIn(("arch", "synthetic_smem_optin_plus_reserved_is_per_sm"),
                      mismatches(statuses(d)))
        # The driver reserves 2 KiB and both classes are 1 KiB short: only the engine's assumed
        # reserved memory (1 KiB) disagrees with the driver.
        d = device_4090()
        d["raw"]["reserved_shared_per_block"] = 2048
        d["queried"]["smem_per_block_optin"] = 100352
        d["synthetic"]["smem_per_block_optin"] = 100352
        self.assertEqual(mismatches(statuses(d)), [("arch", "occupancy_reserved_shared_per_block")])
        d = device_4090()
        d["raw"]["reserved_shared_per_block"] = -1
        self.assertEqual(statuses(d)[("arch", "reserved_shared_per_block")], "unavailable")

    def test_occupancy_limits_against_the_drivers_raw_values(self):
        for field, driver in (("shared_per_block_default", 32768), ("registers_per_block", 32768),
                              ("max_threads_per_block", 512), ("warp_size", 16)):
            d = device_4090()
            d["raw"][field] = driver
            self.assertEqual(mismatches(statuses(d)), [("arch", f"occupancy_{field}")], field)
        d = device_4090()
        d["raw"]["warp_size"] = -1  # the driver could not answer
        self.assertEqual(statuses(d)[("arch", "occupancy_warp_size")], "unavailable")
        # Reserved shared memory 0 is a real answer (Volta, Turing), the engine says 0 there.
        d = device_1650()
        d["raw"]["reserved_shared_per_block"] = 0
        d["engine_occupancy_limits"]["reserved_shared_per_block"] = 0
        d["queried"].update(smem_per_block_optin=65536, smem_per_sm=65536)
        d["synthetic"].update(smem_per_block_optin=65536, smem_per_sm=65536)
        self.assertEqual(statuses(d)[("arch", "occupancy_reserved_shared_per_block")], "match")

    def test_persisting_l2_cap(self):
        for cap, want in ((54 * MIB, "match"), (40 * MIB, "info"), (9 * MIB, "info"),
                          (9 * MIB - 1, "mismatch"), (1, "mismatch"), (73 * MIB, "mismatch"),
                          (0, "unavailable")):
            d = device_4090()
            d["queried"]["persisting_l2_max_bytes"] = cap
            self.assertEqual(statuses(d)[("arch", "persisting_l2_max_bytes")], want, cap)

    def test_features(self):
        d = device_4090()
        d["queried"]["features"]["pdl"] = True  # what the build lacks: information
        self.assertEqual(statuses(d)[("arch", "feature_pdl")], "info")
        d = device_4090()
        self.assertEqual(statuses(d)[("arch", "feature_mma_slow")], "derived")
        d["queried"]["features"]["mma_slow"] = True
        self.assertEqual(statuses(d)[("arch", "feature_mma_slow")], "mismatch")

    def test_mma_slow_is_checked_against_the_row(self):
        # Queried and synthetic read the same row, so only the engine's value against the JSON
        # can catch a row that says mma_slow and an engine that does not.
        d = device_4090()
        s = statuses(d, row_of(0x2684, mma_slow=True))
        self.assertEqual(s[("table", "engine_mma_slow")], "mismatch")
        self.assertEqual(s[("table", "engine_row_mma_slow")], "mismatch")
        self.assertEqual(s[("arch", "feature_mma_slow")], "derived")
        self.assertEqual(statuses(device_1650())[("table", "engine_mma_slow")], "match")
        d = device_1650()
        d["engine_table_facts"]["mma_slow"] = False  # the query did not store the flag
        self.assertEqual(statuses(d)[("table", "engine_mma_slow")], "mismatch")

    def test_the_engine_reading_of_the_row_is_exact(self):
        for field, bad in (("source", 2), ("sku", "GeForce RTX 4080"),
                           ("reference_clock_khz", 2_500_000), ("memory_bus_bits", 320),
                           ("memory_clock_khz", 10_400_000), ("l2_mib", 64.0)):
            d = device_4090()
            d["engine_row_facts"][field] = bad
            self.assertIn(("table", f"engine_row_{field}"), mismatches(statuses(d)), field)

    def test_the_stored_facts_are_the_rows_where_the_driver_agrees(self):
        for field, bad in (("source", 2), ("reference_clock_khz", 2_500_000),
                           ("memory_bus_bits", 320), ("memory_clock_khz", 10_400_000),
                           ("l2_mib", 64.0), ("sku", ""), ("sku", "GeForce RTX 4080")):
            d = device_4090()
            d["engine_table_facts"][field] = bad
            name = {"sku": "engine_sku"}.get(field, f"engine_{field}")
            self.assertIn(("table", name), mismatches(statuses(d)), (field, bad))

    def test_a_row_without_l2_is_a_mismatch(self):
        row = dict(ROWS[0x2684])
        del row["l2_mib"]
        self.assertEqual(statuses(device_4090(), row)[("table", "l2_mib")], "mismatch")

    def test_a_recorded_memory_variant_is_read_from_the_driver(self):
        # A 128-bit board on a slower memory under the row's id (the row records both): the
        # driver's bus and clock are information, the engine reads them (source 2).
        d = device_4090()
        d["queried"].update(memory_bus_bits=256, memory_clock_khz=9_000_000)
        d["engine_table_facts"].update(source=2, memory_bus_bits=256, memory_clock_khz=9_000_000)
        row = row_of(0x2684, variants=[{"bus_bits": 256, "gbps": 18.0}])
        s = no_mismatch(self, statuses(d, row))
        self.assertEqual(s[("table", "engine_memory_bus_bits")], "match")
        d["engine_table_facts"]["memory_clock_khz"] = 11_000_000  # faster than the row's: wrong
        self.assertEqual(statuses(d, row)[("table", "engine_memory_clock_khz")], "mismatch")
        d["engine_table_facts"].update(memory_clock_khz=9_000_000, memory_bus_bits=192)
        self.assertEqual(statuses(d, row)[("table", "engine_memory_bus_bits")], "mismatch")

    def test_a_recorded_memory_size_under_the_id_is_information(self):
        # RTX 5060 Ti (0x2D04): the row is the 16 GB board, and the 8 GB one shares the id.
        def board(global_bytes):
            d = device_4090()
            d["queried"].update(pci_device_id=0x2D04, chip="gb206", major=12, minor=0, sms=36,
                                l2_bytes=32 * MIB, global_bytes=global_bytes, memory_bus_bits=128,
                                memory_clock_khz=14_000_000, sm_clock_khz=2_572_000)
            set_engine(d, sku="GeForce RTX 5060 Ti", reference_clock_khz=2_572_000,
                       memory_bus_bits=128, memory_clock_khz=14_000_000, l2_mib=32.0)
            return d
        self.assertEqual(no_mismatch(self, statuses(board(17_000_000_000)))[("table", "vram_gib")],
                         "match")
        self.assertEqual(no_mismatch(self, statuses(board(8_400_000_000)))[("table", "vram_gib")],
                         "info")
        self.assertEqual(statuses(board(12_600_000_000))[("table", "vram_gib")], "mismatch")
        d = device_4090()
        d["queried"]["global_bytes"] = int(8 * GIB * 0.98)  # 8 GB under another id
        self.assertEqual(statuses(d)[("table", "vram_gib")], "mismatch")

    def test_a_recorded_faster_memory_under_the_id_is_information(self):
        # GTX 1650 (0x1F82): the row is the GDDR5 board (8 Gbps, 1665 MHz), and a GDDR6 one
        # (12 Gbps, 1590 MHz) shares the id. The engine keeps the row's rate.
        self.assertEqual(statuses(device_1650())[("table", "engine_source")], "match")
        no_mismatch(self, statuses(device_1650()))
        gddr6 = device_1650()
        gddr6["queried"].update(memory_clock_khz=6_000_000, sm_clock_khz=1_590_000)
        s = no_mismatch(self, statuses(gddr6))
        self.assertEqual(s[("table", "memory_clock_khz")], "info")
        self.assertEqual(s[("table", "boost_clock_khz")], "info")
        for khz in (5_000_000, 6_100_000, 7_000_000):  # not the recorded board: a failure
            d = device_1650()
            d["queried"]["memory_clock_khz"] = khz
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
        facts_compare.check_variants(dict(row, variants=[{"bus_bits": 64}]))

    def test_rows_are_validated(self):
        good = dict(ROWS[0x2684])
        facts_compare.check_row(good)
        for key, bad in (("gbps", None), ("gbps", "21"), ("bus_bits", "384"), ("sms", 0),
                         ("boost_mhz", True), ("name", ""), ("chip", None), ("l2_mib", "72"),
                         ("id", 0x2684)):
            with self.assertRaises(ValueError, msg=(key, bad)):
                facts_compare.check_row(dict(good, **{key: bad}))
        missing = dict(good)
        del missing["gbps"]
        with self.assertRaises(ValueError):
            facts_compare.check_row(missing)
        # The GTX 16 chips have no tensor cores: their rows must say so.
        for pci in (0x1F82, 0x2182):
            row = dict(ROWS[pci])
            self.assertTrue(row["mma_slow"])
            del row["mma_slow"]
            with self.assertRaises(ValueError, msg=pci):
                facts_compare.check_row(row)

    def test_laptop_is_priced_at_the_bottom_of_its_range(self):
        s = no_mismatch(self, statuses(device_laptop()))
        self.assertEqual(s[("table", "boost_clock_khz")], "info")
        d = device_laptop()
        d["engine_table_facts"]["reference_clock_khz"] = 2_160_000  # the top of the range: wrong
        self.assertEqual(statuses(d)[("table", "engine_reference_clock_khz")], "mismatch")
        d = device_4090()
        d["engine_table_facts"]["source"] = 2  # a desktop board read as extrapolated: wrong
        self.assertEqual(statuses(d)[("table", "engine_source")], "mismatch")
        # The driver's boost inside the published range is information; outside it a wrong row.
        for khz, want in ((1_800_000, "info"), (1_590_000, "info"), (1_500_000, "mismatch"),
                          (2_400_000, "mismatch")):
            d = device_laptop()
            d["queried"]["sm_clock_khz"] = khz
            self.assertEqual(statuses(d)[("table", "boost_clock_khz")], want, khz)

    def test_whole_a100_matches(self):
        no_mismatch(self, statuses(device_a100()))

    def test_slice_keeps_the_rows_values(self):
        # A MIG slice reports fewer SMs; the engine prices its share and keeps the row's values.
        d = device_a100()
        d["mig_mode"] = 1
        d["queried"]["sms"] = 42
        d["queried"]["global_bytes"] = 40_000_000_000
        d["engine_table_facts"]["source"] = 2
        s = no_mismatch(self, statuses(d))
        self.assertEqual(s[("table", "sms")], "info")
        self.assertEqual(s[("table", "engine_row_reference_clock_khz")], "match")
        self.assertEqual(s[("table", "engine_l2_mib")], "match")

    def test_mig_slice_reports_its_part(self):
        # A 3g.40gb slice: 42 SMs, half the L2, memory and bus. Information, not failures; the
        # whole board reporting the same smaller values fails, as does a GeForce board with
        # fewer SMs than its row (no GeForce board partitions) and a larger count anywhere.
        d = device_a100()
        d["mig_mode"] = 1
        d["queried"].update(sms=42, l2_bytes=20 * MIB, global_bytes=40_000_000_000,
                            memory_bus_bits=2560)
        d["engine_table_facts"]["source"] = 2
        s = no_mismatch(self, statuses(d))
        for field in ("sms", "l2_mib", "vram_gib", "bus_bits"):
            self.assertEqual(s[("table", field)], "info", field)
        d["queried"].update(sms=108, global_bytes=85_057_388_544)  # whole memory again
        d["engine_table_facts"]["source"] = 1
        s = statuses(d)
        self.assertEqual(s[("table", "l2_mib")], "mismatch")
        d = device_4090()
        d["queried"].update(sms=64, l2_bytes=36 * MIB)
        s = statuses(d)
        self.assertEqual(s[("table", "sms")], "mismatch")
        self.assertEqual(s[("table", "l2_mib")], "mismatch")
        d = device_a100()
        d["queried"]["sms"] = 120  # more SMs than the row, a wrong row
        self.assertEqual(statuses(d)[("table", "sms")], "mismatch")

    def test_a_laptop_or_a_slice_keeps_the_rows_memory_values_in_the_engine(self):
        # The slice's narrower bus or lower clock copied into the stored facts is wrong: the
        # engine prices the row's memory at the slice's share. Only a whole board's recorded
        # variant is read from the driver.
        d = device_a100()
        d["mig_mode"] = 1
        d["queried"].update(sms=42, l2_bytes=20 * MIB, global_bytes=40_000_000_000,
                            memory_bus_bits=2560)
        d["engine_table_facts"]["source"] = 2
        no_mismatch(self, statuses(d))
        d["engine_table_facts"]["memory_bus_bits"] = 2560
        self.assertEqual(statuses(d)[("table", "engine_memory_bus_bits")], "mismatch")
        d["engine_table_facts"].update(memory_bus_bits=5120, memory_clock_khz=1_000_000)
        self.assertEqual(statuses(d)[("table", "engine_memory_clock_khz")], "mismatch")
        d = device_laptop()
        d["engine_table_facts"]["memory_bus_bits"] = 128
        self.assertEqual(statuses(d)[("table", "engine_memory_bus_bits")], "mismatch")
        d = device_laptop()
        d["engine_table_facts"]["memory_clock_khz"] = 9_000_000
        self.assertEqual(statuses(d)[("table", "engine_memory_clock_khz")], "mismatch")

    def test_mig_mode_decides_a_full_memory_slice(self):
        # A 7g profile of an A100: 98 SMs and the board's whole memory and L2. With MIG enabled
        # that is the slice (information); with MIG disabled it is a wrong row or a wrong count.
        def seven_g(mig_mode):
            d = device_a100()
            d["mig_mode"] = mig_mode
            d["queried"]["sms"] = 98
            d["engine_table_facts"]["source"] = 2
            return d
        self.assertEqual(no_mismatch(self, statuses(seven_g(1)))[("table", "sms")], "info")
        self.assertEqual(statuses(seven_g(0))[("table", "sms")], "mismatch")
        # NVML did not say: the whole memory cannot be told from a wrong row, so unverified, not
        # information; a reduced memory shows the slice itself.
        self.assertEqual(statuses(seven_g(-1))[("table", "sms")], "unavailable")
        d = seven_g(-1)
        d["queried"]["global_bytes"] = 40_000_000_000
        self.assertEqual(statuses(d)[("table", "sms")], "info")

    def test_mig_by_chip_and_segment(self):
        # GB202 partitions on its professional boards, not on the GeForce 5090; L40 (AD102) and
        # V100 (GV100) server boards do not partition at all.
        for pci, want in ((0x2BB1, True), (0x2BB5, True), (0x2B85, False), (0x26B5, False),
                          (0x1DB4, False), (0x20B7, True), (0x2330, True), (0x2901, True)):
            row = ROWS[pci]
            self.assertEqual(facts_compare.mig_capable(row), want, row["name"])


class Cli(unittest.TestCase):
    def run_cli(self, probe, *extra, facts=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.json"
            path.write_text(json.dumps(probe), encoding="utf-8")
            out = Path(tmp) / "summary.json"
            command = [sys.executable, "-B", TOOL, "--probe-json", str(path), "--json-out", str(out)]
            if facts is not None:
                facts_path = Path(tmp) / "facts.json"
                facts_path.write_text(json.dumps(facts), encoding="utf-8")
                command += ["--facts", str(facts_path)]
            done = subprocess.run(command + list(extra), capture_output=True, text=True)
            return done, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None)

    def test_exit_codes_and_summary(self):
        good, summary = self.run_cli({"devices": [device_4090()]})
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.assertTrue(summary["ok"])
        self.assertIn("device 0: NVIDIA GeForce RTX 4090", good.stdout)
        bad_device = copy.deepcopy(device_4090())
        bad_device["index"] = 1
        bad_device["queried"]["l2_bytes"] = 64 * MIB
        bad, summary = self.run_cli({"devices": [device_4090(), bad_device]})
        self.assertEqual(bad.returncode, 1)
        self.assertFalse(summary["ok"])
        self.assertEqual([m["field"] for m in summary["devices"][1]["mismatches"]], ["l2_mib"])
        self.assertEqual(summary["devices"][0]["mismatches"], [])

    def test_a_driver_that_answers_nothing_is_unverified_not_clean(self):
        d = device_4090()
        d["queried"].update(memory_bus_bits=0, memory_clock_khz=0, sm_clock_khz=0)
        d["raw"].update(shared_per_block_default=-1, registers_per_block=-1,
                        max_threads_per_block=-1, warp_size=-1, reserved_shared_per_block=-1)
        done, summary = self.run_cli({"devices": [d]})
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertTrue(summary["ok"])
        fields = {u["field"] for u in summary["unverified"]}
        self.assertTrue({"bus_bits", "memory_clock_khz", "boost_clock_khz",
                         "occupancy_warp_size", "reserved_shared_per_block"} <= fields, fields)
        self.assertIn("unverified", done.stderr)
        strict, _ = self.run_cli({"devices": [d]}, "--strict")
        self.assertEqual(strict.returncode, 1)
        clean, summary = self.run_cli({"devices": [device_4090()]}, "--strict")
        self.assertEqual(clean.returncode, 0)
        self.assertEqual(summary["unverified"], [])

    def test_an_empty_or_partial_recording_is_exit_2(self):
        # No devices, or a device without a part the audit reads, is invalid input (2), never a
        # clean audit (0).
        def without(part, key=None):
            d = device_4090()
            if key is None:
                del d[part]
            else:
                del d[part][key]
            return {"devices": [d]}
        for name, probe in (("empty list", {"devices": []}), ("no devices key", {}),
                            ("devices not a list", {"devices": {"0": device_4090()}}),
                            ("not an object", []), ("device not an object", {"devices": [1]}),
                            ("engine facts not an object",
                             {"devices": [dict(device_4090(), engine_table_facts=None)]}),
                            ("one good, one partial", {"devices": [
                                device_4090(), dict(without("raw")["devices"][0], index=1)]})):
            done, summary = self.run_cli(probe)
            self.assertEqual(done.returncode, 2, name + ": " + done.stdout)
            self.assertIn("invalid probe output", done.stderr, name)
            self.assertIsNone(summary, name)
        for part in facts_compare.DEVICE_PARTS + ("index",):
            done, _ = self.run_cli(without(part))
            self.assertEqual(done.returncode, 2, part)

    def test_each_key_the_comparison_reads_must_be_present_and_typed(self):
        # A missing key, null, a string or a bool where a number is read: exit 2, never 0 (a
        # clean audit) or 1 (a mismatch).
        delete = object()
        cases = [("raw", "reserved_shared_per_block", None),
                 ("raw", "reserved_shared_per_block", "1024"), ("raw", "max_threads_per_block", False),
                 ("queried", "sms", "128"), ("queried", "sms", None), ("queried", "l2_bytes", None),
                 ("queried", "memory_clock_khz", None), ("queried", "sm_clock_khz", "x"),
                 ("queried", "global_bytes", None), ("queried", "persisting_l2_max_bytes", None),
                 ("queried", "name", None), ("queried", "chip", None), ("queried", "sms", True),
                 ("queried", "major", float("nan")), ("queried", "pci_device_id", 9860.0),
                 ("queried", "major", 8.0), ("engine_table_facts", "memory_bus_bits", None),
                 ("engine_table_facts", "sku", None), ("engine_table_facts", "mma_slow", 0),
                 ("engine_row_facts", "mma_slow", None), ("engine_row_facts", "l2_mib", "72")]
        cases += [("engine_occupancy_limits", key, None) for key in facts_compare.OCCUPANCY_LIMITS]
        for part, keys in facts_compare.NUMBER_KEYS.items():
            cases += [(part, key, delete) for key in keys]
        cases += [(part, key, delete) for part, key in facts_compare.STRING_KEYS +
                  facts_compare.BOOL_KEYS + facts_compare.INTEGER_KEYS]
        for part, key, bad in cases:
            d = device_4090()
            if bad is delete:
                del d[part][key]
            else:
                d[part][key] = bad
            done, summary = self.run_cli({"devices": [d]})
            self.assertEqual(done.returncode, 2, f"{part}.{key} = {bad!r}: {done.stdout}")
            self.assertIn("invalid probe output", done.stderr, f"{part}.{key}")
            self.assertNotIn("Traceback", done.stderr, f"{part}.{key}")
            self.assertIsNone(summary, f"{part}.{key}")

    def test_features_and_index_must_be_present(self):
        def mutate(change):
            d = device_4090()
            change(d)
            return d
        for name, d in (
                ("no queried features", mutate(lambda d: d["queried"].pop("features"))),
                ("no synthetic features", mutate(lambda d: d["synthetic"].pop("features"))),
                ("synthetic features empty, queried mma_slow set", mutate(
                    lambda d: (d["synthetic"].update(features={}),
                               d["queried"]["features"].update(mma_slow=True)))),
                ("synthetic features without mma_slow",
                 mutate(lambda d: d["synthetic"]["features"].pop("mma_slow"))),
                ("queried features without mma_slow",
                 mutate(lambda d: d["queried"]["features"].pop("mma_slow"))),
                ("queried features short of a synthetic one",
                 mutate(lambda d: d["queried"]["features"].pop("pdl"))),
                ("both features empty", mutate(lambda d: (d["synthetic"].update(features={}),
                                                          d["queried"].update(features={})))),
                ("a feature that is not a bool",
                 mutate(lambda d: d["queried"]["features"].update(pdl=1))),
                ("index null", mutate(lambda d: d.update(index=None))),
                ("no mig mode", mutate(lambda d: d.pop("mig_mode"))),
                ("mig mode 2", mutate(lambda d: d.update(mig_mode=2))),
                ("mig mode a bool", mutate(lambda d: d.update(mig_mode=True))),
                ("mig mode null", mutate(lambda d: d.update(mig_mode=None))),
                ("pci bus id not text", mutate(lambda d: d.update(pci_bus_id=7))),
                ("no index", mutate(lambda d: d.pop("index")))):
            done, summary = self.run_cli({"devices": [d]})
            self.assertEqual(done.returncode, 2, name + ": " + done.stdout)
            self.assertIn("invalid probe output", done.stderr, name)

    def test_a_duplicate_index_is_exit_2(self):
        done, _ = self.run_cli({"devices": [device_4090(), device_4090()]})
        self.assertEqual(done.returncode, 2)
        self.assertIn("appears twice", done.stderr)

    def test_a_device_the_probe_could_not_read_fails_the_audit_but_keeps_the_others(self):
        done, summary = self.run_cli({"devices": [device_4090(), {"index": 1, "error": "boom"}]})
        self.assertEqual(done.returncode, 1)
        self.assertIn("device 0: NVIDIA GeForce RTX 4090", done.stdout)
        self.assertEqual([m["field"] for m in summary["devices"][1]["mismatches"]], ["device"])

    def test_the_drivers_no_answer_is_still_unavailable(self):
        # Present with -1 is the driver saying nothing: exit 0, not 2. A bool is not that number.
        d = device_4090()
        d["raw"].update(reserved_shared_per_block=-1, warp_size=-1, max_threads_per_block=-1)
        done, summary = self.run_cli({"devices": [d]})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertFalse(facts_compare.unavailable(False))
        self.assertTrue(facts_compare.unavailable(-1))

    def test_board_and_build_features_in_the_cli(self):
        slow = device_4090()
        slow["queried"]["features"]["mma_slow"] = True
        done, summary = self.run_cli({"devices": [slow]})
        self.assertEqual(done.returncode, 1)
        self.assertEqual([m["field"] for m in summary["devices"][0]["mismatches"]],
                         ["feature_mma_slow"])
        build = device_4090()
        build["queried"]["features"]["pdl"] = True  # what the build lacks: information only
        done, summary = self.run_cli({"devices": [build]})
        self.assertEqual(done.returncode, 0, done.stdout)
        warp = device_4090()
        warp["raw"]["warp_size"] = 16
        done, summary = self.run_cli({"devices": [warp]})
        self.assertEqual(done.returncode, 1)
        self.assertEqual([m["field"] for m in summary["devices"][0]["mismatches"]],
                         ["occupancy_warp_size"])

    def test_wrong_fact_rows_are_exit_2_or_a_mismatch(self):
        def facts_with(change):
            rows = [copy.deepcopy(r) for r in json.loads(
                facts_vs_driver.FACTS.read_text(encoding="utf-8"))["gpus"]]
            for r in rows:
                if r["id"] == "0x2684":
                    change(r)
            return {"schema": 1, "gpus": rows}
        for name, change in (("gbps null", lambda r: r.update(gbps=None)),
                             ("gbps missing", lambda r: r.pop("gbps")),
                             ("bus_bits string", lambda r: r.update(bus_bits="384")),
                             ("bad variants", lambda r: r.update(variants=[]))):
            done, _ = self.run_cli({"devices": [device_4090()]}, facts=facts_with(change))
            self.assertEqual(done.returncode, 2, name)
            self.assertNotIn("Traceback", done.stderr, name)
        rows = json.loads(facts_vs_driver.FACTS.read_text(encoding="utf-8"))["gpus"]
        done, _ = self.run_cli({"devices": [device_4090()]},
                               facts={"schema": 1, "gpus": rows + [dict(rows[0])]})
        self.assertEqual(done.returncode, 2)
        self.assertIn("appears twice", done.stderr)
        done, summary = self.run_cli({"devices": [device_4090()]},
                                     facts=facts_with(lambda r: r.update(bus_bits=256)))
        self.assertEqual(done.returncode, 1)  # a wrong row is a mismatch, not information
        self.assertIn("bus_bits", [m["field"] for m in summary["devices"][0]["mismatches"]])

    def test_a_json_out_that_cannot_be_written_is_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.json"
            probe.write_text(json.dumps({"devices": [device_4090()]}), encoding="utf-8")
            done = subprocess.run([sys.executable, "-B", TOOL, "--probe-json", str(probe),
                                   "--json-out", tmp], capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertNotIn("Traceback", done.stderr)

    def test_the_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.json"
            probe.write_text(json.dumps({"devices": [device_4090()]}), encoding="utf-8")
            for args in (["--probe-json", str(probe), "--probe", "x"], [],
                         ["--probe-json", str(probe), "--device", "0"]):
                done = subprocess.run([sys.executable, "-B", TOOL] + args, capture_output=True,
                                      text=True)
                self.assertEqual(done.returncode, 2, args)

    def test_a_missing_probe_is_exit_2(self):
        done = subprocess.run([sys.executable, "-B", TOOL, "--probe", str(ROOT / "no_such_probe")],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)


class RunningTheProbe(unittest.TestCase):
    """main() with subprocess.run replaced: what is run, and what a failing probe gives."""

    def main_with(self, completed=None, raises=None, args=("--probe", "probe")):
        run = mock.Mock(return_value=completed, side_effect=raises)
        with mock.patch.object(facts_vs_driver.subprocess, "run", run), \
                mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            code = facts_vs_driver.main(list(args))
        return code, run

    def test_the_probe_is_run_for_one_device_with_a_timeout(self):
        done = subprocess.CompletedProcess([], 0, json.dumps({"devices": [device_4090()]}), "")
        code, run = self.main_with(done, args=("--probe", "probe", "--device", "1"))
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args[0], ["probe", "--device", "1"])
        self.assertIn("timeout", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")

    def test_a_failing_probe_is_exit_2(self):
        done = subprocess.CompletedProcess([], 3, json.dumps({"devices": [device_4090()]}), "no GPU")
        self.assertEqual(self.main_with(done)[0], 2)

    def test_a_probe_that_hangs_is_exit_2(self):
        self.assertEqual(self.main_with(raises=subprocess.TimeoutExpired("probe", 1))[0], 2)

    def test_a_probe_that_prints_no_json_is_exit_2(self):
        done = subprocess.CompletedProcess([], 0, "not json", "")
        self.assertEqual(self.main_with(done)[0], 2)


class ProbeSource(unittest.TestCase):
    """No GPU: the keys facts_compare.py reads against the keys apps/facts/main.cpp prints, part
    by part, so a rename or a dropped key on either side fails here."""

    SOURCE = (ROOT / "apps" / "facts" / "main.cpp").read_text(encoding="utf-8")

    @classmethod
    def keys_after(cls, start, end):
        """The `{"key",` names between the first `start` and the next `end` in the source."""
        begin = cls.SOURCE.index(start)
        stop = cls.SOURCE.index(end, begin)
        return set(re.findall(r'\{"([a-z_0-9]+)"', cls.SOURCE[begin:stop]))

    def test_each_part_prints_what_the_comparison_reads(self):
        printed = {
            "queried": self.keys_after("Json class_of", "\n}\n"),
            "synthetic": self.keys_after("Json class_of", "\n}\n"),
            "raw": self.keys_after("Json raw{", "// The values resident"),
            "engine_occupancy_limits": self.keys_after("Json occupancy{", "// What the query"),
            "engine_table_facts": self.keys_after("Json facts_json", "\n}\n"),
            "engine_row_facts": self.keys_after("Json facts_json", "\n}\n"),
        }
        for part, keys in facts_compare.NUMBER_KEYS.items():
            self.assertLessEqual(set(keys), printed[part], f"{part} is not printed by the probe")
        for part, key in (facts_compare.STRING_KEYS + facts_compare.BOOL_KEYS +
                          facts_compare.INTEGER_KEYS):
            self.assertIn(key, printed[part], f"{part}.{key}")
        feature_keys = self.keys_after("Json features_of", "\n}\n")
        self.assertLessEqual(set(FEATURES), feature_keys)
        self.assertIn("features", printed["queried"])

    def test_the_device_entry_prints_every_part(self):
        entry = self.keys_after('return Json{{"index", device}', "\n}\n")
        self.assertLessEqual({"index", "pci_bus_id", "mig_mode"} | set(facts_compare.DEVICE_PARTS), entry)
        self.assertIn('{"error", error.what()}', self.SOURCE)

    def test_the_occupancy_values_are_the_engines_constants(self):
        # The probe prints the named constants resident_ctas_per_sm uses, not copies of them.
        kernel = (ROOT / "src" / "core" / "kernel_occupancy.cpp").read_text(encoding="utf-8")
        for name in ("MaxThreadsPerBlock", "RegsPerBlock", "WarpSize", "SharedPerBlock"):
            self.assertIn(f"kAssumed{name}", kernel)
            self.assertIn(f"ninferno::kAssumed{name}", self.SOURCE)
        self.assertIn("reserved_shared_bytes_per_cta(q.major)", self.SOURCE)
        self.assertIn("reserved_shared_bytes_per_cta(d.major)", kernel)


RECORDED_PROBE = ROOT / "tests" / "data" / "facts_probe_recorded.json"


@unittest.skipUnless(RECORDED_PROBE.exists(), "no recorded probe output yet (see RECORDED_PROBE)")
class RecordedProbe(unittest.TestCase):
    """A real ninferno-facts-probe output, recorded on a GPU machine and committed here: the
    comparison must read it without an invalid-input error."""

    def test_the_recording_is_valid_and_has_no_mismatch(self):
        done = subprocess.run([sys.executable, "-B", TOOL, "--probe-json", str(RECORDED_PROBE)],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
