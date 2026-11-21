"""tools/audit/facts_compare.py and facts_vs_driver.py on canned driver output, no GPU (#612 step 2).

The canned device is an RTX 4090 as ninferno-facts-probe prints it: the driver's values in
"queried", synthetic_device_class's architecture limits in "synthetic", the raw attributes the
query does not keep, the engine's readings of the table. Every value is written out here, none is
computed by the tool under test. Each test changes one thing and checks the field that must fail.
A real recorded probe output goes to tests/data/facts_probe_recorded.json (see RecordedProbe).
"""
import copy
import json
import math
import os
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
            "mma_slow": False, "driver_read": False}


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


# llround(snap_to_fact_grid(x)) of src/core/machine_facts.cpp, worked out by hand from
# 2^(round(16 log2 x) / 16): the clock the engine writes for a driver reading of x kHz.
SNAP_9_000_000 = 9_147_842
SNAP_1_800_000 = 1_763_488  # a laptop's driver boost clock, 1 800 000 kHz, on the grid
SNAP_10_001_000 = 9_975_792  # the RTX 4070 GDDR6 board's driver reading, 10 001 000 kHz


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
        # information; anything farther is a wrong row. A GeForce row records the reference
        # boost only, so a driver clock of any height above it is a partner board's (info); a
        # non-GeForce row keeps the 7 % ceiling; below the window is a mismatch on both.
        pro = row_of(0x2684, segment="pro")
        for ratio, row, want in ((1.014, None, "match"), (0.986, None, "match"),
                                 (1.016, None, "info"), (0.984, None, "info"),
                                 (1.069, None, "info"), (1.10, None, "info"),
                                 (1.20, None, "info"), (0.971, None, "info"),
                                 (0.969, None, "mismatch"), (1.069, pro, "info"),
                                 (1.071, pro, "mismatch"), (0.969, pro, "mismatch")):
            d = device_4090()
            d["queried"]["sm_clock_khz"] = int(2_520_000 * ratio)
            self.assertEqual(statuses(d, row or "table")[("table", "boost_clock_khz")], want,
                             (ratio, row and "pro"))

    def test_vram_windows(self):
        # GDDR ECC keeps 1/16 on a professional board; a GeForce board reports nearly all.
        pro = row_of(0x2684, segment="pro")
        for gib_share, row, want in ((0.928, pro, "match"), (0.91, pro, "mismatch"),
                                     (0.928, None, "mismatch"), (0.98, None, "match"),
                                     (0.96, None, "match"), (0.94, None, "mismatch")):
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
        d["engine_table_facts"].update(source=2, memory_bus_bits=256,
                                       memory_clock_khz=SNAP_9_000_000, driver_read=True)
        row = row_of(0x2684, variants=[{"bus_bits": 256, "gbps": 18.0}])
        s = no_mismatch(self, statuses(d, row))
        self.assertEqual(s[("table", "engine_memory_bus_bits")], "match")
        d["engine_table_facts"]["memory_clock_khz"] = 11_000_000  # faster than the row's: wrong
        self.assertEqual(statuses(d, row)[("table", "engine_memory_clock_khz")], "mismatch")
        d["engine_table_facts"].update(memory_clock_khz=SNAP_9_000_000, memory_bus_bits=192)
        self.assertEqual(statuses(d, row)[("table", "engine_memory_bus_bits")], "mismatch")

    def test_a_read_memory_clock_must_be_the_exact_snapped_value(self):
        # The engine writes llround(snap_to_fact_grid(driver clock)): any other value is wrong,
        # a tiny one included, and so is the raw driver reading.
        row = row_of(0x2684, variants=[{"gbps": 18.0}])
        d = device_4090()
        d["queried"]["memory_clock_khz"] = 10_001_000
        d["engine_table_facts"].update(memory_clock_khz=SNAP_10_001_000, source=2, driver_read=True)
        self.assertEqual(facts_compare.engine_snapped_khz(10_001_000), SNAP_10_001_000)
        self.assertEqual(facts_compare.engine_snapped_khz(9_000_000), SNAP_9_000_000)
        self.assertEqual(statuses(d, row)[("table", "engine_memory_clock_khz")], "match")
        for wrong in (1, SNAP_10_001_000 - 1, SNAP_10_001_000 + 1, 10_001_000, 10_500_000):
            d["engine_table_facts"]["memory_clock_khz"] = wrong
            self.assertEqual(statuses(d, row)[("table", "engine_memory_clock_khz")], "mismatch",
                             wrong)

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

    def test_a_variant_is_one_board_not_a_mix_of_two(self):
        # GTX 1650: the only recorded variant is 12 Gbps with a 1590 MHz boost. The base board's
        # memory with the variant's boost, or the variant's memory with the base boost, is
        # neither board.
        base_memory = device_1650()
        base_memory["queried"]["sm_clock_khz"] = 1_590_000
        s = statuses(base_memory)
        self.assertEqual(s[("table", "memory_clock_khz")], "match")
        self.assertEqual(s[("table", "boost_clock_khz")], "mismatch")
        variant_memory = device_1650()
        variant_memory["queried"]["memory_clock_khz"] = 6_000_000
        self.assertEqual(statuses(variant_memory)[("table", "memory_clock_khz")], "mismatch")
        whole = device_1650()
        whole["queried"].update(memory_clock_khz=6_000_000, sm_clock_khz=1_590_000)
        self.assertEqual(no_mismatch(self, statuses(whole))[("table", "boost_clock_khz")], "info")
        # A variant that differs in several fields must fit in all of them (memory size too).
        row = row_of(0x2684, variants=[{"vram_gib": 12, "bus_bits": 192}])
        d = device_4090()
        d["queried"].update(global_bytes=int(12 * GIB * 0.98), memory_bus_bits=384)
        self.assertEqual(statuses(d, row)[("table", "vram_gib")], "mismatch")

    def test_source_of_a_laptop_is_extrapolated(self):
        laptop = device_laptop()
        laptop["engine_table_facts"]["source"] = 1
        self.assertEqual(statuses(laptop)[("table", "engine_source")], "mismatch")
        self.assertEqual(statuses(device_laptop())[("table", "engine_source")], "match")

    def driver_read_of(self, d, row="table"):
        return statuses(d, row)[("table", "engine_driver_read")]

    def test_driver_read_is_false_for_a_plain_whole_board(self):
        d = device_4090()
        self.assertEqual(self.driver_read_of(d), "match")
        self.assertEqual(statuses(d)[("table", "engine_row_driver_read")], "match")
        d["engine_table_facts"]["driver_read"] = True  # the query claims it read the driver
        self.assertEqual(self.driver_read_of(d), "mismatch")
        d = device_4090()
        d["engine_row_facts"]["driver_read"] = True
        self.assertEqual(statuses(d)[("table", "engine_row_driver_read")], "mismatch")

    def test_driver_read_is_true_for_a_slower_memory_variant(self):
        slow_row = row_of(0x2684, variants=[{"gbps": 18.0}])
        slow = device_4090()
        slow["queried"]["memory_clock_khz"] = 9_000_000
        slow["engine_table_facts"].update(memory_clock_khz=SNAP_9_000_000, source=2,
                                          driver_read=True)
        self.assertEqual(self.driver_read_of(slow, slow_row), "match")
        slow["engine_table_facts"]["driver_read"] = False
        self.assertEqual(self.driver_read_of(slow, slow_row), "mismatch")

    def test_driver_read_is_true_for_a_bus_variant(self):
        bus_row = row_of(0x2684, variants=[{"bus_bits": 256}])
        bus = device_4090()
        bus["queried"]["memory_bus_bits"] = 256
        bus["engine_table_facts"].update(memory_bus_bits=256, source=2, driver_read=True)
        self.assertEqual(self.driver_read_of(bus, bus_row), "match")
        bus["engine_table_facts"]["driver_read"] = False
        self.assertEqual(self.driver_read_of(bus, bus_row), "mismatch")

    def test_driver_read_is_false_for_a_slice(self):
        # The engine prices a slice's share apart and reads no driver value into the facts.
        d = device_a100()
        d["mig_mode"] = 1
        d["queried"].update(sms=42, l2_bytes=20 * MIB, global_bytes=40_000_000_000,
                            memory_bus_bits=2560)
        d["engine_table_facts"]["source"] = 2
        self.assertEqual(self.driver_read_of(d), "match")
        d["engine_table_facts"]["driver_read"] = True
        self.assertEqual(self.driver_read_of(d), "mismatch")

    def test_driver_read_is_false_for_a_faster_memory_variant(self):
        fast = device_1650()
        fast["queried"].update(memory_clock_khz=6_000_000, sm_clock_khz=1_590_000)
        self.assertEqual(self.driver_read_of(fast), "match")
        fast["engine_table_facts"]["driver_read"] = True
        self.assertEqual(self.driver_read_of(fast), "mismatch")

    def test_driver_read_of_a_laptop_follows_its_boost_range(self):
        # A laptop row with a boost range is priced from it: nothing read. Without one the engine
        # reads the driver's clock, and sets the flag even for the row alone.
        d = device_laptop()
        self.assertEqual(self.driver_read_of(d), "match")
        d["engine_table_facts"]["driver_read"] = True
        self.assertEqual(self.driver_read_of(d), "mismatch")
        row = row_of(0x2C18)
        del row["boost_min_mhz"]
        d = device_laptop()
        d["engine_table_facts"].update(driver_read=True, reference_clock_khz=SNAP_1_800_000)
        d["engine_row_facts"].update(driver_read=True, reference_clock_khz=2_160_000)
        s = statuses(d, row)
        self.assertEqual(s[("table", "engine_driver_read")], "match")
        self.assertEqual(s[("table", "engine_row_driver_read")], "match")
        d["engine_row_facts"]["driver_read"] = False
        self.assertEqual(statuses(d, row)[("table", "engine_row_driver_read")], "mismatch")

    def test_a_laptop_without_a_range_is_priced_at_the_lower_of_its_top_and_the_drivers_clock(self):
        row = row_of(0x2C18)
        del row["boost_min_mhz"]
        top = row["boost_mhz"] * 1000
        for sm_khz, want in ((1_800_000, SNAP_1_800_000), (3_000_000, top), (0, top),
                             (-1, top)):
            d = device_laptop()
            d["queried"]["sm_clock_khz"] = sm_khz
            d["engine_table_facts"].update(driver_read=True, reference_clock_khz=want)
            d["engine_row_facts"].update(driver_read=True, reference_clock_khz=top)
            key = ("table", "engine_reference_clock_khz")
            self.assertEqual(statuses(d, row)[key], "match", sm_khz)
            for wrong in (1, want - 1, want + 1):  # a stored 1 passed the old loose bound
                d["engine_table_facts"]["reference_clock_khz"] = wrong
                self.assertEqual(statuses(d, row)[key], "mismatch", (sm_khz, wrong))

    def test_source_of_a_slice_is_extrapolated(self):
        sliced = device_a100()
        sliced["mig_mode"] = 1
        sliced["queried"]["sms"] = 42
        sliced["engine_table_facts"]["source"] = 1
        self.assertEqual(statuses(sliced)[("table", "engine_source")], "mismatch")

    def test_source_of_a_bus_variant_is_extrapolated(self):
        bus_row = row_of(0x2684, variants=[{"bus_bits": 256}])
        bus = device_4090()
        bus["queried"]["memory_bus_bits"] = 256
        bus["engine_table_facts"].update(memory_bus_bits=256, source=1)
        self.assertEqual(statuses(bus, bus_row)[("table", "engine_source")], "mismatch")
        bus["engine_table_facts"]["source"] = 2
        self.assertEqual(statuses(bus, bus_row)[("table", "engine_source")], "match")

    def test_source_of_a_slower_memory_variant_is_extrapolated(self):
        slow_row = row_of(0x2684, variants=[{"gbps": 18.0}])
        slow = device_4090()
        slow["queried"]["memory_clock_khz"] = 9_000_000
        slow["engine_table_facts"].update(memory_clock_khz=SNAP_9_000_000, source=1)
        self.assertEqual(statuses(slow, slow_row)[("table", "engine_source")], "mismatch")
        slow["engine_table_facts"]["source"] = 2
        self.assertEqual(statuses(slow, slow_row)[("table", "engine_source")], "match")

    def test_source_of_a_faster_memory_variant_stays_verified(self):
        # The 1650's GDDR6 keeps the row's rate: the engine reads nothing from the driver.
        fast = device_1650()
        fast["queried"].update(memory_clock_khz=6_000_000, sm_clock_khz=1_590_000)
        self.assertEqual(statuses(fast)[("table", "engine_source")], "match")
        fast["engine_table_facts"]["source"] = 2
        self.assertEqual(statuses(fast)[("table", "engine_source")], "mismatch")

    def test_source_of_a_memory_under_2_2_percent_slower_stays_verified(self):
        near_row = row_of(0x2684, variants=[{"gbps": 20.6}])
        near = device_4090()
        near["queried"]["memory_clock_khz"] = 10_300_000
        self.assertEqual(statuses(near, near_row)[("table", "memory_clock_khz")], "info")
        self.assertEqual(statuses(near, near_row)[("table", "engine_source")], "match")
        near["engine_table_facts"]["source"] = 2
        self.assertEqual(statuses(near, near_row)[("table", "engine_source")], "mismatch")

    def test_variants_are_boards_that_differ_from_the_row(self):
        recorded = {i: r["variants"] for i, r in ROWS.items() if "variants" in r}
        self.assertEqual(recorded, {0x2D04: [{"vram_gib": 8}],
                                    0x2BB3: [{"vram_gib": 72}],
                                    0x2D18: [{"vram_gib": 12}],
                                    0x2D58: [{"vram_gib": 12}],
                                    0x1F82: [{"gbps": 12.0, "boost_mhz": 1590}],
                                    0x2786: [{"gbps": 20.0}],
                                    0x2503: [{"vram_gib": 8, "bus_bits": 128}]})
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


MACHINE_FACTS_CPP = (ROOT / "src" / "core" / "machine_facts.cpp").read_text(encoding="utf-8")


def snap_to_grid(x):
    """snap_to_fact_grid of machine_facts.cpp: x on the 2^(1/16) grid."""
    return 2.0 ** (math.floor(16.0 * math.log2(x) + 0.5) / 16.0)


def engine_expect(row, q):
    """(source, driver_read, bus, memory clock) machine_facts stores for a row and the driver's
    values: a transcription of the row branch of machine_facts.cpp, written here without
    facts_compare, so a change of the rule in one place and not in the other fails."""
    laptop = row["form"] == "laptop"
    source, read = 1, False
    bus, clock = row["bus_bits"], math.floor(row["gbps"] * 1e6 / 2.0 + 0.5)
    if laptop:
        source = 2
        if not row.get("boost_min_mhz", 0) > 0:
            read = True
    if q["sms"] > 0 and q["sms"] != row["sms"]:
        source = 2
    whole = not laptop and (q["sms"] <= 0 or q["sms"] == row["sms"])
    if whole and q["memory_bus_bits"] >= 32 and q["memory_bus_bits"] % 32 == 0 and \
            q["memory_bus_bits"] != row["bus_bits"]:
        source, read, bus = 2, True, q["memory_bus_bits"]
    if whole and q["memory_clock_khz"] >= 100000 and q["memory_clock_khz"] * 1000 < clock * 978:
        source, read = 2, True
        clock = math.floor(snap_to_grid(q["memory_clock_khz"]) + 0.5)
    return source, read, bus, clock


def device_of_row(row, **driver):
    """A device of the given row as the engine would store it for the driver's values (sms,
    memory_bus_bits, memory_clock_khz, global_bytes, sm_clock_khz, major, minor)."""
    d = device_4090()
    cc = row["cc"]
    l2 = int(row.get("l2_mib", 1) * MIB)
    base = dict(pci_device_id=int(row["id"], 16), name="NVIDIA " + row["name"], chip=row["chip"],
                major=cc // 10, minor=cc % 10, sms=row["sms"], l2_bytes=l2,
                global_bytes=int(row["vram_gib"] * GIB * 0.99), memory_bus_bits=row["bus_bits"],
                memory_clock_khz=int(row["gbps"] * 1e6 / 2) + 1000,
                sm_clock_khz=row["boost_mhz"] * 1000, persisting_l2_max_bytes=l2 * 3 // 4)
    base.update(driver)
    d["queried"].update(base)
    d["synthetic"].update(pci_device_id=base["pci_device_id"], chip=row["chip"],
                          major=base["major"], minor=base["minor"], sms=base["sms"],
                          persisting_l2_max_bytes=base["persisting_l2_max_bytes"])
    laptop = row["form"] == "laptop"
    ref = row.get("boost_min_mhz", 0) * 1000 or row["boost_mhz"] * 1000
    alone = {"source": 2 if laptop else 1, "sku": row["name"], "reference_clock_khz": ref,
             "memory_bus_bits": row["bus_bits"],
             "memory_clock_khz": math.floor(row["gbps"] * 1e6 / 2.0 + 0.5),
             "l2_mib": float(row.get("l2_mib", 0)), "mma_slow": row.get("mma_slow", False),
             "driver_read": laptop and not row.get("boost_min_mhz", 0) > 0}
    d["engine_row_facts"] = dict(alone)
    source, read, bus, clock = engine_expect(row, d["queried"])
    d["engine_table_facts"] = dict(alone, source=source, driver_read=read, memory_bus_bits=bus,
                                   memory_clock_khz=clock)
    return d


def engine_statuses(d, row):
    return {r["field"]: r["status"] for r in facts_compare.compare_device(d, row)
            if r["field"].startswith("engine_")}


class EngineRule(unittest.TestCase):
    """The audit's expectation of the engine's stored facts is the engine's own rule: computed
    from the driver's values, never from the audit's statuses."""

    def test_the_thresholds_are_the_engines(self):
        # machine_facts.cpp is read, so a threshold changed there and not in the audit fails here.
        text = " ".join(MACHINE_FACTS_CPP.split())
        bus = re.search(r"driver\.memory_bus_bits >= (\d+) && driver\.memory_bus_bits % (\d+) == 0",
                        text)
        self.assertIsNotNone(bus)
        self.assertEqual({int(g) for g in bus.groups()}, {facts_compare.ENGINE_BUS_CHANNEL_BITS})
        clock = re.search(r"driver\.memory_clock_khz >= (\d+) && std::int64_t\{driver\."
                          r"memory_clock_khz\} \* 1000 < std::int64_t\{f\.memory_clock_khz\} "
                          r"\* (\d+)\)", text)
        self.assertIsNotNone(clock)
        self.assertEqual(int(clock.group(1)), facts_compare.ENGINE_MIN_MEMORY_CLOCK_KHZ)
        self.assertEqual(int(clock.group(2)), facts_compare.ENGINE_SLOWER_MEMORY_PER_MILLE)
        self.assertEqual(facts_compare.ENGINE_SLOWER_MEMORY_PER_MILLE, 978)
        # 978/1000 is 2^(-1/32) = 0.97857 truncated to the thousandth.
        self.assertEqual(math.floor(2 ** (-1 / 32) * 1000), 978)
        self.assertIn("const bool whole_board = row->form != FormFactor::Laptop && "
                      "(driver.sms <= 0 || driver.sms == row->sms);", text)
        self.assertIn("if (row->boost_min_mhz > 0) {", text)

    def test_the_grid_is_the_engines(self):
        # engine_snapped_khz is a port of the engine's grid. Both are checked against one fixture
        # of hand-worked points (tests/fixtures/facts/grid_points.txt): the engine's test
        # (tests/test_machine_facts.cpp) checks snap_to_fact_grid against it, this test checks the
        # port. A grid changed in the engine fails the engine's test until the fixture changes, and
        # then the port fails here.
        text = (ROOT / "tests" / "fixtures" / "facts" / "grid_points.txt").read_text(encoding="utf-8")
        points = []
        for line in text.splitlines():
            if line and not line.startswith("#"):
                driver_khz, stored_khz = line.split()
                points.append((int(driver_khz), int(stored_khz)))
        self.assertGreaterEqual(len(points), 20)
        for driver_khz, stored_khz in points:
            self.assertEqual(facts_compare.engine_snapped_khz(driver_khz), stored_khz, driver_khz)

    def test_every_engine_field_follows_the_engine_over_a_grid_of_driver_values(self):
        no_range = row_of(0x2C18)
        del no_range["boost_min_mhz"]
        rows = [ROWS[0x2684], ROWS[0x2C18], no_range, ROWS[0x20B2], ROWS[0x2786], ROWS[0x2503]]
        checked = 0
        for row in rows:
            row_khz = math.floor(row["gbps"] * 1e6 / 2.0 + 0.5)
            threshold = row_khz * 978 // 1000
            clocks = {0, 1, 99999, 100000, row_khz, row_khz + 1000, row_khz - 1000,
                      threshold - 1, threshold, threshold + 1, threshold + 2}
            clocks |= {int(row_khz * f) for f in (0.99, 0.985, 0.98, 0.97, 0.95, 0.9)}
            buses = {-1, 0, 1, 31, 32, 100, 128, 192, 256, 384, row["bus_bits"], 5120}
            sms = {-1, 0, row["sms"], row["sms"] - 1, row["sms"] + 1, max(1, row["sms"] // 4)}
            for s in sms:
                for bus in buses:
                    for clock in clocks:
                        d = device_of_row(row, sms=s, memory_bus_bits=bus, memory_clock_khz=clock)
                        bad = [f for f, st in engine_statuses(d, row).items() if st == "mismatch"]
                        self.assertEqual(bad, [], (row["id"], s, bus, clock))
                        checked += 1
        self.assertGreater(checked, 3000)

    def test_the_grid_is_not_vacuous(self):
        # The same facts with the flag or the source flipped fail, at both sides of each edge.
        row = ROWS[0x2684]
        for clock, read in ((10_268_999, True), (10_269_000, False), (10_269_001, False),
                            (10_200_000, True), (10_300_000, False), (99_999, False)):
            d = device_of_row(row, memory_clock_khz=clock)
            self.assertEqual(d["engine_table_facts"]["driver_read"], read, clock)
            self.assertEqual(engine_statuses(d, row)["engine_driver_read"], "match", clock)
            d["engine_table_facts"]["driver_read"] = not read
            self.assertEqual(engine_statuses(d, row)["engine_driver_read"], "mismatch", clock)
            d["engine_table_facts"]["driver_read"] = read
            d["engine_table_facts"]["source"] = 1 if read else 2
            self.assertEqual(engine_statuses(d, row)["engine_source"], "mismatch", clock)

    def test_a_laptop_with_a_recorded_variant_keeps_the_rows_memory(self):
        # machine_facts reads nothing for a laptop with a boost range, whatever `variants` says
        # (test_machine_facts.cpp, laptops_conservative).
        row = row_of(0x2C18, variants=[{"bus_bits": 192}, {"gbps": 16.0}])
        d = device_of_row(row, memory_bus_bits=192, memory_clock_khz=8_001_000)
        self.assertFalse(d["engine_table_facts"]["driver_read"])
        self.assertEqual(d["engine_table_facts"]["memory_bus_bits"], 256)
        s = engine_statuses(d, row)
        self.assertEqual(s["engine_driver_read"], "match")
        d["engine_table_facts"]["driver_read"] = True  # as if a laptop was read like a board
        self.assertEqual(engine_statuses(d, row)["engine_driver_read"], "mismatch")

    def test_a_slice_never_reads_whatever_the_mig_mode_says(self):
        row = ROWS[0x20B2]
        for mode in (-1, 0, 1):
            d = device_of_row(row, sms=54, memory_bus_bits=2560, memory_clock_khz=1_000_000)
            d["mig_mode"] = mode
            self.assertEqual((d["engine_table_facts"]["source"],
                              d["engine_table_facts"]["driver_read"]), (2, False))
            self.assertNotIn("mismatch", engine_statuses(d, row).values(), mode)
            d["engine_table_facts"]["driver_read"] = True
            self.assertEqual(engine_statuses(d, row)["engine_driver_read"], "mismatch", mode)

    def test_a_failing_board_still_has_the_engines_facts_checked_as_the_engine_computes_them(self):
        # No `variants`, so the table fields fail, but the engine did what its code says: the
        # engine rows must not blame it.
        row = ROWS[0x2684]
        for change in ({"memory_bus_bits": 128}, {"memory_clock_khz": 9_000_000},
                       {"sms": 100}, {"sms": 200}):
            d = device_of_row(row, **change)
            self.assertIn("mismatch", statuses(d, row).values(), change)
            self.assertEqual([f for f, st in engine_statuses(d, row).items() if st == "mismatch"],
                             [], change)
        # An id the driver reports under another compute capability is unknown to the engine.
        d = device_of_row(row, major=9, minor=0)
        d["engine_table_facts"].update(sku="", driver_read=True)
        s = statuses(d, row)
        self.assertEqual(s[("table", "cc")], "mismatch")
        self.assertEqual(s[("table", "engine_sku")], "match")
        self.assertEqual(s[("table", "engine_driver_read")], "match")
        d["engine_table_facts"]["driver_read"] = False
        self.assertEqual(statuses(d, row)[("table", "engine_driver_read")], "mismatch")

    def test_a_recorded_bus_must_be_one_the_engine_reads(self):
        row = ROWS[0x2684]
        for width in (100, 33, 49, 208):
            with self.assertRaises(ValueError, msg=width):
                facts_compare.check_variants(dict(row, variants=[{"bus_bits": width}]))
        for width in (32, 64, 192, 256):
            facts_compare.check_variants(dict(row, variants=[{"bus_bits": width}]))

    def test_the_engines_own_documented_variants_pass_with_the_shipped_table(self):
        # machine_facts.cpp and test_machine_facts.cpp (memory_variants): an RTX 4070 with GDDR6
        # under 0x2786 and an 8 GB RTX 3060 on a 128-bit bus under 0x2503.
        narrow_driver = dict(memory_clock_khz=7_501_000, memory_bus_bits=128,
                             global_bytes=int(8 * GIB * 0.99))
        cases = [
            (0x2786, dict(memory_clock_khz=10_501_000)),   # the GDDR6X board
            (0x2786, dict(memory_clock_khz=10_001_000)),   # the GDDR6 board
            (0x2503, dict(memory_clock_khz=7_501_000)),    # the 12 GB board
            (0x2503, narrow_driver),                       # the 8 GB board
        ]
        for pci, driver in cases:
            d = device_of_row(ROWS[pci], **driver)
            summary, _ = facts_vs_driver.run({"devices": [d]}, ROWS)
            self.assertTrue(summary["ok"], (hex(pci), driver, summary["devices"][0]["mismatches"]))
        g6 = device_of_row(ROWS[0x2786], memory_clock_khz=10_001_000)
        self.assertEqual((g6["engine_table_facts"]["source"], g6["engine_table_facts"]["driver_read"]),
                         (2, True))
        narrow = device_of_row(ROWS[0x2503], **narrow_driver)
        self.assertEqual(narrow["engine_table_facts"]["memory_bus_bits"], 128)
        # A board of that kind that the table does not record still fails: a 128-bit bus with the
        # 12 GB board's memory size.
        wrong = device_of_row(ROWS[0x2503], memory_bus_bits=128)
        summary, _ = facts_vs_driver.run({"devices": [wrong]}, ROWS)
        self.assertFalse(summary["ok"])


class ValidationReview(unittest.TestCase):
    def test_driver_read_is_a_boolean_in_both_parts(self):
        # Written out, not derived from BOOL_KEYS.
        for part in ("engine_table_facts", "engine_row_facts"):
            for bad in (1, 0, None, "true", [True]):
                d = device_4090()
                d[part]["driver_read"] = bad
                with self.assertRaises(ValueError, msg=(part, bad)):
                    facts_compare.validate_probe({"devices": [d]})
            d = device_4090()
            del d[part]["driver_read"]
            with self.assertRaises(ValueError, msg=part):
                facts_compare.validate_probe({"devices": [d]})
        self.assertEqual(len([k for k in facts_compare.BOOL_KEYS if k[1] == "driver_read"]), 2)

    def test_the_probe_prints_driver_read_in_both_facts(self):
        printed = ProbeSource.keys_after("Json facts_json", "\n}\n")
        self.assertIn("driver_read", printed)
        self.assertIn("mma_slow", printed)

    def test_is_number_does_not_overflow(self):
        for big in (10 ** 400, -10 ** 400, float("inf")):
            self.assertFalse(facts_compare.is_number(big))
        self.assertTrue(facts_compare.is_number(10 ** 300))
        self.assertFalse(facts_compare.is_number(True))

    def test_rows_with_a_wrong_boost_min_are_refused(self):
        good = dict(ROWS[0x2C18])
        facts_compare.check_row(good)
        for bad in ("1597", None, True, -1, float("nan")):
            with self.assertRaises(ValueError, msg=bad):
                facts_compare.check_row(dict(good, boost_min_mhz=bad))


class MalformedInput(unittest.TestCase):
    """Every bad input is one line on stderr and exit 2; exit 1 means the audit found a mismatch."""

    def run_text(self, text, *extra, env=None, raw=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.json"
            path.write_bytes(raw if raw is not None else text.encode("utf-8"))
            done = subprocess.run([sys.executable, "-B", TOOL, "--probe-json", str(path)] +
                                  list(extra), capture_output=True, text=True, env=env)
        return done

    def assert_clean_exit_2(self, done, label):
        self.assertEqual(done.returncode, 2, f"{label}: {done.stdout} {done.stderr}")
        self.assertNotIn("Traceback", done.stderr, label)
        self.assertEqual(len(done.stderr.strip().splitlines()), 1, f"{label}: {done.stderr}")

    def test_an_error_device_with_a_queried_that_is_not_an_object(self):
        for queried in ([1], "s", None, 5):
            probe = {"devices": [{"index": 0, "error": "x", "queried": queried}]}
            self.assert_clean_exit_2(self.run_text(json.dumps(probe)), repr(queried))
        probe = {"devices": [{"index": 0, "error": "x", "queried": {"name": 5}}]}
        self.assert_clean_exit_2(self.run_text(json.dumps(probe)), "name 5")
        probe = {"devices": [{"index": 0, "error": "x", "pci_bus_id": [1]}]}
        self.assert_clean_exit_2(self.run_text(json.dumps(probe)), "pci_bus_id")
        ok = {"devices": [{"index": 0, "error": "x", "queried": {"name": "n"}}]}
        self.assertEqual(self.run_text(json.dumps(ok)).returncode, 1)  # a mismatch, not bad input

    def test_huge_numbers(self):
        huge = "1" + "0" * 400
        self.assert_clean_exit_2(self.run_text('{"devices":[{"index":%s,"error":"x"}]}' % huge),
                                 "index")
        for key in ("sms", "memory_clock_khz", "global_bytes"):
            d = device_4090()
            text = json.dumps({"devices": [d]}).replace(
                f'"{key}": {d["queried"][key]}', f'"{key}": {huge}', 1)
            self.assertIn(huge, text)
            self.assert_clean_exit_2(self.run_text(text), key)

    def test_deeply_nested_json(self):
        self.assert_clean_exit_2(self.run_text("[" * 100000), "nesting")

    def test_a_name_the_console_cannot_encode(self):
        d = device_4090()
        d["queried"]["name"] = "NVIDIA GeForce RTX 4090 é中"
        env = dict(os.environ, PYTHONIOENCODING="ascii")
        done = self.run_text(json.dumps({"devices": [d]}), env=env)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("Traceback", done.stderr)
        self.assertIn("RTX 4090", done.stdout)

    def test_a_utf8_bom_is_accepted(self):
        raw = b"\xef\xbb\xbf" + json.dumps({"devices": [device_4090()]}).encode("utf-8")
        self.assertEqual(self.run_text("", raw=raw).returncode, 0)

    def test_an_index_is_a_small_whole_number(self):
        for index in (-1, 0.5, 1.0, "0", True, None, 10 ** 6):
            d = device_4090()
            d["index"] = index
            self.assert_clean_exit_2(self.run_text(json.dumps({"devices": [d]})), repr(index))

    def test_wrong_facts_files(self):
        probe = json.dumps({"devices": [device_4090()]})
        for name, content in (("no gpus", {"schema": 1}), ("null gpus", {"gpus": None}),
                              ("list", []), ("empty", {"gpus": []}), ("number", 5)):
            with tempfile.TemporaryDirectory() as tmp:
                facts = Path(tmp) / "facts.json"
                facts.write_text(json.dumps(content), encoding="utf-8")
                done = self.run_text(probe, "--facts", str(facts))
            self.assert_clean_exit_2(done, name)
            self.assertIn("gpu facts", done.stderr, name)


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


RECORDINGS = ROOT / "tests" / "data" / "facts_probe"
# Mismatches a recording still holds, by field (#612 step 2). Every other recording compares with
# none. The Quadro GV100 file was recorded before the synthetic Volta class stopped overriding the
# architecture row (#698): its "synthetic" block is the old one, so these four fields differ from
# the driver. They go when that card is probed again with a build that has #698; remove the entry
# then.
EXPECTED_MISMATCHES = {
    "quadro_gv100": {("arch", "smem_per_block_optin"), ("arch", "smem_per_sm"),
                     ("arch", "max_threads_per_sm"),
                     ("arch", "synthetic_smem_optin_plus_reserved_is_per_sm")},
}


def recorded_mismatches(path):
    """(source, field) of every mismatch the audit finds in one recording, by the CLI."""
    done = subprocess.run([sys.executable, "-B", TOOL, "--probe-json", str(path)],
                          capture_output=True, text=True)
    summary = json.loads(done.stdout[done.stdout.index("\n{") + 1:])
    found = {(m["source"], m["field"]) for dev in summary["devices"] for m in dev["mismatches"]}
    return done, found


@unittest.skipUnless(RECORDINGS.is_dir(), "no per-card recordings")
class RecordedProbes(unittest.TestCase):
    """One real ninferno-facts-probe output per lab card, recorded on that card."""

    def test_every_recording_is_valid_and_only_the_named_mismatches_remain(self):
        files = sorted(RECORDINGS.glob("*.json"))
        self.assertTrue(files)
        for path in files:
            with self.subTest(card=path.stem):
                done, found = recorded_mismatches(path)
                self.assertIn(done.returncode, (0, 1), done.stdout + done.stderr)
                self.assertEqual(found, EXPECTED_MISMATCHES.get(path.stem, set()), done.stdout)
                self.assertEqual(done.returncode, 1 if found else 0)

    def test_a_clock_or_memory_below_the_window_still_fails_on_a_recording(self):
        # The RTX 3090 and RTX 2080 Ti compare clean (a partner board's clock, a driver-reserved
        # slice of memory); a board below the reference clock window or short of memory does not.
        for card, change in (("rtx3090", {"sm_clock_khz": 1_600_000}),
                             ("rtx2080ti", {"sm_clock_khz": 1_400_000}),
                             ("rtx2080ti", {"global_bytes": int(11 * GIB * 0.94)})):
            with self.subTest(card=card, change=change):
                recording = json.loads((RECORDINGS / f"{card}.json").read_text(encoding="utf-8"))
                recording["devices"][0]["queried"].update(change)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "changed.json"
                    path.write_text(json.dumps(recording), encoding="utf-8")
                    done, found = recorded_mismatches(path)
                self.assertEqual(done.returncode, 1, done.stdout)
                self.assertTrue(found & {("table", "boost_clock_khz"), ("table", "vram_gib")},
                                found)


if __name__ == "__main__":
    unittest.main()
