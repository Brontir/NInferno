"""The comparison of a board's facts with what the driver reports (#612 step 2), no GPU.

Input: one device of ninferno-facts-probe's JSON ("queried": the engine's own query, "synthetic":
synthetic_device_class for the same compute capability, "raw": CUDA attributes the query does not
keep, "engine_table_facts": the MachineFacts the query stored, "engine_row_facts": machine_facts
of the table row alone, "engine_occupancy_limits": the per-block limits resident_ctas_per_sm
assumes) and the board's row of src/core/data/gpu_facts.json. Output: one result per field.
validate_probe rejects a document with a key missing or of the wrong type before any comparison.

  table  the gpu_facts.json row against the driver: chip, cc, SMs, L2, bus width, memory data
         rate, boost clock, memory size. A value off the row passes as information only where the
         row records that board in its `variants` (bus_bits, gbps, boost_mhz, vram_gib), or as a
         MIG slice (a MIG-capable chip reporting fewer SMs with MIG enabled per NVML; MIG mode
         unknown: unverified, or a slice where the memory is reduced too), a laptop's range or a
         vendor's boost window; anything else is a mismatch. The engine's reading of the row
         alone (engine_row_facts) against the JSON, exactly; the facts the query stored
         (engine_table_facts) against that reading. What the engine should have stored (source,
         driver_read, the bus and memory clock it read) follows machine_facts' own rule on the
         driver's values (a whole board, a bus in whole 32-bit channels, a clock under 978/1000
         of the row's), never the audit's statuses: the engine ignores `variants`, so a board
         the audit fails (an unrecorded variant) still has the engine's facts checked as the
         engine computes them. Where the audit's `variants` differ from that rule (a recorded
         bus that is no whole number of channels), check_row refuses the row.
  arch   the architecture limits synthetic_device_class writes for the compute capability
         against the queried ones (shared memory per block and per SM, threads, blocks and
         registers per SM) and the reserved-shared invariant of the queried class; the per-block
         limits and the reserved shared memory the occupancy calculator assumes
         (kernel_occupancy.h, "engine_occupancy_limits") against the driver's raw attributes;
         the persisting-L2 cap; the features.

Not compared, because the driver cannot answer or the probe does not print it: tdp_w, pcie,
segment, form, the driver's name against the row's (the engine's SKU is compared with it), the
MachineFacts share, and every feature but mma_slow (the build's, engine against engine). The
MachineFacts driver_read is compared (engine_driver_read, engine_row_driver_read). feature_mma_slow queried against synthetic is "derived": both classes read the same row,
so it only catches a query that did not store the facts. Whether the engine's mma_slow is the
row's is the engine_mma_slow check. A row without l2_mib is a mismatch on every board of it.

status: match, mismatch (fails the run), unavailable (the driver gave no value: the field is
unverified, listed in the summary; not a failure unless the run is strict), derived (agrees, but
from the same table, not from the driver), info (a difference that is not a failure, see above).
"""
import math

MIB = 1024 * 1024
GIB = 1024 * MIB

# Relative tolerances. Every one of them is a CHOSEN tolerance of this audit: none is a measured
# or a published figure, and none is the engine's (the engine's own thresholds are the ENGINE_*
# constants below, which a test ties to machine_facts.cpp). A board off a tolerance fails the
# audit, so widen one only for a recorded board that needs it.
#   MEMORY_CLOCK_TOLERANCE 0.005: chosen; the issue (#612) asked for +-0.5 % on the memory clock.
#   BOOST_TOLERANCE 0.015: chosen; the driver's peak SM clock of a stock board is expected within
#     1.5 % of the published reference.
#   BOOST_UNDER 0.03, BOOST_OVER 0.07: chosen window for a vendor's boost (a partner board's
#     factory overclock, an OEM board under the reference): information, not a failure.
#   L2_TOLERANCE 0.01: chosen; the driver's L2 against the marketed figure.
#   VRAM_HIGH 1.003: chosen; the driver's total cannot exceed the marketed size, 0.3 % is slack
#     for the marketed figure's rounding.
#   VRAM_LOW_GEFORCE 0.97: chosen; a GeForce board reports nearly all of its memory.
#   VRAM_LOW_ECC 0.92: derived, with a chosen margin. GDDR ECC on (an A6000, an L40, an A10)
#     keeps 1/16 of the memory, so at best 15/16 = 0.9375 of the marketed size; 0.92 leaves about
#     2 % for the driver's own reservation.
#   PERSISTING_LOW 0.125: chosen; a persisting-L2 cap under 1/8 of the L2 is not a cache cap.
L2_TOLERANCE = 0.01
MEMORY_CLOCK_TOLERANCE = 0.005
BOOST_TOLERANCE = 0.015
BOOST_UNDER, BOOST_OVER = 0.03, 0.07
VRAM_LOW_ECC, VRAM_LOW_GEFORCE, VRAM_HIGH = 0.92, 0.97, 1.003
PERSISTING_LOW = 0.125
# The engine's thresholds in machine_facts (the row branch), restated for the audit's expected
# values; tests/test_facts_vs_driver.py reads machine_facts.cpp and fails when one differs.
# It reads a driver memory clock only when it is at least ENGINE_MIN_MEMORY_CLOCK_KHZ and under
# 978/1000 of the row's. 978 is derived: half a step of the 2^(1/16) fact grid is a ratio of
# 2^(-1/32) = 0.9786, which the engine rounds down to 978/1000 in integer arithmetic.
ENGINE_SLOWER_MEMORY_PER_MILLE = 978
ENGINE_MIN_MEMORY_CLOCK_KHZ = 100000
# It reads a driver bus width only as a whole number (at least one) of 32-bit channels.
ENGINE_BUS_CHANNEL_BITS = 32
# A CUDA ordinal: a chosen bound, far above any machine's device count.
MAX_DEVICE_INDEX = 4096
VERIFIED_SOURCE = 1 # FactSource::Verified: the engine reads the row as it is
EXTRAPOLATED_SOURCE = 2  # FactSource::Extrapolated: a laptop, a memory variant or a slice


def result(source, field, expected, actual, status, note=""):
    return {"source": source, "field": field, "expected": expected, "actual": actual,
            "status": status, "note": note}


def is_number(value):
    """A finite int or float; a bool is not one."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def unavailable(actual):
    return actual is None or (is_number(actual) and actual in (0, -1))


def close(expected, actual, tolerance):
    return expected > 0 and abs(actual - expected) <= tolerance * expected


# The chips NVIDIA partitions with MIG: GA100 (A100, A30, A800), GH100 (H100, H200, H800), GB100
# (B200), and GB202 on its professional boards (RTX PRO 6000 Blackwell, workstation and server);
# not GV100 nor any GeForce board.
MIG_CHIPS = {"ga100", "gh100", "gb100"}
MIG_PRO_CHIPS = {"gb202"}
# The GTX 16 chips have no tensor cores: their rows must carry mma_slow (a static rule, no GPU).
NO_TENSOR_CHIPS = {"tu116", "tu117"}


def mig_capable(row):
    chip = row.get("chip", "")
    return chip in MIG_CHIPS or (chip in MIG_PRO_CHIPS and row.get("segment") != "geforce")


def is_laptop(row):
    return row.get("form") == "laptop"


def memory_khz(gbps):
    # The driver spells half the data rate in kHz: 28 Gbps is 14 000 000.
    return gbps * 1.0e6 / 2.0


def row_memory_khz(row):
    return memory_khz(row["gbps"])


# A row's `variants`: the other boards sold under its PCI id, each by the values in which it
# differs from the row (an 8 GB RTX 5060 Ti under the 16 GB board's id, a GDDR6 GTX 1650 under
# the GDDR5 board's, a 128-bit 3060 under the 192-bit board's). A driver value that is such a
# board's is information; any other value off the row (an overclock, a wrong row) stays a
# mismatch.
VARIANT_KEYS = ("vram_gib", "gbps", "boost_mhz", "bus_bits")
ROW_NUMBERS = ("cc", "sms", "bus_bits", "gbps", "boost_mhz", "vram_gib")


def check_variants(row):
    """Raises ValueError unless the row's `variants` (if any) are boards that differ from it."""
    variants = row.get("variants", [])
    where = f"{row.get('id')}: variants"
    if not isinstance(variants, list) or ("variants" in row and not variants):
        raise ValueError(f"{where}: expected a nonempty list")
    for v in variants:
        if not isinstance(v, dict) or not v or set(v) - set(VARIANT_KEYS):
            raise ValueError(f"{where}: each is an object of {', '.join(VARIANT_KEYS)}")
        for key, value in v.items():
            if not is_number(value) or value <= 0:
                raise ValueError(f"{where}: {key} is not a positive number")
        bus = v.get("bus_bits", ENGINE_BUS_CHANNEL_BITS)
        if bus % ENGINE_BUS_CHANNEL_BITS:
            # machine_facts reads a driver bus width only in whole 32-bit channels: any other
            # width is "as good as unreported", so the engine would not take this board's bus.
            raise ValueError(f"{where}: bus_bits is not a whole number of "
                             f"{ENGINE_BUS_CHANNEL_BITS}-bit channels (the engine would not read it)")
        if all(v[key] == row[key] for key in v):
            raise ValueError(f"{where}: a variant that does not differ from the row")


def check_row(row):
    """Raises ValueError unless a gpu_facts.json row has what the audit reads, typed."""
    where = f"{row.get('id') if isinstance(row, dict) else row}"
    if not isinstance(row, dict) or not isinstance(row.get("id"), str):
        raise ValueError(f"{where}: a row is an object with a string id")
    for key in ("name", "chip"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"{where}: {key} is not a nonempty string")
    for key in ROW_NUMBERS:
        if not is_number(row.get(key)) or row[key] <= 0:
            raise ValueError(f"{where}: {key} is missing or not a positive number")
    if "l2_mib" in row and (not is_number(row["l2_mib"]) or row["l2_mib"] < 0):
        raise ValueError(f"{where}: l2_mib is not a number")
    if "boost_min_mhz" in row and (not is_number(row["boost_min_mhz"]) or
                                   row["boost_min_mhz"] < 0):
        raise ValueError(f"{where}: boost_min_mhz is not a number")
    if row.get("chip") in NO_TENSOR_CHIPS and row.get("mma_slow") is not True:
        raise ValueError(f"{where}: a {row['chip']} row must carry mma_slow (no tensor cores)")
    check_variants(row)


def vram_floor(row):
    return VRAM_LOW_GEFORCE if row.get("segment") == "geforce" else VRAM_LOW_ECC


def recorded_variants(row, q):
    """The row's `variants` whose every differing field is the driver's value and whose every
    other field is the row's: one board, not a mix of two. A value the driver did not give fits
    any board."""
    low, boost, mem, bus = vram_floor(row), q["sm_clock_khz"], q["memory_clock_khz"], \
        q["memory_bus_bits"]
    fits = []
    for v in row.get("variants", []):
        vram_ok = low <= q["global_bytes"] / (v.get("vram_gib", row["vram_gib"]) * GIB) <= VRAM_HIGH
        bus_ok = unavailable(bus) or bus == v.get("bus_bits", row["bus_bits"])
        mem_ok = unavailable(mem) or close(memory_khz(v.get("gbps", row["gbps"])), mem,
                                           MEMORY_CLOCK_TOLERANCE)
        if "boost_mhz" in v:
            boost_ok = unavailable(boost) or close(v["boost_mhz"] * 1000, boost, BOOST_TOLERANCE)
        else:
            boost_ok = unavailable(boost) or row["boost_mhz"] * 1000 * (1 - BOOST_UNDER) <= boost \
                <= row["boost_mhz"] * 1000 * (1 + BOOST_OVER)
        if vram_ok and bus_ok and mem_ok and boost_ok:
            fits.append(v)
    return fits


def recorded_has(recorded, key):
    """True when a recorded board that fits the driver differs from the row in `key`."""
    return any(key in v for v in recorded)


# The raw attributes kernel_occupancy.h assumes a value for (engine_occupancy_limits, same names).
OCCUPANCY_LIMITS = ("shared_per_block_default", "registers_per_block", "max_threads_per_block",
                    "warp_size", "reserved_shared_per_block")
# A driver's 0 is a real answer for these (Volta and Turing reserve nothing).
ZERO_IS_REAL = ("reserved_shared_per_block",)
BOARD_FEATURES = ("mma_slow",)
FACTS_PARTS = ("engine_table_facts", "engine_row_facts")
FACTS_NUMBERS = ("source", "reference_clock_khz", "memory_bus_bits", "memory_clock_khz", "l2_mib")
DEVICE_PARTS = ("queried", "synthetic", "raw", "engine_occupancy_limits") + FACTS_PARTS
# The class fields the architecture check reads from both classes.
ARCH_NUMBERS = ("smem_per_block_optin", "smem_per_sm", "max_threads_per_sm", "max_blocks_per_sm",
                "registers_per_sm", "persisting_l2_max_bytes")
# Every number the comparison reads. The probe prints each as a number and never as null.
NUMBER_KEYS = {
    "queried": ARCH_NUMBERS + ("pci_device_id", "major", "minor", "sms", "l2_bytes", "global_bytes",
                               "memory_bus_bits", "memory_clock_khz", "sm_clock_khz"),
    "synthetic": ARCH_NUMBERS,
    "raw": OCCUPANCY_LIMITS,
    "engine_occupancy_limits": OCCUPANCY_LIMITS,
}
NUMBER_KEYS.update({part: FACTS_NUMBERS for part in FACTS_PARTS})
# Whole numbers: an id, a compute capability, an index.
INTEGER_KEYS = (("queried", "pci_device_id"), ("queried", "major"), ("queried", "minor"))
STRING_KEYS = (("queried", "name"), ("queried", "chip"), ("engine_table_facts", "sku"),
               ("engine_row_facts", "sku"))
BOOL_KEYS = (("engine_table_facts", "mma_slow"), ("engine_row_facts", "mma_slow"),
             ("engine_table_facts", "driver_read"), ("engine_row_facts", "driver_read"))


def validate_probe(probe_output):
    """Raises ValueError unless the probe document is a complete recording: a nonempty list of
    devices, each with the driver's query, the synthetic class and the engine's readings, and
    every key the comparison reads present and typed. A partial capture must not pass as a clean
    audit. A device the probe could not read carries "error" and is reported, not compared."""
    devices = probe_output.get("devices") if isinstance(probe_output, dict) else None
    if not isinstance(devices, list) or not devices:
        raise ValueError("invalid probe output: \"devices\" is missing, not a list or empty")
    seen = set()
    for position, device in enumerate(devices):
        if not isinstance(device, dict):
            raise ValueError(f"invalid probe output: device {position} is not an object")
        if type(device.get("index")) is not int or not 0 <= device["index"] < MAX_DEVICE_INDEX:
            raise ValueError(f"invalid probe output: device {position}: \"index\" is not a "
                             f"whole number from 0 to {MAX_DEVICE_INDEX - 1}")
        if device["index"] in seen:
            raise ValueError(f"invalid probe output: device index {device['index']} appears twice")
        seen.add(device["index"])
        # Every device is named in the report, so these are typed on an error device too.
        if "pci_bus_id" in device and not isinstance(device["pci_bus_id"], str):
            raise ValueError(f"invalid probe output: device {position}: pci_bus_id is not text")
        if "error" in device:
            if not isinstance(device["error"], str):
                raise ValueError(f"invalid probe output: device {position}: \"error\" is not text")
            if "queried" in device and not isinstance(device["queried"], dict):
                raise ValueError(f"invalid probe output: device {position}: \"queried\" is not "
                                 "an object")
            if not isinstance(device.get("queried", {}).get("name", ""), str):
                raise ValueError(f"invalid probe output: device {position}: queried.name is not "
                                 "text")
            continue
        for part in DEVICE_PARTS:
            if not isinstance(device.get(part), dict):
                raise ValueError(f"invalid probe output: device {position} has no \"{part}\" object")
        # The probe always prints every key, as a number (the driver's "no answer" is -1, never
        # null) or, for a name, a string; one that is missing or of another type is a bad capture.
        for part, keys in NUMBER_KEYS.items():
            for key in keys:
                if not is_number(device[part].get(key)):
                    raise ValueError(f"invalid probe output: device {position}: {part}.{key} "
                                     "is missing or not a number")
        for part, key in INTEGER_KEYS:
            if type(device[part][key]) is not int:
                raise ValueError(f"invalid probe output: device {position}: {part}.{key} "
                                 "is not a whole number")
        for part, key in STRING_KEYS:
            if not isinstance(device[part].get(key), str):
                raise ValueError(f"invalid probe output: device {position}: {part}.{key} "
                                 "is missing or not a string")
        for part, key in BOOL_KEYS:
            if not isinstance(device[part].get(key), bool):
                raise ValueError(f"invalid probe output: device {position}: {part}.{key} "
                                 "is missing or not a boolean")
        if device.get("mig_mode") not in (-1, 0, 1) or type(device.get("mig_mode")) is not int:
            raise ValueError(f"invalid probe output: device {position}: mig_mode is missing or not "
                             "-1 (unknown), 0 or 1")
        queried, synthetic = device["queried"].get("features"), device["synthetic"].get("features")
        for part, features in (("queried", queried), ("synthetic", synthetic)):
            if not isinstance(features, dict) or \
                    any(not isinstance(features.get(k), bool) for k in BOARD_FEATURES):
                raise ValueError(f"invalid probe output: device {position}: {part}.features is "
                                 f"missing or has no boolean {', '.join(BOARD_FEATURES)}")
        if any(not isinstance(v, bool) for v in synthetic.values()) or \
                any(not isinstance(queried.get(k), bool) for k in synthetic):
            raise ValueError(f"invalid probe output: device {position}: the queried features do "
                             "not cover the synthetic ones, or a feature is not a boolean")


def compare_row(q, row, mig_mode):
    """The table results of a board against its gpu_facts.json row, and the statuses by field."""
    out = []

    def add(field, expected, actual, status, note=""):
        out.append(result("table", field, expected, actual, status, note))

    add("chip", row["chip"], q["chip"], "match" if row["chip"] == q["chip"] else "mismatch",
        "the PCI-ID block table's chip against the row's")
    cc = q["major"] * 10 + q["minor"]
    add("cc", row["cc"], cc, "match" if row["cc"] == cc else "mismatch")
    # A MIG slice reports its part of the SMs (and, but for a full-memory profile, of the L2, the
    # memory and the bus); the engine prices the row at the slice's share of the SMs, so fewer SMs
    # on a MIG-capable chip with MIG enabled is the slice's, not a wrong row. With MIG disabled it
    # is a wrong row, and so it is on a chip that does not partition. Where NVML did not say (mig
    # mode -1), a slice shows by its memory being reduced too; fewer SMs with the memory whole
    # cannot be told from a wrong row, so the field is unverified.
    vram = row["vram_gib"] * GIB
    ratio = q["global_bytes"] / vram
    vram_low = vram_floor(row)
    recorded = recorded_variants(row, q)
    fewer = mig_capable(row) and 0 < q["sms"] < row["sms"]
    shrunk = 0 < ratio < vram_low
    slice_part = fewer and (mig_mode == 1 or (mig_mode == -1 and shrunk))
    part_note = "a slice's part of the board's (MIG): the engine prices the row at its share"
    if row["sms"] == q["sms"]:
        add("sms", row["sms"], q["sms"], "match")
    elif slice_part:
        add("sms", row["sms"], q["sms"], "info", "a slice of the board (MIG): the engine prices "
            "its share")
    elif fewer and mig_mode == -1:
        add("sms", row["sms"], q["sms"], "unavailable",
            "fewer SMs on a MIG-capable chip and NVML did not say whether MIG is on")
    else:
        add("sms", row["sms"], q["sms"], "mismatch")
    l2 = row.get("l2_mib")
    if l2 is None:
        add("l2_mib", None, q["l2_bytes"] / MIB, "mismatch",
            "the row has no l2_mib (the choice then sees 0)")
    elif close(l2 * MIB, q["l2_bytes"], L2_TOLERANCE):
        add("l2_mib", l2, q["l2_bytes"] / MIB, "match")
    elif slice_part and 0 < q["l2_bytes"] < l2 * MIB:
        add("l2_mib", l2, q["l2_bytes"] / MIB, "info", part_note)
    else:
        add("l2_mib", l2, q["l2_bytes"] / MIB, "mismatch")
    bus = q["memory_bus_bits"]
    if unavailable(bus):
        add("bus_bits", row["bus_bits"], bus, "unavailable")
    elif row["bus_bits"] == bus:
        add("bus_bits", row["bus_bits"], bus, "match")
    elif slice_part and bus < row["bus_bits"]:
        add("bus_bits", row["bus_bits"], bus, "info", part_note)
    elif recorded_has(recorded, "bus_bits"):
        add("bus_bits", row["bus_bits"], bus, "info",
            "the bus of a board the table records under the row's id")
    else:
        add("bus_bits", row["bus_bits"], bus, "mismatch")
    mem = q["memory_clock_khz"]
    want_khz = row_memory_khz(row)
    if unavailable(mem):
        add("memory_clock_khz", want_khz, mem, "unavailable")
    elif close(want_khz, mem, MEMORY_CLOCK_TOLERANCE):
        add("memory_clock_khz", want_khz, mem, "match")
    elif recorded_has(recorded, "gbps"):
        add("memory_clock_khz", want_khz, mem, "info",
            "the memory of a board the table records under the row's id")
    else:
        add("memory_clock_khz", want_khz, mem, "mismatch", f"gbps {row['gbps']} (half the data rate)")
    boost = q["sm_clock_khz"]
    want_boost = row["boost_mhz"] * 1000
    low = row.get("boost_min_mhz", 0) * 1000
    if unavailable(boost):
        add("boost_clock_khz", want_boost, boost, "unavailable")
    elif close(want_boost, boost, BOOST_TOLERANCE):
        add("boost_clock_khz", want_boost, boost, "match")
    elif is_laptop(row) and low * (1 - BOOST_TOLERANCE) <= boost < want_boost:
        add("boost_clock_khz", want_boost, boost, "info",
            "a laptop's power limit inside the published boost range")
    elif recorded_has(recorded, "boost_mhz"):
        add("boost_clock_khz", want_boost, boost, "info",
            "the reference clock of a board the table records under the row's id")
    elif want_boost * (1 - BOOST_UNDER) <= boost <= want_boost * (1 + BOOST_OVER):
        add("boost_clock_khz", want_boost, boost, "info",
            "a vendor's boost over or under the reference clock (a factory overclock, an OEM "
            "board): the engine prices the reference")
    else:
        add("boost_clock_khz", want_boost, boost, "mismatch",
            "the driver's peak SM clock is far from the row's")
    if vram_low <= ratio <= VRAM_HIGH:
        vram_status, vram_note = "match", ""
    elif slice_part:
        vram_status, vram_note = "info", part_note
    elif recorded_has(recorded, "vram_gib"):
        vram_status, vram_note = "info", "the memory size of a board the table records under the " \
                                         "row's id"
    else:
        vram_status, vram_note = "mismatch", "marketed capacity against the driver's total"
    add("vram_gib", row["vram_gib"], round(q["global_bytes"] / GIB, 3), vram_status, vram_note)
    return out, slice_part


def compare_engine(device, row):
    """The engine's reading of the row against the JSON, and the query's stored facts against it."""
    out = []

    def add(field, expected, actual, status, note=""):
        out.append(result("table", field, expected, actual, status, note))

    def same(field, want, got, note):
        add(field, want, got, "match" if abs(got - want) <= 1e-6 * max(1, abs(want)) else
            "mismatch", note)

    stored, alone = device["engine_table_facts"], device["engine_row_facts"]
    laptop = is_laptop(row)
    expect_ref = row.get("boost_min_mhz", 0) * 1000 or row["boost_mhz"] * 1000
    json_note = "the engine's reading of the row alone against the JSON"
    # machine_facts of the row alone: the row as it is (Verified), a laptop at the bottom of its
    # boost range (Extrapolated), no driver value read.
    same("engine_row_source", EXTRAPOLATED_SOURCE if laptop else VERIFIED_SOURCE, alone["source"],
         "1 the row as it is, 2 a laptop")
    add("engine_row_sku", row["name"], alone["sku"], "match" if alone["sku"] == row["name"] else
        "mismatch", json_note)
    same("engine_row_reference_clock_khz", expect_ref, alone["reference_clock_khz"], json_note)
    same("engine_row_memory_bus_bits", row["bus_bits"], alone["memory_bus_bits"], json_note)
    same("engine_row_memory_clock_khz", math.floor(row_memory_khz(row) + 0.5),
         alone["memory_clock_khz"], json_note)
    if "l2_mib" in row:
        same("engine_row_l2_mib", row["l2_mib"], alone["l2_mib"], json_note)
    want_slow = row.get("mma_slow", False)
    add("engine_row_mma_slow", want_slow, alone["mma_slow"],
        "match" if alone["mma_slow"] == want_slow else "mismatch", json_note)
    q = device["queried"]
    # The engine takes a row only for its architecture: an id the driver reports with another
    # compute capability is unknown to it, and it reads the driver alone (machine_facts.cpp).
    if q["major"] != 0 and q["major"] * 10 + q["minor"] != row["cc"]:
        add("engine_sku", "", stored["sku"], "match" if stored["sku"] == "" else "mismatch",
            "the compute capability is not the row's: the engine does not use the row (the cc "
            "field fails)")
        add("engine_driver_read", True, stored["driver_read"],
            "match" if stored["driver_read"] is True else "mismatch",
            "an id the engine does not know is read from the driver alone")
        return out
    # What the query stored: the row alone where the driver agrees with the row; where a variant,
    # a slice or a laptop made the engine read the driver, only the bounds of that reading.
    stored_note = "the facts the query stored against the engine's reading of the row"
    add("engine_sku", row["name"], stored["sku"], "match" if stored["sku"] == row["name"] else
        "mismatch", "the table row the engine attached to this id" if stored["sku"] else
        "the engine's table has no row for this id")
    if "l2_mib" in row:
        same("engine_l2_mib", row["l2_mib"], stored["l2_mib"], stored_note)
    add("engine_mma_slow", want_slow, stored["mma_slow"],
        "match" if stored["mma_slow"] == want_slow else "mismatch",
        "the engine's mma_slow (a board fact the driver cannot answer) against the row")
    # The engine's own rule, from the driver's values (machine_facts.cpp, the row branch); no
    # audit status enters it, so a board the audit fails still gets the engine's expected facts.
    sms, bus, mem = q["sms"], q["memory_bus_bits"], q["memory_clock_khz"]
    other_sms = sms > 0 and sms != row["sms"]  # source Extrapolated; fewer SMs: a share
    whole_board = not laptop and not other_sms
    row_khz = math.floor(row_memory_khz(row) + 0.5)  # the engine's llround of the row's clock
    read_bus = whole_board and bus >= ENGINE_BUS_CHANNEL_BITS and \
        bus % ENGINE_BUS_CHANNEL_BITS == 0 and bus != row["bus_bits"]
    read_clock = whole_board and mem >= ENGINE_MIN_MEMORY_CLOCK_KHZ and \
        mem * 1000 < row_khz * ENGINE_SLOWER_MEMORY_PER_MILLE
    # driver_read: a laptop row without a boost range (the driver's clock is read), a bus read, a
    # clock read. A slice keeps the row's values and prices its share apart: nothing is read.
    row_reads = laptop and not row.get("boost_min_mhz", 0) > 0
    add("engine_row_driver_read", row_reads, alone["driver_read"],
        "match" if alone["driver_read"] == row_reads else "mismatch", json_note)
    want_read = row_reads or read_bus or read_clock
    add("engine_driver_read", want_read, stored["driver_read"],
        "match" if stored["driver_read"] == want_read else "mismatch",
        "set when the engine read the driver's values into the facts")
    bound = "the query read the driver: within the row's bounds"
    if laptop:
        # The bottom of the range where the row has it, else at most the row's top.
        ok = stored["reference_clock_khz"] == expect_ref if row.get("boost_min_mhz", 0) > 0 \
            else 0 < stored["reference_clock_khz"] <= expect_ref
        add("engine_reference_clock_khz", expect_ref, stored["reference_clock_khz"],
            "match" if ok else "mismatch", bound)
    else:
        same("engine_reference_clock_khz", alone["reference_clock_khz"],
             stored["reference_clock_khz"], stored_note)
    # The memory values are the driver's only where the engine read them; a laptop and a slice
    # keep the row's (a slice's share is priced apart, a laptop from its published range).
    if read_bus:
        same("engine_memory_bus_bits", bus, stored["memory_bus_bits"],
             "the bus width the driver reported, read into the facts")
    else:
        same("engine_memory_bus_bits", alone["memory_bus_bits"], stored["memory_bus_bits"],
             stored_note)
    if read_clock:
        # The engine writes the driver's clock on the 2^(1/16) grid; only its bounds are checked.
        add("engine_memory_clock_khz", alone["memory_clock_khz"], stored["memory_clock_khz"],
            "match" if 0 < stored["memory_clock_khz"] < alone["memory_clock_khz"] else "mismatch",
            "the driver's clock on the grid: under the row's, never unknown")
    else:
        same("engine_memory_clock_khz", alone["memory_clock_khz"], stored["memory_clock_khz"],
             stored_note)
    # The source is Extrapolated for a laptop, a differing SM count (a slice), a bus read or a
    # slower memory read; a faster memory (an overclock) leaves the row as it is.
    want_source = EXTRAPOLATED_SOURCE if laptop or other_sms or read_bus or read_clock \
        else VERIFIED_SOURCE
    same("engine_source", want_source, stored["source"],
         "1 the row as it is, 2 a laptop, a slice, or a bus or slower memory read from the driver")
    return out


def compare_device(device, row):
    """Results for one device of the probe against its gpu_facts.json row (None: no such row)."""
    if "error" in device:
        return [result("probe", "device", "read by the probe", device["error"], "mismatch",
                       "the probe could not read this device")]
    q = device["queried"]
    out = []
    if row is None:
        out.append(result("table", "row", f"0x{q['pci_device_id']:04X} in gpu_facts.json", None,
                          "mismatch", "no row for this PCI product id: the engine runs it on the "
                          "driver's values alone"))
    else:
        table, _ = compare_row(q, row, device["mig_mode"])
        out += table
        out += compare_engine(device, row)

    s = device["synthetic"]
    for field in ("smem_per_block_optin", "smem_per_sm", "max_threads_per_sm", "max_blocks_per_sm",
                  "registers_per_sm"):
        want, got = s[field], q[field]
        out.append(result("arch", field, want, got, "match" if want == got else "mismatch",
                          "synthetic_device_class against the queried class"))
    raw = device["raw"]
    reserved = raw["reserved_shared_per_block"]
    if reserved == -1:
        out.append(result("arch", "reserved_shared_per_block", None, reserved, "unavailable"))
    else:
        total = q["smem_per_block_optin"] + reserved
        out.append(result("arch", "smem_optin_plus_reserved_is_per_sm", q["smem_per_sm"], total,
                          "match" if total == q["smem_per_sm"] else "mismatch",
                          f"opt-in per block {q['smem_per_block_optin']} + reserved {reserved}"))
        out.append(result("arch", "synthetic_smem_optin_plus_reserved_is_per_sm", s["smem_per_sm"],
                          s["smem_per_block_optin"] + reserved,
                          "match" if s["smem_per_block_optin"] + reserved == s["smem_per_sm"]
                          else "mismatch", "the synthetic class's own pair, with the driver's reserved"))
    cap, l2 = q["persisting_l2_max_bytes"], q["l2_bytes"]
    if unavailable(cap):
        out.append(result("arch", "persisting_l2_max_bytes", s["persisting_l2_max_bytes"], cap,
                          "unavailable"))
    else:
        status = "match" if cap == s["persisting_l2_max_bytes"] else \
            "info" if PERSISTING_LOW * l2 <= cap <= l2 else "mismatch"
        out.append(result("arch", "persisting_l2_max_bytes", s["persisting_l2_max_bytes"], cap,
                          status, "synthetic writes 3/4 of the L2; the driver's cap is the "
                          "board's, between 1/8 of the L2 and all of it"))
    # The per-block limits the occupancy calculator assumes, against what the driver says.
    for name in OCCUPANCY_LIMITS:
        want, got = device["engine_occupancy_limits"][name], raw[name]
        if got == -1 or (got == 0 and name not in ZERO_IS_REAL):
            out.append(result("arch", f"occupancy_{name}", want, got, "unavailable"))
        else:
            out.append(result("arch", f"occupancy_{name}", want, got,
                              "match" if want == got else "mismatch",
                              "the occupancy calculator's assumed limit against the driver's"))
    for name, want in s["features"].items():
        got = q["features"].get(name)
        if name in BOARD_FEATURES:
            # A board fact the driver cannot answer, set on both classes from the same table:
            # agreement verifies nothing about the board (engine_mma_slow checks the row); a
            # difference means the query did not store the facts or the two paths resolved a
            # different chip.
            out.append(result("arch", f"feature_{name}", want, got,
                              "derived" if want == got else "mismatch",
                              "queried and synthetic read the same table: not a driver fact"))
        else:
            out.append(result("arch", f"feature_{name}", want, got,
                              "match" if want == got else "info",
                              "queried features are the device's row in common with the build's"))
    return out


def summarize(results_by_device):
    """The JSON summary: per device the counts and the failing fields, and the unverified fields
    (the driver gave no value, so the row was not checked against it)."""
    devices = []
    unverified = []
    failed = False
    for index, name, results in results_by_device:
        counts = {}
        for r in results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if r["status"] == "unavailable":
                unverified.append({"index": index, "field": r["field"]})
        bad = [r for r in results if r["status"] == "mismatch"]
        failed = failed or bool(bad)
        devices.append({"index": index, "name": name, "counts": counts, "mismatches": bad,
                        "results": results})
    return {"ok": not failed, "unverified": unverified, "devices": devices}


def table_text(index, name, results):
    lines = [f"device {index}: {name}"]
    width = max(len(r["field"]) for r in results) if results else 0
    for r in results:
        note = f"  ({r['note']})" if r["note"] and r["status"] != "match" else ""
        lines.append(f"  {r['status']:<11} {r['source']:<5} {r['field']:<{width}} "
                     f"expected {r['expected']!s:<12} actual {r['actual']!s}{note}")
    return "\n".join(lines)
