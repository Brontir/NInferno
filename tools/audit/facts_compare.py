"""The comparison of a board's facts with what the driver reports (#612 step 2), no GPU.

Input: one device of ninferno-facts-probe's JSON ("queried": the engine's own query,
"synthetic": synthetic_device_class for the same compute capability, "raw": CUDA attributes the
query does not keep, "engine_table_facts": MachineFacts as the engine reads the table, "engine_occupancy_limits": the
per-block limits resident_ctas_per_sm assumes) and the
board's row of src/core/data/gpu_facts.json. Output: one result per field.

  table  the gpu_facts.json row against the driver: cc, SMs, L2, bus width, memory data rate,
         boost clock, memory size; and against the engine's own reading of the table;
  arch   the architecture limits synthetic_device_class writes for the compute capability
         against the queried ones (shared memory per block and per SM, threads, blocks and
         registers per SM) and the reserved-shared invariant of the queried class; the board
         feature mma_slow (a difference fails); the per-block limits the occupancy calculator
         assumes (kernel_occupancy.h, "engine_occupancy_limits": shared memory per block, registers
         per block, threads per block, warp size) against the driver's raw attributes.

status: match, mismatch (fails the run), unavailable (the driver gave no value: not a failure),
info (a difference that is not a failure: a feature the build lacks, the persisting-L2 cap, a
board the row records under its id in `variants`).
"""
import math

MIB = 1024 * 1024
GIB = 1024 * MIB

# Relative tolerances. The memory clock is the issue's +-0.5 %. The boost clock is the VBIOS's
# peak SM clock: a stock board reads the published reference within 1.5 %, a partner board's
# factory boost 2 to 6 % above it and an OEM board's a little under it (information within
# 10 %, not a failure), a laptop anywhere in its published range; farther off is a wrong row.
# The L2 and memory size are what a driver
# reports against a marketed figure; GDDR ECC on (an A6000, an L40, an A10) keeps 1/16 of the
# memory, so the driver's total is down to about 0.928 of the marketed size.
L2_TOLERANCE = 0.01
MEMORY_CLOCK_TOLERANCE = 0.005
BOOST_TOLERANCE = 0.015
BOOST_VENDOR = 0.10
VRAM_LOW, VRAM_HIGH = 0.92, 1.003
VERIFIED_SOURCE = 1  # FactSource::Verified: the engine reads the row as it is
EXTRAPOLATED_SOURCE = 2  # FactSource::Extrapolated: a laptop, a memory variant or a slice


def result(source, field, expected, actual, status, note=""):
    return {"source": source, "field": field, "expected": expected, "actual": actual,
            "status": status, "note": note}


def unavailable(actual):
    return actual is None or actual in (0, -1)


def close(expected, actual, tolerance):
    return expected > 0 and abs(actual - expected) <= tolerance * expected


def snap_to_fact_grid(x):
    """machine_facts.cpp's grid: the nearest 2^(k/16)."""
    return 2.0 ** (math.floor(16.0 * math.log2(x) + 0.5) / 16.0) if x > 0 else 0.0


# The chips NVIDIA partitions with MIG: GA100 (A100, A30, A800), GH100 (H100, H200, H800), GB100
# (B200), and GB202 on its professional boards (RTX PRO 6000 Blackwell, workstation and server);
# not GV100 nor any GeForce board.
MIG_CHIPS = {"ga100", "gh100", "gb100"}
MIG_PRO_CHIPS = {"gb202"}


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
# the GDDR5 board's). A driver value that is such a board's is information; any other value off
# the row (an overclock, a wrong row) stays a mismatch.
VARIANT_KEYS = ("vram_gib", "gbps", "boost_mhz")


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
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{where}: {key} is not a positive number")
        if all(v[key] == row[key] for key in v):
            raise ValueError(f"{where}: a variant that does not differ from the row")


def variant_has(row, key, is_the_drivers):
    """True when a board recorded under the row's id has `key` at the driver's value."""
    return any(key in v and is_the_drivers(v[key]) for v in row.get("variants", []))


def engine_expected(q, row):
    """MachineFacts as machine_facts (src/core/machine_facts.cpp) builds it from the row and
    this driver's report: source, reference clock, bus width, memory clock (kHz)."""
    out = {"source": VERIFIED_SOURCE, "reference_clock_khz": row["boost_mhz"] * 1000,
           "memory_bus_bits": row["bus_bits"], "memory_clock_khz": round(row_memory_khz(row))}
    laptop = is_laptop(row)
    if laptop:
        out["source"] = EXTRAPOLATED_SOURCE
        if row.get("boost_min_mhz", 0) > 0:
            out["reference_clock_khz"] = row["boost_min_mhz"] * 1000
        elif q["sm_clock_khz"] > 0:
            out["reference_clock_khz"] = min(out["reference_clock_khz"],
                                             round(snap_to_fact_grid(q["sm_clock_khz"])))
    if q["sms"] > 0 and q["sms"] != row["sms"]:
        out["source"] = EXTRAPOLATED_SOURCE
    whole_board = not laptop and (q["sms"] <= 0 or q["sms"] == row["sms"])
    bus = q["memory_bus_bits"]
    if whole_board and bus >= 32 and bus % 32 == 0 and bus != row["bus_bits"]:
        out["source"] = EXTRAPOLATED_SOURCE
        out["memory_bus_bits"] = bus
    mem = q["memory_clock_khz"]
    if whole_board and mem >= 100000 and mem * 1000 < out["memory_clock_khz"] * 978:
        out["source"] = EXTRAPOLATED_SOURCE
        out["memory_clock_khz"] = round(snap_to_fact_grid(mem))
    return out


DEVICE_PARTS = ("queried", "synthetic", "engine_table_facts", "engine_occupancy_limits")
# The raw attributes kernel_occupancy.h assumes a value for (engine_occupancy_limits, same names).
OCCUPANCY_LIMITS = ("shared_per_block_default", "registers_per_block", "max_threads_per_block",
                    "warp_size")
BOARD_FEATURES = ("mma_slow",)


def validate_probe(probe_output):
    """Raises ValueError unless the probe document is a complete recording: a nonempty list of
    devices, each with the driver's query, the synthetic class and the engine's table reading.
    A partial capture must not pass as a clean audit."""
    devices = probe_output.get("devices") if isinstance(probe_output, dict) else None
    if not isinstance(devices, list) or not devices:
        raise ValueError("invalid probe output: \"devices\" is missing, not a list or empty")
    for position, device in enumerate(devices):
        if not isinstance(device, dict) or "index" not in device:
            raise ValueError(f"invalid probe output: device {position} is not an object with an index")
        for part in DEVICE_PARTS:
            if not isinstance(device.get(part), dict):
                raise ValueError(f"invalid probe output: device {position} has no \"{part}\" object")
        # The probe always prints raw and the key (-1 when CUDA cannot answer).
        if not isinstance(device.get("raw"), dict) or \
                "reserved_shared_per_block" not in device["raw"]:
            raise ValueError(f"invalid probe output: device {position} has no \"raw\" object "
                             "with reserved_shared_per_block")
        missing = [k for k in OCCUPANCY_LIMITS if k not in device["engine_occupancy_limits"]]
        if missing:
            raise ValueError(f"invalid probe output: device {position}: engine_occupancy_limits "
                             f"has no {', '.join(missing)}")
        if not isinstance(device["synthetic"].get("features"), dict) or \
                not isinstance(device["queried"].get("features"), dict):
            raise ValueError(f"invalid probe output: device {position} has no \"features\" object")
        for part in ("queried", "synthetic"):
            if "persisting_l2_max_bytes" not in device[part]:
                raise ValueError(f"invalid probe output: device {position}: {part} has no "
                                 "persisting_l2_max_bytes")
        if "sku" not in device["engine_table_facts"]:
            raise ValueError(f"invalid probe output: device {position}: engine_table_facts has no sku")


def compare_device(device, row):
    """Results for one device of the probe against its gpu_facts.json row (None: no such row)."""
    q = device["queried"]
    out = []
    if row is None:
        out.append(result("table", "row", f"0x{q['pci_device_id']:04X} in gpu_facts.json", None,
                          "mismatch", "no row for this PCI product id: the engine runs it on the "
                          "driver's values alone"))
    else:
        cc = q["major"] * 10 + q["minor"]
        out.append(result("table", "cc", row["cc"], cc, "match" if row["cc"] == cc else "mismatch"))
        # A MIG slice reports its part of the SMs, the L2, the memory and the bus; the engine
        # prices the board's row at the slice's share of the SMs, so a smaller value is the
        # slice's, not a wrong row. A board without MIG reporting fewer SMs than its row is a
        # wrong row.
        slice_part = mig_capable(row) and 0 < q["sms"] < row["sms"]
        if row["sms"] == q["sms"]:
            out.append(result("table", "sms", row["sms"], q["sms"], "match"))
        elif slice_part:
            out.append(result("table", "sms", row["sms"], q["sms"], "info",
                              "a slice of the board (MIG): the engine prices its share"))
        else:
            out.append(result("table", "sms", row["sms"], q["sms"], "mismatch"))
        expected = engine_expected(q, row)
        part_note = "a slice's part of the board's (MIG): the engine prices the row at its share"
        l2 = row.get("l2_mib")
        if l2 is None:
            out.append(result("table", "l2_mib", None, q["l2_bytes"] / MIB, "mismatch",
                              "the row has no l2_mib (the choice then sees 0)"))
        elif close(l2 * MIB, q["l2_bytes"], L2_TOLERANCE):
            out.append(result("table", "l2_mib", l2, q["l2_bytes"] / MIB, "match"))
        elif slice_part and 0 < q["l2_bytes"] < l2 * MIB:
            out.append(result("table", "l2_mib", l2, q["l2_bytes"] / MIB, "info", part_note))
        else:
            out.append(result("table", "l2_mib", l2, q["l2_bytes"] / MIB, "mismatch"))
        bus = q["memory_bus_bits"]
        if unavailable(bus):
            out.append(result("table", "bus_bits", row["bus_bits"], bus, "unavailable"))
        elif row["bus_bits"] == bus:
            out.append(result("table", "bus_bits", row["bus_bits"], bus, "match"))
        elif slice_part and bus < row["bus_bits"]:
            out.append(result("table", "bus_bits", row["bus_bits"], bus, "info", part_note))
        elif expected["memory_bus_bits"] == bus:
            out.append(result("table", "bus_bits", row["bus_bits"], bus, "info",
                              "a memory variant under the row's id: the engine reads the driver's bus"))
        else:
            out.append(result("table", "bus_bits", row["bus_bits"], bus, "mismatch"))
        mem = q["memory_clock_khz"]
        want_khz = row_memory_khz(row)
        if unavailable(mem):
            out.append(result("table", "memory_clock_khz", want_khz, mem, "unavailable"))
        elif close(want_khz, mem, MEMORY_CLOCK_TOLERANCE):
            out.append(result("table", "memory_clock_khz", want_khz, mem, "match"))
        elif expected["memory_clock_khz"] != round(want_khz):
            out.append(result("table", "memory_clock_khz", want_khz, mem, "info",
                              "a slower memory variant under the row's id: the engine reads the "
                              "driver's clock"))
        elif variant_has(row, "gbps",
                         lambda gbps: close(memory_khz(gbps), mem, MEMORY_CLOCK_TOLERANCE)):
            out.append(result("table", "memory_clock_khz", want_khz, mem, "info",
                              "the memory of a board the table records under the row's id: the "
                              "engine prices the row's rate, never a faster one"))
        else:
            out.append(result("table", "memory_clock_khz", want_khz, mem, "mismatch",
                              f"gbps {row['gbps']} (half the data rate)"))
        boost = q["sm_clock_khz"]
        want_boost = row["boost_mhz"] * 1000
        low = row.get("boost_min_mhz", 0) * 1000
        if unavailable(boost):
            out.append(result("table", "boost_clock_khz", want_boost, boost, "unavailable"))
        elif close(want_boost, boost, BOOST_TOLERANCE):
            out.append(result("table", "boost_clock_khz", want_boost, boost, "match"))
        elif is_laptop(row) and low * (1 - BOOST_TOLERANCE) <= boost < want_boost:
            out.append(result("table", "boost_clock_khz", want_boost, boost, "info",
                              "a laptop's power limit inside the published boost range"))
        elif variant_has(row, "boost_mhz",
                         lambda mhz: close(mhz * 1000, boost, BOOST_TOLERANCE)):
            out.append(result("table", "boost_clock_khz", want_boost, boost, "info",
                              "the reference clock of a board the table records under the row's "
                              "id: the engine prices the row's"))
        elif close(want_boost, boost, BOOST_VENDOR):
            out.append(result("table", "boost_clock_khz", want_boost, boost, "info",
                              "a vendor's boost over or under the reference clock (a factory "
                              "overclock, an OEM board): the engine prices the reference"))
        else:
            out.append(result("table", "boost_clock_khz", want_boost, boost, "mismatch",
                              "the driver's peak SM clock is far from the row's"))
        vram = row["vram_gib"] * GIB
        ratio = q["global_bytes"] / vram if vram else 0
        if VRAM_LOW <= ratio <= VRAM_HIGH:
            vram_status, vram_note = "match", ""
        elif slice_part and 0 < ratio < VRAM_LOW:
            vram_status, vram_note = "info", part_note
        elif variant_has(row, "vram_gib",
                         lambda gib: VRAM_LOW <= q["global_bytes"] / (gib * GIB) <= VRAM_HIGH):
            vram_status, vram_note = "info", ("the memory size of a board the table records "
                                              "under the row's id")
        else:
            vram_status, vram_note = "mismatch", "marketed capacity against the driver's total"
        out.append(result("table", "vram_gib", row["vram_gib"], round(q["global_bytes"] / GIB, 3),
                          vram_status, vram_note))
        engine = device["engine_table_facts"]  # validate_probe: the probe always prints it
        if engine.get("sku"):
            # What machine_facts makes of the row and this driver: the row as it is, or a
            # laptop at the bottom of its boost range, a memory variant at the driver's bus and
            # clock, a slice (source Extrapolated, its share priced apart).
            out.append(result("table", "engine_sku", row["name"], engine["sku"],
                              "match" if engine["sku"] == row["name"] else "mismatch",
                              "the table row the engine attached to this id"))
            out.append(result("table", "engine_source", expected["source"], engine["source"],
                              "match" if engine["source"] == expected["source"] else "mismatch",
                              "1 the row as it is, 2 a laptop, a memory variant or a slice"))
            for field in ("reference_clock_khz", "memory_bus_bits", "memory_clock_khz"):
                want, got = expected[field], engine[field]
                out.append(result("table", f"engine_{field}", want, got,
                                  "match" if abs(got - want) <= 1 else "mismatch",
                                  "the engine's reading of the table against the JSON"))
            if l2 is not None:
                out.append(result("table", "engine_l2_mib", l2, engine["l2_mib"],
                                  "match" if close(l2, engine["l2_mib"], 1e-6) else "mismatch",
                                  "the engine's reading of the table against the JSON"))
        else:
            out.append(result("table", "engine_sku", row["name"], "", "mismatch",
                              "the engine's table has no row for this id"))

    s = device["synthetic"]
    for field in ("smem_per_block_optin", "smem_per_sm", "max_threads_per_sm", "max_blocks_per_sm",
                  "registers_per_sm"):
        want, got = s[field], q[field]
        out.append(result("arch", field, want, got, "match" if want == got else "mismatch",
                          "synthetic_device_class against the queried class"))
    raw = device["raw"]
    reserved = raw.get("reserved_shared_per_block")
    if unavailable(reserved) and reserved != 0:
        out.append(result("arch", "reserved_shared_per_block", None, reserved, "unavailable"))
    else:
        reserved = reserved or 0
        total = q["smem_per_block_optin"] + reserved
        out.append(result("arch", "smem_optin_plus_reserved_is_per_sm", q["smem_per_sm"], total,
                          "match" if total == q["smem_per_sm"] else "mismatch",
                          f"opt-in per block {q['smem_per_block_optin']} + reserved {reserved}"))
        out.append(result("arch", "synthetic_smem_optin_plus_reserved_is_per_sm", s["smem_per_sm"],
                          s["smem_per_block_optin"] + reserved,
                          "match" if s["smem_per_block_optin"] + reserved == s["smem_per_sm"]
                          else "mismatch", "the synthetic class's own pair, with the driver's reserved"))
    persisting = q["persisting_l2_max_bytes"]
    out.append(result("arch", "persisting_l2_max_bytes", s["persisting_l2_max_bytes"], persisting,
                      "match" if s["persisting_l2_max_bytes"] == persisting else "info",
                      "synthetic writes 3/4 of the L2; the driver's cap is the board's"))
    # The per-block limits resident_ctas_per_sm assumes, against what the driver says.
    for name in OCCUPANCY_LIMITS:
        want, got = device["engine_occupancy_limits"][name], raw.get(name)
        if unavailable(got):
            out.append(result("arch", f"occupancy_{name}", want, got, "unavailable"))
        else:
            out.append(result("arch", f"occupancy_{name}", want, got,
                              "match" if want == got else "mismatch",
                              "the occupancy calculator's assumed limit against the driver's"))
    for name, want in s["features"].items():
        got = q["features"].get(name)
        # mma_slow is a board fact set on both classes (it turns tensor routes off), so a
        # difference is a failure; the other features are what the build can launch.
        differs = "mismatch" if name in BOARD_FEATURES else "info"
        out.append(result("arch", f"feature_{name}", want, got, "match" if want == got else differs,
                          "queried features are the device's row in common with the build's"))
    return out


def summarize(results_by_device):
    """The JSON summary: per device the counts and the failing fields."""
    devices = []
    failed = False
    for index, name, results in results_by_device:
        counts = {}
        for r in results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        bad = [r for r in results if r["status"] == "mismatch"]
        failed = failed or bool(bad)
        devices.append({"index": index, "name": name, "counts": counts, "mismatches": bad,
                        "results": results})
    return {"ok": not failed, "devices": devices}


def table_text(index, name, results):
    lines = [f"device {index}: {name}"]
    width = max(len(r["field"]) for r in results) if results else 0
    for r in results:
        note = f"  ({r['note']})" if r["note"] and r["status"] != "match" else ""
        lines.append(f"  {r['status']:<11} {r['source']:<5} {r['field']:<{width}} "
                     f"expected {r['expected']!s:<12} actual {r['actual']!s}{note}")
    return "\n".join(lines)
