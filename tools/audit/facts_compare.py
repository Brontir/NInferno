"""The comparison of a board's facts with what the driver reports (#612 step 2), no GPU.

Input: one device of ninferno-facts-probe's JSON ("queried": the engine's own query,
"synthetic": synthetic_device_class for the same compute capability, "raw": CUDA attributes the
query does not keep, "engine_table_facts": MachineFacts as the engine reads the table) and the
board's row of src/core/data/gpu_facts.json. Output: one result per field.

  table  the gpu_facts.json row against the driver: cc, SMs, L2, bus width, memory data rate,
         boost clock, memory size; and against the engine's own reading of the table;
  arch   the architecture limits synthetic_device_class writes for the compute capability
         against the queried ones (shared memory per block and per SM, threads, blocks and
         registers per SM) and the reserved-shared invariant of the queried class.

status: match, mismatch (fails the run), unavailable (the driver gave no value: not a failure),
info (a difference that is not a failure: a feature the build lacks, the persisting-L2 cap).
"""
MIB = 1024 * 1024
GIB = 1024 * MIB

# Relative tolerances. The memory clock is the issue's +-0.5 %. The boost clock is the VBIOS's
# peak SM clock: a stock board reads the published reference within 1.5 %, a partner board's
# factory boost reads 2 to 6 % above it (information, not a failure), and a clock under the
# reference or more than 10 % over it is a wrong row. The L2 and memory size are what a driver
# reports against a marketed figure; GDDR ECC on (an A6000, an L40, an A10) keeps 1/16 of the
# memory, so the driver's total is down to about 0.928 of the marketed size.
L2_TOLERANCE = 0.01
MEMORY_CLOCK_TOLERANCE = 0.005
BOOST_TOLERANCE = 0.015
BOOST_FACTORY_HIGH = 0.10
VRAM_LOW, VRAM_HIGH = 0.92, 1.003
VERIFIED_SOURCE = 1  # FactSource::Verified: the engine reads the row as it is


def result(source, field, expected, actual, status, note=""):
    return {"source": source, "field": field, "expected": expected, "actual": actual,
            "status": status, "note": note}


def unavailable(actual):
    return actual is None or actual in (0, -1)


def close(expected, actual, tolerance):
    return expected > 0 and abs(actual - expected) <= tolerance * expected


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
        out.append(result("table", "sms", row["sms"], q["sms"],
                          "match" if row["sms"] == q["sms"] else "mismatch"))
        l2 = row.get("l2_mib")
        if l2 is None:
            out.append(result("table", "l2_mib", None, q["l2_bytes"] / MIB, "mismatch",
                              "the row has no l2_mib (the choice then sees 0)"))
        else:
            out.append(result("table", "l2_mib", l2, q["l2_bytes"] / MIB,
                              "match" if close(l2 * MIB, q["l2_bytes"], L2_TOLERANCE) else "mismatch"))
        bus = q["memory_bus_bits"]
        if unavailable(bus):
            out.append(result("table", "bus_bits", row["bus_bits"], bus, "unavailable"))
        else:
            out.append(result("table", "bus_bits", row["bus_bits"], bus,
                              "match" if row["bus_bits"] == bus else "mismatch"))
        mem = q["memory_clock_khz"]
        # The driver spells half the data rate in kHz: 28 Gbps is 14 000 000.
        want_khz = row["gbps"] * 1.0e6 / 2.0
        if unavailable(mem):
            out.append(result("table", "memory_clock_khz", want_khz, mem, "unavailable"))
        else:
            out.append(result("table", "memory_clock_khz", want_khz, mem,
                              "match" if close(want_khz, mem, MEMORY_CLOCK_TOLERANCE) else "mismatch",
                              f"gbps {row['gbps']} (half the data rate)"))
        boost = q["sm_clock_khz"]
        want_boost = row["boost_mhz"] * 1000
        if unavailable(boost):
            out.append(result("table", "boost_clock_khz", want_boost, boost, "unavailable"))
        elif close(want_boost, boost, BOOST_TOLERANCE):
            out.append(result("table", "boost_clock_khz", want_boost, boost, "match"))
        elif want_boost < boost <= want_boost * (1 + BOOST_FACTORY_HIGH):
            out.append(result("table", "boost_clock_khz", want_boost, boost, "info",
                              "a factory boost above the reference clock"))
        else:
            out.append(result("table", "boost_clock_khz", want_boost, boost, "mismatch",
                              "the driver's peak SM clock is under the reference or far over it"))
        vram = row["vram_gib"] * GIB
        ratio = q["global_bytes"] / vram if vram else 0
        out.append(result("table", "vram_gib", row["vram_gib"], round(q["global_bytes"] / GIB, 3),
                          "match" if VRAM_LOW <= ratio <= VRAM_HIGH else "mismatch",
                          "marketed capacity against the driver's total"))
        engine = device.get("engine_table_facts")
        if engine is not None and engine.get("sku") and engine.get("source") != VERIFIED_SOURCE:
            # A laptop is priced at the bottom of its boost range, a variant on another memory
            # at the driver's bus and clock, a slice at its share: the engine departs from the
            # row on purpose, so its reading is not compared with the JSON.
            out.append(result("table", "engine_source", VERIFIED_SOURCE, engine.get("source"),
                              "info", "the engine prices this device off the row (a laptop, a "
                              "memory variant or a slice), not the row as it is"))
        elif engine is not None and engine.get("sku"):
            for field, want in (("reference_clock_khz", want_boost), ("memory_bus_bits", row["bus_bits"]),
                                ("memory_clock_khz", want_khz)):
                got = engine[field]
                out.append(result("table", f"engine_{field}", want, got,
                                  "match" if close(want, got, 1e-6) else "mismatch",
                                  "the engine's reading of the table against the JSON"))
            if l2 is not None:
                out.append(result("table", "engine_l2_mib", l2, engine["l2_mib"],
                                  "match" if close(l2, engine["l2_mib"], 1e-6) else "mismatch",
                                  "the engine's reading of the table against the JSON"))
        elif engine is not None:
            out.append(result("table", "engine_sku", row["name"], "", "mismatch",
                              "the engine's table has no row for this id"))

    s = device["synthetic"]
    for field in ("smem_per_block_optin", "smem_per_sm", "max_threads_per_sm", "max_blocks_per_sm",
                  "registers_per_sm"):
        want, got = s[field], q[field]
        out.append(result("arch", field, want, got, "match" if want == got else "mismatch",
                          "synthetic_device_class against the queried class"))
    raw = device.get("raw", {})
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
    persisting = q.get("persisting_l2_max_bytes", 0)
    out.append(result("arch", "persisting_l2_max_bytes", s["persisting_l2_max_bytes"], persisting,
                      "match" if s["persisting_l2_max_bytes"] == persisting else "info",
                      "synthetic writes 3/4 of the L2; the driver's cap is the board's"))
    for name, want in s["features"].items():
        got = q["features"].get(name)
        out.append(result("arch", f"feature_{name}", want, got, "match" if want == got else "info",
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
