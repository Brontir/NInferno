#!/usr/bin/env python3
"""Compare published NVIDIA PCI identities with gpu_facts.json, without a GPU.

Inputs are saved pci.ids and NVIDIA driver README/supportedchips.html files. No
network request, specification inference or runtime registry update is made.
A product ID present in the registry is identity coverage, not proof that every
subsystem variant has the right facts or that the board has been tested.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re


FAMILIES = {"GV": "Volta", "TU": "Turing", "GA": "Ampere", "AD": "Ada",
            "GH": "Hopper", "GB": "Blackwell"}
NON_GPU = re.compile(r"Audio|USB|UCSI|NVSwitch|PCIe? (?:Bridge|host)|Root Port|"
                     r"Ethernet|HDMI|PCI Express", re.I)


def pci_devices(text):
    """Keep NVIDIA device and subsystem names; never collapse a subsystem ID."""
    result = {}
    vendor = device = None
    for line in text.splitlines():
        match = re.match(r"^([0-9a-fA-F]{4})  (.*)", line)
        if match:
            vendor, device = match[1].lower(), None
        if vendor != "10de":
            continue
        match = re.match(r"^\t([0-9a-fA-F]{4})  (.*)", line)
        if match:
            device = match[1].upper()
            result[device] = {"name": match[2], "subsystems": {}}
        match = re.match(r"^\t\t([0-9a-fA-F]{4}) ([0-9a-fA-F]{4})  (.*)", line)
        if match and device is not None:
            key = f"{match[1].upper()} {match[2].upper()}"
            result[device]["subsystems"][key] = match[3]
    return result


class SupportedChips(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.cells = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.cells = []
        elif tag == "td":
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.cells.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and len(self.cells) >= 2:
            if re.fullmatch(r"[0-9A-Fa-f]{4}(?: [0-9A-Fa-f]{4} [0-9A-Fa-f]{4})?",
                            self.cells[1]):
                self.rows.append((self.cells[1].upper(), self.cells[0]))


def driver_identities(documents):
    result = defaultdict(dict)
    for label, text in documents:
        parser = SupportedChips()
        parser.feed(text)
        if not parser.rows:
            raise ValueError(f"{label}: no supported-chip rows (wrong page or changed format)")
        for identity, name in parser.rows:
            entry = result[identity[:4]].setdefault(identity, {"names": [], "sources": []})
            if name not in entry["names"]:
                entry["names"].append(name)
            if label not in entry["sources"]:
                entry["sources"].append(label)
    return result


def classify(name, identities):
    """Triage labels describe evidence, not CUDA support or release status."""
    if re.search(r"Engineering Sample|Reserved|Initial Dev|INT SKU|Generic-GPU|\bES-A", name, re.I):
        return "engineering-or-reserved"
    if re.search(r"Jetson|Tegra|DRIVE|GB10\b|GB20B|GH200", name, re.I):
        return "platform-review"
    if "GRID" in name:
        return "virtual-or-shared-identity"
    if "CMP" in name:
        return "mining-board"
    if re.search(r"Mobile|Laptop|Embedded|Max-Q|\b(?:GV|TU|GA|AD|GH|GB)\d+(?:GL|B)?M\b",
                 name, re.I):
        return "oem-clock-range-review"
    if identities:
        return "driver-listed"
    return "pci-list-only"


def catalog(pci, drivers, registry):
    registered = {row["id"].removeprefix("0x").upper(): row for row in registry["gpus"]}
    rows, excluded, unclassified = [], [], []
    for device in sorted(pci.keys() | drivers.keys() | registered.keys()):
        entry = pci.get(device, {})
        name = entry.get("name", "")
        identities = drivers.get(device, {})
        row = registered.get(device)
        match = re.match(r"(GV|TU|GA|AD|GH|GB)\d+", name)
        family = FAMILIES.get(match[1]) if match else None
        if family is None and row:
            family = FAMILIES.get(row["chip"][:2].upper())
        base = {"id": "0x" + device, "pci_name": name}
        if family is None:
            # Older/later known chips are outside this six-family census. An identity without
            # a classified chip stays visible, rather than guessing architecture from its name
            # (e.g. RTX 2050 is Ampere; RTX 3050 A is Ada).
            if identities and not re.match(r"(?:(?:G[DFKMPRT]|NV|MCP|C)\d+|G\d+)", name):
                unclassified.append({**base, "nvidia": identities})
            continue
        if NON_GPU.search(name):
            excluded.append({**base, "reason": "non-GPU PCI function"})
            continue
        names = sorted({n for i in identities.values() for n in i["names"]})
        rows.append({**base, "generation": family,
                     "registry": row["name"] if row else None,
                     "triage": classify(name, identities),
                     "multiple_driver_names": len(names) > 1,
                     "nvidia": identities, "pci_subsystems": entry.get("subsystems", {})})
    counts = {}
    for family in FAMILIES.values():
        group = [r for r in rows if r["generation"] == family]
        counts[family] = {"candidate_ids": len(group),
                          "registered_ids": sum(r["registry"] is not None for r in group),
                          "missing_ids": sum(r["registry"] is None for r in group),
                          "missing_driver_listed_ids": sum(r["registry"] is None and bool(r["nvidia"])
                                                           for r in group)}
    return {"scope": "Volta, Turing, Ampere, Ada, Hopper and Blackwell PCI candidates; "
                     "not a claim of all shipped boards or hardware qualification",
            "counts": counts, "rows": rows, "excluded": excluded,
            "unclassified_driver_ids": unclassified,
            "missing_by_triage": dict(sorted(Counter(r["triage"] for r in rows
                                                     if r["registry"] is None).items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pci-ids", type=Path, required=True)
    parser.add_argument("--supported-chips", type=Path, action="append", required=True,
                        help="repeat for current and legacy driver lists")
    parser.add_argument("--registry", type=Path,
                        default=Path(__file__).resolve().parents[2] / "src/core/data/gpu_facts.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        inputs = [args.pci_ids, *args.supported_chips, args.registry]
        content = {p: p.read_bytes() for p in inputs}
        hashes = {p: hashlib.sha256(content[p]).hexdigest() for p in inputs}
        labels = {p: f"{p.name}@{hashes[p]}" for p in inputs}
        pci = pci_devices(content[args.pci_ids].decode("utf-8"))
        if not pci:
            raise ValueError("pci.ids: no NVIDIA devices (wrong file or changed format)")
        drivers = driver_identities([(labels[p], content[p].decode("utf-8"))
                                     for p in args.supported_chips])
        result = catalog(pci, drivers, json.loads(content[args.registry]))
        result["inputs"] = [{"file": p.name, "sha256": hashes[p], "source_id": labels[p]}
                            for p in inputs]
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"gpu_catalog: {exc}\n")
    print(json.dumps({"counts": result["counts"],
                      "missing_by_triage": result["missing_by_triage"],
                      "unclassified_driver_ids": len(result["unclassified_driver_ids"])}, indent=2))


if __name__ == "__main__":
    main()
