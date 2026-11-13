"""Coverage auditing must preserve the variants a device-only registry cannot name."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/audit"))
import gpu_catalog  # noqa: E402


PCI = """10de  NVIDIA Corporation
\t1dba  GV100GL [Quadro GV100]
\t\t10de 12eb  TITAN V CEO Edition
\t22d1  GB20B PCI bridge
\t25a9  GA107M [GeForce RTX 2050]
\t28a3  AD107M [GeForce RTX 3050 A Laptop GPU]
\t2bb3  GB202GL [RTX PRO 5000 Blackwell]
\t2e12  GB20B [GB10]
1234  Another vendor
\t2bb3  Not an NVIDIA board
"""
CHIPS = """<table><tr><th>GPU</th><th>PCI ID</th></tr>
<tr><td>Quadro <b>GV100</b></td><td>1DBA</td><td>I</td></tr>
<tr><td>NVIDIA TITAN V JHH Special Edition</td><td>1DBA 10DE 12EB</td></tr>
<tr><td>NVIDIA RTX PRO 5000 Blackwell</td><td>2BB3 10DE 204D</td></tr>
<tr><td>NVIDIA RTX PRO 5000 72GB Blackwell</td><td>2BB3 10DE 227A</td></tr>
<tr><td>New unnamed chip</td><td>3456</td></tr></table>"""
REGISTRY = {"gpus": [{"id": "0x1DBA", "chip": "gv100", "name": "Quadro GV100"}]}


class CatalogTest(unittest.TestCase):
    def result(self):
        return gpu_catalog.catalog(gpu_catalog.pci_devices(PCI),
                                   gpu_catalog.driver_identities([("current.html", CHIPS),
                                                                  ("legacy.html", CHIPS)]),
                                   REGISTRY)

    def test_subsystems_are_not_collapsed_into_device_coverage(self):
        result = self.result()
        gv = next(r for r in result["rows"] if r["id"] == "0x1DBA")
        self.assertEqual(gv["registry"], "Quadro GV100")
        self.assertTrue(gv["multiple_driver_names"])
        self.assertEqual(gv["pci_subsystems"]["10DE 12EB"], "TITAN V CEO Edition")
        self.assertEqual(gv["nvidia"]["1DBA 10DE 12EB"]["names"],
                         ["NVIDIA TITAN V JHH Special Edition"])
        self.assertEqual(gv["nvidia"]["1DBA"]["sources"], ["current.html", "legacy.html"])
        pro = next(r for r in result["rows"] if r["id"] == "0x2BB3")
        self.assertIsNone(pro["registry"])
        self.assertEqual(len(pro["nvidia"]), 2)
        self.assertEqual(result["counts"]["Blackwell"]["missing_driver_listed_ids"], 1)

    def test_architecture_is_not_inferred_from_marketing_series(self):
        rows = {r["id"]: r for r in self.result()["rows"]}
        self.assertEqual(rows["0x25A9"]["generation"], "Ampere")
        self.assertEqual(rows["0x28A3"]["generation"], "Ada")
        self.assertEqual(rows["0x25A9"]["triage"], "oem-clock-range-review")

    def test_bridge_and_other_vendor_are_not_counted_as_gpus(self):
        result = self.result()
        self.assertNotIn("0x22D1", [r["id"] for r in result["rows"]])
        self.assertEqual([r["id"] for r in result["excluded"]], ["0x22D1"])
        pro = next(r for r in result["rows"] if r["id"] == "0x2BB3")
        self.assertIn("RTX PRO", pro["pci_name"])

    def test_soc_and_unknown_identity_remain_visible(self):
        result = self.result()
        gb10 = next(r for r in result["rows"] if r["id"] == "0x2E12")
        self.assertEqual(gb10["triage"], "platform-review")
        self.assertEqual([r["id"] for r in result["unclassified_driver_ids"]], ["0x3456"])

    def test_changed_driver_document_does_not_report_empty_success(self):
        with self.assertRaisesRegex(ValueError, "no supported-chip rows"):
            gpu_catalog.driver_identities([("error.html", "<h1>Access denied</h1>")])

    def test_cli_records_inputs_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, text in [("pci.ids", PCI), ("chips.html", CHIPS),
                               ("registry.json", json.dumps(REGISTRY))]:
                (root / name).write_text(text, encoding="utf-8")
            output = root / "audit.json"
            cmd = [sys.executable, str(ROOT / "tools/audit/gpu_catalog.py"),
                   "--pci-ids", str(root / "pci.ids"), "--supported-chips", str(root / "chips.html"),
                   "--registry", str(root / "registry.json"), "--output", str(output)]
            subprocess.run(cmd, check=True, capture_output=True)
            first = output.read_bytes()
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertEqual(output.read_bytes(), first)
            self.assertEqual(len(json.loads(first)["inputs"]), 3)
            self.assertTrue(all(len(i["sha256"]) == 64 for i in json.loads(first)["inputs"]))


if __name__ == "__main__":
    unittest.main()
