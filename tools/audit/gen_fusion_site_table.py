#!/usr/bin/env python3
"""Generates src/ops/common/fusion_site_table.h from tools/audit/fusion_sites.json (#617 step H).

The registry is the source of truth; the header is its constexpr form, one row a site in registry
order, so the manifest emitter (ops/common/fusion_site_manifest.h) cannot lack a site: a site the
registry gains is a row here, and tests/test_fusion_site_manifest.py fails until the header is
regenerated. `--check` exits 1 when the header on disk differs from what would be written.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "tools" / "audit" / "fusion_sites.json"
OUT = ROOT / "src" / "ops" / "common" / "fusion_site_table.h"

DELIM = "~~"


def raw(text):
    if (")" + DELIM + '"') in text:
        raise ValueError("a registry string holds the raw-literal delimiter")
    return 'R"' + DELIM + "(" + text + ")" + DELIM + '"'


def render():
    sites = json.loads(REGISTRY.read_text(encoding="utf-8"))["sites"]
    rows = []
    for s in sites:
        rows.append(
            "    {" + ", ".join([
                raw(s["id"]), raw(s["name"]), raw(s["layer"]), raw(s["byte_promise"]),
                raw(s["priced"]["status"]), raw(" | ".join(s["alternatives"])),
                raw(s["rule_today"])]) + "},")
    return ("#pragma once\n\n"
            "// GENERATED from tools/audit/fusion_sites.json by tools/audit/gen_fusion_site_table.py "
            "(#617 step\n"
            "// H). Do not edit: change the registry and regenerate; "
            "tests/test_fusion_site_manifest.py checks it\n"
            "// is current.\n\n"
            "#include <cstddef>\n#include <string_view>\n\n"
            "namespace ninferno::ops::detail::fusion {\n\n"
            "struct SiteRow {\n"
            "    std::string_view id, name, layer, byte_promise, priced, alternatives, rule_today;\n"
            "};\n\n"
            "// clang-format off\n"
            "inline constexpr SiteRow kSiteTable[] = {\n" + "\n".join(rows) + "\n};\n"
            "// clang-format on\n\n"
            "inline constexpr std::size_t kSiteCount = sizeof(kSiteTable) / sizeof(kSiteTable[0]);\n\n"
            "} // namespace ninferno::ops::detail::fusion\n")


def main(argv):
    text = render()
    if "--check" in argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is not current: run tools/audit/gen_fusion_site_table.py",
                  file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
