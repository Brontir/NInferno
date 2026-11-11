#!/usr/bin/env python3
"""Merge two outputs of gen_kernel_roofline_golden into kernel_roofline_golden.inc.

The routes a build offers differ by build set (cluster K-split register-MMA schedules need the
full sm_120a build; the deep staged schedules need the 163 KiB block of an sm_80 build). The
golden is the union: for every call, every (family, schedule) any input has. A route that both
inputs price must agree bit for bit (the same %.17g string, which round-trips a double), or the
merge fails: that is the check that the two build sets' pre-refactor prices are one function of
(call, route, device facts), and the test compares the merged table exactly in either build.

usage: merge_kernel_roofline_golden.py A.inc B.inc OUT.inc
(A and B are the raw generator outputs, made at the same pre-refactor commit; the header of A
is kept without its build label, with a line naming the merge.)
"""
import re
import sys

CALL = re.compile(r"^\{(\d+),(\d+),(\d+),(\d+),(\d+),g::Epilogue::(\w+),(\d+),(\d+)\},$")
ROUTE = re.compile(r"^\{(\d+),(\d+),([^,}]+)\},$")


def parse(path):
    header, boards, calls, routes = [], None, [], []
    mode = "head"
    with open(path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            line = line.strip()
            if line.startswith("constexpr const char* kGoldenBoards"):
                boards = line
            elif line.startswith("constexpr GoldenCall"):
                mode = "calls"
            elif line.startswith("constexpr GoldenRoute"):
                mode = "routes"
            elif line.startswith("};"):
                mode = "tail"
            elif mode == "head" and line.startswith("//"):
                header.append(line)
            elif mode == "calls":
                m = CALL.match(line)
                if m:
                    calls.append(m.groups())
            elif mode == "routes":
                m = ROUTE.match(line)
                if m:
                    routes.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    if boards is None or not calls or not routes:
        sys.exit(f"{path}: not a generator output")
    table = {}
    for b, fmt, n, k, t, epi, first, count in calls:
        key = (int(b), int(fmt), int(n), int(k), int(t), epi)
        table[key] = {
            (fam, sch): sec
            for fam, sch, sec in routes[int(first) : int(first) + int(count)]
        }
    return header, boards, table


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ha, ba, ta = parse(sys.argv[1])
    hb, bb, tb = parse(sys.argv[2])
    if ba != bb:
        sys.exit("the two inputs were made over different board lists")
    merged, common, differ = {}, 0, 0
    for key in sorted(set(ta) | set(tb)):
        row = dict(ta.get(key, {}))
        for route, sec in tb.get(key, {}).items():
            if route in row:
                common += 1
                if row[route] != sec:
                    differ += 1
                    print(f"DIFFER {key} route {route}: {row[route]} vs {sec}", file=sys.stderr)
            else:
                row[route] = sec
        merged[key] = row
    if differ:
        sys.exit(f"{differ} routes priced differently by the two builds")
    calls_out, routes_out = [], []
    for key, row in merged.items():
        first = len(routes_out)
        for (fam, sch), sec in sorted(row.items()):
            routes_out.append(f"{{{fam},{sch},{sec}}},")
        b, fmt, n, k, t, epi = key
        calls_out.append(
            f"{{{b},{fmt},{n},{k},{t},g::Epilogue::{epi},{first},{len(row)}}},"
        )
    with open(sys.argv[3], "w", encoding="utf-8", newline="\n") as out:
        # A's first two lines name the commit and A's own build; the union is of both builds,
        # named on the next lines, so A's build label is dropped.
        for line in ha[:2]:
            out.write(re.sub(r", sm_\w+ build\.$", ".", line) + "\n")
        out.write(
            "// Union of two builds' routes (merge_kernel_roofline_golden.py): the full sm_120a build\n"
            "// (cluster schedules) and the sm_80 build (163 KiB: staged schedules 10-12); %d routes\n"
            "// priced by both agreed bit for bit.\n" % common
        )
        out.write(ba + "\n")
        for line in ha[2:]:
            if not line.startswith("constexpr const char* kGoldenBoards"):
                out.write(line + "\n")
        out.write("constexpr GoldenCall kGoldenCalls[] = {\n")
        out.write("\n".join(calls_out) + "\n};\n")
        out.write("constexpr GoldenRoute kGoldenRoutes[] = {\n")
        out.write("\n".join(routes_out) + "\n};\n")
    print(f"{len(merged)} calls, {len(routes_out)} routes, {common} priced by both, 0 differ")


main()
