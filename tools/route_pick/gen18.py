"""gen18.py (lab only, 24.11): the analyzer's T 1-8 bands for one board, over a list of problems, as a
ninferno-route-pick/1 file. Read by a build whose NINFERNO_LAB_PICK_T18=1 lets predicted bands answer at T 1-8
(route_table.cpp, route_authorities.cpp). Not a shipped table: an e2e A/B of pick.py against the T 1-8 rules.

    python3 tools/route_pick/gen18.py --root . --pci 0x2C05 --shapes FILE --out routes_lab/pick/gb203-70sm.json

FILE: lines "fmt a16 epilogue n k" (route_tune --shapes). Bands: runs of equal picks over T 1..8.
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facts  # noqa: E402
import gen  # noqa: E402
import pick  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--pci', required=True)
    ap.add_argument('--shapes', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    F = facts.load_facts(a.root)
    row = next(r for r in F['rows'] if str(r.get('id', '')).lower() == a.pci.lower())
    entries, seen, skipped = [], set(), 0
    for line in open(a.shapes, encoding='utf-8'):
        p = line.split()
        if len(p) != 5 or p[1] != 'a16':
            continue
        fmt, ep, n, k = p[0], p[2], int(p[3]), int(p[4])
        if (fmt, ep, n, k) in seen:
            continue
        seen.add((fmt, ep, n, k))
        seq = []
        for T in range(1, 9):
            r = pick.pick(row, F, dict(format=fmt, epilogue=ep, n=n, k=k, T=T))
            seq.append(r)
        if None in seq:
            skipped += 1
            continue
        bands, lo = [], 1
        for T in range(2, 10):
            if T == 9 or seq[T - 1] != seq[T - 2]:
                bands.append([lo, T - 1, seq[T - 2]])
                lo = T
        entries.append(dict(format=fmt, compute='a16', epilogue=ep, n=n, k=k, bands=bands))
    doc = {'schema': gen.SCHEMA, 'gpu': {'class': gen.class_id(row), 'board': row['name'], 'pci_id': row['id']},
           'source': 'tools/route_pick/gen18.py (lab): pick.py at T 1-8', 'inputs': {}, 'entries': entries}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(gen.text(doc))
    print('%s: %d entries, %d skipped (unpriced)' % (a.out, len(entries), skipped))


if __name__ == '__main__':
    main()
