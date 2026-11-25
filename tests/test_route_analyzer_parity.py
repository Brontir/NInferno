"""The engine's route analyzer (src/ops/generic/route_analyzer.cpp, `ninferno-serve route-analyzer`) equals
tools/route_pick/analyzer.py on every cell: for one board of every chip the analyzer prices, every A16 shape the
route tables list (store, add, SwiGLU) and a set of shapes no table lists (a new model's), at T 1-8, both give the
same candidate routes in the same order and every price within 1e-9 relative. The C++ law is the product's; the
Python stays the lab's copy until the exam tools read the C++ one.

    python3 tests/test_route_analyzer_parity.py <ninferno-serve>"""
import json
import math
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'route_pick'))
import analyzer  # noqa: E402
import analyzer_facts  # noqa: E402

PRICED_CC = (70, 75, 86, 89, 120)
# Shapes no route table lists (Inkling-2-Mini's decode projections, #1221; a long-K and a tiny-n call).
NEW_SHAPES = [('q8', 'store', 3072, 2304), ('q8', 'store', 1024, 2304), ('q8', 'store', 2304, 3072),
              ('q8', 'store', 384, 2304), ('q8', 'swiglu', 3072, 2304), ('bf16', 'store', 24, 2304),
              ('bf16', 'store', 2304, 64), ('bf16', 'store', 64, 4608), ('q4k', 'store', 7168, 18432),
              ('q6k', 'add', 2880, 2880), ('iq4xs', 'swiglu', 8192, 3072), ('q40', 'store', 4096, 4096),
              ('nvfp4', 'store', 5120, 5120)]
REL = 1e-9


def release_shapes():
    seen = set()
    routes = os.path.join(ROOT, 'routes')
    for fn in sorted(os.listdir(routes)):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(routes, fn), encoding='utf-8') as f:
            for e in json.load(f).get('entries', []):
                if e.get('compute') == 'a16' and e['format'] in analyzer.BITS:
                    seen.add((e['format'], e.get('epilogue', 'store'), int(e['n']), int(e['k'])))
    return sorted(seen | set(NEW_SHAPES))


def boards(F):
    """The first desktop row (else the first row) of each chip of PRICED_CC with the analyzer's facts."""
    out = {}
    for row in F['rows']:
        if not row.get('cc') or int(row['cc']) not in PRICED_CC or row.get('mma_slow'):
            continue
        if analyzer.card_of(row, F) is None:
            continue
        chip = row['chip']
        if chip not in out or (row.get('form') == 'desktop' and out[chip].get('form') != 'desktop'):
            out[chip] = row
    return [out[c] for c in sorted(out)]


def main():
    serve = sys.argv[1]
    F = analyzer_facts.load_facts(ROOT)
    rows = boards(F)
    shapes = release_shapes()
    cells = [(row, fmt, ep, n, k, T) for row in rows for fmt, ep, n, k in shapes for T in range(1, 9)]
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, newline='\n') as f:
        for row, fmt, ep, n, k, T in cells:
            f.write('%s %s %s %d %d %d\n' % (row['id'], fmt, ep, n, k, T))
        path = f.name
    try:
        out = subprocess.run([serve, 'route-analyzer', '--cells', path], capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(path)
    got = {}
    for ln in out.splitlines():
        q = ln.split()
        key = (q[0].lower(), q[1], q[2], int(q[3]), int(q[4]), int(q[5]))
        got.setdefault(key, [])
        if q[6] != '-':
            got[key].append((q[6], float(q[7])))
    bad, priced = [], 0
    for row, fmt, ep, n, k, T in cells:
        key = (str(row['id']).lower(), fmt, ep, n, k, T)
        shape = dict(format=fmt, epilogue=ep, n=n, k=k, T=T)
        want = [(r, analyzer.price_ns(row, F, shape, r)) for r in analyzer.rank(row, F, shape)]
        have = got.get(key)
        if have is None:
            bad.append('%s: no C++ answer' % (key,))
            continue
        if [r for r, _ in want] != [r for r, _ in have]:
            bad.append('%s %s: routes %s (python) vs %s (C++)' % (row['name'], key[1:], [r for r, _ in want][:6],
                                                                   [r for r, _ in have][:6]))
            continue
        for (r, a), (_, b) in zip(want, have):
            priced += 1
            if not (math.isfinite(b) and abs(a - b) <= REL * max(abs(a), 1.0)):
                bad.append('%s %s %s: %.9g ns (python) vs %.9g (C++)' % (row['name'], key[1:], r, a, b))
    for b in bad[:40]:
        print('MISMATCH ' + b)
    print('%d boards x %d shapes x T 1-8: %d cells, %d prices, %d mismatches' % (len(rows), len(shapes), len(cells),
                                                                                priced, len(bad)))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
