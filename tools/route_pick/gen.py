#!/usr/bin/env python3
"""gen.py: the analyzer's predicted route bands (T 9-16) for the board classes no route table was measured on.

    python3 tools/route_pick/gen.py            write routes/pick/<chip>-<sms>sm.json, one file a board class
    python3 tools/route_pick/gen.py --check    exit 1 when a file differs from what the facts give now (CI)
    python3 tools/route_pick/gen.py --class ga102:82sm --out DIR    one class, any (a lab A/B of a tabled card)

A board class is a chip and an SM count (core/device_class.h gpu_class_id). Written for every class of
src/core/data/gpu_facts.json whose chip has no table in routes/ (no routes/<chip>.json, no routes/<chip>-*.json),
whose architecture has the facts pick.py prices from (sm_75, sm_86, sm_89, sm_120a) and whose board has tensor
cores (not mma_slow: GTX 16), and that no table's board is within kCarryDistance of (board_distance from the
facts rows: there the table's answers are Carried, kept, and the engine reads no file). The class's board is its
first desktop row, else its first row.

The problems are every generic entry of the route tables of the same compute capability (the release models'
shapes) that pick.py can price: an A16 entry of a format in pick.BITS with the store or SwiGLU epilogue. For each,
pick.pick() at T 9..16 gives one route a width, written as runs of equal routes. Nothing is measured on the card.

The engine reads the file only where the device has no table of its own chip (src/ops/route_table.cpp): its
answers are Predicted, below a pin and a measured or carried table, above the ranker.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import facts  # noqa: E402
import pick  # noqa: E402

LO, HI = 9, 16
SCHEMA = 'ninferno-route-pick/1'
ARCH_OF_CC = {75: '75', 86: '86', 89: '89', 120: '120a'}


def class_id(row):
    return '%s:%dsm' % (row['chip'], int(row['sms']))


def table_cc(root, path, F):
    """The compute capability of a route table, by its board (gpu.pci_id), else by its class's chip."""
    with open(path, encoding='utf-8') as f:
        t = json.load(f)
    gpu = t.get('gpu', {})
    pci = str(gpu.get('pci_id', '')).lower()
    chip = str(gpu.get('class', '')).split(':')[0]
    for row in F['rows']:
        if pci and str(row.get('id', '')).lower() == pci:
            return int(row['cc']), t
    for row in F['rows']:
        if row.get('chip') == chip:
            return int(row['cc']), t
    return None, t


def table_boards(F, all_tables):
    """The gpu_facts rows of the boards the tables were measured on (gpu.pci_id)."""
    ids = {str(t.get('gpu', {}).get('pci_id', '')).lower() for _, _, t in all_tables}
    return [r for r in F['rows'] if str(r.get('id', '')).lower() in ids]


def tables(root, F):
    out = []
    rdir = os.path.join(root, 'routes')
    for name in sorted(os.listdir(rdir)):
        if name.endswith('.json'):
            cc, t = table_cc(root, os.path.join(rdir, name), F)
            out.append((name[:-5].split('-')[0], cc, t))
    return out


def carry_distance(root):
    """core/measured_boards.h kCarryDistance: within it a table's answers carry (Carried) and the engine reads no pick."""
    with open(os.path.join(root, 'src', 'core', 'measured_boards.h'), encoding='utf-8') as f:
        m = re.search(r'inline constexpr double kCarryDistance\s*=\s*([0-9.]+)\s*;', f.read())
    if not m:
        sys.exit('core/measured_boards.h: no kCarryDistance')
    return float(m.group(1))


def board_distance(a, b):
    """core/device_class_host.cpp board_distance for two gpu_facts rows of one compute capability (else inf)."""
    if int(a['cc']) != int(b['cc']) or bool(a.get('mma_slow')) != bool(b.get('mma_slow')):
        return math.inf
    apart = lambda x, y: abs(math.log2(x) - math.log2(y)) if x > 0 and y > 0 else 0.0
    bw = lambda r: float(r['bus_bits']) * float(r['gbps']) / 8.0
    return apart(float(a['sms']), float(b['sms'])) + apart(bw(a), bw(b)) + 0.5 * apart(float(a['l2_mib']), float(b['l2_mib']))


def classes(F, tabled_chips, table_boards, carry):
    """The classes the engine would read a file for: no table of the chip, and none within `carry` (the loader
    reads routes/pick only past it, where a table's answers would be Transferred)."""
    seen = {}
    for row in F['rows']:
        cc = int(row.get('cc', 0))
        if cc not in ARCH_OF_CC or ARCH_OF_CC[cc] not in F['isa'] or row.get('mma_slow'):
            continue
        if row['chip'] in tabled_chips or any(board_distance(row, t) <= carry for t in table_boards):
            continue
        key = class_id(row)
        if key not in seen or (row.get('form') == 'desktop' and seen[key].get('form') != 'desktop'):
            seen[key] = row
    return seen


def problems(all_tables, cc):
    out = {}
    for _, tcc, t in all_tables:
        if tcc != cc:
            continue
        for e in t.get('entries', []):
            ep = e.get('epilogue', 'store')
            if 'bands' not in e or e['format'] not in pick.BITS or e['compute'] != 'a16' or ep not in ('store', 'swiglu'):
                continue
            out[(e['format'], ep, int(e['n']), int(e['k']))] = True
    return sorted(out)


def runs(seq):
    out = []
    for T, r in seq:
        if out and out[-1][2] == r and out[-1][1] == T - 1:
            out[-1][1] = T
        else:
            out.append([T, T, r])
    return out


def overlay(row, F, probs, inputs):
    entries = []
    for fmt, ep, n, k in probs:
        seq = []
        for T in range(LO, HI + 1):
            r = pick.pick(row, F, dict(format=fmt, epilogue=ep, n=n, k=k, T=T))
            if r is None:
                seq = None
                break
            seq.append((T, r))
        if seq:
            entries.append(dict(format=fmt, compute='a16', epilogue=ep, n=n, k=k, bands=runs(seq)))
    return {
        'schema': SCHEMA,
        'gpu': {'class': class_id(row), 'board': row['name'], 'pci_id': row['id']},
        'source': 'tools/route_pick/gen.py: pick.py over the facts below, nothing measured on this card',
        'inputs': inputs,
        'entries': entries,
    }


def inputs_of(root, F, arch):
    """The files one class's overlay is priced from: the data files and its own architecture's kernel facts."""
    out = {}
    for p in [q for q in facts.input_paths(root, F) if not q.startswith(facts.KERNEL_FACTS) or q.endswith('/sm_%s.txt' % arch)]:
        with open(os.path.join(root, p), 'rb') as f:
            out[p] = hashlib.sha256(f.read().replace(b'\r\n', b'\n')).hexdigest()
    for p in ('tools/route_pick/pick.py', 'tools/route_pick/facts.py', 'tools/route_pick/gen.py'):
        with open(os.path.join(root, p), 'rb') as f:
            out[p] = hashlib.sha256(f.read().replace(b'\r\n', b'\n')).hexdigest()
    return out


def text(doc):
    return json.dumps(doc, indent=1, sort_keys=False) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--class', dest='only')
    ap.add_argument('--out')
    a = ap.parse_args()
    F = facts.load_facts(a.root)
    all_tables = tables(a.root, F)
    tabled = {chip for chip, _, _ in all_tables}
    if a.only:
        chip, sms = a.only.split(':')
        rows = [r for r in F['rows'] if r['chip'] == chip and '%dsm' % int(r['sms']) == sms]
        if not rows:
            sys.exit('no gpu_facts.json row of class %s' % a.only)
        want = {a.only: sorted(rows, key=lambda r: r.get('form') != 'desktop')[0]}
    else:
        want = classes(F, tabled, table_boards(F, all_tables), carry_distance(a.root))
    out_dir = a.out or os.path.join(a.root, 'routes', 'pick')
    stale = []
    expected = set()
    for key, row in sorted(want.items()):
        doc = overlay(row, F, problems(all_tables, int(row['cc'])), inputs_of(a.root, F, ARCH_OF_CC[int(row['cc'])]))
        name = key.replace(':', '-') + '.json'
        expected.add(name)
        path = os.path.join(out_dir, name)
        body = text(doc)
        if a.check:
            old = open(path, encoding='utf-8').read() if os.path.exists(path) else None
            if old != body:
                stale.append(name)
            continue
        os.makedirs(out_dir, exist_ok=True)
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(body)
        print('%s: %s, %d entries' % (name, row['name'], len(doc['entries'])))
    if a.check and not a.only and os.path.isdir(out_dir):
        stale += sorted(set(n for n in os.listdir(out_dir) if n.endswith('.json')) - expected)
    if stale:
        print('routes/pick is not what the facts give (regenerate: python3 tools/route_pick/gen.py): ' + ', '.join(stale))
        sys.exit(1)
    if a.check:
        print('routes/pick: %d classes, in step with the facts' % len(expected))


if __name__ == '__main__':
    main()
