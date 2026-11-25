#!/usr/bin/env python3
"""gen.py: the analyzer's predicted route bands (T 9-16, and T 17-32 on sm_89, #785): for the board classes no route table was measured on,
and for a tabled board, its table's gaps (#1068).

    python3 tools/route_pick/gen.py            write routes/pick/<chip>-<sms>sm.json, one file a board class
    python3 tools/route_pick/gen.py --check    exit 1 when a file differs from what the facts give now (CI)
    python3 tools/route_pick/gen.py --class ga102:82sm --out DIR    one class, any (a lab A/B of a tabled card)
    ... --class ga102:82sm --out DIR --epilogues store,swiglu,add   the same with the add epilogue's bands (#1068)

A board class is a chip and an SM count (core/device_class.h gpu_class_id). Written for every class of
src/core/data/gpu_facts.json whose chip has no table in routes/ (no routes/<chip>.json, no routes/<chip>-*.json),
whose architecture has the facts pick.py prices from (sm_75, sm_86, sm_89, sm_120a) and whose board has tensor
cores (not mma_slow: GTX 16), and that no table's board is within kCarryDistance of (board_distance from the
facts rows: there the table's answers are Carried, kept, and the engine reads no file), and whose SM count lies
between the fewest and the most of the same compute capability's table boards (classes()). The class's board is
its first desktop row, else its first row.

The problems are every generic entry of the route tables of the same compute capability (the release models'
shapes) that pick.py can price: an A16 entry of a format in pick.BITS with the store or SwiGLU epilogue. The add
epilogue (o_proj, down) is priced but not written: on the RTX 4060 Ti (Qwen3-8B Q4_K_M) its bands lost 2.9 % end to
end against the borrowed table, the law's n2.w1x4 against the measured n4.w1x4 on the Q6_K down (#1051). For each,
pick.pick() at T 9..16 gives one route a width, written as runs of equal routes. Nothing is measured on the card.

A tabled board (a table's gpu.pci_id row of an architecture pick.py prices, tabled_classes()) gets a file of its
table's gaps: the same problems, the add epilogue included, at the widths its own table does not answer. Past kCarryDistance an
untabled class's bands answer ahead of the nearest table; within it the loader takes a band only in a gap of the
table (no entry of its own, one borrowed by fill_from, or a width its own entry has no band at), so a measured band
stands above a prediction. The add
bands lost on the 4060 Ti where they replaced measured entries (above); in the gaps alone they were scored by leaving
each tabled card's table out on full route_tune grids (#1068 gate step 1: no worse in median and p90 on GA10x, Ada,
GB20x) and held end to end within the A/A band (4060 Ti, 5070 Ti, 4090).

The engine reads the file only on the board it was priced for, and only under --route-policy evidence
(src/ops/route_table.cpp): its answers are Predicted, below a pin and a measured or carried entry, above a borrowed
entry and the ranker.
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
import analyzer  # noqa: E402
import analyzer_facts  # noqa: E402
import facts  # noqa: E402
import pick  # noqa: E402

# The decode and verify widths the analyzer (analyzer.py) ranks routes at, every class of ARCH_OF_CC (ranked_entries).
RANK_LO, RANK_HI = 1, 8
# The chips whose classes get no ranked lists yet (the Fable r21 gate: no card below -1 % e2e against master): AD102
# (RTX 4090), whose rmma the analyzer prices 12-16 % below its time at T 2-4 (#1211; Qwen3.5-9B Q4_K_M -1.5 / -1.6 /
# -3.0 % at C 2 / 3 / 4). Its calls keep the authorities after the analyzer until the DRAM side is priced from the
# die's activation rate.
HELD_CHIPS = ()
LO, HI = 9, 16
# The narrow band (#785): on these compute capabilities the file also holds T 17..NARROW_HI, which answers the int8
# calls of those widths outside the prefill profile (speculative verify rounds, decode batches) ahead of the cost
# model (route_authorities route_table_entry, table_predicted_answer), and an A16 call's table gaps. Only sm_89:
# route_tune a16 decode-only at T 24 / 32, the cells the cost model answers, mean regret pick.py / cost model
# RTX 4070 Ti +2.0 / +27.1 %, RTX 4090 +4.1 / +19.9 %; on the RTX 5090 (sm_120a) pick.py lost (+5.7 / +3.9 %),
# and no sm_86 grid was scored at those widths.
NARROW_CC = (89,)
NARROW_HI = 32
SCHEMA = 'ninferno-route-pick/2'
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
    """core/measured_boards.h kCarryDistance: within it a table's answers carry (Carried) and a pick fills only its gaps."""
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
    """The untabled classes a file of every problem is written for: no table of the chip, and none within `carry`
    (past it a table's answers would be Transferred; within it a pick fills only the table's gaps, and only a
    tabled board's gaps are written, tabled_classes()). And only between the fewest
    and the most SMs of the boards that hold a table of the same compute capability: the law interpolates
    between measured boards, it is not extrapolated past them (the one measured miss, #1051, was a long-K
    call on a board of few SMs). Today that keeps AD103 and GA104 and leaves out AD107, GA107 (10-22 SMs)
    and Turing (one table board, the 68-SM RTX 2080 Ti)."""
    span = {}
    for t in table_boards:
        lo, hi = span.get(int(t['cc']), (math.inf, 0))
        span[int(t['cc'])] = (min(lo, int(t['sms'])), max(hi, int(t['sms'])))
    seen = {}
    for row in F['rows']:
        cc = int(row.get('cc', 0))
        if cc not in ARCH_OF_CC or ARCH_OF_CC[cc] not in F['isa'] or row.get('mma_slow'):
            continue
        if row['chip'] in tabled_chips or any(board_distance(row, t) <= carry for t in table_boards):
            continue
        lo, hi = span.get(cc, (math.inf, 0))
        if not lo <= int(row['sms']) <= hi:
            continue
        key = class_id(row)
        if key not in seen or (row.get('form') == 'desktop' and seen[key].get('form') != 'desktop'):
            seen[key] = row
    return seen


# The epilogues whose bands the file holds (the module docstring says why not add).
WRITTEN_EPILOGUES = ('store', 'swiglu')
# A tabled board's gap file holds the add epilogue too (#1068 gate step 1, the module docstring).
GAP_EPILOGUES = ('store', 'swiglu', 'add')
EPILOGUES = frozenset(GAP_EPILOGUES)


def covered(entry):
    """The widths of LO..HI a table entry's own bands answer. Past HI the file's bands are written whatever the table
    holds: they answer the int8 calls of T 17-32 the table stands aside for (route_authorities a8_wide_call, #785),
    and where the table's own band covers a width it stands above the file's (route_table.cpp own_covers)."""
    return frozenset(T for T in range(LO, HI + 1)
                     if any(int(b[0]) <= T <= int(b[1]) for b in entry.get('bands', [])))


def tabled_classes(F, all_tables):
    """The tabled boards pick.py can price (a table's gpu.pci_id row of an architecture in ARCH_OF_CC with its isa
    facts, tensor cores), each with the problems its own table holds: its file is of the gaps, the rest."""
    out = {}
    for _, _, t in all_tables:
        pci = str(t.get('gpu', {}).get('pci_id', '')).lower()
        row = next((r for r in F['rows'] if pci and str(r.get('id', '')).lower() == pci), None)
        if row is None or int(row.get('cc', 0)) not in ARCH_OF_CC or ARCH_OF_CC[int(row['cc'])] not in F['isa']                 or row.get('mma_slow'):
            continue
        # {problem: the widths of LO..HI its own table answers}: the file holds bands at the other
        # widths only, the gaps the loader takes a band at (route_table.cpp, #1068).
        own = {(e['format'], e.get('epilogue', 'store'), int(e['n']), int(e['k'])): covered(e)
               for e in t.get('entries', []) if e.get('compute') == 'a16'}
        key = class_id(row)
        if key in out:
            sys.exit('two route tables of board class %s' % key)
        out[key] = (row, own)
    return out


def all_classes(F):
    """Every board class of an architecture in ARCH_OF_CC with its facts and tensor cores (not mma_slow), each by its
    first desktop row, else its first row: the classes the analyzer's T 1-8 ranked lists are written for (no span or
    carry limit: at T 1-8 the analyzer is the engine's first authority on every board it has a file for)."""
    seen = {}
    for row in F['rows']:
        cc = int(row.get('cc', 0))
        if cc not in ARCH_OF_CC or ARCH_OF_CC[cc] not in F['isa'] or row.get('mma_slow'):
            continue
        key = class_id(row)
        if key not in seen or (row.get('form') == 'desktop' and seen[key].get('form') != 'desktop'):
            seen[key] = row
    return seen


def problems(all_tables, cc, epilogues=WRITTEN_EPILOGUES, known=pick.BITS):
    out = {}
    for _, tcc, t in all_tables:
        if tcc != cc:
            continue
        for e in t.get('entries', []):
            ep = e.get('epilogue', 'store')
            if 'bands' not in e or e['format'] not in known or e['compute'] != 'a16' or ep not in epilogues:
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


def ranked_list(row, AF, shape):
    """analyzer.rank's routes of one call, cut after the first register-MMA schedule: the engine takes the first route
    its policy, profile and build admit (route_authorities.cpp analyzer_entry), and an A16 rmma schedule is admitted
    under every policy, so no route past it is ever taken. Empty: no route priced."""
    out = []
    for r in analyzer.rank(row, AF, shape):
        out.append(r)
        if r.startswith('rmma.'):
            break
    return out


def ranked_runs(row, AF, prob):
    """[[t_min, t_max, [route, ...]], ...] over RANK_LO..RANK_HI, runs of equal lists; a width with no priced route is
    left out (the engine's older authorities answer it)."""
    fmt, ep, n, k = prob
    out = []
    for T in range(RANK_LO, RANK_HI + 1):
        lst = ranked_list(row, AF, dict(format=fmt, epilogue=ep, n=n, k=k, T=T))
        if not lst:
            continue
        if out and out[-1][2] == lst and out[-1][1] == T - 1:
            out[-1][1] = T
        else:
            out.append([T, T, lst])
    return out


def overlay(row, F, probs, inputs, own=None, AF=None, rprobs=()):
    """The file of `probs`' bands, at the widths of LO..HI (LO..NARROW_HI on NARROW_CC) `own` ({problem: widths})
    leaves, and of `rprobs`' ranked lists at T RANK_LO..RANK_HI (analyzer.py over `AF`). A problem pick.py cannot price
    at some width of LO..HI has no bands; past HI the bands stop at the first width it cannot price."""
    own = own or {}
    bands = {}
    hi = NARROW_HI if int(row['cc']) in NARROW_CC else HI
    for fmt, ep, n, k in probs:
        seq = []
        for T in range(LO, hi + 1):
            if T in own.get((fmt, ep, n, k), ()):
                continue
            r = pick.pick(row, F, dict(format=fmt, epilogue=ep, n=n, k=k, T=T))
            if r is None:
                if T > HI:
                    break
                seq = None
                break
            seq.append((T, r))
        if seq:
            bands[(fmt, ep, n, k)] = runs(seq)
    ranked = {p: r for p, r in ((p, ranked_runs(row, AF, p)) for p in rprobs) if r}
    entries = []
    for fmt, ep, n, k in sorted(set(bands) | set(ranked)):
        e = dict(format=fmt, compute='a16', epilogue=ep, n=n, k=k)
        if (fmt, ep, n, k) in ranked:
            e['ranked'] = ranked[(fmt, ep, n, k)]
        if (fmt, ep, n, k) in bands:
            e['bands'] = bands[(fmt, ep, n, k)]
        entries.append(e)
    return {
        'schema': SCHEMA,
        'gpu': {'class': class_id(row), 'board': row['name'], 'pci_id': row['id']},
        'source': 'tools/route_pick/gen.py: analyzer.py (ranked, T 1-8) and pick.py (bands, T 9-32) over the facts '
                  'below, nothing measured on this card',
        'inputs': inputs,
        'entries': entries,
    }


def inputs_of(root, F, arch):
    """The files one class's overlay is priced from: the data files and its own architecture's kernel facts."""
    out = {}
    for p in [q for q in facts.input_paths(root, F) if not q.startswith(facts.KERNEL_FACTS) or q.endswith('/sm_%s.txt' % arch)]:
        with open(os.path.join(root, p), 'rb') as f:
            out[p] = hashlib.sha256(f.read().replace(b'\r\n', b'\n')).hexdigest()
    for p in ('tools/route_pick/analyzer.py', 'tools/route_pick/analyzer_facts.py', 'tools/route_pick/pick.py',
              'tools/route_pick/facts.py', 'tools/route_pick/gen.py', pick.MMQ_HEADER.replace(os.sep, '/')):
        with open(os.path.join(root, p), 'rb') as f:
            out[p] = hashlib.sha256(f.read().replace(b'\r\n', b'\n')).hexdigest()
    return out


def text(doc):
    """The file's text: the header indented, one line an entry (the ranked lists would take a line a route)."""
    head = {k: v for k, v in doc.items() if k != 'entries'}
    body = json.dumps(head, indent=1, sort_keys=False)[:-2]
    lines = [json.dumps(e, sort_keys=False) for e in doc['entries']]
    return body + ',\n "entries": [\n  ' + ',\n  '.join(lines) + '\n ]\n}\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--class', dest='only')
    ap.add_argument('--out')
    ap.add_argument('--epilogues', help='with --class: the epilogues to write (a lab A/B, #1068: store,swiglu,add); '
                    'the shipped files keep %s' % ','.join(WRITTEN_EPILOGUES))
    a = ap.parse_args()
    if a.epilogues is not None and not (a.only and a.out):
        sys.exit('--epilogues is for a lab file of one class (--class, --out): the shipped files keep %s' % ','.join(WRITTEN_EPILOGUES))
    epilogues = WRITTEN_EPILOGUES
    if a.epilogues is not None:
        epilogues = tuple(e.strip() for e in a.epilogues.split(','))
        bad = sorted(set(epilogues) - EPILOGUES)
        if bad or not all(epilogues):
            sys.exit('--epilogues: %s is not one of %s' % (','.join(bad) or '(none)', ','.join(GAP_EPILOGUES)))
    F = facts.load_facts(a.root)
    AF = analyzer_facts.load_facts(a.root)
    all_tables = tables(a.root, F)
    tabled = {chip for chip, _, _ in all_tables}
    if a.only:
        chip, sms = a.only.split(':')
        rows = [r for r in F['rows'] if r['chip'] == chip and '%dsm' % int(r['sms']) == sms]
        if not rows:
            sys.exit('no gpu_facts.json row of class %s' % a.only)
        want = {a.only: (sorted(rows, key=lambda r: r.get('form') != 'desktop')[0], epilogues, {})}
    else:
        want = {k: (row, WRITTEN_EPILOGUES, {})
                for k, row in classes(F, tabled, table_boards(F, all_tables), carry_distance(a.root)).items()}
        for k, (row, own) in tabled_classes(F, all_tables).items():
            if k in want:
                sys.exit('board class %s is both tabled and untabled' % k)
            want[k] = (row, GAP_EPILOGUES, own)
        # Every other class gets the T 1-8 ranked lists alone (no bands: `None`).
        for k, row in all_classes(F).items():
            want.setdefault(k, (row, None, {}))
    out_dir = a.out or os.path.join(a.root, 'routes', 'pick')
    stale = []
    expected = set()
    for key, (row, eps, own) in sorted(want.items()):
        full = frozenset(range(LO, (NARROW_HI if int(row['cc']) in NARROW_CC else HI) + 1))
        probs = [] if eps is None else [p for p in problems(all_tables, int(row['cc']), eps)
                                        if own.get(p, frozenset()) != full]
        # The ranked lists answer every problem of the capability's tables at T 1-8, a tabled board's own included.
        rprobs = () if row['chip'] in HELD_CHIPS else problems(all_tables, int(row['cc']), GAP_EPILOGUES,
                                                                analyzer.BITS)
        doc = overlay(row, F, probs, inputs_of(a.root, F, ARCH_OF_CC[int(row['cc'])]), own, AF, rprobs)
        if not doc['entries']:
            continue  # nothing priced for the class: no file
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
    if not a.only and os.path.isdir(out_dir):
        extra = sorted(set(n for n in os.listdir(out_dir) if n.endswith('.json')) - expected)
        if a.check:
            stale += extra
        else:
            # A class the facts no longer give (a span or carry change): its file goes.
            for name in extra:
                os.remove(os.path.join(out_dir, name))
                print('%s: removed (no class of the facts)' % name)
    if stale:
        print('routes/pick is not what the facts give (regenerate: python3 tools/route_pick/gen.py): ' + ', '.join(stale))
        sys.exit(1)
    if a.check:
        print('routes/pick: %d classes, in step with the facts' % len(expected))


if __name__ == '__main__':
    main()
