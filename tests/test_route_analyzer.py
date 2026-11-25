"""The route analyzer (tools/route_pick/analyzer.py) on the repo's facts: for one board of every architecture the
analyzer routes (ROUTED_CC: sm_75, sm_86, sm_89, sm_120a; Volta keeps its HMMA authority, route_authorities.cpp
volta_hmma, until its substitutes' kernel facts are committed), every A16 release shape the route tables list (store,
add, SwiGLU) at T 1-8 has at least one priced route, every price is a positive finite number, and rank() lists the
priced routes cheapest first with pick() its head."""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'route_pick'))
import analyzer  # noqa: E402
import analyzer_facts  # noqa: E402


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
    return sorted(seen)


ROUTED_CC = (75, 86, 89, 120)


def boards(F):
    """The first gpu_facts row of each compute capability of ROUTED_CC; every one must have the analyzer's facts."""
    out = {}
    for row in F['rows']:
        if row.get('cc') and int(row['cc']) in ROUTED_CC and not row.get('mma_slow'):
            assert analyzer.card_of(row, F) is not None, 'no analyzer facts for %s' % row.get('name')
            out.setdefault(int(row['cc']), row)
    return [out[cc] for cc in sorted(out)]


def main():
    F = analyzer_facts.load_facts(ROOT)
    shapes = release_shapes()
    bs = boards(F)
    assert shapes, 'no release shapes'
    assert sorted(int(b['cc']) for b in bs) == sorted(ROUTED_CC), 'a routed architecture without a board'
    failures = 0
    for g in bs:
        for fmt, ep, n, k in shapes:
            for T in range(1, 9):
                shape = dict(format=fmt, epilogue=ep, n=n, k=k, T=T)
                ranked = analyzer.rank(g, F, shape)
                prices = [analyzer.price_ns(g, F, shape, r) for r in ranked]
                ok = bool(ranked) and all(p is not None and p > 0 and math.isfinite(p) for p in prices) \
                    and prices == sorted(prices) and analyzer.pick(g, F, shape) == ranked[0]
                if not ok:
                    failures += 1
                    if failures <= 10:
                        print('FAIL', g.get('name'), shape, ranked[:3], prices[:3])
    print('boards %d, shapes %d, T 1-8: %d failures' % (len(bs), len(shapes), failures))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
