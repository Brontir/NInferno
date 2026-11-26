"""The route analyzer (tools/route_pick/analyzer.py) on the repo's facts: for one board of every architecture the
analyzer routes (ROUTED_CC: sm_75, sm_86, sm_89, sm_120a; Volta keeps its HMMA authority, route_authorities.cpp
volta_hmma, until its substitutes' kernel facts are committed), every A16 release shape the route tables list (store,
add, SwiGLU) at T 1-8 has at least one candidate route (analyzer.candidates: the launchable routes it prices) and every
price is a positive finite number. A card of an architecture without facts gets no price, not an error."""
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
                prices = {r: analyzer.price_ns(g, F, shape, r) for r in analyzer.candidates(g, F, shape)}
                if not prices or not all(p > 0 and math.isfinite(p) for p in prices.values()):
                    failures += 1
                    if failures <= 10:
                        print('FAIL', g.get('name'), shape, sorted(prices.items())[:3])
    # The cluster cap in whole clusters (#1228, #1234): the RTX 5090's measured slots (C 2: 340, C 8: 328 = 41
    # clusters) and the RTX 5070 Ti's own launch (clocc: C 8 17 clusters, 136 CTAs) from the 5090's facts.
    for name, sms, c2, c8 in (('RTX 5090', 170, 340, 328), ('RTX 5070 Ti', 70, 140, 136)):
        card = analyzer.card_of(F['gpus'][name], F)
        assert int(card['sms']) == sms, name
        for C, want in ((2, c2), (8, c8)):
            got = analyzer.cluster_cap(card, C) * card['sms']
            if abs(got - want) > 1e-9:
                failures += 1
                print('FAIL cluster cap', name, C, got, want)
    # A card of an architecture without the analyzer's facts (sm_80 here): no price, no candidate, no error.
    no_facts = dict(bs[0], cc=80)
    one = dict(format='q4k', epilogue='store', n=4096, k=4096, T=1)
    assert analyzer.card_of(no_facts, F) is None
    assert analyzer.price_ns(no_facts, F, one, 'ggml_vec') is None and analyzer.rank(no_facts, F, one) == []
    # The product's refusals: an rmma route at a K off its codec's step (rmma_refusal Extent; Q4_K's is 32), and a cluster
    # schedule below sm_90.
    sm89 = next(b for b in bs if int(b['cc']) == 89)
    odd = dict(format='q4k', epilogue='store', n=4096, k=5200, T=4)
    assert analyzer.price_ns(sm89, F, odd, 'rmma.m1.n1.w1x8') is None
    assert not any(r.startswith('rmma.') for r in analyzer.candidates(sm89, F, odd))
    assert analyzer.price_ns(sm89, F, dict(odd, k=4096), 'rmma.m1.n1.w1x8') is not None
    # A K of whole 32-value steps but not of whole Q4_K superblocks (supports(): K % 256): no route at all.
    assert analyzer.price_ns(sm89, F, dict(odd, k=5216), 'rmma.m1.n1.w1x8') is None
    assert analyzer.rank(sm89, F, dict(odd, k=5216)) == []
    # R8's own K step (format_traits.h r8_k_values, 32 for the groupwise q4): a padded q4 weight takes any K, R8 does not.
    q4 = dict(format='q4', epilogue='store', n=4096, k=4112, T=2)
    assert analyzer.price_ns(sm89, F, q4, 'r8.w4r1') is None and analyzer.price_ns(sm89, F, dict(q4, k=4096), 'r8.w4r1') is not None
    # r8_fits at T 1: a K whose activation passes 48 KB of shared memory (K 49152 x 36 / 32 = 54 KB).
    assert analyzer.price_ns(sm89, F, dict(q4, k=49152, T=1), 'r8.w4r1') is None
    assert analyzer.price_ns(sm89, F, dict(odd, k=4096), 'rmma.m1.n1.w1x8.c2') is None
    # latency_pays (generic_rmma_describe.cpp): on sm_75 SwiGLU's latency instance is the activation ring's form only, a CTA
    # of at most 128 threads; elsewhere the instance launches where it keeps the plain kernel's waves.
    sm75 = next(b for b in bs if int(b['cc']) == 75)
    c75 = analyzer.card_of(sm75, F)
    same = dict(registers='128')
    assert c75['bpc'] >= analyzer.LAT_BYTES['swiglu'], 'the sm_75 board is not latency-bound for SwiGLU'
    assert analyzer.latency_launched(c75, 'swiglu', 1, 4, 1, 12288, same, same)
    assert not analyzer.latency_launched(c75, 'swiglu', 1, 8, 1, 12288, same, same)
    assert not analyzer.latency_launched(c75, 'swiglu', 1, 4, 1, 12288, same, None)
    # A latency kernel that holds fewer CTAs an SM launches only while the grid fits as many waves (SwiGLU w1x4 on the sm_75
    # board, whose bytes a cycle pass SwiGLU's bound: asserted above, so these cannot pass at the gate).
    big = dict(registers='255')
    sms = c75['sms']
    assert analyzer.occupancy(c75, 4, 255.0) < analyzer.occupancy(c75, 4, 128.0), 'the 255-register kernel holds as many CTAs'
    one_wave = int(16 * sms * analyzer.occupancy(c75, 4, 255.0))
    assert analyzer.latency_launched(c75, 'swiglu', 1, 4, 1, one_wave, same, big)
    assert not analyzer.latency_launched(c75, 'swiglu', 1, 4, 1, 16 * sms * int(analyzer.occupancy(c75, 4, 128.0)), same, big)
    print('boards %d, shapes %d, T 1-8: %d failures' % (len(bs), len(shapes), failures))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
