"""analyzer_facts.py: the facts analyzer.py prices from, read from one source tree (`root`, the repo) and nothing else.

    F = load_facts(root)   {'gpus': {name: row}, 'rows': [gpu_facts rows], 'archs': [SM limit rows],
                            'isa': {arch: numbers}, 'kf': {arch: {family: [rows]}}, 'mmq': (rows, K)}

An arch enters F['isa'] only with every fact the price reads (REQUIRED below): a missing one leaves the arch out, and
analyzer.card_of then gives None for its cards (no price) rather than a price that fails or guesses.

Files: src/core/data/gpu_facts.json, isa_facts.json (the isa_probe and launch facts), sm_rate_facts.json (the DRAM
latency L0 and the warp's fill rate, warp_inflight), src/ops/generic/data/kernel_facts/sm_<arch>.txt (rmma, r8, mmq,
vec, vecmma, and on sm_70 the Volta substitutes vhmma / vgemv / vgemvr).
NARROW below holds the narrow-load facts per arch (probe narrow.cu, #1183) until they are in isa_facts.json.
"""
import json
import os
import re

DATA = 'src/core/data'
KERNEL_FACTS = 'src/ops/generic/data/kernel_facts'
FAMILIES = ('rmma', 'r8', 'mmq', 'vec', 'vecmma', 'vhmma', 'vgemv', 'vgemvr')
EXTRA = dict(l1lat='l1_latency_w1', l2lat_ns='l2_latency_ns_w1', int_cyc='int32_cycles_per_warp_instruction',
             fp_cyc='fp32_cycles_per_warp_instruction', shfl='lat_shfl', smem_lat='shared_latency',
             l1cyc='l1_ldg32_cycles_per_sm', l2bpc='l2_bytes_per_sm_clock')
# Probe narrow.cu (analyzer_cov/narrow/nr_l*.out, 24.11): clocks of a warp load at (4 B a lane, 4 B stride) = the issue
# cap; a wavefront's clocks from (16, 16) over its four wavefronts; the bank width whose wavefront count matches (1, 26).
# Equal on every card of an arch (sm_86: 3060 / 3090; sm_89: 4060 Ti / 4070 Ti / 4090; sm_120a: 5070 Ti / 5090).
MMQ_HEADER = 'src/ops/generic/ggml_mmq/ggml_mmq.h'
# The facts every price reads: an arch without one of them is left out (no price for its cards).
REQUIRED = ('l0', 'node', 'mma_lat', 'hmma', 'lat', 'fill', 'nr', 'l2trip') + tuple(EXTRA)
NARROW = {'70': (1.621, 1.027, 8), '75': (2.012, 2.000, 8), '86': (2.000, 1.019, 4), '89': (2.000, 1.019, 4),
          '120a': (2.000, 1.008, 4)}


def load_kernel_facts(path):
    """sm_<arch>.txt -> {family: [rows]}, a row a dict of strings by its section's columns."""
    out = {f: [] for f in FAMILIES}
    cols = None
    with open(path, encoding='utf-8') as f:
        for ln in f:
            if ln.startswith('columns ') or ln.startswith('family_columns '):
                cols = ln.split()[1:]
                # `local` twice: the local-memory frame bytes after `stack`, then the local instructions (`local_ops`).
                if cols.count('local') == 2:
                    cols[len(cols) - 1 - cols[::-1].index('local')] = 'local_ops'
            elif cols is not None and not ln.startswith('#'):
                q = ln.split()
                if not q or q[0] not in out or len(q) != len(cols) or cols[0] != 'family':
                    continue
                if q[0] == 'vec':                              # the vec kernels' table, not the other `type` tables
                    if cols[1] == 'type' and 'gated' in cols:
                        out['vec'].append(dict(zip(cols, q)))
                elif q[0] in ('vecmma', 'vhmma', 'vgemv', 'vgemvr') or (cols[1] in ('decoder', 'codec', 'type') and 'outside_stall' in cols):
                    out[q[0]].append(dict(zip(cols, q)))
    return out


def load_facts(root):
    data = os.path.join(root, DATA)
    with open(os.path.join(data, 'gpu_facts.json'), encoding='utf-8') as f:
        g = json.load(f)
    with open(os.path.join(data, 'isa_facts.json'), encoding='utf-8') as f:
        isa = json.load(f)['arches']
    with open(os.path.join(data, 'sm_rate_facts.json'), encoding='utf-8') as f:
        rate = json.load(f)['arches']
    val = lambda v: v.get('value') if isinstance(v, dict) else v
    F = dict(gpus={}, archs=g.get('archs', []), isa={}, kf={}, rows=g.get('gpus', []))
    for row in g.get('gpus', []):
        F['gpus'].setdefault(str(row.get('name', '')).replace('GeForce ', ''), row)
    for arch, a in isa.items():
        fa, la = a.get('facts', {}), a.get('launch', {}).get('facts', {})
        path = os.path.join(root, KERNEL_FACTS, 'sm_%s.txt' % arch)
        if arch not in rate or 'node_ns' not in la or not os.path.exists(path):
            continue
        rf = rate[arch]['facts']
        d = dict(l0=val(rf['dram_stream_latency_ns']), node=val(la['node_ns']), mma_lat=val(fa['mma_dependent_latency']),
                 hmma=val(fa['mma_cycles_per_partition']), imma_n=val(fa.get('imma_narrow_cycles_per_partition')),
                 imma_w=val(fa.get('imma_wide_cycles_per_partition')), pdl=val(la.get('node_pdl_ns')) is not None,
                 lat=val(fa['lat_imad']))
        for k, name in EXTRA.items():
            d[k] = val(fa[name])
        # The L2 round trip of four warps (the stream-k fixup's walk), the one-warp trip where the probe has no w4 row.
        d['l2trip'] = val(fa.get('l2_latency_w4') or fa.get('l2_latency_w1'))
        d['fill'] = val(rf['warp_fill_bytes_per_clock']) if 'warp_fill_bytes_per_clock' in rf else None
        lsms = a.get('launch', {}).get('sms')
        d['stream'] = val(la['stream_ns']) if 'stream_ns' in la else None
        d['floor'] = sorted((n, val(la[k])) for n, k in ((1, 'floor_ns_1'), (lsms, 'floor_ns_sms'), (4 * lsms if lsms else None, 'floor_ns_4sms'),
                                                         (1024, 'floor_ns_1024'), (4096, 'floor_ns_4096')) if n and k in la)
        if arch in NARROW:
            d['nr'] = dict(zip(('cap', 'wf', 'bank'), NARROW[arch]))
        if any(d.get(f) is None for f in REQUIRED) or len(d['floor']) < 2:
            continue
        F['isa'][arch] = d
        F['kf'][arch] = load_kernel_facts(path)
    F['mmq'] = mmq_multiples(root)
    return F


def mmq_multiples(root):
    """ggml_mmq::kRowMultiple and kKMultiple from the tree's header (mmq_call / mmq_admits take a shape in whole tiles)."""
    with open(os.path.join(root, MMQ_HEADER), encoding='utf-8') as f:
        text = f.read()
    out = []
    for name in ('kRowMultiple', 'kKMultiple'):
        m = re.search(r'inline constexpr int %s\s*=\s*(\d+)\s*;' % name, text)
        if not m:
            raise SystemExit(MMQ_HEADER + ': no ' + name)
        out.append(int(m.group(1)))
    return tuple(out)
