"""facts.py: the repo facts pick.py prices from, read from one source tree.

    F = load_facts(root)   {'gpus': {name: gpu_facts.json row}, 'archs': [SM limit rows], 'isa': {arch: numbers},
                            'kf': {arch: kernel facts by family}}

Files: src/core/data/gpu_facts.json, isa_facts.json (the isa_probe facts, lat_imad among them, and the launch
facts), sm_rate_facts.json, src/ops/generic/data/kernel_facts/sm_<arch>.txt. An architecture without the int8-MMA
and launch-node facts (sm_70) is left out: nothing on it is priced.
"""
import json
import os

DATA = 'src/core/data'
KERNEL_FACTS = 'src/ops/generic/data/kernel_facts'
FILES = ('gpu_facts.json', 'isa_facts.json', 'sm_rate_facts.json')


def load_kernel_facts(path):
    """sm_<arch>.txt -> {'rmma': [rows], 'r8': [...], 'mmq': [...], 'vec': [...]}, a row a dict of strings by column."""
    out = {'rmma': [], 'r8': [], 'mmq': [], 'vec': []}
    cols = None
    with open(path, encoding='utf-8') as f:
        for ln in f:
            if ln.startswith('columns ') or ln.startswith('family_columns '):
                cols = ln.split()[1:]
            elif cols is not None and not ln.startswith('#'):
                q = ln.split()
                if not q or q[0] not in out or len(q) != len(cols) or cols[0] != 'family':
                    continue
                if q[0] == 'vec':                              # the vec kernels' table, not the other `type` tables
                    if cols[1] == 'type' and 'gated' in cols:
                        out['vec'].append(dict(zip(cols, q)))
                elif cols[1] in ('decoder', 'codec', 'type') and 'outside_stall' in cols:
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
        if arch not in rate or 'imma_narrow_cycles_per_partition' not in fa or 'node_ns' not in la or not os.path.exists(path):
            continue
        F['isa'][arch] = dict(l0=val(rate[arch]['facts']['dram_stream_latency_ns']), node=val(la['node_ns']),
                              mma_lat=val(fa['mma_dependent_latency']), hmma=val(fa['mma_cycles_per_partition']),
                              imma_n=val(fa['imma_narrow_cycles_per_partition']),
                              imma_w=val(fa['imma_wide_cycles_per_partition']), pdl=val(la.get('node_pdl_ns')) is not None,
                              lat=val(fa['lat_imad']))
        F['kf'][arch] = load_kernel_facts(path)
    return F


def input_paths(root, F):
    """Every file the facts were read from, relative to `root`, sorted: what a generated overlay depends on."""
    out = [os.path.join(DATA, f) for f in FILES]
    out += [os.path.join(KERNEL_FACTS, 'sm_%s.txt' % a) for a in F['isa']]
    return sorted(p.replace(os.sep, '/') for p in out)
