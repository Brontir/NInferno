"""Engine vs HF reference: dNLL, top-1 agreement and greedy divergence.

usage: python3 tools/moe_quality/compare.py <ref dir> <engine dir> <tag> [names...]
ref:    <tag>_<name>.ref  = FP32 logp[n] | I32 argmax[n] | FP32 top1-top2 gap[n]
engine: <tag>_<name>.eng  = FP32 logp[n] | I32 argmax[n]  (ninferno-dense score --dump)
The engine side: ninferno-dense <artifact> score --tokens <tag>_<name>.i32 --window 2048 --dump ...
and ninferno-dense <artifact> generate --tokens <tag>_chat.i32 --decode 64 --out <tag>_chat.gen.
greedy: <ref>/<tag>_chat.greedy.json vs <engine>/<tag>_chat.gen (int32 tokens)
"""
import json
import os
import sys

import numpy as np


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit('usage: python3 tools/moe_quality/compare.py <ref dir> <engine dir> <tag> [names...]')
    ref_dir, eng_dir, tag = sys.argv[1:4]
    names = sys.argv[4:] or ['wiki1', 'wiki2', 'code', 'chatlong']
    rows = []
    for name in names:
        r = np.fromfile(f'{ref_dir}/{tag}_{name}.ref', dtype='<u4')
        n = len(r) // 3
        r_lp = r[:n].view('<f4'); r_arg = r[n:2 * n].view('<i4'); r_gap = r[2 * n:].view('<f4')
        e = np.fromfile(f'{eng_dir}/{tag}_{name}.eng', dtype='<u4')
        m = len(e) // 2
        e_lp = e[:m].view('<f4'); e_arg = e[m:].view('<i4')
        k = min(n, m)
        d = e_lp[:k].astype(np.float64) - r_lp[:k].astype(np.float64)
        agree = e_arg[:k] == r_arg[:k]
        dis = ~agree
        rows.append({
            'name': name, 'tokens': int(k),
            'ref_nll': float(-r_lp[:k].mean()), 'eng_nll': float(-e_lp[:k].mean()),
            'dnll': float(-(d.mean())), 'mean_abs_dlogp': float(np.abs(d).mean()),
            'max_abs_dlogp': float(np.abs(d).max()),
            'top1_agree': float(agree.mean()),
            'disagree_ref_gap_median': float(np.median(r_gap[:k][dis])) if dis.any() else None,
            'disagree_ref_gap_max': float(r_gap[:k][dis].max()) if dis.any() else None,
        })
    out = {'scored': rows}
    g_ref = f'{ref_dir}/{tag}_chat.greedy.json'
    g_eng = f'{eng_dir}/{tag}_chat.gen'
    if os.path.exists(g_ref) and os.path.exists(g_eng):
        with open(g_ref, encoding="utf-8") as f:
            ref = json.load(f)
        eng = np.fromfile(g_eng, dtype='<i4').tolist()
        # ninferno-dense generate returns the n tokens it fed: the first is the prefill's draw.
        div = next((i for i, (a, b) in enumerate(zip(ref['tokens'], eng)) if a != b), None)
        out['greedy'] = {
            'compared': min(len(ref['tokens']), len(eng)),
            'first_divergence': div,
            'ref_gap_at_divergence': ref['gaps'][div] if div is not None else None,
            'ref_tokens_head': ref['tokens'][:16], 'eng_tokens_head': eng[:16],
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
