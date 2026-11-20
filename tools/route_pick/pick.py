"""pick.py: the route analyzer (analyzer core after round 11; tools/route_pick/README.md), counted from repo facts.

    pick(gpu_row, F, shape, among=None) -> route name       the lowest price; T 1..16, T 17..32 where handover() is False
    price_ns(gpu_row, F, shape, route)  -> ns | None        None: not priced, never picked

`F` is facts.load_facts(root), `gpu_row` a gpu_facts.json row, `shape` dict(format, epilogue, n, k, T). Every input is a
repo fact: the card's SMs, clock, bus and cc; its architecture's registers and shared memory an SM, DRAM latency L0,
graph-node time, MMA latency and cycles, IMMA cycles, dependent ALU latency (lat_imad), PDL; a kernel's registers, K
step, MMA, integer and instruction counts, load kind and stall clocks outside the K loop. The only tables are names.

The price, ns (node = one graph node; pre = node + L0, the activation pre-pass; out = lat_imad x outside_stall /
clock, the first round's times PDL on a PDL architecture):
  rmma      node + rounds of resident CTAs (SMs x CTAs the registers allow); a round = sqrt(SM time^2 + DRAM time^2)
              SM time   = K trips x tau + out,  tau = the largest of
                            (2 - latency) x L0 + MMAs x MMA latency / clock                   one warp's own chain
                            MMS x warps on the SM x MMAs x MMA cycles / 4 / clock             the SM's tensor throughput
                            red x warps on the SM x int / 4 / clock + (1 - red) x own chain   the repeated dequant,
                              red = (passes - 1) / passes: the share of the work that decodes the same weights again
              DRAM time = round's bytes / bus x max(1, rows open on the busiest SM / L1) ^ (G + GN x narrow)
              bytes: the weights once, and once more for every later column pass not sharing its reads in flight
              the row priced: the latency instance where the product launches it (n_tiles 1, the row exists,
              DRAM bytes an SM clock >= 5.5, 4.5 for SwiGLU, cc >= 75), else the plain row
              the pair (round 11): q8 n4.w1x8 = the price of n2.w1x4 x the ratio of the two with the issue bound
              in tau (one instruction a clock a partition: warps on the SM x the K loop's instructions / 4)
  ggml_mmq  K_MMQ x [ node + pre + waves x (units a CTA x (8 warps x IMMAs x cycles / 4 / clock + unit bytes x SMs / bus) + out) ]
  r8        K x [ node + pre (T > 1) + waves x (bus time of the resident rows + out + quantizer stall at T 1) ],  K = K_R8Q at T 1
  ggml_vec  K_VEC x [ node + pre + weight bytes / bus + lat_imad x L0 ]
  family factor (round 10): r8 at T 1, ggml_vec at T 2-4 and ggml_mmq at T 17-32 are multiplied by
              exp(sum(w x feature) / w_margin), a sign classifier against the cheapest rmma route written as a factor.
FITTED, none of them here: P (round 9; K_VEC is round 8's) and FAMILY_W (round 10) are copied unchanged.
Not priced: w4a8*, tiled, sm_70, r8 and ggml_vec above T 8, T > 16 on a compute capability outside VERIFIED_T17.
"""
import math
import re

P = dict(G=0.1240,       # exponent of the DRAM slowdown with the rows open on one SM
         GN=0.0733,      # added exponent where a K step takes a sector-sized piece of a row (q8)
         MMS=0.5979,     # share of the SM's counted tensor-throughput bound that binds
         K_R8Q=0.5958,   # scale of the self-quantizing r8 instance (T 1)
         K_MMQ=1.0505,   # scale of ggml_mmq
         PDL=0.4594,     # share of the first round's outside-loop cost left on a PDL architecture
         K_VEC=1.1269)   # scale of ggml_vec
FAMILY_W = {             # family: (T band, weights); features: 1, ln n, ln k, T / 16, bits, PDL, ln(bus bytes an SM clock)
    'r8': ((1, 1), dict(one=4.7569, margin=-4.6870, ln=1.2825, lk=1.9693, T16=0.2973, bits=-3.0877, pdl=-6.7183, lbpc=-5.8212)),
    'ggml_vec': ((2, 4), dict(one=5.4184, margin=-18.8925, ln=0.3836, lk=1.5004, T16=-18.6269, bits=-1.9643, lbpc=-5.2504)),
    'ggml_mmq': ((17, 32), dict(one=2.6797, margin=-24.6209, pdl=1.4861)),
}
L1_MIN_KIB = 28.0                  # SET: the L1 an SM keeps beside its shared memory
VERIFIED_T17 = (86, 89, 120)       # guard: compute capabilities whose T 17-32 pick was scored on an open grid
PARTITIONS = 4.0                   # tensor partitions of an SM
PAIR = {'rmma.m1.n4.w1x8': ('q8', 'rmma.m1.n2.w1x4')}   # route: (format, the pair member its price is taken from)
MMQ_WARPS = 8.0                    # warps of one ggml_mmq CTA
MMQ_FILL = 0.9                     # ggml_mmq_internal.cuh: a tile a CTA where the grid fills its last wave to 90 %
# generic_gemm_internal.h kRmmaLatencyOneSetBytes (store and add: one weight set, generic_rmma_describe.cpp) / SwiGluBytes
LAT_BYTES = {'store': 5.5, 'add': 5.5, 'swiglu': 4.5}
VEC_KS = (5120, 6144, 17408)       # ggml_vec.cu vec_launch_k: the K values with an unrolled kernel (5120 alone when gated)

BITS = {'q4k': 4.5, 'q5k': 5.5, 'q6k': 6.5625, 'q8': 8.5, 'q3k': 3.4375, 'iq3s': 3.4375, 'iq4xs': 4.25}
RMMA_DEC = {'q4k': 'GgmlQ45KQuadDecoder<4>', 'q5k': 'GgmlQ45KQuadDecoder<5>', 'q6k': 'GgmlQ6KDecoder', 'q8': 'Q8Decoder',
            'q3k': 'GgmlQ3KDecoder', 'iq3s': 'GgmlIq3SDecoder', 'iq4xs': 'GgmlIq4XsDecoder'}
R8_CODEC = {'q4k': 'W4a8Q45K<4>', 'q5k': 'W4a8Q45K<5>', 'q6k': 'W4a8Q6K', 'q8': 'W4a8Q8'}
MMQ_TYPE = {'q4k': '12', 'q5k': '13', 'q6k': '14', 'q8': '8', 'q3k': '11', 'iq3s': '21', 'iq4xs': '23'}   # ggml_type
EPI = {'store': '0', 'add': '1', 'swiglu': '2'}   # the kernel facts' epilogue column
_RMMA = re.compile(r'rmma\.m(\d+)\.n(\d+)\.w(\d+)x(\d+)(?:\.c(\d+))?$')
_R8 = re.compile(r'r8\.w(\d+)r(\d+)$')
_CACHE = {}


def _num(v, d=0.0):
    try:
        x = float(v)
        return d if x != x else x
    except (TypeError, ValueError):
        return d


def kf_row(kf, fam, **want):
    """The first kernel-facts row of `fam` whose key fields equal `want`, or None."""
    key = (id(kf), fam, tuple(sorted(want.items())))
    if key not in _CACHE:
        _CACHE[key] = next((r for r in kf[fam] if all(r.get(k) == str(v) for k, v in want.items())
                            and (fam == 'vec' or _num(r.get('loops', '1'), 1.0) > 0)), None)
    return _CACHE[key]


def card_of(gpu, F):
    """The numbers of a gpu_facts.json row the price uses, or None for a card on an architecture without facts."""
    key = (id(F), gpu.get('id'), gpu.get('name'))
    if key not in _CACHE:
        cc = int(gpu['cc'])
        arch = next((a for a in (str(cc), '%da' % cc) if a in F['isa']), None)       # cc 120 -> the sm_120a snapshot
        lim = next((r for r in F['archs'] if r['cc'] != 0 and r['cc'] <= cc <= r.get('cc_to', r['cc'])),
                   next((r for r in F['archs'] if r['cc'] == 0), None))
        c = None
        if arch is not None:
            sms, clk = float(gpu['sms']), float(gpu['boost_mhz']) * 1e-3              # SM clock, cycles a ns
            bus = float(gpu['bus_bits']) * float(gpu['gbps']) / 8.0                   # DRAM bus, bytes a ns
            c = dict(F['isa'][arch], cc=cc, kf=F['kf'][arch], sms=sms, clk=clk, bus=bus, bpc=bus / (sms * clk),
                     regs_sm=int(lim['registers_per_sm']), l1=(float(lim['smem_per_sm_kib']) + L1_MIN_KIB) * 1024.0)
        _CACHE[key] = c
    return _CACHE[key]


def occupancy(c, warps, regs):
    """CTAs of a kernel one SM holds at once: its registers (a warp takes them in units of 256)."""
    return max(float((c['regs_sm'] // (math.ceil(regs * 32.0 / 256.0) * 256)) // warps), 1.0)


def price_rmma(c, P, r, n, k, T, pitch, M, N, A, B, C, issue=False):
    sms, clk, l0, bus = c['sms'], c['clk'], c['l0'], c['bus']
    passes = math.ceil(T / (8.0 * N))                      # column passes of an n_tiles-wide tile
    rows_cta, warps = 16.0 * M * A, A * B
    ctas = math.ceil(n / rows_cta) * passes * C
    occ = occupancy(c, warps, _num(r['registers']))
    kiter, slots = _num(r['k_iter'], 256.0), sms * occ
    trips = k / (B * C * kiter)
    nfull = math.floor(ctas / slots)
    tail = ctas - nfull * slots
    cp = ctas / passes
    share = min(max((slots - cp) / max(ctas - cp, 1e-9), 0.0), 1.0) if passes > 1 else 1.0
    bcta = n * pitch * (1.0 + (passes - 1.0) * (1.0 - share)) / ctas       # DRAM bytes one CTA draws
    mma, out = _num(r['mma']), c['lat'] * _num(r['outside_stall']) / clk
    own = (2.0 - _num(r['latency'])) * l0 + mma * c['mma_lat'] / clk       # one K trip of a warp: two steps, an L0 each
    out1 = out * (P['PDL'] if c['pdl'] else 1.0)
    narrow = 0.0 if _num(r.get('ld_bytes')) > 0 else (256.0 / max(kiter, 1.0) - 1.0) / 3.0    # 1 at 32 K values a step,
    gexp, red = P['G'] + P['GN'] * narrow, (passes - 1.0) / passes         # 0 at 128 and for the generic LD loads

    def rnd(cps, nc, o):                                   # a round of nc CTAs, cps of them on the busiest SM
        dram = nc * bcta / bus * max(1.0, cps * rows_cta * pitch / c['l1']) ** gexp
        tau = max(own, P['MMS'] * cps * warps * mma * c['hmma'] / PARTITIONS / clk,
                  red * cps * warps * _num(r['int']) / PARTITIONS / clk + (1.0 - red) * own,
                  cps * warps * _num(r['total']) / PARTITIONS / clk if issue else 0.0)      # the issue bound: pair ratios only
        sm = trips * tau + o
        return math.sqrt(sm * sm + dram * dram)

    part = lambda x: max(min(math.ceil(x / sms), occ), 1.0)
    if nfull < 1:
        return c['node'] + rnd(part(ctas), tail, out1)
    return c['node'] + rnd(occ, slots, out1) + (nfull - 1.0) * rnd(occ, slots, out) + (rnd(part(tail), tail, out) if tail > 0 else 0.0)


def price_other(c, P, fam, r, n, k, T, bits, sets, Wp=0, R=0):
    sms, clk, l0, bus, pitch = c['sms'], c['clk'], c['l0'], c['bus'], sets * k * bits / 8.0
    pre = c['node'] + l0                                   # the activation pre-pass: a graph node and a DRAM latency
    out = c['lat'] * _num(r['outside_stall']) / clk if r else 0.0
    first = out * ((P['PDL'] if c['pdl'] else 1.0) - 1.0)  # what the first wave saves under PDL (0 or negative)
    if fam == 'ggml_vec':
        return P['K_VEC'] * (c['node'] + pre + n * pitch / bus + c['lat'] * l0)
    if fam == 'ggml_mmq':                                  # 128-row tiles; one CTA an SM, or a tile a CTA (the fill rule)
        nt = math.ceil(n / 128.0)
        tiles = nt * math.ceil(k / 256.0)
        ctas = tiles if tiles < sms else nt if nt / (math.ceil(nt / sms) * sms) >= MMQ_FILL else sms
        per = nt * (k / 256.0) * sets * math.ceil(T / float(mmq_j(T))) / ctas
        cyc = c['imma_w'] if '16832' in r.get('mma_shape', '') else c['imma_n']
        unit = per * (MMQ_WARPS * _num(r['mma']) * cyc / PARTITIONS / clk + 4096.0 * bits * sms / bus)
        return P['K_MMQ'] * (c['node'] + pre + math.ceil(ctas / sms) * (unit + out) + first)
    ctas = math.ceil(n / float(Wp * R))                    # r8
    slots = sms * occupancy(c, Wp, _num(r['registers']))
    cps = math.ceil(ctas / sms) if ctas <= slots else slots / sms
    pro = k / 1024.0 / Wp * _num(r.get('pro_stall')) / clk if T <= 1 else 0.0        # the CTA quantizes the activation
    t = c['node'] + (0.0 if T <= 1 else pre) + math.ceil(ctas / slots) * (cps * Wp * R * pitch * sms / bus + out + pro) + first
    return t * (P['K_R8Q'] if T <= 1 else 1.0)


def mmq_j(T):
    return 16 if T <= 16 else 32 if T <= 32 else 64 if T <= 64 else 128


def handover(gpu, F, T):
    c = card_of(gpu, F)
    return c is None or (float(T) > 16 and c['cc'] not in VERIFIED_T17)


def price_ns(gpu, F, shape, route, P=P, W=FAMILY_W):
    fmt, anchor = PAIR.get(route, (None, None))
    if fmt == str(shape['format']).lower():                # the pair's ratio by the issue bound, on the anchor's price
        t = [_price(gpu, F, shape, r, P, W, i) for r, i in ((anchor, False), (route, True), (anchor, True))]
        if None not in t:
            return t[0] * t[1] / t[2]
    return _price(gpu, F, shape, route, P, W)


def _price(gpu, F, shape, route, P, W, issue=False):
    c, fmt, ep = card_of(gpu, F), str(shape['format']).lower(), str(shape.get('epilogue', 'store')).lower()
    n, k, T = float(shape['n']), float(shape['k']), max(float(shape.get('T', 1)), 1.0)
    if handover(gpu, F, T) or fmt not in BITS:
        return None
    kf, bits, sets = c['kf'], BITS[fmt], 2.0 if ep == 'swiglu' else 1.0   # SwiGLU: two weight sets
    m = _RMMA.match(route)
    if m:
        M, N, A, B, C = (int(x or 1) for x in m.groups())
        key = dict(decoder=RMMA_DEC.get(fmt), warps_m=A, warps_k=B, m_tiles=M, n_tiles=N, cluster=C, epilogue=EPI.get(ep, '0'), sums=0)
        r = kf_row(kf, 'rmma', latency=0, **key)
        if r is not None and N == 1 and c['bpc'] >= LAT_BYTES.get(ep, 5.5) and c['cc'] >= 75:
            r = kf_row(kf, 'rmma', latency=1, **key) or r                  # the latency instance is the one launched
        return None if r is None else price_rmma(c, P, r, n, k, T, sets * k * bits / 8.0, M, N, A, B, C, issue)
    (fam, Wp, R), m = (route, 0, 0), _R8.match(route)
    if m:
        fam, Wp, R = 'r8', int(m.group(1)), int(m.group(2))
        r = None if T > 8 else kf_row(kf, 'r8', codec=R8_CODEC.get(fmt), warps=Wp, rows=R, nt=1 if T <= 1 else 2 if T <= 2 else 4 if T <= 4 else 8,
                                      epilogue=EPI.get(ep, '0'), pre=0 if T <= 1 else 1)
    elif route == 'ggml_mmq':
        r = kf_row(kf, 'mmq', type=MMQ_TYPE[fmt], J=mmq_j(T), fallback=0)
    elif route == 'ggml_vec':
        gated = ep == 'swiglu'
        ks = int(k) if int(k) in VEC_KS and (not gated or int(k) == 5120) else 0
        r = None if T > 8 or fmt not in R8_CODEC else kf_row(kf, 'vec', type=MMQ_TYPE[fmt], T=int(T), KS=ks, gated=int(gated), bounded=0)
    else:
        return None
    if r is None:
        return None
    t = price_other(c, P, fam, r if fam != 'ggml_vec' else None, n, k, T, bits, sets, Wp, R)
    (lo, hi), w = W.get(fam, ((0, 0), None))
    if w is not None and lo <= T <= hi:                    # the family factor
        x = dict(one=1.0, ln=math.log(n), lk=math.log(k), T16=T / 16.0, bits=bits, pdl=float(c['pdl']), lbpc=math.log(c['bpc']))
        t *= math.exp(sum(w[name] * x[name] for name in w if name != 'margin') / w['margin'])
    return t


def candidates(gpu, F, shape):
    """Every route name this file can price for the shape: the schedules that have a kernel-facts row."""
    c = card_of(gpu, F)
    if c is None:
        return []
    T, fmt = max(float(shape.get('T', 1)), 1.0), str(shape['format']).lower()
    names = set(['ggml_mmq'] if T >= 5 else []) | set(['ggml_vec'] if 2 <= T <= 4 else [])
    for r in c['kf']['rmma']:
        # Thread-block clusters are sm_90+ (generic_select.cpp rmma_index: below it a cluster
        # schedule runs as its C == 1 twin), so a cluster row is no candidate there.
        if r['cluster'] != '1' and c['cc'] < 90:
            continue
        if r['decoder'] == RMMA_DEC.get(fmt) and r['latency'] == '0' and r['sums'] == '0':
            names.add('rmma.m%s.n%s.w%sx%s%s' % (r['m_tiles'], r['n_tiles'], r['warps_m'], r['warps_k'],
                                                  '' if r['cluster'] == '1' else '.c' + r['cluster']))
    for r in c['kf']['r8'] if T <= 8 else ():
        if r['codec'] == R8_CODEC.get(fmt):
            names.add('r8.w%sr%s' % (r['warps'], r['rows']))
    return sorted(r for r in names if price_ns(gpu, F, shape, r) is not None)


def pick(gpu_row, F, shape, among=None):
    """The route with the lowest price; ties go to the earlier name in sorted order."""
    names = sorted(among) if among is not None else candidates(gpu_row, F, shape)
    priced = [(t, r) for t, r in ((price_ns(gpu_row, F, shape, r), r) for r in names) if t is not None]
    return min(priced)[1] if priced else None

