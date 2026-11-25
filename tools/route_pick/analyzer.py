"""analyzer.py: the route analyzer of decode and verify widths (T 1-8): it prices every route of one call from counted
chip and kernel facts and the cheapest wins (analyzer rounds 15-20; the lab's copy prices
every cell of the exam grids the same). The older tools/route_pick/pick.py (round 11) still writes the T 9-32 bands.

    price_ns(gpu_row, F, shape, route) -> ns | None      None: not priced, never picked
    pick(gpu_row, F, shape, among=None) -> route name

`F` is analyzer_facts.load_facts(root), `gpu_row` a gpu_facts.json row, `shape` dict(format, epilogue, n, k, T).
Inputs, all repo facts: the card's SMs, boost clock, bus; its arch's register / thread / block limits, DRAM latency L0
and fill rate (sm_rate_facts), launch facts, MMA latency and cycles, ALU latencies, narrow-load facts; each kernel's
registers, K step, MMAs, loads and their bytes, instruction counts and static stall clocks (kernel_facts sm_<arch>.txt).

The price (ns):
  rmma      launch + rounds of the CTAs the SMs hold; a round = hypot(SM side, DRAM side).
            SM side = K trips x tau + the code outside the loop + the wave term (each warp's K step waits one K step's
            bytes of every row of the round at the bus rate, / warps_k); tau = the warp's chain (its DRAM round trips,
            each a window of the bytes it keeps in flight, + the loop's static stall clocks + an emulated MMA's latency on
            sm_70) or an issue bound. DRAM side = the round's bytes / bus x (rows open on the busiest SM / L1) ^ G.
  r8        launch + pre-pass + rounds; a round = hypot(the warp's chain: dp4a loop stall clocks + a DRAM window per
            global-wait generation, the quantizer's loop at T 1, the code outside; the round's bytes / bus).
  ggml_mmq  launch + pre-pass + waves of 128-row tiles; a unit = max(its MMA issue, its loop's stall clocks) + bytes / bus;
            the stream-k fixup where it runs (its launch, and its bytes at the L2 rate or its walk of L2 round trips);
            the epilogue kernel (its launch and bytes).
  ggml_vec  two launches (quantizer, product) + L0 + rounds of the persistent grid; a round = hypot(SM side: the loop's
            chain, the issue bound and the load unit by the narrow-load rule; the round's bytes / bus).
  sm_70     every rmma.* route is the Volta substitute the launch takes (volta_hmma_takes before volta_gemv_takes),
            priced from its own counted kernel (sections vhmma / vgemv / vgemvr of sm_70.txt; without them, no price).
FITTED: G alone. Not priced (None): the W4A8, tiled and staged tiles.
"""
import math
import re

# The analyzer's settings: G is FITTED (round 9's value), the rest choose a form, no number.
P = dict(G=0.1240,       # FITTED, the one number left at T 1-8 (round 9's): exponent of rmma's DRAM slowdown with the rows open on one SM
         LCH='s',        # a launch's time: 'g' a graph node's floor by its CTAs (the product's decode graph); 's' the same under a stream
                         # launch (route_tune times stream launches: the exam)
         LD='n',         # ggml_vec's loads by the narrow-load rule (the arch's issue cap, wavefront clocks and bank width)
         VEC_FORM='h',   # ggml_vec's round: hypot(SM side, bytes / bus)
         VEC_SM='iol',   # ggml_vec's SM side: the code outside the K loop ('o'), the issue bound ('i'), the load unit ('l')
         MMA_SM='io',    # ggml_vec.mma: the same SM side from its vecmma row
         XK=1,           # a route's second kernel (vec's quantizer, r8's / mmq's pre-pass) costs one more stream launch
         VSUB=1)         # on (1) / off (0). sm_70: every rmma.* route is the Volta substitute the launch takes, priced from its counted kernel
L1_MIN_KIB = 28.0                  # SET: the L1 an SM keeps beside its shared memory
PARTITIONS = 4.0                   # tensor partitions of an SM
# Counted (no fit): the SASS instructions of one emulated MMA call (sm_70, nvcc 12.8.93, generic_gemm_bf16.cu.o:
# mma_bf16_fma_call, 551 instructions: FFMA 120, MOV 146, IMAD 88, LOP3 84, SHF 32, SHFL 32, CALL 32 to the 16-instruction
# __cuda_sm70_shflsync_idx_p, whose bodies are not added). #1165.
EMU_CALL = {'HMMA.16816.F32.BF16.EMU': 551.0}
# Measured (few/bw/emu.cu, the product's call body (552 instructions, 32 shflsync calls, -rdc), Quadro GV100, 24.11):
# a warp's ns a call with 1 / 2 / 4 independent accumulator chains (563.2 / 437.0 / 377.6, warps_sm 1), and the issue
# cost a call, 4 partitions x 1627 MHz / the SM's saturated 18.86 calls a us (16 warps, 4 chains) = 345 instructions.
EMU_LAT_NS = {1: 563.2, 2: 437.0, 3: 437.0, 4: 377.6}
EMU_ISSUE = 4.0 * 1627.0 / 18.86
PAIR = {'rmma.m1.n4.w1x8': ('q8', 'rmma.m1.n2.w1x4')}   # route: (format, the pair member its price is taken from)
MMQ_WARPS = 8.0                    # warps of one ggml_mmq CTA
MMQ_FILL = 0.9                     # core/mmq_columns.h launch_of: a tile a CTA where the grid fills its last wave to 90 %
# generic_gemm_internal.h kRmmaLatencyOneSetBytes (store and add: one weight set, generic_rmma_describe.cpp) / SwiGluBytes
LAT_BYTES = {'store': 5.5, 'add': 5.5, 'swiglu': 4.5}
VEC_KS = (5120, 6144, 17408)       # ggml_vec.cu vec_launch_k: the K values with an unrolled kernel (5120 alone when gated)

# NAMES: format -> (bits a weight, rmma decoder, r8 / W4A8 codec, ggml_type), as generic_rmma_decoders.h rmma_decoder_of,
# generic_select.cpp w4a8_decoder_name and ggml's enum name them. A family is a candidate where its kernel-facts row exists.
FORMATS = {
    'q4k': (4.5, 'GgmlQ45KQuadDecoder<4>', 'W4a8Q45K<4>', '12'), 'q5k': (5.5, 'GgmlQ45KQuadDecoder<5>', 'W4a8Q45K<5>', '13'),
    'q6k': (6.5625, 'GgmlQ6KDecoder', 'W4a8Q6K', '14'), 'q8': (8.5, 'Q8Decoder', 'W4a8Q8', None),
    'q3k': (3.4375, 'GgmlQ3KDecoder', 'W4a8Q3K', '11'), 'q2k': (2.625, 'GgmlQ2KDecoder', 'W4a8Q2K', '10'),
    'iq4xs': (4.25, 'GgmlIq4XsDecoder', 'W4a8Iq4Xs', '23'), 'iq4nl': (4.5, 'Ggml32Decoder<4>', 'W4a8Ggml32<4>', '20'),
    'iq3s': (3.4375, 'GgmlIq3SDecoder', 'W4a8Iq3S', '21'), 'iq3xxs': (3.0625, 'GgmlIq3XxsDecoder', 'W4a8Iq3Xxs', '18'),
    'iq2s': (2.5625, 'GgmlIq2SDecoder', 'W4a8Iq2S', '22'), 'iq2xs': (2.3125, 'GgmlIq2XsDecoder', 'W4a8Iq2Xs', '17'),
    'iq2xxs': (2.0625, 'GgmlIq2XxsDecoder', 'W4a8Iq2Xxs', '16'), 'iq1s': (1.5625, 'GgmlIq1SDecoder', 'W4a8Iq1S', '19'),
    'iq1m': (1.75, 'GgmlIq1MDecoder', 'W4a8Iq1M', '29'), 'q40': (4.5, 'Ggml32Decoder<0>', 'W4a8Ggml32<0>', '2'),
    'q41': (5.0, 'Ggml32Decoder<1>', 'W4a8Ggml32<1>', '3'), 'q50': (5.5, 'Ggml32Decoder<2>', 'W4a8Ggml32<2>', '6'),
    'q51': (6.0, 'Ggml32Decoder<3>', 'W4a8Ggml32<3>', '7'), 'bf16': (16.0, 'Bf16Decoder', None, None),
    'fp8': (8.0, 'Fp8Decoder', None, None), 'fp8r32': (8.0, 'Fp8RowF32Decoder', None, None),
    'nvfp4': (4.5, 'Nvfp4Decoder', 'W4a8Nvfp4', None), 'q4': (4.25, 'GroupwiseG64Decoder<4>', 'W4a8G64<4>', None),
    'q5': (5.25, 'GroupwiseG64Decoder<5>', 'W4a8G64<5>', None), 'q6': (6.25, 'GroupwiseG64Decoder<6>', 'W4a8G64<6>', None),
}
BITS = {f: v[0] for f, v in FORMATS.items()}
RMMA_DEC = {f: v[1] for f, v in FORMATS.items()}
R8_CODEC = {f: v[2] for f, v in FORMATS.items() if v[2]}
# The blocks a lane's group spans (generic_w4a8.cuh: kSpan x kStride of the codec; 1 where it has neither).
R8_GROUP = {'W4a8Q45K': 2, 'W4a8Q6K': 4, 'W4a8Q3K': 4, 'W4a8Q2K': 4}
MMQ_TYPE = {f: v[3] for f, v in FORMATS.items() if v[3]}
EPI = {'store': '0', 'add': '1', 'swiglu': '2'}   # the kernel facts' epilogue column
_RMMA = re.compile(r'rmma\.m(\d+)\.n(\d+)\.w(\d+)x(\d+)(?:\.c(\d+))?$')
_R8 = re.compile(r'r8\.w(\d+)r(\d+)$')


def _num(v, d=0.0):
    try:
        x = float(v)
        return d if x != x else x
    except (TypeError, ValueError):
        return d


def kf_row(kf, fam, **want):
    """The first kernel-facts row of `fam` whose key fields equal `want`, or None (cached in the arch's facts)."""
    cache = kf.setdefault('_rows', {})
    key = (fam, tuple(sorted(want.items())))
    if key not in cache:
        cache[key] = next((r for r in kf[fam] if all(r.get(k) == str(v) for k, v in want.items())
                           and (fam == 'vec' or _num(r.get('loops', '1'), 1.0) > 0)), None)
    return cache[key]


CARD_FIELDS = ('cc', 'sms', 'boost_mhz', 'boost_min_mhz', 'form', 'bus_bits', 'gbps')


def card_of(gpu, F):
    """The numbers of a gpu_facts.json row the price uses, or None for a card on an architecture without facts (cached
    in F by the row's CARD_FIELDS, the fields read)."""
    cache = F.setdefault('_cards', {})
    key = tuple(gpu.get(f) for f in CARD_FIELDS)
    if key not in cache:
        cc = int(gpu['cc'])
        arch = next((a for a in (str(cc), '%da' % cc) if a in F['isa']), None)       # cc 120 -> the sm_120a snapshot
        lim = next((r for r in F['archs'] if r['cc'] != 0 and r['cc'] <= cc <= r.get('cc_to', r['cc'])),
                   next((r for r in F['archs'] if r['cc'] == 0), None))
        c = None
        if arch is not None:
            # A laptop at the bottom of its boost range, as the engine prices it (machine_facts.cpp: the
            # OEM's power limit sets the clock; the unknown is conservative), else the boost clock.
            mhz = gpu.get('boost_min_mhz') if gpu.get('form') == 'laptop' and gpu.get('boost_min_mhz') else gpu['boost_mhz']
            sms, clk = float(gpu['sms']), float(mhz) * 1e-3                           # SM clock, cycles a ns
            bus = float(gpu['bus_bits']) * float(gpu['gbps']) / 8.0                   # DRAM bus, bytes a ns
            c = dict(F['isa'][arch], cc=cc, kf=F['kf'][arch], sms=sms, clk=clk, bus=bus, bpc=bus / (sms * clk),
                     regs_sm=int(lim['registers_per_sm']), l1=(float(lim['smem_per_sm_kib']) + L1_MIN_KIB) * 1024.0,
                     thr_sm=int(lim['max_threads_per_sm']), blk_sm=int(lim['max_blocks_per_sm']),
                     smem_sm=float(lim['smem_per_sm_kib']) * 1024.0, smem_res=float(lim.get('smem_reserved_per_block', 0)))
            c['arch'] = arch
            c['die'] = dict(nr=c['nr'])                      # the arch's narrow-load facts (analyzer_facts.NARROW)
        cache[key] = c
    return cache[key]


def occupancy(c, warps, regs):
    """CTAs of a kernel one SM holds at once: its registers (a warp takes them in units of 256), its threads against the
    SM's thread limit, and the SM's CTA limit (gpu_facts archs: max_threads_per_sm, max_blocks_per_sm)."""
    by_regs = (c['regs_sm'] // (math.ceil(regs * 32.0 / 256.0) * 256)) // warps
    return max(float(min(by_regs, c['thr_sm'] // (warps * 32), c['blk_sm'])), 1.0)


def launch_ns(c, P, ctas):
    """ns one kernel launch takes before its work (round 18). P LCH '' = one graph node whatever the grid (round 17); 'g' = a graph node
    whose floor grows with its CTAs (isa_facts launch facts floor_ns_1 / _sms / _4sms / _1024 / _4096: CTAs of 128 threads, linear between
    the probe points and past the last); 's' = the same under a stream launch (stream_ns), which is how route_tune times a call
    (protocol `copies`: consecutive stream launches). No fitted number."""
    m = P.get('LCH')
    fl = c.get('floor')
    n = max(float(ctas), 1.0)
    f = fl[-1][1] + (n - fl[-1][0]) * (fl[-1][1] - fl[-2][1]) / (fl[-1][0] - fl[-2][0])
    for (n0, f0), (n1, f1) in zip(fl, fl[1:]):
        if n <= n1:
            f = f0 + (f1 - f0) * (n - n0) / (n1 - n0)
            break
    base = c['stream'] if m == 's' and c.get('stream') else c['node']
    return base + max(f - fl[0][1], 0.0)


def second_kernel(c, P):
    """ns a route's second kernel (vec's quantizer, r8's / mmq's activation pre-pass) costs beyond its launch and L0. UNEXPLAINED: measured
    1.0-1.9 us on sm_75..120a once the stream launch is priced (explore/offs.py); P XK prices it as one more stream launch (a form, chosen
    left one card out; no fitted number)."""
    return (c.get('stream') or c['node']) if P.get('XK') else 0.0


def window(c, P, bw):
    """ns of one DRAM round trip of a warp that keeps bw bytes in flight (PR 1180, #771): L0 for the first 512 B, then
    the window fills at F bytes a clock (facts.py `fill`, warp_fill_bytes_per_clock). No fitted number."""
    return c['l0'] + max(bw - 512.0, 0.0) / c['fill'] / c['clk']


def price_rmma(c, P, r, n, k, T, pitch, M, N, A, B, C, issue=False, sets=1.0):
    """ns of one register-MMA launch: rounds of the CTAs the SMs hold (registers), a round = hypot(SM side, DRAM side).
    SM side: K trips x tau + the code outside the loop, plus the wave term (every warp's K step waits one K step's bytes
    of every row of the round at the bus rate, / warps_k); tau = the warp's chain (its DRAM round trips, each a window of
    the bytes it keeps in flight, + the loop's static stalls + an emulated MMA's latency) or an issue bound. DRAM side:
    the round's bytes / bus, slowed by the rows open on the busiest SM (G, the one fitted exponent)."""
    sms, clk, bus = c['sms'], c['clk'], c['bus']
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
    out = c['lat'] * _num(r['outside_stall']) / clk
    red = (passes - 1.0) / passes                          # the share of the work that decodes the same weights again
    steps = 2.0 - _num(r['latency'])                       # DRAM round trips a K trip exposes, each with its share of the bytes
    bw = _num(r['ldg_bytes']) + _num(r.get('ld_bytes')) + _num(r.get('ldgsts_bytes'))
    w_dram = steps * window(c, P, bw / steps)              # the chain's DRAM round trips
    st = _num(r['stall_sum']) / clk                        # and its issue (static stall clocks)
    tot = _num(r['total'])
    # #1165: an emulated MMA (sm_70 NINFERNO_FMA_MMA, shape '...EMU') is a call whose body the loop's `total` leaves out:
    # its issue cost (EMU_ISSUE) on the SM and its measured latency (EMU_LAT_NS, by the independent accumulators) on the chain.
    emulated = str(r.get('mma_shape', '')).endswith('.EMU')
    emu = _num(r['mma']) * EMU_ISSUE if emulated else 0.0
    w_lat = _num(r['mma']) * EMU_LAT_NS[min(4, int(M * N * sets))] if emulated else 0.0
    own = w_dram + st + w_lat

    def rnd(cps, nc, o):                                   # a round of nc CTAs, cps of them on the busiest SM
        dram = nc * bcta / bus * max(1.0, cps * rows_cta * pitch / c['l1']) ** P['G']
        cm = max(nc / sms, 1.0)                            # the CTAs the launch puts on an SM in this round
        tau = max(own,
                  red * cps * warps * _num(r['int']) / PARTITIONS / clk + (1.0 - red) * own,
                  cps * warps * tot / PARTITIONS / clk if issue else 0.0,     # the issue bound: pair ratios only
                  cm * warps * (tot + emu) / PARTITIONS / clk if emu else 0.0)  # the issue bound with emulated calls
        sm = trips * tau + o + nc * bcta / bus / float(B)  # the wave term (round 18, AUDIT r18: 1 + 1 / warps_k)
        return math.sqrt(sm * sm + dram * dram)

    part = lambda x: max(min(math.ceil(x / sms), occ), 1.0)
    ln = launch_ns(c, P, ctas)
    if nfull < 1:
        return ln + rnd(part(ctas), tail, out)
    return ln + rnd(occ, slots, out) + (nfull - 1.0) * rnd(occ, slots, out) + (rnd(part(tail), tail, out) if tail > 0 else 0.0)


MMQ_TILE = 128                     # core/mmq_columns.h kTile: a tile's rows
MMQ_UNIT_K = 256                   # kUnitK: a unit's K (a 32-value block format's ranges round to 8 blocks: the same)


def mmq_launch(n, k, sms, j, T):
    """core/mmq_columns.h launch_of, line for line: the CTAs of ggml's MMQ launch (a tile a CTA where the grid fills its
    last wave to 90 %, else stream-k over min(SMs, work) CTAs), whether the fixup kernel runs (the CTAs do not divide the
    tiles), and the fixup's partial tiles, split tiles and dependent rounds."""
    tiles = math.ceil(n / float(MMQ_TILE)) * math.ceil(T / float(j))
    steps = max(1, math.ceil(k / float(MMQ_UNIT_K)))
    l = dict(tiles=tiles, blocks=tiles, units=steps, split=False, fixup=False, partials=0, split_tiles=0, rounds=0)
    if sms <= 0 or tiles <= 0:
        return l
    waves = math.ceil(tiles / float(sms))
    l['units'] = waves * steps
    if 100 * tiles // (sms * waves) >= round(100 * MMQ_FILL):
        return l
    work = tiles * steps
    blocks = min(sms, work)
    l.update(blocks=blocks, units=math.ceil(work / float(blocks)), split=True, fixup=tiles % blocks != 0)
    last = -1
    for b in range(1, blocks):
        end = b * work // blocks
        if end % steps == 0:
            continue
        l['partials'] += 1
        if end // steps != last:
            l['split_tiles'] += 1
            last = end // steps
    tile, touching, rounds = -1, 0, 0
    for b in range(blocks):
        begin, end = b * work // blocks, (b + 1) * work // blocks
        if begin == end:
            continue
        first, closing = begin // steps, (end - 1) // steps
        if first == tile:
            touching += 1
        else:
            rounds, tile, touching = max(rounds, touching - 1), first, 1
        if closing != first:
            rounds, tile, touching = max(rounds, touching - 1), closing, 1
    l['rounds'] = max(rounds, touching - 1)
    return l


def mmq_fixup_ns(c, P, l, j):
    """ns of the stream-k fixup kernel after a product (mmq_unit.cpp launch_seconds): its launch (the stream-k grid x I / 32
    CTAs) and its work, the bytes it moves (every partial tile read, each split tile's output read and written, 128 x j
    FP32: mmq_columns.h fixup_bytes) at the L2 rate of its working CTAs (a split tile's 128 / 32, at most the SMs), but no
    faster than its walk (the deepest split tile's rounds + 1 L2 round trips of four warps)."""
    if not l['fixup']:
        return 0.0
    ctas = min(max(l['split_tiles'] * (MMQ_TILE // 32), 1), int(c['sms']))
    bytes_ns = (l['partials'] + 2.0 * l['split_tiles']) * MMQ_TILE * j * 4.0 / (c['l2bpc'] * ctas * c['clk'])
    walk_ns = (l['rounds'] + 1) * c['l2trip'] / c['clk'] if l['rounds'] > 0 else 0.0
    return launch_ns(c, P, l['blocks'] * (MMQ_TILE // 32)) + max(bytes_ns, walk_ns)


def price_other(c, P, fam, r, n, k, T, bits, sets, Wp=0, R=0, ep='store', act=0.0):
    """ns of a ggml_mmq or r8 launch. ggml_mmq: waves of 128-row tiles, a unit = max(its MMA issue, its loop's stall
    clocks) + its bytes at the bus. r8: rounds of resident CTAs, a round = hypot(the warp's chain, the round's bytes /
    bus); the chain is the dp4a loop's stall clocks and a DRAM window for each global-wait generation a trip (#1207: that
    charge is the suspect of r8's 2x on NVFP4), the activation quantizer's loop at T 1, the code outside."""
    sms, clk, l0, bus, pitch = c['sms'], c['clk'], c['l0'], c['bus'], sets * k * bits / 8.0
    pre = launch_ns(c, P, 1) + l0 + second_kernel(c, P)    # the activation pre-pass: a launch and a DRAM latency
    out = c['lat'] * _num(r['outside_stall']) / clk if r else 0.0
    if fam == 'ggml_mmq':                                  # 128-row tiles; one CTA an SM, or a tile a CTA (the fill rule)
        # ggml_mmq.cu quantize_activation: a CTA quantizes 512 values of a column, columns x ceil(K / 512) CTAs.
        pre = launch_ns(c, P, T * math.ceil(k / 512.0)) + l0 + second_kernel(c, P)
        nt = math.ceil(n / 128.0)
        launch = mmq_launch(n, k, int(sms), mmq_j(T), T)
        ctas = launch['blocks']
        # The longest CTA's units (launch_of units: a tile a CTA, a wave of them after another; stream-k, ceil(work /
        # blocks)), each set's: a launch takes as long as its longest CTA (mmq_unit.cpp launch_seconds).
        per = launch['units'] * sets / math.ceil(ctas / sms)
        cyc = c['imma_w'] if '16832' in r.get('mma_shape', '') else c['imma_n']
        # The unit as mmq_unit.cpp unit_seconds: its loads (the weight tiles at the bus, or every resident CTA's tile and
        # its j activation columns, two block_q8_1_mmq of 144 B a column, at the L2 rate) and one DRAM latency, then its
        # chain (the loop's stall clocks and its waits on shared loads, a shared latency each); the MMA issue bounds it.
        chain = (_num(r['stall_sum']) + _num(r.get('sb_waits_shared')) * c['smem_lat']) / clk
        issue = max(MMQ_WARPS * _num(r['mma']) * cyc / PARTITIONS / clk, chain)
        # The weights come from DRAM once over the launch (mmq_unit.cpp unit_facts_of): the row tiles' K steps spread over
        # the launch's units, at most one weight tile a resident CTA an iteration (a tile each up to the SMs, or the
        # stream-k CTAs); a tile is 128 rows x 256 values.
        resident = ctas if launch['split'] else min(launch['tiles'], int(sms))
        weight_tiles = min(resident, math.ceil(nt * max(1, math.ceil(k / 256.0)) / float(launch['units'])))
        l2_ns = resident * (4096.0 * bits + mmq_j(T) * 288.0) / (c['l2bpc'] * resident * clk)
        load = max(4096.0 * bits * weight_tiles / bus, l2_ns) + c['dramlat'] / clk
        unit = per * (issue + load)
        # generic_gemm.cpp GgmlMmq: the quantizer (pre), one product launch a weight set (a gated call runs the gate's and
        # the up's, each its own rounds with its own code outside the loop), then the epilogue kernel over the plane
        # (generic_mmq_epilogue.cu: ceil(rows / 1024) x T CTAs of 256 threads, a launch, and its bytes: each output reads
        # its FP32 plane value, the up's too under SwiGLU, an added call its BF16 residual, and writes its BF16 value), at
        # the L2 rate: the plane is the product's just-written output, the residual and the output a few KiB a token.
        epi_bytes = n * T * (4.0 * sets + 2.0 + (2.0 if ep == 'add' else 0.0))
        epi = launch_ns(c, P, math.ceil(n / 1024.0) * T) + epi_bytes / (c['l2bpc'] * sms * clk)
        # Stream-k adds the fixup kernel after each product where the CTAs do not divide the output tiles (mmq_launch).
        fix = mmq_fixup_ns(c, P, launch, mmq_j(T))
        return pre + sets * (launch_ns(c, P, ctas) + fix + math.ceil(ctas / sms) * (unit / sets + out)) + epi
    ctas = math.ceil(n / float(Wp * R))                    # r8
    slots = sms * occupancy(c, Wp, _num(r['registers']))
    trips = k / max(_num(r['k_iter']), 1.0)
    gw = _num(r.get('glb_waits'))
    trip = _num(r['stall_sum']) / clk + gw * window(c, P, _num(r['ldg_bytes']) / max(gw, 1.0))
    chain = trips * trip + out
    if T <= 1:                                             # the CTA quantizes the activation: its loads are L2 latencies
        chain += k / 1024.0 / Wp * (_num(r.get('pro_stall')) / clk + _num(r.get('pro_glb_waits')) * c['l2lat_ns'])
    rounds = math.ceil(ctas / slots)
    last = ctas - (rounds - 1) * slots
    # A round's memory side: its weights at the bus, or its warps' activation L1 requests (act a warp's trip, a
    # pre-quantized call: ncu shows that call L1-bound, 85 % of the L1/TEX throughput, its L2 traffic below T 1's), one
    # wavefront each on the SMs the round fills, whichever is longer.
    rnd = lambda nc: math.hypot(chain, max(nc * Wp * R * pitch / bus,
                                           nc * Wp * trips * act * c['die']['nr']['wf'] / (min(nc, sms) * clk)))
    return launch_ns(c, P, ctas) + (0.0 if T <= 1 else pre) + (rounds - 1) * rnd(slots) + rnd(last)


VEC_WARPS = 4                      # ggml_vec.cu kVecWarps
VEC_TWIN_BLOCKS = 7                # ggml_vec.cu kVecBoundedBlocks: the bounded twin's CTAs an SM (T 4, runtime K, plain)


def vec_ks(ep, k, inst=None):
    """The static K of the instance: the unrolled kernel where the product has one (inst None or 'ks'), 0 = the
    runtime-K loop (inst 'loop': lab/vec-ks0, or any K without an unrolled kernel)."""
    gated = ep == 'swiglu'
    return int(k) if inst != 'loop' and int(k) in VEC_KS and (not gated or int(k) == 5120) else 0


def vec_row(c, fmt, ep, k, T, bounded=0, inst=None):
    return None if fmt not in MMQ_TYPE else kf_row(c['kf'], 'vec', type=MMQ_TYPE[fmt], T=int(T), KS=vec_ks(ep, k, inst),
                                                   gated=int(ep == 'swiglu'), bounded=bounded)


def vec_occ(c, r):
    """CTAs of a vec instance an SM holds: its registers, threads, blocks and static shared memory (vec_slots)."""
    regs = math.ceil(_num(r['registers']) * 32.0 / 256.0) * 256
    occ = min((c['regs_sm'] // regs) // VEC_WARPS, c['thr_sm'] // (VEC_WARPS * 32), c['blk_sm'])
    sh = _num(r['shared'])
    if sh > 0:
        occ = min(occ, int(c['smem_sm'] // (sh + c['smem_res'])))
    return max(int(occ), 1)


def vec_launch(c, fmt, ep, n, k, T, inst=None):
    """The launch as vec_launch_ks makes it: (kernel-facts row, CTAs, row groups). The grid is min(groups, resident
    CTAs); at T 4 the bounded twin where it saves a round and spills nothing (`local`: the local-memory frame bytes)."""
    r = vec_row(c, fmt, ep, k, T, inst=inst)
    groups = math.ceil(n / (VEC_WARPS * (1.0 if ep == 'swiglu' else 2.0)))
    slots = vec_occ(c, r) * int(c['sms'])
    tw = vec_row(c, fmt, ep, k, T, 1, inst) if int(T) == 4 and ep != 'swiglu' and r['KS'] == '0' else None
    if tw is not None and _num(tw['local']) == 0 and _num(tw['stack']) == 0:
        ts = min(vec_occ(c, tw), VEC_TWIN_BLOCKS) * int(c['sms'])
        if math.ceil(groups / ts) < math.ceil(groups / slots):
            r, slots = tw, ts
    return r, int(min(groups, slots)), int(groups)


# ---- round 18: vec's loads from the source (ggml_vec.cu decoders), the narrow-load rule, the executed epilogue, the icache step ----
# NAMES and source counts, no fitted number. A load = (bytes a lane, plane, bytes a block in that plane, offsets by u = lane % (lanes a block)).
# The lane's block is lane // (lanes a block); a 256-value block type has 8 lanes a block (kPerBlock), a 32-value block type 1.
_U8 = range(8)
VEC_LOADS = {     # ggml_vec.cu Decoder<type>::decode, one row of one 1024-value slice
    'q4k': (8, [(4, 'c', 144, [0] * 8), (1, 'c', 144, [4 + u for u in _U8]), (1, 'c', 144, [8 + u for u in _U8]),
                (1, 'c', 144, [4 + (u & 3) for u in _U8]), (1, 'c', 144, [8 + (u & 3) for u in _U8]),
                (16, 'c', 144, [16 + 32 * (u // 2) for u in _U8]), (16, 'c', 144, [32 + 32 * (u // 2) for u in _U8])]),
    'q5k': (8, [(4, 'c', 176, [0] * 8), (1, 'c', 176, [4 + u for u in _U8]), (1, 'c', 176, [8 + u for u in _U8]),
                (1, 'c', 176, [4 + (u & 3) for u in _U8]), (1, 'c', 176, [8 + (u & 3) for u in _U8]),
                (16, 'c', 176, [16] * 8), (16, 'c', 176, [32] * 8),
                (16, 'c', 176, [48 + 32 * (u // 2) for u in _U8]), (16, 'c', 176, [64 + 32 * (u // 2) for u in _U8])]),
    'q2k': (8, [(4, 's', 4, [0] * 8), (1, 'h', 16, [8 * (u // 4) + 2 * (u % 4) for u in _U8]), (1, 'h', 16, [8 * (u // 4) + 2 * (u % 4) + 1 for u in _U8]),
                (16, 'c', 64, [32 * (u // 4) for u in _U8]), (16, 'c', 64, [32 * (u // 4) + 16 for u in _U8])]),
    'q6k': (8, [(2, 's', 2, [0] * 8), (1, 'h', 16, [8 * (u // 4) + 2 * (u % 4) for u in _U8]), (1, 'h', 16, [8 * (u // 4) + 2 * (u % 4) + 1 for u in _U8])]
               + [(2, 'c', 192, [64 * (u // 4) + 32 * (u & 1) + 4 * i + d for u in _U8]) for i in range(8) for d in (0, 2)]
               + [(2, 'c', 192, [128 + 32 * (u // 4) + 4 * i + d for u in _U8]) for i in range(8) for d in (0, 2)]),
    'iq3xxs': (8, [(8, 'c', 64, [8 * u for u in _U8]), (4, 'h', 32, [4 * u for u in _U8]), (2, 's', 2, [0] * 8)]),
    'iq3s': (8, [(8, 'c', 104, [8 * u for u in _U8]), (1, 'c', 104, [64 + u for u in _U8]), (4, 'c', 104, [72 + 4 * u for u in _U8]),
                 (1, 'h', 4, [u // 2 for u in _U8]), (2, 's', 2, [0] * 8)]),
    'q3k': (8, [(2, 's', 2, [0] * 8)] + [(1, 'h', 12, [(8 * (u // 4) + 2 * (u % 4) + e) % 8 for u in _U8]) for e in (0, 1)]
               + [(1, 'h', 12, [8 + (8 * (u // 4) + 2 * (u % 4) + e) % 4 for u in _U8]) for e in (0, 1)]
               + [(16, 'c', 96, [0] * 8), (16, 'c', 96, [16] * 8), (16, 'c', 96, [32 + 32 * (u // 4) for u in _U8]), (16, 'c', 96, [48 + 32 * (u // 4) for u in _U8])]),
    'iq2xxs': (8, [(8, 'c', 64, [8 * u for u in _U8]), (2, 's', 2, [0] * 8)]),
    'iq2xs': (8, [(8, 'c', 64, [8 * u for u in _U8]), (2, 's', 2, [0] * 8), (1, 'h', 8, [u for u in _U8])]),
    'iq2s': (8, [(4, 'c', 72, [4 * u for u in _U8]), (4, 'c', 72, [32 + 4 * u for u in _U8]), (1, 'c', 72, [64 + u for u in _U8]),
                 (2, 's', 2, [0] * 8), (1, 'h', 8, [u for u in _U8])]),
    'iq1s': (8, [(4, 'c', 32, [4 * u for u in _U8]), (2, 'h', 16, [2 * u for u in _U8]), (2, 's', 2, [0] * 8)]),
    'iq1m': (8, [(2, 's', 8, [0] * 8), (2, 's', 8, [2] * 8), (2, 's', 8, [4] * 8), (2, 's', 8, [6] * 8),
                 (4, 'c', 32, [4 * u for u in _U8]), (2, 'h', 16, [2 * u for u in _U8])]),
    'iq4xs': (8, [(2, 's', 4, [0] * 8), (2, 's', 4, [2] * 8), (1, 'h', 4, [u // 2 for u in _U8]),
                  (8, 'c', 128, [16 * u for u in _U8]), (8, 'c', 128, [16 * u + 8 for u in _U8])]),
    'q40': (1, [(16, 'c', 16, [0]), (2, 's', 2, [0])]), 'iq4nl': (1, [(16, 'c', 16, [0]), (2, 's', 2, [0])]),
    'q41': (1, [(16, 'c', 16, [0]), (4, 's', 4, [0])]),
    'q50': (1, [(16, 'c', 16, [0]), (4, 'h', 4, [0]), (2, 's', 2, [0])]),
    'q51': (1, [(16, 'c', 16, [0]), (4, 'h', 4, [0]), (4, 's', 4, [0])]),
}
VEC_ACT = [(4, [4 * l for l in range(32)]), (16, [32 * l for l in range(32)]), (16, [32 * l + 16 for l in range(32)])]   # ds, the two int4 of qs: a token
# The epilogue a warp runs for a row group, instructions (TASK item 4; read off sm_86's SASS of vec_kernel<Q4_K, 8, 0> in
# C:/Users/MixaPC/few/sass/ka86.sass with explore/sassfn.py: the reduction 0x1cd0-0x2700, the first store site 0x2710-0x2d20):
EPI_RED = 10.0      # a value: 5 SHFL + 5 FADD (T != 4, the gated pair)
EPI_SITE = {'store': 51.0, 'add': 54.0, 'swiglu': 51.0}   # one store site on the path of the launch's kind (prefix 15, kind tests 8,
                    # one segment-search pass 12, index 11, F2FP + STG 4, BSYNC 1; the residual adds its LDG + PRMT + FADD)
EPI_GATE = 41.0     # the gate's issue slots of a gated site (0x2880-0x2b00)


def load_clocks(nr, lanes):
    """Clocks one warp load takes of the SM's load unit (AUDIT 19): max(issue cap, wavefronts x clocks a wavefront)."""
    by = {}
    bank = nr['bank']
    for a, w in lanes:
        for b in range(a // bank, (a + w - 1) // bank + 1):
            by.setdefault(b % (128 // bank), set()).add(b * bank // 128)
    return max(nr['cap'], max(len(v) for v in by.values()) * nr['wf'])


_VLU = {}


def vec_load_unit(c, fmt, T):
    """(clocks of a row's weight loads, clocks of a token's activation loads) for one 1024-value slice of one warp, by the
    narrow-load rule with the die's probe facts; the block's alignment in its line averaged over the 4 B offsets (cached by
    the facts' values: a pure function of them)."""
    nr = c['die']['nr']
    key = (nr['cap'], nr['wf'], nr['bank'], fmt)
    if key not in _VLU:
        per, loads = VEC_LOADS[fmt]
        uw = 0.0
        for width, plane, blk, offs in loads:
            tot = 0.0
            for al in range(0, 128, 4):
                tot += load_clocks(nr, [(al + (l // per) * blk + offs[l % per], width) for l in range(32)])
            uw += tot / 32.0
        ua = sum(load_clocks(nr, [(a, w) for a in offs]) for w, offs in VEC_ACT)
        _VLU[key] = (uw, ua)
    return _VLU[key]


def sm_side(c, P, mode, tot, stall, o_tot, o_stall, trips, cps, dram_trips, bw, ld_clk, icf=1.0, warps=VEC_WARPS):
    """ns one CTA's warp takes for its whole K on an SM that holds cps such CTAs: the larger of
      its own chain   trips x the loop's stall clocks + the epilogue's ('o'), times the icache step, and a DRAM window a round trip;
      the issue bound ('i') one warp of each resident CTA shares a partition: cps x the instructions, a clock each, times the icache step;
      the load unit   ('l') every resident warp's loads through the SM's one load unit: cps x warps x ld_clk (the whole K's clocks)."""
    clk = c['clk']
    og = 'o' in mode
    sm = icf * (trips * stall + (o_stall if og else 0.0)) / clk + dram_trips * window(c, P, bw)
    if 'i' in mode:
        sm = max(sm, icf * cps * (trips * tot + (o_tot if og else 0.0)) / clk)
    if 'l' in mode:
        lsu = cps * warps * ld_clk / clk
        sm = max(sm, lsu)
    return sm


def vec_counts(c, P, q, fmt, k, T):
    """The counted terms of a vec instance row q for the price: (instructions a trip, stall clocks a trip, epilogue instructions,
    epilogue stall clocks, trips, load-unit clocks of the whole K, body bytes). P LD 'n': the loads by the narrow-load rule
    (vec_load_unit), else round 17's count x l1_ldg32_cycles_per_sm."""
    tot, stall = _num(q['total']), _num(q['stall_sum'])
    o_tot, o_stall = _num(q['outside_total']), _num(q['outside_stall'])
    unrolled = q['KS'] != '0'
    slices = k / 1024.0
    trips = 1.0 if unrolled else slices
    lu = vec_load_unit(c, fmt, T) if P.get('LD') == 'n' else None
    nld = _num(q['ldg']) + _num(q.get('lds'))
    if lu is None:
        ld = trips * nld * c['l1cyc']
    else:
        cap = c['die']['nr']['cap']
        ld = slices * (2.0 * lu[0] + int(T) * lu[1]) + trips * _num(q.get('lds')) * cap
    body = 16.0 * (tot + (0.0 if unrolled else o_tot))
    return tot, stall, o_tot, o_stall, trips, ld, body


def price_vec(c, P, fmt, ep, n, k, T, bits, sets, inst=None):
    """ggml_vec from its launch: two nodes (the q8_1 quantizer and the product) and the quantizer's DRAM latency, then
    the rounds of the persistent grid, the last with its own CTAs. A warp takes a row pair over the whole K; a round =
    its weight bytes / bus combined with the SM side (sm_side) as rmma's round combines SM and DRAM time. `inst`: None
    = the instance the product launches (unrolled at 5120 / 6144 / 17408), 'loop' = the runtime-K loop at any K.
    No fitted constant."""
    L = vec_launch(c, fmt, ep, n, k, T, inst)
    r, ctas, groups = L
    q = r
    if _num(r['total']) <= 0:                      # unknown flow (a compiled switch, sm_100 / sm_120a gated T 3-8): the
        q = vec_row(c, fmt, 'store', k, T, inst=inst)   # plain instance's loop counts stand in (a row pair a trip either way)
    l0, bus = c['l0'], c['bus']
    bg = n * sets * k * bits / 8.0 / groups        # weight bytes a row group (4 warps, a row pair each)
    rounds = math.ceil(groups / ctas)
    last = groups - (rounds - 1) * ctas
    form, mode = P.get('VEC_FORM', 'h'), P.get('VEC_SM', 'iol')
    tot, stall, o_tot, o_stall, trips, ld, body = vec_counts(c, P, q, fmt, k, T)
    bw = bg / VEC_WARPS / (k / 1024.0)             # DRAM bytes a warp keeps in flight: a 1024-value slice of its rows

    def rnd(nc):
        d = nc * bg / bus
        cps = math.ceil(nc / c['sms'])
        sm = sm_side(c, P, mode, tot, stall, o_tot, o_stall, trips, cps, k / 1024.0, bw, ld, 1.0)
        return math.hypot(sm, d) if form == 'h' else sm + d if form == 's' else max(sm, d)

    return launch_ns(c, P, ctas) + launch_ns(c, P, math.ceil(k / 256.0) * int(T)) + second_kernel(c, P) + l0 + (rounds - 1) * rnd(ctas) + rnd(last)


VEC_MMA_TYPES = ('q4k', 'q5k', 'q6k')   # ggml_vec.cu vec_mma_supported (feat/1162-few)


def vecmma_row(c, fmt, ep):
    rows = c['kf'].get('vecmma') or []      # facts.py: the architecture's own section, or sm_90's as a named stand-in
    t, g = MMQ_TYPE.get(fmt), str(int(ep == 'swiglu'))
    return next((r for r in rows if r['type'] == t and r['gated'] == g and _num(r['mma']) > 0), None)


def price_vec_mma(c, P, fmt, ep, n, k, T, bits, sets):
    """ggml_vec.mma (#1162) from its launch and its kernel-facts row (vecmma, round 17): ceil(rows / 16) CTAs (8
    gated) of 4 warps, not persistent, the CTAs an SM holds from the row's registers and shared memory; a warp takes
    every fourth 256-value superblock (k / 1024 trips), the trip's clocks, issue bound and DRAM window as sm_side.
    T enters neither the grid nor the trips. Without a facts row (the architecture's section missing): no price."""
    if fmt not in VEC_MMA_TYPES or c['cc'] < 80:
        return None
    r = vecmma_row(c, fmt, ep)
    if r is None:
        return None
    rows_cta = 8.0 if ep == 'swiglu' else 16.0
    ctas = math.ceil(n / rows_cta)
    occ = vec_occ(c, r)
    slots = c['sms'] * occ
    trips = k / 256.0 / VEC_WARPS
    bw = _num(r['ldg_bytes']) + _num(r.get('ld_bytes'))
    bc = rows_cta * sets * k * bits / 8.0
    rounds = math.ceil(ctas / slots)
    last = ctas - (rounds - 1) * slots
    mode, form = P.get('MMA_SM', 'io'), P.get('VEC_FORM', 'h')

    def rnd(nc):
        d = nc * bc / c['bus']
        sm = sm_side(c, P, mode, _num(r['total']), _num(r['stall_sum']), _num(r['outside_total']), _num(r['outside_stall']), trips,
                     min(math.ceil(nc / c['sms']), occ), trips, bw, trips * (_num(r['ldg']) + _num(r.get('lds'))) * c['l1cyc'])
        return math.hypot(sm, d) if form == 'h' else sm + d

    return launch_ns(c, P, ctas) + launch_ns(c, P, math.ceil(k / 256.0) * int(T)) + second_kernel(c, P) + c['l0'] + (rounds - 1) * rnd(slots) + rnd(last)


def mmq_j(T):
    return 16 if T <= 16 else 32 if T <= 32 else 64 if T <= 64 else 128


def mmq_takes(shape, F):
    """Whether ggml's MMQ takes the shape (generic_select.cpp mmq_call, mmq_admits): rows and K in whole tiles of
    F['mmq'] (ggml_mmq.h kRowMultiple, kKMultiple); on other rows the launch demotes it, on other K the selection
    refuses it (#1077)."""
    rows, ks = F['mmq']
    return int(shape['n']) % rows == 0 and int(shape['k']) % ks == 0


# Round 19 law 2 (generic_routes.cpp volta_hmma_takes / volta_gemv_takes, generic_gemm.cpp RegisterMma): on a die
# without the register-MMA shapes (sm_70) a lane-local rmma call goes to Volta's HMMA kernel (Q4_K / Q5_K / Q6_K at K % 256,
# Q8 G32 at K % 64, T 1-16) or else its FP32 GEMV (those and BF16 at K % 8, T 1-4): every rmma.* schedule is then one kernel.
VOLTA_HMMA_K = dict(q4k=256, q5k=256, q6k=256, q8=64)
VOLTA_GEMV_K = dict(q4k=256, q5k=256, q6k=256, q8=256, bf16=8)
VOLTA_CTA_ROWS = 32                                    # generic_volta_hmma.cuh kCtaRows: four warps of eight rows


def volta_takes(fmt, k, T):
    if fmt in VOLTA_HMMA_K and 1 <= T <= 16 and k % VOLTA_HMMA_K[fmt] == 0:
        return 'hmma'
    if fmt in VOLTA_GEMV_K and 1 <= T <= 4 and k % VOLTA_GEMV_K[fmt] == 0:
        return 'gemv'
    return None


VOLTA_FMT = dict(q4k=('HK45<4>', 'K45<4>'), q5k=('HK45<5>', 'K45<5>'), q6k=('HK6', 'K6'), q8=('HQ8', 'Q8G32'), bf16=(None, 'Bf16Rows'))


def volta_kf(c):
    """The sm_70 kernel facts of the Volta substitutes (sections vhmma / vgemv / vgemvr), keyed (family, key columns...)."""
    out = c.get('_vkf')
    if out is None:
        out = {}
        for fam, nkey in (('vhmma', ('fmt', 'E', 'NG', 'ahead')), ('vgemv', ('fmt', 'E', 'NT')), ('vgemvr', ('fmt', 'E', 'NT', 'R'))):
            for r in c['kf'].get(fam, []):
                out[(fam,) + tuple(r[k] for k in nkey)] = r
        c['_vkf'] = out
    return out


def volta_sub_price(c, P, n, k, T, bits, sets, fmt=None, ep='store', kind=None):
    """The Volta substitute from its counted kernel (sm_70 kernel facts vhmma / vgemv / vgemvr: its loop's instructions
    and static stalls an iteration of 256 values, its registers / shared memory): an iteration per SM = max(one warp's
    chain, the SM's issue, the SM's share of the DRAM rate), over waves x iterations. None without the kernel's row."""
    E = {'store': '0', 'add': '1', 'swiglu': '2'}.get(ep, '0')
    kf = volta_kf(c)
    if kind == 'hmma':
        warps, rows_cta = 4, VOLTA_CTA_ROWS
        ctas = math.ceil(n / rows_cta)
        ahead = '1' if ctas >= 2 * c['sms'] else '2'
        r = kf.get(('vhmma', VOLTA_FMT[fmt][0], E, '1' if T <= 8 else '2', ahead))
        depth = 2.0 if ahead == '1' else 3.0
    else:
        R = 4 if ep == 'store' and math.ceil(n / 32) >= 4 * c['sms'] else 1
        warps, rows_cta = 8, 8 * R
        ctas = math.ceil(n / rows_cta)
        nt = '1' if T <= 1 else '2' if T <= 2 else '4'
        r = kf.get(('vgemvr', VOLTA_FMT[fmt][1], E, nt, str(R)) if R > 1 else ('vgemv', VOLTA_FMT[fmt][1], E, nt))
        depth = 1.0
    if r is None:                                                  # no counted kernel on this tree: no price
        return None
    regs, shared = _num(r['registers']), _num(r['shared'])
    occ = occupancy(c, warps, regs)
    if shared > 0:
        occ = max(min(occ, float(int(c['smem_sm'] // (shared + c.get('smem_res', 1024.0))))), 1.0)
    tot, stall = _num(r['total']), _num(r['stall_sum'])
    rows_warp = 8.0 if kind == 'hmma' else rows_cta / warps      # hmma: a warp's eight fragment rows; gemv: R rows a warp
    bytes_iter = rows_warp * 256.0 * bits / 8.0 * sets            # a warp's iteration: 256 values of each of its rows
    iters = math.ceil(k / 256.0)                                   # superblocks, the last bounded (k % 256 at K % 64 / 8)
    sms, clk = c['sms'], c['clk']
    # A gated GEMV runs the gate's K loop, then the up's (generic_volta_gemv.cu: kSets loops; the row counts the heavier,
    # `loops` of them): each its share of the bytes. The HMMA kernel holds both sets in one loop.
    kloops = 1.0 if kind == 'hmma' else max(_num(r.get('loops'), 1.0), 1.0)
    t = launch_ns(c, P, ctas) + c['l0']
    left = float(ctas)
    while left > 0:                                                # rounds of the CTAs the SMs hold at once
        nc = min(left, occ * sms)
        w_sm = math.ceil(nc / sms) * warps
        issue = w_sm * tot / PARTITIONS / clk                      # the SM's warps, an instruction a clock a partition
        chain = (tot + stall) / clk                                # one warp's iteration: its instructions and static stalls
        bus = c['bus']
        dram = nc * warps * bytes_iter / bus                       # the round's iteration bytes at the die's rate
        # A warp's load waits one DRAM latency over the superblocks in flight, then the round's bytes stream behind it.
        t += iters * kloops * max(issue, chain, c['l0'] / depth + dram / kloops)
        left -= nc
    return t


def price_ns(gpu, F, shape, route, P=P):
    """ns of one call of `shape` by `route` on the card of `gpu` (a gpu_facts.json row), or None: no facts for the card,
    the format or the route's kernel, or a route the launch would not take as named."""
    fmt, anchor = PAIR.get(route, (None, None))
    if fmt == str(shape['format']).lower():                # the pair's ratio by the issue bound, on the anchor's price
        t = [_price(gpu, F, shape, r, P, i) for r, i in ((anchor, False), (route, True), (anchor, True))]
        if None not in t:
            return t[0] * t[1] / t[2]
    return _price(gpu, F, shape, route, P)


def _price(gpu, F, shape, route, P, issue=False):
    c, fmt, ep = card_of(gpu, F), str(shape['format']).lower(), str(shape.get('epilogue', 'store')).lower()
    if c is None or fmt not in BITS:
        return None
    n, k, T = float(shape['n']), float(shape['k']), max(float(shape.get('T', 1)), 1.0)
    kf, bits, sets = c['kf'], BITS[fmt], 2.0 if ep == 'swiglu' else 1.0   # SwiGLU: two weight sets
    m = _RMMA.match(route)
    if m and P.get('VSUB') and c['cc'] == 70 and volta_takes(fmt, k, T):
        return volta_sub_price(c, P, n, k, T, bits, sets, fmt, ep, volta_takes(fmt, k, T))
    if m:
        M, N, A, B, C = (int(x or 1) for x in m.groups())
        key = dict(decoder=RMMA_DEC.get(fmt), warps_m=A, warps_k=B, m_tiles=M, n_tiles=N, cluster=C, epilogue=EPI.get(ep, '0'), sums=0)
        r = kf_row(kf, 'rmma', latency=0, **key)
        if r is not None and N == 1 and c['bpc'] >= LAT_BYTES.get(ep, 5.5) and c['cc'] >= 75:
            r = kf_row(kf, 'rmma', latency=1, **key) or r                  # the latency instance is the one launched
        return None if r is None else price_rmma(c, P, r, n, k, T, sets * k * bits / 8.0, M, N, A, B, C, issue, sets=sets)
    (fam, Wp, R), m = (route, 0, 0), _R8.match(route)
    if m:
        fam, Wp, R = 'r8', int(m.group(1)), int(m.group(2))
        nt = 1 if T <= 1 else 2 if T <= 2 else 4 if T <= 4 else 8
        r = None if T > 8 else kf_row(kf, 'r8', codec=R8_CODEC.get(fmt), warps=Wp, rows=R, nt=nt, epilogue=EPI.get(ep, '0'),
                                      pre=0 if T <= 1 else 1)
        # A pre-quantized call (T >= 2) reads its int8 activation from global (the pre-pass wrote it): the loads a warp's
        # trip issues beyond the same instance quantizing into shared memory (pre 0) are the activation's, its codes in
        # LDG.128 (512 B a warp) and its scales in LDG.32 (128 B): c and s from the count and bytes they add. Consecutive
        # lanes take consecutive groups of R8_GROUP blocks (generic_r8.cuh: kSpan x kStride blocks of 32 B), so a codes load
        # touches 32 lanes x 16 B at that stride: one L1 tag request a 128 B line it touches (ncu, Q6_K: 32 a load), its
        # scales' load R8_GROUP lines. The rate: one wavefront a request (the die's narrow-load facts).
        r0 = kf_row(kf, 'r8', codec=R8_CODEC.get(fmt), warps=Wp, rows=R, nt=nt, epilogue=EPI.get(ep, '0'), pre=0) if T > 1 else None
        act = 0.0
        if r is not None and r0 is not None:
            n_add = max(_num(r['ldg']) - _num(r0['ldg']), 0.0)
            b_add = max(_num(r['ldg_bytes']) - _num(r0['ldg_bytes']), 0.0)
            codes = max(min((b_add - 128.0 * n_add) / 384.0, n_add), 0.0)
            grp = R8_GROUP.get(str(R8_CODEC.get(fmt)).split('<')[0], 1)
            lines = min(32.0, 8.0 * grp)                                  # 32 lanes x 16 B at a 32 x grp B stride
            act = codes * lines + (n_add - codes) * min(float(grp), 32.0)    # L1 tag requests a warp's trip
    elif route == 'ggml_mmq':
        # Rows and K in whole tiles only: on other rows the launch demotes it, on other K the selection refuses it.
        if c.get('imma_n') is None or c.get('imma_w') is None or not mmq_takes(shape, F):
            return None
        r = kf_row(kf, 'mmq', type=MMQ_TYPE.get(fmt), J=mmq_j(T), fallback=0)
    elif route == 'ggml_vec.mma':
        return price_vec_mma(c, P, fmt, ep, n, k, T, bits, sets) if T <= 8 else None
    elif route in ('ggml_vec', 'ggml_vec.ks', 'ggml_vec.loop'):
        # ggml_vec = the instance the product launches today; .loop = the runtime-K loop (lab/vec-ks0); .ks = unrolled
        inst = 'loop' if route.endswith('.loop') else None
        if vec_row(c, fmt, ep, k, T, inst=inst) is None:          # no vec kernel for the format (q8, bf16, fp8, NVFP4, G64)
            return None
        return price_vec(c, P, fmt, ep, n, k, T, bits, sets, inst)
    else:
        return None
    if r is None:
        return None
    return price_other(c, P, fam, r, n, k, T, bits, sets, Wp, R, ep, act if fam == 'r8' else 0.0)


def candidates(gpu, F, shape):
    """The routes the product can launch for the shape that this file prices: ggml_vec (T <= 8), ggml_mmq (T >= 5, whole
    tiles), every rmma and r8 (T <= 8) schedule with a kernel-facts row. Priced, never listed: the lab's instances
    ggml_vec.ks / .loop, and ggml_vec.mma, the few-row MMA tile (#1162), which the product times only in a8 sweeps and
    never tables (route_tune_tool.cpp), so no exam grid holds it; on the RTX 4070 Ti it lost 8 % end to end on Q6_K
    where its price ranked it first (25.11). It joins the list once the exam grids time it."""
    c = card_of(gpu, F)
    if c is None:
        return []
    T, fmt = max(float(shape.get('T', 1)), 1.0), str(shape['format']).lower()
    # ggml's MMQ on rows and K it takes only (mmq_takes; price_ns refuses the others too) (#1077).
    names = set(['ggml_mmq'] if T >= 5 and mmq_takes(shape, F) else []) | set(['ggml_vec'] if T <= 8 else [])
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


def rank(gpu_row, F, shape, among=None):
    """Every priced route, cheapest first: the list the engine walks, taking the first its policy admits."""
    names = sorted(among) if among is not None else candidates(gpu_row, F, shape)
    priced = [(t, r) for t, r in ((price_ns(gpu_row, F, shape, r), r) for r in names) if t is not None]
    return [r for _, r in sorted(priced)]
