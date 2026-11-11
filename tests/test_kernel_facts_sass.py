"""tools/kernel_facts/sass_counts.py and the loop columns of kernel_facts.py (#613 PR-3): the
K loop found from the control flow of a small SASS text shaped like rmma_kernel's (a prologue,
the double-buffered loop with its break in the middle, a row-bound guard around the loads, an
alternative version of a step, an epilogue, a subroutine after EXIT), the heaviest path counted
once, the min-term MMAs told apart, k_iter from the weight MMAs; then the snapshot round trip and
the check's verdicts. No GPU, no CUDA toolkit."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "kernel_facts"))
import kernel_facts  # noqa: E402
import sass_counts as sc  # noqa: E402


# The high word every fixture instruction carries unless a test gives another: stall 2, yield,
# no scoreboard set or waited on.
DEFAULT_HIGH_WORD = 0x000FE40000000F00


def sass(lines, words=None):
    """A `cuobjdump -sass` function body from (offset, text) pairs, in cuobjdump's layout: the
    low word on the instruction's line, the high word (the control code) alone on the next;
    `words` maps an offset to its high word."""
    out = ["\tcode for sm_89", "\t\tFunction : _Z6kernelv"]
    for offset, text in lines:
        high = (words or {}).get(offset, DEFAULT_HIGH_WORD)
        out.append(f"        /*{offset:04x}*/ {text:60s} /* 0x0000000000000000 */")
        out.append(" " * 80 + f"/* 0x{high:016x} */")
    return "\n".join(out) + "\n"


def high_word(stall, yield_flag=1, write=7, read=7, wait=0):
    """A high word with the given control code (bits 41-57) and nothing else."""
    return (stall | yield_flag << 4 | write << 5 | read << 8 | wait << 11) << 41


# Offsets step by 0x10 as in SASS. The loop runs 0x100 .. 0x300 (the back edge at 0x300); the
# break at 0x1f0 leaves it; 0x120 guards the weight loads of rows past the end; 0x220 picks one of
# two versions of the second step (the tail version at 0x230 with one more check, the plain one
# at 0x280); R52 holds the 1.0 pair for the min-term MMAs.
KERNEL = [
    (0x000, "MOV R1, c[0x0][0x28] ;"),
    (0x010, "S2R R0, SR_TID.X ;"),
    (0x020, "ISETP.GE.AND P0, PT, R0, c[0x0][0x380], PT ;"),
    (0x030, "@P0 EXIT ;"),
    (0x040, "IMAD.MOV.U32 R52, RZ, RZ, 0x3f803f80 ;"),
    (0x050, "MOV R53, 0x3f803f80 ;"),
    (0x060, "LDG.E.128 R4, desc[UR8][R2.64] ;"),
    (0x070, "BRA 0x100 ;"),
    # the loop header: fetch the next step, guarded for rows past the end
    (0x100, "ISETP.GE.AND P1, PT, R10, R11, PT ;"),
    (0x110, "IADD3 R10, R10, 0x1, RZ ;"),
    (0x120, "@P1 BRA 0x160 ;"),
    (0x130, "LDG.E.128 R8, desc[UR8][R2.64+0x100] ;"),
    (0x140, "LDG.E.64 R12, desc[UR8][R2.64+0x200] ;"),
    (0x150, "LD.E.128 R16, desc[UR8][R20.64] ;"),
    # step one: decode, a min-term MMA, two weight MMAs
    (0x160, "LOP3.LUT R24, R4, 0xf0f0f0f, RZ, 0xc0, !PT ;"),
    (0x170, "PRMT R25, R24, 0x5410, R5 ;"),
    (0x180, "FFMA R26, R25, R27, R28 ;"),
    (0x190, "HMMA.16816.F32.BF16 R32, R52, R16, RZ ;"),
    (0x1a0, "HMMA.16816.F32.BF16 R36, R24, R16, R36 ;"),
    (0x1b0, "HMMA.16816.F32.BF16 R40, R24, R18, R40 ;"),
    (0x1c0, "STL [R1], R26 ;"),
    (0x1d0, "ISETP.GE.AND P2, PT, R10, R11, PT ;"),
    (0x1e0, "UIADD3 UR4, UR4, 0x1, URZ ;"),
    (0x1f0, "@P2 BRA 0x400 ;"),
    # step two: the tail version checks between its MMAs, the plain one does not
    (0x200, "LDG.E.128 R4, desc[UR8][R2.64+0x300] ;"),
    (0x210, "ISETP.NE.AND P3, PT, R10, R11, PT ;"),
    (0x220, "@P3 BRA 0x280 ;"),
    (0x230, "HMMA.16816.F32.BF16 R36, R8, R16, R36 ;"),
    (0x240, "ISETP.GE.AND P4, PT, R10, R11, PT ;"),
    (0x250, "@P4 BRA 0x2c0 ;"),
    (0x260, "HMMA.16816.F32.BF16 R40, R8, R18, R40 ;"),
    (0x270, "BRA 0x2c0 ;"),
    (0x280, "HMMA.16816.F32.BF16 R36, R8, R16, R36 ;"),
    (0x290, "HMMA.16816.F32.BF16 R40, R8, R18, R40 ;"),
    (0x2a0, "SHFL.BFLY PT, R44, R26, 0x1, 0x1f ;"),
    (0x2b0, "NOP ;"),
    (0x2c0, "HMMA.16816.F32.BF16 R32, R52, R18, R32 ;"),
    (0x2d0, "LDS.U.128 R48, [R1+0x400] ;"),
    (0x2e0, "LDGSTS.E.BYPASS.128 [R1], desc[UR8][R2.64+0x400] ;"),
    (0x2f0, "ISETP.GE.AND P0, PT, R10, R11, PT ;"),
    (0x300, "@!P0 BRA 0x100 ;"),
    # the epilogue, a slow path called from it, and the exit
    (0x400, "FMUL R36, R36, R30 ;"),
    (0x410, "CALL.REL.NOINC 0x500 ;"),
    (0x420, "STG.E [R6.64], R36 ;"),
    (0x430, "EXIT ;"),
    (0x500, "MUFU.RCP R36, R36 ;"),
    (0x510, "RET.REL.NODEC R20 0x0 ;"),
    (0x520, "BRA 0x520 ;"),
]


def body_of(lines, words=None):
    return sc.parse_functions(sass(lines, words).splitlines())["_Z6kernelv"]


# Control codes read off an sm_86 build of the Q4_K register-MMA kernels (cuobjdump -sass of
# nvcc 13.2), decoded by hand from bits 41-57 of the high word: stall 41-44, yield 45, write
# scoreboard 46-48, read scoreboard 49-51 (7: none), wait mask 52-57.
#   0x000fe200000418ff (HMMA.16816.F32.BF16 R52, R52, R20, RZ): bits 40-63 are 0x000fe2, shifted
#     by one 0x7f1 = 0b111_111_1_0001: stall 1, yield 1, write 7, read 7, wait 0.
#   0x004ff600000418ff (an HMMA after a load): 0x004ff6 >> 1 = 0x27fb = 0b100_111_111_1_1011:
#     stall 11, yield 1, write 7, read 7, wait 0b000100 (scoreboard 2).
#   0x000ea2000c1e1d00 (LDG.E.128 R84, [R10.64]): 0x000ea2 >> 1 = 0x751 = 0b111_010_1_0001:
#     stall 1, yield 1, write 2, read 7, wait 0.
#   0x000fec0000010000 (BAR.SYNC.DEFER_BLOCKING 0x0): 0x7f6: stall 6, yield 1, none.
HMMA_WORD = 0x000FE200000418FF
HMMA_WAITS_SB2 = 0x004FF600000418FF
LDG_SETS_SB2 = 0x000EA2000C1E1D00
BAR_WORD = 0x000FEC0000010000

# The loop of KERNEL with a static schedule: the weight load at 0x130 sets scoreboard 2, the MMA
# at 0x1a0 waits on it; a local load at 0x1c0 sets scoreboard 3, the MMA at 0x2c0 waits on it; a
# local load at the bottom (0x2d0) sets scoreboard 4, the first decode of the next iteration
# (0x160) waits on it through the back edge. Outside the loop: a CTA barrier and a store that
# waits; the unreachable self-branch at 0x520 has stall 0, as ptxas writes it.
SCHEDULED = [(o, {0x1c0: "LDL R26, [R1] ;", 0x2d0: "LDL R48, [R1+0x400] ;",
                  0x400: "BAR.SYNC.DEFER_BLOCKING 0x0 ;"}.get(o, t)) for o, t in KERNEL]
SCHEDULED_WORDS = {
    0x130: LDG_SETS_SB2,
    0x190: HMMA_WORD,
    0x1a0: HMMA_WAITS_SB2,
    0x1c0: high_word(2, write=3),
    0x2c0: high_word(2, wait=1 << 3),
    0x2d0: high_word(2, write=4),
    0x160: high_word(2, wait=1 << 4),
    0x400: BAR_WORD,
    0x420: high_word(2, wait=1 << 1),
    0x520: high_word(0),
}


class LoopFinder(unittest.TestCase):
    def test_counts_of_the_heaviest_iteration(self):
        c = sc.loop_counts(body_of(KERNEL), tiles=2)
        self.assertEqual(c.loops, 1)
        # Both steps, the tail version of the second (one more ISETP and BRA than the plain one,
        # the same MMAs), the guarded loads: never the epilogue, the subroutine or the prologue.
        self.assertEqual(c.mma, 6)
        self.assertEqual(c.mma_ones, 2)
        self.assertEqual(c.mma_shape, "HMMA.16816.F32.BF16")
        # 4 weight MMAs of m16n8k16 over 2 tiles: 32 K per iteration.
        self.assertEqual(c.k_iter, 32)
        self.assertEqual((c.ldg, c.ldg_bytes), (3, 32 * (16 + 8 + 16)))
        self.assertEqual((c.ld, c.ld_bytes), (1, 512))
        self.assertEqual((c.lds, c.lds_bytes), (1, 512))
        self.assertEqual((c.ldgsts, c.ldgsts_bytes), (1, 512))
        self.assertEqual((c.local, c.local_bytes), (1, 128))
        self.assertEqual(c.shfl, 0)  # on the plain version only
        self.assertEqual(c.fp, 1)  # the FFMA; FMUL and MUFU are outside
        self.assertEqual(c.uniform, 1)
        # ISETP x5, IADD3, LOP3, PRMT
        self.assertEqual(c.int, 8)
        # BRA x6: the guard, the break, the version pick, the tail check, the tail's jump to
        # the join, the back edge
        self.assertEqual(c.other, 6)
        self.assertEqual(c.total, 6 + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 8 + 6)

    def test_plain_version_when_it_is_heavier(self):
        # Without its check the tail version is 3 instructions (HMMA, HMMA, BRA) against the plain
        # one's 4 (HMMA, HMMA, SHFL, NOP): the plain path is counted, SHFL included.
        lines = [l for l in KERNEL if l[0] not in (0x240, 0x250)]
        c = sc.loop_counts(body_of(lines), tiles=2)
        self.assertEqual(c.shfl, 1)
        self.assertEqual(c.mma, 6)

    def test_dead_code_takes_no_part(self):
        # A block the entry never reaches that branches into the loop body (a callee would):
        # neither a loop of its own nor a member of the K loop.
        dead = [(0x530, "HMMA.16816.F32.BF16 R36, R8, R16, R36 ;"), (0x540, "BRA 0x160 ;")]
        self.assertEqual(sc.loop_counts(body_of(KERNEL + dead), tiles=2),
                         sc.loop_counts(body_of(KERNEL), tiles=2))

    def test_branch_forms(self):
        # BRA.U on a uniform predicate and BRA.DIV fall through; PT is unconditional; !PT is never
        # taken; the uniform indirect branch is unknown flow; a target outside the function too.
        body = body_of(KERNEL)
        self.assertEqual(sc.branch_edges(False, "BRA.U", "!UP0, 0x530"), (True, True))
        self.assertEqual(sc.branch_edges(False, "BRA.DIV", "~URZ, 0x530"), (True, True))
        self.assertEqual(sc.branch_edges(False, "BRA.U", "UPT, 0x530"), (True, False))
        self.assertEqual(sc.branch_edges(False, "BRA", "!PT, 0x530"), (False, True))
        self.assertEqual(sc.branch_edges(True, "BRA", "0x530"), (True, True))
        self.assertEqual(sc.branch_edges(False, "BRA", "0x530"), (True, False))
        succ = sc.successors(body)
        self.assertEqual(succ[7], [8])  # BRA 0x100 at index 7: the loop header, index 8
        self.assertIsNone(sc.successors(body + [(0x530, False, "BRXU", "UR4 -0x530")]))
        self.assertIsNone(sc.successors(body + [(0x530, False, "BRA", "0x9000")]))

    def test_no_loop_and_unknown_flow(self):
        straight = [l for l in KERNEL if l[0] not in (0x2f0, 0x300)]
        c = sc.loop_counts(body_of(straight), tiles=2)
        self.assertEqual((c.loops, c.mma, c.total, c.k_iter), (0, 0, 0, 0))
        indirect = KERNEL + [(0x530, "BRX R2 -0x530 ;")]
        self.assertEqual(sc.loop_counts(body_of(indirect), tiles=2).loops, 0)
        self.assertEqual(sc.loop_counts([], tiles=2).loops, 0)

    def test_ffma_loop_without_tensor_mma(self):
        # Volta's build: the loop with the FFMAs stands in; no shape, no k_iter.
        lines = [(o, t.replace("HMMA.16816.F32.BF16", "FFMA").replace(", RZ ;", ", R0 ;"))
                 for o, t in KERNEL]
        c = sc.loop_counts(body_of(lines), tiles=2)
        self.assertEqual((c.loops, c.mma, c.k_iter, c.mma_shape), (1, 0, 0, "-"))
        self.assertEqual(c.fp, 7)

    def test_mixed_shapes_and_odd_k(self):
        lines = [(o, t.replace("HMMA.16816.F32.BF16 R40", "HMMA.1688.F32 R40")) for o, t in KERNEL]
        c = sc.loop_counts(body_of(lines), tiles=2)
        self.assertEqual(c.mma_shape, "HMMA.16816.F32.BF16:4+HMMA.1688.F32:2")
        # 2 x 2048 + 2 x 1024 = 6144 over 256 per tile: 24 K.
        self.assertEqual(c.k_iter, 24)
        self.assertEqual(sc.loop_counts(body_of(lines), tiles=5).k_iter, 0)

    def test_two_candidate_loops(self):
        # A second, unrelated MMA loop after the first: two candidates, the heavier one counted.
        second = [(0x600 + i * 0x10, t) for i, t in enumerate([
            "HMMA.16816.F32.BF16 R36, R8, R16, R36 ;",
            "ISETP.GE.AND P0, PT, R10, R11, PT ;",
            "@!P0 BRA 0x600 ;",
            "EXIT ;",
        ])]
        lines = [(o, t) if o != 0x430 else (o, "BRA 0x600 ;") for o, t in KERNEL] + second
        c = sc.loop_counts(body_of(lines), tiles=2)
        self.assertEqual(c.loops, 2)
        self.assertEqual(c.mma, 6)

    def test_ones_definitions(self):
        # A guarded move of the constant establishes nothing and clears the register; a write
        # whose first operand is a predicate (SHFL.IDX PT, R52) still clears it; HFMA2's constant
        # form counts only with the 1.0 halves; a copy carries it; a redefinition later in the loop
        # reaches the loop's top through the back edge.
        def ones(lines):
            body = body_of(lines)
            return sorted(body[i][0] for i in sc.ones_mma_indices(body, sc.successors(body)))

        self.assertEqual(ones(KERNEL), [0x190, 0x2c0])
        guarded = [(o, "@P0 " + t if o == 0x040 else t) for o, t in KERNEL]
        self.assertEqual(ones(guarded), [])
        shfl = KERNEL[:11] + [(0x155, "SHFL.IDX PT, R52, R53, 0x1, 0x1f ;")] + KERNEL[11:]
        self.assertEqual(ones(shfl), [])
        lop = KERNEL[:11] + [(0x155, "LOP3.LUT P0, R52, R53, 0xf, RZ, 0xc0, !PT ;")] + KERNEL[11:]
        self.assertEqual(ones(lop), [])
        hfma = [(o, "HFMA2 R52, -RZ, RZ, 1.5, 1.5 ;" if o == 0x040 else t) for o, t in KERNEL]
        self.assertEqual(ones(hfma), [])
        hfma = [(o, "HFMA2 R52, -RZ, RZ, 1.875, 1.875 ;" if o == 0x040 else t) for o, t in KERNEL]
        self.assertEqual(ones(hfma), [0x190, 0x2c0])
        copied = [(o, "MOV R60, R52 ;" if o == 0x050 else t) for o, t in KERNEL]
        copied = [(o, t.replace("R52, R18", "R60, R18")) for o, t in copied]
        self.assertEqual(ones(copied), [0x190, 0x2c0])
        # A copy with a modifier on its source (-R52, ~R52, |R52|) makes another pair, not ones:
        # the MMA reading R60 drops out, the one still reading R52 stays.
        for source in ("-R52", "~R52", "|R52|"):
            negated = [(o, t.replace("MOV R60, R52", f"IMAD.MOV.U32 R60, RZ, RZ, {source}"))
                       for o, t in copied]
            self.assertEqual(ones(negated), [0x190], source)
        reuse = [(o, t.replace("MOV R60, R52", "IMAD.MOV.U32 R60, RZ, RZ, R52.reuse"))
                 for o, t in copied]
        self.assertEqual(ones(reuse), [0x190, 0x2c0])
        # R52 rewritten after the MMAs inside the body: the first MMA of the next iteration reads
        # it through the back edge, so neither is ones.
        redefined = KERNEL[:38] + [(0x2d5, "IMAD R52, R52, R53, RZ ;")] + KERNEL[38:]
        self.assertEqual(ones(redefined), [])
        self.assertEqual(sc.operand_tokens("R60, R52.reuse, -RZ, 1.5, c[0x0][0x28], [R2.64+0x10]"),
                         ["R60", "R52", "RZ", "1.5", "c[0x0][0x28]", "[R2.64+0x10]"])
        self.assertEqual(sc.written_registers("SHFL.IDX", "PT, R4, R5, 0x1, 0x1f"), {"R4"})
        self.assertEqual(sc.written_registers("ISETP.GE.AND", "P0, PT, R0, R1, PT"), set())
        self.assertEqual(sc.written_registers("LDG.E.128", "R8, desc[UR8][R2.64]"),
                         {"R8", "R9", "R10", "R11"})
        # RED reduces into memory; REDUX writes a uniform register; the FP64 ops write a pair
        # whatever their suffix, F2F / I2F when the result type is F64.
        self.assertEqual(sc.written_registers("RED.E.ADD.F32.FTZ.RN.STRONG.GPU", "[R2.64], R5"),
                         set())
        self.assertEqual(sc.written_registers("REDUX.SUM", "UR4, R5"), {"UR4"})
        self.assertEqual(sc.written_registers("DADD", "R4, R6, R8"), {"R4", "R5"})
        self.assertEqual(sc.written_registers("DFMA", "R4, R6, R8, R10"), {"R4", "R5"})
        self.assertEqual(sc.written_registers("F2F.F64.F32", "R4, R6"), {"R4", "R5"})
        self.assertEqual(sc.written_registers("I2F.F64.S32", "R4, R6"), {"R4", "R5"})
        self.assertEqual(sc.written_registers("F2F.F32.F64", "R4, R6"), {"R4"})
        self.assertEqual(sc.written_registers("STG.E.128", "desc[UR8][R2.64], R8"), set())

    def test_irreducible_flow(self):
        # Two blocks that enter each other's middles (no header dominates the other): no natural
        # loop, so no K loop; nothing crashes.
        lines = [
            (0x000, "ISETP.GE.AND P0, PT, R0, R1, PT ;"),
            (0x010, "@P0 BRA 0x050 ;"),
            (0x020, "HMMA.16816.F32.BF16 R36, R8, R16, R36 ;"),
            (0x030, "ISETP.GE.AND P1, PT, R0, R1, PT ;"),
            (0x040, "@P1 BRA 0x070 ;"),
            (0x050, "HMMA.16816.F32.BF16 R40, R8, R18, R40 ;"),
            (0x060, "@P0 BRA 0x030 ;"),
            (0x070, "EXIT ;"),
        ]
        c = sc.loop_counts(body_of(lines), tiles=1)
        self.assertEqual((c.loops, c.mma), (0, 0))

    def test_classes_and_widths(self):
        self.assertEqual(sc.classify("IMAD.WIDE.U32"), "int")
        self.assertEqual(sc.classify("HADD2.BF16_V2"), "fp")
        self.assertEqual(sc.classify("ULDC.64"), "uniform")
        self.assertEqual(sc.classify("BSSY.RECONVERGENT"), "other")
        self.assertEqual(sc.classify("LDL.LU"), "local")
        self.assertEqual(sc.classify("QMMA.16864.F32.E2M1"), "mma")
        self.assertEqual([sc.access_bytes(o) for o in ("LDG.E.128", "LDS.64", "LDG.E.U8", "LDS")],
                         [16, 8, 1, 4])

    def test_row_round_trip(self):
        c = sc.loop_counts(body_of(KERNEL), tiles=2)
        self.assertEqual(sc.LoopCounts.from_row(c.row()), c)
        self.assertEqual(len(c.row()), len(sc.COLUMNS))
        loop_only = sc.LoopCounts.from_row(c.row()[:len(sc.LOOP_COLUMNS)])
        self.assertFalse(loop_only.scheduled)
        self.assertEqual(loop_only, c.without_schedule())


class StaticSchedule(unittest.TestCase):
    """The control codes (schema 4): the decode of hand-checked words, the sums over the counted
    iteration and the code outside the loop."""

    def test_control_word_decode(self):
        # (stall, yield, write scoreboard, read scoreboard, wait mask, reuse)
        self.assertEqual(sc.Control(HMMA_WORD).fields(), (1, 1, 7, 7, 0, 0))
        self.assertEqual(sc.Control(HMMA_WAITS_SB2).fields(), (11, 1, 7, 7, 0b000100, 0))
        self.assertEqual(sc.Control(LDG_SETS_SB2).fields(), (1, 1, 2, 7, 0, 0))
        self.assertEqual(sc.Control(BAR_WORD).fields(), (6, 1, 7, 7, 0, 0))
        self.assertEqual(sc.Control(DEFAULT_HIGH_WORD).fields(), (2, 1, 7, 7, 0, 0))
        # The reuse flags (58-61) and the instruction bits below 41 are not the control code.
        self.assertEqual(sc.Control(HMMA_WORD | 0xF << 58).fields(), (1, 1, 7, 7, 0, 0xF))
        self.assertEqual(sc.Control(high_word(15, 0, 5, 0, 0x3F)).fields(), (15, 0, 5, 0, 0x3F, 0))

    def test_high_word_follows_its_instruction(self):
        body = body_of(SCHEDULED, SCHEDULED_WORDS)
        self.assertEqual(body[17][0], 0x190)
        self.assertEqual(body[17][4], HMMA_WORD)
        self.assertTrue(all(ins[4] is not None for ins in body))
        # A text without the high words (an older dump) parses with None and no schedule.
        text = "\n".join(l for l in sass(KERNEL).splitlines() if "/*0" in l or "Function" in l)
        bare = sc.parse_functions(text.splitlines())["_Z6kernelv"]
        self.assertTrue(all(ins[4] is None for ins in bare))
        c = sc.loop_counts(bare, tiles=2)
        self.assertFalse(c.scheduled)
        self.assertEqual(c.without_schedule(), sc.loop_counts(body_of(KERNEL), tiles=2)
                         .without_schedule())

    def test_schedule_of_the_fixture(self):
        c = sc.loop_counts(body_of(SCHEDULED, SCHEDULED_WORDS), tiles=2)
        self.assertTrue(c.scheduled)
        self.assertEqual((c.loops, c.mma, c.total, c.local), (1, 6, 29, 2))
        # 29 instructions at stall 2, except the LDG and the first weight MMA at 1 and the MMA
        # that waits on the LDG at 11.
        self.assertEqual(c.stall_sum, 29 * 2 - 1 - 1 + 9)
        # Waits: 0x1a0 on the LDG's scoreboard, 0x2c0 on the LDL at 0x1c0, 0x160 on the LDL at
        # 0x2d0 of the previous iteration; the two last are local.
        self.assertEqual((c.sb_waits, c.sb_waits_local), (3, 2))
        # Outside: the prologue (0x000-0x070) and the epilogue (0x400-0x430); not the loop's
        # plain version of step two (0x280-0x2b0, a member of the loop off the counted path),
        # not the callee or the self-branch after EXIT.
        self.assertEqual(c.outside_total, 8 + 4)
        self.assertEqual(c.outside_stall, 11 * 2 + 6)
        self.assertEqual((c.outside_waits, c.outside_bar), (1, 1))
        self.assertEqual(c.schedule_problems(), [])
        self.assertGreaterEqual(c.stall_sum, c.total)

    def test_local_wait_needs_the_ldl(self):
        # The same scoreboard set by a shared load is no local wait; a wait in between clears it.
        lds = [(o, "LDS R26, [R1] ;" if o == 0x1c0 else t) for o, t in SCHEDULED]
        c = sc.loop_counts(body_of(lds, SCHEDULED_WORDS), tiles=2)
        self.assertEqual((c.sb_waits, c.sb_waits_local), (3, 1))
        words = dict(SCHEDULED_WORDS)
        words[0x200] = high_word(2, wait=1 << 4)  # waits on the bottom LDL's scoreboard first
        c = sc.loop_counts(body_of(SCHEDULED, words), tiles=2)
        self.assertEqual((c.sb_waits, c.sb_waits_local), (4, 2))  # 0x160 still local, 0x200 not

    def test_no_loop_counts_everything_outside(self):
        straight = [l for l in SCHEDULED if l[0] not in (0x2f0, 0x300)]
        c = sc.loop_counts(body_of(straight, SCHEDULED_WORDS), tiles=2)
        self.assertEqual((c.loops, c.total, c.stall_sum, c.sb_waits), (0, 0, 0, 0))
        self.assertTrue(c.scheduled)
        # Every reachable instruction: 0x000-0x070, 0x100-0x2e0 (0x2f0, 0x300 removed: the
        # alternative step two included), 0x400-0x430.
        self.assertEqual(c.outside_total, 8 + 31 + 4)
        self.assertEqual(c.schedule_problems(), [])

    def test_problems(self):
        c = sc.loop_counts(body_of(SCHEDULED, SCHEDULED_WORDS), tiles=2)
        c.stall_sum = c.total - 1
        self.assertIn("stall_sum", c.schedule_problems()[0])
        c = sc.loop_counts(body_of(SCHEDULED, SCHEDULED_WORDS), tiles=2)
        c.sb_waits_local = c.sb_waits + 1
        self.assertEqual(len(c.schedule_problems()), 1)
        c = sc.loop_counts(body_of(SCHEDULED, SCHEDULED_WORDS), tiles=2)
        c.outside_bar = c.outside_total + 1
        self.assertEqual(len(c.schedule_problems()), 1)
        self.assertEqual(c.without_schedule().schedule_problems(), [])


class Snapshot(unittest.TestCase):
    KEY = ("Bf16Decoder", 1, 4, 1, 1, 1, 0, 0, 0)

    def rows(self):
        c = sc.loop_counts(body_of(KERNEL), tiles=1)
        return {self.KEY: (48, 0, 0, 1536, c)}

    def test_render_and_parse(self):
        text = kernel_facts.render("89", self.rows(), "13.4.92", "-O3", "abc")
        header, rows = kernel_facts.parse_snapshot(text)
        self.assertEqual(header["arch"], "89")
        self.assertEqual(rows, self.rows())
        self.assertIn("columns " + kernel_facts.COLUMNS, text)
        self.assertEqual(sum(1 for l in text.splitlines() if not l.startswith("#")), 6 + 1)

    def test_schema_one_parses_without_loops(self):
        text = ("arch 89\nnvcc 13.4.92\nkernels 1\ncolumns " +
                " ".join(kernel_facts.COLUMNS.split()[:14]) +
                "\nrmma Bf16Decoder 1 4 1 1 1 0 0 0 48 0 0 1536\n")
        _, rows = kernel_facts.parse_snapshot(text)
        self.assertEqual(rows[self.KEY], (48, 0, 0, 1536, None))

    def test_loop_audit(self):
        rows = self.rows()
        self.assertEqual(kernel_facts.loop_audit(rows), [])
        other = ("Bf16Decoder", 4, 1, 1, 1, 1, 0, 0, 0)
        c = sc.LoopCounts()
        rows[other] = (48, 0, 0, 0, c)
        lines = kernel_facts.loop_audit(rows)
        self.assertEqual(len(lines), 3)
        self.assertIn("without a K loop", lines[0])
        self.assertIn("k_iter differs", lines[1])
        self.assertIn("1 of 2 kernels have no control codes", lines[2])
        c.scheduled = True
        c.stall_sum = -1
        self.assertIn("stall_sum -1 < total 0", kernel_facts.loop_audit(rows)[2])

    def test_rows_match_the_columns_line(self):
        text = kernel_facts.render("89", self.rows(), "13.4.92", "-O3", "abc")
        lines = text.splitlines()
        row = next(l for l in lines if l.startswith("rmma "))
        short = text.replace(row, " ".join(row.split()[:14]))
        with self.assertRaises(SystemExit):
            kernel_facts.parse_snapshot(short)  # a 14-field row under the loop columns
        columns = next(l for l in lines if l.startswith("columns "))
        schema1 = text.replace(columns, " ".join(columns.split()[:15]))
        with self.assertRaises(SystemExit):
            kernel_facts.parse_snapshot(schema1)  # a 37-field row under the resource columns
        with self.assertRaises(SystemExit):
            kernel_facts.parse_snapshot(text.replace(columns + "\n", ""))  # no columns line
        _, rows = kernel_facts.parse_snapshot(schema1.replace(row, " ".join(row.split()[:14])))
        self.assertEqual(rows[self.KEY][4], None)

    def test_name_batches(self):
        names = [f"_Z{i:04d}" + "x" * 90 for i in range(100)]  # 96 characters each
        batches = kernel_facts.name_batches(names, batch=256, max_chars=1000)
        self.assertEqual(sum(len(b) for b in batches), 100)
        self.assertTrue(all(len(",".join(b)) <= 1000 for b in batches))
        self.assertEqual(len(batches[0]), 10)  # 10 x 96 + 9 commas = 969
        self.assertEqual([len(b) for b in kernel_facts.name_batches(names, batch=30)], [30, 30, 30, 10])
        self.assertEqual(kernel_facts.name_batches(names, batch=0), [names])
        self.assertEqual(kernel_facts.name_batches(["a" * 50], batch=4, max_chars=10), [["a" * 50]])

    def test_key_tiles(self):
        self.assertEqual(kernel_facts.key_tiles(("D", 4, 1, 2, 4, 1, 0, 0, 0)), 8)
        self.assertEqual(kernel_facts.key_tiles(("D", 4, 1, 2, 4, 1, 2, 0, 0)), 16)


class Check(unittest.TestCase):
    """`check` against a saved resource dump and SASS: equal, a changed count, another nvcc, the
    resources alone, a schema 1 snapshot."""

    SYMBOL = "_ZN3ops11rmma_kernelIvEEvv"
    DUMP = "\n".join(("Fatbin elf code:", "================", "arch = sm_89", "",
                      "Resource usage:", " Common:", "  GLOBAL:0", " Function " + SYMBOL + ":",
                      "  REG:48 STACK:0 SHARED:1536 LOCAL:0", ""))

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        (self.root / "usage.txt").write_text(self.DUMP)
        (self.root / "sass.txt").write_text(sass(KERNEL).replace("_Z6kernelv", self.SYMBOL))
        # The demangler is the toolkit's: stand in for it, and for the symbol's key.
        self.saved = (kernel_facts.demangle, kernel_facts.rmma_key)
        kernel_facts.demangle = lambda names, cuda_bin: {n: n for n in names}
        kernel_facts.rmma_key = lambda name: Snapshot.KEY if name == self.SYMBOL else None

    def tearDown(self):
        kernel_facts.demangle, kernel_facts.rmma_key = self.saved
        self.dir.cleanup()

    def run_check(self, expected_text, nvcc="13.4.92", extra=()):
        expected = self.root / "expected.txt"
        expected.write_text(expected_text)
        argv = ["check", "--dump", str(self.root / "usage.txt"), "--sass-dump",
                str(self.root / "sass.txt"), "--expected", str(expected), "--nvcc", nvcc, *extra]
        return kernel_facts.main(argv)

    def expected_text(self, edit=None):
        rows = kernel_facts.extract_rows(None, None, "89", str(self.root / "usage.txt"),
                                         str(self.root / "sass.txt"))
        if edit:
            edit(rows)
        return kernel_facts.render("89", rows, "13.4.92", "-O3", "abc")

    def test_extract(self):
        rows = kernel_facts.extract_rows(None, None, "89", str(self.root / "usage.txt"),
                                         str(self.root / "sass.txt"))
        self.assertEqual(list(rows), [Snapshot.KEY])
        self.assertEqual(rows[Snapshot.KEY][:4], (48, 0, 0, 1536))
        self.assertEqual(rows[Snapshot.KEY][4].k_iter, 64)  # tiles 1: 4 weight MMAs x 16 K

    def test_verdicts(self):
        self.assertEqual(self.run_check(self.expected_text()), 0)

        def bump(rows):
            rows[Snapshot.KEY][4].mma += 1

        self.assertEqual(self.run_check(self.expected_text(bump)), 1)
        self.assertEqual(self.run_check(self.expected_text(bump), nvcc="13.0.1"), 77)
        self.assertEqual(self.run_check(self.expected_text(bump), extra=["--no-sass"]), 0)
        text = self.expected_text()
        resources_only = [" ".join(l.split()[:14]) if l.startswith("rmma ") else
                          " ".join(l.split()[:15]) if l.startswith("columns ") else l
                          for l in text.splitlines()]
        self.assertEqual(self.run_check("\n".join(resources_only) + "\n"), 0)

    def test_stamp(self):
        stamp = self.root / "stamps" / "sm_89"
        text = self.expected_text()
        self.assertEqual(self.run_check(text, extra=["--stamp", str(stamp)]), 0)
        self.assertTrue(stamp.is_file())
        # The same inputs pass again without extracting anything.
        saved = kernel_facts.extract_rows

        def no_extract(*args, **kwargs):
            raise AssertionError("extracted although the stamp holds")

        kernel_facts.extract_rows = no_extract
        try:
            self.assertEqual(self.run_check(text, extra=["--stamp", str(stamp)]), 0)
        finally:
            kernel_facts.extract_rows = saved

        def bump(rows):
            rows[Snapshot.KEY][4].mma += 1

        # Another snapshot (or nvcc) is another check: it runs, fails, and the stamp stays the
        # old pass's, so the next run checks again.
        before = stamp.read_text()
        self.assertEqual(self.run_check(self.expected_text(bump), extra=["--stamp", str(stamp)]), 1)
        self.assertEqual(stamp.read_text(), before)
        self.assertEqual(self.run_check(text, nvcc="13.0.1", extra=["--stamp", str(stamp)]), 77)
        self.assertEqual(stamp.read_text(), before)

    def test_stamp_without_inputs(self):
        # A missing snapshot is said as without a stamp, not a traceback.
        argv = ["check", "--dump", str(self.root / "usage.txt"), "--no-sass", "--expected",
                str(self.root / "none.txt"), "--stamp", str(self.root / "stamp")]
        self.assertEqual(kernel_facts.main(argv), 1)
        self.assertFalse((self.root / "stamp").exists())

    def older_schema(self, text, extra=()):
        """`text` (schema 4) rewritten as schema 2, or as schema 3 with `extra` loop columns."""
        n = len(kernel_facts.ROW_HEAD) + len(sc.LOOP_COLUMNS)
        out = []
        for l in text.splitlines():
            if l.startswith("rmma "):
                l = " ".join(l.split()[:n] + ["0"] * len(extra))
            elif l.startswith("columns "):
                l = " ".join(l.split()[:n + 1] + list(extra))
            out.append(l)
        return "\n".join(out) + "\n"

    def test_older_schemas_are_rejected(self):
        # Schema 2 (the loop without the schedule) and schema 3 (plus the critical path) are no
        # longer read: `check` fails on them instead of comparing without the schedule.
        text = self.expected_text()
        chain = ("cp_mma", "cp_ld", "cp_lds", "cp_shfl", "cp_imad", "cp_iadd", "cp_lop3",
                 "cp_prmt", "cp_shf", "cp_hfma2", "cp_ffma", "cp_cvt", "cp_f2f", "cp_other")
        for old in (self.older_schema(text), self.older_schema(text, chain)):
            with self.assertRaises(SystemExit):
                kernel_facts.parse_snapshot(old)
            with self.assertRaises(SystemExit):
                self.run_check(old)
        self.assertEqual(self.run_check(text), 0)

    def test_render_refuses_rows_without_a_schedule(self):
        rows = {k: (*v[:4], v[4].without_schedule()) for k, v in self.rows_of().items()}
        with self.assertRaises(SystemExit):
            kernel_facts.render("89", rows, "13.4.92", "-O3", "abc")

    def rows_of(self):
        return kernel_facts.extract_rows(None, None, "89", str(self.root / "usage.txt"),
                                         str(self.root / "sass.txt"))

    def test_schedule_is_compared(self):
        def later(rows):
            rows[Snapshot.KEY][4].stall_sum += 1

        self.assertEqual(self.run_check(self.expected_text(later)), 1)

    def test_impossible_schedule_fails(self):
        # The revert check: a snapshot whose stall sum is below its instruction count (or whose
        # waits exceed their instructions) is no kernel's and fails `check` before any compare.
        def short(rows):
            c = rows[Snapshot.KEY][4]
            c.stall_sum = c.total - 1

        with self.assertRaises(SystemExit) as raised:
            self.run_check(self.expected_text(short))
        self.assertEqual(raised.exception.code, 3)

        def waits(rows):
            c = rows[Snapshot.KEY][4]
            c.outside_waits = c.outside_total + 1

        with self.assertRaises(SystemExit):
            self.run_check(self.expected_text(waits))


class CommittedSnapshots(unittest.TestCase):
    """Every committed snapshot parses under the reader `check` uses, and every kernel's static
    schedule is one a kernel can have: the stall sums at least the instruction counts, the
    waits within their parts. Nothing ties a snapshot to another table."""

    def test_bounds(self):
        root = Path(__file__).resolve().parents[1] / "src" / "ops" / "generic" / "data" / \
            "kernel_facts"
        files = sorted(root.glob("sm_*.txt"))
        self.assertTrue(files)
        scheduled = 0
        for path in files:
            header, rows = kernel_facts.parse_snapshot(path.read_text())
            self.assertEqual(header["arch"], path.stem[3:])
            for key, value in rows.items():
                c = value[4]
                if c is None or not c.scheduled:
                    continue
                scheduled += 1
                self.assertGreaterEqual(c.stall_sum, c.total, f"{path.name} {key}")
                self.assertGreaterEqual(c.outside_stall, c.outside_total, f"{path.name} {key}")
                self.assertLessEqual(c.sb_waits_local, c.sb_waits, f"{path.name} {key}")
                self.assertEqual(c.schedule_problems(), [], f"{path.name} {key}")
        print(f"committed snapshots: {len(files)} files, {scheduled} kernels with a schedule")


def elf64(sections):
    """A minimal little-endian ELF64 file: the header, the sections' bytes, then the section
    header table (a null section, `sections` as (name, bytes), and .shstrtab)."""
    import struct
    names = b"\0" + b"".join(n.encode() + b"\0" for n, _ in sections) + b".shstrtab\0"
    body, offsets, at = b"", [], 64
    for _, data in sections:
        offsets.append(at + len(body))
        body += data
    names_at = at + len(body)
    body += names
    shoff = at + len(body)
    headers = [bytes(64)]
    name_at = 1
    for (n, data), offset in zip(sections, offsets):
        headers.append(struct.pack("<IIQQQQIIQQ", name_at, 1, 0, 0, offset, len(data), 0, 0, 1, 0))
        name_at += len(n) + 1
    headers.append(struct.pack("<IIQQQQIIQQ", name_at, 3, 0, 0, names_at, len(names), 0, 0, 1, 0))
    head = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9)
    head += struct.pack("<HHIQQQIHHHHHH", 2, 62, 1, 0, 0, shoff, 0, 64, 0, 0, 64, len(headers),
                        len(headers) - 1)
    return head + body + b"".join(headers)


class StampKey(unittest.TestCase):
    """The stamp's key reads a binary's device code (.nv_fatbin): stripping it or setting its run
    path leaves the key; other device code changes it."""

    def key(self, root, data):
        binary = root / "engine"
        binary.write_bytes(data)
        expected = root / "expected.txt"
        expected.write_text("snapshot\n")

        class Args:
            pass

        a = Args()
        a.binary, a.dump, a.sass_dump, a.expected = str(binary), None, None, str(expected)
        a.nvcc, a.no_sass, a.cuda_bin, a.sass_batch = "13.4.92", False, None, 64
        return kernel_facts.check_key(a)

    def test_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fat = b"fatbin image"
            built = elf64([(".text", b"code" * 8), (".symtab", b"symbols"), (".nv_fatbin", fat)])
            (root / "e").write_bytes(built)
            self.assertEqual(kernel_facts.elf_section(root / "e", ".nv_fatbin"), fat)
            self.assertIsNone(kernel_facts.elf_section(root / "e", ".nv_missing"))
            # Stripped (no .symtab) and with another run path (other .dynamic bytes): same key.
            stripped = elf64([(".text", b"code" * 8), (".dynamic", b"rpath=$ORIGIN/../lib"),
                              (".nv_fatbin", fat)])
            self.assertEqual(self.key(root, built), self.key(root, stripped))
            other = elf64([(".text", b"code" * 8), (".nv_fatbin", b"fatbin imagf")])
            self.assertNotEqual(self.key(root, built), self.key(root, other))
            # A relocatable image the objects left counts too; the images' addresses do not.
            rel = elf64([(".nv_fatbin", fat), ("__nv_relfatbin", b"object image")])
            rel2 = elf64([(".nv_fatbin", fat), ("__nv_relfatbin", b"object imagf")])
            self.assertNotEqual(self.key(root, rel), self.key(root, rel2))
            moved = elf64([(".nv_fatbin", fat), (".nvFatBinSegment", b"addresses 1")])
            moved2 = elf64([(".nv_fatbin", fat), (".nvFatBinSegment", b"addresses 2")])
            self.assertEqual(self.key(root, moved), self.key(root, moved2))
            # Not ELF: the whole file.
            (root / "t").write_bytes(b"not an elf")
            self.assertIsNone(kernel_facts.elf_section(root / "t", ".nv_fatbin"))
            self.assertNotEqual(self.key(root, b"text a"), self.key(root, b"text b"))


if __name__ == "__main__":
    unittest.main()
