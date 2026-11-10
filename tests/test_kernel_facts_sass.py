"""tools/kernel_facts/sass_counts.py and the loop columns of kernel_facts.py (#613 PR-3): the
K loop found from the control flow of a small SASS text shaped like rmma_kernel's (a prologue,
the double-buffered loop with its break in the middle, a row-bound guard around the loads, an
alternative version of a step, an epilogue, a subroutine after EXIT), the heaviest path counted
once, the min-term MMAs told apart, k_iter from the weight MMAs; then the snapshot round trip and
the check's verdicts. No GPU, no CUDA toolkit."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "kernel_facts"))
import kernel_facts  # noqa: E402
import sass_counts as sc  # noqa: E402


def sass(lines):
    """A `cuobjdump -sass` function body from (offset, text) pairs, in cuobjdump's layout."""
    out = ["\tcode for sm_89", "\t\tFunction : _Z6kernelv"]
    for offset, text in lines:
        out.append(f"        /*{offset:04x}*/ {text:60s} /* 0x0000000000000000 */")
        out.append(" " * 80 + "/* 0x000fe40000000f00 */")
    return "\n".join(out) + "\n"


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


def body_of(lines):
    return sc.parse_functions(sass(lines).splitlines())["_Z6kernelv"]


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

    def test_critical_path(self):
        # The fixture's two HMMAs accumulate into R32 one after the other, so both sit on the
        # longest chain; every class count is bounded by the class's own count, and the path's
        # counts round-trip through the row like the others.
        c = sc.loop_counts(body_of(KERNEL), tiles=1)
        self.assertEqual(c.cp_mma, 2)
        int_classes = ("cp_imad", "cp_iadd", "cp_lop3", "cp_prmt", "cp_shf")
        self.assertGreaterEqual(sum(getattr(c, f) for f in sc.CP_FIELDS) - c.cp_mma, 1)
        self.assertLessEqual(sum(getattr(c, f) for f in int_classes), c.int)
        self.assertLessEqual(c.cp_hfma2 + c.cp_ffma, c.fp)
        self.assertEqual(sc.LoopCounts.from_row(c.row()), c)
        # One class per measured latency: an instruction is priced at its own fact, the ones
        # with none at their family's (IMAD for the integer ops, HFMA2 for FP16x2, FFMA for FP32).
        for opcode, name in (
                ("HMMA.16816.F32.BF16", "cp_mma"), ("LDS.128", "cp_lds"), ("LDG.E.128", "cp_ld"),
                ("SHFL.IDX", "cp_shfl"), ("IMAD.WIDE.U32", "cp_imad"), ("MOV", "cp_imad"),
                ("ISETP.GE.AND", "cp_imad"), ("IADD3", "cp_iadd"), ("IADD3.X", "cp_iadd"),
                ("LOP3.LUT", "cp_lop3"), ("PRMT", "cp_prmt"), ("SHF.R.U32.HI", "cp_shf"),
                ("HFMA2", "cp_hfma2"), ("HADD2.F32", "cp_hfma2"), ("FFMA", "cp_ffma"),
                ("FMUL", "cp_ffma"), ("I2FP.F32.S32", "cp_cvt"), ("F2I.TRUNC.NTZ", "cp_cvt"),
                ("F2F.F16.F32", "cp_f2f"), ("F2FP.BF16.F32.PACK_AB", "cp_f2f"),
                ("BRA", "cp_other")):
            self.assertEqual(sc.cp_class(opcode), name, opcode)
        self.assertEqual(set(sc.CP_NOMINAL), set(sc.CP_FIELDS))
        # A chain of three dependent IMADs is three deep; two independent ones are one.
        chain = [(0x000, "IMAD R1, R0, R0, RZ ;"), (0x010, "IMAD R2, R1, R1, RZ ;"),
                 (0x020, "IMAD R3, R2, R2, RZ ;"), (0x030, "IMAD R4, R0, R0, RZ ;")]
        counts, short = sc.critical_path(body_of(chain), [0, 1, 2, 3])
        self.assertEqual((counts["cp_imad"], short), (3, 0.0))

    def test_critical_path_follows_the_latencies(self):
        # Two chains from R0: a load into an IMAD, and five converts. Which is the longest is a
        # matter of the clocks per class: nominally the load's (48 + 4.5 over 5 x 6), with a
        # card's measured 11.5-clock converts the converts' (57.5 over 46.987 + 4.033). The path
        # is the first weight set's; the others say by how many clocks it is not theirs.
        lines = [(0x000, "LDG.E R1, desc[UR8][R0.64] ;"), (0x010, "IMAD R2, R1, R1, RZ ;"),
                 (0x020, "I2F R3, R0 ;"), (0x030, "F2I R4, R3 ;"), (0x040, "I2F R5, R4 ;"),
                 (0x050, "F2I R6, R5 ;"), (0x060, "I2F R7, R6 ;")]
        body, order = body_of(lines), list(range(len(lines)))
        isa = {"mma_dependent_latency": 32.893, "l1_latency_w1": 46.987, "l1_latency_w8": 50.0,
               "shared_latency": 23.106, "lat_shfl": 26.002, "lat_imad": 4.033, "lat_iadd": 5.029,
               "lat_lop3": 4.033, "lat_prmt": 4.053, "lat_shf": 4.033, "lat_hfma2": 4.11,
               "lat_ffma": 4.033, "lat_cvt_i2f_f2i": 11.5, "lat_cvt_f32_f16": 5.047}
        measured = sc.chain_weights(isa)
        self.assertEqual((measured["cp_ld"], measured["cp_lds"], measured["cp_cvt"]),
                         (46.987, 23.106, 11.5))
        self.assertEqual((measured["cp_iadd"], measured["cp_prmt"], measured["cp_f2f"]),
                         (5.029, 4.053, 5.047))
        self.assertEqual(sc.chain_weights(isa, 8)["cp_ld"], 50.0)
        # A row without one of the ALU latencies (an architecture not probed for them) gives no
        # weights.
        for fact in ("lat_imad", "lat_iadd", "lat_ffma", "lat_cvt_f32_f16"):
            self.assertIsNone(sc.chain_weights({k: v for k, v in isa.items() if k != fact}), fact)
        self.assertIsNone(sc.chain_weights(isa, 4))
        nominal, short = sc.critical_path(body, order)
        self.assertEqual((nominal["cp_ld"], nominal["cp_imad"], nominal["cp_cvt"], short),
                         (1, 1, 0, 0.0))
        chosen, short = sc.critical_path(body, order, (measured,))
        self.assertEqual((chosen["cp_ld"], chosen["cp_imad"], chosen["cp_cvt"], short),
                         (0, 0, 5, 0.0))
        self.assertAlmostEqual(sc.chain_clocks(chosen, measured), 57.5)
        # Chosen nominally and checked against the measured set: 57.5 - (46.987 + 4.033) short.
        counts, short = sc.critical_path(body, order, (sc.CP_NOMINAL, measured))
        self.assertEqual(counts, nominal)
        self.assertAlmostEqual(short, 57.5 - 51.02)
        # The other way round the chosen chain is the measured set's, and nominally 22.5 short.
        counts, short = sc.critical_path(body, order, (measured, sc.CP_NOMINAL))
        self.assertEqual(counts, chosen)
        self.assertAlmostEqual(short, 52.5 - 30.0)
        c = sc.loop_counts(body_of(KERNEL), tiles=1, weights=(measured, sc.CP_NOMINAL))
        self.assertGreaterEqual(c.cp_short, 0.0)
        # Each convert at its own latency: five F32 / F16 converts (5.047 clocks each, 25.235)
        # are a shorter chain than three I2F / F2I ones (11.5 each, 34.5), and an IADD3 chain
        # (5.029) is longer than as many IMADs (4.033).
        lines = [(0x000, "F2F.F16.F32 R1, R0 ;"), (0x010, "F2F.F32.F16 R2, R1 ;"),
                 (0x020, "F2F.F16.F32 R3, R2 ;"), (0x030, "F2F.F32.F16 R4, R3 ;"),
                 (0x040, "F2F.F16.F32 R5, R4 ;"), (0x050, "I2F R6, R0 ;"), (0x060, "F2I R7, R6 ;"),
                 (0x070, "I2F R8, R7 ;")]
        chosen, _ = sc.critical_path(body_of(lines), list(range(len(lines))), (measured,))
        self.assertEqual((chosen["cp_f2f"], chosen["cp_cvt"]), (0, 3))
        self.assertAlmostEqual(sc.chain_clocks(chosen, measured), 34.5)
        lines = [(0x000, "IADD3 R1, R0, R0, RZ ;"), (0x010, "IADD3 R2, R1, R1, RZ ;"),
                 (0x020, "IMAD R3, R0, R0, RZ ;"), (0x030, "IMAD R4, R3, R3, RZ ;")]
        chosen, _ = sc.critical_path(body_of(lines), list(range(len(lines))), (measured,))
        self.assertEqual((chosen["cp_iadd"], chosen["cp_imad"]), (2, 0))

    def test_reads_and_writes(self):
        rw = sc.reads_and_writes
        # An MMA names its fragments by their first register: m16n8k16 of bf16 reads four of A,
        # two of B and the accumulator it writes; an f16 accumulator is two registers; the s8
        # and tf32 shapes hold as many per lane, m16n8k8 of f16 half of them.
        self.assertEqual(rw("", "HMMA.16816.F32.BF16", "R36, R24.reuse, R16, R36"),
                         ({"R24", "R25", "R26", "R27", "R16", "R17", "R36", "R37", "R38", "R39"},
                          {"R36", "R37", "R38", "R39"}))
        self.assertEqual(rw("", "HMMA.16816.F16", "R36, R24, R16, RZ"),
                         ({"R24", "R25", "R26", "R27", "R16", "R17"}, {"R36", "R37"}))
        self.assertEqual([sc.mma_fragments(o) for o in (
            "HMMA.16816.F32", "HMMA.1688.F32", "HMMA.1688.F32.TF32", "HMMA.1684.F32.TF32",
            "IMMA.16832.S8.S8", "IMMA.8816.S8.S8", "IMMA.16864.S4.S4", "HMMA.884.F32.F32.STEP0",
            "DMMA.884", "QMMA.16832.F32.E4M3.E4M3", "HMMA.99.F32")],
            [(4, 2), (2, 1), (4, 2), (2, 1), (4, 2), (1, 1), (4, 2), (2, 2), (2, 2), (4, 2),
             (1, 1)])
        # A compare writes its leading predicates and reads the one it combines with; PLOP3's
        # third and fourth predicates are sources; FCHK writes one predicate and no register.
        self.assertEqual(rw("", "ISETP.GT.AND", "P1, PT, R5.reuse, 0x7, !P0"),
                         ({"R5", "P0"}, {"P1"}))
        self.assertEqual(rw("", "UISETP.NE.U32.AND", "UP0, UPT, UR4, URZ, UPT"),
                         ({"UR4"}, {"UP0"}))
        self.assertEqual(rw("", "ISETP.NE.U64.AND", "P0, PT, R4, RZ, PT"),
                         ({"R4", "R5"}, {"P0"}))
        self.assertEqual(rw("", "PLOP3.LUT", "P0, PT, P2, P3, PT, 0x2f, 0xf2"),
                         ({"P2", "P3"}, {"P0"}))
        self.assertEqual(rw("", "FCHK", "P0, R4, R5"), ({"R4", "R5"}, {"P0"}))
        self.assertEqual(rw("", "HSETP2.GT.AND", "P0, P1, R4, R5, PT"),
                         ({"R4", "R5"}, {"P0", "P1"}))
        # The predicates around a register destination are written (LOP3's zero test, the
        # carries out); the trailing ones are read (the carries in, a selector).
        self.assertEqual(rw("", "LOP3.LUT", "P0, RZ, R4, UR5, RZ, 0xc0, !PT"),
                         ({"R4", "UR5"}, {"P0"}))
        self.assertEqual(rw("", "IADD3", "R4, P0, P1, R1, 0x10, -R2"),
                         ({"R1", "R2"}, {"R4", "P0", "P1"}))
        self.assertEqual(rw("", "IADD3.X", "R5, R6, R7, RZ, P0, P1"),
                         ({"R6", "R7", "P0", "P1"}, {"R5"}))
        self.assertEqual(rw("", "LEA", "R4, P0, R1.reuse, R2, 0x2"), ({"R1", "R2"}, {"R4", "P0"}))
        self.assertEqual(rw("", "LEA.HI.X", "R5, R1, R3, RZ, 0x2, P0"),
                         ({"R1", "R3", "P0"}, {"R5"}))
        self.assertEqual(rw("", "SEL", "R4, RZ, 0x1, !P2"), ({"P2"}, {"R4"}))
        self.assertEqual(rw("", "SEL.64", "R4, R6, RZ, P2"), ({"R6", "R7", "P2"}, {"R4", "R5"}))
        self.assertEqual(rw("", "SHFL.IDX", "PT, R4, R5, R6, R7"), ({"R5", "R6", "R7"}, {"R4"}))
        self.assertEqual(rw("", "P2R", "R4, PR, R5, 0x7f"),
                         ({"R5"} | set(sc.PREDICATE_REGISTERS), {"R4"}))
        self.assertEqual(rw("", "R2P", "PR, R4, 0x7f"), ({"R4"}, set(sc.PREDICATE_REGISTERS)))
        # The mask names the predicates: 0x20 is P5 alone, a register mask may be any of them.
        self.assertEqual(rw("", "P2R", "R0, PR, RZ, 0x20"), ({"P5"}, {"R0"}))
        self.assertEqual(rw("", "R2P", "PR, R4, 0x3"), ({"R4"}, {"P0", "P1"}))
        self.assertEqual(rw("", "R2P", "PR, R4, R6"), ({"R4", "R6"}, set(sc.PREDICATE_REGISTERS)))
        # The 64-bit read of a special register writes a pair, the 32-bit one a register.
        self.assertEqual(rw("", "CS2R", "R112, SRZ"), (set(), {"R112", "R113"}))
        self.assertEqual(rw("", "CS2R.32", "R112, SR_CLOCKLO"), (set(), {"R112"}))
        # A branch reads its condition, as an operand or as its guard.
        self.assertEqual(rw("", "BRA", "!P1, 0x100"), ({"P1"}, set()))
        self.assertEqual(rw("", "BRA.U", "UP0, 0x100"), ({"UP0"}, set()))
        self.assertEqual(rw("!P1", "BRA", "0x100"), ({"P1"}, set()))
        # 64-bit operands are pairs: an address register written .64, the operands of a 64-bit
        # add, the addend of a wide multiply, a store's data as wide as the access; SHF names
        # both halves itself.
        self.assertEqual(rw("", "LDG.E.128", "R8, desc[UR8][R2.64+0x100]"),
                         ({"UR8", "R2", "R3"}, {"R8", "R9", "R10", "R11"}))
        self.assertEqual(rw("", "LDGSTS.E.BYPASS.128", "[R1+0x10], desc[UR8][R2.64], P0"),
                         ({"R1", "UR8", "R2", "R3", "P0"}, set()))
        self.assertEqual(rw("", "IADD.64", "R4, R6, UR8"),
                         ({"R6", "R7", "UR8", "UR9"}, {"R4", "R5"}))
        self.assertEqual(rw("", "IMAD.WIDE.U32", "R4, R6, UR8, R10"),
                         ({"R6", "UR8", "R10", "R11"}, {"R4", "R5"}))
        self.assertEqual(rw("", "STG.E.128", "desc[UR8][R2.64], R8"),
                         ({"UR8", "R2", "R3", "R8", "R9", "R10", "R11"}, set()))
        self.assertEqual(rw("", "STL.64", "[R1+0x8], R6"), ({"R1", "R6", "R7"}, set()))
        self.assertEqual(rw("", "SHF.L.U64.HI", "R4, R5, 0x2, R6"), ({"R5", "R6"}, {"R4"}))
        self.assertEqual(rw("", "F2F.F32.F64", "R4, R6"), ({"R6", "R7"}, {"R4"}))
        self.assertEqual(rw("", "F2F.F64.F32", "R4, R6"), ({"R6"}, {"R4", "R5"}))
        self.assertEqual(rw("", "DADD", "R4, R6, R8"), ({"R6", "R7", "R8", "R9"}, {"R4", "R5"}))
        # Under a guard the instruction also reads the guard; under !PT it is a no-op.
        self.assertEqual(rw("!P0", "IADD.64", "R4, R4, R6"),
                         ({"P0", "R4", "R5", "R6", "R7"}, {"R4", "R5"}))
        self.assertEqual(rw("P3", "MOV", "R2, R5"), ({"P3", "R5"}, {"R2"}))
        self.assertEqual(rw("!PT", "LDS", "RZ, [RZ]"), (set(), set()))
        self.assertEqual(rw("!UPT", "UIADD3", "URZ, UPT, UPT, URZ, URZ, URZ"), (set(), set()))

    def test_dependence_graph(self):
        def preds(lines):
            return sc.dependence_graph(body_of(lines), list(range(len(lines))))[1]

        # The parser keeps the guard's name; the instruction's own predicates stay operands.
        body = body_of([(0x000, "@!P0 IADD3 R2, R3, 0x1, RZ ;"), (0x010, "@PT NOP ;"),
                        (0x020, "@!PT LDS RZ, [RZ] ;"), (0x030, "MOV R1, R2 ;")])
        self.assertEqual([guard for _, guard, _, _ in body], ["!P0", "", "!PT", ""])
        # A producer of any register of a fragment is a dependence of the MMA: R25 of the A
        # fragment R24..R27, R17 of the B fragment R16..R17, R39 of the accumulator; R18 is
        # outside the B fragment.
        self.assertEqual(preds([
            (0x000, "PRMT R25, R4, 0x5410, R5 ;"),
            (0x010, "LOP3.LUT R17, R4, 0xf, RZ, 0xc0, !PT ;"),
            (0x020, "FFMA R39, R1, R2, R3 ;"), (0x030, "IMAD R18, R0, R0, RZ ;"),
            (0x040, "HMMA.16816.F32.BF16 R36, R24, R16, R36 ;")])[4], [0, 1, 2])
        # A compare feeds the instruction it guards and the branch on it. A guarded write
        # replaces the value on some lanes only: the reader of R2 depends on the MOV and on the
        # IMAD before it, side by side (the MOV does not wait for the IMAD's R2); an unguarded
        # write ends that, and two guarded loads into one register are not a chain.
        self.assertEqual(preds([
            (0x000, "IMAD R2, R0, R0, RZ ;"), (0x010, "ISETP.GE.AND P0, PT, R1, R11, PT ;"),
            (0x020, "@P0 MOV R2, R5 ;"), (0x030, "IADD3 R6, R2, 0x1, RZ ;"),
            (0x040, "BRA !P0, 0x000 ;"), (0x050, "@!P0 BRA 0x000 ;"),
            (0x060, "MOV R2, R7 ;"), (0x070, "IADD3 R8, R2, 0x1, RZ ;")]),
            [[], [], [1], [0, 2], [1], [1], [], [6]])
        self.assertEqual(preds([
            (0x000, "@P0 LDG.E R4, desc[UR8][R2.64] ;"),
            (0x010, "@P1 LDG.E R4, desc[UR8][R6.64] ;"),
            (0x020, "IMAD R5, R4, R4, RZ ;")]), [[], [], [0, 1]])
        # A zeroed pair ends the reach of what its registers held: the B fragment R112..R113
        # of the MMA comes from the guarded load and the zeroes it may leave, not from the MMA
        # that wrote R112..R115 before them.
        self.assertEqual(preds([
            (0x000, "HMMA.16816.F32.BF16 R112, R68, R120, RZ ;"), (0x010, "CS2R R112, SRZ ;"),
            (0x020, "CS2R R114, SRZ ;"), (0x030, "@P4 LD.E.128 R112, desc[UR8][R42.64] ;"),
            (0x040, "HMMA.16816.F32.BF16 R56, R52, R112, RZ ;")])[4], [1, 3])
        # Under the writer's own guard a reader sees that writer alone (the lanes are the
        # same); under the opposite guard none of it; once the predicate is written again the
        # guards are no longer the same lanes.
        lanes = [(0x000, "FFMA R4, R1, R2, R3 ;"), (0x010, "@!P3 IADD3 R4, R5, 0x1, RZ ;"),
                 (0x020, "@!P3 IMAD R6, R4, R4, RZ ;"), (0x030, "@P3 IMAD R7, R4, R4, RZ ;"),
                 (0x040, "@P1 IMAD R8, R4, R4, RZ ;"),
                 (0x050, "ISETP.GE.AND P3, PT, R9, R11, PT ;"),
                 (0x060, "@!P3 IMAD R10, R4, R4, RZ ;")]
        self.assertEqual(preds(lanes), [[], [], [1], [0], [0, 1], [], [0, 1, 5]])
        # A guard saved in a register and read back is the guard it was: the load under the
        # restored P5 takes its address from the two moves under the first P5 and not from the
        # MMA they replaced. Read back from a register that holds anything else, it is not.
        saved = [(0x000, "HMMA.16816.F32.BF16 R80, R112, R82, R120 ;"),
                 (0x010, "ISETP.GE.AND P5, PT, R68, R9, PT ;"), (0x020, "P2R R78, PR, RZ, 0x20 ;"),
                 (0x030, "@!P5 MOV R80, R2 ;"), (0x040, "@!P5 MOV R81, R3 ;"),
                 (0x050, "ISETP.GE.AND P5, PT, R4, R5, PT ;"),
                 (0x060, "ISETP.NE.AND P5, PT, R78, RZ, PT ;"),
                 (0x070, "@!P5 LD.E.128 R100, desc[UR6][R80.64] ;")]
        self.assertEqual(preds(saved)[7], [3, 4, 6])
        saved[2] = (0x020, "P2R R78, PR, RZ, 0x60 ;")  # two predicates: not P5's value alone
        self.assertEqual(preds(saved)[7], [0, 3, 4, 6])
        saved[2] = (0x020, "P2R R78, PR, RZ, 0x20 ;")
        saved[5] = (0x050, "IADD3 R78, R78, 0x1, RZ ;")
        self.assertEqual(preds(saved)[7], [0, 3, 4, 6])
        # The same through one bit of a word: cleared, set under the guard, read back. Under
        # the other polarity the reader sees none of the guarded write; a bit set under @!P1
        # holds the negation.
        def word(set_guard, write_guard, read_guard, clear="0xfffffffe"):
            return preds([
                (0x000, "FFMA R84, R1, R2, R84 ;"), (0x010, "ISETP.GE.AND P1, PT, R0, R13, PT ;"),
                (0x020, f"LOP3.LUT R25, R25, {clear}, RZ, 0xc0, !PT ;"),
                (0x030, f"{set_guard} LOP3.LUT R25, R25, 0x1, RZ, 0xfc, !PT ;"),
                (0x040, f"{write_guard} MOV R84, R7 ;"),
                (0x050, "ISETP.GE.AND P1, PT, R4, R5, PT ;"),
                (0x060, "LOP3.LUT P1, RZ, R25, 0x1, RZ, 0xc0, !PT ;"),
                (0x070, f"{read_guard} IADD3 R9, R84, 0x8, RZ ;")])[7]

        self.assertEqual(word("@P1", "@!P1", "@!P1"), [4, 6])
        self.assertEqual(word("@P1", "@P1", "@!P1"), [0, 6])
        self.assertEqual(word("@!P1", "@!P1", "@P1"), [4, 6])
        self.assertEqual(word("@!P1", "@!P1", "@!P1"), [0, 6])
        # A bit not known clear before it is set holds more than the guard.
        self.assertEqual(word("@P1", "@!P1", "@!P1", clear="0xfffffffd"), [0, 4, 6])
        # The carry of a wide add links its low and high halves, as LEA / LEA.HI.X do.
        self.assertEqual(preds([
            (0x000, "IADD3 R4, P0, P1, R1, R2, RZ ;"), (0x010, "IADD3.X R5, R6, R7, RZ, P0, P1 ;"),
            (0x020, "LEA R8, P2, R4, R2, 0x2 ;"), (0x030, "LEA.HI.X R9, R4, R3, RZ, 0x2, P2 ;")]),
            [[], [0], [0], [0, 2]])
        # The high half of a 64-bit address is a dependence of the load.
        self.assertEqual(preds([
            (0x000, "IADD3.X R3, R6, R7, RZ, P0, P1 ;"), (0x010, "LDG.E R8, desc[UR8][R2.64] ;")]),
            [[], [0]])
        # The fixture's first step (0x160 .. 0x1b0): both weight MMAs take R24 from the LOP3 and
        # R25 from the PRMT after it, and R26 from the FFMA; the min-term MMA none of them.
        step = [l for l in KERNEL if 0x160 <= l[0] <= 0x1b0]
        self.assertEqual(preds(step), [[], [0], [1], [], [0, 1, 2], [0, 1, 2]])

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
        # With the latencies that chose the critical paths: two more header lines, the same rows.
        rows = self.rows()
        rows[self.KEY][4].cp_short = 1.25
        chain = kernel_facts.chain_line((sc.CP_NOMINAL,), "nominal")
        self.assertEqual(chain, "nominal mma=33 ld=48 lds=25 shfl=27 imad=4.5 iadd=4.5 lop3=4.5 "
                                "prmt=4.5 shf=4.5 hfma2=4.1 ffma=4.1 cvt=6 f2f=6 other=2")
        text = kernel_facts.render("89", rows, "13.4.92", "-O3", "abc", chain)
        header, parsed = kernel_facts.parse_snapshot(text)
        self.assertEqual((header["chain"], header["chain_short"]), (chain, "1 1.250"))
        self.assertEqual(parsed, self.rows())

    def test_chain_weight_sets(self):
        # An architecture whose row holds every latency is chosen with them, one set per resident
        # warp count; one without a row, or whose row lacks one, with the nominal set alone.
        facts = {"mma_dependent_latency": 32.9, "shared_latency": 23.1, "lat_shfl": 26.0,
                 "lat_imad": 4.0, "lat_iadd": 5.0, "lat_lop3": 4.0, "lat_prmt": 4.0,
                 "lat_shf": 4.0, "lat_hfma2": 4.1, "lat_ffma": 4.0, "lat_cvt_i2f_f2i": 11.5,
                 "lat_cvt_f32_f16": 5.0,
                 **{f"l1_latency_w{w}": 46.0 + w for w in sc.CP_WARPS}}
        table = {"arches": {
            "89": {"facts": {k: {"value": v, "unit": "clk"} for k, v in facts.items()}},
            "70": {"facts": {k: {"value": v, "unit": "clk"} for k, v in facts.items()
                             if not k.startswith("lat_")}}}}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "isa.json")
            with open(path, "w") as f:
                json.dump(table, f)
            weights, basis = kernel_facts.chain_weight_sets("89", path)
            self.assertEqual((basis, [w["cp_ld"] for w in weights]),
                             ("measured", [47.0, 48.0, 50.0, 54.0]))
            self.assertEqual(kernel_facts.chain_line(weights, basis),
                             "measured mma=32.9 ld=47 lds=23.1 shfl=26 imad=4 iadd=5 lop3=4 prmt=4 "
                             "shf=4 hfma2=4.1 ffma=4 cvt=11.5 f2f=5 other=4")
            for arch in ("70", "75"):
                self.assertEqual(kernel_facts.chain_weight_sets(arch, path),
                                 ((sc.CP_NOMINAL,), "nominal"))

    def test_committed_chains_are_the_isa_facts(self):
        # A committed snapshot with the critical path's columns names the latencies that chose
        # them, and they are what the ISA facts table gives its architecture today: a row
        # measured (or measured again) moves the chains, so the snapshot is extracted again with
        # it. No toolkit: this holds on a build whose nvcc is not the snapshot's too.
        data = Path(__file__).resolve().parents[1] / "src/ops/generic/data/kernel_facts"
        seen = 0
        for path in sorted(data.glob("sm_*.txt")):
            header, _ = kernel_facts.parse_snapshot(path.read_text())
            if header["columns"].split() != kernel_facts.COLUMNS.split():
                continue
            seen += 1
            self.assertEqual(header.get("chain"),
                             kernel_facts.chain_line(*kernel_facts.chain_weight_sets(header["arch"])),
                             f"{path.name}: extract it again with today's ISA facts")
        self.assertGreaterEqual(seen, 1)

    def test_replica_prices_the_chain(self):
        # The replica's timeline law (tools/bench/rmma_counted_replica.py) prices a committed
        # kernel's chain with the same clocks per class the snapshot chose it with, read at the
        # resident warps: a shared load at the shared-memory latency, not the L1 one.
        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo / "tools" / "bench"))
        import rmma_counted_replica as replica
        chip = replica.chip_of(str(repo), "GeForce RTX 4090")
        snap = replica.load_snapshot(str(repo), chip["arch"])
        sched = (1, 1, 4, 1, 1)  # m_tiles, n_tiles, warps_m, warps_k, cluster
        inst, how = replica.find_instance(snap, "bf16", sched, replica.EPILOGUE["store"], -1, -1)
        self.assertEqual(how, "fallback")  # the plain instance of the schedule
        call = dict(n=4096, k=4096, T=24, sets=1.0, bpw=replica.BPW["bf16"], heavy=False)
        seconds, terms = replica.price(chip, call, inst, sched, "timeline")
        self.assertGreater(seconds, 0.0)
        lat = sc.chain_weights(chip["isa"], replica.isa_warps(terms["tl_warps"]))
        clocks = sc.chain_clocks(inst, lat)
        self.assertGreater(clocks, 0.0)
        self.assertEqual(terms["tl_chain"], max(float(inst["total"]), clocks))
        more = dict(inst, cp_lds=inst["cp_lds"] + 1000)
        _, terms = replica.price(chip, call, more, sched, "timeline")
        self.assertAlmostEqual(terms["tl_chain"] - clocks, 1000 * chip["isa"]["shared_latency"])
        self.assertNotEqual(chip["isa"]["shared_latency"], lat["cp_ld"])

    def test_replica_keeps_both_local_columns(self):
        # A snapshot names two fields `local`: the kernel's local memory in bytes and the K
        # loop's local-memory instructions. The replica matches a sweep's kernel by its stack and
        # local bytes and prices the loop's reloads by their count: a spilled kernel (a stack
        # frame, reloads in the loop) is found by its own bytes, not by its instruction count.
        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo / "tools" / "bench"))
        import rmma_counted_replica as replica
        snap = replica.load_snapshot(str(repo), "86")
        spilled = [(key, i) for key, insts in snap.items() for i in insts
                   if i["stack"] > 0 and i["local"] > 0 and i["local"] != i["local_mem"]]
        self.assertTrue(spilled)
        fmt_of = {dec: fmt for fmt, decs in replica.DECODERS.items() for dec in decs}
        found = 0
        for (dec, wm, wk, mt, nt, c, epilogue), i in spilled:
            if dec not in fmt_of:
                continue
            inst, how = replica.find_instance(snap, fmt_of[dec], (mt, nt, wm, wk, c), epilogue,
                                              i["registers"], i["stack"] + i["local_mem"])
            self.assertIn(how, ("ok", "ambiguous"))
            self.assertEqual((inst["registers"], inst["stack"] + inst["local_mem"]),
                             (i["registers"], i["stack"] + i["local_mem"]))
            found += 1
        self.assertGreater(found, 0)

    def test_replica_skips_a_board_without_facts(self):
        # The timeline law has no facts on an arch without an ISA row. Asked for that law alone,
        # the replica says so and goes on to the next board: it prices nothing, so it reads no
        # sweep (the directory named here does not exist). Every swept board's architecture has
        # a row today, so the table is read here as one that lacks sm_75's.
        import contextlib
        import io
        from unittest import mock
        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo / "tools" / "bench"))
        import rmma_counted_replica as replica
        self.assertTrue(replica.chip_of(str(repo), "GeForce RTX 2080 Ti")["isa"])
        isa_of = replica.isa_of
        argv = ["replica", "--repo", str(repo), "--law", "timeline",
                "--sweep", "GeForce RTX 2080 Ti=" + os.path.join(str(repo), "no_such_sweep")]
        out = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), \
                mock.patch.object(replica, "isa_of",
                                  lambda r, arch: None if arch == "75" else isa_of(r, arch)):
            replica.main()
        self.assertIn("no facts (sm_75 has no ISA row)", out.getvalue())
        self.assertIn("board skipped", out.getvalue())

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
        self.assertEqual(len(lines), 2)
        self.assertIn("without a K loop", lines[0])
        self.assertIn("k_iter differs", lines[1])

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
        # An ISA facts table without the architecture: the critical path under nominal latencies.
        (self.root / "isa.json").write_text('{"arches": {}}')
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
                str(self.root / "sass.txt"), "--expected", str(expected), "--nvcc", nvcc,
                "--isa-facts", str(self.root / "isa.json"), *extra]
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

    def test_chain_of_other_latencies(self):
        # A snapshot whose critical paths were chosen with the latencies the ISA facts give its
        # architecture equals the build; one chosen with others (the row moved since) does not,
        # whatever its rows say. Neither matters to a resources-only check or another nvcc's.
        rows = kernel_facts.extract_rows(None, None, "89", str(self.root / "usage.txt"),
                                         str(self.root / "sass.txt"))
        today = kernel_facts.chain_line((sc.CP_NOMINAL,), "nominal")
        text = kernel_facts.render("89", rows, "13.4.92", "-O3", "abc", today)
        self.assertEqual(self.run_check(text), 0)
        moved = text.replace("chain " + today, "chain " + today.replace("cvt=6", "cvt=11.5"))
        self.assertNotEqual(moved, text)
        self.assertEqual(self.run_check(moved), 1)
        self.assertEqual(self.run_check(moved, extra=["--no-sass"]), 0)
        self.assertEqual(self.run_check(moved, nvcc="13.0.1"), 77)


if __name__ == "__main__":
    unittest.main()
