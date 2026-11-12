# Counted MMQ units per generation (#785, PR-B)

The kernel-facts snapshots (`src/ops/generic/data/kernel_facts/sm_<arch>.txt`) now hold two
sections after the register-MMA rows:

- `family mmq`: ggml's matrix kernel `mul_mat_q<type, J, fallback, prec>`, every format and
  column tile the build compiles;
- `family w4a8`: the W4A8 tiles `w4a8_kernel` and `w4a8_ws_kernel`.

Each section has the schema 4 loop counts and static schedule of the register-MMA rows. It
also has the pipe counts of the same iteration: I2F, I2FP, FFMA, FMUL, FADD, IMAD, BAR. The
counts are data. No price, cost or selector code reads them in this PR.
`ninferno_kernel_facts_lookup_test` checks every row against a kernel's bounds. It also checks
the rows the launch laws read: one or two loops, four barriers and `k_iter` 32 J per MMQ unit,
and one loop, two barriers and `k_iter` 16 x 32 for the W4A8 32 x 128 tile.
`ninferno_kernel_facts_test` compares a build's kernels with the sections. The comparison is
full where the build's nvcc is the one the section records. Under another nvcc only the kernel
set is compared.

`tools/kernel_facts/mmq_units.py` turns the sections into the tables below. Its docstring gives
the pipe model and the sources of every rate. One MMQ unit is one iteration of the K loop: a
128-row x J-column tile over 256 K values. The kernel runs 8 warps on 4 schedulers, so one
scheduler issues two warps' instructions per unit. Every clock figure below is SM clocks per
scheduler and unit.

## How the sections were made

The sections were read from 13.2.86 builds (`family_nvcc 13.2.86`). WSL has no 13.4.92, and
the register-MMA rows (13.4.92) were left byte for byte as they were. Each file's old text is
the prefix of its new text.

13.2.86 is close to 13.4.92 but not equal. On sm_89, 13.2.86 reproduces the committed
register-MMA rows except in these:

| Part of the row | Rows that differ (of 2696) |
|---|---|
| Resources | 12 |
| Loop or schedule | 28 |
| Code outside the loop | 416 |

So the sections should be regenerated on the release toolchain before a release reads them:

    python3 tools/kernel_facts/kernel_facts.py extract --binary <build's linked exe> --arch <label> \
      --nvcc 13.4.92 --families mmq,w4a8 --update src/ops/generic/data/kernel_facts/sm_<label>.txt

`ninja ninferno_kernel_facts_update` does the same for every family. It leaves the 13.4.92
register-MMA rows as they are. Until then, a 13.4.92 build compares the sections' kernel set
only.

The Turing rows agree with the counts DECISIONS 78 read from a 13.4 build. Q4_K has 195
registers at J 32 and a 48-byte stack at J 128. Per copy of the loop there are 4 J IMMA.8816,
and the local loads and stores appear at J 128 only.

## Counted units

| arch | nvcc | format | J | registers | instr / warp | IMMA | INT | FP32 pipe | I2F | stall_sum | issue | tensor | alu | xu | shared | bound | 2 x stall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sm_75 Turing | 13.2.86 | Q4_K | 32 | 195 | 1108 | 128 8816 | 122 | 549 | 136 | 2280 | 2216 | 1024 | 2196 | 2176 | 1776 | **2216** | 4560 |
| sm_80 GA100 | 13.2.86 | Q4_K | 32 | 205 | 1013 | 32 16832 | 122 | 549 | 136 | 2112 | 2026 | 512 | 2196 | 2176 | 1776 | **2196** | 4224 |
| sm_86 GA10x | 13.2.86 | Q4_K | 32 | 158 | 1012 | 32 16832 | 121 | 677 | 8 | 1576 | 2024 | 1024 | 1596 | 128 | 1776 | **2024** | 3152 |
| sm_89 Ada | 13.2.86 | Q4_K | 32 | 158 | 1012 | 32 16832 | 121 | 677 | 8 | 1576 | 2024 | 1024 | 1596 | 128 | 1776 | **2024** | 3152 |
| sm_90 Hopper | 13.2.86 | Q4_K | 32 | 178 | 1018 | 32 16832 | 128 | 677 | 8 | 1600 | 2036 | 512 | 1354 | 128 | 1776 | **2036** | 3200 |
| sm_100 GB100 | 13.2.86 | Q4_K | 32 | 206 | 1088 | 32 16832 | 200 | 677 | 8 | 1679 | 2176 | 512 | 1354 | 128 | 1776 | **2176** | 3358 |
| sm_120a GB20x | 13.2.86 | Q4_K | 32 | 227 | 1074 | 32 16832 | 186 | 677 | 8 | 1713 | 2148 | 1024 | 1726 | 128 | 1776 | **2148** | 3426 |
| sm_75 Turing | 13.2.86 | Q4_K | 128 | 255 | 3407 | 512 8816 | 175 | 1861 | 520 | 6718 | 6814 | 4096 | 7444 | 8320 | 3392 | **8320** | 13436 |
| sm_80 GA100 | 13.2.86 | Q4_K | 128 | 255 | 3096 | 128 16832 | 239 | 1861 | 520 | 6442 | 6192 | 2048 | 7444 | 8320 | 3392 | **8320** | 12884 |
| sm_86 GA10x | 13.2.86 | Q4_K | 128 | 221 | 2959 | 128 16832 | 132 | 2373 | 8 | 4035 | 5918 | 4096 | 5010 | 128 | 3392 | **5918** | 8070 |
| sm_89 Ada | 13.2.86 | Q4_K | 128 | 221 | 2959 | 128 16832 | 132 | 2373 | 8 | 4035 | 5918 | 4096 | 5010 | 128 | 3392 | **5918** | 8070 |
| sm_90 Hopper | 13.2.86 | Q4_K | 128 | 255 | 2998 | 128 16832 | 173 | 2373 | 8 | 4162 | 5996 | 2048 | 4746 | 128 | 3392 | **5996** | 8324 |
| sm_100 GB100 | 13.2.86 | Q4_K | 128 | 231 | 3118 | 128 16832 | 292 | 2373 | 8 | 4206 | 6236 | 2048 | 4746 | 128 | 3392 | **6236** | 8412 |
| sm_120a GB20x | 13.2.86 | Q4_K | 128 | 251 | 3110 | 128 16832 | 284 | 2373 | 8 | 4386 | 6220 | 4096 | 5314 | 128 | 3392 | **6220** | 8772 |
| sm_75 Turing | 13.2.86 | Q6_K | 32 | 215 | 1437 | 128 8816 | 787 | 161 | 128 | 2980 | 2874 | 1024 | 3148 | 2048 | 1752 | **3148** | 5960 |
| sm_80 GA100 | 13.2.86 | Q6_K | 32 | 227 | 1374 | 64 16816 | 787 | 161 | 128 | 2759 | 2748 | 512 | 3148 | 2048 | 1752 | **3148** | 5518 |
| sm_86 GA10x | 13.2.86 | Q6_K | 32 | 128 | 1375 | 64 16816 | 788 | 289 | 0 | 2256 | 2750 | 1024 | 3152 | 0 | 1752 | **3152** | 4512 |
| sm_89 Ada | 13.2.86 | Q6_K | 32 | 128 | 1375 | 64 16816 | 788 | 289 | 0 | 2256 | 2750 | 1024 | 3152 | 0 | 1752 | **3152** | 4512 |
| sm_90 Hopper | 13.2.86 | Q6_K | 32 | 168 | 1397 | 64 16816 | 811 | 289 | 0 | 2269 | 2794 | 512 | 3244 | 0 | 1752 | **3244** | 4538 |
| sm_100 GB100 | 13.2.86 | Q6_K | 32 | 240 | 1381 | 64 16816 | 795 | 289 | 0 | 2231 | 2762 | 512 | 3180 | 0 | 1752 | **3180** | 4462 |
| sm_120a GB20x | 13.2.86 | Q6_K | 32 | 255 | 1379 | 64 16816 | 793 | 289 | 0 | 2319 | 2758 | 1024 | 3172 | 0 | 1752 | **3172** | 4638 |
| sm_75 Turing | 13.2.86 | Q6_K | 128 | 255 | 3718 | 512 8816 | 1638 | 641 | 512 | 7905 | 7436 | 4096 | 6552 | 8192 | 3352 | **8192** | 15810 |
| sm_80 GA100 | 13.2.86 | Q6_K | 128 | 255 | 3453 | 256 16816 | 1635 | 641 | 512 | 7313 | 6906 | 2048 | 6540 | 8192 | 3352 | **8192** | 14626 |
| sm_86 GA10x | 13.2.86 | Q6_K | 128 | 254 | 3406 | 256 16816 | 1605 | 1153 | 0 | 5206 | 6812 | 4096 | 6420 | 0 | 3352 | **6812** | 10412 |
| sm_89 Ada | 13.2.86 | Q6_K | 128 | 254 | 3406 | 256 16816 | 1605 | 1153 | 0 | 5206 | 6812 | 4096 | 6420 | 0 | 3352 | **6812** | 10412 |
| sm_90 Hopper | 13.2.86 | Q6_K | 128 | 255 | 3511 | 256 16816 | 1695 | 1153 | 0 | 5650 | 7022 | 2048 | 6780 | 0 | 3352 | **7022** | 11300 |
| sm_100 GB100 | 13.2.86 | Q6_K | 128 | 255 | 3627 | 256 16816 | 1826 | 1153 | 0 | 5713 | 7254 | 2048 | 7304 | 0 | 3352 | **7304** | 11426 |
| sm_120a GB20x | 13.2.86 | Q6_K | 128 | 255 | 3606 | 256 16816 | 1803 | 1153 | 0 | 6388 | 7212 | 4096 | 7212 | 0 | 3352 | **7212** | 12776 |

## The decision point

Each column below is one counted form. A form serves every generation only if every measured
unit is within 10 % of it.

The measured units:

- **RTX 2080 Ti**: the unit DECISIONS 78 measured, a tile a CTA on 64 CTAs. It is 4.56 us at
  J 32 and 11.60 us at J 128, at the 1.95 GHz the board ran. `kTuringMmqLaw` keeps these at the
  1545 MHz reference clock, which gives the 7047 and 17920 it lists.
- **RTX 4070 Ti**: about 6960 at J 32, at full load (the plan).
- **RTX 3090**: 5600-7500 at J 32, at full load (the plan), at its 1695 MHz boost clock.

The DRAM column is the unit's weight bytes at the board's bus share of an SM. The board's
facts come from `gpu_facts.json`.

| board (arch) | J | measured unit | counted bound | measured / bound | 2 x stall | measured / 2 x stall | DRAM (weights) | bound + DRAM | measured / (bound + DRAM) | measured / (2 x stall + DRAM) |
|---|---|---|---|---|---|---|---|---|---|---|
| RTX 2080 Ti, 64 CTAs (sm_75) | 32 | 8892 | 2216 | 4.01 | 4560 | 1.95 | 3968 | 6184 | 1.44 | 1.04 |
| RTX 2080 Ti, 64 CTAs (sm_75) | 128 | 22620 | 8320 | 2.72 | 13436 | 1.68 | 3968 | 12288 | 1.84 | 1.30 |
| RTX 4070 Ti, full load (sm_89) | 32 | 6960 | 2024 | 3.44 | 3152 | 2.21 | 5727 | 7751 | 0.90 | 0.78 |
| RTX 3090, full load (sm_86) | 32 | 5600-7500 | 2024 | 2.77-3.71 | 3152 | 1.78-2.38 | 2737 | 4761 | 1.18-1.58 | 0.95-1.27 |

**Result: no counted form is within 10 % on every generation, so a microbench night is
needed.**

- **The counted issue bound is short of every measured unit**, by 2.7x to 4.0x on the 2080 Ti,
  3.4x on the 4070 Ti and 2.8x to 3.7x on the 3090. The ratio is not even constant within
  Turing: 4.0 at J 32 and 2.7 at J 128. This matches the "about three times" DECISIONS 78 found.
- **The static schedule (2 x stall_sum) is short too**, by 1.7x to 2.4x.
- **Adding the unit's DRAM time to either form does not close the gap on every board.** The
  best form, 2 x stall + DRAM, reads 1.04 and 1.30 on the 2080 Ti (J 32 and J 128). It reads
  0.78 on the 4070 Ti and 0.95 to 1.27 on the 3090.

So the stall sum does not carry the measured excess. The loaded-latency term, the part of a
unit that neither the pipes nor the control codes see, has no counted fact.

GA10x and Ada have identical counts at every format and width: the same SASS, with no
difference in issue between them. Their measured units still differ (the 4070 Ti at about 6960
and the 3090 at 5600-7500). That difference is the boards' DRAM share and clocks, not the
kernel.

GB20x counts within 6 % of Ada: Q4_K is 2148 against 2024 at J 32 and 6220 against 5918 at
J 128. But no GB20x unit has been measured at all, so nothing here says what its law is.

The microbench night: `tools/bench/mmq_launch_microbench.cu`, as DECISIONS 78 ran it. Measure
the tile-a-CTA unit by J (16 to 128) and the stream-k ratio on two cards a generation:

- Ada: RTX 4070 Ti and RTX 4090;
- GA10x: RTX 3060 and RTX 3090;
- GB20x: RTX 5070 Ti and RTX 5090.

This gives the law rows of PR-C: `kAdaMmqLaw`, `kGa10xMmqLaw` and `kGb20xMmqLaw`. Since GA10x
and Ada count the same, one law could serve both if their two cards' units agree once DRAM is
taken out. That is for the night to show, not this table.

GA100, Hopper and GB100 (sm_80, sm_90, sm_100) do not take MMQ as a cost-model candidate
(`mmq_candidate` is false on the full-rate rows). Their counts are data only, and no law is
needed for them.

## The W4A8 format term

| arch | format | registers | instr / warp | IMMA | INT | FP32 | stall_sum | instr vs Q4_K | stall vs Q4_K |
|---|---|---|---|---|---|---|---|---|---|
| sm_75 Turing | Q4_K | 195 | 658 | 64 | 211 | 268 | 1407 | 1.00 | 1.00 |
| sm_75 Turing | Q5_K | 207 | 702 | 64 | 247 | 268 | 1465 | 1.07 | 1.04 |
| sm_75 Turing | Q6_K | 212 | 947 | 64 | 365 | 394 | 1811 | 1.44 | 1.29 |
| sm_80 GA100 | Q4_K | 217 | 664 | 16 | 263 | 268 | 1522 | 1.00 | 1.00 |
| sm_80 GA100 | Q5_K | 224 | 714 | 16 | 311 | 268 | 1592 | 1.08 | 1.05 |
| sm_80 GA100 | Q6_K | 224 | 861 | 32 | 322 | 394 | 2153 | 1.30 | 1.41 |
| sm_86 GA10x | Q4_K | 212 | 664 | 16 | 263 | 268 | 1295 | 1.00 | 1.00 |
| sm_86 GA10x | Q5_K | 221 | 714 | 16 | 311 | 268 | 1356 | 1.08 | 1.05 |
| sm_86 GA10x | Q6_K | 255 | 861 | 32 | 322 | 394 | 1311 | 1.30 | 1.01 |
| sm_89 Ada | Q4_K | 212 | 664 | 16 | 263 | 268 | 1295 | 1.00 | 1.00 |
| sm_89 Ada | Q5_K | 221 | 714 | 16 | 311 | 268 | 1356 | 1.08 | 1.05 |
| sm_89 Ada | Q6_K | 255 | 861 | 32 | 322 | 394 | 1311 | 1.30 | 1.01 |
| sm_90 Hopper | Q4_K | 210 | 674 | 16 | 253 | 268 | 1314 | 1.00 | 1.00 |
| sm_90 Hopper | Q5_K | 221 | 726 | 16 | 300 | 268 | 1407 | 1.08 | 1.07 |
| sm_90 Hopper | Q6_K | 255 | 892 | 32 | 335 | 394 | 1320 | 1.32 | 1.00 |
| sm_100 GB100 | Q4_K | 202 | 555 | 16 | 158 | 268 | 913 | 1.00 | 1.00 |
| sm_100 GB100 | Q5_K | 199 | 629 | 16 | 230 | 268 | 1011 | 1.13 | 1.11 |
| sm_100 GB100 | Q6_K | 248 | 786 | 32 | 250 | 394 | 1086 | 1.42 | 1.19 |
| sm_120a GB20x | Q4_K | 146 | 555 | 16 | 154 | 268 | 1114 | 1.00 | 1.00 |
| sm_120a GB20x | Q5_K | 150 | 628 | 16 | 231 | 268 | 1129 | 1.13 | 1.01 |
| sm_120a GB20x | Q6_K | 138 | 777 | 32 | 243 | 394 | 1188 | 1.40 | 1.07 |

The plan measured a Q6_K CTA at about 1.5x a Q4_K one's time per K on the RTX 4070 Ti (60
against 40 us per 4096 K). The Ada counts give Q6_K 1.30x Q4_K's instructions and 1.01x its
static stall sum per K tile. So the counted issue does not carry the measured format term
either.

## Open points

- **The sections are from 13.2.86.** Regenerate them on 13.4.92 (the command above). Until
  then the 13.4.92 check compares their kernel set only.
- **Q8_0 has no MMQ kernel in this build.** `ggml_mmq` instantiates Q4_K, Q5_K, Q6_K, the other
  K-quants, Q4_0, Q4_1, Q5_0, Q5_1 and the IQ formats, but not Q8_0. So the selector never
  prices Q8_0 as MMQ, and Q8_0 has no row.
- **The rates in the pipe model are listed figures, not measured ones.** I2F's 16 results per
  SM-clock (8 clocks a warp instruction) comes from the CUDA guide's table, and Turing's J 128
  unit is bound by it. The model also assumes IMAD issues on the INT32 datapath. The int8 MMA
  rate is the generation row's `s8`. A different I2F rate moves only the `xu` column, on sm_75
  and sm_80, whose builds convert with I2F; sm_86 on uses I2FP.
- **Q2_K's `k_iter` is 5/4 of its unit**, in both families. Its min term is a scale and a min
  per 16 values, and that term takes MMAs of its own, which the ones-register search of
  `sass_counts.py` does not tell apart from the weight MMAs.
- **sm_70 has no sections.** No 12.8 toolkit is on the build host, and Volta takes no MMQ law.
  The check notes the sm_70 build's MMQ and W4A8 kernels as unchecked.
