# NInferno autonomous work log (Claude session 62a7d67e)

## Mode (set by the owner, 2026-10-22 ~19:00 local)
- The owner (user) said: the developer is away for 4 days (until about 2026-10-26); "everything is on you,
  keep developing the project until I say stop".
- Authorized: push branches to Brontir/NInferno and open DRAFT PRs. NOT authorized: merging to master
  (ask first), force-pushing others' branches, deleting anything.
- Reports to the owner in Russian; PRs/commits in English, repo style (AGENTS.md), Co-Authored-By trailer,
  PR bodies end with the Claude Code line.

## Stands (root via <the stands' SSH key>; config: scratchpad/lab/ssh_config)
| host | IP | GPU | arch | my build dirs |
|---|---|---|---|---|
| llm2080ti | <LAN address> | RTX 2080 Ti 11 GB | 75 | /root/claude/wt_tst, build_tst_75 |
| llm4090 | <LAN address> | RTX 4090 24 GB | 89 | (busy: developer queue ~40 h) |
| llm4070ti | <LAN address> | RTX 4070 Ti 12 GB | 89 | /root/claude/wt_inc, build_inc (+ hf cache) |
| llm5070ti | <LAN address> | RTX 5070 Ti 16 GB | 120a | /root/claude/wt_tst, build_tst_120a |
- Ubuntu 26.04, CUDA 13.4, cmake 4.2.3, gcc 15, ccache (/root/.ccache, shared; mirror the developer's
  relative layout wt_X + build_X for hits). Internet OK but HF downloads ~6 MB/s.
- Protocol: the developer's queue (/root/queue/{pending,running,done}, runner.sh) waits while
  /root/GPU_BUSY exists. My helper /root/claude/lab.sh: `lock PURPOSE` / `unlock` (only my own tag).
  Never touch /root/wt_*, /root/build_*, /root/bins, /root/queue jobs of the developer.
- Run long jobs with `(setsid nohup bash X.sh > X.out 2>&1 < /dev/null &)`; wait with a background ssh loop.

## Branches / PRs
- #72 det/gdn-state-chunk-rounding: FP16/BF16 GDN state rounded at every chunk end + segment-relative
  early stop + tests + bench --state-dtype. Evidence: 5070 Ti tests pass; perf +1.2-1.5 % kernel.
  PENDING: E2E (4070 Ti, run_e2e_combo.sh), KLD (run_kld.sh queued after), sm_75 (2080 Ti run_2080.sh).
- #73 det/facts-memory-variants: K16. PENDING: sm_75 ctest (machine_facts, fingerprint, selector golden).
- #74 det/host-tier-declared-pin: D3 (rung declares configured Host-tier pinning). 5070 Ti CPU tests pass.
- #75 det/fingerprint-chat-template: D10 part (chat template content in sources). 5070 Ti tests pass.
- #76 det/spec-gate-served: perf_matrix records served_speculation; gate judges it, 0.98, --release,
  verify-route files in DECISION_FILES. 27 unittest pass; on bench.jsonl --release blocks (21 stale +12 unpaired).
- #72 E2E DONE (4070 Ti, 9B Q4_K_M fast): s1 alone 393/401 dumps differ + hit!=miss; s1+#72 0/401 + hit==miss.
  Commented on #66. sm_75 K16 tests pass (#73 updated). Pending for #72: KLD, sm_75 GDN tests.
- Exact-mode E2E determinism run queued on 4070 Ti (run_det.sh, 9B, NINFERNO_DET_SPEC=off, master).
- #72 COMPLETE evidence: KLD (9B, quick, 4096/2048): f16 0.00981 -> 0.01045 mean, top-1 97.22 -> 97.14,
  PPL ratio ~1; sm_75 GDN tests pass. Body updated.
- Dense-tune A/B on 2080 Ti running (run_tune_ab.sh: routes as shipped vs tu102 entries re-keyed to
  s20.13.3.4.6; 9B Q4_K_M copied via `scp -3` from 4070 Ti in 62 s (LAN ~90 MB/s; HF ~6 MB/s)).
- #77 det/dense-tune-rekey: guard test (route_table_test) + all 356 entries re-keyed s20.9->s20.13.
  2080 Ti A/B (9B, spec none): C8 315.0 -> 346.6 (+10 %), C1/prefill unchanged. Guard fails before
  ("356 of 356"), passes after (sm_75).
- Running: full GDN/Qwen ctest of #72 vs master on 2080 Ti (run_gdn_suite.sh, waiter bb6g5qgqw).
- #78 det/warmup-capacity-error: warm-up not settling -> CapacityError (fallbacks catch it). Compiles sm_89.
- #79 det/audit-env-memory: D7 audit + getenv/cudaMemGetInfo, + plan_ladder/generic_gemm/causal_partition
  (92 files clean); allowed NINFERNO_ROUTE_RECORD dump. #57 runs it in CI (cpu label).
- Integration: all my branches merge with each other and all dev PRs; only #54 conflicts with #72
  (text: mode renames) -> comment on #72 with the resolution.
- Queued (waiters): 4070 Ti run_det.sh -> run_x4lite.sh (9B tables incl. borrowed vs none); 5070 Ti
  run_x4.sh -> run_det_moe.sh (exact det on 35B-A3B + 27B, spec off); 2080 Ti run_retune.sh (fresh tu102
  tune vs #77 re-keyed). #72 sm_75 regression sweep: 26/26 GDN/Qwen tests same as master.
- Developer's X13 gpt-oss (4090, done 19:48): acc/rnd short 2.22 -> long 8-24K 0.96 (tok/rnd 3.22 -> 1.96,
  dec 297 -> 169 tok/s): explains S11. Factors need a 2nd generation (developer's rule) -> queued
  run_x13oss.sh on 5070 Ti (same commit 5191d5466, same prompts; after det_moe; waiter b5zxkjtpx).
- Retune tu102: 73 fresh entries (10 changed, 2 new vs re-keyed); A/B rekeyed vs fresh running (bxzzqwx3g).
- Prepared idea (not committed): determinism test D->B under exact via exact_bytes() at batch/aba/
  cachecold/boundary/twins/chunks/pollute(d)/hosttier call sites; open PR only if MoE/27B also identical.
- Exact determinism 9B DONE (4070 Ti, 2725 s): 546 identical, 0 divergences, bugs=0 across solo/params/
  batch/aba/structured/cachecold/boundary/twins/chunks/pollute/hosttier; fullpool per contract.
- Exact determinism on 4070 Ti (9B): 222/222 identical so far (solo..chunks); D-class phases are in fact
  byte-identical -> next: run on MoE 35B-A3B + 27B on 5070 Ti after X4 before proposing D->B tightening.
- Developer's 4090 queue: a_s1e2e done 18:40 (35B: new cut 704 still 493/501 = D2), now b_p666 X13 gpt-oss,
  then b_p669 (18 h), x4.
- X4 on 5070 Ti running: run_x4.sh (27B + 35B-A3B UD-IQ3_XXS, tables vs --route-dir empty, ABBA, C1/C8);
  outputs /root/claude/x4/, summary in /root/claude/run_x4.out. Developer's own X4 (4090, older master
  a2ec3190 with NINFERNO_ROUTE_DIR, still valid) is queued behind ~30 h of jobs.
- Formatting: clang-format 23 at /root/claude/venv on 5070 Ti; use the separate clone /root/claude/fmt
  (never the worktree a running job uses); `git clang-format --binary ... master`.

## Cron loop
- Job da29ffa1 (`7,37 * * * *`, session-only, expires in 7 days): wakes me every ~30 min to run the loop
  prompt (read this log, check stands, process results, update PRs, report briefly in Russian).

## 20:08 tick
- Retune A/B (2080 Ti, 9B) interim: rekeyed C8 346.5 vs fresh C8 314.2 (= dead level); C1 equal. A quick
  retune (warm-up + one C8 wave) likely missed the decode bands (graph capture times nothing) -> keep #77's
  re-keyed entries; verify by comparing the fresh file's band routes when the A/B ends.
- X4-lite (4070 Ti, 9B, spec none) interim: tables vs none C1 equal; C8 387.0 vs 374.5 (+3.3 %).
- 5070 Ti: X4 still downloading the 35B (13.2 GB @ ~6 MB/s since 19:41).
- 4090: developer's b_p669 perf campaign running, x4 pending.

## 20:40 tick
- Retune A/B final: rekeyed C8 346.5/345.7 vs fresh 314.2/314.4; fresh tu102 entries have null decode AND
  prefill bands (graph capture times nothing) -> #77 keeps re-keyed entries; comment posted on #77.
- X4-lite final (4070 Ti, 9B): C8 387.0/387.3 tables vs 374.5/373.4 none (+3.5 %), C1 identical.
  => tables/tune pay only at small-T batched widths (T=2..16).
- Integration branch integ/claude-prs (master + #72..#79, local + stands only, NOT on GitHub) running the
  full ctest (-E real) on 4070 Ti (sm_89) and 2080 Ti (sm_75); failures re-run on master (run_integ.sh).

## 20:49 tick
- X4 5070 Ti, 27B UD-IQ3_XXS (product defaults, spec auto): C1 tables vs none decode 121.85/121.72 vs
  119.20/119.33 (+2.1 %), chats 120.9/120.7 vs 118.0/118.0 (+2.4 %), prefill equal; C8 262.1/262.7 vs
  261.9/259.8 (~0). 35B-A3B arms running next.
- Integration suites building done (2080 Ti 468 s, 4070 Ti 736 s), ctest running.

## 21:10 X4 done (5070 Ti)
- 35B-A3B UD-IQ3_XXS: long24k decode tables 360.9 vs none 374.8 (-3.7 %, both pairs within 0.1 %); chats
  C1 +0.6 %, C8 equal, prefill equal. gb203.json is the RTX 5080's table (84 SMs); the 5070 Ti (70 SMs)
  takes it + 172 borrowed problems. Same plan and speculation (dflash:7) in both arms. Posted on #77.
- Summary across cards: tables/tune help or hurt only at small-T widths; prefill never moves.
- Possible follow-up: find which gb203 entries lose for the 5070 Ti on the 35B (bisect) -> carry rule.
- det_moe (exact, 35B then 27B) started on the 5070 Ti; X13 gpt-oss queued after it.

## 21:40 integration finding
- Full suite (-E real, 270 tests) on master + #72..#79: sm_89 and sm_75 both 265/270; the 5 device tests
  (ninferno_device_test + 4 sync modes) failed; master passes them (sm_75) -> regression from #73 (K16):
  the test's placeholder driver values (=1) read as a slower memory. Fix d0960efe on #73: implausible
  driver memory (bus not multiple of 32, clock < 100 MHz) = unreported; the note names "another memory".
  8 device/facts/fingerprint/golden tests pass on sm_75. #73 body updated.
- Integration rerun (fixed script: master ref e651009a, --output-on-failure) on 4070 Ti (waiter bw3wq63xi).

## 21:22 tick: exact det on MoE 35B-A3B (5070 Ti) 220 identical / 0 other so far; det75 (2080 Ti) building;
   integ rerun (4070 Ti) running; 4090 developer's b_p669.

## 21:49 tick: EXACT on MoE 35B-A3B (5070 Ti, sm_120a) DONE: 546 identical, bugs=0, divergences 0, 1824 s
   (same as dense 9B on sm_89). Running: 27B on 5070 Ti (76 identical so far), 9B on 2080 Ti sm_75 (99 so far),
   integ rerun ctest on 4070 Ti.

## 22:00 integ rerun 4070 Ti: 270/270 PASS (master + #72..#79 incl. K16 fix).
- 4070 Ti now: run_det_fast.sh: determinism probe under fast (aba,cachecold,boundary,chunks,hosttier,batch,
  twins) on master then combo-s1-72 (S1 d4e0e4ed + all 5 #72 commits = 27905280, pushed to the stand's
  wt_tst ref); counts divergences per phase (waiter bw8mhkans).

## 22:19 fast-mode probe, master (9B, 4070 Ti, 716 s, bugs=6 scored=156):
   aba 0/4, cachecold 0/16, boundary 0/9 differ (whole-prompt hits restore at the prompt end);
   chunks 16/56 differ (lengths chunk*n+1..2 -> early stop + FP16 state + MMQ), hosttier 86/320 (mid-prompt
   restores), batch 62/70 and twins 46/56 (T<=16 batch composition: not promised). Combo run in progress.
   Exact: 27B 99 identical so far (5070 Ti); 9B sm_75 221 identical so far (2080 Ti).

## Backlog (priority order; update as done)
1. Finish #72/#73 evidence; comment on #66 with the combined E2E (S1 + #72).
2. Bench provenance: perf_matrix rows carry commit + nf1; gen_performance marks unversioned/stale.
3. D3: Host tier sized from options / RAM class, not MemAvailable (model_instance.cpp, contended).
4. D5: decision manifest + perturbation test (contended with #57/#46/#54/#65/#71).
5. spec_gate: 0.98, served configuration, stale = release blocked.
6. dense_tune (#29): evidence on 2080 Ti (re-tune tu102 entries? A/B) -> owner decision.
7. Monitor the developer's 4090 queue results (a_s1e2e/a_s1op for #66, b_p666 X13 gpt-oss, b_p669 perf
   campaign, x4) and act on them.
Owner decisions pending: ladder granularity (<=4 levels?), MoE offload, GTX 16xx support, card tables.

## 22:31 fast-mode probe, combo-s1-72 (S1 + #72, 728 s, bugs=6 scored=145):
   aba/cachecold/boundary 0; chunks 14/56 (master 16): all chunk*n+1 lengths now identical; remaining diffs
   are exactly lengths chunk*n+2 (1026/3074/4098/8194) at token 0/1 (last call 66 vs 65 tokens);
   hosttier 86/320 (unchanged); batch 55/70, twins 49/56 (not promised in fast).
   Next: find the op whose fast plan depends on the call's token count (attention split?).

## 22:55 fast-mode root cause (4070 Ti, combo-s1-72, 9B, int8 KV, chunk_probe + NINFERNO_QWEN_DUMP)
- causal scoring normalizes the chunk to 1024 (scored "chunk 1024" for both) -> no use for chunk A/B.
- generation prefill dumps of the last 2 tokens, one call (4097/4098) vs chunk 1024 ([..,960,65/66]):
  layers 0-2 (GDN + MLP + MMQ) byte-identical; FIRST difference L03.attn (first attention layer), then
  everything after. Same for 4097 and 4098 (the +1/+2 token pattern in the probe was luck of margins).
- Cause: causal_softmax_attention reads only invariant_routes_active(); under fast prefill the INT8
  plan takes Grouped (W<=8), ParallelGrouped (<=256, KV splits from batch*kv_heads*ceil(W/8) tiles)
  or Tiled (>256 = prompt_fast int8-MMA kernel on sm80+) by the call's width -> a token's attention
  bytes follow the call (chunk size / cache frontier). Not in the pillar gap list; contradicts
  neutral_frontier.h ("a point on the 64-token grid leaves every token's bytes") and the prefill
  route profile's intent (prefill_routes.h). Same structure in bf16/fp8/k8v4/nvfp4 plans.
- Next: (1) is prompt_fast per-row invariant (dump mid rows)? (2) prototype fast-prefill attention
  = invariant instances for W>=17 (or one family), A/B bytes + prefill speed; (3) PR or gap doc.

## 23:10 tick
- Exact runs still going: 2080 Ti 9B sm_75 (226 identical, 0 div so far), 5070 Ti 27B (220 identical);
  the non-identical result lines are "ok"/"refused at start"/"finished at the context limit" (expected).
- det/determinism-exact-bytes: clang-format 23 fixes folded in (local a939377e, not pushed yet).
- 4070 Ti: run_fastattn.sh (local prototype int8/plan.cpp, env NINFERNO_LOCAL_PREFILL_ATTN 0/1/2):
  E1 mid-row dumps (is prompt_fast per-row invariant), E2 last-token dumps under 1 (prefill >=17 ->
  tiled/prompt kernel) and 2 (prefill -> invariant instances), E4 op bench (append, h16-kv4 int8,
  T 17..4096 x ctx 0/4K/16K), E3 det fast (aba,cachecold,boundary,chunks,hosttier). Waiter blhg3mlmt.
- All 5 KV storages share the width rule (Grouped <= token tile, ParallelGrouped <= 256, Tiled above;
  invariant override) -> one uniform fix once the A/B picks the family.
- 4090: developer's x4_l4090.sh running. No new PR comments/reviews from others.

## 23:45 fast-mode attention: prototype results (4070 Ti, 9B, int8 KV, combo-s1-72 + local plan.cpp)
- E1: prompt_fast is per-row invariant: rows 2000-2001 and 4030-4031 byte-identical in ALL 401 dumps,
  one 4097 call vs chunk 1024 (widths 1024/960 vs 4097).
- E2: mode 1 (prefill W>=17 -> tiled) and mode 2 (prefill -> invariant instances): last two tokens of
  4097/4098 byte-identical in all 401 dumps.
- E3 det fast (aba,cachecold,boundary,chunks,hosttier): mode 1 chunks 0/56 (was 14), bugs 0 (was 6),
  hosttier 58/320; mode 2 chunks 0/56, hosttier 48/320. Remaining hosttier = reused 0, first diff
  at decode token 3..44: batch neighbours (C=8 waves), not promised in fast.
- E4 op bench (append h16-kv4 int8 eager cold, us): wide calls equal under mode 1 (1024/4096 at any
  ctx); W=256 faster (575->356 at 4K, 2111->1226 at 16K); narrow slower (W=17: 53->203 at 4K,
  152->690 at 16K; W=66: 133->219, 457->730). Mode 2 is 3-4x slower on wide calls (4096@16K
  12.2->38.2 ms) -> rejected.
- 2080 Ti EXACT 9B sm_75 DONE: 546 identical, bugs 0, divergences 0 (5346 s). Third arch for D->B.
- Branch det/fast-prefill-attention (local abff9259): INT8 plan: prefill profile && W>=17 -> Tiled,
  not on Volta (split-D only >256); test run_prefill_shape_invariance_cases (int8, 3 geometries,
  ctx 300..40000, W 17/64/130/257 vs 300, cuts at 256/283, cached) + workspace contract under the
  prefill profile; docs: prefill_routes.h, neutral_frontier.h, pillar gap line (other storages).
  clang-format clean. run_fastattn2.sh on 4070 Ti (waiter b7rnn2jks): finer bench, test fails on
  master plan / passes with fix (all storages), det fast on S1+#72+fix.
- 2080 Ti: run_det75b.sh = the tightened test itself (73aaaa0f = a939377e) built OK, running (bgrtugf15).
- 4090: developer's b_p669 finished 22:27 (f306faa57: 27B long24k prefill ~815-822, 35B ~6233-6331);
  x4_l4090 running.
- #80 opened (draft): det/determinism-exact-bytes a939377e (exact -> B for batch/aba/cachecold/
  boundary/twins/chunks/pollute(d)/hosttier). Evidence rows: 9B sm_89 546/546 (2725 s), 9B sm_75
  546/546 (5346 s), 35B-A3B sm_120a 546/546 (1824 s); TODO add 27B row (5070 Ti, in pollute at
  23:21) and det75b (the branch's own test on sm_75).

## 23:55 tick
- 4070 Ti run_fastattn2: (1) finer bench: crossover W~96-128; W>=128 tiled 40-44 % faster at every ctx,
  W 17-96 slower (W=17: 17.8->44 us @0, 57->203 @4K, 160->676 @16K, 278->1310 @32K).
  (2a) master plan + new test: 84 failures (7 narrow checks x 12 geometry/ctx; W=257 passes).
  (2b) fix: bf16, int8-g64 (incl. prefill width), fp8, nvfp4 PASS; k8v4 running; then (3) det fast.
- Caveat found reading prompt_fast.cuh: the diagonal tile's FP16 V shift (tile_shift, vmax over keys up
  to the CTA's last row, zero-filled past it) can differ between a narrow and a wide call when a V
  group scale > 8 lies past a row's own key -> possible FP16-subnormal rounding difference. Not hit by
  the probe/test; state it in the PR.
- Queued: run_ttft.sh (4070 Ti, after fastattn2): follow-up-turn prefill over a cached ~15K doc,
  master vs fix, ABBA; run_fastattn75.sh (2080 Ti, after det75b): attention test on sm_75.
- 5070 Ti 27B exact: fullpool done, hosttier next. 4090: developer x4.
- 00:05 27B exact DONE (5070 Ti): 546 identical, bugs 0, divergences 0, 7928 s -> #80 body updated (4 rows).
  run_x13oss.sh (5070 Ti) now takes the lock (waits on det_moe DONE); waiter b5zxkjtpx still armed.
- 00:10 (2b) attention test with the fix (sm_89): rc=0 1337 s, all 5 storages PASS, 0 prefill-width
  failures. (3) det fast on S1+#72+fix running, then run_ttft.
- 00:20 (3) det fast on S1+#72+fix (the real commit): chunks 0/56, bugs 0, hosttier 58/320 (batch),
  aba/cachecold/boundary 0 -> same as prototype mode 1. run_ttft.sh now on 4070 Ti.

## 00:45 #81 opened (draft): det/fast-prefill-attention abff9259
- E2E follow-up turn over a cached 15,812-token prefix (9B, 4070 Ti, C=1, ABBA, prompt_ms medians):
  29 tok 46.6->51.0 (+9 %), 42 +8 %, 57 +7 %, 74 +4 %, 91/127 ~0, 162 80.0->75.8 (-5 %); cold 2.71 s both.
- #81 body: probe table, dumps, change, tests (sm_89 all storages PASS; master plan 84 fails), op bench
  table, E2E table, limits (FP16 V shift corner, Turing pending), follow-up (absolute key splits).
- Comments posted on #72 and #66: master 16/56 (6 bugs) -> #66+#72 14/56 -> +#81 0/56, 0 bugs.
- Pending: 2080 Ti det75b (tightened test itself) then run_fastattn75 (attention test on sm_75) ->
  update #80 and #81. 5070 Ti: X13 gpt-oss (b5zxkjtpx). 4090: developer x4.
- 01:05 #81 amended (comment-only, d45fb5f8, force-with-lease on my own branch) so master + #72..#81 merge
  with no conflict (neutral_frontier.h adjacent-line clash with #72). Body notes stands ran abff9259.
- Gemma 4 / dense family already take gemma4_attention's prefill kernel for every call > 16 columns
  (absolute grid) -> #81 aligns Qwen INT8 with that rule (added to #81 body).
- Decided NOT to start the split-key prompt kernel now (E2E cost of #81 is modest; split traffic would
  cost wide calls ~5-10 % at long ctx) -> documented follow-up for the developer to weigh.
- 4070 Ti: run_hosttier1.sh (hosttier at C=1, master / #66+#72 / +#81), waiter bltf40ivh.
- 01:20 tick (stands 00:19): hosttier1 rerun (C=1 with kv_capacity 4096*C patch) on 4070 Ti, waiter
  ba1b5o9yw; det75b 222 identical / 0 BUG (late phases); x13oss started 23:49 (prompts ready);
  4090 developer x4. Nothing to act on.

## 01:50 owner: may merge to main after re-checks (memory updated). Plan:
- Re-checks queued: run_integ2.sh (master + #72..#81 merged in order; full ctest -E real; PROBES=1 on 89:
  exact probe + fast probe) on 4070 Ti (after hosttier1) and 2080 Ti (after fastattn75); run_perf2.sh
  (4070 Ti, after integ2): perf_matrix 9B master vs integ2 ABBA. Waiters bkgezg9gp, b9g7sbb0y, b1tu2rre0.
- Merge #72..#80 after green re-checks; HOLD #81 until the key-split prompt kernel removes the
  narrow-call regression (branch det/prompt-key-splits on top of #81, S = 4096 keys, partials folded
  in piece order; in progress).
- hosttier at C=1 (fast): master 88/320, #66+#72 76/320, #66+#72+#81 0/320 (reuse ~0; the divergences
  were the capture cuts of a cache-enabled engine) -> added to #81 body.
- 02:00 det75b (the #80 test itself, sm_75): rc 0, 5358 s, 554 contract-B lines, 546 identical, bugs 0
  -> #80 body updated. integ2 restarted (stand wt_tst origin is local wt_inc -> fetch from GitHub
  into refs/remotes/gh/*; integ2 9f98af77 = e651009a + #72..#81) on 4070 Ti (waiter b4rpbizbk), perf2
  after it (bseywde90); 2080 Ti integ2 restarted (old waiting process killed by PID), after fastattn75.
- 02:20 X13 gpt-oss-20b dflash on 5070 Ti DONE (460 s client): pooled acc/rnd chat 1.67, short 2.32, mid 1.78,
  long 0.98 vs 4090 1.76/2.22/1.86/0.96 -> posted on #65 (developer's PR). 5070 Ti now FREE.
- integ2 attempt 3 on 4070 Ti (benchmarks OFF: master's bench/ops/q8_linear_add_bench.cu does not build
  with NINFERNO_BUILD_BENCHMARKS=ON -> separate small fix later); waiter baollphq4; perf2 waits on it.
- det/prompt-key-splits local 0acd00e4 (on #81 d45fb5f8): pieces of 4096 keys, partials + merge in piece
  order, chunked launches (64 MiB budget), workspace query; clang-format clean. Next: GPU tests on 5070 Ti.
- 02:45 owner: Codex review is connected on GitHub, use it fully. Triggers: "@codex review" comment, PR opened
  for review, draft marked ready. Requested on #72..#81. Gate for merges: Codex "no major issues" (or findings
  fixed/answered) + stand re-checks. Session PR bar: bound to #65 (developer's) + #66, #72..#81 (monitor off).
- 5070 Ti: run_keysplit.sh (attention test int8/all + op bench prefill profile master/#81/#81+pieces, h16-kv4
  and h24-kv4) started 02:40.
- 03:30 Codex round 1 results: no major issues on #74, #77, #80; findings fixed & pushed + "@codex review":
  #73 a43c5e14 (MIG: driver memory read on whole boards only + test; fingerprint contract docs),
  #76 bfc46a6a (perf_matrix per-request timings.speculation; stale pass blocks --release; 28 py tests OK),
  #78 f6d1da24 (warm-up message counts passes run), #79 568713ee (audit: sys::env, _wgetenv, _wdupenv_s).
  #72/#75 Codex still running at 02:47. #81 P2 (tile_shift CTA-dependent) -> per-row prefix-max shift on P
  (planned in det/prompt-key-splits).
- key-split int8 test on 5070 Ti FAILED: workspace interval mismatch (non-monotone launch columns) and
  query/execution high-water (chunk metadata without valid columns). Fix: columns = min(width, chunk);
  reserve the metadata slot whenever chunking. Bench part of run_keysplit continues.
- 04:10 #75 Codex P2 (template TOCTOU) fixed 70480ed1: compile_chat_template names the bytes it compiled
  (named_bytes_sha256), Frontend::chat_template_source(), LoadSummary::chat_template_source from the served
  Frontend -> fingerprint; plan without Frontend falls back to reading the path; test_frontend checks it.
  Pushed + "@codex review". Build check: integ2-75 (2080 Ti) fetches branch heads at start (after fa75).
- det/prompt-key-splits (local, not pushed): 48d01ed8 pieces (monotone launch columns, chunk-metadata slot),
  0f86dfe0 bench switch, f3611d14 per-row FP16 bound (prefix-max scan, P shift, V shift only > 511) + test
  with growing V. 5070 Ti run_keysplit2.sh: int8 test, no-shift A/B, bench pieces 4096 vs 8192, all
  storages (waiter b5tgciire). Bench v1 (S=4096): narrow W=17 ~2x master at 16-32K, W=128-256 -45 %,
  W=4096 +8..12 % (9B) / +14..17 % (27B) -> consider S=8192.
- 04:20 fa75 (sm_75 tiled MMA): master plan 84 prefill-width fails; #81 all storages PASS (1547 s) -> #81 body.
  integ2-75 started on 2080 Ti with FRESH heads 5c3e0005 (Codex fixes on #73/#75/#76/#78/#79 included);
  4070 Ti integ2 runs OLD heads (2bfd6137) -> rerun there after the fixes settle. Waiter b1tu2rre0 (2080 Ti).
- 04:45 Codex round 2: #75 70480ed1 and #78 f6d1da24 -> no major issues. New findings fixed:
  #73 2d85fa0c (laptops excluded from memory-variant detection + test), #76 edc94a93 (stale exemptions
  block --release; 29 py tests OK). Re-review requested. #79 running, #72 stuck "running" since 21:44 UTC.
- 04:50 keysplit2 (5070 Ti): int8 test with pieces + per-row bound PASS (workspace contracts fixed, growing-V
  cases pass); WITHOUT the per-row bound also PASS -> the growing-V test guards the path but does not catch
  the old behaviour (power-of-two scaling rarely moves bytes). Bench 4096 vs 8192 next.
  2080 Ti integ2-75 (fresh heads) built in 511 s, ctest running. #72 Codex re-triggered.

## 05:10 owner: "finish the task; model and card lists are fixed; user settings are input; show the bounds and
## find the optimum for the current hardware/model/settings" + ultracode on (use workflows).
- Workflow wf_b8863084-908 (decision-goal-map): 6 readers (inventory, planner, speculation, kernels, options,
  determinism) over a read-only master worktree C:\Users\Pers_PC\Desktop\Claude\Inferno\NInferno-read ->
  synthesis (gap analysis + plan) -> critic. Task w2yr8eyq3.
- keysplit2 (5070 Ti) final: all storages PASS with pieces + per-row bound. Bench (vs master): pieces 4096:
  W=17 @16-32K ~2x master (#81 alone 3.5-7x); W=4096 +13..17 % (9B) / +18..22 % (27B); pieces 8192: W=4096
  +3..13 %, W=17 @16K 3.5x. The per-row prefix scan cost ~3 % on every tile -> full tiles now take the warp
  max (same value; the scan only on diagonal tiles): 2899c672. keysplit3 (5070 Ti) re-benches 4096/8192.
- Thinking: fast mode's documented contract (serving.md) does not promise cache/chunk byte identity; #81's
  invariance costs real speed -> keep #81 as a decision for the developer/owner with data; do not merge it.
- 05:40 Codex: #73 2d85fa0c and #76 edc94a93 -> no major issues. #72 findings: P1 independent oracle for
  per-chunk rounding; P2 bench byte metrics ignore the state width. Local WIP on det/gdn-state-chunk-rounding:
  rounded_state_oracle_case (FP64 gdn_ref run per 16-token chunk, state rounded to FP16/BF16 like
  stored_value; production under ScopedPrefillRoutes; prints agreement vs the per-chunk and per-call oracles)
  + Problem::state_element in the GDN bench. Queued on 5070 Ti after keysplit3 (run_gdnoracle.sh, waiter
  bwl3htk41) to get statistics before fixing thresholds.
- 4070 Ti integ2 (old heads) ctest 270/270 PASS; probes then perf2 running. 2080 Ti integ2-75 (fresh) ctest.
  4090: developer queue idle.
- 06:00 keysplit3: full-tile fast path recovered ~1 %; pieces 8192 chosen (wide +1..12 %, narrow 17@32K
  199->377 us). #81 branch now d45fb5f8 + f1b94848 (pieces 8192) + 46766822 (bench switch) + 7825abd0
  (per-row FP16 bound). #81 body rewritten with "Should it merge?" (product choice, not merged); title
  marked "decision needed"; Codex re-review requested.
- 06:40 #72 Codex fixes pushed (b2811bde decay-only rounded-state FP64 oracle: per-chunk oracle 99.98-100 %
  equal within 1 spacing vs per-call 3-39 %; 9032421f bench state bytes) + "@codex review".
  2080 Ti integ2 (fresh heads, #72..#81) 270/270 PASS. FINAL pre-merge config = master + #72..#80 (no #81):
  run_integ3.sh 75 on 2080 Ti (5d6f9354), run_integ3.sh 89 PROBES=1 queued on 4070 Ti after perf2.
- 07:15 (local 02:15) 4070 Ti integ2 attempt 3 (old heads, #72..#81, no #66): ctest 270/270; exact probe 554
  contract-B, bugs 0; fast probe bugs 8: chunks 8/56 (aba/cachecold/boundary 0) -> expected without #66
  (projections). Codex: #72 9032421f and #81 7825abd0 -> no major issues; #79 568713ee stuck "running"
  since 21:45 UTC -> "@codex review" again at 23:18 UTC.
- integ3 (master + #72..#80, tree 0efd6cb7 on both stands): 2080 Ti ctest running (waiter byfzlr1sz);
  4070 Ti building, then PROBES + perf3 (waiter bxzuc5t3i).
- 5070 Ti: 9B artifact copied from the 4070 Ti (scp -3); run_fastfinal.sh = fast probe on master + #66 + #72
  + #81 final (aba,cachecold,boundary,chunks,hosttier) and chunks on #66 + #72 (waiter blmk4pns7).
- Goal map (wf_b8863084-908) saved: scratchpad/goal_map.json, gap table gap_table.md. Plan S0..S11:
  S0 land fixes + docs drift; S1 fixed lists as data (models + cards, drift test, --list-models/--list-cards);
  S2 weight-free plan (#60, developer's); S3 admissibility + ninferno-explain/1; S4 deterministic budget;
  S5 speculation as plan axis; S6 stand route data; S7 per-op route plan; S8 spec profiles by settings;
  S9 declared objective + chooser; S10 matrix; S11 remove machine reads. Critic: B1/B2 axes, stands are
  2080 Ti/4090/4070 Ti/5070 Ti, objective needs owner sign-off, G17 (UI auto spec-tune) + G20 (context-cost
  key) into S0, bounds-only matrix right after S3, S1 adds build axis + (PCI id, memory variant).
- 08:00 integ3 (master + #72..#80, tree 0efd6cb7): 2080 Ti ctest 270/270 (2787 s); 4070 Ti ctest 270/270, exact
  probe 554 contract-B bugs 0, fast probe chunks 16/56 (= master; #81/#66 not included, expected); perf3 9B
  ABBA master vs integ3: prefill 5682/5683 vs 5687/5684 t/s, decode 80.64/80.62 vs 80.62/80.60, C8 387.0 vs
  392.2/390.5 -> neutral.
- MERGED to master (squash, owner's go-ahead after re-checks; Codex clean): #72 832f898a, #73 8d9ab867,
  #74 64981537, #75 494b064d, #76 622ef74f, #77 bcaa53a7, #78 e1338107, #80 77141fc5. master 77141fc5 +
  #79 = tree 0efd6cb7 exactly (the tested tree). #79 waits for Codex (re-requested 23:18 UTC). No GitHub CI
  checks exist on master (#57 adds CI; developer's).
- fastfinal on 5070 Ti restarted: #66 and #81 conflict only in deterministic-decisions.md gap list (#66
  removes the MMQ line, #81 adds the attention line after it) -> resolved with ours for the test; note on #81.
- Workflow wf_1c118b79-d80 (task wx683v9f7): G20 context-cost keyed by class, G19 constrained DFlash by count,
  G17 UI no auto --spec-profile, bench build fix; worktrees ../wt-g20-context-cost-class, ../wt-g19-constraint-count,
  ../wt-g17-ui-no-auto-tune, ../wt-bench-build-fix (branches det/<name>), implement -> 2 reviews -> fix.
- 08:15 UTC (the PC slept ~8.5 h between 23:3x and 08:0x UTC). Codex #79 568713ee -> no major issues ->
  MERGED 3809353d; master tree == integ3 tree 0efd6cb7 exactly. S0 merges complete (except #81, decision).
- fastfinal (5070 Ti, sm_120a, on e651009a): #66+#72+#81 final: chunks 0/56, aba/cachecold/boundary 0,
  hosttier C=8 40/320 (batch neighbours), bugs 0; #66+#72 alone: chunks 16/56, bugs 8 -> #81 body updated
  (+ note: docs conflict with #66 in the gap list).
- PR #60 (developer's weight-free plan) = critical path to "bounds + optimum before download". "@codex review"
  posted on #60. Worktree ../wt-p60 = master 3809353d + #60 (clean textual merge, d7a209f7). Workflow
  wf_f126bd83-eab (task wq6b9azjv): 4 review lenses (semantic merge with #72-#80, determinism, correctness,
  goal fit) x 2 verifiers + a stand script (dry plan vs real start per artifact). Asked owner whether I
  may merge developer PRs (#60, #66, #54, #71) after my re-checks.
- Stands: 2080 Ti, 4070 Ti, 4090 free; 5070 Ti free after fastfinal.
- 08:50 UTC p60b (master 3809353d + #60 98266d17) built on all three stands (2080 Ti 176 s, 4070 Ti 135 s,
  5070 Ti 719 s), ctest running (waiter bfabj1tt3). Codex on #60 (08:18 UTC): P1 plan_offline.cpp:171
  --vision mmproj projector bytes dropped from the quant plan; P2 plan_offline.cpp:184 dry plan picks GGUF rows
  while an unpinned fetch prefers safetensors. Workflows wx683v9f7 (G20/G19/G17/bench) and wq6b9azjv (#60 review)
  still running.
- 13:30 UTC master + #60 ctest: 2080 Ti 274/274, 4070 Ti 274/274, 5070 Ti 280/280. p60cmp (4070 Ti, 9B Q4_K_M):
  dry vs real start SAME 3 / DIFF 1 (board C=4: dry one rung lower, conservative); dry JSON byte-identical x3.
  CLI `ninferno --plan --no-load` without --prompt fails (= review finding correctness-2).
- #60 review workflow: 1 P1 (plan_quants.cpp:93 own-context probe drops desktop reserve at C=1 auto), several P2;
  Codex on #60: P1 vision mmproj bytes, P2 safetensors vs GGUF. Not yet posted to #60.
- G-branches (local worktrees, not pushed): det/g20-context-cost-class 12bebd04, det/g19-constraint-count d33e83a3,
  det/g17-ui-no-auto-tune 76d65033, det/bench-build-fix 0495590b -> need stand builds/tests, then PRs.
- Owner: no more workflows; sub-agents only Sonnet for review/idea, never editing (memory saved).
- Artifact "Карта выбора NInferno" https://claude.ai/artifact/UZg4s86MQZpZ7UwfrW2nkN (scratchpad/viz: graph_data.js,
  template.html, build.py; raw/ = 208 dry plans: 13 models x 8 cards x C{1,4}, run_matrix.sh on all 4 stands).
  Findings: 204/208 plain speculation; 22 cells < 3 bpw; C=4 anomaly (3.8-27B 4070 Ti IQ1_S 145K vs C=1 Q2_K_XL 49K);
  64-token contexts accepted (desktop tier).
- 14:20 UTC G-branches rebased on master 3809353d (G20 conflicts in engine.cpp + pillar resolved: both sides kept)
  and pushed: det/g20-context-cost-class e1822167, det/g19-constraint-count 59a11af4, det/g17-ui-no-auto-tune 9b11b9f0,
  det/bench-build-fix dcbfe5f2. run_gb.sh (master + 4 branches, BENCHMARKS=ON, ctest -E real) on 2080 Ti + 4070 Ti
  (waiter b3owr6nmy). Then: PRs + Codex.
- Review summary posted on #60 (issuecomment-6365945706): P1 C=1 own probe drops reserve; P2 shown plan != launched
  under Auto x C>1 (explains the matrix C=4 anomaly); CLI --plan --no-load needs --prompt; local drafter bytes;
  hub cache decides quant; desktop label on headless; spec_gate/audit lists. Artifact v3 updated.
- 15:30 UTC build dirs build_tst_75/89 broke (ninja 1.13.2 RefreshDyndepDependents assertion even on master after a
  BENCHMARKS=ON reconfigure) -> recreated fresh (Release, arch, ccache, BUILD_TESTING=ON — off by default!).
  master + G20/G19/G17/bench builds on sm_75 (995 s) and sm_89 (540 s); q8_linear_add + context_cost benches build
  (bench fix confirmed). ctest rerun with tests on (waiter bs4ccdcz8).
- Artifact: matrix now shows the launch context (pick.context); 52 C>1 cells marked "ladder is the uncapped plan" (#60 P2).
  Mobile + dark checked (no page overflow). Sonnet fact-check of the page vs code running (read-only).
- 16:10 UTC Artifact v4 "Сверено с кодом": Sonnet fact-check (read-only) found 16 discrepancies, all fixed: decode-steps
  rule (cgroup quota + graphs + no spec; Qwen/Nemotron always 1), reserve (WDDM/WSL2/integrated only, 0..4 GiB,
  --kv-capacity / --max-context@C=1), states 2C->C->0, request fields (logit_bias!=0 etc. refused; budget via
  Anthropic thinking.budget_tokens), facts fallback (driver readings on 2^(1/16) grid), gates 3-8 boards, sm_70..120a,
  activations int8, class chain neighbours, rung no longer reads MemAvailable, spec profile lookup (lineage, requires),
  pool_head, spec-select thinking, kv dtype per family, context-cost key slug+-sm<cc>, ladder constants 11, +flags node.
  Plain-spec reason corrected from the dry-plan notes: profile found but requires a drafter (Qwen GGUF no MTP head;
  assistant/DFlash not attached); plan mislabels it "measured faster without speculation". Qwen3.6-27B matches
  qwen3.8-27b by shape.
- 17:00 UTC gb (master + G20/G19/G17/bench, c7b3b593): 2080 Ti ctest 270/270, 4070 Ti 270/270, both benches build.
  Draft PRs opened + "@codex review": #82 G20 context-cost by class, #83 G19 constrained DFlash declared mask
  times, #84 G17 UI no auto spec tune, #85 bench build fix. Merge gate: Codex clean (then squash-merge, own PRs).
- 17:40 UTC Codex clean on #82/#83/#84 -> MERGED: #82 0df6f0e9, #83 4c7d4a09, #84 81795da7 (master 81795da7);
  master + #85 tree == tested gb tree 41ac883f. #85 Codex still running.
- #83 made the developer's #65 conflict (openai_chat_response.cpp outcome_timings: one counter each side) ->
  resolution posted on #65. #54 still conflicting (from #72, note posted earlier). #60/#66/#70/#71/#81 mergeable.
- Owner asked for a 3-pillar view (memory / speed / accuracy), formulas instead of ladders/tables, capability tables.
  New artifact "Память, скорость, точность" https://claude.ai/artifact/93RBdj5bD3pjuNRwvSP5BE (viz/pillars.py +
  pillars_template.html). Memory closed form ctx_max = (free − W − D − (C+K)·s − ws − graphs − Δ)/(C·b_kv) matches
  the #60 ladder within one g-step in 48/50 cuts (Nemotron 5090 +19 %: missing drafter bytes D; Muse 4070 Ti +5 %).
  Speed: decode ≈ η_d·BW·C/(W_act + C·ctx·b_kv), prefill ≈ η_p·SM·MHz·rate/(2·P_act); η fitted on 4 perf runs:
  decode dense 0.83 / MoE 0.42, prefill dense 0.29 / MoE 0.18; decode formula overpredicts at C=4-8 (needs compute
  term). Accuracy: measured δ for activations, GDN state, FP8 KV, two quant pairs (IQ3_XXS vs Q4 +3.6/3.7 % ppl);
  gaps: int8/k8v4/nvfp4 KV, quants < 3 bpw. RTX 4060 Ti in the matrix is the 8 GB board (0x2803).
- 15:40 UTC #85 Codex stuck "running" since 14:45 -> re-requested. Old map v5: RTX 4060 Ti label 8 GB.
  run_calib.sh (decode agg tok/s at C=1,2,4,8 short chats, plain, max-context 8192; C=1 also long24k) started:
  4070 Ti (4B, 9B, 27B IQ2_XXS), 5070 Ti (9B, 27B IQ3_XXS, 35B-A3B IQ3_XXS), 2080 Ti (9B after scp). Goal: fit the
  batch term of the decode formula (pillars page). 4090 skipped (no build/artifacts of mine).
- 16:10 UTC #85 Codex clean -> MERGED 393368c2; master tree == tested gb tree 41ac883f. All my PRs except #81 merged.
- Calibration (short chats, agg tok/s C=1/2/4/8): 9B 2080Ti 92.9/166.6/267.9/343.6; 4070Ti 85.8/154.7/267.5/390.7;
  5070Ti 146.4/256.9/425.1/618.0; 27B IQ3_XXS 5070Ti 63.7/110.7/194.4/259.1; 35B-A3B 5070Ti 267.6/428.9/... (running).
  Model t_step(C) = t0 + C*tau fits (9B 5070Ti C4 pred 423 vs 425). tau ~ 2*P_act/(eta_c*peak_int8), eta_c ~0.045-0.07.
  Long24k decode C=1: 2080Ti 85.4, 4070Ti 80.6, 5070Ti 137.8 (9B); 27B 60.7; 35B 247.9.
- 16:40 UTC 35B-A3B 5070Ti C4 605.8, C8 758.5. Two-term decode t(C) = (W_act(C)+C·ctx·b_kv)/(eta_d·BW) +
  C·2·P_act/(eta_c·peak_int8); MoE W_act(C) by routing (k/E: Qwen3.6-35B 8/256, Gemma 26B 8/128, gpt-oss 4/32).
  Fit: dense eta_d 0.91, eta_c 0.055, eta_p 0.285 (RMS 5.9 % on 24 pts); MoE eta_d 0.50, eta_c 0.037, eta_p 0.125
  (RMS 15.8 % on 9 pts, one model). Turing prefill overpredicted +44 % -> eta_p per generation. Pillars page v2.
- 19:30 local. Owner asked: inventory every switch (memory/speed/accuracy), implement missing ones, test, show
  with formulas. Done so far:
  * viz/switches.py: 16 switches + 5 gaps; pillars page section "Переключатели" (tables: switches, accuracy per
    switch, speed per switch, gaps) wired to meas/acc.json + meas/sw_*.out.
  * run_acc.sh on 5070 Ti (perplexity --quick, 9B: ref exact/bf16/f32 then kv int8/fp8/k8v4/nvfp4, state
    f16/bf16, act fast/int8, default, step1; then quants Q8_0..UD-IQ2_XXS vs Q8_0). kv_int8: KLD 0.00099,
    top-1 99.14 %, ppl ratio 1.0000. KLD is in report.json kld.overall.{mean_kld,top1_agreement}.
  * run_sw.sh on 4070 Ti + 2080 Ti (12 switch variants, C=1 long24k + C=8). exact on 4070 Ti: prefill
    5707 -> 1567, decode@22K 80.6 -> 69.9.
  * Gap implemented: per-request draft depth "ninferno": {"draft_tokens": K} (branch det/request-draft-depth,
    97d6c552 + a6b93674 after Sonnet review: floor at depth K via request_speculation_at, early 400,
    docs). Needs stand build + unit tests + e2e with a drafter, then PR.
- 20:10 local: acc so far kv_int8 KLD 0.00099 / 99.14 %; kv_fp8 0.0030 / 98.61 %; kv_k8v4 ppl 5.0130 (+0.27 %).
  sw: exact 4070Ti prefill -73 %, decode@22K -13 %, C8 -12 %; 2080Ti prefill -71 %, decode -28 %, C8 -12 %;
  int8 act = fast on Q4_K_M; kv_bf16 4070Ti decode@22K 75.6 (-6 %). Pillars page v3 published with the
  switches section (fixed .sw class clash). run_dd.sh queued on 4070 Ti after run_sw (build det/request-draft-depth,
  unit tests, e2e on q38_iq2xxs_dflash2 exact: K=0/3/7/15/16/bad member, byte equality vs server default).
- Owner review of switch cards (23.10 evening). Decisions/tasks recorded on the pillars page (v7):
  * TASK: all KV types (int8/k8v4/nvfp4) for all families (Gemma, dense, Nemotron) — "must be covered".
  * TASK: drafter hints for fine-tuned / re-quantized models whose drafter lives elsewhere.
  * TASK: default concurrency from the formula (card, model, reserve) instead of 1.
  * Not switches: CUDA graphs (mandatory), prefix cache + saved states (automatic), decode steps, sampling
    (model's recommended defaults) -> group "Должно работать само"; context + YaRN -> "Вход задачи".
- 21:00 local. acc (9B, vs exact/bf16/f32): kv int8 KLD 0.00099/99.14 %; fp8 0.0030/98.61 %; k8v4 0.0053/97.77 %
  (+0.27 % ppl); nvfp4 0.0098/97.10 % (+0.40 %); act fast = int8 0.0039/98.51 %; default 0.0040/98.45 %.
  state f16/bf16 refused under exact -> run_acc2.sh queued (under fast). sw done on 4070 Ti; 2080 Ti nearly:
  KV bf16 on 2080 Ti decode@22K -24 % vs int8 (4070 Ti -6 %); fp8 on Turing (emulated) -4 %. Page v8.
- dd on 4070 Ti: branch build 44 s, 7 unit tests pass; e2e running (q38_iq2xxs_dflash2, dflash2:7 fixed).
- dd e2e PASS (K0 37 tok/s plain, default/K3/K7 ~97 dflash2, K15/K16/bad member 400, bytes identical). PR #86 opened + @codex review.
- sw done both stands (page v9). Surprises: --no-cuda-graph only -2 % decode (4070 Ti) / -1 % (2080 Ti) on 9B C=1; chunk 512/2048 vs 1024: prefill -2..+1 % — tiny for 9B.
- acc: step1 (exact, one token per call) ppl 4.99966 = ref (exact decode == prefill). Quants: preparing downloads. #86 Codex running.

## 2026-10-23 (вечер) — конструктор explain.html (артефакт https://claude.ai/artifact/QDsehkqrPBVj4FxsfL5xS3, v1)
- Новая концепция по запросу владельца: шаги модель → карта → задача (контекст, C, квант) → переключатели; справа три опоры.
  У каждого переключателя таблица «если → контекст / декод / префилл / точность» против текущего.
- Паспорт модели: формулы с подставленными числами из config.json и сверкой с движком
  (KV 13/13, GDN-состояние, Mamba2, кольца).
- Правила по умолчанию, предложенные владельцем:
  - кольцо = тип KV (bf16 при KV bf16, иначе fp8);
  - состояние на ступень точнее KV (bf16 → f32, иначе f16; exact → f32).
  В коде пока другое: состояние f16 при fast, f32 при exact.
- Точность: слагаемые KLD складываются как корень из суммы квадратов.
  Проверка на дефолте: формула 0,0044 при замере 0,0040; простая сумма 0,0067 — верхняя граница.
  Для не-Qwen семейств слагаемые помечены «по Qwen 9B».
- Найдено: serving.md неполно перечисляет AllowA8. По formats.cpp kept_block_policy int8-активации получают
  все GGUF-форматы (Q2_K…Q8_0, IQ1–IQ4, MXFP4); у Q3_K префилл от 17 токенов идёт в BF16.
  Задача: поправить документацию.
- Nemotron: после конвертации веса ≈18 ГБ при любом GGUF-кванте, на 16 ГБ не влезает (так и в движке).
- Спекуляция: профиль ищется по классу карты (chip:Nsm, chip), иначе берётся ближайший по цепочке похожести.
  Профиль снят при C=1, при C>1 в сумму не входит.
- 5070 Ti: run_acc.sh дошёл до кривой квантов (идёт prepare Q8_0); run_acc2 (состояние под fast) ждёт.
- 20:55 тик.
  - 5070 Ti: run_acc готовит кванты для кривой (Q8_0 и Q6_K готовы, идёт Q5_K_M); run_acc2 ждёт.
  - Новый run_chunk.sh (чанки 256…4096, C=1 с промптом 22K и C=4) запущен на двух стендах:
    - 4070 Ti: Gemma 4 12B Q4_K_M и Gemma 26B-A4B UD-IQ2_XXS;
    - 2080 Ti: Qwen3.8-27B UD-IQ2_XXS.
  - 4090 свободен, но на нём нет ни моих сборок, ни артефактов.
  - PR #87 (черновик, только комментарии): по коду (kept_block_policy, w4a8_supports) int8-активации под fast
    получают все GGUF-форматы, включая Q2_K/Q3_K/Q8_0/IQ1–IQ3/MXFP4. Заметка в formats.h
    «у Q3_K префилл от 17 токенов в A16» устарела. Отправлен на @codex review.
  - Конструктор v2: исключение для Q3_K убрано.
    Префилл 35B на 5070 Ti по формуле 7332 ток/с при замере 7334 (это точка подгонки).
  - #86: Codex ещё работает.

## 2026-10-23 21:30 — единый выбор конфигурации (поручение владельца)
- План и файл решений для владельца:
  - C:\Users\Pers_PC\Desktop\Claude\Inferno\choice\PLAN.md
  - C:\Users\Pers_PC\Desktop\Claude\Inferno\choice\DECISIONS.md
- PR A #88 (черновик; база — det/plan-no-load, то есть #60 разработчика): src/product/choice.
  - Модель скорости, точности и полезность −ln T − λ·KLD, λ = 25.
  - Квант пока выбирается правилом #60 (кривой KLD по квантам ещё нет).
  - ctest на sm120 проходит: декод 5,21 % rms, префилл 2,03 % rms на 49/15 точках подгонки.
  - @codex review запрошен.
- PR B (ветка det/choice-wire, PR ещё не открыт):
  - product/plan/plan_choice: выбор при весах модели, память — план без весов на каждого кандидата;
  - serve: выбирает KV, активации и состояние, которые не заданы флагами (choose_served_axes), пишет строку «choice |»;
  - --plan --no-load: раздел JSON "choice";
  - /api/plan в GUI: раздел "choice".
  - Сборка идёт на 5070 Ti (run_cb.sh, BR=det/choice-wire).
- Проверка на смеси задач (mix_tasks.py: 8 случайных задач из examples/cli/messages, ответы 96–512 токенов),
  RTX 4090, бинарник p60, артефакты скопированы со стендов. Конфигурации не из подгонки:
  - C=1, средневзвешенное время: 9B Q4_K_M −3,7 %, Q6_K −1,7 %, KV bf16 −4,0 %, Qwen3-8B (новое семейство) −4,5 %,
    27B IQ3_XXS −6,9 %;
  - выбросы: 9B exact −10,4 %, Qwen3.5-4B −15,1 % (постоянные расходы на шаг), C=4 +16,1 % (префилл прячется в декоде).
  - Второй проход (seed 4, с длинными промптами 9K и 22K) — в очереди на 4090.
- Чанк: run_chunk.sh на 4070 Ti (Gemma 12B, Gemma 26B) и на 2080 Ti (Qwen3.8-27B IQ2_XXS) идёт.
- 22:10 проверка на смеси задач, RTX 4090, 16 конфигураций не из подгонки (две выборки задач: seed 7 и seed 4 с промптами 9K/22K).
  Средневзвешенное время: rms 9,3 %, худшее 16,4 %.
  - C=1, 9B (Q4_K_M, Q6_K, Q8_0, KV bf16): −1,7…−4,0 %; Qwen3-8B −4,5/−8,7 %; 27B IQ3_XXS −6,9 %; exact −8…−10 %.
  - Быстрые модели переоценены: 4B −15 %, 35B-A3B IQ3_XXS C1 −16 %
    (накладные расходы на запрос и малые матрицы / формат IQ3 — в следующую подгонку).
  - C>1: 27B C2 +5,5 %, 35B C4 +0,8 %, 9B C4 +16 %.
  Отчёт: C:\Users\Pers_PC\Desktop\Claude\Inferno\choice\validation_4090.txt
- PR B (det/choice-wire): оценка скорости в каждой строке квантов (на глубине контекст/4), why_code у альтернатив.
  Сборка serve на 5070 Ti идёт (542/1155).
- PR C (det/choice-ui): GUI берёт скорость из estimate и choice. Своя JS-модель удалена:
  genTps, KIND_LOAD, SPEC_KEEP, calibOf, measuredReq, prefillTps, CARDS и другие (−159 строк).
  Добавлена панель «Выбор движка / что изменится». test_page_plan: 587 проверок проходят.
- Следующее:
  1. собрать B и C, ctest, проверить --plan --no-load и старт serve («choice |»), GUI вживую;
  2. открыть PR B и C;
  3. прогнать смесь на 2080 Ti, 4070 Ti, 5070 Ti;
  4. переподогнать с поправкой на запрос и проверить на других картах.
- 22:50 тик.
  - Codex #88 (5 замечаний, главное P1): KV и активации — решения B1, по deterministic-decisions.md
    они не могут читать частоту и полосу платы.
    Исправлено (114d4987): решение считается на эталонной карте поколения (kReferenceCards, среднее лабораторных
    карт), поэтому все платы одного поколения решают одинаково, а частота и полоса платы идут только в оценки.
    Решение по полезности включается, только если множители KV и exact сняты на двух картах поколения.
    Сейчас таких поколений нет (у Ada только 4070 Ti), поэтому стоят умолчания семейства.
    Заодно: Volta (у неё нет int8), порядок подстановки η префилла для MoE, ступени Auto перед квантами
    ниже 3,5 bpw, состояние и кольца в объяснении. ctest на sm120 зелёный.
  - Codex #87: TQ1_0/TQ2_0/Q1_0/Q2_0 остаются в BF16. Исправлено, Codex запрошен снова.
  - PR B/C: исправления A влиты в обе ветки. Сборка wire упала на моей опечатке в строке (исправлено);
    полная сборка det/choice-ui (A + B + C) идёт на 5070 Ti, run_cb4.out.
  - 4090: run_sw2.sh (переключатели на 9B, бинарник p60) идёт — это вторая карта Ada для множителей KV/exact.
    Пока: default 150,2 ток/с декод / 11845 префилл; exact 127,9 / 3076.
  - Чанк, промежуточно:
    - Gemma 12B (4070 Ti): префилл 2855 → 3057 → 3112 (чанк 256 → 512 → 1024), декод без изменений;
    - Qwen3.8-27B IQ2_XXS (2080 Ti): 624 → 670.
- 23:40 тик.
  - 4090 run_sw2 готов. Ada, вторая карта:
    - KV: bf16 1,09 / fp8 1,08 / k8v4 1,04 / nvfp4 1,22 (4070 Ti: 1,03 / 1,03 / 1,01 / 1,25) — совпадают в пределах 6 %;
    - exact 3,52 (4070 Ti 2,98) — в пределах 20 %.
    Берём средние; Ada решает по полезности (kv_two_cards / exact_two_cards = true).
    Turing exact пересчитан тем же способом: 4,28.
  - --plan --no-load (сборка det/choice-ui, 5070 Ti), 9B Q4_K_M, --plan-device:
    - 4090 и 4070 Ti — KV int8 + fast по полезности (exact −0,18, KV bf16 −0,004);
    - 5070 Ti и 2080 Ti — умолчания семейства.
    Хаб: оценка в каждой строке квантов.
  - Исправлено по живой проверке:
    - exact не попадал в альтернативы (сравнивалось связанное состояние);
    - глубина оценки теперь одна на всю таблицу;
    - метка кванта в выборе;
    - округление в JSON;
    - префилл exact у MoE: η int8 MoE × отношение bf16/int8 у плотных (−72 %, а не −35 %).
  - GUI вживую (ninferno-ui на 5070 Ti через туннель, Qwen3.6-35B-A3B GGUF): таблица и панель «что изменится»
    работают; панель держится открытой при перерисовке, цвета свои.
  - Открыты PR #89 (B, база det/choice-model) и #90 (C, база det/choice-wire); @codex review для обоих.
  - Сборка det/choice-ui для sm89 на 4090 (run_cb.out) — для проверки настоящего старта serve («choice |»).
  - Чанк, 4070 Ti, Gemma 12B: префилл 2855 / 3057 / 3112 / 3094 / 3042 при чанке 256 / 512 / 1024 / 2048 / 4096;
    C=4 с чанком 4096 — старт отклонён (не влезает). 2080 Ti, 27B IQ2_XXS: 624 / 670 / 673.
- 00:30 (24.10)
  - Codex #88: квант ранжируется только среди допустимых кандидатов (тест ranks_eligible_only).
    #89: маршрут кванта берётся по метке строки; умолчания в serving.md, reference и cli.md обновлены.
    #90: панель и скорость относятся к показанному кванту (choiceFor + запрос плана выбранного кванта).
    Ответы и @codex review отправлены.
  - Новое:
    - CLI ninferno берёт выбор (choose_for_artifact / start_axes / apply_start_axes);
    - связанные хранилища по правилу владельца;
    - r0 = 0,135 с только в оценках;
    - η префилла exact для MoE;
    - заметка перенесена в docs/maintainer/configuration-choice.md, раздел в deterministic-decisions.md,
      choice в аудите decision_inputs (96 файлов, чисто);
    - место под кривую KLD по квантам (пусто, пока нет данных).
  - 4090: полная сборка det/choice-ui + ctest -E real (run_ct.out).
    Смеси для проверки поставлены в очередь на 2080 Ti, 4070 Ti, 5070 Ti.
  - 5070 Ti: кривая квантов — Q8_0 (эталон) и Q6_K готовы.
- 01:30 (24.10)
  - #87 влит (c039f717, Codex чисто).
  - Настоящий старт serve (5070 Ti, сборка det/choice-ui), 9B:
    - по умолчанию — choice KV int8 / fast / f16, строка capacity совпадает;
    - --kv-dtype bf16 — состояние f32 по связке (runtime 508 МиБ против 312);
    - --activations exact — f32.
  - GUI вживую: после щелчка по другому кванту выбор пересчитывается (запрос плана этого кванта);
    скорости в таблице на одной глубине.
  - Проверка смесью на 4070 Ti (вне подгонки):
    - с r0 оценки +8 %, без r0 в пределах ±5 % (Gemma 12B +5 %, Qwen3-8B +3 %);
    - на 4090 без r0 крупные модели −2…−7 %, быстрые (4B, 35B) −15 %.
    r0 убран: это расход хоста, из фактов GPU его не вывести (DECISIONS 14а).
  - Чанк, MoE (Gemma 26B, 4070 Ti): префилл 4314 / 5514 / 6567 / 6987 / 6994 при 256…4096 — насыщение к 2048.
    Плотные: ±5 %. Тот же замер на 4090 (Gemma 26B, 35B, 9B) поставлен после ctest — вторая карта и проверка
    гипотезы «эффективность MoE зависит от токенов на эксперта».
  - Кривая квантов (KLD к Q8_0): Q6_K 0,0062, Q5_K_M 0,0106, Q4_K_M — готов; остальные идут.
  - ctest det/choice-ui на 4090: 145/275, падений нет.
- 02:10 (24.10)
  - Ось «чанк» в ядре: префилл MoE по токенам на эксперта (kMoeChunkShare по Gemma 26B на 4070 Ti);
    кандидаты 1024/2048 только у MoE. Пока стоит 1024 (одна карта), альтернатива показывается с причиной.
    Сервер применяет самый широкий чанк (лестница может сузить), в CLI чанк закреплён.
    Документация обновлена; тест на sm120 зелёный.
  - Проверка смесью на 2080 Ti: 9B −1,8 / −1,1 %, exact +3 %, KV bf16 −4 %, C4 +4 %;
    27B IQ2_XXS −19 % (модель быстрее реальности) — вероятно, распаковка IQ2 на Turing.
    Тот же 27B IQ2_XXS поставлен на 4070 Ti (q_mix3.sh) — проверить, Turing это или формат.
  - Ждём: кривую квантов (фоновое ожидание), ctest на 4090 (идут тяжёлые тесты внимания),
    чанк на 4090 после ctest.
- 03:00 (24.10)
  - Кривая квантов готова (KLD к Q8_0, 9B): Q6_K 0,0062 / Q5_K_M 0,0106 / Q4_K_M 0,0193 / IQ4_XS 0,0283 /
    Q3_K_M 0,068 / UD-IQ3_XXS 0,107 / UD-IQ2_M 0,161 / UD-IQ2_XXS 0,334.
    Вписана в ядро (kQuantKld).
  - Квант по полезности внутри ступени #60 на поколении с правом решать: plan_quants, best_worth.
    9B на 4090 и 4070 Ti → UD-Q5_K_XL (самый крупный — UD-Q8_K_XL / Q8_0); на 5070 Ti → UD-Q8_K_XL (правило).
    План занимает 0,76 с против 0,58 с. GUI знает best_worth. DECISIONS 18 — прошу подтверждения владельца.
  - acc2: состояние под fast на 9B — f16 0,00398, bf16 0,00422 против 0,00388 у одного fast.
    Свой вклад состояния: f16 0,00089, bf16 0,00166 (вписано вместо значений 27B из документации).
  - 27B UD-IQ2_XXS на 4070 Ti: оценка +3 % (на 2080 Ti −19 %) — это особенность Turing, записано.
  - Проверка смесью: 4070 Ti — 10 конфигураций, rms 5,1 %. 5070 Ti — смесь идёт.
  - ctest на 4090: 150/275, падений нет. Codex перезапрошен для #88–#90.
- 03:40 (24.10)
  - ctest det/choice-ui (037fb7ee) на 4090 (sm89): 275/275 пройдено, 2925 с.
  - Полный ctest на свежем коде запущен на 4070 Ti (sm89) и 2080 Ti (sm75), run_ct.out.
  - Codex (#88: 3 замечания, #89: 1) исправлено:
    - точка кривой по метке — только для измеренного файла;
    - короткий промпт заполняет только свою часть чанка;
    - кванты, ранжированные по полезности, объясняются через полезность;
    - явные флаги передаются в план как есть (DryRunInputs::pins).
  - Проверка смесью на 5070 Ti: rms 5,1 %, худшее +8,7 % (exact с поправкой, перенесённой с Ada).
    Общий отчёт: C:\Users\Pers_PC\Desktop\Claude\Inferno\choice\validation.md.
  - 4090: замер чанка (Gemma 26B, 35B, 9B, бинарник p60) после ctest — идёт.
- 04:30 (24.10)
  - Чанк MoE на двух картах и двух моделях — одна кривая по токенам на эксперта (в пределах 6 %):
    Gemma 26B (4070 Ti, 4090), 35B (4090). Ada решает чанк (1024/2048/4096);
    35B на 4090 → чанк 2048 (оценка префилла 16079).
  - Codex, третья волна: последний неполный чанк; квант ниже порога объясняется правилом.
    Четвёртая волна (#89):
    - P1: выбор считается на простаивающей карте (не зависит от свободной памяти);
    - P1: все строки оцениваются на своём выборе до фильтра памяти;
    - P2: пресет exact закрепляет bf16.
  - Ускорение: при выборе кванта сначала верхняя оценка без памяти, планы только пока строка может выиграть.
    9B 0,63 с; 35B 8,7 с против 8,3 с у самого #60 (время — перебор квантов в #60).
  - Отчёт для владельца: C:\Users\Pers_PC\Desktop\Claude\Inferno\choice\README.md.
  - Идут: ctest на 4070 Ti (147/…) и 2080 Ti (сборка); на 4090 после чанка — сборка, старт 35B без флагов
    и смесь (q_v.sh).
- 05:00 (24.10) Codex, пятая волна:
  - #88: равная полезность объясняется правилом ничьей (tie_context);
  - #89 P1: перевыбранная строка передаёт свой контекст на поток в запуск;
  - #89 P2: незакреплённый чанк запускает выбор.
  Проверено на sm120: 9B на 4090 при C=4 → Q6_K, у каждого потока свои 32K.
  Идут: 4090 (чанк 9B → сборка → старт 35B → смесь), ctest на 4070 Ti, сборка/ctest на 2080 Ti.
- 05:30 (24.10)
  - Проверка на 4090: настоящий старт 35B UD-IQ3_XXS без флагов → чанк 2048 применён (capacity: chunk 2048).
    Но KV выбран k8v4: ошибка кванта (0,22) под корнем из суммы квадратов «съедала» вклад KV.
    Исправлено: слагаемое кванта прибавляется линейно, корень — только между переключателями
    (DECISIONS 19); теперь 35B → KV int8, выбор кванта 9B не изменился.
  - Смесь 35B с чанком 2048: 7,9 с, как и с 1024 (промпты короткие, чанк не работает).
  - Полный ctest на свежем коде (42e32f0a) запущен на 4090. 4070 Ti — 149/…, 2080 Ti — 137/…, без падений.
- 06:00 (24.10)
  - Разбор нескольких потоков (C>1): систематического сдвига по C нет. Остаток зависит от формата весов
    (27B UD-IQ3_XXS на 5070 Ti −15 %, UD-Q4_K_XL на 4090 +3…+10 % при 4–8 потоках, 9B Q4_K_M ±4 %).
    Записано в choice_constants.h как известный остаток. Поправка по форматам требует своих замеров.
  - choice_check.py на 5070 Ti: 9B C=1 +4,3 %, C=4 +9,3 %; дальше Q8_0 — идёт.
  - Копирование квантов на 4070 Ti: 6 из 8. run_acc_b: эталон ref9 готов, слагаемые идут.
  - ctest: 2080 Ti (6c1cc7d7) и 4090 (42e32f0a) — идут.
  - Sonnet: ревью ядра (только идеи) — идёт.
- 06:10 (24.10) Проверка стендов: 4090 ctest 242/275, 2080 Ti ctest идёт; 5070 Ti choice_check 9B Q8_0 +5,3 %, сейчас 35B;
  4070 Ti — слагаемое kv_int8 точности идёт, копируется последний квант (UD-IQ2_XXS). Очереди разработчика пусты.
- 06:25 (24.10)
  - choice_check на 5070 Ti, 6 прогонов: от −4,8 % до +9,3 % (validation.md, новый раздел).
  - 4090: q_cc.sh (8 прогонов по формам моделей, Ada решает) — стартует после ctest → /root/claude/cc.out, CCDONE.
  - 5070 Ti: run_fmt.sh — скорость при C=1/2/4/8 по 7 квантам 9B (остаток при нескольких потоках по формату) → run_fmt.out.
  - 4070 Ti: кванты скопированы, COPIED поставлен — кривая точности пойдёт после слагаемых.
- 07:10 (24.10) Ревью Sonnet (только идеи) — 10 замечаний, проверены и исправлены (#88 7e3ca0de, #89, #90 7953615d):
  - путь «решает полезность» теперь под тестом (параметр Gates, тест открывает флаги);
  - чанк MoE решается по скорости за своим флагом (DECISIONS 20) → сейчас Ada берёт 2048 (проверено: Gemma 26B на 4070 Ti);
  - explain: ничьи у квантов, порог 3,5 бита, tie_order по оси; кривая кванта без скачка у Q8_0; противоречивые пины;
  - JSON: chunk_by_speed; UI: слова для закрытого флага и порога.
  - DECISIONS 21, 22 (вопрос владельцу: перенос слагаемых на другие семейства).
  - Сборка и 7 тестов на 4070 Ti (7953615d) — ок. Codex позван на #88.
- 08:00 (24.10) Цель записана в начало choice/PLAN.md; таймер начинается со сверки с целью; хук Stop (choice/goal_reminder.bat, в ~/.claude/settings.json) раз в 30 мин возвращает к цели.
  - 4090: ctest 275/275 (42e32f0a); choice_check 8 прогонов, 5 в пределах 5 %, все в пределах 13 % (validation.md).
  - **Находка (критический путь):** точность на 4070 Ti — KV сходится с 5070 Ti (int8 0,00092 против 0,00099; fp8 0,0026 против 0,0030; k8v4 0,00535 против 0,00532; nvfp4 0,00983 против 0,00981),
    но `--activations fast` даёт 0,0294 KLD (top-1 95,8 %) против 0,0039 (98,5 %) на 5070 Ti — в 7,6 раза больше.
    Проверяю: 4090 run_actx.sh (fast/int8 против exact), 5070 Ti — то же на сборке det/choice-ui (Q4_K_M и Q8_0) после run_fmt.
    Гипотезы: маршруты префилла на Ada (таблица ad104, MMQ) / int8-префилл для Q4-6 G64 (qx_prefill_a8_pays — но он и на Blackwell true).
- 09:30 (24.10) **Активации по поколениям** (#88 a7e9b29c, #89, #90 32f54537; тесты на 4090 — ок):
  - fast на одной сборке 7953615d: 4090 0,0302 (top-1 95,75 %), 4070 Ti 0,0294, 5070 Ti 0,00388 (98,51 %) → свойство поколения (ядра int8-префилла Ada), не кода и не таблиц маршрутов (без таблиц на 4090 — бит в бит то же).
  - KV-слагаемые — константа модели (две карты, ≤13 %). kKldTwoCards убран: accuracy_decides(gen) по картам слагаемых; ждём кривую квантов 4070 Ti (kQuantKldCards).
  - С открытыми слагаемыми Ada выберет exact для 9B (λ=25: точка безразличия 0,012 KLD; на реальной смеси exact +34 % времени, −0,029 KLD). DECISIONS 23 — вопрос владельцу.
  - Решает только семейство замеров (qwen3_5_text), остальные — правило и умолчания (DECISIONS 22→23).
  - Следующая крупная задача: найти ядро Ada, дающее ошибку int8-префилла (тогда fast на Ada станет и быстрым, и точным).
  - 5070 Ti: run_fmt готов (разбор по форматам — ниже), идёт Q8_0 fast/exact.
- 10:40 (24.10)
  - 2080 Ti ctest 275/275 (6c1cc7d7): все три поколения зелёные.
  - Остаток при нескольких потоках по форматам: 9B, 8 квантов, 1–8 потоков — в пределах 10 %, поправка не нужна (константы, комментарий).
  - Смесь на 4090 (32f54537), сид 7: Q4_K_M fast 12,86 / exact 17,55; Q6_K 15,98 / 22,88; Q8_0 20,15 / 28,30 с.
  - fast против exact: 5070 Ti Q8_0 0,00493; **4090 Q6_K 0,00394** (как Blackwell) против Q4_K_M 0,0302 → дефект int8-пути Q4_K на Ada.
    Отладка run_dbg_q4k.sh на 4090: A — без Ada-тайла 128x128 (w4a8_wide_q4k), B — Q4_K на producer-warp тайле → run_dbg_q4k.out.
  - 4070 Ti кривая: Q4_K_M 0,01920, Q6_K 0,00635, Q5_K_M 0,01063 (5070 Ti: 0,0193 / 0,00623 / 0,01058) — идёт дальше.
  - 2080 Ti: fast/exact на Turing — идёт (сборка perplexity).
- 12:00 (24.10) **Корень ошибки fast найден.**
  - Кривая квантов 4070 Ti = 5070 Ti (8 точек, ≤1,8 %) → kQuantKldCards=2 (локально, не закоммичено до решения по Q4_K).
  - fast на Q4_K_M: Turing 0,0297, Ada 0,0302, Blackwell(5070 Ti) 0,0039; Q6_K/Q8_0 везде 0,004–0,005.
  - Отладка 4090: A (без 128x128) и B (producer-warp) — без изменений; **C (MMQ-префилл выключен) → 0,0039**, префилл −11 %.
  - Механизм: квантователь ggml MMQ (ggml_mmq.cu) пишет в DS4 сумму ИСХОДНЫХ активаций, а точка считается по квантованным → у форматов с минимумом (Q4_0/Q4_1/Q5_1/Q4_K/Q5_K) большие слагаемые не гасят шум. W4A8 NInferno берёт сумму по кодам — точен.
  - D: сумма из кодов (d·Σq), MMQ сохраняется → run_dbg_d.out на 4090.
- 12:40 (24.10) **Исправление MMQ** — ветка det/mmq-code-sums (от master, WIP-коммит запушен):
  - ggml_mmq.cu и ggml_vec.cu: сумма q8_1-блока из кодов (d·Σq). Отладочный D на 4090: 0,0302 → 0,0040, скорость та же.
  - Идёт: 5070 Ti — ninferno_linear_generic_test master против ветки (run_mmqt.out); 4090 — 4B Q4_K_M и 27B UD-IQ3_XXS до/после (run_fixx.out);
    4070 Ti и 2080 Ti — ветка исправления, fast против exact (run_fixb.out).
  - Ядро выбора (локально, не закоммичено): кривая на двух картах, активации — константа модели 0,00394; тест: Ada решает, 9B → Q4_K_M fast int8. DECISIONS 24.
  - 13:10 4B Q4_K_M на 4090: fast 0,01237 → 0,00428 (top-1 97,34 → 98,44 %), 9,05k → 9,01k tok/s.
  - 27B UD-IQ3_XXS на 4090: 0,00231 → 0,00149, 1,72k tok/s оба. 4090: полный ctest det/mmq-code-sums → run_ct_mmq.out.
- 13:40 (24.10) Ядро выбора закоммичено (#88 20a0b120 → #89, #90 94e683ac): кривая на двух картах, активации — константа модели (опирается на исправление MMQ), Ada решает.
  - lab/choice-mmq = det/choice-ui + cherry-pick исправления (только для стенда). 4070 Ti: q_integ.sh (план хаба 9B, choice_check Q4_K_M C1/C4, Q6_K) после run_fixb → run_integ.out.
  - Описание PR исправления: scratchpad/pr_mmq.md (ждёт тест 5070 Ti, 4070 Ti/2080 Ti, ctest 4090).
  - 14:20 4070 Ti, ветка исправления: 9B Q4_K_M fast 0,0294 → 0,00393 (98,48 %), 2,76k → 2,77k tok/s. Тест check_mmq_code_sums (порог по данным) — 5070 Ti run_mmqt2.out.
- 14:40 (24.10) validation.md: таблица доказательства по реальным временам 4090 и KLD после исправления → лучшая Q6_K fast (−3,03), не Q8_0 и не Q4_K_M.
  - 4070 Ti: после интеграции — q_q5xl.sh (UD-Q5_K_XL из хаба, её KLD против Q8_0) → run_q5xl.out; затем скорость на 4090.
  - 14:50 2080 Ti, ветка исправления: 0,0297 → 0,00398 (98,46 %), 1,06k → 1,08k. 2080 Ti: полный ctest ветки → run_ct_mmq.out.
- 15:20 (24.10)
  - Интеграция на 4070 Ti (lab/choice-mmq): план хаба 9B → UD-Q5_K_XL, fast, KV int8, by_worth=true; альтернативы объяснены (bf16 KV −0,008, exact −0,152, int8 tie_order).
    choice_check: Q4_K_M C1 +4,1 %, C4 +13,2 %; Q6_K C1 +4,5 %.
  - UD-Q5_K_XL: KLD против Q8_0 = 0,00936 (интерполяция давала ~0,0098, −4 %).
  - Синтетический тест на случайных весах не различал суммы (0,00043/0,00042) → переписан: центрированные веса Q4_K/Q5_K + выброс в блоке активаций; перезапуск на 5070 Ti → run_mmqt3.out.
- 16:10 (24.10) **GUI вживую на Ada** (4070 Ti, lab/choice-mmq, туннель 18091): хаб 9B → ★ UD-Q5_K_XL, fast, KV int8 по полезности; панель «что изменится» с живыми решениями.
  Найдено и исправлено (#90 b5eee9ca): заголовок карточки «Largest that fits» при best_worth → «The best balance of speed and accuracy». Снимок: choice/gui_ada_4070ti.jpg (отправлен владельцу).
- 16:40 (24.10) **PR #91** (draft) fix(ops): ggml's q8_1 block sums from the quantized codes — один коммит ae04b678 (код + тест 2f; порог 0,01: с исправлением 0,0027, без 0,056 на 5070 Ti). Codex позван.
  #88: комментарий о новых коммитах (зависит от #91), Codex позван. ctest #91: 4090 и 2080 Ti идут (на коде 21642e35 = тот же код).
- 17:10 (24.10) ctest #91: 270/270 на 4090 (sm_89) и 2080 Ti (sm_75).
  - Codex P1 #91: тест сравнивал MMQ с другим ядром (W4A8) → переделан на FP64-оракул (независимая раскодировка весов + quantize_a8); 146622ef. Порог — по прогону на 5070 Ti (run_mmqt4.out).
  - 4090: смесь UD-Q5_K_XL идёт (q_proof2.sh).
  - 17:20 4090: UD-Q5_K_XL fast 15,35 с → полезность −3,07; лучшая измеренная Q6_K fast −3,03: выбор движка в пределах 0,04 (≈4 % времени), validation.md.
- 17:50 (24.10) #91 3ce50e1b: тест 2f на FP64-оракуле (0,0024 против 0,056), порог 0,01; проходит на sm_120/89/75 бит в бит; описание обновлено, Codex позван.
  - 4090 план хаба 9B: UD-Q5_K_XL «best balance» (largest UD-Q8_K_XL) — тот же выбор, что на 4070 Ti (эталонная карта поколения).
  - choice/README.md обновлён для владельца (статус, #91, доказательство, вопросы).
  - Стенды свободны. Дальше: описания #88–#90 под итоговое состояние, ответы Codex, память.
- 18:10 (24.10) Описания #88, #89, #90 переписаны под итоговое состояние (порядок #91 → #60 → #88 → #89 → #90; Ada решает; доказательство; GUI вживую).
- 18:30 (24.10) Охват: слагаемые точности Qwen-MoE (35B UD-IQ3_XXS) на 4090 и 5070 Ti (run_acc_moe.sh, сборка #91) → run_acc_moe.out.
- 18:50 (24.10) refactor: kKldFamilies + kQuantKldFamily (#88 → #89 JSON quant_by_worth → #90 6df7083f); поведение то же, тест ок.
- 19:40 (24.10) MoE: слагаемые 35B на 4090/5070 Ti (≤6 %), отличаются от dense (int8 KV ×3,4) → слагаемые по семействам (#88 61bf279d → #89 → #90 1846c380); MoE на Ada решает KV/активации/чанк, квант по правилу. DECISIONS 25. 4090: q_moe.sh (план хаба 35B, choice_check C1/C4) → run_moe.out.
- 20:00 (24.10) #91 Codex: замечаний нет. 4090: хаб 35B → UD-Q4_K_XL (правило), KV bf16 по полезности (int8 −0,001), fast, чанк 2048; choice_check 35B C1 −15,1 % (остаток хоста). #88: Codex позван на 61bf279d.
- 20:30 (24.10) 35B choice_check C4 на 4090: −12,7 % (остаток хоста). Codex P2 #88 (ничья квантов — меньшая метка, как quant_ranks_above) исправлен, ответ + @codex review; #89/#90 (4d738341) — сборка и 7 тестов на 4070 Ti ок.
- 21:00 (24.10) Охват другими семействами требует режима CausalScoring в программах dense/Gemma/Nemotron (зона разработчика) — записано в README как рекомендованный шаг, не начато. Ждём Codex #88 и решений владельца.
- 21:30 (24.10) Codex P2 #88 (документ/комментарий о ничьей) — исправлено (#88 → #89 serving.md → #90), Codex позван. Все PR mergeable; #89/#90/#91 Codex чисто.
- 22:00 (24.10) Codex P2 #88 (объяснение ничьей правила) — исправлено, Codex позван.
- 22:20 (24.10) Финальный перемер (цель п.5): q_final.sh на всех 4 стендах, сборка lab/choice-mmq (cf82711a + исправление MMQ), choice_check по моделям → run_final.out.
- 23:00 (24.10) Финальный перемер (итоговый код, 4 карты, 10 прогонов): 7 в пределах 5 %, 27B C2 +12,5 %, 35B −13,5 % (остатки). validation.md, PLAN п.5 закрыт.
- (25.10) Владелец: «анализируй, используй Codex, если всё ок — вливай #60 и #91». #91 влит (4576eb47). #60: открыты Codex P1/P2 + мой P1 → исправления в det/p60-fixes (ac582703), PR #92 в ветку #60, Codex позван; lab/p60-master (master+#60+исправления) без конфликтов; ctest на 4070 Ti → run_ct_p60.out; сборка+тесты на 4090 (bpt3gmgcr).
- (25.10) Владелец принял λ=25 (DECISIONS 26). Проверено и описано: одна функция выбора во всех местах (раздел Where it runs, #88 → #89 → #90).
- (25.10) Владелец: убрать опасные запасные пути. Аудит Sonnet: 20 находок; самые опасные — молчаливые подмены маршрута GEMM (MMQ/R8/W4A8 → A16/Scalar). Лаборатория lab/route-demotions печатает каждую подмену; 4090: ctest + реальные старты + perplexity → run_demote.out.
- (25.10) det/no-silent-fallbacks: таблица маршрутов (битая → отказ), устройство (без выдумок), drafter-ы по имени (ошибка). ctest на 5070 Ti → run_ct_nsf.out.
- (25.10, день) det/no-silent-fallbacks, ещё три исправления:
  - d464c3e6 — хаб: снимок ветки/тега из кэша при недоступном хабе только с HF_HUB_OFFLINE=1 (коммит — сам себе снимок); сохранённый артефакт без загрузки — только пока хаб называет тот же коммит, недоступный хаб = ошибка с двумя выходами; 6a45bc54 — тест 11d.
  - 7bb60a9b — кэш конвертации локального пути: штамп источника (путь, размер, mtime файлов + ресурсы) рядом с артефактом; изменился при наличии источника → конвертация заново; старый кэш без штампа штампуется при первой встрече; тест в test_cache_revision.
  - Разобраны без изменений: #6 (device_class кэшируется после первого успеха, движок читает его на старте — значения по умолчанию только в host-only инструментах), #12 (сброс спекуляции детерминирован по файлам и пишется в лог), #14 (dispatch: движок сам отказывает неверному --device), #15/#16 (детерминировано и видно / умолчания HF), #17/#18 (внешний контракт), #20 (выключено по умолчанию), #7/#8 (известные пробелы).
  - Осталось: #1–3 подмены маршрутов GEMM — ждут данных 4090 (lab/route-demotions исправлен 2d64a01e: перевод строки в fprintf; choice_check взят с det/choice-wire) → run_demote.out.
  - Стенды: 2080 Ti ctest 6a45bc54 → run_ct_nsf2.out; 5070 Ti ctest 86e8d1e8 → run_ct_nsf.out; 4070 Ti ctest lab/p60-master → run_ct_p60.out; 4090 q_demote.
- (24.10, 12:20) PR #93 (draft) det/no-silent-fallbacks → master: 8 исправлений + список оставленного с причинами; Codex позван; ревью Sonnet (только находки) запущено. Ожидатель стендов: wait_any.sh.
  - Предпроверка цепочки: #92 сливается в #88/#89/#90 без конфликтов; master → det/choice-wire даёт конфликт в docs/reference/serving.md (строки --kv-dtype/--sliding-kv-dtype/--state-dtype/--activations). Решение: строки из #89, а в --state-dtype — текст master об округлении по сетке 16-токенных чанков + правило «Omitted» из #89.
  - #92 Codex: «running» с 08:17 UTC.
- (24.10, 12:40) 4070 Ti: ctest lab/p60-master (master+#60+#92) 274/274 — комментарий в #92. Ревью Sonnet #93: 3 дефекта исправлены (c2115f59: отказ таблицы повторяется; второй движок без поиска по умолчанию; штамп относительный/UTF-8/только читаемые файлы/исчезнувший ≠ изменение; failed-статусы drafter-ов под --with-drafters). 4070 Ti: ctest c2115f59 → run_ct_nsf4.out; 2080 Ti цепочка → run_ct_nsf3.out. Codex #92 «running» >1 ч.
- (24.10, 13:40) Владелец: сдвинутая ветка при удалённом оригинале → GUI предупреждает и даёт выбор (оставить навсегда / перекачать и стереть), консоль — таймер и старые веса по умолчанию. Сделано в #93 (DECISIONS 27):
  - e5d042f9 ядро acquire: MovedRef Notice/Accept/Refetch, accepted.json, недоступный хаб → старые веса + предупреждение; ninferno-serve и ninferno-convert --fetch: --accept-converted / --refetch; таймер 10 с только на терминале; тест fetch_test 12b (нужен torch: на 2080 Ti venv-torch).
  - 7ca021df GUI: GET/POST /api/moved, метка и 2 кнопки в «Моих моделях», 10 языков; --refetch удаляет только заменённую сборку; тест test_ui.
  - ec471554 Codex #93: incompatible drafter под --with-drafters = ошибка; удалённый конфиг/токенизатор = изменение.
  - Вопрос владельцу: то же правило при неудалённом оригинале?
- (24.10, 14:30) Codex #92 (P1: проектор в плане ≠ проектор при старте) → 398558b2 det/p60-fixes: один выбор convert::best_projector по листингу; тест test_ui; 5070 Ti ctest (120a) → run_ct_p60f.out; Codex позван.
  - 5070 Ti ctest #93 (86e8d1e8, sm_120): 2 падения. selector_golden — моё строгое правило «любая таблица обязана разобраться» (таблица валидируется по фактам этого устройства; ad102.json отвергается на V100) → 9f6e4aed вернул пропуск чужих таблиц. pdl_sass_audit — падает и на master/#91 при сборке ARCH=120; сборка 120a проходит → мой run_ct должен собирать 120a на 5070 Ti (не продукт).
  - Codex #93 P2: --refetch без хаба = ошибка → 756c1544 (+ тест 12b).
- (24.10) 4070 Ti ctest c2115f59: 269/270 (только selector_golden — исправлено 9f6e4aed). 4070 Ti: ctest 756c1544 → run_ct_nsf5.out.
- (24.10, 15:00) Codex #92 2×P2 (раздельный проектор; план под --max-context-cap автоматический) → 3f4ee1b3; 5070 Ti цепочка → run_ct_p60g.out; Codex позван.
- (24.10, 15:10) 4090 q_demote: ctest 270/270, 20 подмен MMQ→rmma только в dense_tune_test (тест сам даёт буфер под W4a8, меньше нужного MMQ); perplexity 9B fast — 0 подмен; choice_check не работает на ветке без #60 → q_demote2.sh (perplexity fast/exact + генерация spec auto по всем моделям) → run_demote2.out.
- (24.10, 15:40) q_demote2: генерация spec auto на 9 моделях (Qwen3-8B, 3.5-4B, 9B×4 кванта, 35B MoE, 27B, Gemma 26B) — 0 подмен; perplexity упал (папка вывода существовала) → q_demote3 (Qwen3.5, fast/exact) → run_demote3.out.
- (24.10, 16:00) Codex #92 ещё 2×P2 (дубль consolidated.safetensors в счёте плана; маленький mmproj без метки выпадал из листинга) → 99228af1 + тест через fetch_head_probe --list. 5070 Ti цепочка p60g возьмёт этот head. q_demote3 (perplexity Qwen3.5) идёт: пока 0 подмен.
- (24.10, 16:10) Подмены маршрутов — проверка на других поколениях: q_demote_gen.sh (сборка lab/route-demotions, perplexity fast/exact + генерация spec auto) после ctest: 5070 Ti (gpt-oss-20b MXFP4 MoE + 9B Q8/Q6/Q5/Q3/IQ3_XXS/IQ2_M), 2080 Ti (9B Q4_K_M, 27B IQ2_XXS) → run_demote_gen.out. Решение «подмена → ошибка» только при нуле на всех поколениях.
- (24.10) 5070 Ti ctest #92 398558b2 (120a): 280/280.
- (24.10, 16:50) 4090 q_demote3: perplexity fast/exact на 7 моделях Qwen3.5 — 0 подмен (Ada в реальной работе: 0). Codex #92 P2 (MTP-часть чекпойнта) → b23a271e; 5070 Ti: p60g (99228af1) идёт, p60h (b23a271e) в очереди.
- (24.10, 17:20) #93 756c1544: ctest 4070 Ti 270/270; fetch_test с torch (2080 Ti venv-torch) — все случаи ок, включая сдвинутую ветку целиком (уведомление/accept/недоступный хаб/refetch). Комментарий в #93.
- (24.10) 4070 Ti: ctest lab/p60-master2 (master + #92 b23a271e, слияние чистое) → run_ct_p60m2.out — итоговая проверка перед вливанием.
- (24.10, 17:40) 4090: подмены под нагрузкой — lab/route-demotions-p60 (+#92), choice_check C=1/C=4 на 6 моделях → run_demote_serve.out.
- (24.10) 2080 Ti ctest #93 756c1544: 270/270. #93 зелёный на sm_89 и sm_75; 2080 Ti дальше: q_demote_gen.
- (24.10, 18:20) choice_check требовал выбор из #89 → lab/route-demotions-p60 = демоции + choice-ui + p60-fixes (94b10bfa; конфликт serving.md взят из #89 — только лаборатория); 4090 перезапуск q_demote_serve.
- (24.10, 18:50) Codex #92 круг 5: 2×P2 (native рядом с safetensors; mtp_bytes в GUI) → 4262dd67. Правило вливания: P1 — всегда до вливания; P2 — дешёвые исправляю; круг Codex только с P2 + чистые тесты → вливаю #92 → #60, остаток P2 — продолжением. q_demote_serve: choice_check копия искала примеры не там → запуск из дерева ветки; перезапущен на 4090.
- (24.10) 5070 Ti ctest #92 99228af1 (120a): 280/280.
- (24.10, 19:20) 4090 q_demote_serve (choice_check C=1/C=4, 6 моделей: 9B Q4_K_M/UD-Q5_K_XL, 35B MoE, 27B, Gemma 26B, Qwen3-8B): 0 подмен (лог сервера реальный: 36k prompt / 7.6k answer токенов на последнем). Ada итого: ctest (только dense_tune_test сам даёт малый буфер), perplexity 7 моделей fast/exact, генерация 9 моделей, сервер C=1/C=4 — 0 подмен. Ждём Blackwell (5070 Ti) и Turing (2080 Ti).
- (24.10, 19:40) 4070 Ti ctest master+#92(b23a271e): 274/274. Codex #92 круг 6 (4262dd67): только 1×P2 (GUI: смешанный листинг safetensors+.ninferno) → по правилу вливаю после ctest 4262dd67 (5070 Ti p60h); P2 — продолжением.
- (24.10, 20:30) **#92 влит в det/plan-no-load (cf768381), #60 влит в master squash (058cb27d)** — разрешение владельца; проверки: master+#92 274/274 (4070 Ti), ветка 4262dd67 280/280 (5070 Ti 120a), Codex круг 6 только P2 → #94 (det/plan-followups, 614fbcfb). #88 → base master; цепочка: #88 ← plan-no-load + master (3 тривиальных конфликта: оставлены блоки #88), #89 ← #88 (serving.md по плану), #90 ← #89 (чисто). Запущено: ctest det/choice-ui 4070 Ti → run_ct_chain.out; choice_check 9B/35B C1/C4 на 4090 → run_chain_cc.out; Codex #88/#89/#90/#94.
- (24.10, 21:00) Codex #90 P2 (документ плана без choice → защита в page.html) → a116de24. Подмены: Turing (2080 Ti) 9B fast/exact/gen, 27B IQ2_XXS fast — 0; Blackwell (5070 Ti) gpt-oss + 9B Q8_0 — 0 (идёт дальше). Идут ctest цепочки (4070 Ti) и choice_check (4090).
- (24.10, 21:20) Turing q_demote_gen: 9B Q4_K_M и 27B IQ2_XXS — perplexity fast/exact + генерация — 0 подмен. 2080 Ti: ctest det/choice-ui (a116de24) sm_75 → run_ct_chain75.out.
- (24.10, 21:40) 4090 choice_check цепочки на master: 9B Q4_K_M C1 −3,5 %, C4 +2,3 %; UD-Q5_K_XL C1 −3,9 %; 35B C1 −14,6 %, C4 −12,7 % (как до слияния) — выбор KV int8/fast/f16 (+чанк 2048 MoE). Регрессии нет; комментарии в #88–#90.
- (24.10, 22:20) Codex P2: #88 неизвестные биты ≥ порога (146091ca + тест), #89 choice_check план = спекуляция сервера (6d51ebe1); цепочка → #90 b11e14b4. 4090: ninferno_choice_test + plan_dry_run_test на b11e14b4 → run_choice_test.out.
- (24.10, 23:00) Blackwell q_demote_gen: gpt-oss-20b + 9B Q8/Q6/Q5/Q3/IQ3_XXS/IQ2_M — 0 подмен. Итог 3 поколения: 0 в продукте → DECISIONS 28: подмены видимы (предупреждение на пару маршрутов + счётчик + тест), не ошибки. det/route-demotion-visible (838a6e65, b36089c4) → ctest 5070 Ti 120a → run_ct_rdv.out.
- (24.10, 23:15) 4090: choice_test + plan_dry_run_test на b11e14b4 — пройдены; полный ctest b11e14b4 на 4090 → run_ct_chain4090.out.
- (24.10) 4070 Ti ctest цепочки c7c70d06: 275/275.
- (24.10, 23:40) Codex круг: #88 P2 (неизвестные биты не взвешиваются по кривой; квант — правилу) → 9e4b1bbf + тест; #89 P2 (--count) → исправлено; #90 → 53244ac1. 4070 Ti: choice_test на 53244ac1 → run_choice_test.out.
- (24.10) choice_test+plan_dry_run на 53244ac1: пройдены. 4070 Ti: полный ctest 53244ac1 → run_ct_chain_final.out.
- (24.10, 00:00) Codex #89 P2 (choice_check: отклонённый запрос = ошибка) → только Python; #90 → bb5ad119 (C++ как в 53244ac1, ctest которого идёт на 4070 Ti). Идут: 2080 Ti (a116de24), 4090 (b11e14b4), 4070 Ti (53244ac1), 5070 Ti (route-demotion-visible).
- (25.10, 00:30) det/route-demotion-visible ctest 5070 Ti 120a: 280/280 → PR #95 (draft), Codex позван. #89 Codex P2 (ошибка потока не роняла прогон) → исправлено, #90 a46dc2c7.
- (25.10, 01:00) 4090 ctest цепочки b11e14b4: 275/275. Codex: #95 P2 (поздняя подмена Volta Tiled→scalar) → dd451ebb, ctest 5070 Ti → run_ct_rdv2.out; #89 P2 (choice_check закрепляет квант плана для хаба) → #90 c4aa9891 (только Python). Ждём ctest 4070 Ti 53244ac1 (C++ финал).
- (25.10) 2080 Ti ctest цепочки a116de24 (sm_75): 275/275.
- (25.10, 01:40) **#88, #89, #90 влиты в master** (34ba85c0, 41a62a53, cd9ad51b): финал 53244ac1 ctest 275/275 (4070 Ti), b11e14b4 275/275 (4090), a116de24 275/275 (2080 Ti sm_75); choice_check в продукте как до слияния; Codex только P2 (исправлены). Конфликт #89 с master после squash #88 — тривиальный (содержимое совпало, diff пустой). master == проверенная вершина. 5070 Ti: ctest master 120a в очереди → run_ct_master.out.
- (25.10, 02:20) Пока владелец не за ПК: #95 Codex P2 (RMMA K не кратен шагу → scalar) → e946b406, проверены все launch-переходы; ctest 4070 Ti → run_ct_rdv3.out. #93: master влит (4 add/add конфликта — обе стороны) + e6ab0596 (задвоенный селектор кликов после слияния; node --check ок); 2080 Ti: ctest sm_75 + fetch_test с torch → run_ct_nsf6.out. Финальный перемер на master (пункт 5): q_final_master.sh на 4 стендах, те же артефакты/C/seed, что до слияния → run_final_master.out.
- (25.10, 02:50) Codex #93 P2 (GUI refetch терял рецепт) → 0c86d8b1 + test_ui (надо прогнать). Финальный перемер master на 4090 (cd9ad51b): 9B Q4_K_M C1 −3,1 %, UD-Q5_K_XL C1 −4,0 %, 27B C2 +12,5 % (как до слияния); остальное идёт.
- (25.10, 03:20) Финальный перемер master на 4090: 9B −3,1/−4,0 %, 27B C2 +12,5 %, 35B −13,6 %, Gemma −1,4 % — как до слияния (−13,5 у 35B). 4090: test_ui/cache/fetch для #93 0c86d8b1 → run_nsf_ui.out.
- (25.10) 4090: #93 0c86d8b1 — ui_test, cache_revision, fetch_test пройдены.
- (25.10) 5070 Ti #95 dd451ebb (120a): 280/280. Следом на 5070 Ti: ctest master, затем финальный перемер.
- (25.10, 04:00) #93 e6ab0596 ctest sm_75 275/275 (fetch_test с torch идёт); #95 e946b406 sm_89 274/274. Комментарии в PR. Финальный перемер master: 4090 готов; 4070 Ti/2080 Ti/5070 Ti в очереди.
- (25.10) #93 e6ab0596 fetch_test с torch: rc 0, 0 FAILED, 4 случая сдвинутой ветки ок.
- (25.10) Финальный перемер master 4070 Ti: 9B Q4_K_M C1 +4,0 %, UD-Q5_K_XL C2 +0,9 %, Q6_K C1 +3,2 % (KV int8/fast/f16).
- (25.10) Финальный перемер master 2080 Ti: 9B Q4_K_M C1 −0,8 %, C4 +3,0 %. Осталось 5070 Ti (после ctest master).
- (25.10, 05:00) ctest master cd9ad51b на 5070 Ti (120a): 281/281. Master зелёный на sm_89/sm_75/sm_120a.
- (25.10) Финальный перемер master 5070 Ti: 9B Q4_K_M C1 +3,6 %, Q8_0 C4 +13,8 %. Итог master на 4 картах: 8 из 12 в пределах 5 %; крайние: 27B C2 +12,5, 35B −13,6, 5070 Ti Q8_0 C4 +13,8 (оценка ниже замера: выбор консервативен).
- (25.10, 05:30) Владелец: вопрос 1 → А + фоновое обновление в GUI с одним перезапуском без зацикливания (DECISIONS 29). Сделать после вопросов.
- (25.10, 05:40) Владелец: GUI — фоновая загрузка не стартует сама, всплывающее окно + отметка в списке (DECISIONS 29 уточнено). Вопрос 2 → полностью: CausalScoring для Gemma/gpt-oss/Nemotron, perplexity, KLD на 2 картах, выбор по полезности (DECISIONS 30).
- (25.10) DECISIONS 30: det/score-all-families 123f9951 — CausalScoring для Gemma/dense/Nemotron (общий causal_score Program; семьи дают workspace causal_score), Engine::document_start_token + perplexity BOS в начале окна. Смоук (ref exact/bf16 + fast --kld): 4090 Gemma 26B + Qwen3-8B; 5070 Ti gpt-oss-20b; 4070 Ti Gemma 12B + Qwen3-8B → run_score.out. Nemotron: артефакта нет; 30B влезает только в 4090.
- (25.10) DECISIONS 29 часть А: det/moved-ref-everywhere 08f8955c (acquire: правило и при неудалённой загрузке — закрепление на коммите; --keep-replaced; fetch_test 12c; тестовый хаб с историей коммитов). Часть Б (GUI окно + фоновая загрузка + один перезапуск) — на паузе: владелец попросил виды спекуляции в конструкторе.
- (24.10, 19:30 стенд) Конструктор v6 опубликован: виды спекуляции модели (только то, что движок запускает), ★ лучший по замеру, поле «токенов вперёд» до максимума движка. Спросил владельца про то же в page.html.
- (24.10) DECISIONS 30 смоук: gpt-oss на 5070 Ti — «Program model does not match the frozen sequence plan»: Gemma/dense/Nemotron не передавали purpose в load_options → 85932a51 (det/score-all-families). Смоук заменён на q_acc_fam.sh (ref exact/bf16, KV fp8 exact, fast bf16, fast fp8): 4070 Ti Gemma 26B+12B+Qwen3-8B, 5070 Ti gpt-oss-20b; 4090 — после копирования gpt-oss/12B (scp -3). 2080 Ti: ctest+fetch_test det/moved-ref-everywhere → run_ct_mre.out.
- (24.10) Nemotron-H: движок поддерживает только MoE-вариант; минимальный GGUF Lightning 30B-A3B = 18 ГБ → только 4090 (одна карта). Качаю UD-IQ3_XXS на 4090 (run_nemo_fetch.out). Вопрос владельцу: одна карта для Nemotron — решать по ней или оставлять умолчания.
- (24.10, 20:10 стенд) Конструктор v7: «авто» = выбор профиля среди драфтеров артефакта (без --with-drafters — MTP из файла или выкл; с ними — ★). Владельцу объяснено по коду: <artifact>.speculation.json пишет convert только при подключении драфтеров (GGUF — только --with-drafters); профили — из реестра в бинарнике, ключ модель+класс GPU (+applies_to репо), НЕ квант; без манифеста — встроенное умолчание MTP3/DFlash 7 или plain. Предложено: учесть квант в ускорении через модель скорости (ждёт ответа).
- (24.10) DECISIONS 30: оценка других семейств даёт абсурд: Gemma 26B/12B PPL англ. 700–12000, gpt-oss 1000–4600 (код 4,8). Не скользящее окно (ctx 128..4096 одинаково), не decode/prefill (step1 = то же). Термы KLD с этих прогонов НЕ использовать. Сравнение Program.causal_score vs ninferno-gemma4 score на тех же токенах (lab/score-debug c83e12f4, 4090) → run_score_cmp.out.
- (24.10) Nemotron Lightning 30B-A3B UD-IQ3_XXS сконвертирован на 4090 (17,5 ГиБ, 905 с). В реестре есть профиль gb203 на «pirola-IQ3_XXS 15.4 GB» — возможно, влезает в 16 ГБ (5070 Ti): проверить для второй карты.
- (24.10) DECISIONS 29 часть Б: det/moved-ref-gui 5ab9099f (поверх det/moved-ref-everywhere): окно + отметка в списке, фоновая подготовка --refetch --keep-replaced, один перезапуск в простое (/health active_requests), один откат, без повторов; tests/ui_moved_update_test.py (поддельные хаб+serve). 2080 Ti: сборка+тесты → run_mrg.out (после run_ct_mre).
- (24.10, 20:40) Оценка других семейств: Qwen3-8B (dense, тот же код что gpt-oss) — PPL нормальная (англ. ~9, код 2,3) → путь Program для dense верен. Gemma 12B: Program и независимый ninferno-gemma4 score дают одинаково абсурдно (739 / 2943 без BOS) на токенах, ИДЕНТИЧНЫХ HF tokenizers (6661/6661) → не токенизация, не путь оценки. BOS: Gemma 739 с / 1952 без; gpt-oss 1185 с / 831 без → правило BOS исправлено ac9e593c (add_bos_token → класс токенизатора Llama/Gemma → TemplateProcessing; тест template_prefix). Эталон: llama.cpp b11539 на 4090 на тех же GGUF → run_llamacpp_ref.out.
- (24.10) Dense-термы: Qwen3-8B на 4070 Ti — KV fp8 KLD 0,0049 (top-1 97,5 %), fast 0,00125 (98,6 %); вторая карта — 5070 Ti → run_acc_fam2.out.
- (24.10) Владелец: реестр спекуляции онлайн (ежедневный перетест на нашем железе, скачивание, офлайн — предыдущий, копия в установке, читаемый файл). DECISIONS 31 записан; вопрос 1 (тот же сервер и ключ, что у релизов?) — ждёт ответа.
- (24.10, 21:30) Dense-термы на 2 картах: Qwen3-8B 5070 Ti — KV fp8 0,004945 (97,58 %), fast 0,001215 (98,61 %); 4070 Ti — 0,004947 / 0,001254 → константы модели. det/choice-families ae1cd5f5: dense_decoder решает KV/активации по полезности (квант — правило) + тест; ring-терм по замеру, если есть. 5070 Ti: ninferno_choice_test → run_cb_cf.out.
- (24.10) Владелец: глубоко продумать спекуляцию (драфтер как микромодель: память, время, настройки по карте), компактный алгоритм → обсудить → потом реализация. choice/SPECULATION.md v2 (с 24 замечаниями Sonnet-рецензента, ключевые проверены). ПОДТВЕРЖДЕНО на 5070 Ti планом: Gemma 12B (unsloth GGUF, --with-drafters) → plain, «measured faster without speculation» ложно: профиль «*» thinking=dflash:7 без served, движок Gemma только MTP → меню plain. GUI не передаёт --with-drafters, но показывает ускорение отдельных драфтеров (app.cpp:968). Формула цены раунда надёжна для 1 ячейки, held-out 17–21 % > 10 %.
- (24.10, 22:10) llama.cpp b11539 (4090, те же GGUF, тот же текст, -c 2048): Gemma 12B PPL 628±61, gpt-oss 720±39 — абсурдная PPL у IT-моделей на сыром wikitext — свойство моделей, не наша ошибка (у нас 739/831: другая схема окон). Точное сравнение по окнам (ninferno-gemma4/-dense score --dump, схема llama.cpp) → run_exact_cmp.out. Вопрос методики: KLD для IT-моделей мерить на тексте в их режиме (чат/собственные генерации). dense_decoder: choice_test на det/choice-families ок (5070 Ti).
- (24.10, 22:40) Точное сравнение с llama.cpp по окнам (схема -c 2048): Gemma 12B 480,7/591,5/647,9 против 479,9/596,5/627,7 — совпадает (0,2–3 %), движок Gemma верен. gpt-oss 1341/970/768 против 1132/894/720 (+18/+8/+7 %) при идентичных токенах (6615, без BOS) — разница в весах (HF safetensors vs GGUF) или в прямом проходе: конвертирую тот же GGUF нашим конвертером и считаю → run_oss_gguf.out.
- (24.10, 23:00) gpt-oss тот же GGUF через наш конвертер: 1177/935/748 против llama.cpp 1132/894/720 (+4 %, как у Gemma +3 % на финальном) — прямой проход примерно верен (остаток, вероятно, BF16-активации exact против F32 у llama.cpp). Артефакт из HF safetensors (bf16 внимание + mxfp4) хуже GGUF-артефакта (q8 внимание + mxfp4): 1341/970/768, в целом 831 против 816 — подозрительно. KLD HF-артефакта против GGUF-артефакта на quick-корпусе → run_oss_kld.out.
- (24.10, 23:30) gpt-oss: KLD HF-артефакта против GGUF-артефакта 0,068 (top-1 86,7 %, PPL 369 vs 347) — слишком много для разницы bf16/q8 внимания; конфиги идентичны; побайтовое сравнение объектов → run_artcmp.out (4090).
- (24.10) Владелец: строго ограничить список моделей → choice/MODELS.md: 10 моделей × до 5 вариантов (HF API, только принимаемые форматы, ≤24 ГБ). Ждёт утверждения.
- (24.10) Обновление модели: часть А ctest 275/275 sm_75 + fetch_test с torch (6 случаев) → PR #96 (draft); часть Б ui_test + ui_moved_update_test 12/12 → PR #97 (draft, поверх #96). Codex позван на обоих.
- (24.10, 23:50) gpt-oss HF- vs GGUF-артефакт: 313 объектов одного формата побайтно равны (mxfp4 эксперты, смещения, sinks, нормы, эмбеддинги), различаются только 50 матриц внимания bf16 vs q8_0 → ошибки HF-конвертации нет; KLD 0,068 = q8-внимание на «растерянном» сыром тексте. Вывод: для IT-моделей сырой wikitext раздувает KLD — мерить на тексте в их режиме (DECISIONS 32, предложение).
- (24.10) Владелец утвердил список моделей (MODELS.md, DECISIONS 33). Просит переформулировать вопросы проще.
- (24.10) Владелец: реестр спекуляции — тот же сервер и ключ, что у релизов (DECISIONS 31, вопрос 1 = А). Вопрос KLD: владелец сомневается в методе; предложена прямая проверка задачами с известными ответами (ждёт «делать?»).
- (25.10, 00:40) det/score-all-families ac9e593c ctest 275/275 sm_89 → PR #98 (draft, Codex). det/choice-families ae1cd5f5 (dense_decoder решает KV/активации) → PR #99 (draft, Codex). Владелец: реестр — сервер/ключ релизов (А). Тесты точности: предложено 7 (MMLU-Pro, GSM8K, MATH-500, GPQA-Diamond, IFEval, ARC-C, HellaSwag) — ждёт согласования. EvalScope 1.10.0 (без дополнений) на 4090; проверка оснастки smoke (30 вопр./тест) Qwen3.5-9B Q4_K_M → run_eval_smoke.out.
- (25.10) Владелец согласовал тесты: MMLU-Pro, GSM8K, MATH-500 (без thinking) + AIME 25/26 с thinking (DECISIONS 34).
- (25.10, 01:10) SPECULATION.md v3: экономика драфтера = E(k) токенов за раунд (пара моделей, разброс по картам 2–8 %) ÷ R(k) цена раунда (карта+квант, разброс до 40 %); ускорение = E/R подтверждено на 248 замерах реестра (медиана 0,8 %). E — в реестре, R — формулой в программе, память — в плане. Вопрос владельцу.
- (25.10, 01:40) Матрица спекуляции 50 вариантов (план без весов, 5070 Ti): GGUF Qwen3.5 — нет; Gemma 12B — пусто (ошибка); остальные — «похожая карта» с поднятыми порогами (Muse 1,37→4,14). Ошибки списка: sanskar003 AWQ — 401 (замена cyankiwi); cdiamond iMatrix и meta-models Muse — исправлено в det/gguf-file-roles 32010d81 (+тест), ctest 2080 Ti → run_ct_gfr.out. Владелец: драфтеры искать вручную — карточки и файлы на HF, базовая модель и соседние варианты.
- (25.10, 02:20) Ручной разбор драфтеров (карточки + файлы HF) → choice/DRAFTERS.md. Новые: z-lab Qwen3.5-9B/4B DFlash, incoai Qwen3.6-35B DFlash2, GGUF-ассистенты Gemma в репо unsloth/HauhauCS, MTP-файл Qwen3.8 у unsloth; unsloth Qwen3.5 GGUF без MTP (есть MTP-GGUF репо). Движок не умеет: DFlash Gemma/Nemotron, DFlash2 dense, EAGLE3. Конвертер умеет GGUF-ассистента Gemma и MTP Qwen из GGUF.
- (25.10) Первый замер E(k)/R(k): gpt-oss dflash:1..7 spec-tune на 5070 Ti → run_st_gptoss.out (evalq/q_spectune.sh).
- (25.10, 03:00) Аудит решений по имени: (1) list_quants skip «imatrix» где угодно — cdiamond пропадал из GUI; (2) drafter = «dflash/eagle/assistant» где угодно — модели nerkyor …-DFlash2-GGUF и т.п. как драфтеры → det/gguf-file-roles dc7064fb (одно правило gguf_auxiliary_file + fetch_test «file roles»); (4) converted_artifact подстрокой (Q4_K_M ⊂ UD-Q4_K_M → повторная конвертация) → #96 77e8be09 + test_ui; #97 смерджен с ним. (3) размер драфтера в GUI по имени (assistant / kind содержит mtp / делим на 2, 3.5) — уйдёт с реестром v2. (5) --quant подстрокой (Q4_XS ⊂ IQ4_XS) — только ручной ввод. 2080 Ti: ctest gguf-file-roles → fetch_test + test_ui → run_names.out.
- (25.10, 03:20) Оснастка тестов работает (4090, Qwen3.5-9B Q4_K_M, C=8, без thinking): MMLU-Pro 420 вопросов / 409 с (62,9 %), MATH-500 150 / 192 с (85,3 %), GPQA 30 / 98 с (23 %). GSM8K — тайм-аут загрузки с ModelScope (докачается). Оценка полного прогона: MMLU-Pro 12 032 ≈ 3,3 ч на конфигурацию 9B — 20 конфигураций (10 моделей × 2 кванта) ≈ 80–150 ГПУ-часов на 4 картах. Короткий прогон согласованного набора (с AIME thinking) → run_eval_smoke2.out.
- (25.10, 03:50) Первая кривая E/R: gpt-oss-20b DFlash на 5070 Ti, k=1..7 по разделам корпуса. R почти не зависит от раздела и растёт линейно (k=2 2,04 → k=7 2,46–2,50), E — от раздела (k=7: math 4,32, thinking 4,30, code 3,18, prose 2,83) → ускорение 1,18–1,75. Конвейер spec-tune → E/R работает.
- (25.10) Владелец: убрать решения по именам файлов (DECISIONS 36): роли из реестра v2 / заголовка GGUF.
- (25.10, 04:40) Codex P1 по всем PR исправлены: #96 0490f1bd (закрепление только если снимок держит запрошенную модель: snapshot_holds_model + fetch_test 12c), #97 9bf5e059 (перезапуск/откат по пути артефакта, без чужого --spec-profile, docs/ui.md), #98 099f7066 (эталон KLD хранит токен начала окна, NKLDREF2), #99 — нужна 2-я Ada: Qwen3-8B на 4090 в очереди → run_acc_fam_q8b.out; затем документация выбора.
- (25.10, 05:00) Схема реестра v2 упрощена (владелец: «много лишнего»): только drafters{e, repo@commit}, variants{drafters, files}. q_names на 2080 Ti перезапущен на свежем коммите det/gguf-file-roles (+ gguf_test). Gemma 12B: 5070 Ti замер идёт (assistant+dflash сконвертированы), 4070 Ti конвертация.
- (25.10, 05:20) Черновик реестра v2 (choice/registry_v2_draft.json, 32 КБ): 10 моделей × 50 вариантов @commit, файлы по меткам (+mmproj, safetensors → "weights"), драфтеры моделей @commit (z-lab/incoai/google/ggml-org/meta-models), e пустые. Дальше: (1) проверить «какие драфтеры подходят варианту» самим продуктом (ninferno-convert --plan-drafters на каждый вариант); MTP-файл Qwen3.8 у unsloth (MTP/mtp-…Q4_0.gguf) — схема должна позволять файл драфтера внутри варианта; (2) метки файлов сверить с заголовками GGUF; (3) e из замеров spec-tune.
- (25.10, 05:40) Короткий прогон согласованного набора (Qwen3.5-9B Q4_K_M): MMLU-Pro 61,7 % (420/346 с), GSM8K 96,7 % (30/54 с), MATH-500 86 % (150/150 с), AIME25 5/5 (236 с), AIME26 3/5 (473 с) с thinking. Полный ≈ 4 ч на конфигурацию 9B. Полный прогон 9B Q4_K_M в очереди на 4090 после Qwen3-8B → run_eval_full_9b.out.
- (25.10, 06:00) Владелец: не нести сырые данные; глубоко проверить: (1) опознание модели по файлу (архитектура/геометрия/словарь/base_model) без списка вариантов — на 50 вариантах и чужих моделях; (2) перенос E между квантами, файнтюнами и их собственными драфтерами; (3) минимальный реестр. Прийти с выводами.
- (25.10, 06:30) Опознание: все 47 прочитанных вариантов продукт сам связал с правильной записью реестра (repo / base_model / форма) → список вариантов для опознания не нужен. Внешние данные (WaveCut, NInfer upstream): uncensored-файнтюн Qwen3.8-27B + базовый DFlash2 → 2,90 ток/раунд против 4,5–4,6 у базы (−37 %): E на файнтюны не переносится. Идея: цепочка base_model на HF (quantized vs finetune/merge) решает, наследует ли вариант E модели. Замер на 5070 Ti: Qwen3.5-9B MTP-GGUF Q4/Q8/IQ2 + HauhauCS + JEV, MTP и z-lab DFlash (лаб. реестр) → run_e_variants.out.
- (25.10, 06:45) Проверка идеи «цепочка base_model решает наследование E» на 50 вариантах: ОПРОВЕРГНУТА — HauhauCS Qwen3.5-9B / Gemma 12B Uncensored и 0bserverx Muse Heretic помечены на HF как quantized (на деле другие веса); QAT Gemma — finetune (верно). Метаданным HF доверять нельзя → вариантам нужен свой подтверждённый замер E (как у базы / своё).
- (25.10, 07:10) #99: Qwen3-8B на 4090 (2-я Ada): fp8 0,004947 (=4070 Ti), fast 0,001236 → e96d5761 (две Ada + Blackwell, docs). Codex позван снова на #96–#99. Спекуляция: R(k) прямая (gpt-oss 1,84+0,093k, ≤4 %), E(k) = 2 числа (a, d) ≤2 % — проверить на MTP/ассистенте/вариантах.
- (25.10, 08:00) Перенос R между картами: Gemma 12B ассистент 4070 Ti (Ada) против 5070 Ti (Blackwell) — R(k) совпадает ≤1,7 % при k≤6, +14 % при k≥8 (порог ширины ядра); E ≤1,8 %; выбор k по задачам одинаков, потеря ≤2,7 %. Прежние линейные подгонки (1,107+0,146k против 1,02+0,183k) расходились только из-за ступени k≥8. Значит кривая x(k)=E/R пары измеряется один раз и переносится. Модель скорости speed_of для R не годится: Gemma недооценивает на ~0,13+0,017k (накладные раунда+проход ассистента), gpt-oss MoE переоценивает проверку в 2,5 раза (0,24 против 0,093 за строку). MTP Qwen3.5-9B: доп. цена ≈0,12 за проход ≈ доля байт MTP-слоя+головы (1 млрд параметров словаря).
- (25.10, 08:00) z-lab Qwen3.5-9B DFlash движок отказывал на всех k («QKV input projection: unsupported logical geometry»): fused Q/K/V-ядро драфтера скомпилировано только под 2048/5120, а 9B (4096, Q8 одним родителем) шёл в него. det/dflash-generic-qkv 338bd96d: fused только для своих геометрий, иначе общий банк (+тест). 4070 Ti: сборка, тест, конвертация, spec-tune dflash:1..15 → run_df9.out. 5070 Ti: E-варианты продолжают MTP (DFlash в них падает до фикса).
- (25.10, 08:40) PR: #99 смержен (fa5b37d6; Codex без замечаний, CI ок). Codex на новых коммитах: #96 P1 split-GGUF (снимок с одной частью из двух считался полным → закрепление на старом коммите) + P2 метка --vision-конверсии → 651d75b5 (+fetch_test 12c, test_ui); #97 P2×2 (драфтеры старого коммита оставались; провал отката не сообщался) → 808b3141 (правило конвертера по манифесту спекуляции, +ui_moved_update_test). 2080 Ti после q_names: сборка+тесты → run_p1fix.out.
- (25.10, 08:40) Профили Gemma 12B на двух картах (spec-tune c=1): thinking mtp:6 ×2,73 (Ada) / ×2,79 (Blackwell), без thinking mtp:4 ×2,16 / ×2,17, смесь mtp:5 ×2,27 / mtp:6 ×2,28. Пороги (floor) = R(k). Проверка переноса таблицы нагрузки (spec-tune --load 2,4,8) на обеих картах → run_load.out (после df9 / e-вариантов). gpt-oss DFlash на 4090 (Ada) после полного прогона тестов → run_oss_ada.out.
- (25.10, 09:30) E между квантами одних весов (Qwen3.5-9B MTP, 5070 Ti): Q4_K_M против Q8_0 — E ≤2,3 % (переносится); R у Q8 на 3–13 % ниже (шаг тяжелее 10,7 против 7,0 мс). Модель времени раунда T(k) = шаг·(1+v·k) + a + b·k: v=0,092 (проверка, растёт с шагом), a=0,85 мс на раунд, b=0,82 мс на черновой проход (не зависят от кванта цели) — оба кванта ≤3,4 %. Проверка вне выборки — UD-IQ2_M (идёт). Выбор k Q4→Q8 теряет ≤3,5 %, ускорение Q8 ×2,10 против Q4 ×1,89 (смесь).
- (25.10, 10:00) det/gguf-file-roles: ctest 275/275 sm_75 + gguf_test + fetch_test 0 падений («file roles» ок) → PR #100 (draft, Codex). Мои опечатки экранирования (перевод строки внутри литерала) в тестах: det/dflash-generic-qkv b871747b, det/moved-ref-everywhere 355c3d3e (влит в #97 b7604555) — сборки перезапущены (4070 Ti run_df9, 2080 Ti run_p1fix). Урок: C++-правки через Edit, не через python-heredoc.
- (25.10, 10:40) DFlash Qwen3.5-9B на det/dflash-generic-qkv (4070 Ti, Q4_K_M, z-lab@5fc3b3d4): грузится и работает; attn_input_proj_test OK. E: thinking k=7 5,74 / k=15 7,67; код k=15 10,9. R: k=1..7 1,65–1,74 (ширина блока 16 — драфтер считает весь блок), но k=9 2,57 > k=11 2,13 > k=15 1,99 — немонотонно: проверка 10 строк дороже 16 (маршрут цели?). Ускорение thinking ×3,32 (k=7) … ×3,85 (k=15), код ×5,45. Значит для DFlash R(k) не прямая — хранить замеренную кривую. Аномалию ширины 10 сверить на Blackwell (5070 Ti после E-вариантов).
- (25.10, 10:40) q_load ждал сам себя (pgrep -f находил свою командную строку) — перезапущен с шаблоном [q]_…; 4070 Ti уже меряет нагрузку.
- (25.10, 10:50) PR #101 (draft, Codex): DFlash Qwen3.5-9B через общий банк Q/K/V.
- (25.10, 11:30) #97 b7604555: ui_test ок, ui_moved_update_test ок («драфтеры старого коммита удалены»). fetch_test: новый случай split ок, но следующий «--refetch» падал — выборка другого кванта по новому коммиту переставляет refs/main (кэш HF), тест восстановлен 0d3e8e13 (#97 733ffceb), перезапуск run_p1fix. Замечание по продукту: после выборки ДРУГОГО кванта сдвинутой ветки старая конверсия (Q8_0) больше не находится kept_artifact (ref → новый коммит) — следующий старт молча скачает новый коммит. Узкий случай, записать в #96.
- (25.10, 12:10) E-варианты (5070 Ti, MTP): UD-IQ2_M — E ниже Q4 на 6–14 % (k=4 thinking 3,60 против 4,19; код 5,40 против 5,88): на 2-битный квант E НЕ переносится (Q4↔Q8 — да, ≤2,3 %). R вне выборки по модели T(k)=шаг·(1+0,092k)+0,85+0,82k мс: −4…+7 % (k=1 хуже всего). HauhauCS — только DFlash (до фикса падал, перезамер в q_e_dflash). JEV-9B: конвертация падала «missing source tensor mtp.fc.weight» — конфиг объявляет MTP (mtp_num_hidden_layers 1), весов mtp.* нет (427 тензоров): план драфтеров брал объявление → det/mtp-from-weights 3f19d034 (решает индекс весов / заголовок; без списка — объявление; NextN — объявление) + test_drafter_plan. 2080 Ti: тест + конвертация JEV → run_mtpw.out.
- (25.10, 12:40) #96/#97 на 733ffceb (2080 Ti): ui_test, ui_moved_update_test, fetch_test — все проходят. Codex позван на обоих.
- (25.10, 13:10) Нагрузка Gemma 12B ассистент на 4070 Ti (temp 1.0): C=2/4/8 → глубина 3 (×1,99/×1,69/×1,42), load «2:3». Ждём 5070 Ti.
- (25.10, 13:40) #101 смержен (9895ebc6; Codex без замечаний; attn_input_proj_test + spec-tune 9B на 4070 Ti). #96 новый P1 Codex: синтетический листинг кэша без размеров → выбор по ширине (--quant 8bit) не находил скачанное → 7f3d14cc (+fetch_test 12c по ширине), влит в #97 d9b258be; 2080 Ti перепроверка после q_mtpw → run_p1fix. #100 P2 (ассистент «…-assistant-Q8_0.gguf») — гипотетический: опубликованные ассистенты Gemma называются mtp-… (unsloth, HauhauCS) и распознаются; ответил фактами (DECISIONS 36: роль неизвестного имени — из реестра/заголовка). det/mtp-from-weights: drafter_plan_test ок, конвертация JEV-9B идёт.
- (25.10, 14:00) Черновик решения реестра → scratchpad/registry_design_draft.md. Длина контекста: замеры X13 в текущем реестре (27B, 4090 и 5070 Ti) — множитель E по классу промпта (short/mid/long) одинаков на картах ≤6 % (dflash2 1/0,87/0,61 vs 1/0,86/0,59) → тоже свойство пары, 3 числа. Открыто: множитель не зависит от k?
- (25.10, 14:30) Таблица нагрузки Gemma 12B (temp 1): E при нагрузке одинакова на картах (≤2 %), но выгода НЕ переносится: C=2 обе глубина 3 ×1,99; C=4 Ada гл.3 ×1,69 / Blackwell гл.1 ×1,51 (гл.3 ×1,46); C=8 Ada гл.3 ×1,42 / Blackwell гл.1 ×1,08 (гл.3 ×1,03). Load: Ada «2:3», Blackwell «2:3,3:1». Физика: проверка C·(k+1) строк (до 32) — на Blackwell 5070 Ti раньше упирается в вычисления (отношение вычисления/память ниже, чем у 4070 Ti). Вывод для схемы: кривая одной заявки (C=1, ≤7 строк, упор в память) — свойство пары и переносится; таблица нагрузки — свойство пары И поколения карт (замерять на двух картах поколения, как требует deterministic-decisions).
- (25.10, 15:00) Владелец: JEV — не чат-модель (верно: text-classification, LoRA + голова решений; JEV-27B так же). Моя ошибка — выбор по популярности без чтения карточки. JEV остановлен везде. Аудит карточек 50 вариантов: не-чат только JEV-9B/27B; gpt-oss-safeguard — разметка безопасности (вопрос владельцу). det/mtp-from-weights отложен (у моделей списка MTP-веса есть). PLAN.md: план остатка, сверенный с целью (A1–A3 спекуляция, B1 точность, C1 PR) + отложенное с причинами. Память: feedback-plan-relevance.
- (25.10, 15:20) #96/#97 на d9b258be (2080 Ti): ui_test, ui_moved_update_test, fetch_test (0 падений, включая выбор по ширине) — проходят. Ждём Codex.
- (25.10, 15:50) DECISIONS 37: JEV-9B/27B и safeguard убраны; 27B safetensors-файнтюны (BF16 54 ГБ) не помещаются на 24 ГБ → их GGUF; замены MiMo-V2.6-Distill-Qwen-9B-GGUF (bartowski), Huihui-Qwen3.8-27B-abliterated-GGUF, Dark-Scarlett GGUF, Qwen3.8-27B-Uncensored GGUF, gpt-oss-20b heresy. Находка: MiMo-9B safetensors объявляет MTP без весов, план пишет builtin_mtp True — det/mtp-from-weights снова нужен (реальный случай) → 2080 Ti run_mimo (план master vs фикс + конвертация). Проверка всего списка планом на 24/16 ГБ → run_fit.out.
- (25.10, 16:30) План без весов брал MTP только из config.json (plan_facts) → MiMo-9B «builtin_mtp True». det/mtp-from-weights 641061f5: одно правило convert::index_lists_mtp для конвертера и плана (хаб и локальная папка). 2080 Ti: сборка serve + план MiMo master vs фикс → run_mimo2; конвертация MiMo фиксом → run_mimo; список 50 вариантов планом 24/16 ГБ → run_fit (медленно: плану нужны данные хаба).
- (25.10, 17:00) DFlash Qwen3.5-9B на вариантах (5070 Ti, #101): E — Ada↔Blackwell ≤1,1 %; Q4↔Q8 ≤1,7 %; IQ2 −2…−6 % (у MTP −6…−14: драфтер DFlash свой Q8, меняется только цель); HauhauCS (файнтюн) = база −2…+0,5 % (а WaveCut 27B −37 %) → файнтюны непредсказуемы, мерить каждый. R: k≤7 (≤8 строк проверки) Ada≈Blackwell ≤9 %; k≥9 форма разная (Ada 2,57/2,12/1,99 при k=9/11/15, Blackwell 2,27/2,26/2,34) → выбор thinking Ada k=15 ×3,86, Blackwell k=7 ×3,36; чужой выбор теряет 8–15 %. Ускорение растёт с тяжестью кванта (Q8 ×3,96, Q4 ×3,36, IQ2 ×2,67 thinking). ОБЩИЙ ВЫВОД с нагрузкой: цена раунда переносится между картами, пока проверка ≤8 строк (упор в память); шире (глубокий DFlash или C>1) — свойство поколения карт.
- (25.10, 17:40) Правило широкой проверки из фактов карты: не выводится (SM-правило stand_in_load противоречит Gemma 4070 Ti 60 SM ×1,42 vs 5070 Ti 70 SM ×1,03; отношение вычисления/пропускная немонотонно на 6 точках). → замер пары на поколении; непромеренное поколение — без широкой проверки. PLAN.md сверка 17:30.
- (25.10, 18:20) Проверка 50 вариантов планом (24/16 ГБ): ERR на master — cdiamond iMatrix и meta-models Muse (оба исправлены #100); 31B nvidia NVFP4 32,6 ГБ / RedHat 23,3 / AWQ 20,5+3,9 не помещаются на 24 ГБ → заменены HauhauCS 31B QAT GGUF, LilaRest NVFP4-turbo, ggml-org GGUF (проверены планом: 42–44 ток/с на 4090). DECISIONS 37 дополнен. На 16 ГБ не помещаются NVFP4/MoE ≥26B — ожидаемо.
- (25.10, 19:00) det/mtp-from-weights: MiMo-V2.6-Distill-Qwen-9B конвертируется (rc=0, «declared MTP layers are absent from the checkpoint»; на master такой случай падал — как JEV). План MiMo master vs фикс → run_mimo2 (сборка serve). Ландшафт HF: топ-2000 генеративных, ≤40B: покрытие архитектур 74,8 % скачиваний; пробелы — Qwen-VL семейство ~6 %, Gemma3 GGUF 1,1 M, LFM2, GLM4-MoE-Lite, Gemma2; проверка точности метаданных на 48 моделях планом → run_compat.
- (25.10, 20:00) Владелец: архитектуры НЕ расширяем; нужно понять, как выбирать и фильтровать модели, которые поддерживаем. Факты:
  (1) план без весов поддержку не проверяет: из 48 моделей выдал план и скорость для GPT-NeoX, LFM2, LFM2.5-GGUF, Spark-GGUF, Qwen3-VL, MLX, bnb-4bit;
  (2) конвертер: архитектуру safetensors отказывает до скачивания (config.json) — верно; архитектуру GGUF — нет (LFM2.5 упёрся в место, Spark качался); формат MLX / bnb и отсутствие чат-шаблона — только после скачивания, непонятными ошибками (U32, «expected source shape», «cannot read chat_template.jinja»);
  (3) GUI-поиск решает сам по подстрокам имён («family» по префиксу ≥5 букв, драфтер по «draft/assistant»), mlx по library, bnb/HQQ по quantization.
  Три места решают поддержку по-разному → предложение владельцу: одна функция «поддерживается ли и почему» по фактам до скачивания (архитектура из config/заголовка GGUF, поля конфига, формат весов из quantization_config + заголовков safetensors/GGUF, чат-шаблон), ею пользуются конвертер, план, GUI и лабораторный обход HF. Закрытые репо (401) — «нужен токен», не «не поддерживается».
  Дополнение: compressed-tensors w4a16 (nm-testing SmolLM) — отказ после скачивания («dense_encoded would not run as published»); GPTQ-Int4 Qwen2.5-7B и Gemma4-e4b файнтюн — конвертер начал качать (поддержка GPTQ не ясна — проверить по коду). Предложение «одна функция поддержки» отправлено владельцу, жду ответа.
- (25.10, 21:30) Владелец: «да, делай одну функцию поддержки». det/model-support 9d94cdf5: convert::model_support(SupportFacts) — факты до скачивания (карточка HF одним запросом: задача, library, gated, часть config, GGUF arch+шаблон, dtypes, файлы; полный config.json → выбор рецепта и проверки полей самого конвертера; заголовок выбранного GGUF). Правила: не генерация → MLX/bnb/упакованный U32 → архитектура (списки конвертера; драфтер вида, который никто не запускает) → закрытый без токена → нет чат-шаблона (у GGUF с base_model — шаблон придёт от базы, правило не применяется). Подключено: конвертер (отказ до скачивания), план без весов (отказ до плана). Тесты: test_model_support (все правила), fetch_test 12a (GGUF lfm2, MLX — отказ, кэш пуст). 2080 Ti: сборка + тесты + выборка 48 моделей → run_support. Дальше: GUI (заменить угадывание по именам), обход HF.
- (25.10, 22:10) det/model-support 8ccc07c3: GUI — поиск, /api/arch и карточка модели берут статус из model_support (поиск просит у HF ещё safetensors и gated); уровни: full / drafter / token (новый, 10 языков) / projector / unknown / none + причина; «probably supported» (префикс имени) и отдельный «format» удалены; docs/weight-conversion.md и docs/ui.md описывают правило. 2080 Ti: run_support (тесты PR-A + 48 моделей), затем run_support2 (ui_test, поиск по живому HF: qwen3/pythia/mlx/deepseek-math/lfm2/llama-3.1).
- (25.10, 23:00) det/model-support: на 2080 Ti план, тест правил и ninferno-ui падали (SEGFAULT) — моя ошибка: range-for по .items() временного json (value(...).items()) в search_support_facts → 8b3b9748; то же в det/mtp-from-weights index_lists_mtp → 70921776; fetch_test: мой case() с 3 аргументами → ac00fe33. Перезапуск run_support + run_support2.
- (25.10, 23:30) Владелец (уходит спать): проверять поддерживаемые модели; дать скачанную модель и токен HF/авторизацию; доделать недостающее (KV-кеш разных типов не для всех семейств); тесты из бенчмарков; автономно до зелёного. План A–G с критикой → PLAN.md «Остаток (25.10, ночь)». Факты: GUI токен не принимает (только HF_TOKEN env), свои модели — только .ninferno из --models-dir; KV: Gemma4/dense/Nemotron только bf16/fp8 (kv_fp8), int8/nvfp4/k8v4 — только Qwen; OAuth требует регистрации приложения — делаю токен + подхват hf auth login.
- (26.10, 00:30) C: det/hf-token e91ac392 (поверх det/model-support): токен HF из $HF_TOKEN, иначе из файла hf auth login ($HF_TOKEN_PATH/$HF_HOME/token/~/.cache/huggingface/token); GUI «Найти»: строка Hugging Face (аккаунт через whoami; ввод токена → проверка у хаба → сохранение туда же, 0600, только с этой машины); «Мои модели»: «Модель с диска» (GGUF/папка) → /api/disk (local_support_facts + model_support) → запуск; serve и план отказывают локальной неподдерживаемой модели до конвертации. A: MLX по блоку quantization (MiniCPM5-MLX проходила) → 23036d6b. 2080 Ti: q_hf (тесты, выборка 48, GUI вживую: /api/hf, /api/disk, поиск, запуск Qwen3-0.6B GGUF с диска и ответ 2+2).
- (26.10, 00:30) B: lab/integration 0f8c2ba2 (master + #96/#97 + #100 + mtp-from-weights + model-support + hf-token; конфликты #96↔model-support в acquire/fetch_server разрешены — понадобится при мерже). q_verify.sh: по каждому варианту списка — план продукта на карте → ninferno-serve (скачивание, конвертация, загрузка) → 3 вопроса с известным ответом + ток/с → удаление скачанного. 4070 Ti: 4B/9B/Gemma 12B (15), 5070 Ti: 27B/35B/26B/31B GGUF/gpt-oss/Nemotron NVFP4/Muse (19), 4090 после тестов точности: 16 (NVFP4 35B/26B/31B, Nemotron GGUF, Muse NVFP4...).
- (26.10, 01:00) D (KV других типов): исследовано; польза по плану продукта узкая — Gemma 4 31B на 24 ГБ: bf16 48K, fp8 120K, nvfp4 ≈214K (родной 262K); Gemma 12B на 12 ГБ и Qwen3-8B — пользы нет (моя ручная прикидка была неверна, исправлено). Цена — новые пути ядер gemma4_attention (дни, риск точности) → DECISIONS 38, вопрос владельцу. E: окружение EvalScope ставится на 4070 Ti и 5070 Ti (прогоны Q8/IQ2 9B и Gemma 12B после проверки моделей).
- (26.10, 01:40) #96 Codex на 7f3d14cc: P1 «закреплять только при нужных весах» — уже так (snapshot_holds_model + fetch_test 12c), ответил; P1 split — повтор, исправлено 651d75b5; P2 --vision без проектора в кэше → b11a2d71 (snapshot_holds_model(projector), fetch_head_probe --holds/--holds-vision, fetch_test 12c), влит в #97 b930e07b. Урок закреплён: C++-правки со спецсимволами — только через Edit; scratchpad/check_literals.py ловит разорванные строки и управляющие байты.
- (26.10, 03:00) B первые: 4070 Ti Qwen3.5-9B unsloth UD-Q5_K_XL и HauhauCS Q6_K — запуск, 3/3. 5070 Ti: unsloth/Qwen3.8-27B-GGUF UD-Q4_K_S — план выбрал, сервер отказал до скачивания своим правилом (вес + 10 % карты ≥1 ГиБ: 14,30+1,56 > 15,32) — два кода решают память (цель 1). det/plan-before-download 7936b341: проверка перед скачиванием = план без весов (plan_without_load, свободная память сейчас), serving_reserve/keeps_weights удалены, docs/serving.md. Влито в lab/integration; отказанные варианты перепроверить после текущего прохода.
- (26.10, 05:00) B идёт: 4070 Ti — 10/10 вариантов Qwen3.5-9B/4B (включая MiMo GGUF, AxionML NVFP4, cyankiwi AWQ) запущены продуктом, ответы 3/3; 5070 Ti — 27B файнтюны GGUF (Huihui IQ4_XS, Dark-Scarlett Q3_K_M, Uncensored IQ4_XS) 3/3; unsloth 27B — отказ старым правилом (исправлено det/plan-before-download, перепроверка). ток/с в отчёте — по коротким ответам (префилл+задержка), не замер скорости. 4090: тесты точности — MMLU-Pro готов, идут gsm8k/math500/aime.
- (26.10, 05:30) E: полный прогон 9B Q4_K_M (4090, ≈7 ч): MMLU-Pro 60,1 % (12032), GSM8K 94,7 % (1319), MATH-500 82,6 % (500), AIME25 43,3 % / AIME26 43,3 % (thinking, по 30). Дальше для отбора различающих вопросов: 5070 Ti — 9B Q8_0 и UD-IQ2_M, 4070 Ti — Gemma 12B Q4_K_M (q_disc.sh после проверки моделей). 4090: gpt-oss DFlash на Ada (F) → затем проверка 16 больших вариантов списка (B).
- (26.10, 06:00) E: конвейер отбора готов — evalq/extract_scores.py (на стенде: {датасет: {subset:index: acc}}) + evalq/discriminative.py (вопросы, где конфигурации расходятся; «все верно»/«все неверно» исключаются); на 9B Q4 совпадает с отчётами EvalScope (60,13/94,69/82,6/43,33 %).
- (26.10, 06:30) A+C проверены на 2080 Ti (e91ac392): model_support_test ok, fetch_test 0 падений, ui_test ok, ctest convert/fetch/plan/ui без падений; выборка 48: MiniCPM-MLX теперь отказ, 32 PLANNED / 16 REFUSED, все причины верные; GUI вживую: поиск помечает эмбеддинги/речь/MLX/bnb/базовые/закрытые/lfm2 верно; /api/hf без токена и с неверным (400, ничего не сохранено); /api/disk (GGUF full, NeoX — причина, нет пути — 404); запуск Qwen3-0.6B GGUF с диска через GUI → «2 + 2 = 4». PR #102 (функция поддержки, draft, Codex), PR #103 (токен HF + модель с диска, поверх #102, draft, Codex). Известный пробел: compressed-tensors w4a16 (SmolLM nm-testing) отказ только после скачивания.
- (26.10, 07:00) F: gpt-oss DFlash на 4090 (Ada) против 5070 Ti: R ≤1,6 %, E ≤2,6 %, выбор k тот же/соседний, потеря ≤3,4 % → правило переноса подтверждено и для MoE. Итоговое решение спекуляции → choice/SPECULATION.md v4 (таблица фактов + схема записи + что удаляется) — показать владельцу. 4090 дальше: проверка 16 больших вариантов (B).
- (26.10, 07:40) B нашла ошибку движка: unsloth/Qwen3.6-35B-A3B-MTP-GGUF UD-IQ3_S (выбор плана на 5070 Ti) сконвертирован, но старт упал «attn_input_proj workspace: unsupported parent qtype» — у MoE MTP-слой всегда шёл в слитое ядро (только BF16/NVFP4/FP8/Q8), а здесь веса MTP в ggml-блоках. det/mtp-ggml-projection 6bd30bb7: слитое ядро только для его форматов, иначе общий linear + разбиение (путь dense). Влито в lab/integration a4faae44. 5070 Ti: после прохода — перепроверка unsloth 27B (старое правило памяти) и 35B MTP-GGUF (этот фикс), затем тесты точности. Остальное по B: 4070 Ti 15/15 вариантов 3/3 (вкл. Gemma 12B GGUF/QAT/NVFP4); 5070 Ti 27B файнтюны и 35B GGUF 3/3.
- (26.10, 08:30) B 4070 Ti завершена: 14/14 запущенных вариантов 3/3 (Qwen3.5-9B ×5, 4B ×5, Gemma 12B GGUF/QAT/NVFP4/lmstudio), google/gemma-4-12B-it BF16 — no fit (эталон, ожидаемо). → stdata/verify_4070ti.txt. 4070 Ti: тесты точности Gemma 12B Q4 (E).
- (26.10, 09:30) Codex: #102 P1 (явный шаблон --resource/--chat-template; сводка tokenizer_config ≠ «нет шаблона» → читаем файл, в поиске «неизвестно»), P2 (смешанный репо с --quant → GGUF) → ec8f9695, e7d12269; --chat-template теперь доходит до конвертации (базовую модель можно запустить), docs/reference/serving.md. #103 P2×4 (запись токена через прокси запрещена; после смены токена кэши сброшены и поиск повторён; с диска запускается только «full»; --chat-template для локальной) → f62240bd (перебазирован на #102). #96: P1 повтор — ответил; P2 несколько конверсий при --refetch → 835cfc1a (+fetch_test); P2 рейтинг проектора — ответил (правило выбора, не закрепления); влито в #97 ca3764fc. 2080 Ti: q_hf (#102+#103) → затем q_p1fix (#96/#97).
- (26.10, 09:30) B: найдено cdiamond Qwen3.8-27B iMatrix NVFP4-MTP GGUF (2,5 M) — смешанный GGUF (nvfp4 ×374 + q5_k/q6_k/q8_0; in_proj_qkv в q5_k): рецепт NVFP4 не подходит, общий рецепт не сохраняет NVFP4-блоки GGUF (намеренно: правила упаковки/делителей) → модель не конвертируется. Вопрос владельцу (DECISIONS 39): научить общий рецепт сохранять NVFP4-блоки GGUF или отказывать до скачивания. det/mtp-from-weights проверен: план MiMo master builtin_mtp True → фикс False.
- (26.10, 10:30) #102/#103 после исправлений Codex (0c799269 / 0b7aa1af, 2080 Ti): ctest convert|fetch|plan|ui|choice 51/51, model_support ok, fetch_test 0, ui_test ok, GUI вживую ок (диск, поиск, запуск 2+2=4). Отписал в PR. det/mtp-from-weights (PR #104): MiMo план master True → фикс False (после пересборки; зависание было от старой сборки с висячей ссылкой).
- (26.10, 10:45) #96/#97 на ca3764fc (2080 Ti): ui_test, ui_moved_update_test, fetch_test 0 (вкл. --vision без проектора и --refetch при нескольких конверсиях). Ждём Codex.
- (26.10, 11:30) Codex, новые замечания → исправлено и запушено:
  - #104 P1: NextN-слои (DeepSeek/GLM), объявленные в config, план без весов терял (индекс не содержит mtp.*) → convert::sharded_checkpoint_mtp (индекс ИЛИ объявленный NextN, в т.ч. text_config) для хаба и локальной папки, тест в drafter_plan → 85d0cd3d.
  - #102 P2×3: --include *.gguf в смешанном репо — проверяется выбранный GGUF; для GGUF сбрасываются library/dtypes карточки; /api/quants отдаёт полный вердикт репозитория (arch_json), а не по одной архитектуре → 739f330c, fetch_test 12a «org/mixed-include».
  - #103 P2×4: барьер кэшей при смене токена (счётчик поколения под sizes_mutex_); токен из HF_TOKEN — сохранение 409, страница не предлагает форму; замена файла токена на Windows; fetch_text берёт hub_token() (файл тоже) → 1cd22298 (+тест токена в model_support_test).
  - #96 P2×2: закреплённый старт берёт точную метку модели из снапшота (ширина «8bit» не тянет больший файл старого коммита), проектор — рядом; --refetch удаляет конверсию по точной метке → 559e2fe1 (+fetch_test 12c), влито в #97 648657aa.
  2080 Ti: q_fix3 (сборка + тесты трёх веток + /api/hf вживую) параллельно ct2 (ctest mtp-ggml-projection идёт, 147/275 без падений). 5070 Ti: verify gpt-oss GGUF ×3 — 3/3; 4090: NVFP4 Qwen3.6-35B ×2 и Gemma-4-26B NVFP4 — 3/3.
- (26.10, 12:20) Все исправления Codex проверены на 2080 Ti: #104 85d0cd3d (drafter_plan ok, fetch 0, ui ok); #102+#103 1cd22298 (model_support ok вкл. токен, fetch_test 0 вкл. mixed-include, ui ok; /api/hf вживую: env → 409 без файла, два сохранения → второй, 0600; прокси → 403; удаление → нет файла); #96/#97 1bd3aafb (fetch_test 0 вкл. «по ширине, больший того же класса в листинге — модель здесь, ничего не скачано», ui_moved ok). Отписано, Codex позван на всех.
- (26.10, 12:20) Пробел «compressed-tensors w4a16 SmolLM отказ после скачивания» разобран: масштабы int4 в F32 (257951 из 262144 неточны в FP16) → честный отказ «не сохраняется как опубликовано». У RedHatAI w4a16 (Qwen3-8B, Qwen2.5-7B, Gemma-3-12B) масштабы BF16 — проходят. Точность масштабов зависит от значений → до скачивания не решается детерминированно; оставлено как есть (редкий тестовый репо).
- (26.10, 12:50) #100 (роли GGUF-файлов) смержен в master 60527776: Codex только P2 (отвечено фактами), ctest 275/275 sm_75 + fetch_test, работает в lab/integration на 3 стендах. #96 после этого конфликтовал (комментарий fetch_head_probe) → слит master, #97 4f9a13fa; перепроверка q_fix3c на 2080 Ti. #98: Codex не ответил на повторный запрос (только 👀) → позван снова.
- (26.10, 13:40) Codex, третий круг (6 замечаний, все по делу) → исправлено:
  - #104: индекс весов решает MTP и без объявления в config (план хаба и папки; индекс не прочитан — объявление) → 997cf01e, тест 839a3b3e.
  - #102: в смешанном репо safetensors судятся без сводки GGUF рядом (факты пересобраны из карточки без "gguf") → cab4a467 + fetch_test «mixed … judged by their config».
  - #103: драфтер не запускается отдельно (convert::serve_refusal: отказ или «драфтер работает рядом со своей моделью, --drafter») — serve, план хаба и папки; новый токен будит поток каталога (приватная лента); файл токена — отказ, если права 0600 не выставились; страница узнаёт токен окружения по имени переменной → 4ceaff9b, тест ee0f460e.
  - #97: «done» с неудалённой старой конверсией показывает причину → 8a784ffd.
  - #96: старт с весами здесь всегда закреплён за их коммитом (гонка «спросили → ветка сдвинулась → скачали молча»); точная метка — только если сдвинулась → 22d4acd5, #97 6916b853.
  2080 Ti q_fix4: mtp-from-weights и hf-token — тесты зелёные (fetch 0, model_support ok, ui ok); идёт moved-ref-gui и живые проверки (драфтер отдельно, MiMo).
- (26.10, 14:20) lab/int2 97ce72c2 = master + #103(#102) + #104 + #97(#96) + plan-before-download + mtp-ggml-projection. Конфликты (понадобятся при слиянии): plan_offline includes; docs/ui.md (диск + сдвинутая ветка); fetch_server.py (history + card); acquire.cpp — проверка поддержки на закреплённом коммите и точной метке (ask.revision = at, ask.quant = pinned_quant), preflight там же. 2080 Ti: q_int2 после q_fix4. Codex позван на #102/#103/#104 (тесты зелёные).
- (26.10, 15:00) 2080 Ti: ctest det/mtp-ggml-projection 275/275 (sm_75) — фикс MoE MTP в ggml-блоках без регрессий; далее ct plan-before-download. Codex: #98 👍 на 099f7066 (без замечаний) — к слиянию после полного ctest; #104 P2 (MTP в отдельном model_mtp.safetensors без объявления; нечитаемый локальный индекс) → 9e7bd7f0; #102 P2 (поиск/карточка смешанного репо — по safetensors, как берёт fetch) → 76c0cc7f + юнит-тест, #103 46fb2ece. lab/int2 3a111403 = master + #98 + #103(#102) + #104 + #97(#96) + plan-before-download + mtp-ggml-projection → q_ctint2 (полный ctest + fetch/ui/support) после текущих; по зелёному — слияние по порядку. Проверка списка: 5070 Ti Muse-Glimmer-30B IQ4_XS 3/3; 4090 gemma-4-31B qat UD-Q4_K_XL 3/3.
- (26.10, 15:30) Проверка списка на 5070 Ti завершена: 16/18 запусков 3/3 (27B файнтюны GGUF, 35B GGUF/HauhauCS, Gemma 26B/31B GGUF, gpt-oss ×6, Muse GGUF ×3); unsloth 27B и 35B MTP-GGUF — перепроверка с фиксами (идёт run_retest); nvidia …Nemotron-3.5-Lightning…-NVFP4-DSpark ОТКАЗАН — по карточке это драфтер DSpark (Qwen3DSparkModel, 0,97B, «not a standalone target model»), моя ошибка выбора по имени. Причина отказа была невнятной (в реестре dspark не было Qwen3DSparkModel) → det/model-support 65802e46: «a dspark drafter: no engine runs that kind» + юнит-тест; влито в #103 и lab/int2 f2cfb427. Список: заменён целевой nvidia/…-NVFP4 (0,95 M) — добавлен в очередь проверки 4090; MODELS.md, DECISIONS 37.
- (26.10, 15:50) q_fix4 зелёный целиком: moved-ref-gui 6916b853 (fetch 0, ui, ui_moved ok); вживую — z-lab/Qwen3.5-9B-DFlash как модель → «a speculative drafter, which runs beside its model … --drafter»; unsloth 9B Q4_K_M планируется; MiMo builtin_mtp master True → фикс False. Codex позван на #96/#97. Заметка: план 9B GGUF на 4090 — «no speculation profile for this model on this GPU» (пробел реестра; закрывает схема v4).
- (26.10, 16:30) Перепроверка в продукте (5070 Ti, lab/integration): unsloth/Qwen3.8-27B-GGUF UD-Q4_K_S (раньше отказ старым правилом памяти) — стартует, 3/3; unsloth/Qwen3.6-35B-A3B-MTP-GGUF UD-IQ3_S (раньше падал attn_input_proj) — стартует, 3/3, служит mtp:3, принято 3/3 за запрос. → PR #105 (det/mtp-ggml-projection, ctest 275/275 sm_75) и #106 (det/plan-before-download, ctest идёт), draft, Codex позван. Итог проверки списка 5070 Ti: 18/18 моделей работают (DSpark-драфтер исключён из списка). Заметка: спекуляция у 35B MTP-GGUF — «медиана реестра, вариант нигде не мерен» (пробел реестра, схема v4).
- (26.10, 17:30) #105 (MoE MTP в ggml-блоках) смержен в master c4eea877: Codex 👍, ctest 275/275 sm_75, в продукте 35B MTP-GGUF стартует с mtp:3. Codex, новый круг: #96 P1 (неиндексированные safetensors-шарды — закрепление только при всех частях) → 6b76206f + fetch_test; #97 P1 (обновление Vision-конверсии теряло Vision → LaunchRequest.vision → --vision) → 7c4fb249 + ui_test; #102 P2 (GGUF без шаблона в заголовке берёт chat_template.jinja репо) → d7c0e736; #104 P2 (индекс читается при любом числе файлов) → 9fa41c52; #106 P2×2 (селектор ширины пропускал проверку памяти; сдвинутая ветка с сохранённым артефактом проверялась на новом коммите) → проверка перенесена в before_download acquire: точная метка выбранного GGUF, коммит, который реально качается, и ничего, если веса здесь → 45610b85. lab/int2 3f129106 (все PR + исправления) → q_ctint2 после ct2.
- (26.10, 17:30) 4090: useful-quants Nemotron-3.5 W4A16 не стартует («explicit grouping has incompatible formats»). Причина проверена по байтам: «мёртвые» строки экспертов up_proj с масштабами ~1,6e-38 (BF16), неточные в FP16 → эксперт не сохраняется «как опубликовано», банк смешанный. det/expert-bank-kept 55702ced: банк сохраняется целиком или никак → штатная проверка точности называет причину. Правило для мёртвых групп (записывать 0) — DECISIONS 40, вопрос владельцу. q_after (2080 Ti): живая проверка #106 (27B --quant 4bit на 11 ГБ) и конверсия W4A16 этой веткой.
- (26.10, 18:40) 2080 Ti: ctest det/plan-before-download (7936b341) 275/275; lab/int2 3f129106 полный ctest идёт. Codex, новый круг:
  - #106 P1 (верно): официальный рецепт Qwen3.6-35B-A3B для BF16 квантует экспертов в Q4/Q5 — проверка по байтам файлов отказала бы → d2a0c17b: до скачивания проверяется только то, что сохраняет размер (GGUF, рецепты *_published/*_bf16), иначе — при старте, как до PR; P2: выбранный правилом квант тоже проверяется по свободной памяти сейчас.
  - #96 P2 (--vision и текстовый чекпойнт) → d85cf17e + fetch_test; #97 P2 (поток остановки старого сервера отсоединён) → d71bf44d (владеемый поток, join в ~App); #102 P2 (шаблон-сайдкар при --include) → include_selects; #104 P2 (нечитаемый индекс в describe_checkpoint) → try/catch.
  - Находка по ходу (пункт 1 цели): план без весов для Qwen/Qwen3.6-35B-A3B (3,9 M скачиваний, BF16) говорит «не влезает: 67 ГиБ», а настоящая причина иная — у MoE Qwen3.5 нет рецепта без квантования, конвертер отказывает собственной квантизации экспертов ПОСЛЕ скачивания 67 ГиБ (convert.cpp: bf16_recipe = "" для MoE; check_fidelity own_refused). Функция поддержки теперь отказывает до скачивания: Qwen3.5-архитектура + num_experts + нет метода квантования + все dtypes BF16/F16/F32 → «an unquantized Qwen3.5/3.6 MoE checkpoint… take NVFP4/GGUF/GPTQ release» (det/model-support, тест). Hub и локальная папка (dtypes из заголовков).
  lab/int3 70a8451b (всё вместе) → q_int3 (CPU-тесты) на 2080 Ti.
- (26.10, 19:30) Codex ещё круг (только P2): #97 (failed до окончания соседних подготовок) → 5c3265fd; #102 (tokenizer_config вне --include) → include_selects; #104 (локальный один файл весов — заголовок) → 01ddd9fb, convert::checkpoint_mtp — одно правило для конвертера и плана папки; #106 (GGUF без метки кванта) → e214cb1d. P1 нет. Решение: после зелёных полного ctest lab/int2 и CPU-тестов финальной интеграции — слияние (раунды Codex дают только всё более узкие P2). Стенды: 2080 Ti ctint2 155/275, int3 сборка; 5070 Ti 9B Q8 eval (MMLU-Pro 32 %); 4070 Ti Gemma 12B eval (MMLU-Pro 50 %); 4090 проверка списка (Gemma 31B LilaRest NVFP4-turbo 3/3, 9 ток/с).
- (26.10, 20:10) lab/int3 70a8451b CPU-тесты зелёные (drafter_plan, ui, model_support, fetch 0, ui_moved). Финальная интеграция lab/int4 13327c57 (+ последние P2: 01ddd9fb, 5c3265fd, e214cb1d, include tokenizer) → q_int4. Полный ctest lab/int2 идёт.
- (26.10, 20:40) Полный ctest lab/int2 3f129106 (master + #98 + #102/#103 + #104 + #96/#97 + #106 + #105): 277/277 sm_75, fetch_test 0, ui_moved ok, model_support ok. Ждём lab/int4 (последние P2) → слияние.
- (26.10, 21:00) #98 смержен (1dc9a9e1). Живая проверка #106: unsloth 27B --quant 4bit на 2080 Ti (11 ГБ) — «does not fit … (nothing was downloaded)», кэш 40 КБ. Проверка списка на 4090 завершена: 14/17 3/3 (→ stdata/verify_4090.txt); не стартуют cdiamond (DEC 39), W4A16 Nemotron (DEC 40), RadixArk Muse NVFP4 (ModelOpt MXFP8-слои) → функция поддержки отказывает до скачивания (det/model-support, тест; #103 c16c6d3a). Nemotron NVFP4 (замена DSpark) 3/3, 50 ток/с. Итого по списку: 4070 Ti 14/14, 5070 Ti 18/18, 4090 14/17 (3 — форматы, вопросы владельцу/отказ). Gemma 31B NVFP4-turbo служит mtp:10 по профилю реестра другого репо (gemma-4-31b-it, «*»), принято 2/10 — довод к SPECULATION v4 (ток/с проверки — не скорость).
- (26.10, 21:20) 4090 свободен → 9B UD-IQ2_M полный набор точности (q_iq2_4090, master build_89), чтобы не ждать за Q8 на 5070 Ti; на 5070 Ti после Q8 вместо IQ2 — Gemma 12B (вторая конфигурация Gemma).
- (26.10, 22:30) СЛИЯНИЕ: lab/int4 CPU-тесты зелёные → в master: #104 f9f1839e, #102 14393efc, #103→#107 04a25a31 (#103 был поставлен на ветку #102 и слился в неё; перенёс тем же содержимым через #107, diff проверен равным), #96 6edb3bc7, #97 0ac11c08 (база переключена на master), #106 f74f7764. Конфликты разрешены версиями из проверенной lab/int4 (acquire.cpp, fetch_server.py, docs/ui.md, model_source.cpp). Master f74f7764 → q_ctmaster (полный ctest + fetch/ui_moved/support) на 2080 Ti.
- (26.10, 22:30) det/expert-bank-kept проверена на W4A16 Nemotron (2080 Ti): вместо падения планировщика — штатный отказ точности по каждой матрице с подсказкой --round-tiny-scales (флаг разработчика уже есть, по умолчанию выключен) → PR #108 (draft, Codex). DECISIONS 40 сужен: давать ли --round-tiny-scales серверу/GUI.
- (26.10, 23:00) #108 Codex 👍 без замечаний → полный ctest det/expert-bank-kept (= master + #108) в очереди после ctest master на 2080 Ti; по зелёному — слияние. Точность: 4070 Ti Gemma 12B Q4 MMLU-Pro 63 %, 5070 Ti 9B Q8 54 %, 4090 9B IQ2 18 % (1,3 it/s — быстрее).
- (27.10, 00:30) Полный ctest master f74f7764 (после всех слияний): 276/277 sm_75; fetch_test 0, ui_moved ok. Упал мой юнит-тест model_support «Model Optimizer NVFP4 layers» — ошибка теста (унаследовал U32-dtypes MLX-случая, сработало правило упаковки), не продукта → det/test-modelopt-layers 303f0aa9 (PR #109), влит в #108; ctest #108 (f7b8840f) перезапущен.
- (27.10, 02:00) Владелец: «да» на DECISIONS 38 (KV nvfp4 Gemma 4), 39 (отказ до скачивания), 40 (--round-tiny-scales в serve/GUI), KLD для не-Qwen; спекуляция — попросил подробнее; предложил: непромеренный вариант → запрос на наш сервер, там LLM-анализ страницы HF находит драфтер (быстрый рецепт) + триггер замера в лаборатории.
- (27.10, 02:00) Память спекуляции (проверено кодом и планом, master, 9B Q4_K_M на 4070 Ti, 262K): считают байтовые функции семейства для каждой конфигурации (веса драфтера, его KV, кольцо, рабочие буферы проверки, графы по глубине), растёт с контекстом и числом запросов, при нехватке план откатывается: глубже→мельче→другой драфтер→без спекуляции. MTP: +168 МБ веса, +554 МБ KV, +34 МБ графы, +20 МБ ≈ 0,78 ГБ (на ступень лестницы меньше). НАХОДКА: DFlash в плане без весов до скачивания драфтера — «the DFlash drafter's shape is unknown: its KV and ring are estimated», учтено ~+80 МБ, а драфтер z-lab/Qwen3.5-9B-DFlash — 1,29 B параметров (2,6 ГБ BF16) + KV полного слоя ~0,5 ГБ при 262K → план занижает ~2 ГБ. Ещё: план для unsloth 9B GGUF — «no cost model for this card, target or drafter: depth 0» (спекуляция не планируется вовсе).
- (27.10, 03:00) Владелец: v4+сервер с LLM-поиском драфтера и очередью замеров — да; офлайн и блокировка по настройке (приватность); глубина для непромеренного — по данным пар (проверил сам: «лучшая в среднем по группе, k≤7, класс <4 бит отдельно» — потеря в среднем 1,4 %, худшая 13,8 % (Ada k=15); файнтюн E−37 % — не медленнее ×1,01; группы с худшим <×1,05 — без быстрого рецепта) → SPECULATION.md v5. Память: «не показывать проверки, решать самому».
- (27.10, 03:00) Точность 9B (temp 0): MMLU-Pro Q4 60,13 / Q8 60,88 / IQ2 56,32 %; GSM8K (639 общих) 94,7 / 93,6 / 90,5 %. Различающих: MMLU-Pro 2847 из 12032 (все верно 5694, все неверно 3491), GSM8K 60 из 639. Но Q4↔Q8 расходятся в 1520 вопросах при разнице 0,75 п. — похоже на шум (пакетная обработка при concurrency 8?) → порог шума: повтор IQ2 тем же билдом/картой (q_noise_4090 после IQ2-полного) — вопросы, что «прыгают» между одинаковыми прогонами, — шум, не квант.
- (27.10, 05:00) Исправление: у WaveCut 2,90 ток/раунд против 4,5–4,6 у базы — 0,535 от черновиков после первого (не 0,63, как я моделировал). С фактом без защиты 2 режима медленнее обычного: gpt-oss plain ×0,92, IQ2+MTP plain ×0,99 → защита «≥×1,05 при приёме файнтюна» нужна (SPECULATION v5 исправлен).
- (27.10, 05:00) S1: det/spec-pairs d50c060d → PR #110 (draft, Codex): spec/pairs.{h,cpp} — MeasuredPair (веса цели по sha256 файлов, spec-tune пары, биты, поколение), profile_for_pair (select() по замерам пары; на своём поколении все глубины и нагрузка, на чужом k≤7 и один запрос), pair_group/profile_for_group (движок+вид+класс бит; лучшая в среднем глубина ≤7; порог — наибольший в группе; plain, если при приёме файнтюна < ×1,05). Тесты на 9 наших замерах (fixtures/spec_pairs): ок на 2080 Ti вместе с ninferno_spec_test.
- (27.10, 06:00) #109 и #108 в master (e3c7059e, da0c464e): ctest det/expert-bank-kept (= master + #108 + #109) 277/277 sm_75, fetch 0, ui_moved ok, model_support ok → master зелёный.
- (27.10, 06:00) Правило проекта (AGENTS: константы — с двух карт поколения): глубина >7 и нагрузка на «своём» поколении — только если пара замерена на ДВУХ картах поколения и согласна; S1 пускает по одной карте → исправить в S2 (MeasuredPair по картам, выбор согласия).
- (27.10, 06:00) Кампания пар (evalq/q_pairs.sh: prepare --with-drafters → spec-tune 3600 с → замер + манифест → удаление скачанного) поставлена после прогонов точности: 4070 Ti 14 вариантов, 5070 Ti 18, 4090 14 (пройденные 3/3 из проверки списка, квант плана). На 5070 Ti дубль IQ2-полного выключен (артефакт переименован *.held).
- (27.10, 06:00) Слагаемые точности: в choice_constants есть только qwen3_5_text, qwen3_5_moe_text, dense_decoder. Старый замер Gemma (4070 Ti, 85932a51 до #98): PPL эталона Gemma 12B ≈376, Gemma 26B KLD fp8 0,445 / fast 0,69 — счёт Gemma тогда был неверен. 2080 Ti: q_kld_gemma (master, Gemma 12B Q4_K_M: ref / kv fp8 / fast / fast+fp8) — проверить разумность PPL.
- (27.10, 06:30) Владелец: отбирать вопросы, на которых квант даёт ЛОЖНЫЙ ответ (другие токены не всегда плохо, ложный ответ — всегда). По 9B MMLU-Pro: Q8 верно/Q4 неверно 805, обратное 715 → реальная потеря Q4 ~90 вопросов (0,75 п.), остальное — шум прогона; IQ2: 1336 / 787 → 549 (4,56 п.). GSM8K: Q4 «лучше» Q8 (9/16) — шум. discriminative.py переписан: эталон верно во всех прогонах И квант неверно во всех своих прогонах; «исправления» и нестабильные — шум, не берутся. Повтор (IQ2 noise на 4090) даст второй прогон; повтор эталона Q8 тоже нужен.
- (27.10, 06:45) Повторы для стабильности ложных ответов: 4090 — Q8 (после IQ2-повтора), 5070 Ti — Q4 (после Q8-полного); на другой карте, чем исходный прогон (ложный ответ должен держаться и на другой карте). Кампании пар — после повторов.
- (27.10, 07:30) #110 P1 Codex (пары старой версии движка спекуляции) → c3da1399: pair_current; такая пара — обычный декод с причиной, в группы не входит; тест ок (2080 Ti).
- (27.10, 08:30) #110 дополнен 368708ee: profile_for_target — глубина >7 и нагрузка только при двух согласных картах поколения; weights_identity/artifact_weights; тесты ок (2080 Ti).
- (27.10, 09:00) ПРОБЛЕМА: перплексия Gemma 4 12B Q4_K_M на master (2080 Ti) — wikitext 677 / pg19 427 / zhwiki 1177 (итог 252), при BOS (window_start_token 2). Qwen3-8B на тех же частях 8,7 / 9,8 / 6,2. Softcap в головe и финальная норма на месте (gemma4 output_logits → pass_.head). Значит KLD-слагаемые Gemma пока мерить бессмысленно (q_kld_gemma остановлен). Перекрёстная проверка q_gscore: токены HF vs движка, ninferno-gemma4 score (свой путь семейства) vs ninferno-perplexity (#98) на wikitext-00.
- (27.10, 10:00) Gemma 4 12B перплексия, перекрёстно (2080 Ti, wikitext-00, W 2048): токены HF 64 310 = движка 64 309 (токенизация совпадает); ninferno-gemma4 score (свой путь семейства) PPL 688, ninferno-perplexity (#98) PPL 611 → два независимых пути согласны, #98 не виноват. Правдоподобно ли ~650 для 12B-it? → независимый эталон llama.cpp llama-perplexity на том же GGUF (q_llamacpp).
- (27.10, 10:40) Эталон llama.cpp (b3fa699, llama-perplexity, тот же GGUF Q4_K_M, wikitext-00, c 2048): PPL 654,6 ± 20,5 — совпадает с нашими 611 (perplexity) / 688 (gemma4 score). Ошибки нет: у Gemma 4 12B-it на сыром тексте PPL ≈ 650 (инструктивная модель). Следствие: KLD-слагаемые Gemma на этом корпусе по величине несравнимы с Qwen (у той PPL ≈ 5–9), а выбор переводит KLD в цену одной λ → вопрос калибровки (DECISIONS 41). q_kld_gemma возобновлён.
- (27.10, 11:30) #110 P1 Codex (дубли одной карты считались двумя) → b2f389b7: различные карты по классу GPU в ключе замера; тест ок.
- (27.10, 12:30) Владелец: интересны ВСЕ вопросы, где когда-либо был ложный ответ, самые интересные — «то да, то нет» (на грани: квант сдвигает вероятность верного). discriminative.py переписан: p(верно) по прогонам каждой конфигурации; берём вопрос, если где-то был ложный и где-то верный ответ; «borderline» (прогоны одной конфигурации расходятся) — первыми, затем «false» (устойчиво неверно при устойчиво верном у другой); мера конфигурации — средняя p на отобранных, потеря — падение против эталона. С повторами (Q8 на 4090, Q4 на 5070 Ti, IQ2 noise) p оценивается точнее; дальше — N прогонов каждой конфигурации на сокращённом наборе (он маленький).
- (27.10, 14:00) S2 (det/spec-pairs-resolve, поверх #110): 018429c9 — пары в реестре (Registry::pairs/measured), pair_choice (замер пары по весам и носимым драфтерам → иначе группа), resolve_speculation сначала из пар, затем старые профили; собрано (serve) на 2080 Ti, тесты ок. + план без весов: list_quants хранит sha256 файлов, ModelRead.weights, PlannedTarget.weights/quant, resolve_planned_speculation — пары первыми. Проверка в продукте: тестовый реестр (поставляемый + 6 пар с настоящими sha256 файлов unsloth 9B MTP-GGUF) — план 9B Q4 на 4070 Ti / 5070 Ti с реестром и без (q_s2plan).
- (25.10 по часам стендов, 10:10) S2 разбор «почему mtp:4, а не DFlash»: (а) в поставляемом реестре qwen3.5-9b объявлен только MTP → план не подключал DFlash вообще, пары DFlash отфильтровывались; (б) spec-tune мерит по драфтеру за прогон → у одной карты две пары (ev MTP, edf DFlash), выбор брал первую. Исправлено 8081e7bb: пары одной карты сливаются (замеры объединены, нагрузка/fit_fallback — только если совпадают); в реестр qwen3.5-9b добавлен dflash z-lab/Qwen3.5-9B-DFlash@5fc3b3d4 (component dflash — как мерила лаборатория); вес драфтера, встроенного из своего репозитория, теперь считается в плане (раньше нет — недосчёт ~2,4 ГБ). d35b0ccd: форма DFlash-драфтера из его config.json до скачивания → KV/кольцо по функции семейства, не оценка. Тест: MTP+DFlash одной карты в любом порядке → dflash:7.
- (10:10) #110 Codex P1 (группа из одной карты = константа с одной карты, AGENTS: ≥2 карт) + P2 (вид драфтера, не только режим) → 5d291d0b: группа только при ≥2 различных картах, цель одной карты один раз, вид DrafterRecord совпадает; тесты. Влито в S2. Codex позван.
- (10:10) KLD Gemma 12B Q4_K_M (2080 Ti): kv fp8 mean 0,207 (median 0,011, top-1 87,3 %), fast 0,241 (top-1 86,0 %) — на порядки больше Qwen (fp8 0,0028) при PPL ≈650: подтверждает DECISIONS 41 (KLD семейств несравним, калибровать по ложным ответам).
- (10:10) q_s2b на 2080 Ti: сборка d35b0ccd → план 9B Q4 (4070 Ti/5070 Ti, реестр с парами и без) → полный ctest (ждёт блокировку kldg).
- (13:40) S2 в продукте (d35b0ccd, 2080 Ti): с парами 4070 Ti и 5070 Ti → dflash:7 «measured … (2 lab card(s))», расчёт памяти family_planner (форма драфтера известна). На 12 ГБ (4070 Ti) DFlash урезает авто-контекст 262K → 163K (MTP держит 262K), 5070 Ti — 262K → DECISIONS 42 (вопрос владельцу: скорость ×2,49 против ×1,89 ценой контекста; предложено А — как сейчас, показать альтернативу). Классы контекста «depth 0 (no estimate)» — старый round_cost_of (только отчёт, не служит) → S4 удалит.
- (13:40) Найдено ещё: без --quant план читал заголовок САМОГО МАЛОГО кванта и применял его пару ко всем строкам таблицы → 7134e716: свой субъект спекуляции у каждой строки (DryRunInputs::subject_of, веса по sha256 листинга) в плане сервера и в GUI (app.cpp, листинг несёт weights); выбор и fits-now — у выбранной строки. weights_identity → "" при неполных дайджестах (одно правило). Проверка q_s2c — 4070 Ti (CPU).
- (13:40) ctest d35b0ccd — 2080 Ti (идёт).
- (14:10) Кампания пар шла на master-реестре: у 9B/4B GGUF без MTP драфтеров нет → unsloth 9B UD-Q5_K_XL измерен «plain ×1,00, drafters []» (пусто). Перезапущена на всех трёх стендах: BR=det/spec-pairs-resolve (DFlash 9B в реестре) + REG=lab_registry_pairs.json (кандидаты DRAFTERS.md, не в поставке: z-lab/Qwen3.5-4B-DFlash@9a1996cc, incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b69 — карточки HF: Apache-2.0, base Qwen3.5-4B / Qwen3.6-35B-A3B). Пустые пары удалены для перемера; q_pairs.sh: BR/REG параметры. Порядок ожиданий прежний (точность первой); 4070 Ti — ещё после q_s2c.
- (14:30) #110 Codex на 5d291d0b: P1 нет, P2 «порог сторожа от первой карты» → 86bff73c/b6ceee12: у согласных карт — наибольший порог (порядок записей не влияет); тест (вторая карта с plain на 2 % быстрее; 3 % на k=15 меняли выбор → это уже «несогласие»); 2080 Ti ok. Влито в S2 (f281b23f). Codex позван.
- (14:50) q_s2c (7134e716, 4070 Ti CPU): таблица квантов 9B MTP-GGUF без --quant — у каждой строки свой выбор: Q4_K_M/Q8_0/UD-IQ2_M «measured» (свой замер), прочие ≥4 бит — группа dflash:7 «preliminary (3 measured pairs)», <4 бит — опубликованный mtp:5 (в группе <4 бит одна карта → нет группы, по правилу). 4070 Ti: выбран UD-Q5_K_XL 115K (наибольший влезающий Q8_0) — 2,4 ГБ BF16 DFlash сдвигают и квант; Q8_0 с DFlash не влезает → MTP head mtp:3 (92K); UD-Q6_K_XL+DFlash → 8,9K. Дописано в DECISIONS 42 (q8-компонент DFlash как третий путь — мерить в кампании).
- (15:20) Кампания пар идёт (4070 Ti): HauhauCS 9B Q6_K — DFlash ×2,55 (thinking dflash:11 ×3,69 на Ada), MiMo-V2.6-Distill 9B Q5_K_M — ×2,23 (thinking ×3,03). unsloth 9B UD-Q5_K_XL снова «без драфтера»: в кэше лежал артефакт, сконвертированный в 08:05 на master-реестре, и prepare --with-drafters взял его молча. Кампания: перед каждой подготовкой rm -rf $DIR/ninferno (скачанное остаётся), перезапущена на трёх стендах.
- (15:20) ДЕФЕКТ ПРОДУКТА (тот же корень): кэшированный GGUF с --with-drafters никогда не получал драфтер, добавленный обновлением реестра (drafters_incomplete смотрел только на «failed»). 3414dae2: манифест пишет registry_sha256; другой или отсутствующий → повторное подключение (новый компонент доконвертируется существующим путём companions); тест test_drafter_plan. Сборка теста — 4070 Ti (lab/s2-test, отдельный каталог).
- (15:20) #110: Codex «Didn't find any major issues» — P1 нет.
- (15:50) Пункт 3 для S2: GUI печатал основания «measured»/«preliminary» сырыми словами → d96e0772: подписи EN/RU («замерено для этих весов», «предварительно: по похожим замеренным моделям»), подсказка начинается с источника (пара / группа из N пар); test_page_plan 598 ok (локально, node).
- (16:10) Полный ctest S2 (d35b0ccd, sm_75): 277/278; упал только model_support «Model Optimizer NVFP4 layers» — ошибка теста, исправленная в master #109 (ветка отставала). → #110 слит в master (b8313863, squash; P1 нет, тесты зелёные). S2 обновлена от master (cc677076; add/add-конфликт pairs.* — версия ветки S2, расхождение только в комментарии) → **PR #111** (draft, Codex позван).
- (16:40) 3414dae2 доказан в продукте (2080 Ti, unsloth/Qwen3.5-9B-GGUF Q4_K_M --prepare-only --with-drafters): master → манифест без отпечатка, только mtp (unsupported); ветка (cc677076) → «companion drafter z-lab/Qwen3.5-9B-DFlash -> component dflash … converting again with its companion drafters», манифест registry b539cbf4, dflash embedded; третий запуск — готовый артефакт без переконвертации. Тесты 3414dae2 (4070 Ti): drafter_plan ok, spec_pairs ok, plan_dry_run 8240 ok.
- (17:10) Кампания пар (4070 Ti, DFlash 4B из лаб. реестра): lmstudio 4B Q6_K ×2,65 (thinking ×3,72), bartowski 4B Q5_K_S ×2,61 (×3,66). Ошибка скрипта: xargs ломается на «the model's weights» (кавычка) → AxionML/cyankiwi/RedHatAI «the model's weights» пропущены → q_pairs_v2.sh (обрезка sed); 5070 Ti/4090 перезапущены на v2 (ещё ждут точность), 4070 Ti — второй проход после первого (pairs_pass2.sh).
- (17:10) #111 Codex: 2×P2 → 992e336a: пары не спрашиваются для bundle-манифеста (своя декларация чекпойнта главнее реестра, и при конвертированном, и в плане); вес драфтера из своего репо — по ревизии (repo@revision). Тесты (4070 Ti, sm_89): spec, spec_pairs, plan_dry_run, drafter_plan — ok. Codex позван.
- (17:40) DECISIONS 39 (владелец «да»): det/gguf-mixed-nvfp4 c5f61bb5 — SupportFacts считает матрицы NVFP4 и прочих квантованных GGML-блоков по tensor infos заголовка (локально и с хаба); model_support отказывает Qwen3.5 GGUF со смесью (рецепты NVFP4 держат NVFP4 только рядом с неквантованными, GGUF-рецепты — блоки без NVFP4; как в synthetic_gguf auto-mixed-nvfp4); тест model_support. Проверка в продукте (cdiamond: план и старт без скачивания; K-quant 27B планируется) — 2080 Ti q_mixnv.
- (18:10) Кампания 4070 Ti: unsloth 4B MTP-GGUF Q5_K_M — DFlash ×2,57 (thinking ×3,50; MTP тоже принят, DFlash быстрее); HauhauCS Gemma 12B Q4_K_M — ассистент mtp:3 ×1,96 (thinking mtp:5 ×2,40). unsloth Gemma 12B UD-Q6_K_XL: 10,73 ГиБ весов + ассистент не влезают в 12 ГБ (факт карты) → пара без включённых замеров (pair_choice её отбрасывает — ложного plain нет); вариант добавлен в список 5070 Ti.
- (18:10) DFlash Gemma: конвертер качает z-lab gemma DFlash (реестр: dflash.engines ∋ gemma4 — ради ninferno-gemma4 Session), серверный Engine его не запускает → DECISIONS 43 (предложено: план драфтеров — только виды, что запускает Engine). Ветка с удалением из реестра отменена (ломала профили Session и проверку реестра). В лаб. реестре кампании gemma4 из dflash убран (не качать зря).
- (18:30) DECISIONS 39 доказан в продукте (2080 Ti, c5f61bb5): cdiamond 27B NVFP4-GGUF — план и старт отказывают за 6 с, прочитав 42 МБ заголовка («mixes NVFP4 matrices (374) with other quantized GGML blocks (132)»), кэш пуст; unsloth 27B UD-Q4_K_S планируется; model_support test ok → **PR #112** (draft, Codex позван).
- (19:00) DECISIONS 40 (владелец «да»): det/round-tiny-scales-serve a05a9f2a — ninferno-serve --round-tiny-scales (в конвертацию), ninferno-ui: LaunchRequest.round_tiny_scales → флаг; в окне журнала отказанного запуска/подготовки с «--round-tiny-scales keeps it» — кнопка «Запустить с округлёнными мелкими масштабами» + пояснение; тесты serve_options/ui, docs/serving.md; test_page_plan 590 ok (node). Проверка в продукте: 2080 Ti q_rts (Nemotron W4A16 prepare без/с флагом).
- (19:40) Кампания (4070 Ti, второй проход v2): AxionML 9B NVFP4 — DFlash ×2,21 (thinking ×2,94), cyankiwi 4B AWQ — ×2,22 (×3,23), lmstudio Gemma 12B Q6_K — ассистент mtp:5 ×2,43.
- (19:40) #112 Codex 2×P2 → 9cc76166: split GGUF — все части (fetch_gguf_head part / GgufHead::parts; локально -0000N-of-0000M), простые типы ggml::is_plain не считаются блоками; тесты (две части NVFP4+Q8_0 → отказ; F64 рядом с NVFP4 — не блок); model_support + gguf ok (4070 Ti). Codex позван.
- (20:30) DECISIONS 40 доказан (2080 Ti): Nemotron W4A16 prepare без флага — отказ с «--round-tiny-scales keeps it with those scales rounded»; с флагом — конвертирован, отчёт считает округлённые масштабы (1890/84/42/168/504…), provenance.round_tiny_scales=true; serve_options/ui тесты ok → **PR #113** (Codex позван).
- (20:30) Владелец по 42: «драфтер — ось единого выбора, рекомендуем по скорости, но с учётом запрошенного контекста и потоков, показываем, что выбранное не оптимально по скорости из-за контекста» → S3 так и проектировать. По 43: «разберись, зачем было нужно и где ошибка, сделай единый источник».
- (20:30) 43 разбор: Session (ninferno-gemma4, models/gemma4/session.cu) запускает DFlash/адаптивные/ngram; серверный Engine (Program, gemma4 program_family) — только ассистент (порт DFlash не доделан). Реестр kinds.engines (данные эпохи Session) и профили, настроенные на Session, расходятся с engine_support; заплатка served есть не везде. ДЕФЕКТ (план на master): unsloth Gemma 12B Q4_K_M на RTX 5070 Ti → plain «measured faster without speculation» (профиль «*» thinking dflash:7, без served); 4070 Ti → mtp:5. Исправление det/engine-support-truth b0eeae2c: Registry::supported = engine_runs_kind (engine_support — единственная правда), поле engines удалено; Gemma — без DFlash-драфтеров; незапускаемые выборы профилей без served → default профиля (с заметкой), 2 профиля nvidia 26B NVFP4 gb202 без единого запускаемого выбора удалены; test_spec держит драфтеры и выборы профилей к Engine; drafters_test: DFlash Gemma «unsupported», не скачивается. Проверка q_est (2080 Ti): ctest + планы Gemma 12B / nvidia 26B NVFP4 на 5070 Ti/4070 Ti/4090.
- (21:00) 4070 Ti: кампания пар закончена (RedHatAI Gemma 12B NVFP4 — ассистент mtp:5 ×2,44, thinking ×2,95). Свободную карту занял замер для DECISIONS 42: 9B Q4_K_M + DFlash-драфтер q8 против BF16 (q_q8dflash, лаб. реестр с quant q8).
- (21:00) #113 Codex: P1 (локальная модель-путь не получала флаг) + P2 (повтор в GUI терял настройки запуска) → коммит: cached_artifact(round_tiny_scales), /api/retry_rounded повторяет исходный LaunchSpec + флаг (with_tiny_scales_rounded, тест); LaunchRequest.round_tiny_scales удалён (не нужен). page 590 ok; сборка/тесты — 4070 Ti.
- (22:00) #113: P1/P2 исправлены (47bd12f5), ui/serve_options ok (4070 Ti), Codex позван.
- (22:00) S3 начат (det/choice-drafter-axis от S2): 73b119bf ядро — DrafterOption в ModelFacts (label, speedup, plain_from_streams), Candidate.drafter, Pins.drafter, speed_of умножает декод на ускорение (до границы нагрузки), выбор по ценности; без решающих констант — драфтер профиля; explain: «быстрее, но не держит контекст — xN» (по слову владельца); тест weighs_the_drafters (DFlash при запасе памяти, MTP когда DFlash не держит 32K, plain при 2 потоках, пин, закрытые ворота) — choice ok (4070 Ti). ecbc6c5f подключение — drafter_axis из субъекта (по режиму: замеры пары этого режима переизбираются; опубликованный — свой выбор), память кандидата планом с данным драфтером, кэш по драфтеру, драфтер в JSON выбора. Проверка q_s3a (4070 Ti CPU): 9B MTP GGUF Q4/Q8/таблица на 4070 Ti / 5070 Ti.
- (23:00) Единый источник (43) доказан: Gemma 12B на 5070 Ti → mtp:6 (было plain); Session-выборы Gemma — запись "session", опубликованное = Engine (served поднят); spec_test ok (102 пороги, 10,9 %), drafter_plan ok, drafters_test.py 11/11 (venv), plan_dry_run OK; подпись «замерено быстрее без» заменена на drafter_missing/engine_refuses → **PR #114** (Codex позван). Замечено: synthetic_dflash2_gguf.py падает и на master (с venv-питоном; под ctest пропускается) — разобрать.
- (23:00) S3 шаг 2 в продукте (4070 Ti, план без весов, реестр пар): Q4_K_M → dflash:7 (декод 265 против 80 без, 163K), Q8_0 → plain + «DFlash не держит 32K, хотя ×3,96» — но план всё ещё служит mtp:3 (откат на MTP-голову) → шаг 3 (сервер служит выбор) и варианты без замера (MTP-голова) в оси.
- (23:00) Владелец: глубина для рассуждений/без — по смыслу; GUI вариант + дельта; CLI 5:4 при возможности → SPECULATION.md S3 дополнение (режим глубины по данным; GUI дельта; --draft-tokens T:N с проверками).
- Вопрос в очереди (DECISIONS 44, записать): GGUF без --with-drafters по умолчанию (GUI так запускает) — Gemma GGUF никогда не получает ассистента (×2,4), Qwen GGUF без MTP — DFlash.
- (00:10) Владелец: адаптивная глубина «работает только на синтетике, в реале нет» → удалить. det/remove-adaptive-depth 3215ef37 (поверх #114): SpecConfig без k_min (K:K0 отвергается), декодер — каждый раунд своей глубины (без расписания и синхронизации на раунд), Session Gemma/dense без min_draft_tokens, ninferno-gemma4 без --draft-min и mtp:8:2/mtp:15:2 в spec-tune, ninferno-dense без --min-draft-tokens, меню/манифест/engine_support без adaptive, реестр — 28 адаптивных записей session удалены, тесты (spec, product_speculation, real gemma, draft_depth удалён). Полная сборка + ctest — 2080 Ti (q_noadapt). Память: feedback-no-adaptive-depth.
- (00:10) #113 P2 (повтор подготовки перемещённой модели не вёл обновление) → 6a93c46b; ui ok. q8 DFlash: квант компонента не применился (приём побитно как у BF16) — возможный дефект пути GGUF+компонент (у 27B DFlash2 q8 тоже) — разобрать.
- (00:40) Адаптивная глубина: заметки, чтобы не вернуть случайно — AGENTS.md (инварианты), spec/config.h (parse отвергает K:K0), декодер, docs/serving.md, DECISIONS 44, SPECULATION.md, память (f41ba516). Полная сборка + ctest det/remove-adaptive-depth — 2080 Ti.
- (00:40) S3 шаг 3: 2b50df38 — StartAxes.drafter; --spec auto: выбор «plain» выключает спекуляцию, иной режим → ServeOptions.speculation_mode → GenerationService резолвит всё (профиль, откаты, мельче, MTP-голова) через один субъект subject_of_mode; d489d9db — план без весов перестраивается под драфтер выбора. Проверка q_s3b (4070 Ti).
- (01:10) S3 в продукте (d489d9db, 4070 Ti CPU, реестр пар): 4070 Ti Q4_K_M → dflash:7 (265 ток/с против 80, 163K) и план служит dflash:7; Q8_0 → plain («DFlash не держит 32K, хотя ×3,96»), план теперь тоже plain (раньше mtp:3 откат); таблица → UD-Q5_K_XL + dflash:7 (preliminary); 5070 Ti (без решающих констант) — драфтер профиля dflash:7. Подпись plain «choice» — 81fd7321 (page 602 ok). Следствие: MTP-голова без замера пары для Q8_0 не предлагается (нет данных → не выбираем); кампания пар даст MTP-группу.
- (25.10, 15:20) Владелец: «распутывайся; карты поколений позже — придут, замеришь». Порядок распутывания: снос модели цены раунда → один порядок старта → одна память драфтера → GUI (скорости таблицы с драфтером, подпись оси) → один порядок ранжирования в choose().
- (15:20) det/no-round-cost-model 257b5f09 (поверх remove-adaptive): удалены spec/load_model.* и spec/context_class.* (round_cost_of, modeled_load, class_depths, классы контекста — только показывались), реестр без context_classes (версия 114), отчёт плана без modeled_load/model_note/classes (JSON, текст, подсказка GUI), ask без model_repo/artifact_id/source_identity, shape.kquant, GpuKey.mma_slow/facts_extrapolated, проверка имён маршрутов в route_table_test; −1950 строк. page 597 ok, registry_add ok. Полная сборка + ctest sm_89 (4070 Ti, q_norc) идёт.
- (15:20) S3-ветка: ad30eb18 — один порядок старта: product::fit_fallbacks (откат профиля → мельче → MTP-голова) и product::pool_rule (голова при 4/5, иначе plain) — сервер, CLI и план зовут одно. Найдено расхождение: план искал «мельче» от выбора профиля, сервер — от отката профиля (план теперь как сервер); CLI искал мельче от неподошедшего отката, читал профиль по артефакту и не брал драфтер выбора — теперь один субъект, subject_of_mode, ask.automatic=!spec_explicit; сервер пропускал выбор, если 5 осей закреплены, хотя --spec auto оставляет драфтер. Обёртки по пути артефакта удалены. 4ba08bb9 — product::attached_drafter_memory: одна функция веса+состояния драфтера для плана сервера и GUI (GUI брал размер из реестра для драфтера, которого не подключает, и не учитывал состояние DFlash — ~2 ГБ на 262K). Сборка/тесты/сравнение шагов плана старый↔новый/CLI-старт — q_s3c (4070 Ti). Ревью Sonnet идёт.
- (16:10) Ревью Sonnet порядка старта: дефектов нет; учтено — CLI снова пишет заметки профиля, тест глубин точный (2121bff5). GUI: скорость строки таблицы — с драфтером, который служит план строки (drafter_axis публичная), ax_drafter/choiceDrafter/why_default_drafter (EN/RU/UK), page 608 ok (2121bff5). d93540b3 — один компаратор в choose() (рулевой «наибольший» + ценность в одном порядке; выбор тот же).
- (16:10) q_s3c (4070 Ti, 4ba08bb9): юнит-тесты 5/5; план старый↔новый — шаги отката совпали во всех сценариях (9B Q8_0 262K: 4070 Ti plain, 5070 Ti head mtp:3; Q4_K_M ×4: dflash:7). CLI-старт на артефакте без профиля: plain + заметки. q_s3d (CLI на артефакте с DFlash, таблица) — ждёт.
- (16:10) Пропущено ранее: #111 Codex 09:18 — P1 (GUI размер драфтера из опубликованного выбора, не из пары) и P2 (у профиля пары нет floor_sources → откат головы по медиане реестра). Исправлено: ec14e90e (floors в обоих путях пары + тест через Registry::use_file), 0e005b4b (attached_drafter_memory перенесён в S2). S3 слит с S2 (360c9fd3). Тесты — q_s2p (2080 Ti).
- (16:10) norc: сборка падала дважды на пропущенных местах (apps/cli ask.source_identity, tests/test_plan_bench_fits model_repo) — поиск теперь по всему дереву; скрипты q_* теперь выходят при упавшей сборке (раньше брали блокировку и гоняли ctest на неполной сборке). Перезапущено (ec3dba59).
- (17:00) q_s3d (4070 Ti, d93540b3): юнит 5/5; таблица 9B MTP GGUF с драфтером строки (Q4_K_M dflash:7 260 ток/с вместо ~80; строки с MTP-откатом — скорость без спекуляции: у MTP нет замера); **CLI следует выбору**: артефакт 9B Q4_K_M + DFlash, лаб. пары — CLI при 32K и 262K запустил DFlash (замер 238,7 ток/с, оценка 219) с тем же KV, что план сервера (int8 / k8v4).
- (17:00) Двойной учёт в GUI: страница умножала оценку (уже с драфтером) на свой коэффициент → 7f4fb68b (estimate.drafter; свой коэффициент только если драфтер не учтён).
- (17:00) DECISIONS 45 (GGUF в GUI без --with-drafters — вопрос), 46 (модель цены удалена) записаны.
- (17:00) eef0cb9a — глубина по запросу по данным: spec::per_request_gain + product::per_request_pays (порог 3 % > разброс 1,8–2,3 %); сервер/план/GUI без --spec-select решают одинаково. Данные пар: только Ada 9B DFlash различается (15/7): при thinking по умолчанию потеря 2 % → fixed; при выключенном — 16 % → по запросу.
- (17:00) **Дефект S3 найден живой проверкой GUI** (4070 Ti, встроенный реестр): профиль «*» без скоростей → ускорение драфтера 1,0 → на Ada выбор брал plain («та же ценность»), старт --spec auto выключил бы спекуляцию профиля. 74187cd8: DrafterOption::measured; ось драфтера взвешивается, только если все выигрыши замерены, иначе стоит выбор профиля (объяснение «выигрыш не замерен на этой карте»); оценки считают только замеренный выигрыш. Тест choice. q_s3e (сборка, 8 юнит-тестов, GUI) идёт.
- (17:00) #111: P1/P2 Codex исправлены и проверены (2080 Ti: spec_pairs/plan_dry_run/ui/spec ok), Codex позван; #112 Codex позван повторно (P2 исправлены ещё в 9cc76166). norc: plan_dry_run тест проверял поле classes → 5540ae1f; полный ctest идёт (дальше — пересборка этого теста).
- (18:00) remove-adaptive: spec_test падал — слияние #114 вернуло адаптивные метки (mtp:5:2/mtp:8:2) в test_spec → b279bc59; на голове spec/registry_add/plan_dry_run зелёные (2080 Ti) → **PR #115** (draft, поверх #114, Codex позван); norc слит с ним (91cfee64).
- (18:00) q_s3e (4070 Ti, 74187cd8): 8/8 юнит-тестов (choice, plan_dry_run, product_speculation, spec, spec_pairs, ui, serve_options, plan_bench_fits). **Живая проверка GUI после фикса:** 9B MTP GGUF на 4070 Ti — выбран mtp:5+head (профиль), plain объяснён «выигрыш не замерен на этой карте», оценки без засчитанного драфтера (GUI применяет свой коэффициент) — дефект закрыт в продукте.
- (18:00) norc полный ctest (4070 Ti, ec3dba59) идёт: 145/277.
- (18:40) **#114 слит в master (8ef5c247)** (Codex без замечаний на 7e01c57b, тесты зелёные). #115 переведён на master (слит master, реестр v114 против master), MERGEABLE. norc: реестр v115 против #115. #111: P2 Codex (документировать basis measured/preliminary) → c7166497; слит master (конфликт подписей GUI — объединение) → 44ae8ac6, MERGEABLE. ОШИБКА процесса: первая попытка разрешения закоммитила и отправила маркеры конфликта (heredoc сломал regex, цепочка `| tail &&` не остановила) — исправлено amend + force-with-lease, урок в памяти. S3 слит (0d10d3ee, page 616 ok). Сборки+юнит: S2 (2080 Ti), S3 (4070 Ti) идут.
- (19:00) После слияния с master: S2 (44ae8ac6, 2080 Ti) и S3 (0d10d3ee, 4070 Ti) — по 8/8 юнит-тестов зелёные. Ждём Codex (#111, #112, #115) и полный ctest norc.
- (19:40) Codex второй круг (только P2/P3): #115 P3 — строка mtp:K:K0 в синтаксисе config.h удалена (c32a8167; P2 версия реестра — уже v114). #111 2×P2 → c6dcf9c3: профиль пары выбирается по тому, что запускает движок семейства (served заполнен; группа берёт только запускаемые глубины), устаревшие записи (pair_current) отсеиваются до объединения карт; 2 новых теста в spec_pairs. #112 P2 → 266101b4: части GGUF — только своей группы разбиения, а не все файлы --include; fetch_test случай + probe печатает parts. Проверка обоих — q_112 (2080 Ti). Перезапуск q_112 после правки имени цели (подмена работающего скрипта опасна — убит по PID через файл).
- (20:10) #111 (c6dcf9c3) 8/8 юнит, #112 (266101b4) fetch_test+model_support — зелёные (2080 Ti), ответы в PR, Codex позван. S3 слит с #111 (1a166373; конфликт теста — оба блока), юнит S3 — 4070 Ti.
- (20:20) norc полный ctest (ec3dba59, sm_89): 275/277 — провалы plan_dry_run (поле classes; исправлено 5540ae1f) и spec_test (адаптивные метки; исправлено b279bc59, слито); в продукте план без «modeled load» и строк классов, JSON без classes/modeled_load. Перепроверка 4 тестов на голове — q_norc2 (4070 Ti); затем PR.
- (21:00) norc на голове (9fd6ef81): plan_dry_run/route_table/spec/registry_add зелёные → **PR #116** (draft, поверх #115, Codex позван). S3 (1a166373) 8/8 юнит.
- (21:00) **Точность — калибровка (DECISIONS 47)**: извлечены прогоны (5070 Ti/4090), по 2 на квант. Средняя точность воспроизводится ±0,2 п. (MMLU-Pro Q8 60,88/60,67, Q4 60,13/59,97, IQ2 56,32/56,28), а ответы прогонов одной конфигурации расходятся в 10–11 % (шум пакетов при 8 потоках); Q4↔Q8 — 12,8 % (≈шум). Устойчиво ложные: Q4 нетто 0,07 п., IQ2 3,0 п. ΔKLD к Q8 (модель выбора): Q4 0,0172 → 0,73 п. (0,042 п./0,001), IQ2 0,114 → 4,48 п. (0,039) — **≈0,04 п. MMLU-Pro на 0,001 KLD**; при λ=25 один пункт ≈ ×1,87 времени. Gemma 12B Q6/Q8 не влезает на 12 ГБ → q_gemma_cal на 5070 Ti после кампании (Q8_0 + повтор Q4_K_M).
- (22:00) Codex: #115 и #116 — без замечаний. **#115 слит в master (62839ca8)** после перепроверки головы (spec/registry_add/plan_dry_run, 2080 Ti). #116 переведён на master (слит master, реестр v115 против master), MERGEABLE; тесты головы — q_116 (2080 Ti), затем слияние. #111: P2 TQ1_0/TQ2_0 как низкобитные → bf2d020f (тест ok); master влит, реестр v115 (46a20a68); юнит S2 — после q_116; Codex позван.
- (22:40) S3 слит с S2+master (1ba355c6, без конфликтов) → **PR #117** (draft, поверх #111, Codex позван); юнит S3 — 4070 Ti, #116/S2 — 2080 Ti.
- (23:00) S3 после слияния (1ba355c6): 8/8 юнит. #116 — полная сборка sm_75 с нуля на 2080 Ti, затем тесты; S2 юнит — следом.
- (23:40) **#112 слит в master (cc89a459)** (Codex без замечаний на 266101b4, fetch_test+model_support зелёные). #117 Codex: P1 (план CLI без весов показывал драфтер профиля, не выбора) + P2 (по запросу при plain-умолчании драфтер не грузился) → e1c0729a: product::plan_at_choice — одна перестройка плана под драфтер выбора для сервера и CLI; меню засевается включённым выбором другого флага (тест). #111 Codex 2×P2 → 6dcc6959: артефакт без манифеста (GGUF без --with-drafters со встроенным MTP) берёт пару (resolve_entry_pair), локальный GGUF — класс бит из имени файла. S3 слит (041475a8), юнит — 4070 Ti; #116 тесты и S2 юнит — 2080 Ti.
- (00:10) **#116 слит в master (90abaf59)** (5/5 тестов на голове, Codex +1). Модель цены раунда и классы контекста удалены из master.
- (00:40) master (#116) влит в S2 (3730db68: конфликты — сигнатура plan_with_speculation без gpu + drafter_state, реестр без context_classes + пары, GUI-подписи; реестр v116) и в S3 (вместе с S2; объявление per_request_of после SpeculationAsk). page 605/615 ok, registry_add ok. Юнит: S2 (2080 Ti), S3 (4070 Ti).
- (01:00) S2 (3730db68) и S3 (5898ecf2) после master: по 8/8 юнит; ответы в #111/#117, Codex позван.
- (01:40) Codex: #117 P2 (по запросу с plain-умолчанием ось драфтера брала «замеренный ×1» → выбор plain) → b8f91b49: опция другого флага не взвешивается (стоит выбор профиля). #111 P2 (пара сверялась с драфтером только по виду) → 8dfa67ef: MeasuredPair::drafters (вид → repo@revision, в JSON пары), манифест передаёт подключённые драфтеры, замер чужого драфтера — не замер; same_drafter; тесты. ВАЖНО для лаборатории: сборщик реестра пар должен записывать drafters (repo@commit) — в текущих записях их нет (сверка тогда «неизвестно» → принимается). S3 слит (328a8faf); юнит S2 (2080 Ti), S3 (4070 Ti). Кампания: 5070 Ti 12/19, 4090 6/14.
- (02:00) Следующий шаг к «данные из замеров в продукте» (S4): кампания хранит замер (TAG.json), манифест (драфтеры repo@revision, engine, путь артефакта с коммитом) и лог; для пары в реестре не хватает веса цели (sha256 файлов) — его надо снимать из провенанса артефакта ДО удаления (q_pairs_v2 удаляет новые скачивания). План: (1) в q_pairs_v2 после spec-tune писать TAG.weights (artifact_weights через tools/artifact/reader.py); (2) инструмент репозитория tools/spec/registry_add.py --pair TAG.json TAG.manifest.json TAG.weights → запись пары с drafters (repo@revision), generation по классу GPU, target_bits по метке; (3) пары кампании → лабораторный реестр → проверка плана в продукте → после #111 — PR с парами в поставляемый реестр.
- (02:20) S2 (8dfa67ef) и S3 (328a8faf): 8/8 юнит; ответы в #111/#117, Codex позван.
- (03:30) Владелец: «продолжай, пока всё не будет готово; потом повторное ревью — насколько всё прозрачное, элегантное и соответствует исходным требованиям».
- (03:30) **#111 слит в master (92159426)** (Codex без замечаний на 8dfa67ef, 8/8). #117 переведён на master; Codex 4×P2 → ba368607: ось драфтера предлагает запасной вариант опубликованного профиля другого вида (27B GB203: DFlash2 → MTP), subject_of_mode — fit_fallback как выбор, таблица нагрузки чужого вида → один запрос, выбор по тому, что запускает движок; явный --spec планируется как задан; fits_now — с драфтером выбора. 8/8 юнит (4070 Ti). Проверка 27B на GB203 (q_s3t) идёт.
- (03:30) **S4 инструмент**: det/pairs-registry d1ada2f4 — registry_add.py --pair (замер + манифест + цель repo@commit + метка + sha256 весов → пара с drafters repo@revision, поколение по таблице фактов, класс бит; замена пары той же цели/поколения/карты; тесты 17/17). Лаб. помощник campaign_pairs.py: sha256 из дерева хаба по коммиту и имени артефакта → 38 пар кампании (4070 Ti 14, 4090 9, 5070 Ti 15) в campaign_registry.json. Проверка в продукте (находит ли план пару по весам) — q_pairs_check (4070 Ti).
- (04:00) #117: Codex без замечаний на ba368607 (unresolved=false), юнит 8/8 → полный ctest sm_75 перед слиянием (q_s3full, 2080 Ti). 27B GB203 (без --with-drafters): mtp:4+head по профилю на 32K/128K, 262K не влезает — разумно. Проверка пар кампании в плане (q_pairs_check, 4070 Ti) идёт.
- (04:30) Проверка пар (частично): HauhauCS Gemma 12B / Qwen 9B на 4070 Ti — measured по весам; Muse 0bserverx — пара без замера драфтера (не найдена правильно). Найдено: 4 тюна кампании не замерили ни одного драфтера (не влезли/бюджет) → registry_add --pair теперь отказывает (c?); лаб. реестр пересобран: 34 пары. Пары ad102 (4090) проверялись на 4070 Ti — перепроверить на 4090.
- (05:30) Проверка пар в продукте (38, лаб. реестр): найдены по весам почти все; разобраны расхождения: (1) Qwen3.5-4B — в поставляемом реестре нет объявления DFlash 4B → добавлено (z-lab@9a1996cc); (2) RedHatAI 35B и ggml-org Nemotron — ошибка МОЕГО скрипта ключа (MTP-часть чекпойнта не взята; mtp-*.gguf взят как вес) → исправлено; **все 34 ключа сверены с листингом продукта (/api/quants): 34/34**; (3) AxionML NVFP4 на 4070 Ti — plain верно (с DFlash 11,1 ГиБ, 32K не влезает; объяснение «×2,94, но не держит контекст» — по DECISIONS 42); (4) подпись группового plain «measured faster» → «preliminary» (e8ac5da1). det/pairs-registry 950652ab: 34 пары + DFlash 4B в поставляемом реестре (v117, без локальных путей, 745 КБ). Проверка с поставляемым реестром (без --drafter-registry) + юнит — q_ship_check (4070 Ti). #117 полный ctest — 2080 Ti.
- (06:00) Поставляемый реестр (950652ab) в продукте: 34/34 пар measured по весам на картах своего класса, все планы влезают; юнит 4/4 → **PR #118** (draft, Codex позван).
- (06:40) **PR #118** (пары в поставляемом реестре + инструмент) открыт. S3-дополнение владельца п.2: det/thinking-variants 541e9151 (поверх #117) — отчёт спекуляции несёт выбор профиля для запросов с рассуждениями и без (+ замеренный выигрыш, + «по запросу»), текстовый план и GUI показывают оба и дельту («с рассуждениями dflash:15 ×3,86, без dflash:7 ×2,22 (−1,64)»); page 618 ok. П.3 (CLI 5:4) — не делается: CLI — один запрос с одним флагом, уже берёт его выбор; сервер решает по замерам. Проверка — q_variants (4070 Ti). #117 полный ctest — 2080 Ti (сборка с нуля).
- (07:30) det/thinking-variants 541e9151: 8/8 юнит; в продукте 9B Q4 на 4070 Ti (2 карты Ada) — «thinking dflash:7 ×3,32, without dflash:7 ×2,22 (one for every request)» (одна глубина — строка в GUI не показывается); Q8_0 → plain (выбор). Юнит #118 на голове e8ac5da1 — идёт.
- (08:00) **#118 слит в master** (Codex без замечаний на e8ac5da1, 8/8 юнит, 34/34 пар в продукте): замеры кампании теперь в поставляемом реестре.
- (08:40) #117 полный ctest (2080 Ti) падал на линковке: **диск стенда заполнен на 100 %** (не код: «final link failed: No space left on device»). Удалены 28 моих старых каталогов сборки (~48 ГБ каждый; оставлены master и #117) → свободно 692 ГБ. Другие стенды: 49–66 %. q_s3full перезапущен на голове 77646951 (master + реестр v117). **PR #119** (thinking-variants, поверх #117, draft, Codex позван). Кампания: 5070 Ti и 4090 ещё идут; q_gemma_cal ждёт.
- (10:30) **Повторное ревью** (3 Sonnet-агента: ядро выбора, порядок старта спекуляции, GUI/λ; каждую находку проверил сам) → choice/REVIEW-REPEAT.md. Главное: (1) второй читатель профиля спекуляции в GUI (app.cpp speculation_json: своя цепочка классов, без пар S2, своя память драфтера, specGain на странице); (2) квант выбирается двумя кодами (choose мульти-квант — только тесты; продукт — plan_quants), тесты проверяли путь, которым продукт не ходит; (3) неверная причина драфтера в GUI; (4) незамеренный выигрыш в оценке (split); (5) λ не виден; (6) сервер принимал plain-выбор профиля за спекуляцию (CapacityError → fit_fallbacks).
- (10:30) Исправления стопкой поверх #117: R1 det/review-r1 (#120: п.3,4,6) → R2 (GUI из продукта, −185 строк) → R3 (choose при одном кванте, один предикат порога, защита «биты известны» в plan_quants, тесты кванта — на путь продукта) → R4 (λ виден: обмен против крупнейшего в строке правила/GUI, строка λ у выбора, EN/RU/UK) → R5 (мёртвые тексты, docs/serving.md). #119 (Codex P2: варианты только при подаче выбора профиля, из поданного меню) → 84e566e0, база #119 = R1. Юнит R1→R2→R3 — 4070 Ti цепочкой; #117 ctest — 2080 Ti (147/277).
- (11:30) R1 юнит (4070 Ti): 7/8 — plan_dry_run падал на проверке порядка старта: после слияния #117 с master (реестр v117, 34 пары) синтетический 27B на Ada получает выбор группы пар (preliminary → plain, так по v5), а тест требовал опубликованный DFlash2. Решение: test_plan_dry_run работает на реестре без пар (механика плана; пары — test_spec_pairs) → b324c843 на #117, протянуто по R1–R5 и #119. Полный ctest #117 (77646951) упадёт на том же — перепроверю тест на b324c843.
- (11:30) R5+: одна глубина профиля для явного --spec (product::profile_depth; сервер брал served-выбор, план — сырой профиль; profile_speculation/speculation_of удалены). Ревью Sonnet стопки R1–R5: ошибка компиляции test_choice (Pins::quant) — исправлена; защита «биты известны» читала биты метки, выбор — bits_per_weight → читает bits_per_weight, тест неизвестных бит переписан (params=0); блок ускорения в GUI скрывался для репо вне каталога → показывается и при подаче спекуляции планом; -Wswitch, пример в docs/reference/serving.md.
- (12:10) PR #121 (R2), #122 (R3), #123 (R4), #124 (R5) открыты стопкой (draft, Codex позван); #119 Codex P1 (строка by_thinking в схеме docs/reference/serving.md) исправлен; #120 Codex позван повторно. Кампания 5070 Ti закончена (PAIRSDONE) → q_gemma_cal идёт (Q8_0); 4090 кампания продолжается. Новые пары (после v117) — второй пакет в реестр после 4090.
- (13:30) #117 полный ctest sm_75 (77646951): 276/277, единственный провал plan_dry_run — тест, исправленный b324c843; на b324c843 8/8 юнит (2080 Ti) → **#117 слит в master (871769b0)**. #120 переведён на master (master влит, конфликты squash — сторона R1, дерево не изменилось), стопка R2–R5 и #119 обновлены. R5 не собирался (profile_depth объявлен до SpeculationSubject) → e4d2efd3; юнит R5 + сравнение планов R1↔R5 (24 случая) — 4070 Ti. Сборка #117 на 2080 Ti удалена.
- (14:30) R5 юнит 8/8 (e4d2efd3); **сравнение планов R1↔R5 (24 случая, 4 карты): решения одинаковы везде**, отличаются только тексты (причина «(the one quantization)» убрана; строка правила с обменом: «Q8_0: 207 % more time … 0.0089 less KLD» — Q8_0 на 12 ГБ теряет DFlash). Живая проверка GUI R5 (4070 Ti): строка λ, обмен в правиле — работают; найдено «each request has its own .» при авто → b59fef03 (planned_context, тест страницы).
- (14:30) **Пробел данных найден живой проверкой**: unsloth/Qwen3.5-9B-MTP-GGUF (флагман) нет среди 34 поставляемых пар (прежние «measured» были из лаб. test_registry_pairs, S2); на Ada MTP 9B не мерян вовсе (тюн 4070 Ti — только DFlash). GUI (GGUF без --with-drafters, DEC 45) → группа qwen3_5+mtp → plain по правилу v5 (худший выигрыш при приёме файнтюна < ×1,05); до #118 было mtp:5 по опубликованному «*». Решение по данным: замерить. Пары 9B MTP GGUF Q4_K_M / UD-Q5_K_XL / Q8_0 поставлены на 4070 Ti и 4090 (кампания 4090 закончена, PAIRSDONE), затем второй пакет пар в реестр.
- (15:30) Codex по стопке: #119 2×P2 (варианты — из поданного меню шага profile; выбранный профилем plain = замеренный ×1) → 8e4df6d1; #120 P2 (unmeasured и ось drafter в схеме CHOICE) → dca2afca; #121 P2 (блок ускорения не перестраивался, когда план приходит позже) → 2c356f90; #122 — схема CHOICE без оси кванта/quant_by_worth; #123 P2 (строка λ только при by_worth) + P1 (схема time_ratio/kld_delta/accuracy_weight). Codex позван повторно на всех. Юнит R5 и #119 на голове — 2080 Ti (sm_75); 4070 Ti/4090 — замеры пар 9B MTP GGUF (без сборок рядом).
- (17:30) R5 и #119 юнит на финальных головах (sm_75) — R5 8/8 (e261431f); #119 идёт. Пары 9B MTP GGUF замерены: 4070 Ti и 4090 × Q4_K_M/UD-Q5_K_XL/Q8_0, MTP (mtp:2–5) и DFlash; DFlash ×2,2–2,6, thinking ×3,1–3,8. **Второй пакет пар (det/pairs-registry-2, v118): 47 пар (34 прежних без изменений + 13)**; в продукте с новым реестром: 9B MTP GGUF без драфтеров (путь GUI) → mtp:5 measured на 4070 Ti/4090/5070 Ti, с драфтерами — DFlash measured.
- (17:30) **Дефект найден той же проверкой**: пара, замеренная с MTP и DFlash, предлагала оси драфтера только DFlash (resolved.drafters — только драфтеры выбранных конфигураций; меню MTP «драфтер не подключён») → Q8_0 на 12 ГБ при 32K: plain вместо замеренного MTP. Исправлено (spec::measured_only: подключаются все замеренные драфтеры, без шумовых заметок) + тест spec_pairs → d2580fc5. Проверка (юнит, план, реальный старт CLI Q8_0 32K auto/none) — q_p2, 4070 Ti.
- (18:40) Проверка пакета пар (q_p2, 4070 Ti, 9846aa06): юнит 8/8; план 9B MTP GGUF без драфтеров → mtp:5 measured (Q8_0/UD-Q5_K_XL/Q4_K_M); **реальный старт CLI Q8_0 32K: --spec auto 90,4 ток/с против --spec none 54,1 (×1,67)**. Найдено расхождение план↔старт: план по хабу считал DFlash 9B в BF16 (2,41 ГиБ) → «не влезает» → head mtp:3, а старт на артефакте запустил DFlash (рецепт Qwen хранит проекции компаньона в Q8: 25 объектов q8_g32_fp16, 1,28 ГиБ) → convert::companion_drafter_bytes (одно правило рядом с конвертером; DFlash2 — 10,1 % BF16 по заголовку 27B), attached_drafter_memory по model_type цели (d76f2501). Проверка план по хабу ↔ план на артефакте (32K/64K/128K) — q_p3.
- (20:00) **#120 (R1) в master (1a498e11)**, **#119 (варианты с рассуждениями) в master (7ff02b46)** — Codex без замечаний, юнит 8/8. Стопка R2–R5 обновлена на master (дерево R2 не изменилось; конфликт R5 speculation.h — оба объявления). Codex #121 P2 (кэшированный план не рисовал блок ускорения) → d8f8d0b3; #123 P2 (λ при всех закреплённых осях) → строка только если ценность сравнивала альтернативу. #122/#124 — без замечаний. Юнит R5 (ee4004f6) — 2080 Ti.
- (20:00) Пары, план↔артефакт (q_p3): после companion_drafter_bytes **план по хабу = план на артефакте** на 32K/64K/128K (DFlash 32K fits rung 3; 64K head; 128K plain). Найдено: при 64K выбор взял замеренный mtp:5, а план показывал head mtp:3 (plan_at_choice сравнивал только режим) → 6bb90bd9 (перестройка, если конфигурация не та). Проверка + реальный старт CLI 64K auto/none — q_p4 (4070 Ti).
- (21:00) q_p4 (4070 Ti, 6bb90bd9): юнит 8/8; **план по хабу = план на артефакте** (32K dflash:7 rung 3; 64K **mtp:5** — после фикса, было head mtp:3; 128K plain); реальный старт CLI Q8_0 64K: mtp:5 66,6 ток/с против 54,0 без спекуляции (проза: замер mtp:5 ×1,02 на прозе, ×2,14 thinking — оценка взвешивает умолчание thinking). **PR #125** (пары v118 + 3 фикса план↔старт), Codex позван. R5 (ee4004f6, вся стопка на master с #119/#120) — юнит 8/8 (2080 Ti). Ждём Codex #121/#123/#125.
- (22:20) Codex: #121 2×P2 (перерисовка по ряду плана после тела; режим кроме Auto держит переключатель) → fa4767d5; #123 P2 (λ только по осям с точностью: kv/activations/state/ring) → R4; #125 P2 (объявленный quant драфтера: q8 у DFlash2 27B квантует весь драфтер) → 0ff6b1f1. Codex позван снова. Юнит: #125 (4070 Ti), R5 (2080 Ti).
- (23:30) **#125 в master (8d2950f9)**: реестр v118 (47 пар, 9B MTP GGUF на Ada) + 3 фикса план↔старт + объявленный quant драфтера; Codex без замечаний, юнит 8/8. Стопка R2–R5 обновлена на master (без конфликтов), page 619 ok; юнит R5 (4ea16d05) — 2080 Ti. DECISIONS 45 дополнен замером MTP против DFlash на 9B (DFlash +24–34 % смесь, ~+45 % thinking) — вопрос владельцу. Ждём Codex #121/#123.
- (00:30) Codex #121 P2: план не читал base_model карточки (дообученная Qwen3.6-27B без записи → nearest Qwen3.8-27B: DFlash2 вместо MTP 3.6; конвертер берёт по README) — пробел планировщика, был и до R2 → det/plan-lineage d557e53d (convert::hub_base_model из cardData, readme_base_models для локального, PlanSourceFacts.shape.base_model; тест). Юнит — 4070 Ti. R5 на master+#125: юнит 8/8. Gemma cal: Q8_0 полный набор идёт (3+ ч).
- (01:30) det/plan-lineage: тест сначала падал (PlannedTarget.gguf по умолчанию true → ветка GGUF без манифеста, она и у старта по форме — согласовано); тест на чекпойнт → 8/8. В продукте: huihui-ai Qwen3.6-27B-abliterated и AEON-7 Qwen3.6-27B — master: группа qwen3_5+mtp (драфтеры 3.8), ветка: qwen3_5+dflash (DFlash 3.6, как подключит конвертер); обе — plain (непромеренные файнтюны, v5). **PR #126** (draft, Codex позван).
- (02:10) **#121 (R2, GUI-спекуляция из продукта) в master (c9c1b8ef)**, **#122 (R3, choose при одном кванте) в master (f73e200b)** — Codex без замечаний, юнит стопки 8/8. #123 (R4, λ) → база master, Codex позван снова (молчал после последней правки); #124 (R5) — за ним. #126 (lineage) — Codex идёт. Gemma cal Q8_0 — идёт.
- (02:40) Codex #123 P2: подсказка λ ссылалась на DECISIONS 47 вне репо → калибровка записана в docs/maintainer/configuration-choice.md (прогоны, воспроизводимость, две точки, предел), подсказка — туда (R4). R5: конфликт подсказки → разрешён в пользу R4; при этом вернулись recWhy — снова удалены (3523b81f), page 619 ok. Codex #123 позван снова.
- (03:00) **#123 (R4, λ виден + калибровка в docs) в master (4ff3692a)** — Codex без замечаний. #124 (R5) → база master (дерево не изменилось), Codex позван. #126 — ждёт Codex. Стопка повторного ревью: R1–R4 в master.
- (03:30) **#124 (R5) в master** — Codex без замечаний, юнит 8/8 на финальной голове (1f10461f). **Стопка повторного ревью R1–R5 вся в master.** Остаётся #126 (lineage, Codex идёт); живая проверка GUI на master; Gemma cal.
- (03:50) **Живая проверка GUI на master (252aa213, 4070 Ti, 9B MTP GGUF, 32K)**: mtp:5 measured (было plain preliminary), варианты thinking/без (mtp:5 ×2,44 / mtp:3 ×2,00, по запросу), декод ≈166 ток/с (было ≈68), λ, обмен в правиле, ось драфтера. Сверка обновлена.
- (04:10) Перемер после слияния (треб. 5/7): смесь задач (run_mix, 8 задач, без thinking) на master с --spec auto против none, 9B MTP GGUF Q4_K_M / UD-Q5_K_XL (MTP, путь GUI) и Q4_K_M + DFlash; план рядом (выбор без thinking и его замеренный выигрыш) — q_mixspec на 4070 Ti и 4090.
- (04:50) Смесь задач (8 задач, без thinking), master, **4070 Ti**: MTP Q4_K_M план mtp:3 ×1,92 → смесь ×1,62 (общее время, с префиллом); UD-Q5_K_XL ×2,00 → ×1,71; DFlash Q4_K_M ×2,22 → ×1,84 — стабильно ≈0,84 плана (префилл 9K токенов не ускоряется). 4090: прогон на старой сборке master (без v118: plain preliminary) — недействителен; пересборка master и повтор — q_mix4090.
- (05:40) **Поиск драфтера для неизвестной модели (просьба владельца), эмуляция одним вызовом Haiku на модель.** Промпт v1→v4 (choice/drafter_lookup_prompt.md), отчёт choice/DRAFTER-LOOKUP.md. v1: 8/10 (вендорная цепочка, сторонние издатели, неподдерживаемые kinds); v2: 8/10 — поиск HF без author= не возвращает z-lab (проверено curl); v3: 10/10 (поиски author=z-lab и author=<вендор>); v4: 6/6 новых вне реестра + правило веса MTP в индексе (mlx заявляет MTP в config без весов). 3–6 запросов, 14–35 с. Вывод: шаги детерминированы — их можно сделать кодом, LLM только для неоднозначного. Вопрос владельцу: Gemma 4 ассистент vs z-lab DFlash (DFlash не замерен).
- (00:00) **4090 смесь (master 252aa213, 8 задач, без thinking):** MTP Q4_K_M план ×1,92 → смесь ×1,84 (269,5/146,6); UD-Q5_K_XL ×2,00 → ×1,75; DFlash Q4_K_M ×2,22 → ×1,84. **Дефект план↔старт найден:** q4km (без --with-drafters) и q4km_dfl дали время по задачам до мс одинаковое — старт без флага служил dflash:7 (лог: «registry pair … default dflash:7»), план говорил mtp:5. Причина: кэшированная конвертация Q4_K_M (и Q8_0) несёт компонент dflash (`.conversion.json` components text,mtp,dflash) от прежнего --with-drafters; старт её переиспользует, план по хабу и GUI её не видят. Значит 4090 MTP Q4_K_M ×1,84 — на самом деле DFlash; MTP-точка 4090 — только UD-Q5_K_XL (×1,75, 0,87 плана). → det/plan-cached-drafters (ffe6d29f, 7085286a): convert::cached_conversion_has_drafters; план без весов, таблица квантов и GUI /api/plan планируют такой квант с драфтерами и считают их память; юнит-тест в test_ui. Сборка+юнит 4090 идёт, затем q_pcd (план master↔ветка + старт). #126: Codex висел 4 ч — перезапрошен.
- (00:25) **det/plan-cached-drafters доказан на 4090 (тот же кэш):** master-план Q4_K_M mtp:5 → ветка dflash:7 (драфтер 1,21 ГБ учтён), UD-Q5_K_XL mtp:5 у обоих, в таблице Q4_K_M/Q8_0 dflash:7, остальные строки как были; реальный старт ветки без флага: «serves dflash:7», ступень 1 из 6 (32K) — план = старт. Юнит 8/8 sm_89 (первая сборка: ui_test SegFault — моя ошибка, цикл по .items() временного value(); исправлено 924bda34, по коду других таких мест нет). → **PR #127** (draft, Codex запрошен). Перемер MTP-точки 4090 на смеси (--spec mtp против none, Q4_K_M) — q_mixmtp идёт; сборка ветки на 4070 Ti для живой проверки GUI /api/plan.
- (00:40) 4090 смесь MTP Q4_K_M перемер (--spec mtp → mtp:3, глубина по умолчанию: профиль пары артефакта с DFlash выбирает dflash, R5 берёт глубину профиля только того же бэкенда — известное ограничение, для запросов без thinking совпадает с замеренной mtp:3): 245,1 против 146,6 ток/с = ×1,67, план без thinking ×1,92 → 0,87. Две карты: смесь 0,83–0,87 плана.
- (01:05) #127 живая проверка GUI на 4070 Ti (/api/plan, тот же кэш): master Q4_K_M/Q8_0 mtp:5 → ветка dflash:7 с памятью драфтера (1,21/1,11 ГБ), UD-Q5_K_XL mtp:5 у обоих; юнит 8/8 на 4070 Ti. Описание PR дополнено. Ждём Codex на #126 и #127.
- (01:20) **Разбор смеси по задачам (4090 UD-Q5_K_XL, master):** оценка плана без спекуляции (prefill 11436, decode 138,3) — 14,71 с против замера 14,99 (−1,9 %), по задачам ±2 % кроме коротких (фикс. ~50 мс). Префилл ≈0,8 с из 15 — **моё прежнее объяснение разрыва 0,84 «префиллом» было неверным (не проверено замером)**. Со спекуляцией (mtp:3 по запросу без thinking, профиль ×2,005): предсказано 7,74 с, замерено 8,58 (+11 %); декод ×1,83 (0,91 профиля); по задачам: история ×1,39, перевод ×1,6, якобиан ×2,2, код ×2,0, SQL ×2,04, NIAH/чат — фикс. накладные. Вывод: оценка времени верна, выигрыш спекуляции — от состава задач (смесь богаче прозой, чем набор spec-tune). Вопрос на будущее: показывать в GUI разброс выигрыша по типам задач (данные есть в профиле секций).
- (01:35) Проверка «выигрыш по типам задач из профиля пары» (секции math/code/prose/thinking в measurements реестра) на смеси 4090 UD-Q5_K_XL, mtp:3: взвешенное по задачам предсказание ×2,06 против замера ×1,82 — хуже агрегата профиля (×2,00). Секции переносятся для прозы (1,38 / 1,39) и математики (2,22 / 2,2), но не для кода (профиль ×3,25, смесь ×2,0: код spec-tune слишком лёгкий для драфтера). Вывод: показывать в GUI выигрыш по типам задач из текущих секций нельзя — сначала сделать секцию кода spec-tune реалистичной (кандидат в DECISIONS, вопрос владельцу). Код не менялся.
- (01:55) #127 Codex 2×P2 исправлены (8612b986): выбор той же конвертации, что берёт старт (рецепт, --vision; несколько рецептов без имени — что несут все), кэш планов GUI по отпечатку конвертаций на диске; юнит 8/8 sm_89; перепроверка 4090: план Q4_K_M dflash:7, Q5 mtp:5, старт dflash:7. Codex перезапрошен.
- (02:20) #127 Codex круг 2 (2×P2: --vision без проектора → текстовая конвертация; ref в отпечатке GUI) исправлен e460f059, юнит 8/8, Codex перезапрошен. 4070 Ti: замер пар соседних непромеренных квантов 9B MTP GGUF (Q4_K_S, Q5_K_M, Q6_K) — план даёт им plain (группа с защитой файнтюна), а братья Q4_K_M/UD-Q5_K_XL/Q8_0 замерены ×1,6–2,0 (mtp:3 без thinking) → проверка правила «сосед того же репо вместо группы» (q_sib).
- (02:50) Пары соседних квантов 9B MTP GGUF на 4070 Ti (spec-tune, mtp:3 без/с thinking): Q4_K_S ×1,90/×2,13 (брат Q4_K_M ×1,90/×2,23), Q5_K_M ×1,99/×2,29 (брат UD-Q5_K_XL ×1,99/×2,35) — перенос 0–4 %, а план даёт им plain (группа с защитой файнтюна). DECISIONS 48 (предложение). Ветка det/spec-sibling-pairs (56928f85): pair_choice — между своей парой и группой пары брата того же репозитория, ближайшего по битам в своём классе (basis preliminary, «registry sibling»); план передаёт target.repo, старт — manifest.repo или каталог кэша хаба; тест в test_spec_pairs. Сборка+юнит 4090 идёт; Q6_K замер идёт. #127: Codex круг 3 (P2: манифест с упавшим драфтером) исправлен 6d1c43c1, юнит 8/8, Codex перезапрошен.
- (03:20) det/spec-sibling-pairs в продукте (4090): таблица 9B MTP GGUF — master: 9 строк ≥4 бит plain (группа), ветка: все mtp:5 «registry sibling»; **Q6_K на смеси задач: старт ветки --spec auto (профиль брата, по запросу mtp:3 без thinking) 208,7 ток/с против 118,2 без спекуляции = ×1,77 (план без thinking ×1,75); master давал plain ×1,0**. Q6_K spec-tune 4070 Ti: ×2,00/×2,39 против брата UD-Q5_K_XL ×1,99/×2,35. Найдена ошибка ветки: ближайший брат брался по слитой записи карты (биты первого кванта) → Q5_K_M/Q6_K получали Q4_K_M; исправлено e3680d86 (+тест), пересборка и таблица идут.
- (03:50) det/spec-sibling-pairs исправлен (e3680d86 ближайший брат по своим битам; 8def2ce8 тест; 6bd8ce96 docs), юнит 8/8 sm_89; таблица 4090: Q5_K_M/Q5_K_S/Q6_K/UD-Q6_K_XL → UD-Q5_K_XL, UD-Q8_K_XL → Q8_0, 4-битные → Q4_K_M — **PR #128** (draft, Codex запрошен). Q6_K замер 4070 Ti ×2,00/×2,39 против брата ×1,99/×2,35.
- (04:20) #127 Codex круг 4 (P2: план синтезировал манифест из реестра, а старт читает кэшированный — --drafter-artifact/старая запись реестра) → 766c8f0a: product::hub_quant_subject — кэшированная конвертация с драфтерами планируется как её разрешает старт (speculation_subject артефакта), объявленные драфтеры рядом — для их памяти; convert::cached_conversion_with_drafters возвращает путь. Сборка+юнит+q_pcd на 4090 идут.
- (04:45) #127 766c8f0a: юнит 8/8, 4090 план Q4_K_M dflash:7 (1,21 ГБ) = старт dflash:7; Codex перезапрошен. #128 Codex идёт (👀), #126 Codex висит (👀).
- (05:00) Проверка правила братьев (#128) на других семействах, сравнение на одной карте: 4070 Ti — gemma-4-12b-it-GGUF Q4_K_M и UD-Q6_K_XL (ассистент); 4090 — Qwen3.8-27B-GGUF UD-Q4_K_S и Q4_K_M (MTP, DFlash2) — q_pairs_v2 (master), q_sibg / q_sib27.
- (05:20) Gemma 12B Q4_K_M на 4070 Ti: mtp:5 ×2,27 (thinking ×2,70, без thinking mtp:4 ×2,16); UD-Q6_K_XL на 4070 Ti не влезает с ассистентом (12 ГБ) — сравнение братьев Gemma на одной карте поставлено на 4090 после 27B (q_after27).
- (05:50) det/pairs-registry-3 (68df50af, реестр v119, 51 пара): + 4 пары 4070 Ti — 9B MTP GGUF Q4_K_S/Q5_K_M/Q6_K, Gemma 12B GGUF Q4_K_M. Юнит + план таблицы (--plan-device 4070 Ti) на 2080 Ti идут (q_reg3). 4090: 27B пары качаются (~6 МБ/с), затем Gemma Q4_K_M/UD-Q6_K_XL.
- (06:30) det/pairs-registry-3 проверен (2080 Ti, юнит 8/8; план 4070 Ti: 9B Q4_K_S/Q5_K_M/Q6_K → measured) → **PR #129** (draft). Gemma GGUF без --with-drafters — plain drafter_missing (DECISIONS 45 открыт).
- (06:50) DECISIONS 45 дополнено: в GGUF-репо Gemma (unsloth/bartowski) лежит mtp-*.gguf (ассистент в GGUF); NInferno его не читает (gguf_auxiliary_file, рецепт только у Nemotron-H) — кандидат: рецепт ассистента Gemma из того же репо.
- (07:00) **#129 (реестр v119) в master (62a1f10c)** — Codex 👍, юнит 8/8. 27B UD-Q4_K_S на 4090: dflash2:7 ×3,21 (thinking ×3,46, без ×3,13); Q4_K_M идёт.
- (07:30) 27B GGUF на 4090: Q4_K_M dflash2:11 ×3,46 (thinking ×4,02); против брата UD-Q4_K_S на его глубине dflash2:7: −2 %/−7 %, mtp:3 ≤2 %; своя глубина на 3–9 % лучше — правило братьев консервативно (DECISIONS 48). Gemma Q4_K_M/UD-Q6_K_XL на 4090 — идёт.
- (07:50) #126 Codex P2 (merge с несколькими base_model: план брал первый) → 5f4465e6: PlanTargetShape.base_models, convert::hub_base_models (все cardData.base_model), локальный чекпоинт — все из README; тест merge (первый родитель неизвестен). Юнит 9 тестов на 2080 Ti идёт. Gemma 12B Q4_K_M на 4090: mtp:5 ×2,28 (thinking ×2,71) — как на 4070 Ti (×2,27/×2,70); UD-Q6_K_XL идёт.
- (08:30) Gemma 12B GGUF на 4090: UD-Q6_K_XL mtp:5 ×2,34/×2,80 против Q4_K_M ×2,28/×2,71 — брат завысил бы на 3–4 %. По трём семействам ближайший брат −7…+4 % (DECISIONS 48, комментарий в #128).
- (08:50) 4090: проверка #128 в продукте на втором семействе — 27B GGUF Q4_K_M (своей пары в реестре нет): план master против #128 (брат UD-Q4_K_S), затем смесь --spec auto против none на сборке #128 (q_sib27mix; перекачка 17 ГБ). 2080 Ti: тесты #126 (5f4465e6) идут.
- (09:10) #126 5f4465e6 юнит 9/9 sm_75; Codex перезапрошен.
- (10:10) **#126 (lineage) в master** — Codex 👍, юнит 9/9. Найден дефект план↔старт: --quant Q4_K_M у unsloth/Qwen3.8-27B-GGUF — скачивание берёт UD-Q4_K_M (подстрока, точная метка выигрывает), план отказывал «no GGUF quantization Q4_K_M»; ширина «4bit» тоже. → det/plan-quant-selector: одно правило convert::gguf_group_for_quant (вынесено из fetch), listed_quant_for для листинга плана; тест в model_support. Тесты 2080 Ti идут. q_sib27mix на 4090 — план упал на этом же дефекте; смесь идёт (скачивание).
- (10:40) **#128 доказан в продукте на 2-м семействе**: 27B GGUF UD-Q4_K_M (без своей пары) на 4090 — master plain (группа), ветка mtp:5/mtp:3 по брату UD-Q4_K_S; смесь 96,8 против 50,3 ток/с = ×1,92 (план ×2,39 без thinking, 0,80). Комментарий в #128, DECISIONS 48, PLAN.
- (11:30) det/plan-quant-selector проверен на 4090 (Q4_K_M → UD-Q4_K_M 16,1 ГБ, 4bit → 17,2 ГБ; master отказывал), юнит 9/9 sm_75/sm_89; найдена и исправлена своя ошибка ширины (cbe1dc8f) → PR открыт.
- (12:00) #130 Codex P2 (разбитый GGUF без метки: листинг по частям, выбор брал одну часть) → 9e5f43ef: listed_quant_for возвращает одну запись всех частей группы, тест; перепроверка 4090 идёт. Калибровка Gemma Q8 (5070 Ti): MMLU-Pro 86 % (12 ч), ещё ~1 ч.
- (12:20) #130 9e5f43ef юнит 9/9 sm_89, 4090 перепроверка как раньше (Q4_K_M 16,1 ГБ, 4bit 17,2 ГБ); Codex перезапрошен.
- (13:00) #130 081c8f4c (каждый файл в своей группе, размеры файлов в RepoQuant) юнит 9/9, 4090 как раньше; Codex перезапрошен.
- (13:30) #130 Codex P2 (одно имя файла в двух папках: сводное имя без папки → повторный выбор при чтении заголовка неоднозначен) → 1ff14a58: имя записи — вся группа с папкой, тест; перепроверка 4090 идёт. Gemma Q8 cal: MMLU-Pro 97 %.
- (14:00) #130 1ff14a58 юнит 9/9, 4090 как раньше; Codex перезапрошен. lab/int5 = master + #127 + #128 + #130 (конфликт в read_hub: ревизия #127 + optional named #130 — обе строки); полный ctest sm_89 + fetch_test + ui_moved на 4090 идёт (q_ctint5).
- (14:30) **#130 в master (a032f9b7)** — Codex 👍. #127 обновлён с master (тот же конфликт read_hub, обе строки) aee127ad. lab/int5: собран (1226 с), полный ctest sm_89 идёт. Gemma Q8 cal: MMLU-Pro готов, идёт GSM8K (16 %).
- (15:30) lab/int5 (master+#127+#128+#130) полный ctest sm_89 277/277, ui_moved_update ok, model_support ok; fetch_test на 4090 не запустился (нет venv-torch) → 2080 Ti (q_int5cpu).
- (16:00) #127 Codex P2 (устаревшая ревизия с загрузкой → старт переконвертирует без драфтеров, план считал их) → e7ee83d8: convert::cache_served (cache_current без штампа) при наличии загрузки; тест; юнит+q_pcd на 4090 идут. Gemma cal: AIME25.

(08:18) #127: юнит упал на сборке — в тесте "5
" стал переводом строки (скрипт правки) → c88da2cb; юнит+q_pcd8 на 4090 заново. #128: повторно @codex review. 2080 Ti: int5 ещё собирается. 5070 Ti: Gemma cal в eval; пары Gemma UD-Q6_K_XL на 5070 Ti готовы (×2,65/×3,16).
(08:21) #127 c88da2cb на 4090: юнит 8/8, план Q4_K_M dflash:7 (драфтер 1,21 ГБ) = старт dflash:7 (master mtp:5); комментарий + @codex review. 4090 свободен.
(08:30) lab/int5 (master + #127 до e7ee83d8 + #128 + #130) CPU на 2080 Ti: drafter plan ok, ui ok, model support ok, fetch_test 0 провалов, ui_moved_update ok — вместе с полным ctest sm_89 277/277 интеграция зелёная.
(08:46) Codex P2: #128 — ближайший брат, замеренный с другим драфтером, отбрасывался после выбора → уход в группу; теперь поиск по carried_of (41fbaa19, тест); #127 — отпечаток плана GUI не видел отдельных драфтеров манифеста → 606fb78c. Юнит обоих на 4090 (sib4, затем pcd9). Gemma Q8 cal: AIME25 19/30. Очередь разработчика пуста. PLAN сверка обновлена.
(08:49) #128 41fbaa19 юнит 8/8 (4090) → комментарий + @codex. Gemma Q4_K_M полный прогон (4070 Ti, 24.10) извлечён: MMLU-Pro 72,58, GSM8K 95,98, MATH-500 92,2, AIME25 6,7, AIME26 0 (cal/gemma12_q4_full.scores.json). ΔKLD кванта в модели выбора для Gemma Q4_K_M = 0,0785 (интерполяция Qwen по bpw, перенос), Q8_0 = 0. Ждём Q8 (AIME).
(08:50) #127 606fb78c юнит 8/8 (4090) → комментарий + @codex.
(09:18) #128 Codex P2 «брат из старого коммита может быть другими весами» → 8757942c: брат только того же коммита (план: коммит листинга, RepoQuant.commit → PlannedTarget.commit, CLI и GUI; старт: каталог ninferno/<commit> артефакта); без коммита — нет брата; тесты (другой коммит, без коммита, полный хэш и регистр). Сборка упала (hub_repo_of_artifact после использования) → d4e619a4; юнит sib6 + q_sibproof2 (план 9B Q6_K/Q5_K_S, 27B UD-Q4_K_M, старый ref; старт Q6_K) на 4090. Коммиты реестра = HEAD хаба для 9B/27B/Gemma. DECISIONS 48 дополнен. Gemma Q8: AIME25 готов, идёт AIME26.
(09:22) #128 d4e619a4: юнит 8/8 (4090); в продукте план 9B Q6_K/Q5_K_S и 27B UD-Q4_K_M → брат, mtp:5; на предыдущем коммите → группа plain; старт 9B Q6_K → брат mtp:5 = план. Комментарий + @codex.
(09:45) #128: Codex 👍 на d4e619a4 («no major issues»); конфликт с master (#130) в read_hub → влит master cf701034 (обе строки; commit и у сводного RepoQuant listed_quant_for). Перепроверка: 4090 юнит 10 (с model_support) + q_sibproof3; 2080 Ti CPU+fetch_test (q_sibcpu). Затем — мерж #128 (по полномочию из памяти, после перепроверок).
(10:14) Gemma калибровка: Q8 (5070 Ti) MMLU-Pro 73,30 / Q4 (4070 Ti) 72,58 → 0,71 п. (Qwen 0,73); GSM8K 96,06/95,98; MATH 94,2/92,2. Кванты по метке переносятся в валюте ответов. Найдено: план завышает параметры Gemma ~18% (BF16 13,59 бит, Q8_0 7,22) → KLD-показ 0,0785 и порог качества (Q3_K_M «ниже порога»). DECISIONS 47 дополнен. Обзор bpw семейств (q_bpw.sh, 4070 Ti) идёт.
(10:22) **#128 в master (5716bec9)** (Codex 👍, 4090 юнит 10/10 + план=старт, 2080 Ti CPU/fetch_test зелёные). #127: влит master 02e6ce79 (конфликты: + .commit), 4090 юнит 10/10, план dflash:7 = старт; @codex. Новая ветка det/gguf-bits-per-weight: биты на вес по элементам таблицы тензоров GGUF (Gemma 4 формула +18%; Qwen −1,5…−4%), кривая Qwen пересчитана (×0,98454), фикстура — в работе.
(10:25) Причина завышения параметров Gemma 4 найдена: формула геометрии берёт head_dim 512 (key_length глобальных) для всех 48 слоёв, у 40 скользящих 256 (key_length_swa), kv-головы по слоям [8…,1], у глобальных нет attn_v (K=V) → внимание 4,5 млрд вместо 2,4 (+2,1 млрд = +18%). Влияет и на скорость (active_params → компьют префилла). Следующая задача: формула по слоям + замер префилла Gemma (отдельно от det/gguf-bits-per-weight). Сборка bpw на 4070 Ti идёт. 5070 Ti: Gemma Q4_K_M cal (повтор на той же карте).
(10:44) #127: Codex 👍 на 02e6ce79 → CPU+fetch_test на 2080 Ti (q_pcdcpu), затем мерж. det/gguf-bits-per-weight: юнит 10/10 (4070 Ti, sm_89); сравнение планов master↔ветка перезапущено (скрипт f-строки).
(10:56) Формула параметров Gemma 4: по тензорам q 881M = 40×(3840×4096)+8×(3840×8192), k 330M (глобальные kv=1×512), v только у 40 скользящих (K=V у глобальных; в метаданных GGUF не сказано — видно лишь по таблице тензоров). Отложено (влияет на компьютную часть префилла ≈5% времени запроса); после #127 и PR бит на вес.
(11:17) Владелец: 45 — да (драфтер GGUF — ось выбора без флага); 47 — потом, константы; 48 — «выключать спекуляцию всегда мимо». **#127 в master (5a16244d)**. det/gguf-drafter-axis 11a3099b (план/GUI с кандидатами, старт докачивает драфтер, если выбор им пишет) — юнит+проверка на 4090 идут. Анализ группы (LOO по реестру, 165 случаев): было plain 102 (×1,63, 71% своего лучшего), новое правило «средний выигрыш при приёме файнтюна» plain 0 (×2,19, 98%) → det/group-guarded-gain 1140c8fc, юнит+планы на 4070 Ti. DECISIONS 45/47/49. bpw what-if на 7 картах: решения меняются у 9B MTP (4060: квант; 5060/5090: plain→mtp:5) — разобрать.
(11:35) det/group-guarded-gain: юнит 10/10 (4070 Ti); план bartowski 9B Q4_K_M: master plain → ветка mtp:3 (группа) → **PR #131** (draft, @codex); смесь задач auto/none на 4070 Ti идёт. Владелец: периодические ревью и рефакторинг (код для сотен разработчиков) → память; ревью качества (Sonnet, только чтение) по #126–#130/#127/#128 и трём открытым веткам запущено. 2080 Ti: свежий master vs биты на вес на 7 картах.
(11:46) Ревью качества 1 (Sonnet): 15 находок → choice/REVIEW-QUALITY-1.md. Сделано: п.1/6/7/5ч — det/gguf-drafter-axis влит master (#127) + product::HubSubjects (один план субъектов хаба для сервера и GUI, без attaching/with_drafters-лямбд), backend_of_drafter_kind один (bffe4a30), юнит на 5070 Ti; п.15 — #131 рефакторинг группы (0ebe0ff0), юнит на 2080 Ti после boards. 4090: старт Gemma без флагов решил «attached», качается Q4_K_M.
(11:48) Рефакторинг по ревью: det/refactor-review-1 dbf7a5d6 (комментарии на местах, одно пространство имён, мёртвое PlannedTarget::lineage удалено, cache_serves/serves_at — одно чтение ревизии); юнит 12 на 2080 Ti в очереди. drafter-axis f8ba43ed (комментарий planned_subject). 4090: смесь Gemma (prepare → старт без флагов, auto/none) идёт; 4070 Ti: смесь #131 идёт.
(11:54) **#131 в продукте (4070 Ti)**: bartowski 9B Q4_K_M (нет своей пары) — группа mtp:3, смесь 122,9 против 76,1 ток/с = ×1,61 (master plain). Комментарий в PR.
(12:00) det/gguf-drafter-axis bffe4a30 (master + HubSubjects): юнит 10/10 sm_120a (5070 Ti). Ждём смесь Gemma на 4090 → PR.
(12:01) **Пункт 45 в продукте (4090)**: Gemma 12B Q4_K_M как в GUI (prepare → старт без флагов) — «attached», mtp:5 по паре, смесь 188,4 против 102,8 ток/с = ×1,83 (master plain) → **PR #132** (draft, @codex).
(12:14) Биты на вес: со свежим master 36/36 решений одинаковы (6 карт × 6 репо; «5060 Ti 16GB» — нет в таблице), юнит 10/10 sm_75; рефакторинг п.15 (метод PlanModelFacts::bits_per_weight, t.elements()) 487fddf4 → **PR #133** (draft, @codex).
(12:16) Рефакторинг п.8/14: det/refactor-pair-choice dc5edd71 (поверх #131): pair_choice по шагам (PairQuery, carried_of, pool_cards, own_cards, nearest_siblings, try_mode_of_kind, same_bit_class, same_repo_revision). 4070 Ti: юнит + сравнение планов с #131 (4 карты × 7 репо, каждая строка: served/basis/source).
(12:19) Codex P2: #131 — выигрыш при приёме файнтюна по разделам до взвешивания (Over::fine_tune_speed) 6cc4c15c; LOO заново: plain 0/165, ×2,20, 98%. #132 — локальный GGUF без отдельных драфтеров (PlannedSubjects, бывш. HubSubjects: with_drafters только у хаба), --drafter → не докачивать драфтер репо (ad82d406, тест). Ответы в PR + @codex. Тесты: 4090 gda3 + план локального файла; 4070 Ti grp3, pcr2 (refactor-pair-choice 16e2130a = + #131 фикс), сравнение планов; 2080 Ti grp2/rr1/bpw76.
(12:22) #132 ad82d406 на 4090: юнит 10/10; локальный файл Gemma Q4_K_M → plain drafter_missing (как master), хаб → mtp:5 с ассистентом (7,95 ГБ). Комментарий в PR.
(12:50) Рефакторинг pair_choice (dc5edd71) против #131 (1140c8fc): планы 4 карты × 7 репо — same 28, diff 0 (решение, основание, источник каждой строки). #131 6cc4c15c юнит 10/10 (4070 Ti). GUI-проверка #132 перезапущена (ls master-ui упал → && не запустил).
(12:51) #132 GUI /api/plan на 4090 (ad82d406): Gemma Q4_K_M/UD-Q6_K_XL mtp:5 measured, Q3_K_M mtp:5 own; строки таблицы с drafter_bytes 0,85 ГБ. Комментарий в PR.
(13:04) Рефакторинг pair_choice 16e2130a: юнит 10/10, планы против #131 same 28/28 (до и после фикса #131) → **PR #134** (draft, поверх #131, @codex).
(13:16) Codex P2: #131 — стоимость раунда раздела из замеренных секунд: seconds/(rounds + FT×accepted) (63b15611; LOO без изменений: plain 0/165, ×2,20), влито в #134; #132 — память драфтера из файла --drafter (supplied_drafter_memory: тензоры артефакта + состояние DFlash из конфига) bcccbd77/1286cf37; 4090: юнит + план с --drafter. det/refactor-review-1 юнит 12/12 (2080 Ti) → **PR #135** (@codex). #134 Codex 👍.
(13:18) bpw 487fddf4 юнит 10/10 sm_75. Рефакторинг п.3/4: det/refactor-convert-cache 4a6ce9fb (convert::local_commit, converted_name, conversion_carries_drafters, ManifestDrafter::present; тесты); 2080 Ti: CPU + fetch_test + cache_revision (q_cccpu) в очереди.
(13:19) #132 1286cf37 на 4090: юнит 10/10; --drafter → память по тензорам файла (845 713 936 Б), без флагов — драфтер репо, plain — без драфтера. Комментарий + @codex.
(13:44) #132 Codex P2: --drafter у цели-артефакта не считался (ранний возврат) → d4cd11a2; 4090 юнит + план артефакта с --drafter. #134, #135 — Codex 👍. #131/#133 — Codex идёт.
(13:44) **#135 в master** (рефакторинг по ревью: комментарии, одно чтение ревизии кэша, мёртвое поле; Codex 👍, юнит 12/12).
(13:47) #132 d4cd11a2: юнит 10/10 (4090); артефакт + --drafter → +845 713 936 Б. Комментарий + @codex.
(13:48) lab/int6 = master + #131/#134 + #132 + #133 + refactor-convert-cache (без конфликтов) → полный ctest sm_89 + fetch_test + ui_moved на 4090 (q_ctint6). Очищены свои сборки влитых веток: 4090 285→425 ГБ, 4070 Ti 228→387 ГБ свободно.
(13:49) Gemma 12B Q4_K_M на 4090, смесь без спекуляции: оценка плана 18,53 с против 18,84 замера (−1,7 %); префилл оценки 6357 против 6708 ток/с по длинному промпту (−5 %, осторожно). Завышение параметров формулой (+18 %) на скорость почти не влияет → формула по слоям — полировка, не критический путь. CCCPU (первый прогон): fetch_test 0 провалов, ui_moved ok, drafter plan/ui/model support ok; cache_revision не собран (скрипт) — во втором прогоне.
(13:51) refactor-convert-cache db315bd2 на 2080 Ti: cache revision (новые тесты), drafter plan, ui, model support, fetch_test 0 провалов, ui_moved ok → **PR #136** (@codex).
(15:25) int6 ctest на 4090: 128/277 без падений, идёт. Codex: #132/#136 👍; #131/#133 «running» 1–2 ч. Рефакторинг по ревью: **#137** одна ninferno::quant_label_bits (core/quant_label.h, optional; fetch → 0, пары → 16; TQ теперь с шириной) d1e344e7; **#138** одно чтение base_model GGUF (convert::gguf_metadata_base_repository: три ключа, хвост /tree срезается — план и старт теперь видят одну родословную) 87275c5a. 2080 Ti: q_lb → q_bm (CPU-тесты, fetch_test, drafters_test).
(15:27) **#139** (поверх #132): declared_drafters — один шаг объявления драфтеров; субъект кэшированной конвертации больше не планирует целый субъект ради attached (п.2 ревью) d9220cca; 2080 Ti q_dd (юнит + планы против #132, 4×7). 4070 Ti: q_int6plan — master против lab/int6 (что меняет стопка, 4×7).
(15:28) **#140** (поверх #136): convert::conversions_stamp рядом с правилом + каталог snapshots (дефект: после удаления загрузки GUI отдавал устаревший план) c6db960e; 2080 Ti q_cs в очереди. #133 — Codex перезапрошен (висел 2 ч). Остались из ревью: п.9 HubRevision (после мержа #131/#134 — те же строки pairs.cpp), п.10 RepoListing.
(15:30) Ревью Sonnet #137–#140 (проверено в коде): #137 нет include в plan_offline.cpp (не собралось бы) → 8830f1b1; #140 перевод строки в тесте (heredoc!) → dde25225 + refs по пути; #138 ссылки datasets/spaces/другой хост без схемы/?query давали ложный репозиторий → febc21e2 + тесты; #139 geometry_of 2f4bcdb3. **#141**: GUI читал геометрию модели второй копией (KvModel, ~300 строк, расходилась с plan_facts) — теперь kv_summary из фактов плана (−342/+28) 354a7776. 2080 Ti очередь: bm → dd → cs → lb2 → ui1 (устаревшая сборка lb убита по PID).
(15:40) #133 Codex 👍 (487fddf4). #131 Codex ещё идёт. @codex на #137–#141. 4070 Ti: собирается lab/int6 для сравнения планов. Мерж стопки — после полного ctest int6 (4090, 146/277, медленные тесты внимания ~5 мин).
(15:47) int6 ctest 148/277 без падений. Сравнение планов q_int6plan: скрипт обрезал вывод diff (видны только строки master) → cmp_plan.py (по строкам: спекуляция/основание, биты, выбор) на 4070 Ti. #137/#141 перенесены на 4070 Ti (q_lb89 → q_ui89, sm_89); 2080 Ti: bm (febc21e2) → dd → cs.
(15:58) #138 febc21e2 на 2080 Ti: gguf reader, drafter_plan, plan_bench_fits, fetch_test 0 провалов, drafters_test — зелёные.
(16:00) #137 8830f1b1 на 4070 Ti: собрано, spec_pairs/ui/plan_bench_fits/model_support 4/4; fetch_test там не запускается (нет venv) → q_lbf на 2080 Ti после cs.
(16:05) #141 354a7776 (4070 Ti): ui_test + plan_facts зелёные, ui_moved ok; живое сравнение /api/quants kv с master на 10 репо (q_uikv) поставлено.
(16:20) **lab/int6 полный ctest 277/277 (4090, sm_89)** → **в master: #136 (1db37984), #133 (4e1696a0), #132 (03959c81)** (Codex 👍 у всех; master = int6 без #131/#134 — разница только pairs.*). #139/#140 перенацелены на master, влит master (#140: конфликт app.cpp решён заново от master — «ours» откатил бы #132; 499ec5dc). #131 — Codex перезапрошен.
(16:20) Сравнение планов master(47d2b405)↔int6, 4 карты × 7 репо (cmp_i6.out): 548 строк меняют спекуляцию. Главное — #132: у GGUF драфтеры как ось: 9B MTP-GGUF mtp:5 → dflash:7 (measured: на 4090 смесь DFlash ×1,84 > MTP ×1,75 — согласно); 9B без MTP и bartowski plain → dflash:7 (preliminary/builtin); 27B plain → dflash2 (own); 35B/gpt-oss plain → dflash:7 где влезает; Gemma другие кванты drafter_missing → mtp:5 (группа). Выбор кванта меняется ради драфтера: 4060 9B Q5_K_S → IQ4_XS (+DFlash). В продукте доказано только на Gemma → 4090: q_dfmix (новый master, 9B GGUF Q4_K_M без флагов против none на смеси).
(16:25) #141 вживую: /api/quants kv на 10 репо = master (все поля; только 256 вместо 256.0).
(16:20 по стенду) **#131 Codex 👍 → в master (fb8d5bc4); #134 влит master → дерево = lab/int6 (проверено 277/277) → в master (9a735f16).** Master теперь = lab/int6. #137–#141 влит новый master (без конфликтов), запушены. 4090: q_dfmix — план 9B GGUF Q4_K_M = dflash:7 (preliminary, брат UD-Q5_K_XL), качается модель (~6 МБ/с), затем смесь auto против none.
(16:25) **DFlash-ось в продукте (master 03959c81, 4090): 9B GGUF Q4_K_M без MTP — план dflash:7 (брат UD-Q5_K_XL), старт без флагов «attached (a separate download)», служит dflash:7 = план; смесь 265,4 против 149,7 ток/с (×1,77), старый master — plain.** Следом: 27B UD-Q4_K_M (кэш есть) — план dflash2:11 «own» (опубликованный профиль, не лаб) → q_df27 auto против none.
(16:30) **27B UD-Q4_K_M (4090) без флагов: mtp:5 по брату UD-Q4_K_S, смесь 96,7 против 50,3 ток/с (×1,92).** Но таблица плана без --quant (и GUI) для тех же весов обещала dflash2:11 (опубликованный профиль). **Дефект план≠старт найден:** план без --quant берёт факты из самого малого GGUF; у unsloth 27B UD-IQ1_S/UD-IQ2_XXS нет MTP-слоёв (остальные с ними) → все строки планировались «без MTP», пара брата (замер с MTP) не применялась. Исправление det/rows-pair-resolution 5f23fdf5: у каждой строки свой признак MTP из её файла (convert::gguf_carries_mtp: nextn_predict_layers из первого МиБ, parse_gguf_metadata_prefix), в плане сервера и GUI (кэш по коммиту и кванту); тесты. 4070 Ti: q_rmtp (юнит + таблица = --quant по строкам + время). 4090: q_d2mix — 27B с --spec dflash2 на смеси (правда о профиле ×3,84).
(16:45) **Замер 27B UD-Q4_K_M (4090), смесь: dflash2:7 113,1 ток/с > mtp:5 96,7 > plain 50,3.** Значит второй дефект: выбор взвешивал только режим, которым замерена пара (брат — только MTP), и не видел DFlash2-драфтер, который цель несёт (−15 % скорости). Исправление в той же ветке aae1ad0b: ResolvedSpeculation::other_kinds — каждый несомый вид драфтера со своими лучшими данными (пара/брат/группа этого вида, иначе опубликованный профиль), одним кодом в плане и при старте (spec::resolve_other_kinds); ось драфтера предлагает все виды, subject_of_mode берёт свой. Тест на 27B (брат MTP + DFlash2 отдельно). 4070 Ti q_rmtp перезапущен на aae1ad0b; 4090 q_d2start — старт без флагов на ветке, ожидание dflash2 ≈113 ток/с.
(17:00) **ДОКАЗАНО в продукте (4090, ветка aae1ad0b): 27B UD-Q4_K_M без флагов — выбор взвесил DFlash2 рядом с MTP-братом, «attached», служит dflash2:7; смесь 113,2 ток/с (master: mtp:5 96,7; plain 50,3) = замер --spec dflash2.** Хвосты: (1) тест в plan_dry_run не видел пар (тот тест очищает реестр от пар) → свой тест test_plan_drafter_kinds.cpp на поставляемом реестре; (2) строки таблицы показывали профиль по умолчанию, а старт кванта — выбор драфтера → 1337d7da: каждая строка планируется при своём выборе (choose_at_weights + plan_at_choice), fit с памятью драфтера; (3) чтение MTP 27 файлов последовательно (таблица 16 → 34 с) → b76ae2b6: по 8 параллельно, в сервере заранее, в GUI с кэшем. 4070 Ti: q_rmtp на 1337d7da идёт (потом перемер времени на b76ae2b6).
(17:10) **PR #142** (det/rows-pair-resolution, b76ae2b6): юнит 5/5, таблица = --quant по строкам 27B на 4090/4070 Ti, 27B на смеси ×2,25 (113,2) без флагов; @codex. 4070 Ti q_rmtp2: время таблицы с параллельным чтением + master↔ветка 4×7.
(17:15) PLAN сверка обновлена. 4090: q_tune27 — spec-tune 27B UD-Q4_K_M (MTP + DFlash2, 1 ч) — лабораторная пара вместо опубликованного профиля (обещает ×3,25, смесь ×2,25).
(17:20) Gemma Q4_K_M повтор на 5070 Ti: MMLU-Pro 72,51 (4070 Ti 72,58: 0,07 п. между картами), потеря от Q8 на одной карте 0,79 п. (Qwen 0,73) → DECISIONS 47 дополнен; AIME Q8-прогона аномально низок (шёл ×2,5 дольше; вероятно обрезка контекста на 16 ГБ, не проверено). 4070 Ti q_rmtp2: таблица 27B с параллельным чтением 22,1 с (было 35, master 16), юнит 5/5, строки = --quant; сравнение 4×7 идёт. 4090: spec-tune 27B идёт.
(17:25) 5070 Ti: q_142b — #142 на Blackwell 16 ГБ: 27B без --quant и флагов (квант выбирает таблица, драфтер — выбор), смесь против --spec none на том же кванте.
(17:30) #139 45c81c1e: юнит 4/4, планы 28/28 = база #132 (2080 Ti).
(17:40) Codex: #137 P2 (TQ в gguf_quant_label) → 03543bde; #138 P2 (general.source.repo_url) → 8c872762; #142 P1 (кап контекста решался до выбора драфтера) → 9da84215. #139/#140/#141 Codex 👍. #140 e41d4a88 тесты зелёные (2080 Ti).
(17:50) **В master: #141 (25433fd5; GUI без второй копии геометрии, −342 строки), #140 (2ef95f43; штамп конвертаций рядом с правилом, с загрузками), #139 (41bd8556; одно объявление драфтеров)** — тесты зелёные, Codex 👍, живая проверка #141 (10 репо = master). #142 влит master (конфликт app.h решён, 07fff7f9). #137 fetch_test зелёный (a191b987); ждут Codex после фиксов: #137, #138, #142. 2080 Ti: q_mpost (master после слияний: CPU, fetch_test, ui_moved). 4070 Ti: сравнение master↔#142 4×7 идёт. 4090: spec-tune 27B. 5070 Ti: #142 на Blackwell.
(18:00) spec-tune 27B UD-Q4_K_M 4090: dflash2:11 ×3,30 (без рассуждений), ×4,02 (с ними) = опубликованный профиль; по задачам dflash2:7 математика ×4,6, код ×5,5, проза ×1,86 → смесь ×2,25 (0,68 оценки). DECISIONS 50 — вопрос владельцу (веса задач в оценке).
(18:05) master 41bd8556 после #139–#141: CPU 8/8, fetch_test 0 провалов, ui_moved ok (2080 Ti).
(18:10) #142 против master 4×7: 124 строки спекуляции (27B 4090 строки dflash2:11→:7 = старт; bartowski 9B dflash:7 builtin→mtp:5 generic — MTP-слои видны; 27B 5090 dflash2 own→mtp:5 брат Blackwell — не замерено; 4060 9B одна строка plain по DEC 42). Время таблицы 22 с.
(18:20) Codex второй круг: #142 P1 (пробы собственного контекста по умолчанию) → 01a67061 (PlanAt: все планы строки при одном выборе); #138 P2 (preflight поддержки не видел source.repo_url) → 795cdeb8 (тот же gguf_metadata_base_repository). #137 Codex 👍. Перепроверки: 2080 Ti q_cx (#137, #138), 4070 Ti q_rmtp3 (#142). 5070 Ti: #142 на Blackwell — план 27B pick UD-IQ4_XS mtp:5 (брат), смесь идёт.
(18:35) #142 01a67061 GUI на 4090: /api/plan 27B = план сервера 25/25 строк, выбор UD-Q5_K_XL одинаков; GUI 16 с первый раз, 0 с из кэша.
(18:40) #142 01a67061 на 4070 Ti: юнит 5/5, строки = --quant (4090/4070 Ti, что-если), таблица 22 с.
(18:50) **#137 в master (45fcc29c)** (одна метка→биты; TQ-метки читаются; тесты зелёные, Codex «no major issues»). Codex третий круг: #142 P2 (чтение MTP всех квантов при --quant) → d04c02d4 (только для таблицы); #138 P2 (точки в голом id: acme.ai/model) → 68fe7a00. #142 в GUI на 4090: /api/plan = план сервера 25/25. Ждут Codex: #138, #142.
(19:00) Codex 4-й круг: #142 — MTP «нет» только при прочитанном блоке архитектуры или всех метаданных (порядок ключей спецификацией не задан) + GUI с явным квантом читает только свой файл → ffc1dd45; #138 — source.repo_url раньше домашней страницы source.url → 984b68e2. 5070 Ti: мой скрипт делал --prepare-only без --quant (а prepare по замыслу ничего не выбирает — качал Q8_0) → перезапущен без prepare. Перепроверки: 2080 Ti #138, 4070 Ti #142.
(19:15) #138 984b68e2 зелёный (2080 Ti), ждёт Codex. Ревью п.9: det/hub-revision 98ea6e9b — core/hub_revision.h (HubRevision parse/id/same_as; имена каталогов кэша), spec/convert/serve/ui берут оттуда; разорванные комментарии на местах; −30 строк. 2080 Ti q_hr (CPU, fetch_test, ui_moved, планы против master).
(19:20) #142 ffc1dd45 (4070 Ti): юнит 5/5, строки = --quant, таблица 22 с. Ждёт Codex и Blackwell.
(19:30) **#138 в master (5badea7c)** (одно чтение base_model GGUF: план, старт, preflight; Codex чисто на 984b68e2, тесты зелёные). #142: Codex чисто на ffc1dd45; влит master (конфликты gguf.h/test_gguf — обе стороны), 82550771 → тесты на 4070 Ti, затем мерж. Blackwell: старт 27B без --quant выбрал UD-IQ4_XS (лог «the largest that fits … 16 GB at 32K»), качается.
(19:45) **#142 в master (c6dee000)**: тесты слитой головы 82550771 зелёные (4070 Ti), Codex чисто на ffc1dd45. Весь круг #131–#142 в master.
(19:50) 4090: полный ctest master c6dee000 (q_ctmaster).
(20:00) **Новые стенды от владельца:** llm3060 <LAN address> (RTX 3060 12 ГБ, sm_86, 0x2504 — в gpu_facts есть) и llmgv100 <LAN address> (Quadro GV100 32 ГБ, Volta sm_70, 0x1DBA — **в gpu_facts нет**; CUDA 13 не собирает Volta → CUDA 12.9 там уже стоит, run_cb.sh переключается при ARCH=70). Оба чистые: ставлю cmake/ninja/ccache/pkg-config/dev-библиотеки (apt, без драйвера и CUDA), venv-torch, /root/claude (lab.sh, run_cb.sh, run_mix, mix_tasks, messages), клон wt_tst. ssh_config и память обновлены.
(20:15) GV100 venv: python -m pip (первый pip упал segfault). CUDA-свойства: Quadro GV100 80 SM, 1627 МГц, HBM2 850 МГц ×4096 бит, 32 ГиБ; RTX 3060 = таблице. **det/gpu-facts-quadro-gv100 b2cc0b70**: строка 0x1DBA в gpu_facts.json + в список лабораторных карт теста machine_facts. Первые прогоны: GV100 master sm_70 (CUDA 12.9) сборка + полный ctest; 3060 master sm_86 + полный ctest.
(20:40) **#142 на Blackwell (5070 Ti 16 ГБ): 27B без --quant и флагов — таблица выбрала UD-IQ4_XS + mtp:5 = старт; смесь 92,5 против 48,5 ток/с (×1,91).** GV100: сборка master под 70 с CUDA 12.9 падает — segfault cicc (компилятор) на ggml_vec.cu (и с g++-14); проверенный тулкит для Volta — 12.8 → ставлю CUDA 12.8.1 (runfile, только toolkit, /usr/local/cuda-12.8) + g++-14 (CUDA 12.x — gcc ≤ 14). Владелец: GV100 ≈ V100 (тот же чип gv100).
(20:55) **PR #143** (det/hub-revision, ревью п.9): тесты зелёные на a6e3fea0, влит master (#142) с решением конфликтов (manifest.cpp/pairs.cpp), b97843bb → перепроверка на 2080 Ti против свежего master; @codex.
(21:15) Стенды: 4090 полный ctest master c6dee000; 3060 ctest master sm_86 (собрано за 901 с); 2080 Ti #143 перепроверка; GV100 ставится CUDA 12.8.1; 4070 Ti q_shipm — все пары реестра на картах своего класса без флагов (регрессия данных после #131–#142); 5070 Ti q_bwq9 — 9B GGUF Q4_K_M без флагов на Blackwell (DFlash-ось, доказано пока только на Ada).
(21:40) **Регрессия пар без флагов (4070 Ti, master c6dee000): 51 пара реестра на картах своего класса — все «measured», все fits** (5 строк в прогоне дали «no plan», при ручном повторе все 5 measured; временный сбой хаба — q_shipm2 повторяет со stderr, чтобы назвать причину). #143 на b97843bb: юнит 5/5, fetch_test, ui_moved зелёные; Codex 👍; ждёт сравнения планов против master (идёт). GV100: CUDA 12.8 распакован вручную (runfile не ставился без libxml2), но с glibc 2.43 cudafe падал на mathcalls.h (rsqrt/sinpi/cospi без noexcept) — в 12.9 NVIDIA это исправила; перенёс те же 6 объявлений в заголовок 12.8 (копия .orig), run_cb.sh для 70 = cuda-12.8 + g++-14; сборка master sm_70 пошла. Ревью п.10 + остаток п.12 — det/repo-listing: RemoteFile вместо трёх параллельных массивов, RepoListing{commit, quants} (6f6d1314); базовые модели из той же карточки, что проверка поддержки (минус один HTTP-запрос плана, ec86c0cf). 2080 Ti: q_rl после q_hr; удалены 16 сборок смерженных веток (−49 ГБ).
(21:55) **Blackwell DFlash-ось доказана (5070 Ti, master c6dee000): 9B GGUF Q4_K_M без флагов — план dflash:7 (брат UD-Q5_K_XL), старт «attached», служит dflash:7; смесь 241,4 против 132,6 ток/с (×1,82).** Причина «no plan» в регрессии пар найдена по stderr: **HTTP 429 хаба** (все стенды ходят с одного внешнего IP; анонимный лимит API 500/5 мин), повтор request() ждал 1+2+4 с — внутри того же окна. Хуже в таблице: неудачное чтение MTP-признака строки молча берёт факты другого файла. Исправление det/hub-rate-limit b9d91bf5: 429 ждёт сброса окна по заголовку RateLimit t= (документация HF, IETF draft), иначе Retry-After, не больше окна (300 с), как huggingface_hub; юнит. 4070 Ti q_rlim: те же 51 пара на ветке. #143: сравнение планов дало 17 изменённых строк + 2 NOPLAN (27B 4090/5090) — вероятно тот же 429 (чтение MTP строк); перезапуск сравнения с полным выводом (cmp_hr2) до мержа.
(22:05) **GV100: сбои не CUDA, а железо хоста.** Сборка на CUDA 12.8 + g++-14 тоже упала — ICE g++ (Aborted) в cuda_error.h, segfault cicc на ggml_mmq_iq3_s, abort на ggml_vec; dmesg: все 6 segfault cicc «likely on CPU 8 (core 16)»; CPU i7-14700KF (Raptor Lake), микрокод 0x137, плата B760M GAMING X DDR4, BIOS F25, без ECC. Ранний segfault pip — туда же. Значит вывод «CUDA 12.9 cicc падает на sm_70» не доказан (вероятно, то же ядро). Очищен ccache стенда и сборка; q_first заново под taskset -c 0-7,10-27 (без CPU 8/9 — одно физическое ядро). Если соберётся — сообщить владельцу (ядро деградировало/нестабильно), перепроверить 12.9. PLAN сверка 22:05.
(22:10) **4090: полный ctest master c6dee000 — 278/278 зелёные** (после всего круга #131–#142).
(22:15) **3060 (Ampere): полный ctest master c6dee000 — 278/278.** Там же q_amq9: 9B GGUF Q4_K_M без флагов (Ampere — пар своего поколения нет) против --spec none на смеси.
(22:25) 3060 план 27B без --quant (32K): pick UD-Q2_K_XL (2,92 бита, ниже порога) + mtp:3 «similar» (своего поколения пар нет); строки 1–2 бита dflash2:7. q_amq9 (9B Q4_K_M без флагов) качает модель.
(22:35) **#143 в master (84743071)**: повтор сравнения планов — одно различие (4060 bartowski 9B Q5_K_S plain→mtp:3), при повторе оба бинарника mtp:3 → это сбой чтения MTP в прогоне master (дефект 429, чинит det/hub-rate-limit); реальных изменений 0. Codex 👍, тесты зелёные.
(22:45) det/hub-rate-limit b9d91bf5 на 4070 Ti: 51/51 measured, 0 no plan, юнит зелёный — но 429 в этом прогоне не случился (путь ожидания не проверен). Поэтому fefe447f: в поддельном хабе fetch_test — репозиторий с окном лимита 9 с (дольше 1+2+4 с), случай «rate limit waited out»; q_rlim2 — ветка и контроль на master (должен упасть).
(23:05) GV100: master sm_70 собран (CUDA 12.8 + патч, taskset без CPU 8/9) за 1783 с, ни одного нового segfault → ядро подтверждено; владельцу отправлено уведомление; полный ctest идёт. det/repo-listing (q_rl): 6/6, fetch_test, ui_moved зелёные; сравнение планов против master 84743071 идёт. Полировка: det/gemma-layer-params 77ac6d00 — параметры внимания по слоям (свои ширины; ключи/значения только у слоёв со своим KV; attention_k_eq_v — одна проекция); 4090 q_gp: тесты плана + params master/ветка против числа элементов тензоров GGUF (Gemma 12B/31B/26B-A4B/E4B, Qwen 9B/27B/35B — у Qwen без изменений).
(23:15) **Ampere (3060 12 ГБ), master без флагов: 9B GGUF Q4_K_M — план dflash:7 (брат UD-Q5_K_XL, перенос с Ada), старт «attached», служит dflash:7; смесь 84,3 против 52,9 ток/с (×1,59).** План: plain 53,7 ток/с (замер 52,9 — точно), со спекуляцией ×2,31 без рассуждений → смесь 0,69 обещанного (Ada тот же 9B — 0,84). Перенос профиля с Ada на Ampere завышает (у 3060 мало вычислений на байт: проверка 8 строк дороже). Решение не меняется (спекуляция ×1,59 > plain), но оценке нужен замер поколения → 3060 q_tune9a: spec-tune 9B Q4_K_M + DFlash (1 ч) — первая пара Ampere.
(23:25) Перенос профиля на поколение без пар (spec/pairs.cpp profile_for_target): берётся selected() одной карты целиком — множитель ускорения. Ampere (×1,59 при обещанных ×2,31·0,84≈1,94) показывает, что цена раунда (проверка k+1 строк) зависит от поколения. Жду spec-tune 3060: floor (сек/раунд к шагу plain) Ampere против Ada той же пары → если различие большое, переносить приём (E(k), токенов за раунд — свойство пары), а цену раунда брать по карте.
(23:35) det/repo-listing: 6/6, fetch_test, ui_moved зелёные, планы против master 0 изменений → влит master (#143, без конфликтов) → **PR #144** (draft, @codex). 2080 Ti: ждуны с pgrep -f совпадали сами с собой (команда ждуна содержит искомое имя) — не запускались; теперь seq_2080.sh: q_rlim3 (лимит хаба, ветка + контроль master), затем q_rl2 (слитая голова #144).
(23:55) **Первая пара Ampere (3060, 9B Q4_K_M + DFlash): dflash:7 ×2,00 без рассуждений, ×3,10 с ними** (перенос с Ada обещал ×2,31/×3,40). Токенов за раунд = Ada (3,79 против 3,77) — приём свойство пары; цена раунда в шагах plain 1,90 против 1,63 (+16 %) — свойство поколения. Смесь ×1,59 = 0,80 своего профиля (Ada 0,84 — состав задач, DECISIONS 50). → det/pairs-ampere 5a5cba7f (реестр v120). Вторая пара Ampere: 9B MTP Q4_K_M (MTP+DFlash) — q_tune9m на 3060: держится ли поправка цены раунда поколения для другой пары/драфтера. **Лимит хаба:** fetch_test «rate limit waited out» ok на ветке, FAILED на master (контроль) → **PR #145** (@codex); полный fetch_test с пробником — q_rlim4 после q_rl2.
(00:05) **#144 в master (9d360406)** — ревью п.10 и остаток п.12: слитая голова 6/6, fetch_test, ui_moved зелёные; Codex чисто; планы 0 изменений. Ревью-круг 1 закрыт целиком (пп. 1–15 + два «+»).
(00:15) det/gemma-layer-params (4090): тесты плана 5/5; параметры против подсчёта хаба по GGUF (gguf.total): Gemma 12B 11,922 против 11,907 (+0,13 %; master 14,03, +18 %), 31B 30,806 / 30,697 (+0,35 %; master +24 %), 26B-A4B 25,236 / 25,233 (master +4 %); Qwen 9B/27B/35B без изменений. E4B 4,56 против 7,52 — таблицы эмбеддингов по слоям (поиск, не вычисление: для компьюта верно). Ключи=значения у глобальных слоёв GGUF в метаданных не сказаны (лишние 16 M — 0,13 %). q_gpre: оценка префилла master/ветка против замера 22K (Gemma 12B, 26B-A4B, контроль Qwen 9B) — не стала ли оценка оптимистичной.
(00:30) #145: полный fetch_test с пробником зелёный, юнит зелёный; Codex P2 (ждать сброса только исчерпанной квоты r=0, иначе Retry-After) → 3456acd2 + юнит-случаи; q_rlim5 перепроверка. 4090 q_gpre: оценка префилла Gemma 12B Q4_K_M — master 6357, ветка 7480 ток/с; замер идёт. 3060: q_tune9m качает 9B MTP Q4_K_M. GV100: полный ctest sm_70 идёт.
(00:40) #145 3456acd2: юнит и полный fetch_test зелёные (2080 Ti); ждёт Codex.
(01:05) **Вторая пара Ampere (3060, 9B MTP Q4_K_M, MTP + DFlash): dflash:7 ×2,03 / ×3,09.** Против записи 4070 Ti тех же весов: токены за раунд совпадают на всех конфигурациях (dflash:7 3,79/3,80, mtp:3 3,15/3,15) — приём = свойство пары; цена раунда на Ampere выше и растёт с числом проверяемых строк: dflash:3 ×1,02, :5 ×1,08, :7 ×1,09, :11 ×1,21; mtp:2 ×1,06 … mtp:5 ×1,11 (прежние ×1,165 — артефакт сравнения разных квантов). Перенос на чужое поколение уже отсекает глубины >7 (selected(home=false)), остаётся 2–11 %. DECISIONS 51 (черновик). det/pairs-ampere a9566302 (реестр v121). q_pamp на 3060: тесты, план 4 целей master/ветка, смесь 9B MTP Q4_K_M без флагов против none. Скрипт roundcost.py — цена раунда по конфигурациям двух замеров.
(01:25) **GV100 (Volta sm_70): полный ctest master 277/278** — единственный провал ninferno_linear_generic_test: «nvfp4 A8 T=17/64/200: tiled.128x128» — тест ждал W4A8 «на любой сборке», а его int8 MMA m16n8k32 входит в mma_bf16, которого у Volta нет; выбор маршрута верен (tiled A16), ошибка теста → det/volta-tests (0ea0521e + строка GV100 в gpu_facts 193a79e8); q_vt на GV100 и 2080 Ti. q_pamp (3060) перезапущен: неверная цель ninferno_registry_add_test и экранирование в однострочнике — проверка вынесена в plan_spec_row.py.
(01:45) **Префилл 22K на 4090 (q_gpre): Gemma 12B Q4_K_M замер 6144 ток/с — оценка master 6357 (+3,5 %), ветка gemma-layer-params 7480 (+22 %); Qwen 9B контроль 11828 против 11436 (−3 %).** Модель префилла = eta·peak/(2·params), без FLOP внимания по контексту; eta подогнан на Qwen (полное внимание 8/32 слоёв ≈ +8 % FLOP). У Gemma на 22K глобальные слои (головы 512) ≈ +12 %, скользящие ≈ +3 % → точность master была компенсацией двух ошибок. Вывод: параметры по слоям нельзя вливать одни — нужна пара «FLOP внимания в префилле + переподгонка eta». 26B-A4B: оценка декода master 169 → ветка 198 — ждём замер (декод важнее). #145: Codex чисто на 3456acd2; влит master (#143/#144) 638ef5bf → q_rlim6 на 2080 Ti после q_vt.
(02:10) **Пары Ampere в продукте (3060, det/pairs-ampere a9566302): 9B Q4_K_M ×2,31 preliminary (Ada) → ×2,00 measured; 9B MTP Q4_K_M ×2,22 → ×2,03 measured (своя Ampere); смесь 9B MTP Q4_K_M без флагов 85,7 против 52,4 (×1,64) = 0,81 своего профиля (Ada 0,84) — со своей парой остаётся только разрыв состава задач (DECISIONS 50).** Но Q5_K_M / Q6_K (на Ampere не мерены) брали Ada как есть. → 770ae5b2: generation_round_costs — по конфигурации медиана отношения цены раунда по целям, замеренным на обоих поколениях; профиль чужого поколения масштабирует секунды раундов; токены за раунд — от пары. В реестре: Ada→Ampere dflash:7 1,095, mtp:5 1,110 (2 карты Ada согласны); Ada↔Blackwell общих целей нет — без изменений. Юнит (синтетическая Ampere-копия ×1,10). q_pamp2: тесты, планы 4 целей, смесь 9B MTP Q6_K (непромеренная на Ampere) без флагов против none.
(02:40) **Передача работы.** Итоги последних прогонов: (1) 3060, 9B MTP Q6_K (на Ampere не мерен): план master ×2,34 → #146 ×2,14; смесь без флагов 79,2 против 41,8 (×1,89 = 0,88 плана) — поправка поколения доказана на непромеренной комбинации. (2) 4090 Gemma 26B-A4B 22K: замер префилл 13196 / декод 196,3; master 8570 / 169,2 (−14 %), ветка #147 10702 / 197,8 (+0,8 %) — для MoE ветка сильно точнее, для плотной 12B префилл хуже → #147 draft с условием. (3) **Volta GV100: 9B GGUF Q4_K_M без флагов — DFlash подключён, смесь 72,3 против 70,9 (×1,02) при плане ×2,31 — нужна пара Volta.** (4) #145 слитая голова зелёная → **в master (11659583)**. (5) det/volta-tests: и вторая правка теста (по route_admits, 9ff2ec3e) на GV100 падает — маршрут tiled при T≥17; причина (допуск/выбор/owned_prefill_w4a8) не выяснена. PR: #146 (Ampere + generation_round_costs, draft), #147 (Gemma, draft). Заметки: choice/HANDOFF.md; материалы: choice/handoff/ (WORKLOG, lab, stand_kit, scripts, data/ampere). Блокировок GPU моих нет ни на одном стенде.
