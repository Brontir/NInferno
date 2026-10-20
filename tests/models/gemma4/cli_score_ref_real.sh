#!/bin/bash
# ninferno-gemma4 score of one real Gemma 4 artifact against a reference on the same weights: the
# first 1024 tokens of wiki (ids_wiki.bin) scored with one window; mean NLL within TOL of the
# reference's and top-1 agreement at least AGREE where the reference's top-1/top-2 gap exceeds 0.5.
#   cli_score_ref_real.sh <ninferno-gemma4> <case>
# NINFERNO_TEST_GEMMA4_DIR holds gemma4_<case>.ninferno, NINFERNO_TEST_GEMMA4_REF holds <case>/ with
# ids_wiki.bin, <prefix>_lp_wiki.bin, <prefix>_top1_wiki.bin and <prefix>_gap_wiki.bin: the reference
# run's per-position log-probabilities, top-1 ids and top-1/top-2 gaps (FP32, I32, FP32), from HF
# transformers or llama.cpp, supplied with the reference set. Without either the test skips (77).
#   e2b_bf16      google/gemma-4-E2B-it BF16 vs HF transformers BF16 (prefix hf)
#   e2b_qat_q4_0  Google's QAT Q4_0 GGUF vs llama.cpp on the same GGUF (prefix lcpp)
#   e4b_bf16      google/gemma-4-E4B-it BF16 vs HF transformers BF16 (prefix hf)
set -u
BIN=$1
CASE=$2
DIR=${NINFERNO_TEST_GEMMA4_DIR:-}
REF=${NINFERNO_TEST_GEMMA4_REF:-}
if [ -z "$DIR" ] || [ -z "$REF" ]; then
  echo "skip: NINFERNO_TEST_GEMMA4_DIR or NINFERNO_TEST_GEMMA4_REF is not set"; exit 77
fi
case "$CASE" in
  e2b_bf16)     PREFIX=hf;   TOL=0.02; AGREE=0.99 ;;
  e2b_qat_q4_0) PREFIX=lcpp; TOL=0.02; AGREE=0.99 ;;
  e4b_bf16)     PREFIX=hf;   TOL=0.02; AGREE=0.99 ;;
  *) echo "unknown case $CASE"; exit 2 ;;
esac
ARTIFACT=$DIR/gemma4_$CASE.ninferno
R=$REF/$CASE
if [ ! -e "$ARTIFACT" ] || [ ! -e "$R/ids_wiki.bin" ]; then
  echo "skip: $ARTIFACT or $R/ids_wiki.bin is missing"; exit 77
fi
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
"$BIN" "$ARTIFACT" score --tokens "$R/ids_wiki.bin" --window 1024 --context 4096 \
  --dump "$WORK/score.dump" > "$WORK/score.json" 2> "$WORK/score.err"
rc=$?
if [ $rc -ne 0 ]; then echo "score failed (rc=$rc)"; tail -5 "$WORK/score.err"; exit 1; fi
python3 - "$WORK/score.dump" "$R" "$PREFIX" "$TOL" "$AGREE" <<'PY'
import sys
import numpy as np
dump, ref, prefix, tol, agree = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
raw = np.fromfile(dump, dtype=np.uint8)
half = len(raw) // 8
lp, top1 = raw[: half * 4].view(np.float32), raw[half * 4:].view(np.int32)
ref_lp = np.fromfile(f"{ref}/{prefix}_lp_wiki.bin", dtype=np.float32)
ref_top1 = np.fromfile(f"{ref}/{prefix}_top1_wiki.bin", dtype=np.int32)
gap = np.fromfile(f"{ref}/{prefix}_gap_wiki.bin", dtype=np.float32)
n = min(len(lp), len(ref_lp))
delta = float(-lp[:n].mean() + ref_lp[:n].mean())
clear = gap[:n] > 0.5
agreement = float((top1[:n] == ref_top1[:n])[clear].mean())
ok = n >= 1000 and abs(delta) <= tol and agreement >= agree
print(f"scored {n}: nll {-lp[:n].mean():.4f} vs {prefix} {-ref_lp[:n].mean():.4f} (delta {delta:+.4f}, "
      f"tolerance {tol}), top-1 agreement {agreement:.4f} where the gap > 0.5 (at least {agree}) -> "
      f"{'ok' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
