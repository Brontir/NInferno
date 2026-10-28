#!/bin/bash
# Audit of synchronous CUDA copies and fills on the legacy default stream.
#
# Every NInferno device stream is created cudaStreamNonBlocking, so the legacy stream does not order
# against them: a cudaMemcpy/cudaMemset that races work queued on a device stream reads or writes
# memory mid-flight. Such calls are allowed only where the per-file allowlist below says why they
# cannot race (startup before any stream work, a debug dump behind a stream synchronize, a
# single-stream session that synchronizes its stream first). A changed count in a listed file, or
# any such call in another file, fails the audit and asks for a review of the new call.
#
# The same holds for tests: seeding device memory with a plain cudaMemset and then running the code
# under test on a DeviceContext stream is a race (ninferno_state_store_test failed that way under GPU
# load). tests/ has its own allowlist.
#
# A stream parameter with a default (`cudaStream_t stream = nullptr`) silently puts a caller that
# forgets the argument on the legacy stream, so none is allowed in src/ or tests/.
#
#   legacy_stream_calls.sh <source root>
set -u
ROOT=${1:-.}
cd "$ROOT" || exit 2
# file|expected count|why the calls cannot race
ALLOW=$(cat <<'EOF'
src/core/arena.cu|3|DeviceBuffer::fill/copy_from_host/copy_to_host: startup and test utilities; fill and copy_from_host settle the default stream before returning, the Engine runtime paths pass streams
src/artifact/materializer.cpp|1|--debug-poison fill of the weight arena, followed by cudaDeviceSynchronize before the first upload
src/core/paged_kv_cache.cpp|1|--debug-poison fill at pool construction, followed by cudaDeviceSynchronize
src/models/common/layer_dump.h|1|the debug layer dump (dense, Gemma 4, Qwen): D2H after cudaDeviceSynchronize
src/models/gemma4/target.cu|1|NINFERNO_GEMMA4_DUMP: the first position of a dumped pass, after cudaStreamSynchronize
src/models/dense/target.cu|1|NINFERNO_DENSE_DUMP: the first position of a dumped call (not under capture), after cudaStreamSynchronize
src/models/speculation/decoder.cu|5|single-stream decoder (moved from gemma4/speculation.cu): each read follows device_.synchronize()
src/models/gemma4/load_target.cpp|1|router_scale D2H at load, after demands.materialize synchronized its transfer stream and before any Engine stream work
tests/artifact/materialization_cuda_errors.cpp|1|D2H after materialize, which synchronizes its transfer stream
tests/artifact/test_materialization.cpp|9|D2H after materialize, which synchronizes its transfer stream
tests/models/qwen3_5/test_dflash_prefill_real.cpp|3|cache and block-table D2H, every prefill synchronizes device.stream first
tests/models/qwen3_5/test_context_store.cpp|1|block-table D2H, every caller synchronizes device.stream first
tests/models/qwen3_5/test_loading_real.cpp|1|D2H after materialize_model, which synchronizes its transfer stream
tests/models/qwen3_5/test_state_image.cpp|1|expect_bytes D2H, every caller synchronizes device.stream first
tests/ops/test_fp8_e4m3.cu|1|D2H after the widening kernel on the same legacy stream; the test makes no other stream
tests/ops/test_gemma4_attention_fp8.cu|4|zero fills on the legacy stream before the appends and launches on the same stream (stream nullptr)
tests/ops/test_gemma4_fused.cpp|1|H2D of the input before scale_bf16 runs on the legacy stream (stream nullptr), cuda_synchronize before the read
tests/ops/test_ggml_decode.cu|3|legacy-stream only: upload, fill and launches on the default stream, cudaDeviceSynchronize before the read
tests/ops/linear/linear_test_common.cpp|1|D2H after the launch stream is synchronized
tests/ops/linear/test_generic_linear.cpp|8|legacy-stream reference, fills and launches (stream nullptr), cudaDeviceSynchronize before reads
tests/ops/linear/test_side_by_side_projection.cpp|2|poison fills before launches on the legacy stream (stream nullptr), cudaDeviceSynchronize before reads
tests/ops/linear/test_dense_tune.cpp|1|download(): D2H after the call stream is synchronized
tests/ops/linear/test_linear_gemma4_shapes.cpp|6|staging on the legacy stream the launches also use
tests/ops/linear/test_linear_gemma4_token_invariance.cpp|1|D2H after cudaDeviceSynchronize
tests/ops/linear/test_qwen_prefill_token_invariance.cpp|3|the test's launch stream is the legacy stream; cuda_synchronize before reads
tests/ops/linear/test_segmented_swiglu.cpp|1|poison fill before the launches on the legacy stream (stream nullptr), cudaDeviceSynchronize before reads
tests/ops/linear_add/linear_add_test_common.cpp|1|D2H after the launch stream or the device is synchronized
tests/ops/linear_pair/linear_pair_test_common.cpp|1|D2H after cudaDeviceSynchronize
tests/ops/linear_swiglu/linear_swiglu_test_common.cpp|1|D2H after the launch stream or the device is synchronized
tests/ops/op_tester.h|4|GuardedDeviceBuffer fill/copy_from_host settle the default stream; reads follow a synchronize
tests/ops/softmax_attention/causal_cache.cpp|1|BatchDeviceCache::copy_from settles the default stream after the copy
tests/ops/test_gdn_replay_fold.cpp|19|legacy-stream reference path, fenced by cuda_synchronize before the graph and every read
tests/ops/test_kv_cache_append.cpp|2|legacy-stream only, after cuda_synchronize
tests/ops/test_linear_topk.cu|2|legacy stream; the graph streams are cudaStreamCreate (blocking) streams
tests/ops/test_moe.cpp|8|legacy-stream fills, copies and launches (stream nullptr): run() synchronizes the device before its read; check_warp_plan launches on the legacy stream between its synchronous copies
tests/ops/test_moe_frag.cpp|2|poison fill and launches on the legacy stream (stream nullptr), cudaDeviceSynchronize before the read
tests/ops/test_rmsnorm.cpp|2|uploads fenced by the output fill's settle and cuda_synchronize before launch
tests/ops/test_sparse_moe.cpp|2|legacy-stream eager path, cuda_synchronize before the graph
tests/test_bench_fixtures.cu|4|D2H of fixtures whose uploads settle the default stream; the timing checks follow measure_*_prepared, which synchronizes its stream
tests/test_arena.cpp|1|fill of a buffer no stream work touches, before cudaFree
tests/test_decode_graph.cpp|1|D2H after device.synchronize
tests/test_device_buffer_fill_order.cpp|2|the deliberately unfenced control arm of the fill-order test, read after cudaDeviceSynchronize
tests/test_kv_cache.cpp|2|D2H after context.synchronize
tests/test_state_store.cpp|1|expect_device_byte D2H, every caller synchronizes ctx.stream first
EOF
)
SOURCES=(--include=*.cpp --include=*.cu --include=*.h --include=*.cuh)
fail=0
found=$(grep -rnE 'cuda(Memcpy|Memset|Memcpy2D|Memset2D|Memcpy3D|MemcpyToSymbol|MemcpyFromSymbol)\(' src tests \
          "${SOURCES[@]}" | grep -vE '^\S+:[0-9]+:\s*//')
files=$(echo "$found" | grep -v '^$' | cut -d: -f1 | sort | uniq -c | awk '{print $2"|"$1}')
for entry in $files; do
  file=${entry%|*}; count=${entry#*|}
  line=$(echo "$ALLOW" | grep "^$file|")
  if [ -z "$line" ]; then
    echo "UNLISTED legacy-stream calls in $file ($count):"; echo "$found" | grep "^$file:" | sed 's/^/  /'
    fail=1; continue
  fi
  want=$(echo "$line" | cut -d'|' -f2)
  if [ "$count" != "$want" ]; then
    echo "CHANGED $file: $count legacy-stream calls, the allowlist reviewed $want:"
    echo "$found" | grep "^$file:" | sed 's/^/  /'
    fail=1
  fi
done
# An allowlisted file whose calls are all gone keeps a stale reason: drop it from the list.
for file in $(echo "$ALLOW" | cut -d'|' -f1); do
  if ! echo "$files" | grep -q "^$file|"; then
    echo "STALE allowlist entry $file: it has no legacy-stream calls now"; fail=1
  fi
done
# Explicit stream 0 / nullptr in an Async call is the legacy stream too.
explicit=$(grep -rnE 'cuda[A-Za-z0-9]*Async\([^;]*,\s*(0|nullptr|NULL|cudaStreamLegacy)\s*\)' src \
             "${SOURCES[@]}")
if [ -n "$explicit" ]; then echo "Async call on the legacy stream:"; echo "$explicit" | sed 's/^/  /'; fail=1; fi
# A defaulted stream parameter: a caller that omits it runs on the legacy stream.
defaulted=$(grep -rnE 'cudaStream_t\s+\w+\s*=\s*(0|nullptr|NULL|cudaStreamLegacy)\s*[,)]' src tests \
              "${SOURCES[@]}")
if [ -n "$defaulted" ]; then
  echo "Stream parameter with a default:"; echo "$defaulted" | sed 's/^/  /'; fail=1
fi
total=$(echo "$found" | grep -c . )
echo "legacy-stream audit: $total synchronous calls in $(echo "$files" | grep -c .) files, all reviewed: $([ $fail = 0 ] && echo yes || echo NO)"
exit $fail
