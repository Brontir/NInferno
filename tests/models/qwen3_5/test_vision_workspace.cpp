#include "core/device.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/vision.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/weight_input.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/program/model_family.h"
#include "real_artifact.h"
#include <cstdlib>

#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

int main() {
    namespace qwen       = ninferno::models::qwen3_5;
    const char* artifact = std::getenv("NINFERNO_TEST_ARTIFACT");
    if (!artifact || !*artifact) { return 77; }
    // The route is sized beside MTP: a text-only artifact, or one without the head, has neither.
    if (const auto& parts = ninferno::test::real_artifact(artifact); !parts.vision || !parts.mtp) {
        std::cout << "skip: the artifact has no " << (parts.vision ? "MTP" : "vision")
                  << " component\n";
        return 77;
    }
    static_assert(ninferno::models::qwen3_5::kMaximumVisionItemTokens == 16384);
    // The Vision route's size is a law of its tower, not one number (#429: Qwen3.5-4B 768.5 MiB,
    // Qwen3.6-35B-A3B and Qwen3.8-27B 916.5 MiB): its widest scope holds the residual, the MLP's up
    // projection and its norm, (2 H + I) BF16 values a patch, beside the position ids (8 bytes a
    // patch) and the scratch its linear calls declare (their routes' activation planes: an A8
    // fc1 of a 1152-wide tower adds 90 MiB at 65536 patches). The workspace must not grow with the
    // text context past one item's.
    try {
        int devices       = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && devices == 0)) {
            return 77;
        }
        CUDA_CHECK(status);
        ninferno::DeviceContext device;
        ninferno::models::LoadOptions selected;
        selected.vision        = true;
        selected.speculative   = ninferno::SpeculativeBackend::Mtp;
        // The optimized head where the artifact's serves MTP (one converted --no-proposal has none).
        const ninferno::ProposalHead head = ninferno::test::proposal_head(
            artifact, ninferno::ProposalHead::Optimized, ninferno::SpeculativeBackend::Mtp);
        selected.proposal_head = head;
        auto model             = qwen::load_model(artifact, selected, device);
        const qwen::execution::Parameters parameters(*model);
        const auto execution = qwen::detail::make_qwen3_5_execution(parameters);
        const auto capacity = [&](std::uint32_t max_context) {
            ninferno::EngineOptions options;
            options.max_context         = max_context;
            options.kv_capacity = ninferno::KvCapacityPolicy::explicit_capacity(max_context);
            options.prefill_chunk       = 1024;
            options.kv_cache            = ninferno::KvCacheStorage::Fp8E4M3Row256;
            options.speculative.backend = ninferno::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens         = 3;
            options.speculative.proposal_head        = head;
            options.enable_vision                    = true;
            options.use_cuda_graph                   = false;
            options.context_cache.device_state_slots = 1;
            auto planner     = qwen::make_sequence_planner(*execution, device, options);
            const auto pages = planner.capacity_curve().minimum_main_page_groups;
            return std::move(planner).finalize(pages).workspace_capacity_bytes();
        };
        // Increasing total context cannot exceed the single Vision item's 16K workspace.
        const auto at_item_limit    = capacity(16384);
        const auto above_item_limit = capacity(131072);
        // The Vision route alone (no text region beneath its handoff): the item's encode peak and
        // its handoff, what the ceiling bounds.
        const auto alone = qwen::execution::VisionContext::plan_workspace(
            *parameters.model.config().vision, *parameters.vision,
            static_cast<std::uint32_t>(qwen::kMaximumVisionItemTokens), 1);
        std::cout << "workspace at 16K " << at_item_limit << " at 128K " << above_item_limit
                  << " | Vision alone: encode peak " << alone.encode_peak_bytes << " handoff at "
                  << alone.handoff_offset_bytes << " + " << alone.handoff_capacity_bytes
                  << " capacity " << alone.capacity_bytes << '\n';
        const auto& vc = *parameters.model.config().vision;
        const auto& vp = *parameters.vision;
        const std::uint64_t merge  = std::uint64_t(vc.spatial_merge_size) * vc.spatial_merge_size;
        const std::uint64_t P      = qwen::kMaximumVisionItemTokens * merge;
        const std::uint64_t tensors =
            (2 * std::uint64_t(vc.hidden_size) + vc.intermediate_size) * P * 2;
        const auto declared = [&](const auto& linear) {
            return ninferno::ops::linear_workspace_capacity_bytes(
                linear, static_cast<std::int32_t>(merge), static_cast<std::int32_t>(P));
        };
        std::uint64_t scratch = P * 8 + declared(vp.patch_embedding);
        std::uint64_t qkv = 0, out = 0, up = 0, down = 0;
        for (const auto& layer : vp.layers) {
            qkv  = std::max<std::uint64_t>(qkv, declared(layer.qkv));
            out  = std::max<std::uint64_t>(out, declared(layer.output));
            up   = std::max<std::uint64_t>(up, declared(layer.fc1));
            down = std::max<std::uint64_t>(down, declared(layer.fc2));
        }
        scratch += qkv + out + up + down;
        constexpr std::uint64_t kAlignmentSlack = 1ULL << 20;
        std::cout << "law: tensors " << tensors << " + declared scratch " << scratch << '\n';
        if (above_item_limit != at_item_limit) {
            throw std::runtime_error(
                "Vision workspace grew with the text context: " + std::to_string(at_item_limit) +
                " bytes at a 16K context, " + std::to_string(above_item_limit) + " at 128K");
        }
        if (alone.encode_peak_bytes < tensors ||
            alone.encode_peak_bytes > tensors + scratch + kAlignmentSlack ||
            at_item_limit < alone.encode_peak_bytes) {
            throw std::runtime_error(
                "Vision workspace off its law: encode peak " +
                std::to_string(alone.encode_peak_bytes) + " bytes, tensors " +
                std::to_string(tensors) + " .. tensors + declared scratch " +
                std::to_string(tensors + scratch + kAlignmentSlack) + "; workspace " +
                std::to_string(at_item_limit));
        }
        std::cout << "Vision workspace remains bounded for long text context\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
