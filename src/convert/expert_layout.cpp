#include "convert/expert_layout.h"

#include "core/ggml_frag.h"

#include <stdexcept>
#include <string>

namespace ninferno::convert {

ExpertLayout parse_expert_layout(std::string_view name) {
    if (name == "auto") { return ExpertLayout::Auto; }
    if (name == "published") { return ExpertLayout::Published; }
    if (name == "frag") { return ExpertLayout::Frag; }
    throw std::runtime_error("expert layout must be auto, published or frag, not " +
                             std::string(name));
}

std::string_view expert_layout_name(ExpertLayout choice) noexcept {
    switch (choice) {
    case ExpertLayout::Auto: return "auto";
    case ExpertLayout::Published: return "published";
    case ExpertLayout::Frag: return "frag";
    }
    return "auto";
}

// The register-MMA expert GEMM on ggml_frag_v1 against the published layout (same kernel and
// schedule, output byte-identical) is faster, at one to sixteen tokens, for Q4_0 from K = 1024 on
// (below it the streaming split gives a warp at most two K steps, and frag gains nothing), Q4_K,
// IQ3_XXS and IQ2_S, by 1-7 %. Q6_K was slower at the expert K of 512 and has no fragment-order
// kernel; the other formats keep the published layout.
bool frag_measured_faster(QType format, std::uint64_t k) noexcept {
    switch (format) {
    case QType::GGML_Q4_0: return k >= 1024;
    case QType::GGML_Q4_K:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ2_S: return true;
    default: return false;
    }
}

QuantLayout expert_bank_layout(QType format, std::uint64_t k, ExpertLayout choice,
                               QuantLayout published) {
    if (!is_ggml_block(format) || !ggml_frag_format(format)) { return published; }
    switch (choice) {
    case ExpertLayout::Published: return published;
    case ExpertLayout::Frag: return QuantLayout::GgmlFrag;
    case ExpertLayout::Auto: break;
    }
    return frag_measured_faster(format, k) ? QuantLayout::GgmlFrag : published;
}

} // namespace ninferno::convert
