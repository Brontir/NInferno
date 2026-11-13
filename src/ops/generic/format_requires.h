#pragma once

// The arch requirements of the generic path's format predicates (#615 F5), declared once each over
// the discrete facts of core/arch_requires.h. A predicate reads its format's row
// (format_traits.h), what this binary compiles (compiled_codecs.h) and one of these, resolved from
// the board's facts (current_arch_facts, generic_select_detail.h); it queries no device field of
// its own. Each says what kind of knowledge it is, as the format rows do.

#include "core/arch_requires.h"

#include <array>
#include <string_view>

namespace ninferno::ops::detail::generic::format_requires {

// [measured rule] The architectures where a measured dense-tune choice may move a w4a8_ws_tuned
// format to the producer-warp tile (w4a8_ws_extra): sm_89 and sm_120, exactly. Evidence: product,
// 24K prefill, A B B A, greedy byte-identical: RTX 5060 Ti 981 -> 1011 t/s, RTX 4060 Ti 900 -> 901
// (926 with IQ3_S on its shared-table codec); generic_select.cpp has the rest. Other architectures
// were not measured and keep the classic tiles.
inline constexpr ArchRequires kW4a8WsTunedArch{.cc_any_of = {{{8, 9}, {12, 0}}}};

// [measured rule, on facts] Where int8 activations pay at prefill for the Q4/Q5/Q6 G64 weights
// (qx_prefill_a8_pays, generic_gemm.cpp): tensor mma.sync in this build, a board with tensor cores
// (not mma_slow, as for ggml's mmq), and a generation whose FP32-accumulate HMMA runs below full
// rate. A generation without a row of its own has no measured kind and keeps A16 (#300).
inline constexpr ArchRequires kQxPrefillA8Pays{
    .mma_tensor = Need::Yes, .mma_slow = Need::No, .fp32acc_full_rate = Need::No};

// [measured rule] The R8 decode rules of a card without BF16 operands in its MMA (rules_r8: Turing
// and Volta): RTX 2080 Ti, ninferno_r8_schedule_bench on the IQ3 16 GB artifacts' 63 problems.
inline constexpr ArchRequires kR8WithoutNativeBf16{.mma_bf16_native = Need::No};

// [compiled fact] The R8 kernel's one-slot form: an sm_70 tree (generic_r8.cuh
// NINFERNO_R8_ONE_SLOT; a tree compiles one architecture) on a GPU without tensor mma.sync. Its
// Q6_K rule (rules_r8) is measured on the GV100.
inline constexpr ArchRequires kR8OneSlot{.mma_tensor = Need::No, .build_arch_below = 75};

struct NamedRequires {
    std::string_view name;
    const ArchRequires* arch;
};

// Every requirement above, for the tests and a listing.
inline constexpr std::array kDeclared{
    NamedRequires{"w4a8_ws_tuned_arch", &kW4a8WsTunedArch},
    NamedRequires{"qx_prefill_a8_pays", &kQxPrefillA8Pays},
    NamedRequires{"r8_without_native_bf16", &kR8WithoutNativeBf16},
    NamedRequires{"r8_one_slot", &kR8OneSlot},
};

} // namespace ninferno::ops::detail::generic::format_requires
