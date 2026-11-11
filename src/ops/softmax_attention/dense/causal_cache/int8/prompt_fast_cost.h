#pragma once

// The INT8-KV prompt kernel's one fitted figure, plain C++ so that the kernel's launch layout
// (int8/template_launch.cuh int8_prompt_fast_layout) and the configuration choice's estimate of it
// (product/choice/choice.cpp, choice_constants.h kAttentionHalfCta) read the same definition.
//
// Both prompt-fast shapes run one CTA per SM and every CTA of a launch sweeps a similar key range.
// A four-warp CTA, or an eight-warp CTA owning only 64 rows, sweeps in about 0.72 of a full
// eight-warp CTA's time (one RTX 5090 timing at 64K context) but covers half the rows. Fitted on
// that one board (core/fitted_holdouts.h holds its record).

namespace ninferno::ops {

// A half-height (64-row) CTA's time, in percent of a full 128-row CTA's.
inline constexpr int kInt8PromptHalfCtaPercent = 72;

} // namespace ninferno::ops
