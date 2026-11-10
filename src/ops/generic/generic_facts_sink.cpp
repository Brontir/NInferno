// The register-MMA fact reads a selector test records (ScopedRmmaFactsSink, #612): the part of
// generic_describe.cpp that needs no kernel launch, so it builds into ninferno_ops_host (#620).

#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"

#include <vector>

namespace ninferno::ops::detail::generic {

namespace {
thread_local std::vector<RmmaFactRead>* t_fact_sink = nullptr;
} // namespace

ScopedRmmaFactsSink::ScopedRmmaFactsSink(std::vector<RmmaFactRead>& sink) noexcept
    : previous_(t_fact_sink) {
    t_fact_sink = &sink;
}

ScopedRmmaFactsSink::~ScopedRmmaFactsSink() { t_fact_sink = previous_; }

void note_rmma_fact_read(const RmmaFactRead& read) noexcept {
    // A sink that cannot grow loses the read (the test then fails on its digest); pricing goes on.
    try {
        if (t_fact_sink != nullptr) t_fact_sink->push_back(read);
    } catch (...) {}
}

} // namespace ninferno::ops::detail::generic
