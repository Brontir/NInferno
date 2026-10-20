#pragma once

// Recipe.prepare: turn a Model and its Recipe into ordered artifact objects (parents), the
// parameter bindings, the Use records and their auxiliary values, with payload offsets fixed.

#include "convert/expert_layout.h"
#include "convert/model.h"
#include "core/weight_view.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ninferno::convert {

struct WeightJob {
    std::string id;
    Shape shape;
    QType format           = QType::BF16;
    QuantLayout layout     = QuantLayout::Contiguous;
    std::uint64_t divisors = 1;
    Method method          = Method::CastDirect;
    std::vector<std::size_t> parameters;
    std::vector<SourcePtr> sources; // one per parameter, rows stacked in this order
    WeightGeometry geometry;
    std::uint64_t offset = 0;                 // payload-relative
    std::vector<std::uint32_t> divisor_words; // NVFP4: one per divisor
    KOrder order; // RowGroupZpOrdered: the plane's column order (every source's)
};

struct AuxiliaryJob {
    std::string id;
    std::uint32_t word   = 0;
    std::uint64_t offset = 0;
};

struct ResourceJob {
    std::string id;
    const std::string* bytes = nullptr;
    std::uint64_t offset     = 0;
};

struct Plan {
    std::vector<ResourceJob> resources;
    std::vector<WeightJob> weights;
    std::vector<AuxiliaryJob> auxiliaries;
    OJson objects               = OJson::array();
    OJson bindings              = OJson::object();
    OJson uses                  = OJson::array();
    std::uint64_t payload_bytes = 0;
};

// `expert_layout`: the layout of the routed-expert banks of ggml block formats
// (convert/expert_layout.h).
[[nodiscard]] Plan prepare(const Model& model, const Recipe& recipe,
                           ExpertLayout expert_layout = ExpertLayout::Auto);

} // namespace ninferno::convert
