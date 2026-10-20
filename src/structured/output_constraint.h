#pragma once

#include "structured/constraint.h"

#include <memory>
#include <utility>

namespace ninferno {

// Definition of the public opaque handle returned by Engine::compile_output_constraint.
class OutputConstraint {
public:
    OutputConstraint(std::shared_ptr<const structured::CompiledConstraint> compiled,
                     const void* owner)
        : compiled_(std::move(compiled)), owner_(owner) {}

    [[nodiscard]] const std::shared_ptr<const structured::CompiledConstraint>&
    compiled() const noexcept {
        return compiled_;
    }

    // The compiler (one per Engine vocabulary) that produced this constraint.
    [[nodiscard]] const void* owner() const noexcept { return owner_; }

private:
    std::shared_ptr<const structured::CompiledConstraint> compiled_;
    const void* owner_ = nullptr;
};

} // namespace ninferno
