#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <array>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno {

struct LayoutRegion {
    std::size_t offset    = 0;
    std::size_t bytes     = 0;
    std::size_t alignment = 1;

    [[nodiscard]] DeviceSpan bind(DeviceSpan backing) const;
};

struct TensorRegion {
    LayoutRegion region;
    DType dtype = DType::BF16;
    std::array<std::int32_t, 4> shape{1, 1, 1, 1};

    [[nodiscard]] Tensor bind(DeviceSpan backing) const;
};

// Checked offset builder for project-owned backing allocations. Model-specific code owns the
// region list; this mechanism only guarantees that the total and every later binding share the
// exact offsets produced here.
class LayoutBuilder {
public:
    class Scope {
    public:
        ~Scope() noexcept;
        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        friend class LayoutBuilder;
        explicit Scope(LayoutBuilder& builder) noexcept;
        LayoutBuilder* builder_   = nullptr;
        std::size_t saved_cursor_ = 0;
    };

    [[nodiscard]] LayoutRegion add(std::size_t bytes, std::size_t alignment,
                                   std::string_view label);
    [[nodiscard]] TensorRegion add_tensor(DType dtype, std::initializer_list<std::int32_t> shape,
                                          std::size_t alignment, std::string_view label);
    [[nodiscard]] Scope scope() noexcept;
    [[nodiscard]] std::size_t finish(std::size_t alignment  = 1,
                                     std::string_view label = "layout") const;

private:
    std::size_t cursor_ = 0;
    std::size_t peak_   = 0;
};

// A region a dry-run builder handed out (#615 W5): the op that took it and its bytes in the arena.
// Regions of one scope never overlap; a scope's regions are reused by the next scope at the same
// offsets, which is what a check of concurrent launches reads (ops/generic/scratch_lifetime.h).
struct WorkspaceRegion {
    std::string op;
    std::size_t offset = 0;
    std::size_t bytes  = 0;
    bool scratch       = false; // an op's raw scratch (alloc_bytes), not a tensor (alloc)
};

// Dry-run counterpart of WorkspaceArena. Target allocation helpers can run against this builder
// and the real arena, including identical nested scope lifetimes, without maintaining byte
// formulas.
class WorkspaceLayoutBuilder {
public:
    class Scope {
    public:
        ~Scope() noexcept;
        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        friend class WorkspaceLayoutBuilder;
        explicit Scope(WorkspaceLayoutBuilder& builder) noexcept;
        WorkspaceLayoutBuilder* builder_ = nullptr;
        std::size_t saved_cursor_        = 0;
    };

    [[nodiscard]] Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape,
                               std::size_t alignment = 256);
    [[nodiscard]] DeviceSpan alloc_bytes(std::size_t bytes, std::size_t alignment = 256);
    [[nodiscard]] Scope scope() noexcept;
    [[nodiscard]] std::size_t peak_bytes(std::size_t alignment = 256) const;

    // Records every region handed out from here on into `sink` (nullptr: stops), under the op the
    // caller names last (`op`). Recording changes no offset and no peak.
    void record(std::vector<WorkspaceRegion>* sink) noexcept { sink_ = sink; }

    void op(std::string name) { op_ = std::move(name); }

private:
    void note(std::size_t offset, std::size_t bytes, bool scratch);

    std::size_t cursor_                 = 0;
    std::size_t peak_                   = 0;
    std::vector<WorkspaceRegion>* sink_ = nullptr;
    std::string op_;
};

} // namespace ninferno
