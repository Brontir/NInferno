#pragma once

#include "ninferno/ops/sampling.h"
#include "ninferno/types.h"
#include "runtime/contract/token_mask.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninferno::runtime {

// Per-lane token masks and the release counter the sampling kernels observe, plus per-lane
// fault flags. Model-independent; a Program owns one.
//
// The host writes a lane's rows into pinned memory after the unit is launched; release() then
// queues, on a private copy stream, the rows into device memory followed by the unit's epoch into
// the device release word. The samplers poll that word in device memory and read the rows from
// device memory: polling or reading mapped host memory from hundreds of sampler blocks costs
// hundreds of microseconds per step (PCIe round trips), device memory costs nothing measurable.
// Each unit is synchronized before the next one fills, so rows are never rewritten while a copy
// or a kernel may still read them; the copy stream is drained when a unit starts as well.
// Fault flags stay in mapped host memory: a kernel writes one only after its bounded wait
// expires. So does the progress counter the host advances after every mask row it computes: a
// kernel reads it only once its wait is long, and renews the bound while the host is working.
class TokenMaskStaging {
public:
    TokenMaskStaging(std::uint32_t lanes, std::uint32_t columns, std::uint32_t words);
    ~TokenMaskStaging();
    TokenMaskStaging(const TokenMaskStaging&)            = delete;
    TokenMaskStaging& operator=(const TokenMaskStaging&) = delete;

    [[nodiscard]] std::uint32_t columns() const noexcept { return columns_; }
    [[nodiscard]] std::uint32_t words() const noexcept { return words_; }
    [[nodiscard]] std::uint32_t* host_rows(std::uint32_t lane) const;
    [[nodiscard]] const std::uint32_t* device_rows(std::uint32_t lane) const;
    [[nodiscard]] const std::uint32_t* device_release() const noexcept { return device_release_; }
    [[nodiscard]] std::uint32_t* device_fault(std::uint32_t lane) const;
    [[nodiscard]] const std::uint32_t* device_progress() const noexcept;
    [[nodiscard]] std::uint32_t* host_progress() const noexcept;

    [[nodiscard]] std::uint32_t next_epoch() noexcept;
    // Waits for the copies of the previous release (normally long complete).
    void drain() noexcept;
    // Queue the first rows[i] rows of lanes[i] and then the epoch. False when a copy could not
    // be queued; the samplers then time out and raise their fault flags.
    bool release(std::uint32_t epoch, std::span<const std::uint32_t> lanes,
                 std::span<const std::uint32_t> rows) noexcept;
    void clear_fault(std::uint32_t lane) noexcept;
    [[nodiscard]] bool take_fault(std::uint32_t lane) noexcept;

private:
    std::uint32_t lanes_           = 0;
    std::uint32_t columns_         = 0;
    std::uint32_t words_           = 0;
    std::uint32_t epoch_           = 0;
    std::uint32_t* host_masks_     = nullptr;
    std::uint32_t* device_masks_   = nullptr; // device memory
    std::uint32_t* device_release_ = nullptr; // device memory, one word
    // Word 0: epoch copy source; words 16*(1+lane): fault flags; word 16*(1+lanes): progress
    // counter (mapped).
    std::uint32_t* host_control_   = nullptr;
    std::uint32_t* device_control_ = nullptr;
    void* copy_stream_             = nullptr; // cudaStream_t
};

// One execution unit's use of the staging. Rows are the unit's membership order.
class MaskedUnit {
public:
    // A null staging or source makes every row unconstrained and costs nothing.
    MaskedUnit(TokenMaskStaging* staging, TokenMaskSource* source,
               std::span<const std::uint32_t> lanes) noexcept;
    ~MaskedUnit();
    MaskedUnit(const MaskedUnit&)            = delete;
    MaskedUnit& operator=(const MaskedUnit&) = delete;

    [[nodiscard]] bool any() const noexcept { return any_; }
    [[nodiscard]] bool active(std::size_t row) const noexcept { return any_ && active_[row]; }

    // The sampling configuration a row's kernels read: base, plus this unit's mask for an active
    // row. Base must not carry a mask of its own.
    [[nodiscard]] ops::SamplingConfig config(std::size_t row,
                                             const ops::SamplingConfig& base) const noexcept;

    // What an active row's constraint declares about the host time of its mask rows
    // (TokenMaskSource::row_class(), plan_step_masks()); nothing measured.
    [[nodiscard]] MaskRowClass row_class(std::size_t row) const noexcept {
        return active(row) ? source_->row_class(row) : MaskRowClass::Shaped;
    }

    // Host side, after launch: compute an active row's masks. drafts.size()+1 must not exceed the
    // staging columns.
    void fill(std::size_t row, std::span<const TokenId> drafts);
    // Publish every filled row. Idempotent; the destructor releases on every other path.
    void release() noexcept;
    // After the unit is synchronized: report rows whose kernels gave up waiting.
    void collect_faults() noexcept;

private:
    TokenMaskStaging* staging_ = nullptr;
    TokenMaskSource* source_   = nullptr;
    std::span<const std::uint32_t> lanes_;
    std::array<bool, kMaximumConcurrency> active_{};
    std::array<std::uint32_t, kMaximumConcurrency> filled_rows_{};
    std::uint32_t epoch_ = 0;
    bool any_            = false;
    bool released_       = false;
};

} // namespace ninferno::runtime
