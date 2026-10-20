#include "runtime/contract/token_mask_staging.h"

#include "core/cuda_error.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <string>

namespace ninferno::runtime {
namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) { throw std::runtime_error(cuda_error_message(operation, status)); }
}

std::uint32_t* allocate_mapped(std::size_t words, std::uint32_t** device) {
    void* host = nullptr;
    check(cudaHostAlloc(&host, words * sizeof(std::uint32_t),
                        cudaHostAllocMapped | cudaHostAllocPortable),
          "token mask staging cudaHostAlloc");
    void* mapped = nullptr;
    const cudaError_t status = cudaHostGetDevicePointer(&mapped, host, 0);
    if (status != cudaSuccess) {
        cudaFreeHost(host);
        check(status, "token mask staging cudaHostGetDevicePointer");
    }
    std::fill_n(static_cast<std::uint32_t*>(host), words, 0U);
    *device = static_cast<std::uint32_t*>(mapped);
    return static_cast<std::uint32_t*>(host);
}

cudaStream_t as_stream(void* stream) { return static_cast<cudaStream_t>(stream); }

} // namespace

TokenMaskStaging::TokenMaskStaging(std::uint32_t lanes, std::uint32_t columns, std::uint32_t words)
    : lanes_(lanes), columns_(columns), words_(words) {
    if (lanes == 0 || lanes > kMaximumConcurrency || columns == 0 || words == 0 ||
        words % 8U != 0U) {
        throw std::invalid_argument("token mask staging geometry is invalid");
    }
    const std::size_t mask_words = static_cast<std::size_t>(lanes) * columns * words;
    try {
        std::uint32_t* unused = nullptr;
        host_masks_           = allocate_mapped(mask_words, &unused);
        // Control words live on their own 64-byte lines, apart from mask rows.
        host_control_ = allocate_mapped(16U * (2U + lanes), &device_control_);
        void* device  = nullptr;
        check(cudaMalloc(&device, mask_words * sizeof(std::uint32_t)),
              "token mask staging cudaMalloc rows");
        device_masks_ = static_cast<std::uint32_t*>(device);
        check(cudaMalloc(&device, 64), "token mask staging cudaMalloc release");
        device_release_ = static_cast<std::uint32_t*>(device);
        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "token mask staging copy stream");
        copy_stream_ = stream;
        // The release word is zeroed on the copy stream that later publishes epochs into it, and
        // completed before construction returns: the samplers poll it from other streams, which
        // nothing would order after this fill otherwise.
        check(cudaMemsetAsync(device_release_, 0, 64, stream), "token mask staging release reset");
        check(cudaStreamSynchronize(stream), "token mask staging release reset settle");
    } catch (...) {
        if (copy_stream_ != nullptr) { cudaStreamDestroy(as_stream(copy_stream_)); }
        if (device_release_ != nullptr) { cudaFree(device_release_); }
        if (device_masks_ != nullptr) { cudaFree(device_masks_); }
        if (host_control_ != nullptr) { cudaFreeHost(host_control_); }
        if (host_masks_ != nullptr) { cudaFreeHost(host_masks_); }
        throw;
    }
}

TokenMaskStaging::~TokenMaskStaging() {
    cudaStreamSynchronize(as_stream(copy_stream_));
    cudaStreamDestroy(as_stream(copy_stream_));
    cudaFree(device_release_);
    cudaFree(device_masks_);
    cudaFreeHost(host_control_);
    cudaFreeHost(host_masks_);
}

std::uint32_t* TokenMaskStaging::host_rows(std::uint32_t lane) const {
    if (lane >= lanes_) { throw std::out_of_range("token mask lane is out of range"); }
    return host_masks_ + static_cast<std::size_t>(lane) * columns_ * words_;
}

const std::uint32_t* TokenMaskStaging::device_rows(std::uint32_t lane) const {
    if (lane >= lanes_) { throw std::out_of_range("token mask lane is out of range"); }
    return device_masks_ + static_cast<std::size_t>(lane) * columns_ * words_;
}

std::uint32_t* TokenMaskStaging::device_fault(std::uint32_t lane) const {
    if (lane >= lanes_) { throw std::out_of_range("token mask lane is out of range"); }
    return device_control_ + 16U * (1U + lane);
}

const std::uint32_t* TokenMaskStaging::device_progress() const noexcept {
    return device_control_ + 16U * (1U + lanes_);
}

std::uint32_t* TokenMaskStaging::host_progress() const noexcept {
    return host_control_ + 16U * (1U + lanes_);
}

std::uint32_t TokenMaskStaging::next_epoch() noexcept {
    // Epochs compare by signed difference on the device, so wraparound is harmless.
    ++epoch_;
    return epoch_;
}

void TokenMaskStaging::drain() noexcept { (void)cudaStreamSynchronize(as_stream(copy_stream_)); }

bool TokenMaskStaging::release(std::uint32_t epoch, std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> rows) noexcept {
    const cudaStream_t stream = as_stream(copy_stream_);
    bool queued               = true;
    for (std::size_t index = 0; index < lanes.size() && index < rows.size(); ++index) {
        if (rows[index] == 0U || lanes[index] >= lanes_) { continue; }
        const std::size_t offset = static_cast<std::size_t>(lanes[index]) * columns_ * words_;
        const std::size_t bytes =
            static_cast<std::size_t>(std::min(rows[index], columns_)) * words_ * sizeof(std::uint32_t);
        queued = queued && cudaMemcpyAsync(device_masks_ + offset, host_masks_ + offset, bytes,
                                           cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }
    if (!queued) { return false; }
    // The epoch is copied after the rows on the same stream, so a sampler that observes it reads
    // complete rows. host_control_[0] stays unchanged until this copy is drained.
    host_control_[0] = epoch;
    return cudaMemcpyAsync(device_release_, host_control_, sizeof(std::uint32_t),
                           cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

void TokenMaskStaging::clear_fault(std::uint32_t lane) noexcept {
    std::atomic_ref<std::uint32_t>(host_control_[16U * (1U + lane)])
        .store(0U, std::memory_order_relaxed);
}

bool TokenMaskStaging::take_fault(std::uint32_t lane) noexcept {
    return std::atomic_ref<std::uint32_t>(host_control_[16U * (1U + lane)])
               .exchange(0U, std::memory_order_acquire) != 0U;
}

MaskedUnit::MaskedUnit(TokenMaskStaging* staging, TokenMaskSource* source,
                       std::span<const std::uint32_t> lanes) noexcept
    : staging_(staging), source_(source), lanes_(lanes) {
    if (staging_ == nullptr || source_ == nullptr) { return; }
    for (std::size_t row = 0; row < lanes_.size() && row < active_.size(); ++row) {
        active_[row] = source_->constrained(row);
        any_         = any_ || active_[row];
    }
    if (!any_) { return; }
    staging_->drain();
    epoch_ = staging_->next_epoch();
    for (std::size_t row = 0; row < lanes_.size(); ++row) {
        if (active_[row]) { staging_->clear_fault(lanes_[row]); }
    }
}

MaskedUnit::~MaskedUnit() { release(); }

ops::SamplingConfig MaskedUnit::config(std::size_t row,
                                       const ops::SamplingConfig& base) const noexcept {
    ops::SamplingConfig out = base;
    if (!active(row)) {
        out.token_mask          = nullptr;
        out.token_mask_release  = nullptr;
        out.token_mask_fault    = nullptr;
        out.token_mask_progress = nullptr;
        out.token_mask_epoch    = 0;
        out.token_mask_words    = 0;
        return out;
    }
    const std::uint32_t lane = lanes_[row];
    out.token_mask           = staging_->device_rows(lane);
    out.token_mask_release   = staging_->device_release();
    out.token_mask_fault     = staging_->device_fault(lane);
    out.token_mask_progress  = staging_->device_progress();
    out.token_mask_epoch     = epoch_;
    out.token_mask_words     = static_cast<std::int32_t>(staging_->words());
    return out;
}

void MaskedUnit::fill(std::size_t row, std::span<const TokenId> drafts) {
    if (!active(row)) { return; }
    if (drafts.size() + 1U > staging_->columns()) {
        throw std::logic_error("token mask draft window exceeds the staging columns");
    }
    source_->fill(row, drafts, staging_->host_rows(lanes_[row]), staging_->words(),
                  staging_->host_progress());
    filled_rows_[row] = static_cast<std::uint32_t>(drafts.size() + 1U);
}

void MaskedUnit::release() noexcept {
    if (!any_ || released_) { return; }
    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    std::size_t count = 0;
    for (std::size_t row = 0; row < lanes_.size() && row < active_.size(); ++row) {
        lanes[row] = lanes_[row];
        count      = row + 1;
    }
    // A failed queue leaves the samplers to time out; collect_faults() then fails the rows.
    (void)staging_->release(epoch_, std::span<const std::uint32_t>(lanes.data(), count),
                            std::span<const std::uint32_t>(filled_rows_.data(), count));
    released_ = true;
}

void MaskedUnit::collect_faults() noexcept {
    if (!any_) { return; }
    for (std::size_t row = 0; row < lanes_.size(); ++row) {
        if (active_[row] && staging_->take_fault(lanes_[row])) { source_->fault(row); }
    }
}

} // namespace ninferno::runtime
