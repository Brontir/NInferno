#include "core/expert_residency.h"

#include "core/cuda_error.h"
#include "core/sys.h"

#include "core/checked.h"
#include "core/host_memory.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ninferno {
namespace {

std::uint64_t round_up(std::uint64_t bytes) {
    return align_up(bytes, ExpertResidency::kGroupAlignment, "MoE expert offload group");
}

void check(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message(("MoE expert offload: " + operation).c_str(), status));
    }
}

void log_cleanup(const char* operation, cudaError_t status) noexcept {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "MoE expert offload cleanup failed during %s: %s\n", operation,
                     cudaGetErrorString(status));
    }
}

void* device_alloc(std::uint64_t bytes, const std::string& what) {
    void* pointer = nullptr;
    const cudaError_t status = cudaMalloc(&pointer, static_cast<std::size_t>(bytes));
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        std::size_t free_bytes  = 0;
        std::size_t total_bytes = 0;
        (void)cudaMemGetInfo(&free_bytes, &total_bytes);
        throw std::runtime_error("MoE expert offload: " + what + " needs " + format_gib(bytes) +
                                 " of Device memory, " + format_gib(free_bytes) + " is free (" +
                                 cudaGetErrorName(status) + ")");
    }
    return pointer;
}

} // namespace

ExpertResidency::~ExpertResidency() { destroy(); }

void ExpertResidency::destroy() noexcept {
    if (copy_stream_) { log_cleanup("synchronize", cudaStreamSynchronize(copy_stream_)); }
    for (auto& slot : slots_) {
        if (slot.copied) { log_cleanup("cudaEventDestroy", cudaEventDestroy(slot.copied)); }
        if (slot.released) {
            log_cleanup("cudaEventSynchronize", cudaEventSynchronize(slot.released));
            log_cleanup("cudaEventDestroy", cudaEventDestroy(slot.released));
        }
        slot = {};
    }
    if (copy_stream_) { log_cleanup("cudaStreamDestroy", cudaStreamDestroy(copy_stream_)); }
    copy_stream_ = nullptr;
    if (staging_) { log_cleanup("cudaFree", cudaFree(staging_)); }
    if (resident_) { log_cleanup("cudaFree", cudaFree(resident_)); }
    staging_  = nullptr;
    resident_ = nullptr;
    for (auto& group : groups_) {
        if (group.registered) {
            log_cleanup("cudaHostUnregister", cudaHostUnregister(group.host));
            uncount_pinned_host(round_up(group.bytes));
        }
        sys::aligned_free(group.host);
        group.host       = nullptr;
        group.registered = false;
    }
}

std::byte* ExpertResidency::add_group(std::string name, std::uint64_t bytes) {
    if (resolved_) { throw std::logic_error("MoE expert offload: a group after resolve()"); }
    if (bytes == 0) { throw std::invalid_argument("MoE expert offload: an empty group"); }
    void* host = sys::aligned_alloc(kGroupAlignment, static_cast<std::size_t>(round_up(bytes)));
    if (!host) {
        throw std::runtime_error("MoE expert offload: cannot allocate " + format_gib(bytes) +
                                 " of Host memory for " + name);
    }
    groups_.push_back({.name  = std::move(name),
                       .bytes = bytes,
                       .begin = reinterpret_cast<std::uintptr_t>(host),
                       .host  = static_cast<std::byte*>(host)});
    return groups_.back().host;
}

std::uint64_t ExpertResidency::group_bytes(std::size_t group) const {
    return groups_.at(group).bytes;
}

std::uint64_t ExpertResidency::total_bytes() const noexcept {
    std::uint64_t total = 0;
    for (const auto& group : groups_) { total += group.bytes; }
    return total;
}

std::uint64_t ExpertResidency::staging_bytes(std::size_t offloaded) const {
    std::uint64_t largest = 0;
    for (std::size_t i = 0; i < std::min(offloaded, groups_.size()); ++i) {
        largest = std::max(largest, groups_[i].bytes);
    }
    return 2 * round_up(largest);
}

std::uint64_t ExpertResidency::device_bytes(std::size_t offloaded) const {
    std::uint64_t resident = 0;
    for (std::size_t i = offloaded; i < groups_.size(); ++i) { resident += round_up(groups_[i].bytes); }
    return resident + staging_bytes(offloaded);
}

std::uint64_t ExpertResidency::pinned_bytes(std::size_t offloaded) const {
    std::uint64_t pinned = 0;
    for (std::size_t i = 0; i < std::min(offloaded, groups_.size()); ++i) {
        pinned += groups_[i].bytes;
    }
    return pinned;
}

void ExpertResidency::resolve(std::size_t offloaded, int device) {
    if (resolved_) { throw std::logic_error("MoE expert offload: resolved twice"); }
    if (offloaded > groups_.size()) {
        throw std::invalid_argument("MoE expert offload: more groups offloaded than exist");
    }
    check(cudaSetDevice(device), "cudaSetDevice");
    check(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking),
          "create the expert copy stream");
    // Resident groups: one Device allocation, each group at a kGroupAlignment offset, copied
    // from its Host bytes, which are then freed.
    std::uint64_t resident_bytes = 0;
    for (std::size_t i = offloaded; i < groups_.size(); ++i) { resident_bytes += round_up(groups_[i].bytes); }
    if (resident_bytes != 0) {
        resident_ = device_alloc(resident_bytes, std::to_string(groups_.size() - offloaded) +
                                                     " resident MoE layers' experts");
        std::uintptr_t at = reinterpret_cast<std::uintptr_t>(resident_);
        for (std::size_t i = offloaded; i < groups_.size(); ++i) {
            auto& group = groups_[i];
            check(cudaMemcpyAsync(reinterpret_cast<void*>(at), group.host,
                                  static_cast<std::size_t>(group.bytes), cudaMemcpyHostToDevice,
                                  copy_stream_),
                  "copy " + group.name + " to Device memory");
            group.device = at;
            at += round_up(group.bytes);
        }
        // Pageable sources: the copies have read them once the stream is done.
        check(cudaStreamSynchronize(copy_stream_), "complete the resident expert copies");
        for (std::size_t i = offloaded; i < groups_.size(); ++i) {
            sys::aligned_free(groups_[i].host);
            groups_[i].host = nullptr;
        }
    }
    if (offloaded != 0) {
        // The materializer checked every group against the pinning limit before loading them
        // (they are this process's memory already); pinning them is counted against the cap.
        for (std::size_t i = 0; i < offloaded; ++i) {
            auto& group = groups_[i];
            check(cudaHostRegister(group.host, static_cast<std::size_t>(round_up(group.bytes)),
                                   cudaHostRegisterMapped | cudaHostRegisterPortable),
                  "pin " + group.name + " (" + format_gib(group.bytes) + ")");
            group.registered = true;
            count_pinned_host(round_up(group.bytes));
            void* alias      = nullptr;
            check(cudaHostGetDevicePointer(&alias, group.host, 0),
                  "map " + group.name + " to the device");
            group.alias = reinterpret_cast<std::uintptr_t>(alias);
        }
        const std::uint64_t slot_bytes = staging_bytes(offloaded) / 2;
        staging_ = device_alloc(2 * slot_bytes, "the two MoE expert staging slots");
        for (int s = 0; s < 2; ++s) {
            slots_[s].base = reinterpret_cast<std::uintptr_t>(staging_) + s * slot_bytes;
            check(cudaEventCreateWithFlags(&slots_[s].copied, cudaEventDisableTiming),
                  "create a staging event");
            check(cudaEventCreateWithFlags(&slots_[s].released, cudaEventDisableTiming),
                  "create a staging event");
        }
    }
    offloaded_ = offloaded;
    resolved_  = true;
}

const ExpertResidency::Group* ExpertResidency::find(std::uintptr_t address,
                                                    std::size_t& index) const {
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        // Resident groups' Host bytes are freed, but the weights still hold those addresses:
        // the group's Host range is its identity. It is recorded before the free.
        const auto begin = groups_[i].begin;
        if (address >= begin && address - begin < groups_[i].bytes) {
            index = i;
            return &groups_[i];
        }
    }
    return nullptr;
}

bool ExpertResidency::in_host(const void* pointer) const {
    if (!resolved_ || pointer == nullptr) { return false; }
    std::size_t index = 0;
    const Group* group = find(reinterpret_cast<std::uintptr_t>(pointer), index);
    return group != nullptr && index < offloaded_ &&
           reinterpret_cast<std::uintptr_t>(group->host) == group->begin;
}

void ExpertResidency::fill(std::size_t slot, std::size_t group) {
    auto& s = slots_[slot];
    // The slot's previous reads end before the copy starts (its last Lease recorded `released`
    // on its own stream; an event never recorded is complete).
    check(cudaStreamWaitEvent(copy_stream_, s.released, 0), "order a staging copy");
    const auto& g = groups_[group];
    check(cudaMemcpyAsync(reinterpret_cast<void*>(s.base), g.host,
                          static_cast<std::size_t>(g.bytes), cudaMemcpyHostToDevice, copy_stream_),
          "stream " + g.name + " to a staging slot");
    check(cudaEventRecord(s.copied, copy_stream_), "record a staging copy");
    s.group = group;
}

ExpertResidency::Lease ExpertResidency::bind(const void* pointer, bool whole_group,
                                             cudaStream_t stream) {
    if (!resolved_) { throw std::logic_error("MoE expert offload: bind() before resolve()"); }
    Lease lease;
    std::size_t index  = 0;
    const Group* group = find(reinterpret_cast<std::uintptr_t>(pointer), index);
    if (!group) { return lease; }
    lease.begin_ = group->begin;
    lease.bytes_ = group->bytes;
    if (group->device != 0) {
        lease.target_ = group->device;
        lease.source_ = Lease::Source::Resident;
        return lease;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(stream, &capture), "query graph capture");
    if (!whole_group || capture != cudaStreamCaptureStatusNone) {
        lease.target_ = group->alias;
        lease.source_ = Lease::Source::Host;
        return lease;
    }
    std::lock_guard lock(mutex_);
    int slot = slots_[0].group == index ? 0 : slots_[1].group == index ? 1 : -1;
    if (slot < 0) {
        slot = 1 - last_slot_;
        if (slots_[slot].leased) { slot = last_slot_; }
        if (slots_[slot].leased) {
            throw std::logic_error("MoE expert offload: both staging slots are leased");
        }
        fill(static_cast<std::size_t>(slot), index);
    }
    // Double buffering: the next offloaded group streams into the other slot while this one
    // is read (a guess only: a group found in no slot is streamed when it is bound).
    const std::size_t next  = (index + 1) % offloaded_;
    const int other         = 1 - slot;
    if (next != index && slots_[other].group != next && !slots_[other].leased) {
        fill(static_cast<std::size_t>(other), next);
    }
    check(cudaStreamWaitEvent(stream, slots_[slot].copied, 0), "wait for a staging copy");
    slots_[slot].leased = true;
    last_slot_          = slot;
    lease.owner_        = this;
    lease.slot_         = slot;
    lease.stream_       = stream;
    lease.target_       = slots_[slot].base;
    lease.source_       = Lease::Source::Staged;
    return lease;
}

void ExpertResidency::release(int slot, cudaStream_t stream) noexcept {
    std::lock_guard lock(mutex_);
    log_cleanup("record a staging release", cudaEventRecord(slots_[slot].released, stream));
    slots_[slot].leased = false;
}

ExpertResidency::Lease::~Lease() {
    if (owner_ && slot_ >= 0) { owner_->release(slot_, stream_); }
}

ExpertResidency::Lease::Lease(Lease&& other) noexcept
    : owner_(other.owner_), begin_(other.begin_), bytes_(other.bytes_), target_(other.target_),
      slot_(other.slot_), stream_(other.stream_), source_(other.source_) {
    other.owner_ = nullptr;
    other.slot_  = -1;
}

} // namespace ninferno
