#pragma once

// MoE expert offload (EngineOptions::moe_offload): where the routed-expert banks of MoE layers
// live, and the pointer a MoE Op call reads them through.
//
// The materializer places each offloaded layer's expert banks (one "group") in one page-aligned
// Host allocation, laid out exactly as the device weight arena would hold them, and points the
// weights at it. Before anything runs, resolve() decides per group:
//   - resident: copied to one Device allocation (the Host copy is freed);
//   - offloaded: pinned in place (cudaHostRegister, mapped), so a kernel reads it over PCIe
//     through its device alias (zero-copy), or a whole group is streamed into one of two Device
//     staging slots (double buffered on a copy stream) for calls that read most of its experts.
// A MoE call binds its weights first (bind()): the Lease maps every pointer inside the group to
// the bytes the call reads. The bytes are the same in every placement and the Op is the same, so
// the result is too: offload changes where weights are read from, never what is computed.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace ninferno {

class ExpertResidency {
public:
    // A group's bytes are aligned to this in every placement (Host, resident, staging), so every
    // object keeps its alignment relative to the group base.
    static constexpr std::uint64_t kGroupAlignment = 4096;

    ExpertResidency() = default;
    ~ExpertResidency();
    ExpertResidency(const ExpertResidency&)            = delete;
    ExpertResidency& operator=(const ExpertResidency&) = delete;

    // Materialization: a new group of `bytes` (named for messages, e.g. "text/layers/3"); returns
    // its page-aligned Host base, which the caller fills before resolve().
    std::byte* add_group(std::string name, std::uint64_t bytes);

    [[nodiscard]] std::size_t group_count() const noexcept { return groups_.size(); }
    [[nodiscard]] std::uint64_t group_bytes(std::size_t group) const;
    // Bytes of every group together.
    [[nodiscard]] std::uint64_t total_bytes() const noexcept;

    // The Device bytes resolve(offloaded) allocates: the resident groups (groups [offloaded,
    // count)) and, when anything is offloaded, the two staging slots of the largest offloaded
    // group; each allocation rounded to kGroupAlignment.
    [[nodiscard]] std::uint64_t device_bytes(std::size_t offloaded) const;
    // The Host bytes resolve(offloaded) pins: groups [0, offloaded).
    [[nodiscard]] std::uint64_t pinned_bytes(std::size_t offloaded) const;
    [[nodiscard]] std::uint64_t staging_bytes(std::size_t offloaded) const;

    // Places the groups once: [0, offloaded) stay in Host memory, pinned and mapped (counted,
    // core/host_memory.h count_pinned_host); the rest are copied to Device memory and their Host
    // copies freed. Throws, with the numbers, when a Device allocation fails. No bind() before it.
    void resolve(std::size_t offloaded, int device);

    // Whether `pointer` lies in an offloaded group: its bytes are in Host memory at that very
    // address, which the CPU may read (MoE expert offload's CPU decode path). False before
    // resolve() and for resident groups, whose Host copies are gone.
    [[nodiscard]] bool in_host(const void* pointer) const;

    // What one MoE call reads a group's bytes through. Pointers outside the group are unchanged.
    // A Lease on a staging slot keeps the slot until it ends; its end is recorded on the call's
    // stream, after which the slot may be refilled.
    class Lease {
    public:
        Lease() = default;
        ~Lease();
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&&) = delete;
        Lease(const Lease&)       = delete;

        template <class T>
        [[nodiscard]] T* map(T* pointer) const noexcept {
            const auto address = reinterpret_cast<std::uintptr_t>(pointer);
            if (pointer == nullptr || address < begin_ || address - begin_ >= bytes_) {
                return pointer;
            }
            return reinterpret_cast<T*>(target_ + (address - begin_));
        }

        // Where the call reads the group: resident, zero-copy (Host), or a staging slot.
        enum class Source : std::uint8_t { Unmanaged, Resident, Host, Staged };
        [[nodiscard]] Source source() const noexcept { return source_; }

    private:
        friend class ExpertResidency;

        ExpertResidency* owner_ = nullptr;
        std::uintptr_t begin_   = 0;
        std::uint64_t bytes_    = 0;
        std::uintptr_t target_  = 0;
        int slot_               = -1;
        cudaStream_t stream_    = nullptr;
        Source source_          = Source::Unmanaged;
    };

    // The Lease for the group holding `pointer` (any pointer into a weight of the group; another
    // pointer gives an Unmanaged Lease, which maps nothing). `whole_group`: the call reads most
    // of the group's experts, so an offloaded group is streamed into a staging slot rather than
    // read over PCIe per access; ignored while `stream` is capturing a graph (a captured call
    // reads resident or Host bytes only, which do not move between replays).
    [[nodiscard]] Lease bind(const void* pointer, bool whole_group, cudaStream_t stream);

private:
    struct Group {
        std::string name;
        std::uint64_t bytes    = 0;
        std::uintptr_t begin   = 0;       // the Host address the weights were bound to
        std::byte* host        = nullptr; // pageable until resolve(); pinned when offloaded
        std::uintptr_t alias   = 0;       // the device address of the pinned Host bytes
        std::uintptr_t device  = 0;       // resident copy; 0 when offloaded
        bool registered        = false;
    };

    struct Slot {
        std::uintptr_t base = 0;
        std::optional<std::size_t> group;
        cudaEvent_t copied   = nullptr; // on the copy stream, after the group's bytes landed
        cudaEvent_t released = nullptr; // on the last Lease's stream, after its reads
        bool leased          = false;
    };

    [[nodiscard]] const Group* find(std::uintptr_t address, std::size_t& index) const;
    void fill(std::size_t slot, std::size_t group);
    void release(int slot, cudaStream_t stream) noexcept;
    void destroy() noexcept;

    std::vector<Group> groups_;
    bool resolved_          = false;
    std::size_t offloaded_  = 0;
    void* resident_         = nullptr; // one Device allocation for every resident group
    void* staging_          = nullptr; // one Device allocation for both slots
    cudaStream_t copy_stream_ = nullptr;
    Slot slots_[2];
    int last_slot_ = 1;
    std::mutex mutex_;
};

} // namespace ninferno
