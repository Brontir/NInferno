#pragma once

#include "core/dtype.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <utility>
#include <stdexcept>
#include <string>

namespace ninferno {

// The ints of an owning DeviceArena's ticket page (#889): last-block tickets of the ops that run
// on the arena, outside its scoped region and its capacity(). Each ticket returns to zero when its
// grid ends, so a page zeroed once at construction stays valid across calls, layers and graph
// replays in the arena's stream order. Two layers share a slot: each leaves it at zero.
inline constexpr std::size_t kArenaTicketInts = 1056;

// The slots of the page: the generic MoE route ticket, the sparse MoE's D1 and S1 tickets, and
// from kMoeCombine the fused combine's run tickets (ops/moe kCombineTickets of them).
namespace arena_tickets {
inline constexpr int kMoeRoute   = 0;
inline constexpr int kSparseD1   = 1;
inline constexpr int kSparseS1   = 2;
inline constexpr int kMoeCombine = 32;
} // namespace arena_tickets

struct DeviceSpan {
    void* data        = nullptr;
    std::size_t bytes = 0;
    // The ticket page of the arena the span was carved from (nullptr: none): an arena over the span
    // inherits it (DeviceArena(DeviceSpan)).
    int* ticket_page = nullptr;
};

// Owning device allocation for long-lived buffers. DeviceArena remains the
// suballocation primitive for workspaces; this type owns exactly one cudaMalloc.
class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    explicit DeviceBuffer(std::size_t size_bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void fill(int byte_value = 0);
    // Completes the upload before returning. Callers must first order any prior device
    // accesses to the destination range.
    void copy_from_host(const void* source, std::size_t count, std::size_t byte_offset = 0);
    void copy_to_host(void* destination, std::size_t count, std::size_t byte_offset = 0) const;

    // Raw access is intentional: Tensor and Weight are non-owning views.
    void* p           = nullptr;
    std::size_t bytes = 0;

private:
    void require_range(std::size_t byte_offset, std::size_t count, const char* operation) const;
};

class DeviceArena {
public:
    class Scope {
    public:
        ~Scope() noexcept;

        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        friend class DeviceArena;

        explicit Scope(DeviceArena& arena) noexcept;

        DeviceArena* arena_       = nullptr;
        std::size_t saved_offset_ = 0;
        // The arena state the scope was opened in: after a reset() the saved offset names
        // another generation's layout, so the scope no longer rolls back.
        std::uint64_t saved_generation_ = 0;
    };

    // Allocates capacity_bytes and the ticket page (zeroed here, outside any stream capture).
    explicit DeviceArena(std::size_t capacity_bytes);
    // Non-owning arena over an already allocated device region; it shares storage.ticket_page,
    // the page of the arena the region came from (none for a region of another origin).
    explicit DeviceArena(DeviceSpan storage);
    ~DeviceArena();

    DeviceArena(const DeviceArena&)            = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&& other) noexcept;
    DeviceArena& operator=(DeviceArena&& other) noexcept;

    DeviceSpan alloc_bytes(std::size_t bytes, std::size_t align = 256);
    Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape, std::size_t align = 256);
    [[nodiscard]] Scope scope() noexcept;
    void reset() noexcept;

    void* base() const noexcept;

    // kArenaTicketInts zeroed ints of this arena's own or its parent's ticket page (#889), or
    // nullptr: an op then zeroes a ticket of its workspace in stream order before its grid.
    [[nodiscard]] int* ticket_page() const noexcept { return tickets_; }
    std::size_t used() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t peak_used() const noexcept;
    void reset_peak() noexcept;

private:
    void* base_       = nullptr;
    std::size_t cap_  = 0;
    std::size_t off_  = 0;
    std::size_t peak_ = 0;
    std::uint64_t generation_ = 0;
    bool owns_        = true;
    int* tickets_             = nullptr;
    bool owns_tickets_        = false;
};

// A pin the driver refused (cudaMallocHost), with the bytes asked for: a caller sizing an
// automatic tier may retry smaller (#473); the message names the size and what this process
// already pins.
class PinnedHostError : public std::runtime_error {
public:
    PinnedHostError(const std::string& message, std::size_t bytes)
        : std::runtime_error(message), bytes_(bytes) {}

    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
    std::size_t bytes_ = 0;
};

class PinnedHostBuffer {
public:
    // Throws PinnedHostError when the driver refuses the pin.
    explicit PinnedHostBuffer(std::size_t size_bytes);
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void* data() const noexcept;
    std::size_t size() const noexcept;

private:
    void* data_       = nullptr;
    std::size_t size_ = 0;
};

// Pinned bytes this process holds in PinnedHostBuffers (the live ones).
[[nodiscard]] std::size_t pinned_host_bytes() noexcept;

// The pins of the context cache's Host tier, owned by whoever measured them: the engine pins the
// tier once while it plans (under the host's pin lock) and hands the pins on to the Program, which
// gives each to the pool that sizes it, so the memory a plan measured stays pinned until it is
// used (another process starting beside this one cannot take it, #473) and nothing pins a size
// twice. Move-only; one per construction, never process-global.
struct HostTierPins {
    std::optional<PinnedHostBuffer> states; // exactly host slots * StateImage bytes
    std::optional<PinnedHostBuffer> kv;     // exactly the Host KV capacity

    HostTierPins()                               = default;
    HostTierPins(const HostTierPins&)            = delete;
    HostTierPins& operator=(const HostTierPins&) = delete;

    // A moved-from set holds nothing, and says so (states and kv disengaged), so it is never
    // taken for pins of 0 bytes.
    HostTierPins(HostTierPins&& other) noexcept
        : states(std::exchange(other.states, std::nullopt)),
          kv(std::exchange(other.kv, std::nullopt)) {}

    HostTierPins& operator=(HostTierPins&& other) noexcept {
        if (this != &other) {
            states = std::exchange(other.states, std::nullopt);
            kv     = std::exchange(other.kv, std::nullopt);
        }
        return *this;
    }

    [[nodiscard]] std::size_t bytes() const noexcept {
        return (states ? states->size() : 0) + (kv ? kv->size() : 0);
    }
};

using WorkspaceArena = DeviceArena;

} // namespace ninferno
