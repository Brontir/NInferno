#pragma once

#include "core/dtype.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninferno {

struct DeviceSpan {
    void* data        = nullptr;
    std::size_t bytes = 0;
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

    explicit DeviceArena(std::size_t capacity_bytes);
    // Non-owning arena over an already allocated device region.
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

    [[nodiscard]] std::size_t bytes() const noexcept {
        return (states ? states->size() : 0) + (kv ? kv->size() : 0);
    }
};

using WorkspaceArena = DeviceArena;

} // namespace ninferno
