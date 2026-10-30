#include "core/device.h"
#include "models/qwen3_5/state/state_image.h"
#include "support/cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q36 = ninferno::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

struct PlannedPool {
    q36::StateImageDeviceLayout layout;
    std::size_t bytes = 0;
};

PlannedPool plan_pool(bool dflash, std::int32_t slots = 4, bool dflash2 = false, bool rings = false,
                      bool linear = true, bool fp8_rings = false, bool windows = false) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 2,
                .conv_channels  = 5,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 3,
                .slot_count     = slots,
                .conv_dtype     = ninferno::DType::BF16,
            },
        .hidden = 7,
    };
    if (dflash) {
        spec.dflash_local = dflash2
                                ? q36::DFlashLocalStateSpec{.layers   = 5,
                                                            .capacity = 2048,
                                                            .kv_heads = 8,
                                                            .head_dim = 128}
                                : q36::DFlashLocalStateSpec{
                                      .layers = 2, .capacity = 17, .kv_heads = 2, .head_dim = 4};
    }
    if (!linear) { spec.linear = {.layers = 0, .slot_count = slots}; }
    if (rings) {
        // Two sliding groups of different geometry (a Gemma 4 shape in miniature).
        spec.rings = {{.layers = 3, .capacity = 19, .kv_heads = 2, .head_dim = 8},
                      {.layers = 1, .capacity = 5, .kv_heads = 1, .head_dim = 16}};
        for (auto& ring : spec.rings) { ring.fp8 = fp8_rings; }
        if (windows) {
            spec.rings[0].window = 6;
            spec.rings[1].window = 2;
        }
    }
    ninferno::LayoutBuilder builder;
    q36::StateImageDeviceLayout layout = q36::plan_state_image_device_pool(builder, spec);
    return {.layout = std::move(layout), .bytes = builder.finish(256)};
}

// Seeds on the stream the pool operations under test run on. It is non-blocking, so a plain
// cudaMemset (legacy default stream, returns before the fill runs) is not ordered before them.
void set_bytes(const ninferno::Tensor& tensor, unsigned char value, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(tensor.data, value, tensor.bytes(), stream));
}

void expect_bytes(const ninferno::Tensor& tensor, unsigned char expected, std::string_view label) {
    std::vector<unsigned char> host(tensor.bytes());
    CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
    for (const unsigned char value : host) {
        if (value != expected) {
            ++failures;
            std::cerr << "FAIL: " << label << " differs\n";
            return;
        }
    }
}

void fill_slot(q36::StateImageDevicePool& pool, std::int32_t slot, unsigned char base,
               cudaStream_t stream) {
    const std::uint32_t linear_layers = pool.has_linear() ? pool.linear().layer_count() : 0;
    for (std::uint32_t layer = 0; layer < linear_layers; ++layer) {
        set_bytes(pool.linear().conv_slot(layer, slot), static_cast<unsigned char>(base + layer),
                  stream);
        set_bytes(pool.linear().recurrent_slot(layer, slot),
                  static_cast<unsigned char>(base + 0x10 + layer), stream);
    }
    set_bytes(pool.continuation_hidden_slot(slot), static_cast<unsigned char>(base + 0x20),
              stream);
    if (ninferno::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            set_bytes(view.k.slice(3, slot, 1), static_cast<unsigned char>(base + 0x30 + layer),
                      stream);
            set_bytes(view.v.slice(3, slot, 1), static_cast<unsigned char>(base + 0x40 + layer),
                      stream);
        }
    }
    for (std::size_t group = 0; group < pool.ring_groups(); ++group) {
        for (std::uint32_t layer = 0; layer < pool.ring_spec(group).layers; ++layer) {
            const auto planes = pool.ring_planes(group, layer);
            const auto tag    = static_cast<unsigned char>(base + 0x50 + 8 * group + layer);
            set_bytes(planes.k.slice(3, slot, 1), tag, stream);
            set_bytes(planes.v.slice(3, slot, 1), static_cast<unsigned char>(tag + 0x40), stream);
            if (planes.k_scale.data != nullptr) {
                set_bytes(planes.k_scale.slice(3, slot, 1), static_cast<unsigned char>(tag + 0x80),
                          stream);
                set_bytes(planes.v_scale.slice(3, slot, 1), static_cast<unsigned char>(tag + 0xC0),
                          stream);
            }
        }
    }
}

void expect_slot(q36::StateImageDevicePool& pool, std::int32_t slot, unsigned char base,
                 std::string_view label) {
    const std::uint32_t linear_layers = pool.has_linear() ? pool.linear().layer_count() : 0;
    for (std::uint32_t layer = 0; layer < linear_layers; ++layer) {
        expect_bytes(pool.linear().conv_slot(layer, slot), static_cast<unsigned char>(base + layer),
                     label);
        expect_bytes(pool.linear().recurrent_slot(layer, slot),
                     static_cast<unsigned char>(base + 0x10 + layer), label);
    }
    expect_bytes(pool.continuation_hidden_slot(slot), static_cast<unsigned char>(base + 0x20),
                 label);
    if (ninferno::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            expect_bytes(view.k.slice(3, slot, 1), static_cast<unsigned char>(base + 0x30 + layer),
                         label);
            expect_bytes(view.v.slice(3, slot, 1), static_cast<unsigned char>(base + 0x40 + layer),
                         label);
        }
    }
    for (std::size_t group = 0; group < pool.ring_groups(); ++group) {
        for (std::uint32_t layer = 0; layer < pool.ring_spec(group).layers; ++layer) {
            const auto planes = pool.ring_planes(group, layer);
            const auto tag    = static_cast<unsigned char>(base + 0x50 + 8 * group + layer);
            expect_bytes(planes.k.slice(3, slot, 1), tag, label);
            expect_bytes(planes.v.slice(3, slot, 1), static_cast<unsigned char>(tag + 0x40), label);
            if (planes.k_scale.data != nullptr) {
                expect_bytes(planes.k_scale.slice(3, slot, 1),
                             static_cast<unsigned char>(tag + 0x80), label);
                expect_bytes(planes.v_scale.slice(3, slot, 1),
                             static_cast<unsigned char>(tag + 0xC0), label);
            }
        }
    }
}

void expect_zero_slot(q36::StateImageDevicePool& pool, std::int32_t slot, std::string_view label) {
    const std::uint32_t linear_layers = pool.has_linear() ? pool.linear().layer_count() : 0;
    for (std::uint32_t layer = 0; layer < linear_layers; ++layer) {
        expect_bytes(pool.linear().conv_slot(layer, slot), 0, label);
        expect_bytes(pool.linear().recurrent_slot(layer, slot), 0, label);
    }
    expect_bytes(pool.continuation_hidden_slot(slot), 0, label);
    if (ninferno::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            expect_bytes(view.k.slice(3, slot, 1), 0, label);
            expect_bytes(view.v.slice(3, slot, 1), 0, label);
        }
    }
    for (std::size_t group = 0; group < pool.ring_groups(); ++group) {
        for (std::uint32_t layer = 0; layer < pool.ring_spec(group).layers; ++layer) {
            const auto planes = pool.ring_planes(group, layer);
            expect_bytes(planes.k.slice(3, slot, 1), 0, label);
            expect_bytes(planes.v.slice(3, slot, 1), 0, label);
            if (planes.k_scale.data != nullptr) {
                expect_bytes(planes.k_scale.slice(3, slot, 1), 0, label);
                expect_bytes(planes.v_scale.slice(3, slot, 1), 0, label);
            }
        }
    }
}

void test_host_roundtrip(bool dflash, ninferno::DeviceContext& device, bool dflash2 = false,
                         bool rings = false, bool linear = true, bool fp8_rings = false) {
    PlannedPool planned = plan_pool(dflash, 2, dflash2, rings, linear, fp8_rings);
    if (dflash2) {
        expect(q36::dflash_local_transfer_work(planned.layout.host).payload_bytes ==
                   40ULL * 1024 * 1024,
               "DFlash2 local snapshot must transfer exactly 40 MiB");
    }
    ninferno::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);
    fill_slot(pool, 0, dflash ? 0x19 : 0x25, device.stream);
    pool.zero_slot(1, device.stream);

    q36::HostStatePool host(planned.layout.host, 1);
    const auto handle = host.allocate();
    expect(handle.has_value(), "HostStatePool allocates its fixed slot");
    expect(!host.allocate().has_value(), "HostStatePool reports capacity exhaustion");
    expect(host.occupied() == 1, "HostStatePool occupied count after allocation");

    pool.copy_to_host(0, host.writable_view(*handle), device.stream);
    device.synchronize();
    pool.copy_from_host(host.view(*handle), 1, device.stream);
    device.synchronize();
    expect_slot(pool, 1, dflash ? 0x19 : 0x25,
                dflash ? "DFlash Host roundtrip" : "common Host roundtrip");

    const q36::HostStateSlotHandle stale = *handle;
    expect(host.release(stale), "HostStatePool releases a live handle");
    expect(host.occupied() == 0, "HostStatePool occupied count after release");
    bool stale_view_rejected = false;
    try {
        (void)host.view(stale);
    } catch (const std::invalid_argument&) { stale_view_rejected = true; }
    expect(stale_view_rejected, "HostStatePool rejects a stale view");
    const auto reused = host.allocate();
    expect(reused && reused->index == stale.index && reused->generation != stale.generation,
           "HostStatePool reuse advances generation");
    expect(!host.release(stale), "HostStatePool rejects stale release");
    expect(host.release(*reused), "HostStatePool releases the reused slot");
}

// A pool of sliding-window rings without linear state: every slot operation moves the rings, and
// the ring planes have the images layout the Gemma 4 attention reads.
void test_rings(ninferno::DeviceContext& device) {
    PlannedPool planned = plan_pool(false, 3, false, true, false);
    ninferno::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);
    expect(!pool.has_linear(), "ring pool has no linear state");
    bool linear_rejected = false;
    try {
        (void)pool.linear();
    } catch (const std::logic_error&) { linear_rejected = true; }
    expect(linear_rejected, "ring pool rejects a linear state request");
    expect(pool.slot_count() == 3 && pool.ring_groups() == 2, "ring pool geometry");
    const auto planes = pool.ring_planes(0, 2);
    expect(planes.k.dtype == ninferno::DType::BF16 && planes.k.ne[0] == 8 && planes.k.ne[1] == 19 &&
               planes.k.ne[2] == 2 && planes.k.ne[3] == 3 && planes.v.ne[1] == 19,
           "ring planes are [head_dim, capacity, kv_heads, slots]");
    expect(q36::state_image_transfer_work(planned.layout.host).payload_bytes ==
               7U * 2U + 2U * 3U * (8U * 19U * 2U * 2U) + 2U * (16U * 5U * 1U * 2U),
           "ring transfer payload counts hidden and every ring plane");

    fill_slot(pool, 0, 0x03, device.stream);
    fill_slot(pool, 1, 0x61, device.stream);
    pool.copy_slot(0, 2, device.stream);
    device.synchronize();
    expect_slot(pool, 2, 0x03, "ring pool D2D destination");
    expect_slot(pool, 1, 0x61, "ring pool D2D isolation");
    pool.zero_slot(2, device.stream);
    pool.copy_rings(1, 0, device.stream);
    device.synchronize();
    expect_zero_slot(pool, 2, "ring pool zero");
    for (std::size_t group = 0; group < pool.ring_groups(); ++group) {
        for (std::uint32_t layer = 0; layer < pool.ring_spec(group).layers; ++layer) {
            const auto tag = static_cast<unsigned char>(0x61 + 0x50 + 8 * group + layer);
            expect_bytes(pool.ring_planes(group, layer).k.slice(3, 0, 1), tag, "copy_rings K");
        }
    }
    expect_bytes(pool.continuation_hidden_slot(0), 0x03 + 0x20, "copy_rings leaves the hidden");
}

// FP8 rings: E4M3 code planes of the BF16 shape and FP32 scale planes [1, capacity, kv_heads,
// slots] beside them, and every slot operation (D2D copy, fork copy, zero, Host transfer) moves
// the scales with the codes.
void test_fp8_rings(ninferno::DeviceContext& device) {
    PlannedPool planned = plan_pool(false, 3, false, true, false, true);
    ninferno::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);
    const auto planes = pool.ring_planes(0, 2);
    expect(planes.k.dtype == ninferno::DType::FP8_E4M3FN && planes.k.ne[0] == 8 &&
               planes.k.ne[1] == 19 && planes.k.ne[2] == 2 && planes.k.ne[3] == 3,
           "FP8 ring planes are E4M3 [head_dim, capacity, kv_heads, slots]");
    expect(planes.k_scale.dtype == ninferno::DType::FP32 && planes.k_scale.ne[0] == 1 &&
               planes.k_scale.ne[1] == 19 && planes.k_scale.ne[2] == 2 &&
               planes.k_scale.ne[3] == 3 && planes.v_scale.ne[1] == 19,
           "FP8 ring scale planes are FP32 [1, capacity, kv_heads, slots]");
    expect(pool.ring_planes(1, 0).k_scale.ne[1] == 5, "second FP8 ring group has its own scales");
    expect(q36::state_image_transfer_work(planned.layout.host).payload_bytes ==
               7U * 2U + 2U * 3U * (8U * 19U * 2U + 19U * 2U * 4U) +
                   2U * (16U * 5U * 1U + 5U * 1U * 4U),
           "FP8 ring transfer payload counts codes and scales");
    expect(q36::state_image_transfer_work(planned.layout.host).copy_operations == 1U + 4U * 4U,
           "FP8 ring transfer copies four planes per layer");
    PlannedPool bf16 = plan_pool(false, 3, false, true, false, false);
    expect(planned.bytes < bf16.bytes, "FP8 rings take fewer bytes than BF16 rings");

    fill_slot(pool, 0, 0x05, device.stream);
    fill_slot(pool, 1, 0x33, device.stream);
    pool.copy_slot(0, 2, device.stream);
    device.synchronize();
    expect_slot(pool, 2, 0x05, "FP8 ring pool D2D destination");
    expect_slot(pool, 1, 0x33, "FP8 ring pool D2D isolation");
    pool.zero_slot(2, device.stream);
    pool.copy_rings(1, 0, device.stream);
    device.synchronize();
    expect_zero_slot(pool, 2, "FP8 ring pool zero");
    for (std::size_t group = 0; group < pool.ring_groups(); ++group) {
        for (std::uint32_t layer = 0; layer < pool.ring_spec(group).layers; ++layer) {
            const auto tag = static_cast<unsigned char>(0x33 + 0x50 + 8 * group + layer);
            expect_bytes(pool.ring_planes(group, layer).v_scale.slice(3, 0, 1),
                         static_cast<unsigned char>(tag + 0xC0), "copy_rings V scale");
        }
    }
}

// One slot's plane [inner, capacity, kv_heads]: the rows in `live` hold `inside`, every other row
// `outside`.
void expect_rows(const ninferno::Tensor& plane, std::uint32_t capacity,
                 const std::vector<std::uint32_t>& live, unsigned char inside,
                 unsigned char outside, std::string_view label) {
    std::vector<unsigned char> host(plane.bytes());
    CUDA_CHECK(cudaMemcpy(host.data(), plane.data, host.size(), cudaMemcpyDeviceToHost));
    const auto heads     = static_cast<std::size_t>(plane.ne[2]);
    const std::size_t row = host.size() / (capacity * heads);
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::uint32_t r = 0; r < capacity; ++r) {
            const bool in = std::find(live.begin(), live.end(), r) != live.end();
            for (std::size_t b = 0; b < row; ++b) {
                if (host[(head * capacity + r) * row + b] != (in ? inside : outside)) {
                    ++failures;
                    std::cerr << "FAIL: " << label << " row " << r << " head " << head << '\n';
                    return;
                }
            }
        }
    }
}

// A checkpoint's Host round trip with its frontier: a ring moves the rows of its last `window`
// positions (wrapping at the capacity; fewer at a short frontier) and leaves every other row of
// the destination as it was; the hidden moves whole.
void test_ring_live_rows(ninferno::DeviceContext& device, bool fp8) {
    PlannedPool planned = plan_pool(false, 2, false, true, false, fp8, true);
    ninferno::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);
    q36::HostStatePool host(planned.layout.host, 1);
    const auto handle = host.allocate();
    expect(handle.has_value(), "live-row Host slot");
    const auto work = q36::state_image_transfer_work(planned.layout.host);
    expect(work.payload_bytes == (fp8 ? 7U * 2U + 2U * 3U * (8U * 6U * 2U + 6U * 2U * 4U) +
                                            2U * (16U * 2U + 2U * 4U)
                                      : 7U * 2U + 2U * 3U * (8U * 6U * 2U * 2U) +
                                            2U * (16U * 2U * 2U)),
           "a windowed ring's transfer prices its window's rows");
    expect(work.copy_operations == 1U + (fp8 ? 4U : 2U) * 2U * (3U + 1U),
           "a windowed ring's transfer prices two copies a plane");
    // Group 0: capacity 19, window 6; group 1: capacity 5, window 2.
    const struct {
        std::uint64_t frontier;
        std::vector<std::uint32_t> rows[2];
    } cases[] = {
        {.frontier = 23, .rows = {{17, 18, 0, 1, 2, 3}, {1, 2}}}, // wraps in both groups
        {.frontier = 4, .rows = {{0, 1, 2, 3}, {2, 3}}},          // shorter than the window
        {.frontier = 12, .rows = {{6, 7, 8, 9, 10, 11}, {0, 1}}},
    };
    for (const auto& c : cases) {
        fill_slot(pool, 0, 0x11, device.stream);
        fill_slot(pool, 1, 0x47, device.stream);
        pool.copy_to_host(0, host.writable_view(*handle), device.stream, c.frontier);
        pool.copy_from_host(host.view(*handle), 1, device.stream, c.frontier);
        device.synchronize();
        const std::string label = "live rows at frontier " + std::to_string(c.frontier);
        expect_bytes(pool.continuation_hidden_slot(1), 0x11 + 0x20, label + ": hidden");
        for (std::size_t group = 0; group < pool.ring_groups(); ++group) {
            const std::uint32_t capacity = pool.ring_spec(group).capacity;
            for (std::uint32_t layer = 0; layer < pool.ring_spec(group).layers; ++layer) {
                const auto planes = pool.ring_planes(group, layer);
                const auto in     = static_cast<unsigned char>(0x11 + 0x50 + 8 * group + layer);
                const auto out    = static_cast<unsigned char>(0x47 + 0x50 + 8 * group + layer);
                expect_rows(planes.k.slice(3, 1, 1), capacity, c.rows[group], in, out,
                            label + ": K");
                expect_rows(planes.v.slice(3, 1, 1), capacity, c.rows[group],
                            static_cast<unsigned char>(in + 0x40),
                            static_cast<unsigned char>(out + 0x40), label + ": V");
                if (!fp8) { continue; }
                expect_rows(planes.k_scale.slice(3, 1, 1), capacity, c.rows[group],
                            static_cast<unsigned char>(in + 0x80),
                            static_cast<unsigned char>(out + 0x80), label + ": K scale");
                expect_rows(planes.v_scale.slice(3, 1, 1), capacity, c.rows[group],
                            static_cast<unsigned char>(in + 0xC0),
                            static_cast<unsigned char>(out + 0xC0), label + ": V scale");
            }
        }
    }
    // Without a frontier every row moves.
    fill_slot(pool, 0, 0x11, device.stream);
    fill_slot(pool, 1, 0x47, device.stream);
    pool.copy_to_host(0, host.writable_view(*handle), device.stream);
    pool.copy_from_host(host.view(*handle), 1, device.stream);
    device.synchronize();
    expect_slot(pool, 1, 0x11, "every row without a frontier");
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);

    ninferno::DeviceContext device(0);
    PlannedPool planned = plan_pool(true);
    ninferno::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);

    expect(pool.slot_count() == 4, "StateImage slot count");
    expect(pool.linear().layer_count() == 2, "StateImage Linear Attention layer count");
    expect(pool.continuation_hidden_slot(0).dtype == ninferno::DType::BF16 &&
               pool.continuation_hidden_slot(0).ne[0] == 7,
           "StateImage continuation hidden geometry");
    expect(pool.dflash_local() != nullptr && pool.dflash_local()->layer_count() == 2 &&
               pool.dflash_local()->lane_capacity() == 4,
           "StateImage DFlash local geometry");
    expect(pool.has_linear() && pool.linear().layer_count() == 2 &&
               pool.continuation_hidden_slot(0).ne[0] == 7,
           "StateImage slot exposes every fixed component");

    fill_slot(pool, 0, 0x11, device.stream);
    fill_slot(pool, 1, 0x52, device.stream);
    pool.copy_slot(0, 2, device.stream);
    device.synchronize();
    expect_slot(pool, 0, 0x11, "StateImage D2D source isolation");
    expect_slot(pool, 2, 0x11, "StateImage D2D complete destination");

    pool.zero_slot(1, device.stream);
    device.synchronize();
    expect_zero_slot(pool, 1, "StateImage zero complete destination");
    expect_slot(pool, 0, 0x11, "StateImage zero source isolation");

    test_host_roundtrip(false, device);
    test_host_roundtrip(true, device);
    test_host_roundtrip(true, device, true);

    test_rings(device);
    test_host_roundtrip(false, device, false, true, false);
    test_host_roundtrip(true, device, false, true, true);
    test_fp8_rings(device);
    test_host_roundtrip(false, device, false, true, false, true);
    test_ring_live_rows(device, false);
    test_ring_live_rows(device, true);

    return failures == 0 ? 0 : 1;
}
