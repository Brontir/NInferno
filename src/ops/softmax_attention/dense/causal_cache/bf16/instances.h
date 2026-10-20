#pragma once

#include "ops/softmax_attention/dense/causal_cache/bf16/schedule.cuh"

namespace ninferno::ops::detail {

enum class Bf16KvInstance { Grouped32, Grouped64, Tiled64, Tiled128, GroupedDecode };

struct Bf16KvInstanceDescription {
    int query_rows;
    int key_rows;
    bool grouped;
};

template <Bf16KvInstance>
struct Bf16KvInstanceTraits;
#define NINFERNO_BF16_KV_INSTANCE(ID, GROUPED, ...)                                                \
    template <>                                                                                    \
    struct Bf16KvInstanceTraits<Bf16KvInstance::ID> {                                              \
        using Schedule = __VA_ARGS__;                                                              \
        static constexpr Bf16KvInstanceDescription description{Schedule::kQueryRows,               \
                                                               Schedule::kKeyRows, GROUPED};       \
    }
NINFERNO_BF16_KV_INSTANCE(Grouped32, true, Bf16KvGroupedMmaSchedule<32, 32>);
NINFERNO_BF16_KV_INSTANCE(Grouped64, true, Bf16KvGroupedMmaSchedule<64, 32>);
// The tiled prefill stages Q and a K/V pair of key tiles: 96 KiB at D=256 for both tuned
// tiles. Turing's 64 KiB takes 64 query rows over 32-key tiles (64 KiB) instead.
NINFERNO_BF16_KV_INSTANCE(Tiled64, false,
                        fitting_schedule<Bf16KvTiledMmaSchedule<64, 64>,
                                         Bf16KvTiledMmaSchedule<64, 32>>);
NINFERNO_BF16_KV_INSTANCE(Tiled128, false,
                        fitting_schedule<Bf16KvTiledMmaSchedule<128, 32>,
                                         Bf16KvTiledMmaSchedule<64, 32>>);
NINFERNO_BF16_KV_INSTANCE(GroupedDecode, true, Bf16KvGroupedMmaSchedule<16, 32, 2, 2, 1>);
#undef NINFERNO_BF16_KV_INSTANCE

constexpr Bf16KvInstanceDescription bf16_kv_instance_description(Bf16KvInstance id) {
    switch (id) {
#define NINFERNO_BF16_KV_DESCRIBE(ID)                                                              \
    case Bf16KvInstance::ID:                                                                       \
        return Bf16KvInstanceTraits<Bf16KvInstance::ID>::description
        NINFERNO_BF16_KV_DESCRIBE(GroupedDecode);
        NINFERNO_BF16_KV_DESCRIBE(Grouped32);
        NINFERNO_BF16_KV_DESCRIBE(Grouped64);
        NINFERNO_BF16_KV_DESCRIBE(Tiled64);
        NINFERNO_BF16_KV_DESCRIBE(Tiled128);
#undef NINFERNO_BF16_KV_DESCRIBE
    }
    return {};
}

} // namespace ninferno::ops::detail
