#pragma once

// Which model family serves an artifact is data: the model_type its text component declares
// (qwen3_5_text, gemma4_text, dense_decoder, ...). The Engine looks it up in one table of
// factories and never names a family itself.

#include "core/device.h"
#include "ninferno/types.h"
#include "runtime/engine/backend.h"
#include "runtime/engine/model_instance.h"

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace ninferno::runtime {

struct FamilyEntry {
    std::string_view model_type;
    FamilyLoader load = nullptr;
};

// Every family this build serves.
[[nodiscard]] std::span<const FamilyEntry> families() noexcept;

// The KV storage the family declaring `model_type` takes when the Engine chooses it. Throws like
// family_loader when no family declares it.
[[nodiscard]] KvCacheStorage automatic_kv_cache(std::string_view model_type);

// The automatic KV storage on a GPU of `compute_capability` (major * 10 + minor): the family's,
// except that FP8 rows fall back to BF16 on a chip without E4M3 instructions (plan_dry_run.h
// kv_storage_on, the one rule for the Engine and the plan).
[[nodiscard]] KvCacheStorage automatic_kv_cache(std::string_view model_type,
                                                int compute_capability);

// The recurrent state the family declaring `model_type` takes under --activations fast when the
// Engine chooses it. Throws like family_loader when no family declares it.
[[nodiscard]] RecurrentStateStorage automatic_recurrent_state(std::string_view model_type);

// The text component's model_type of the artifact at `path`, read from its directory only.
[[nodiscard]] std::string artifact_model_type(const std::filesystem::path& path);

// The loader of the family that declares `model_type`. Throws std::invalid_argument naming the
// served model types when no family declares it.
[[nodiscard]] FamilyLoader family_loader(std::string_view model_type);

// Loads the artifact named by `options` with the family its model_type selects.
[[nodiscard]] std::unique_ptr<Backend> make_backend(const EngineOptions& options,
                                                    DeviceContext& device);

} // namespace ninferno::runtime
