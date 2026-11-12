#pragma once

#include "models/qwen3_5/model.h"

#include <filesystem>
#include <memory>
#include <span>

namespace ninferno::artifact {
class Reader;
struct Directory;
struct ParameterReference;
} // namespace ninferno::artifact

namespace ninferno::models::qwen3_5 {

// Cold load plan borrows its Reader until materialization. Its selected Host bytes and parsed
// resources already have owners; no Program or device allocation is needed for plan_load.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const ModelWeights& weights() const;
    [[nodiscard]] const FrontendResources& resources() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    [[nodiscard]] const artifact::ParameterReference& parameter(WeightId id) const;
    [[nodiscard]] std::span<const WeightUse> uses(WeightId id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
};

[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {});
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);
[[nodiscard]] std::unique_ptr<Model> load_model(const std::filesystem::path& path,
                                                LoadOptions options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

// A model described by an artifact's directory alone, without its payload (the plan without the
// weights, #615): the config and the weights plan_load binds for the same options, each a view
// of its objects as a start's materialized parents hold them, at addresses that are never read.
// What a start prepares its forms from (execution::prepare_parameters reads this as it reads a
// Model), so a plan sizes the start's workspace over the same forms.
class DescribedModel {
public:
    DescribedModel(const DescribedModel&)            = delete;
    DescribedModel& operator=(const DescribedModel&) = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }

    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    [[nodiscard]] ops::WeightInput input(WeightUseId id) const;
    [[nodiscard]] ops::WeightInput input(WeightId id) const;

private:
    DescribedModel() = default;
    friend std::unique_ptr<DescribedModel> describe_model(const artifact::Directory&, LoadOptions,
                                                          std::uint32_t);

    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    std::vector<WeightParent> parents_; // one an object; bound_'s views point into it
    std::vector<BoundWeight> bound_;
};

// `public_tokens`: the tokenizer's public ids (public_token_count), what plan_load checks an
// indexed proposal head and a DFlash2 selector against. Throws (an ArtifactError,
// std::invalid_argument) where plan_load would refuse the directory for `options`, or where a
// binding reads the payload: a Use's activation divisor (an NVFP4 A4 route's calibration). An
// indexed proposal head's ids and the frontend's resources are not read (no form depends on them).
[[nodiscard]] std::unique_ptr<DescribedModel> describe_model(const artifact::Directory& directory,
                                                             LoadOptions options,
                                                             std::uint32_t public_tokens);

// The ids a request may carry or sample, as plan_load reads them (FrontendResources::
// public_token_count): the artifact's tokenizer resources, the only objects read (no weight is).
// Throws where plan_load's resources refuse.
[[nodiscard]] std::uint32_t public_token_count(const artifact::Reader& reader);

} // namespace ninferno::models::qwen3_5
