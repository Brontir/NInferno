#include "models/qwen3_5/load.h"

#include "artifact/reader.h"
#include "models/qwen3_5/load/bindings.h"

#include <utility>

namespace ninferno::models::qwen3_5 {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

namespace {

// The weights a load binds for `options` (plan_load's, describe_model's): the text, the Vision
// encoder, the MTP layer, the drafter and the proposal head the config and the options select.
// `read_ids` false: an indexed proposal head's ids stay unread (no payload behind the directory).
ModelWeights bind_model(loading::Bindings& bindings, const artifact::Directory& directory,
                        const Config& config, const LoadOptions& options,
                        std::uint32_t public_tokens, bool read_ids) {
    ModelWeights out;
    const auto& text = config.text;
    out.text         = loading::bind_text(bindings, text, options);
    if (config.vision) { out.vision = loading::bind_vision(bindings, *config.vision, text); }
    if (config.mtp) { out.mtp = loading::bind_mtp(bindings, text, out.text); }
    if (config.draft) {
        out.draft = loading::bind_draft(bindings, *config.draft, text, out.text,
                                        std::string(options.speculative_component()));
    }
    if (options.proposal_enabled()) {
        const auto& proposal = directory.component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out.proposal =
            loading::bind_proposal(bindings, *proposal, text, options, public_tokens, read_ids);
        if (config.draft && config.draft->dflash2) {
            const auto domain = proposal->indexed ? proposal->rows : public_tokens;
            if (config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out.mtp) { out.mtp->output_head = out.proposal->head; }
        if (out.draft) { out.draft->output_head = out.proposal->head; }
    }
    out.text.output_head_use = bindings.use(out.text.output_head, "text/final_hidden");
    if (out.mtp) {
        out.mtp->output_head_use = bindings.use(out.mtp->output_head, "mtp/final_hidden");
    }
    if (out.draft) {
        out.draft->output_head_use = bindings.use(
            out.draft->output_head, std::string(options.speculative_component()) + "/final_hidden");
    }
    return out;
}

} // namespace

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder);
    const auto& text = out->config.text;
    out->weights     = bind_model(bindings, reader.directory(), out->config, options,
                                  out->resources.public_token_count, true);
    out->pending     = std::move(bindings.weights);
    // Every binding of a component this model loads must be bound.
    out->materialization =
        std::move(binder).finish(artifact::UnusedBindings::RefuseInLoadedComponents);
    out->info.name = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data    = std::move(plan.impl_);
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), device, observer);
    auto bound   = loading::resolve_weights(std::move(data->pending), backing);
    DeviceBuffer yarn;
    if (auto& rope = data->config.text.rope_parameters; rope && rope->yarn()) {
        yarn = DeviceBuffer(rope->yarn_inv_freq.size() * sizeof(float));
        yarn.copy_from_host(rope->yarn_inv_freq.data(), yarn.bytes);
        rope->yarn_frequencies = static_cast<const float*>(yarn.p);
    }
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(data->resources), std::move(data->info), std::move(backing), std::move(yarn)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

ops::WeightInput DescribedModel::input(WeightUseId id) const {
    return bound_weight_input(weight(id.parameter), id.use_index, options_);
}

ops::WeightInput DescribedModel::input(WeightId id) const {
    return bound_weight_input(weight(id), options_);
}

std::uint32_t public_token_count(const artifact::Reader& reader) {
    const Config config = parse_config(reader.directory(), {});
    artifact::Binder binder(reader);
    return loading::bind_resources(binder, config).public_token_count;
}

std::unique_ptr<DescribedModel> describe_model(const artifact::Directory& directory,
                                               LoadOptions options, std::uint32_t public_tokens) {
    std::unique_ptr<DescribedModel> out(new DescribedModel);
    out->options_ = options;
    out->config_  = parse_config(directory, options);
    // The binder over the directory alone: a binding that reads the payload (a Use's activation
    // divisor) throws, and the plan says so.
    const auto reader = artifact::Reader::detached(directory);
    artifact::Binder binder(reader);
    loading::Bindings bindings(binder);
    // The tokenizer's public ids, as plan_load checks the proposal head and the selector against.
    out->weights_ = bind_model(bindings, directory, out->config_, options, public_tokens, false);
    // One parent an object, as materialize gives each object its device parent: its geometry, an
    // address never read (distinct a parent; the forms compare parents, not addresses), and for
    // NVFP4 a positive weight divisor in place of the payload's (the forms read whether it is
    // there, not its value).
    out->parents_.resize(directory.objects.size());
    for (std::size_t i = 0; i < directory.objects.size(); ++i) {
        if (!std::holds_alternative<artifact::TensorObject>(directory.objects[i])) { continue; }
        const auto& geometry = reader.geometry(artifact::ObjectHandle{i});
        out->parents_[i]     = WeightParent{
            geometry, reinterpret_cast<const std::byte*>((std::uintptr_t{i} + 1) << 32),
            geometry.format == QType::NVFP4 ? 1.0F : 0.0F};
    }
    out->bound_.reserve(bindings.weights.size());
    for (auto& pending : bindings.weights) {
        const auto& reference = pending.reference;
        WeightView view;
        view.shape = reference.shape;
        for (const auto& part : reference.binding.parts) {
            view.parts.push_back({&out->parents_.at(part.object.index), part.begin, part.end});
        }
        out->bound_.push_back({reference.name, std::move(pending.source_objects), std::move(view),
                               std::move(pending.uses)});
    }
    return out;
}

} // namespace ninferno::models::qwen3_5
