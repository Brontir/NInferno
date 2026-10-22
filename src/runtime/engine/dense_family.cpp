#include "runtime/engine/model_instance.h"

#include "runtime/engine/hf_text_frontend.h"

#include "core/startup.h"
#include "models/dense/model.h"
#include "models/dense/program_family.h"
#include "models/speculation/dflash.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninferno::runtime {
namespace {

// --activations as the model's load profile: 16 keeps BF16 activations on every quantized linear, 8
// and 4 allow eight and four bits everywhere.
models::dense::ActivationProfile dense_profile(ActivationMode mode) {
    switch (mode) {
    case ActivationMode::Auto:
        return models::dense::ActivationProfile::Artifact;
    case ActivationMode::Bits16:
        return models::dense::ActivationProfile::A16;
    case ActivationMode::Bits8:
        return models::dense::ActivationProfile::A8;
    case ActivationMode::Bits4:
        return models::dense::ActivationProfile::A4;
    }
    return models::dense::ActivationProfile::A16;
}

// The loaded decoder, its DFlash drafter (an artifact of its own) and the frontend resources its
// tokenizer and chat template were built from.
class DenseLoadedModel final : public LoadedModel {
public:
    std::unique_ptr<models::dense::Model> model;
    std::unique_ptr<models::speculation::DFlashDrafter> drafter;
    HfTextFiles text;
};

} // namespace

FamilyLoad load_dense(const EngineOptions& options, DeviceContext& device) {
    refuse_qwen_only_options(options, "the dense decoder");
    if (options.speculative.backend != SpeculativeBackend::None &&
        options.speculative.backend != SpeculativeBackend::DFlash) {
        throw std::invalid_argument(
            "the dense decoder speculates with a DFlash drafter only (--spec dflash --drafter)");
    }
    auto loaded = std::make_unique<DenseLoadedModel>();
    std::string drafter_note;
    {
        StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
        loaded->text = read_hf_text_files(options.artifact_path);
        inspect.complete();
    }
    {
        StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
        // --activations 16: BF16 activations on every projection, and (below) the invariant
        // route profile on every call.
        loaded->model = models::dense::Model::load(options.artifact_path, device,
                                                   dense_profile(options.activations));
        if (options.speculative.backend == SpeculativeBackend::DFlash) {
            if (options.speculative.drafter.empty()) {
                throw std::invalid_argument(
                    "dense decoder DFlash speculation needs its drafter artifact: --drafter, or "
                    "the drafter named in the model's speculation profile");
            }
            loaded->drafter = models::speculation::DFlashDrafter::load(
                options.speculative.drafter, device, loaded->model->config());
            // Split by tap, the drafter's feature projection lets the Program sum a prefill chunk's
            // projection as its forward reaches each tapped layer: the chunk keeps one [H, chunk]
            // sum instead of every tapped layer's [H, chunk] features. Whole (a format that does
            // not split, or no memory for the transient copy), the Program keeps the features and
            // projects them at once.
            std::string why;
            if (loaded->drafter->split_feature_projection(device, &why)) {
                drafter_note = "DFlash feature projection split by tap (" +
                               std::to_string(loaded->drafter->config().taps.size()) +
                               " slices): prefill sums it tap by tap";
            } else {
                drafter_note = "DFlash feature projection whole (" + why +
                               "): prefill keeps every tapped layer's features";
            }
        }
        device.synchronize();
        binding.complete();
    }
    const models::dense::Model& model = *loaded->model;
    const auto& config                = model.config();

    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    FamilyLoad out;
    const std::int32_t vocabulary = attach_hf_text_frontend(
        out, loaded->text, options, static_cast<std::size_t>(config.vocab_size), "dense decoder");
    if (config.moe) {
        out.routed_experts    = static_cast<std::uint32_t>(config.moe->experts);
        out.experts_per_token = static_cast<std::uint32_t>(config.moe->op.top_k);
    }
    out.execution = models::dense::make_program_execution(model, loaded->drafter.get(), device,
                                                          vocabulary, options.activations);
    frontend.complete();

    LoadSummary& summary         = out.summary;
    summary.architecture         = config.moe ? "moe_decoder" : "dense_decoder";
    summary.model_name           = options.artifact_path.stem().string();
    summary.upload_seconds       = model.upload_seconds();
    summary.host_to_device_bytes = model.host_to_device_bytes();
    // A separate drafter uploads its own weights: the plan ladder's weight bytes and the startup
    // log count them with the target's.
    if (loaded->drafter) { summary.host_to_device_bytes += loaded->drafter->device_bytes(); }
    summary.prefill_signature    = "dense-prefill-1:" + std::to_string(config.layers) + "x" +
                                std::to_string(config.hidden_size) + ":" +
                                std::to_string(config.vocab_size) +
                                (config.moe ? ":moe" + std::to_string(config.moe->experts) : "");
    summary.drafter = std::move(drafter_note);
    out.model       = std::move(loaded);
    return out;
}

} // namespace ninferno::runtime
