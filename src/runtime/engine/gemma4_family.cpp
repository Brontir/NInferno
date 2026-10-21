#include "runtime/engine/model_instance.h"

#include "runtime/engine/hf_text_frontend.h"

#include "core/startup.h"
#include "models/gemma4/model.h"
#include "models/gemma4/program_family.h"

#include <memory>
#include <string>
#include <utility>

namespace ninferno::runtime {
namespace {

// The loaded target and the frontend resources its tokenizer and chat template were built from.
class Gemma4LoadedModel final : public LoadedModel {
public:
    std::unique_ptr<models::gemma4::Model> model;
    std::unique_ptr<models::gemma4::Drafter> drafter;
    HfTextFiles text;
};

} // namespace

FamilyLoad load_gemma4(const EngineOptions& options, DeviceContext& device) {
    refuse_qwen_only_options(options, "Gemma 4");
    // Before the weights load, as the dense family checks (validate_options repeats it after).
    if (options.speculative.backend != SpeculativeBackend::None &&
        options.speculative.backend != SpeculativeBackend::Mtp) {
        throw std::invalid_argument(
            "Gemma 4 speculates with its MTP assistant only (--spec mtp --drafter ASSISTANT)");
    }
    auto loaded = std::make_unique<Gemma4LoadedModel>();
    {
        StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
        loaded->text = read_hf_text_files(options.artifact_path);
        inspect.complete();
    }
    {
        StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
        // --activations exact: BF16 activations on every quantized linear, MoE experts included.
        loaded->model =
            models::gemma4::Model::load(options.artifact_path, device,
                                        options.activations == ActivationMode::Exact
                                            ? models::gemma4::ActivationProfile::A16Full
                                            : models::gemma4::ActivationProfile::Artifact,
                                        options.enable_vision);
        if (options.speculative.backend == SpeculativeBackend::Mtp) {
            if (options.speculative.drafter.empty()) {
                throw std::invalid_argument(
                    "Gemma 4 MTP speculation needs its assistant artifact: --drafter, or the "
                    "assistant named in the artifact's .speculation.json");
            }
            loaded->drafter = models::gemma4::Drafter::load(options.speculative.drafter, device,
                                                            loaded->model->config());
        }
        device.synchronize();
        binding.complete();
    }
    const models::gemma4::Model& model = *loaded->model;
    const auto& config                 = model.config();

    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    FamilyLoad out;
    const std::int32_t vocabulary = attach_hf_text_frontend(
        out, loaded->text, options, static_cast<std::size_t>(config.vocab_size), "Gemma 4",
        model.vision_config().has_value());
    if (model.config().experts > 0) {
        out.routed_experts    = static_cast<std::uint32_t>(model.config().experts);
        out.experts_per_token = static_cast<std::uint32_t>(model.config().experts_per_token);
    }
    out.execution = models::gemma4::make_program_execution(model, loaded->drafter.get(), device,
                                                           vocabulary, options.activations);
    frontend.complete();

    LoadSummary& summary         = out.summary;
    summary.architecture         = config.moe() ? "gemma4_moe" : "gemma4";
    summary.model_name           = options.artifact_path.stem().string();
    summary.upload_seconds       = model.upload_seconds();
    summary.host_to_device_bytes = model.host_to_device_bytes();
    summary.prefill_signature    = "gemma4-prefill-1:" + std::to_string(config.layers) + "x" +
                                std::to_string(config.hidden_size) + ":" +
                                std::to_string(config.vocab_size) +
                                (config.moe() ? ":moe" + std::to_string(config.experts) : "");
    out.model = std::move(loaded);
    return out;
}

} // namespace ninferno::runtime
