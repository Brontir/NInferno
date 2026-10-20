#include "runtime/engine/model_instance.h"

#include "runtime/engine/hf_text_frontend.h"

#include "core/startup.h"
#include "models/nemotron_h/model.h"
#include "models/nemotron_h/program_family.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninferno::runtime {
namespace {

// The loaded decoder and the frontend resources its tokenizer and chat template were built from.
class NemotronLoadedModel final : public LoadedModel {
public:
    std::unique_ptr<models::nemotron_h::Model> model;
    std::unique_ptr<models::nemotron_h::MtpDrafter> mtp;
    HfTextFiles text;
};

} // namespace

FamilyLoad load_nemotron_h(const EngineOptions& options, DeviceContext& device) {
    refuse_qwen_only_options(options, "Nemotron-H");
    // Every expert loads on the GPU: no expert offload, which the Engine would otherwise skip.
    if (options.moe_offload.mode != MoeOffloadMode::Off) {
        throw std::invalid_argument(
            "Nemotron-H places every MoE expert on the GPU: --moe-offload is not available for it");
    }
    // Before the weights load, as the dense family checks (validate_options repeats it after).
    if (options.speculative.backend != SpeculativeBackend::None &&
        options.speculative.backend != SpeculativeBackend::Mtp) {
        throw std::invalid_argument(
            "Nemotron-H speculates with its MTP head only (--spec mtp on an artifact with one)");
    }
    auto loaded = std::make_unique<NemotronLoadedModel>();
    {
        StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
        loaded->text = read_hf_text_files(options.artifact_path);
        inspect.complete();
    }
    {
        StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
        // --activations exact: BF16 activations on every projection, and (below) the invariant
        // route profile on every call.
        loaded->model =
            models::nemotron_h::Model::load(options.artifact_path, device,
                                            options.activations == ActivationMode::Exact
                                                ? models::nemotron_h::ActivationProfile::A16
                                                : models::nemotron_h::ActivationProfile::Artifact);
        if (options.speculative.backend == SpeculativeBackend::Mtp) {
            if (options.speculative.drafter.empty()) {
                throw std::invalid_argument(
                    "nemotron_h MTP speculation needs the head's drafter artifact (the converted "
                    "mtp-*.gguf sidecar): --drafter PATH; ninferno-serve also finds it in the "
                    "target's .speculation.json once attached with ninferno-convert "
                    "--attach-drafters --target-artifact TARGET --drafter-artifact mtp=PATH");
            }
            loaded->mtp = models::nemotron_h::MtpDrafter::load(options.speculative.drafter, device,
                                                               loaded->model->config());
        }
        device.synchronize();
        binding.complete();
    }
    const models::nemotron_h::Model& model = *loaded->model;
    const auto& config                     = model.config();

    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    FamilyLoad out;
    const std::int32_t vocabulary = attach_hf_text_frontend(
        out, loaded->text, options, static_cast<std::size_t>(config.vocab_size), "nemotron_h");
    out.execution = models::nemotron_h::make_program_execution(
        model, loaded->mtp ? &loaded->mtp->weights() : nullptr, device, vocabulary,
        options.activations);
    frontend.complete();

    LoadSummary& summary         = out.summary;
    summary.architecture         = "nemotron_h";
    summary.model_name           = options.artifact_path.stem().string();
    summary.upload_seconds       = model.upload_seconds();
    summary.host_to_device_bytes = model.host_to_device_bytes();
    if (loaded->mtp) { summary.host_to_device_bytes += loaded->mtp->host_to_device_bytes(); }
    summary.prefill_signature = "nemotron-h-prefill-1:" + std::to_string(config.layers) + "x" +
                                std::to_string(config.hidden_size) + ":" +
                                std::to_string(config.vocab_size);
    out.model = std::move(loaded);
    return out;
}

} // namespace ninferno::runtime
