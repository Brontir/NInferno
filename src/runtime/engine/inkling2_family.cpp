#include "runtime/engine/model_instance.h"

#include "runtime/engine/hf_text_frontend.h"

#include "core/startup.h"
#include "models/inkling2/model.h"
#include "models/inkling2/program_family.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninferno::runtime {
namespace {

// The loaded model and the frontend resources its tokenizer and chat template were built from.
class Inkling2LoadedModel final : public LoadedModel {
public:
    std::unique_ptr<models::inkling2::Model> model;
    HfTextFiles text;
};

} // namespace

FamilyLoad load_inkling2(const EngineOptions& options, DeviceContext& device) {
    refuse_qwen_only_options(options, "Inkling 2");
    // Before the weights load (validate_options repeats it after).
    if (options.speculative.backend != SpeculativeBackend::None) {
        throw std::invalid_argument("Inkling 2 runs no speculative backend (--spec off)");
    }
    auto loaded = std::make_unique<Inkling2LoadedModel>();
    {
        StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
        loaded->text = read_hf_text_files(options.artifact_path);
        inspect.complete();
    }
    {
        StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
        loaded->model = models::inkling2::Model::load(options.artifact_path, device);
        device.synchronize();
        binding.complete();
    }
    const models::inkling2::Model& model = *loaded->model;
    const auto& config                   = model.config();

    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    FamilyLoad out;
    const std::int32_t vocabulary = attach_hf_text_frontend(
        out, loaded->text, options, static_cast<std::size_t>(config.vocab_size), "inkling2");
    out.execution = models::inkling2::make_program_execution(model, device, vocabulary,
                                                             options.activations, options.purpose);
    frontend.complete();

    LoadSummary& summary         = out.summary;
    summary.architecture         = "inkling2";
    summary.model_name           = options.artifact_path.stem().string();
    summary.upload_seconds       = model.upload_seconds();
    summary.host_to_device_bytes = model.host_to_device_bytes();
    summary.prefill_signature    = "inkling2-prefill-1:" + std::to_string(config.visits()) + "x" +
                                std::to_string(config.hidden_size) + ":" +
                                std::to_string(config.vocab_size);
    out.model = std::move(loaded);
    return out;
}

} // namespace ninferno::runtime
