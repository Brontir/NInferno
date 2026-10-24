#include "serve/model_source.h"
#include "artifact/source_identity.h"
#include "serve/plan_offline.h"

#include "convert/acquire.h"
#include "convert/support.h"
#include "convert/cache.h"
#include "core/host_memory.h"
#include "core/sys.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace ninferno::serve {
namespace {

// The artifact's source identity (artifact/source_identity.h), as the Engine computes it at
// startup from the same directory: "none" for an artifact converted before its inputs and content
// digests were recorded (the Engine then keys it as "none:<artifact id>").
void log_source_identity(const std::function<void(const std::string&)>& log,
                         const std::string& artifact) {
    if (!log) { return; }
    try {
        log("source identity " +
            ninferno::artifact::source_identity(std::filesystem::path(artifact)).value_or("none") +
            " (" + artifact + ")");
    } catch (const std::exception& e) {
        log(std::string("source identity unreadable: ") + e.what());
    }
}

} // namespace

namespace {

namespace fs = std::filesystem;

// Before a hub model is downloaded with its quantization named (--quant) or not chosen by the
// rule (a safetensors repository): whether it fits this GPU now, by the plan without weights (the
// same memory plan the start makes). A model the plan cannot read is left to the start.
void check_fits(const ServeOptions& options) {
    OfflinePlan plan;
    try {
        plan = plan_without_load(options);
    } catch (const std::exception&) { return; }
    if (plan.fits) { return; }
    throw std::runtime_error(options.model_source + " does not fit GPU " +
                             std::to_string(options.device) + " (nothing was downloaded): " +
                             plan.reason +
                             ". Options: a smaller quantization (--quant), a shorter context "
                             "(--max-context), fewer requests (--max-concurrency), or a GPU with "
                             "more free memory.");
}

} // namespace

ModelSource resolve_model_source(const ServeOptions& options,
                                 const std::function<void(const std::string&)>& log) {
    const std::string& name = options.model_source;
    ModelSource out;
    std::error_code error;
    if (fs::exists(name, error)) {
        if (!options.recipe.empty() || !options.quant.empty()) {
            throw std::runtime_error("--recipe and --quant choose the conversion of a hub id; " +
                                     name + " is a local path (convert it with ninferno-convert "
                                            "--model " + name + " --recipe NAME --out FILE)");
        }
        if (convert::is_convertible(name)) {
            // Whether the engine runs it: the converter's rule set, before converting
            // (convert/support.h).
            if (const auto support = convert::model_support(convert::local_support_facts(name));
                support.verdict == convert::Verdict::Unsupported) {
                throw std::runtime_error(name + " cannot be run: " + support.reason);
            }
            if (log) { log("checkpoint " + name); }
            const auto cached = convert::cached_artifact(name, {}, log, false, options.enable_vision);
            out.kind            = ModelSource::Kind::Checkpoint;
            out.artifact_path   = cached.path.string();
            out.recipe          = cached.recipe;
            out.converted       = cached.converted;
            out.convert_seconds = cached.seconds;
            if (log) {
                log(std::string(cached.converted ? "converted" : "cached") + " artifact " +
                    out.artifact_path + " (recipe " + out.recipe + ")");
            }
            log_source_identity(log, out.artifact_path);
            return out;
        }
        if (fs::is_directory(name)) {
            throw std::runtime_error(name + " is a directory but neither a Hugging Face checkpoint "
                                            "(config.json beside safetensors weights) nor a GGUF "
                                            "model");
        }
        out.artifact_path = name;
        return out;
    }
    if (!convert::is_repository_spec(name)) {
        throw std::runtime_error("'" + name + "' is neither a file or directory nor a hub id "
                                 "(org/name[@revision])");
    }
    convert::AcquireOptions acquire;
    acquire.fetch.quant = options.quant;
    // A GGUF repository without --quant: the largest quantization that fits the asked context x
    // --max-concurrency on this GPU's class (serve/plan_offline.h), or a refusal before anything
    // is downloaded. Not for --prepare-only, which needs no GPU: it keeps the fetch's default.
    bool fitted = false;
    if (options.quant.empty() && !options.prepare_only) {
        if (const auto choice = largest_fitting_hub_quant(options, log)) {
            acquire.fetch.quant = choice->quant;
            out.planned_context = choice->launch_context;
            fitted              = true;
        }
    }
    if (!fitted && !options.prepare_only) { check_fits(options); }
    acquire.fetch.log      = log;
    acquire.convert.recipe = options.recipe;
    acquire.convert.log    = log;
    acquire.reuse          = true;
    acquire.repository     = fs::current_path();
    acquire.log            = log;
    if (options.with_drafters) { acquire.with_drafters = true; }
    acquire.keep_replaced = options.keep_replaced;
    acquire.moved_ref = options.refetch            ? convert::MovedRef::Refetch
                        : options.accept_converted ? convert::MovedRef::Accept
                                                   : convert::MovedRef::Notice;
    // On a terminal the notice stays on screen for a moment before the start goes on with the
    // converted weights (Ctrl+C stops it to choose); a service or ninferno-ui does not wait. The
    // weights served are the same either way.
    if (sys::stderr_is_terminal()) {
        acquire.on_moved_ref = [](const std::string&) {
            for (int left = 10; left > 0; --left) {
                std::fprintf(stderr, "\rserving the converted weights in %2d s (Ctrl+C to stop) ",
                             left);
                std::fflush(stderr);
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            std::fprintf(stderr, "\r%60s\r", "");
        };
    }
    acquire.vision = options.enable_vision;
    auto acquired       = convert::acquire_model(name, std::move(acquire));
    out.kind            = ModelSource::Kind::Hub;
    out.artifact_path   = acquired.artifact.string();
    out.repo            = acquired.fetched.repo;
    out.commit          = acquired.fetched.commit;
    out.recipe          = acquired.recipe;
    out.converted       = acquired.conversion.has_value();
    out.convert_seconds = acquired.conversion ? acquired.conversion->seconds : 0.0;
    if (log && acquired.drafters) {
        for (const auto& drafter : acquired.drafters->drafters) {
            log("drafter " + drafter.decl.name + " (" + drafter.decl.kind + "): " +
                (drafter.status.empty() ? (drafter.compatible ? "ok" : drafter.reason)
                                        : drafter.status));
        }
    }
    if (log) {
        log(std::string(out.converted ? "converted" : "cached") + " artifact " +
            out.artifact_path + " (" + out.repo + "@" + out.commit.substr(0, 12) + ", recipe " +
            out.recipe + ")");
    }
    log_source_identity(log, out.artifact_path);
    return out;
}

} // namespace ninferno::serve
