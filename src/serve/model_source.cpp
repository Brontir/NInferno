#include "serve/model_source.h"
#include "artifact/source_identity.h"
#include "serve/plan_offline.h"

#include "convert/acquire.h"
#include "convert/support.h"
#include "convert/cache.h"
#include "core/ascii.h"
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
// same memory plan the start makes). Asked when the fetch is about to download (FetchPlan), so it
// plans the files the fetch chose (a width selector resolved, the commit a moved branch is pinned
// to) and nothing is asked when the weights are here. A model the plan cannot read is left to the
// start.
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

// Recipes whose artifact is about as large as the checkpoint's weights (nothing quantized); a
// GGUF model's blocks are kept as published too.
bool keeps_weights(const std::string& recipe) {
    return recipe == "dense_published" || recipe == "moe_published" || recipe == "dense_bf16" ||
           recipe == "qwen3_5_bf16" || recipe == "qwen3_5_published";
}

void check_fits(const ServeOptions& options, const convert::FetchPlan& fetch,
                const std::string& recipe) {
    if (fetch.missing_bytes == 0) { return; }
    auto asked         = options;
    asked.model_source = fetch.repo + "@" + fetch.commit;
    bool gguf          = false;
    for (const auto& file : fetch.files) {
        const auto name = ascii_lower(fs::path(file).filename().string());
        if (name.ends_with(".gguf") && !name.starts_with("mmproj")) {
            asked.quant = convert::gguf_quant_label(fs::path(file).filename().string());
            gguf        = true;
            // A name without a quantization label (model.gguf): the plan could not name this
            // file, so the start checks it.
            if (asked.quant.empty()) { return; }
            break;
        }
    }
    // The plan reads the files' bytes: a recipe that quantizes a checkpoint (the official
    // Qwen3.6-35B-A3B one takes its BF16 experts to 4-5 bits) makes a smaller artifact, whose
    // size the start knows.
    if (!gguf && !keeps_weights(recipe)) { return; }
    check_fits(asked);
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
            auto facts = convert::local_support_facts(name);
            if (!options.chat_template_path.empty()) { facts.chat_template = true; } // --chat-template
            if (const auto refusal = convert::serve_refusal(convert::model_support(facts));
                !refusal.empty()) {
                throw std::runtime_error(name + " cannot be run: " + refusal);
            }
            if (log) { log("checkpoint " + name); }
            const auto cached = convert::cached_artifact(name, {}, log, false, options.enable_vision,
                                                         options.round_tiny_scales);
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
    // A GGUF model's separate drafters (choice/DECISIONS 45): attached when the choice drafts with
    // one, as with --with-drafters; --prepare-only, without a GPU to choose for, only with it.
    bool drafters = options.with_drafters;
    if (options.quant.empty() && !options.prepare_only) {
        if (const auto choice = largest_fitting_hub_quant(options, log)) {
            acquire.fetch.quant = choice->quant;
            out.planned_context = choice->launch_context;
            drafters            = drafters || choice->drafters;
        }
    } else if (!options.quant.empty() && !options.prepare_only && !drafters) {
        drafters = hub_start_attaches_drafters(options, log);
    }
    if (drafters && !options.with_drafters && log) {
        log("drafters | the choice drafts with a separate drafter: attached (a separate download)");
    }
    acquire.fetch.log      = log;
    acquire.convert.recipe            = options.recipe;
    acquire.convert.round_tiny_scales = options.round_tiny_scales;
    // --chat-template: the conversion's template too (a base model ships none).
    if (!options.chat_template_path.empty()) {
        acquire.convert.resources.emplace_back("chat_template.jinja", options.chat_template_path);
    }
    acquire.convert.log    = log;
    acquire.reuse          = true;
    acquire.repository     = fs::current_path();
    acquire.log            = log;
    if (drafters) { acquire.with_drafters = true; }
    acquire.keep_replaced = options.keep_replaced;
    // Also for the quantization chosen above: that choice is for an idle card of this class,
    // this check for the memory free now.
    if (!options.prepare_only) {
        acquire.before_download = [&options](const convert::FetchPlan& fetch,
                                             const std::string& recipe) {
            check_fits(options, fetch, recipe);
        };
    }
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
