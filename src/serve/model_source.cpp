#include "serve/model_source.h"
#include "artifact/source_identity.h"
#include "serve/plan_offline.h"

#include "convert/acquire.h"
#include "convert/support.h"
#include "convert/cache.h"
#include "core/ascii.h"
#include "core/host_memory.h"
#include "core/sys.h"
#include "product/speculative_options.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>
#include <optional>
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
void check_fits(const ServeOptions& options, std::optional<std::uint64_t> weight_bytes = {}) {
    OfflinePlan plan;
    try {
        plan = plan_without_load(options, weight_bytes);
    } catch (const std::exception&) { return; }
    if (plan.fits) { return; }
    throw std::runtime_error(options.model_source + " does not fit GPU " +
                             std::to_string(options.device) + " (nothing was downloaded): " +
                             plan.reason +
                             (weight_bytes ? " (the least its conversion writes)" : "") +
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

// The official recipes of a published quantization (NVFP4, FP8) keep its encoded matrices as
// published and quantize the rest themselves (Q8 projections and tables, a Q6 head): their
// artifact is smaller than the files, by how much the headers tell. nvidia/Qwen3.6-35B-A3B-NVFP4's
// 21.5 GiB of files were downloaded onto a 16 GB card before the start refused 19.4 GiB of weights.
bool keeps_encoded(const std::string& recipe) {
    return recipe == "dense_encoded" || recipe == "gemma4_published" ||
           recipe == "gemma4_26b_a4b_nvfp4" || recipe == "gemma4_31b_nvfp4" ||
           recipe == "qwen3_6_27b_nvfp4" || recipe == "qwen3_8_27b_nvfp4" ||
           recipe == "qwen3_6_35b_a3b_nvfp4";
}

// The least such a recipe writes: the encoded tensors (U8 codes, FP8, their scales) as they are,
// and every plain float tensor (16, 32 or 64 bits) at `plain_bits` a weight: 4 for the recipes that
// quantize them, below the smallest encoding they give one (Q6); 16 for gemma4_published, which
// quantizes nothing and keeps a plain tensor in BF16 at the least. A text artifact drops the vision
// and audio towers, so without --vision they are not counted. nullopt when a header cannot be read.
std::optional<std::uint64_t> encoded_floor_bytes(const convert::FetchPlan& fetch, bool vision,
                                                 std::uint64_t plain_bits) {
    std::map<std::string, std::uint64_t> dtypes;
    try {
        convert::FetchOptions where;
        where.cache = fetch.root;
        dtypes      = convert::safetensors_dtype_bytes(fetch, where, [vision](const std::string& name) {
            return vision || (name.find("vision") == std::string::npos &&
                              name.find("visual") == std::string::npos &&
                              name.find("audio") == std::string::npos);
        });
    } catch (const std::exception&) { return std::nullopt; }
    std::uint64_t bytes = 0;
    for (const auto& [dtype, size] : dtypes) {
        if (dtype == "BF16" || dtype == "F16") {
            bytes += size * plain_bits / 16;
        } else if (dtype == "F32") {
            bytes += size * plain_bits / 32;
        } else if (dtype == "F64") {
            bytes += size * plain_bits / 64;
        } else {
            bytes += size;
        }
    }
    return bytes == 0 ? std::nullopt : std::optional<std::uint64_t>(bytes);
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
    if (gguf || keeps_weights(recipe)) {
        check_fits(asked);
        return;
    }
    // A recipe that quantizes a BF16 checkpoint (the official Qwen3.6-35B-A3B one takes its experts
    // to 4-5 bits) makes an artifact whose size the start knows.
    if (!keeps_encoded(recipe)) { return; }
    const std::uint64_t plain_bits = recipe == "gemma4_published" ? 16 : 4;
    if (const auto floor = encoded_floor_bytes(fetch, options.enable_vision, plain_bits)) {
        check_fits(asked, floor);
    }
}

} // namespace

std::string drafters_required_by(const ServeOptions& options) {
    if (options.with_drafters) { return "--with-drafters"; }
    // --spec dflash / dflash2 runs a separate drafter; one given with --drafter replaces the
    // published one, which then need not be fetched.
    const SpeculativeBackend backend = options.speculative.backend;
    if ((backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2) &&
        options.speculative.drafter.empty()) {
        return std::string("--spec ") + product::speculative_backend_name(backend);
    }
    return {};
}

convert::Drafters hub_drafters(const ServeOptions& options, bool attach) {
    // Named by the user (--with-drafters, or a --spec backend that runs a separate drafter):
    // required. Only the automatic choice drafts: attached when it can be, else the model is
    // served without the drafter, with a warning.
    if (!drafters_required_by(options).empty()) { return convert::Drafters::Require; }
    return attach ? convert::Drafters::Attach : convert::Drafters::Default;
}

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
    // A GGUF model's separate drafters (DECISIONS 45): attached when the choice drafts with
    // one, as with --with-drafters. --prepare-only asks the same choice where a GPU can be read,
    // so the start with the same flags converts nothing more (#284); without one, only with
    // --with-drafters.
    bool drafters = options.with_drafters;
    if (options.quant.empty() && !options.prepare_only) {
        if (const auto choice = largest_fitting_hub_quant(options, log)) {
            acquire.fetch.quant = choice->quant;
            out.planned_context = choice->launch_context;
            drafters            = drafters || choice->drafters;
        }
    } else if (!options.quant.empty() && !drafters) {
        drafters = hub_start_attaches_drafters(options, log);
    }
    if (drafters && !options.with_drafters && log) {
        log("drafters | the choice drafts with a separate drafter: it will be attached (a separate "
            "download)");
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
    acquire.drafters       = hub_drafters(options, drafters);
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
