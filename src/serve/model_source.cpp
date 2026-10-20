#include "serve/model_source.h"

#include "convert/acquire.h"
#include "convert/cache.h"
#include "core/host_memory.h"
#include "core/units.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace ninferno::serve {
namespace {

namespace fs = std::filesystem;

// Recipes whose artifact is about as large as the checkpoint's weights (nothing quantized).
bool keeps_weights(const std::string& recipe) {
    return recipe == "dense_published" || recipe == "moe_published" || recipe == "dense_bf16" ||
           recipe == "qwen3_5_bf16";
}

// What a served model needs beside its weights at the least: a context's KV cache, the
// workspaces and the CUDA graphs (a tenth of the device, at least 1 GiB).
std::uint64_t serving_reserve(std::size_t total) {
    return std::max<std::uint64_t>(total / 10, kGiB);
}

// Before a hub model is downloaded: weights kept as published that exceed the device's free
// memory cannot be served on it, whatever the context.
void check_device(const convert::FetchPlan& plan, const std::string& recipe, int device) {
    if (!keeps_weights(recipe)) { return; }
    std::size_t free = 0, total = 0;
    if (cudaSetDevice(device) != cudaSuccess || cudaMemGetInfo(&free, &total) != cudaSuccess) {
        cudaGetLastError();
        return; // the Engine reports the device itself
    }
    if (plan.total_bytes + serving_reserve(total) <= free) { return; }
    const bool moe  = recipe == "moe_published";
    const bool qwen = recipe.starts_with("qwen3_5_");
    throw std::runtime_error(
        plan.repo + " as published is " + format_gib(plan.total_bytes) + " of weights; GPU " +
        std::to_string(device) + " has " + format_gib(free) + " free of " + format_gib(total) +
        ", and serving needs " + format_gib(serving_reserve(total)) +
        " beside the weights, so it "
        "cannot be served here (nothing was downloaded). Options: a quantized release of "
        "the model (GGUF, FP8, NVFP4, MXFP4), --recipe " +
        (qwen  ? std::string("qwen3_5_q8 (8-bit, about half the size)")
         : moe ? std::string("moe_q8")
               : std::string("dense_q8 (8-bit, about half the size) or --recipe dense_nvfp4")) +
        ", or a GPU with more free memory.");
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
    acquire.fetch.quant    = options.quant;
    acquire.fetch.log      = log;
    acquire.convert.recipe = options.recipe;
    acquire.convert.log    = log;
    acquire.reuse          = true;
    acquire.repository     = fs::current_path();
    acquire.log            = log;
    if (options.with_drafters) { acquire.with_drafters = true; }
    acquire.vision = options.enable_vision;
    if (!options.prepare_only) {
        const int device        = options.device;
        acquire.before_download = [device](const convert::FetchPlan& plan,
                                           const std::string& recipe) {
            check_device(plan, recipe, device);
        };
    }
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
    return out;
}

} // namespace ninferno::serve
