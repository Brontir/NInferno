#pragma once

// What `ninferno-serve <model>` names, made into an artifact the Engine loads:
//
//   a .ninferno artifact     loaded as is;
//   a checkpoint           a Hugging Face checkpoint directory or a GGUF model: converted once with
//                          the official recipe into <dir>/.ninferno-cache/ (convert/cache.h);
//   a hub id               org/name[@revision]: fetched into the Hugging Face cache, converted
//                          (the official recipe, or --recipe) into <cache>/.../ninferno/<commit>/
//                          with its drafters attached (a GGUF model's with --with-drafters),
//                          and reused by later starts (convert/acquire.h). Before the weights
//                          are downloaded, the architecture, the cache's free space and, for a
//                          model kept as published, the device's free memory are checked, and
//                          a model that cannot be served is refused with what to do instead.
//
// A path that exists wins over a hub id of the same spelling. Nothing else on the machine is
// searched.

#include "serve/serve_options.h"

#include <functional>
#include <string>

namespace ninferno::serve {

struct ModelSource {
    enum class Kind { Artifact, Checkpoint, Hub };
    Kind kind = Kind::Artifact;
    std::string artifact_path;
    std::string repo;   // the hub id without its revision (Hub)
    std::string commit; // the snapshot served (Hub)
    std::string recipe; // the conversion (Checkpoint, Hub)
    bool converted = false; // converted by this start
    double convert_seconds = 0.0;
};

// Throws std::runtime_error naming what the argument is not, or why the model cannot be served
// and what would serve it.
[[nodiscard]] ModelSource resolve_model_source(const ServeOptions& options,
                                               const std::function<void(const std::string&)>& log);

} // namespace ninferno::serve
