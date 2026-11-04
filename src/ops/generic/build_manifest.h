#pragma once

// The compiled-variant manifest as a document (#307): build_manifest (generic_gemm.h) with the
// compilers that built the binary, so that a resource fact (registers, spills, static shared
// memory, which change with the codec, the epilogue and the compiler) can name the build it holds
// for. `ninferno-serve --build-manifest` prints it with the version and the git commit.

#include "ops/generic/generic_gemm.h"

#include <map>
#include <string>

namespace ninferno::ops::detail::generic {

// {"arch", "arch_specific", "native_nvfp4", "cuda_compiler", "cxx_compiler", "cuda_architectures",
//  "families": {"<family>": [{"route", "compiled", "only_formats"?, "omitted_formats"?}, ...]},
//  "r8_verify_formats": [...]}: formats by their route-table names (qtype_name), an omission at
// K multiples as "<format>@k<multiple>". `build` is the caller's own section (the decision
// fingerprint's build fields: version, git commit), kept as given under "build"; the document is
// indented by two.
[[nodiscard]] std::string build_manifest_json(const BuildManifest& manifest,
                                              const std::map<std::string, std::string>& build = {});

} // namespace ninferno::ops::detail::generic
