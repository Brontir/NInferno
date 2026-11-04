#include "ops/generic/build_manifest.h"

#include "ops/generic/route_table.h"

#include <nlohmann/json.hpp>

// The compilers and the architecture list are defined on this file alone (sources.cmake), so that
// no other object rebuilds when they change.
#ifndef NINFERNO_CUDA_COMPILER
#    define NINFERNO_CUDA_COMPILER "unknown"
#endif
#ifndef NINFERNO_CXX_COMPILER
#    define NINFERNO_CXX_COMPILER "unknown"
#endif
#ifndef NINFERNO_MANIFEST_CUDA_ARCHITECTURES
#    define NINFERNO_MANIFEST_CUDA_ARCHITECTURES "unknown"
#endif

namespace ninferno::ops::detail::generic {

namespace {

const char* family_key(Family family) {
    switch (family) {
    case Family::Scalar: return "scalar";
    case Family::RegisterMma: return "rmma";
    case Family::W4a4Mma: return "w4a4";
    case Family::TiledMma: return "tiled";
    case Family::W4a8Mma: return "w4a8";
    case Family::StagedMma: return "staged";
    case Family::Nvfp4TiledMma: return "nvfp4t";
    case Family::R8Mma: return "r8";
    case Family::GgmlMmq: return "ggml_mmq";
    case Family::GgmlVec: return "ggml_vec";
    case Family::I8Mma: return "i8";
    }
    return "unknown";
}

} // namespace

std::string build_manifest_json(const BuildManifest& m,
                                const std::map<std::string, std::string>& build) {
    nlohmann::json out;
    if (!build.empty()) {
        auto& section = out["build"] = nlohmann::json::object();
        for (const auto& [key, value] : build) section[key] = value;
    }
    out["arch"]             = m.arch;
    out["arch_specific"]      = m.arch_specific;
    out["native_nvfp4"]       = m.native_nvfp4;
    out["cuda_compiler"]      = NINFERNO_CUDA_COMPILER;
    out["cxx_compiler"]       = NINFERNO_CXX_COMPILER;
    out["cuda_architectures"] = NINFERNO_MANIFEST_CUDA_ARCHITECTURES;
    nlohmann::json families   = nlohmann::json::object();
    for (const ManifestSchedule& s : m.schedules) {
        nlohmann::json entry;
        entry["route"]    = route_name(s.route);
        entry["compiled"] = s.compiled;
        if (!s.only_formats.empty()) {
            auto& only = entry["only_formats"] = nlohmann::json::array();
            for (const QType q : s.only_formats) only.push_back(qtype_name(q));
        }
        if (!s.omitted_formats.empty()) {
            auto& omitted = entry["omitted_formats"] = nlohmann::json::array();
            for (const OmittedFormat& f : s.omitted_formats) {
                omitted.push_back(f.k_multiple == 0 ? std::string(qtype_name(f.qtype))
                                                    : std::string(qtype_name(f.qtype)) + "@k" +
                                                          std::to_string(f.k_multiple));
            }
        }
        families[family_key(s.route.family)].push_back(std::move(entry));
    }
    out["families"]          = std::move(families);
    auto& verify             = out["r8_verify_formats"] = nlohmann::json::array();
    for (const QType q : m.r8_verify_formats) verify.push_back(qtype_name(q));
    return out.dump(2);
}

} // namespace ninferno::ops::detail::generic
