// Per-GPU-class route_as overrides for the shape-owned Linear tables (ops/common/shape_routes.h),
// read by the one route-table loader (include/ninferno/ops/route_table.h): the band lookup, the
// class check, the probe bypass and the refusals.
#include "core/device_class.h"
#include "ninferno/ops/route_table.h"
#include "ops/common/route_probe.h"
#include "ops/common/shape_routes.h"
#include "core/sha256.h"
#include "support/temp_dir.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

using namespace ninferno;
using namespace ninferno::ops::detail;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "shape routes: " << what << '\n';
        ++failures;
    }
}

// The table's path in UTF-8, as the route-table API takes it; removed with the process's scratch
// directory.
std::string write_file(const std::string& name, const std::string& text) {
    static const TempDir scratch("ninferno_shape_routes");
    const auto path = scratch.path() / name;
    std::ofstream(path) << text;
    return utf8_path(path);
}

std::string table_json(const std::string& gpu_class, const std::string& entries) {
    return R"({"schema": "ninferno-route-table/1", "gpu": {"class": ")" + gpu_class +
           R"("}, "entries": [)" + entries + "]}";
}

bool throws_on_load(const std::string& path) {
    try {
        (void)ops::load_route_table(path);
    } catch (const std::runtime_error&) { return true; }
    return false;
}

constexpr ShapeRouteKey kA4{QType::NVFP4, ShapeCompute::A4, 2816, 4096};
constexpr ShapeRouteKey kA16{QType::NVFP4, ShapeCompute::A16, 2816, 4096};
constexpr ShapeRouteKey kQ8{QType::Q8_G32_FP16, ShapeCompute::A16, 4096, 1024};

} // namespace

int main() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver || count == 0) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    const auto& device = device_class();
    const std::string chip = device.chip;

    const std::string entries =
        R"({"format": "nvfp4", "compute": "a4", "epilogue": "store", "n": 2816, "k": 4096,
            "route_as": [[256, 384, 512], [512, 768, 1024]]},
           {"format": "q8", "compute": "a16", "epilogue": "store", "n": 4096, "k": 1024,
            "route_as": [[2, 2, 3]]},
           {"format": "q8", "compute": "a16", "epilogue": "store", "n": 2048, "k": 4096,
            "bands": [[1, 16, "rmma.m1.n1.w1x8"]]})";
    const auto good = write_file("good.json", table_json(chip, entries));

    // Loaded for the device's chip; the entry that carries only bands is another consumer's.
    const auto status = ops::load_route_table(good);
    expect(status.loaded, "own-chip table not loaded: " + status.message);
    expect(status.route_as_entries == 2, "expected 2 route_as problems, got " +
                                             std::to_string(status.route_as_entries));

    // Bands: inside, at the edges, between and outside; never below the call's width.
    expect(route_tokens(kA4, 256) == 512, "a4 T=256 -> 512");
    expect(route_tokens(kA4, 384) == 512, "a4 T=384 -> 512");
    expect(route_tokens(kA4, 385) == 385, "a4 T=385 between bands stays");
    expect(route_tokens(kA4, 600) == 1024, "a4 T=600 -> 1024");
    expect(route_tokens(kA4, 255) == 255, "a4 T=255 below every band stays");
    expect(route_tokens(kA4, 2048) == 2048, "a4 T=2048 above every band stays");
    expect(route_tokens(kA16, 256) == 256, "the A16 table of the shape has no override");
    expect(route_tokens(kQ8, 2) == 3, "q8 T=2 -> 3");
    expect(route_tokens(kQ8, 1) == 1, "q8 T=1 stays");
    expect(route_tokens({QType::Q8_G32_FP16, ShapeCompute::A16, 2048, 4096}, 4) == 4,
           "a bands-only entry does not steer the shape table");

    // A live route probe measures the shape table's own routes: overrides stand aside.
    {
        const RouteProbe probe(0, 1);
        expect(route_tokens(kA4, 256) == 256, "forced-class probe bypasses the override");
    }
    {
        const RouteProbe probe(300);
        expect(route_tokens(kA4, 256) == 300, "width probe answers, not the table");
    }
    expect(route_tokens(kA4, 256) == 512, "override back after the probe");

    // A table of another class is not applied and the loaded one stays.
    const auto other = write_file("other.json", table_json("gb999", entries));
    const auto refused = ops::load_route_table(other);
    expect(!refused.loaded, "a table of another class was applied");
    expect(route_tokens(kA4, 256) == 512, "the loaded table did not stay");

    // The architecture key of the chain is a class of this device too.
    const auto arch = write_file("arch.json", table_json(device.arch(), entries));
    expect(ops::load_route_table(arch).loaded, "the architecture key was refused");

    // Refusals.
    const auto overlap = write_file(
        "overlap.json",
        table_json(chip, R"({"format": "q8", "compute": "a16", "epilogue": "store", "n": 1,
                             "k": 1, "route_as": [[2, 8, 12], [8, 16, 24]]})"));
    expect(throws_on_load(overlap), "overlapping bands accepted");
    const auto bad_band = write_file(
        "badband.json", table_json(chip, R"({"format": "q8", "compute": "a16", "epilogue":
                                              "store", "n": 1, "k": 1, "route_as": [[0, 8, 12]]})"));
    expect(throws_on_load(bad_band), "a band from T=0 accepted");
    const auto epilogue = write_file(
        "epilogue.json", table_json(chip, R"({"format": "q8", "compute": "a16", "epilogue": "add",
                                              "n": 1, "k": 1, "route_as": [[2, 8, 12]]})"));
    expect(throws_on_load(epilogue), "route_as on a fused epilogue accepted");
    const auto format = write_file(
        "format.json", table_json(chip, R"({"format": "q3", "compute": "a16", "epilogue": "store",
                                            "n": 1, "k": 1, "route_as": [[2, 8, 12]]})"));
    expect(throws_on_load(format), "an unknown format accepted");
    const auto dup = write_file(
        "dup.json", table_json(chip, R"({"format": "q8", "compute": "a16", "n": 1, "k": 1,
                                         "route_as": [[2, 2, 3]]},
                                        {"format": "q8", "compute": "a16", "n": 1, "k": 1,
                                         "route_as": [[4, 4, 6]]})"));
    expect(throws_on_load(dup), "a duplicate problem accepted");
    const auto schema = write_file("schema.json", R"({"schema": "other", "gpu": {"class": "x"},
                                                     "entries": []})");
    expect(throws_on_load(schema), "another schema accepted");
    const auto broken = write_file("broken.json", "{not json");
    expect(throws_on_load(broken), "malformed JSON accepted");

    ops::clear_route_table();
    expect(route_tokens(kA4, 256) == 256, "cleared table still steers");
    expect(!ops::route_table_status().loaded, "cleared table reported loaded");

    if (failures == 0) std::cout << "shape routes: ok\n";
    return failures == 0 ? 0 : 1;
}
