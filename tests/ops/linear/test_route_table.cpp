// The per-GPU-class route table (include/ninferno/ops/route_table.h): the default lookup along the
// class chain (a ':' in a key is '-' in its file name), the generic path's band lookup and prefill
// route, route_as through the same loader, the refusals (malformed file, other GPU class, a prefill
// route that is not token-invariant, a route_as width below its band, compute classes a format does
// not have, duplicates) and tolerance of keys no consumer reads. Every shipped routes/*.json must
// pass the same validation.
#include "core/device_class.h"
#include "ninferno/ops/route_table.h"
#include "ops/common/shape_routes.h"
#include "ops/generic/route_table.h"
#include "ops/moe/moe_internal.h"
#include "ops/op_tester.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

namespace generic = ninferno::ops::detail::generic;
namespace detail  = ninferno::ops::detail;

const std::filesystem::path& scratch_dir() {
    static const std::filesystem::path dir = [] {
        auto p = std::filesystem::temp_directory_path() /
                 ("ninferno_route_table_test_" + std::to_string(::getpid()));
        std::filesystem::create_directories(p);
        return p;
    }();
    return dir;
}

std::string write_file(const std::string& name, const std::string& text) {
    const auto path = scratch_dir() / (name + ".json");
    std::ofstream(path) << text;
    return path.string();
}

std::string table(const std::string& gpu_class, const std::string& entries) {
    return "{\"schema\": \"ninferno-route-table/1\", \"gpu\": {\"class\": \"" + gpu_class +
           "\", \"name\": \"test\"}, \"entries\": [" + entries + "]}";
}

int check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL route_table: " << what << '\n'; }
    return ok ? 0 : 1;
}

std::string route_or_none(const std::optional<generic::Route>& r) {
    return r ? generic::route_name(*r) : std::string("none");
}

bool refused(const std::string& gpu, const std::string& name, const std::string& entries) {
    try {
        (void)ninferno::ops::load_route_table(write_file(name, table(gpu, entries)));
    } catch (const std::runtime_error&) { return true; }
    return false;
}

} // namespace

// A malformed default table (NINFERNO_ROUTE_TABLE) is not applied, and the status says which file
// and why instead of reporting that nothing was loaded. Run in its own process: the
// default lookup happens once.
int malformed_default() {
    const auto path = scratch_dir() / "malformed-default.json";
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << "{\"schema\": \"ninferno-route-table/1\", \"gpu\": ";
    ::setenv("NINFERNO_ROUTE_TABLE", path.string().c_str(), 1);
    const auto s = ninferno::ops::route_table_status();
    std::cout << "malformed default table: loaded=" << s.loaded << " message: " << s.message
              << '\n';
    int failures = check(!s.loaded, "a malformed default table is not applied");
    failures += check(s.path == path.string() && s.message.find("refused") != std::string::npos &&
                          s.message.find(path.string()) != std::string::npos,
                      "the status names the refused file and the reason: " + s.message);
    std::error_code ec;
    std::filesystem::remove_all(scratch_dir(), ec);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " route_table malformed default\n";
    return failures == 0 ? 0 : 1;
}

// The default lookup fills the problems the device's own table lacks from its most similar chip's
// table, keeps its own entry where both have one, and says so in the status; an explicit table
// stands alone. Run in its own process: the default lookup happens once.
int similar_default() {
    const auto similar = ninferno::device_class().similar_chips();
    if (similar.empty()) {
        std::cout << "SKIP: no similar chip of this architecture\n";
        return 77;
    }
    const std::string gpu = ninferno::ops::route_gpu_class();
    std::string file      = gpu;
    for (char& c : file) {
        if (c == ':') c = '-';
    }
    const auto dir = scratch_dir() / "similar";
    std::filesystem::create_directories(dir);
    const auto entry = [](int k, const char* route) {
        return std::string(R"({"format": "q8", "compute": "a16", "epilogue": "store", "n": 4096, "k": )") +
               std::to_string(k) + R"(, "bands": [[1, 16, ")" + route + R"("]]})";
    };
    std::ofstream(dir / (file + ".json")) << table(gpu, entry(1024, "rmma.m1.n1.w1x8"));
    std::ofstream(dir / (similar.front() + ".json"))
        << table(similar.front(), entry(1024, "rmma.m1.n1.w1x4") + ", " + entry(2048, "rmma.m1.n1.w2x2"));
    // A table measured on a board of another instruction set lends nothing, however near its SMs.
    const int foreign_cc = ninferno::device_class().compute_capability() == 70 ? 75 : 70;
    std::ofstream(dir / "foreign.json")
        << "{\"schema\": \"ninferno-route-table/1\", \"gpu\": {\"class\": \"zz999\", \"name\": \"test\", "
           "\"board\": {\"cc\": " + std::to_string(foreign_cc) + ", \"sms\": " +
               std::to_string(ninferno::device_class().sms) + "}}, \"entries\": [" + entry(4096, "rmma.m1.n1.w1x4") +
               "]}";
    ::unsetenv("NINFERNO_ROUTE_TABLE");
    ::setenv("NINFERNO_ROUTE_DIR", dir.string().c_str(), 1);
    const generic::Problem own{ninferno::QType::Q8_G32_FP16, ninferno::ops::LinearPolicy::A16Only,
                               4096, 1024, generic::Epilogue::Store};
    const generic::Problem lent{ninferno::QType::Q8_G32_FP16, ninferno::ops::LinearPolicy::A16Only,
                                4096, 2048, generic::Epilogue::Store};
    const auto s = ninferno::ops::route_table_status();
    std::cout << "similar default table: " << s.message << '\n';
    int failures = check(s.loaded && s.gpu_class == gpu, "the own table is the default: " + s.message);
    failures += check(route_or_none(generic::table_route(own, 4, false)) == "rmma.m1.n1.w1x8",
                      "a problem of the own table keeps its route");
    failures += check(route_or_none(generic::table_route(lent, 4, false)) == "rmma.m1.n1.w2x2",
                      "a problem the own table lacks takes the similar chip's entry");
    failures += check(s.borrowed_entries == 1 && s.borrowed_from.find(similar.front()) == 0 &&
                          s.message.find("from the next nearest (" + similar.front() + " 1)") != std::string::npos,
                      "the status says what was borrowed from whom: " + s.message);
    failures += check(!generic::table_claims(lent, 4, false), "a borrowed entry claims nothing");
    const generic::Problem foreign{ninferno::QType::Q8_G32_FP16, ninferno::ops::LinearPolicy::A16Only,
                                   4096, 4096, generic::Epilogue::Store};
    failures += check(!generic::table_route(foreign, 4, false).has_value(),
                      "a table of another instruction set lends nothing");
    // An explicit table stands alone.
    (void)ninferno::ops::load_route_table((dir / (file + ".json")).string());
    failures += check(!generic::table_route(lent, 4, false).has_value(),
                      "an explicit table borrows nothing");
    std::error_code ec;
    std::filesystem::remove_all(scratch_dir(), ec);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " route_table similar default\n";
    return failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (ninferno::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (argc == 2 && std::string(argv[1]) == "--malformed-default") { return malformed_default(); }
    if (argc == 2 && std::string(argv[1]) == "--similar-default") { return similar_default(); }
    int failures = 0;
    const auto& device    = ninferno::device_class();
    const std::string gpu = ninferno::ops::route_gpu_class();

    // Default lookup, before anything else touches the table: the device's variant key
    // ("gb202:170sm") is found as "gb202-170sm.json" in NINFERNO_ROUTE_DIR, and keeps its ':'
    // inside.
    {
        std::string file = gpu;
        for (char& c : file) {
            if (c == ':') c = '-';
        }
        const auto dir = scratch_dir() / "dir";
        std::filesystem::create_directories(dir);
        std::ofstream(dir / (file + ".json")) << table(gpu, "");
        ::unsetenv("NINFERNO_ROUTE_TABLE");
        ::setenv("NINFERNO_ROUTE_DIR", dir.string().c_str(), 1);
        const auto s = ninferno::ops::route_table_status();
        failures += check(s.loaded && s.gpu_class == gpu &&
                              std::filesystem::path(s.path).filename() == file + ".json",
                          "the default lookup finds " + file + ".json for " + gpu + ": " + s.message);
        failures += check(ninferno::ops::route_table_dir() == dir.string(),
                          "NINFERNO_ROUTE_DIR names the dir");
    }

    const generic::Problem q8{ninferno::QType::Q8_G32_FP16, ninferno::ops::LinearPolicy::A16Only,
                              4096, 1024, generic::Epilogue::Store};
    const generic::Problem other{ninferno::QType::Q8_G32_FP16, ninferno::ops::LinearPolicy::A16Only,
                                 4096, 2048, generic::Epilogue::Store};
    const detail::ShapeRouteKey a4{ninferno::QType::NVFP4, detail::ShapeCompute::A4, 14336, 5120};
    const detail::ShapeRouteKey both{ninferno::QType::Q8_G32_FP16, detail::ShapeCompute::A16, 4096,
                                     1024};
    const detail::ShapeRouteKey a8{ninferno::QType::FP8_E4M3FN_ROW_BF16, detail::ShapeCompute::A8,
                                   5120, 6144};

    // One file, both dialects: bands + prefill for the generic path, route_as for the shape
    // tables (one entry carries both), an "a8" route_as entry, an entry without an epilogue
    // (store), and keys no consumer reads.
    const std::string good = table(
        gpu, R"({"format": "q8", "compute": "a16", "epilogue": "store", "n": 4096, "k": 1024,
                 "bands": [[1, 4, "rmma.m1.n1.w1x8"], [5, 63, "rmma.m1.n1.w1x4"],
                           [64, 16777216, "rmma.m1.n2.w2x2"]],
                 "prefill": "rmma.m1.n8.w4x1", "route_as": [[2, 3, 4]], "note": "ignored"},
                {"format": "nvfp4", "compute": "a4", "epilogue": "store", "n": 14336,
                 "k": 5120, "route_as": [[48, 96, 192]], "evidence_us": [[48, 1.0, 0.5]]},
                {"format": "fp8", "compute": "a8", "n": 5120, "k": 6144,
                 "route_as": [[20, 40, 64]]})");
    const auto status = ninferno::ops::load_route_table(write_file("good", good));
    failures += check(status.loaded && status.entries == 3 && status.generic_entries == 1 &&
                          status.route_as_entries == 3,
                      "a valid table loads, both dialects counted: " + status.message);
    failures += check(route_or_none(generic::table_route(q8, 1, false)) == "rmma.m1.n1.w1x8",
                      "T=1 takes the first band");
    failures += check(route_or_none(generic::table_route(q8, 16, false)) == "rmma.m1.n1.w1x4",
                      "T=16 takes the second band");
    failures += check(route_or_none(generic::table_route(q8, 100, false)) == "rmma.m1.n2.w2x2",
                      "T=100 outside the prefill profile takes the third band");
    failures += check(route_or_none(generic::table_route(q8, 16, true)) == "rmma.m1.n1.w1x4",
                      "T=16 under the prefill profile keeps the band (prefill starts at 17)");
    failures += check(route_or_none(generic::table_route(q8, 17, true)) == "rmma.m1.n8.w4x1",
                      "T=17 under the prefill profile takes the prefill route");
    failures += check(route_or_none(generic::table_route(q8, 5000, true)) == "rmma.m1.n8.w4x1",
                      "T=5000 under the prefill profile takes the prefill route");
    failures += check(!generic::table_route(other, 1, false), "an absent problem has no route");
    failures += check(detail::route_tokens(both, 2) == 4 && detail::route_tokens(both, 3) == 4 &&
                          detail::route_tokens(both, 5) == 5,
                      "the entry with bands also carries route_as for the shape table");
    failures += check(detail::route_tokens(a4, 48) == 192 && detail::route_tokens(a4, 97) == 97,
                      "an a4 route_as band");
    failures += check(detail::route_tokens(a8, 20) == 64 && detail::route_tokens(a8, 41) == 41,
                      "an a8 route_as band is keyed by the FP8 A8 table");
    failures += check(detail::route_tokens({ninferno::QType::FP8_E4M3FN_ROW_BF16,
                                            detail::ShapeCompute::A16, 5120, 6144},
                                           20) == 20,
                      "the a8 band does not steer the shape's A16 table");

    // Every key of the device's class chain is accepted: chip, architecture, "*".
    for (const std::string& key : {device.chip, device.arch(), std::string("*")}) {
        const auto s = ninferno::ops::load_route_table(write_file("key", table(key, "")));
        failures += check(s.loaded, "a table keyed " + key + " is applied: " + s.message);
    }
    (void)ninferno::ops::load_route_table(write_file("good", good));

    // Another chip (or a chip variant of another SM count): not applied, the previous table stays.
    const auto foreign = ninferno::ops::load_route_table(write_file("foreign", table("zz999", "")));
    failures += check(!ninferno::ops::load_route_table(
                           write_file("variant", table(device.chip + ":1sm", "")))
                           .loaded,
                      "a table of another SM variant of this chip is not applied");
    failures += check(!foreign.loaded, "a table of another class is not applied");
    failures += check(route_or_none(generic::table_route(q8, 1, false)) == "rmma.m1.n1.w1x8" &&
                          detail::route_tokens(a4, 48) == 192,
                      "the loaded table stays after a refused one");

    // Prefill bands: under the prefill profile a band of routes that sum as the prefill route does
    // (WarpsK == 1 register MMA, staged) narrows the prefill route by width; outside the profile
    // and between the bands the entry's own routes answer.
    {
        const generic::Problem bf16{ninferno::QType::BF16, ninferno::ops::LinearPolicy::A16Only,
                                    3072, 4096, generic::Epilogue::Store};
        const auto s = ninferno::ops::load_route_table(write_file("pbands", table(
            gpu, R"({"format": "bf16", "compute": "a16", "epilogue": "store", "n": 3072, "k": 4096,
                     "bands": [[1, 16777216, "rmma.m1.n1.w1x8"]], "prefill": "staged.128x128.w2x4.s3",
                     "prefill_bands": [[17, 1023, "staged.64x128.w1x4.s4"],
                                       [2048, 4095, "rmma.m1.n8.w4x1"]]})")));
        failures += check(s.loaded && s.generic_entries == 1, "prefill bands load: " + s.message);
        failures += check(route_or_none(generic::table_route(bf16, 16, true)) == "rmma.m1.n1.w1x8",
                          "T=16 under the prefill profile keeps the band");
        failures += check(route_or_none(generic::table_route(bf16, 17, true)) == "staged.64x128.w1x4.s4" &&
                              route_or_none(generic::table_route(bf16, 1023, true)) ==
                                  "staged.64x128.w1x4.s4",
                          "T=17..1023 under the prefill profile take the first prefill band");
        failures += check(route_or_none(generic::table_route(bf16, 1024, true)) ==
                              "staged.128x128.w2x4.s3",
                          "T=1024 (between prefill bands) takes the prefill route");
        failures += check(route_or_none(generic::table_route(bf16, 3000, true)) == "rmma.m1.n8.w4x1",
                          "T=3000 takes the second prefill band (a WarpsK == 1 register MMA)");
        failures += check(route_or_none(generic::table_route(bf16, 512, false)) == "rmma.m1.n1.w1x8",
                          "outside the prefill profile the bands answer");
        failures += check(refused(gpu, "pband_split", R"({"format": "bf16", "compute": "a16",
                                     "n": 3072, "k": 4096, "prefill": "staged.128x128.w2x4.s3",
                                     "prefill_bands": [[17, 64, "rmma.m1.n1.w1x8"]]})"),
                          "a prefill band that sums in another order (K split) is refused");
        failures += check(refused(gpu, "pband_low", R"({"format": "bf16", "compute": "a16",
                                     "n": 3072, "k": 4096, "prefill": "staged.128x128.w2x4.s3",
                                     "prefill_bands": [[16, 64, "staged.64x128.w1x4.s4"]]})"),
                          "a prefill band below T = 17 is refused");
        failures += check(refused(gpu, "pband_alone", R"({"format": "bf16", "compute": "a16",
                                     "n": 3072, "k": 4096, "bands": [],
                                     "prefill_bands": [[17, 64, "staged.64x128.w1x4.s4"]]})"),
                          "prefill bands without a prefill route are refused");
        (void)ninferno::ops::load_route_table(write_file("good", good));
    }

    // Refusals. Chunk invariance first: a prefill route with a K split (not token-invariant), and
    // a route_as band that would ask a table about fewer tokens than the band's calls have.
    failures += check(refused(gpu, "splitk", R"({"format": "q8", "compute": "a16", "epilogue": "store",
                                     "n": 4096, "k": 1024, "bands": [],
                                     "prefill": "rmma.m1.n1.w1x8"})"),
                      "a K-split prefill route is refused");
    failures += check(refused(gpu, "narrow", R"({"format": "q8", "compute": "a16", "n": 4096,
                                     "k": 1024, "route_as": [[17, 64, 32]]})"),
                      "a route_as width below its band's t_max is refused");
    failures += check(refused(gpu, "overlap", R"({"format": "q8", "compute": "a16", "epilogue": "store",
                                      "n": 4096, "k": 1024,
                                      "bands": [[1, 8, "rmma.m1.n1.w1x8"], [8, 9, "rmma.m1.n1.w1x4"]]})"),
                      "overlapping bands are refused");
    failures += check(refused(gpu, "overlap_as", R"({"format": "q8", "compute": "a16", "n": 1,
                                     "k": 1, "route_as": [[2, 8, 12], [8, 16, 24]]})"),
                      "overlapping route_as bands are refused");
    failures += check(refused(gpu, "unknown", R"({"format": "q8", "compute": "a16", "epilogue": "store",
                                      "n": 4096, "k": 1024, "bands": [[1, 8, "rmma.m9.n9.w9x9"]]})"),
                      "an unknown route name is refused");
    failures += check(refused(gpu, "a8bands", R"({"format": "fp8", "compute": "a8", "n": 5120,
                                      "k": 6144, "bands": [[1, 8, "rmma.m1.n1.w1x8"]]})"),
                      "bands on a8 (no generic A8 schedule) are refused");
    failures += check(refused(gpu, "a4q8", R"({"format": "q8", "compute": "a4", "n": 4096,
                                      "k": 1024, "route_as": [[2, 2, 3]]})"),
                      "a4 on a format without W4A4 is refused");
    failures += check(refused(gpu, "a8nvfp4", R"({"format": "nvfp4", "compute": "a8", "n": 4096,
                                      "k": 1024, "route_as": [[2, 2, 3]]})"),
                      "a8 on a format without W8A8 is refused");
    failures += check(refused(gpu, "fused", R"({"format": "q8", "compute": "a16", "epilogue": "add",
                                      "n": 4096, "k": 1024, "route_as": [[2, 8, 12]]})"),
                      "route_as on a fused epilogue is refused");
    failures += check(refused(gpu, "dup", R"({"format": "q8", "compute": "a16", "n": 4096, "k": 1024,
                                      "bands": [[1, 8, "rmma.m1.n1.w1x8"]]},
                                     {"format": "q8", "compute": "a16", "epilogue": "store",
                                      "n": 4096, "k": 1024, "route_as": [[2, 2, 3]]})"),
                      "a duplicate problem (one per dialect) is refused");
    failures += check(refused(gpu, "zero", R"({"format": "q8", "compute": "a16", "n": 1, "k": 1,
                                      "route_as": [[0, 8, 12]]})"),
                      "a band from T=0 is refused");
    failures += check(refused(gpu, "format", R"({"format": "q3", "compute": "a16", "n": 1, "k": 1,
                                      "route_as": [[2, 8, 12]]})"),
                      "an unknown format is refused");
    failures += check(route_or_none(generic::table_route(q8, 1, false)) == "rmma.m1.n1.w1x8",
                      "a refused file leaves the active table");
    bool threw = false;
    try {
        (void)ninferno::ops::load_route_table(write_file("malformed", "{\"schema\": "));
    } catch (const std::runtime_error&) { threw = true; }
    failures += check(threw, "malformed JSON is refused");
    threw = false;
    try {
        (void)ninferno::ops::inspect_route_table(write_file("schema", R"({"schema": "other"})"));
    } catch (const std::runtime_error&) { threw = true; }
    failures += check(threw, "another schema is refused by inspect as well");

    // The generic MoE Op's entries: a plan per band and a prefill plan per exact layer; a prefill
    // plan with K split across warps is refused.
    {
        namespace moe = ninferno::ops::detail::moe;
        const auto moe_table = [&](const std::string& moe_entries) {
            return "{\"schema\": \"ninferno-route-table/1\", \"gpu\": {\"class\": \"" + gpu +
                   "\"}, \"entries\": [], \"moe\": [" + moe_entries + "]}";
        };
        const std::string entry =
            R"({"format": "mxfp4", "experts": 32, "top_k": 4, "hidden": 2880, "inter": 2880,
                "bands": [[1, 1, "rmma.m1.n1.w1x8|rmma.m1.n1.w1x4"],
                          [2, 16, "rmma.m1.n1.w4x1|rmma.m1.n4.w4x1"]],
                "prefill": "rmma.m1.n8.w4x1|rmma.m2.n8.w4x1"})";
        const auto s = ninferno::ops::load_route_table(write_file("moe", moe_table(entry)));
        failures += check(s.loaded && s.entries == 1, "a moe entry loads: " + s.message);
        const moe::MoeRouteKey key{
            ninferno::QType::MXFP4, ninferno::QType::MXFP4, 32, 4, 2880, 2880};
        const auto name = [](const std::optional<moe::MoePlan>& p) {
            return p ? moe::plan_name(*p) + " job " + std::to_string(p->job_columns)
                     : std::string("none");
        };
        failures += check(name(moe::table_plan(key, 1, false)) ==
                              "rmma.m1.n1.w1x8|rmma.m1.n1.w1x4 job 8",
                          "moe T=1 takes the first band");
        failures += check(name(moe::table_plan(key, 9, false)) ==
                              "rmma.m1.n1.w4x1|rmma.m1.n4.w4x1 job 32",
                          "moe T=9 takes the second band; the job width is the wider schedule's");
        failures += check(name(moe::table_plan(key, 17, true)) ==
                              "rmma.m1.n8.w4x1|rmma.m2.n8.w4x1 job 64",
                          "moe T=17 under the prefill profile takes the prefill plan");
        failures += check(!moe::table_plan(key, 40, false), "moe T=40 outside every band: none");
        failures += check(!moe::table_plan({ninferno::QType::MXFP4, ninferno::QType::MXFP4, 32, 4,
                                            2880, 2816},
                                           1, false),
                          "another layer: none");
        // A mixed layer (a GGUF mixture: q5_1 gate/up over q5_0 down) has its own key, and
        // neither same-format layer of its geometry takes its bands.
        {
            const auto mixed_status = ninferno::ops::load_route_table(write_file(
                "moe_mixed",
                moe_table(R"({"format": "q51", "down_format": "q50", "experts": 128,
                    "top_k": 6, "hidden": 2688, "inter": 1856,
                    "bands": [[1, 16, "rmma.m1.n1.w1x8|rmma.m1.n1.w1x4"]]})")));
            failures += check(mixed_status.loaded && mixed_status.entries == 1,
                              "a mixed moe entry loads: " + mixed_status.message);
            const moe::MoeRouteKey mixed{
                ninferno::QType::GGML_Q5_1, ninferno::QType::GGML_Q5_0, 128, 6, 2688, 1856};
            auto same_up       = mixed;
            same_up.down_qtype = ninferno::QType::GGML_Q5_1;
            auto same_down     = mixed;
            same_down.qtype    = ninferno::QType::GGML_Q5_0;
            failures += check(name(moe::table_plan(mixed, 4, false)) ==
                                  "rmma.m1.n1.w1x8|rmma.m1.n1.w1x4 job 8",
                              "the mixed layer takes its band");
            failures += check(!moe::table_plan(same_up, 4, false) &&
                                  !moe::table_plan(same_down, 4, false),
                              "a same-format layer of that geometry: none");
            (void)ninferno::ops::load_route_table(write_file("moe", moe_table(entry)));
        }
        bool refused_moe = false;
        try {
            (void)ninferno::ops::load_route_table(write_file(
                "moe_bad", moe_table(R"({"format": "mxfp4", "experts": 32, "top_k": 4,
                    "hidden": 2880, "inter": 2880, "prefill": "rmma.m1.n1.w1x8|rmma.m1.n8.w4x1"})")));
        } catch (const std::runtime_error&) { refused_moe = true; }
        failures += check(refused_moe, "a moe prefill plan that splits K is refused");
        // R8 bands: only on a format the R8 kernel reads and only up to T = 4 (the A8 decode
        // widths).
        const auto r8_entry = [](const char* format, int t_max) {
            return std::string(R"({"format": ")") + format +
                   R"(", "experts": 32, "top_k": 4, "hidden": 2816, "inter": 512, "bands": [[1, )" +
                   std::to_string(t_max) + R"(, "r8.w4r1|rmma.m1.n1.w1x4"]]})";
        };
        const auto loads = [&](const std::string& name, const std::string& entry_json) {
            try {
                const auto path = write_file(name, moe_table(entry_json));
                return ninferno::ops::load_route_table(path).loaded;
            } catch (const std::runtime_error&) { return false; }
        };
        failures +=
            check(loads("moe_r8_ok", r8_entry("iq2s", 4)), "an iq2s R8 band to T = 4 loads");
        failures += check(!loads("moe_r8_wide", r8_entry("iq2s", 8)),
                          "an R8 band beyond T = 4 is refused");
        failures += check(!loads("moe_r8_fp8", r8_entry("fp8", 4)),
                          "an R8 band on a format without the R8 kernel is refused");
        (void)ninferno::ops::load_route_table(write_file("good", good));
    }

    // Every shipped class file passes the loader's validation (the default lookup would otherwise
    // drop it silently).
    ::unsetenv("NINFERNO_ROUTE_DIR");
    const std::string shipped = ninferno::ops::route_table_dir();
    int files = 0;
    if (!shipped.empty() && std::filesystem::is_directory(shipped)) {
        for (const auto& f : std::filesystem::directory_iterator(shipped)) {
            if (f.path().extension() != ".json") continue;
            ++files;
            try {
                const auto s = ninferno::ops::inspect_route_table(f.path().string());
                std::cout << f.path().filename().string() << ": " << s.gpu_class << ", "
                          << s.message << '\n';
                failures += check(f.path().stem().string() ==
                                      [k = s.gpu_class]() mutable {
                                          for (char& c : k) {
                                              if (c == ':') c = '-';
                                          }
                                          return k;
                                      }(),
                                  f.path().string() + " is named after its class key");
            } catch (const std::exception& e) {
                // Summation order is judged with this device's kernels (a one-tile register MMA
                // sums lane by lane without tensor mma.sync): another class's table, which this
                // device never loads, is checked on its own GPU.
                const std::string why = e.what();
                const std::string own = ninferno::ops::route_gpu_class();
                const std::string stem = f.path().stem().string();
                const bool foreign = own.substr(0, own.find(':')) != stem.substr(0, stem.find('-'));
                if (foreign && (why.find("token-invariant") != std::string::npos ||
                                why.find("sums in another order") != std::string::npos)) {
                    std::cout << f.path().filename().string() << ": not judged on " << own
                              << " (summation order is the device's): " << why << '\n';
                    continue;
                }
                failures += check(false, std::string("shipped table refused: ") + why);
            }
        }
    }
    std::cout << files << " shipped route tables in " << shipped << '\n';

    // fp8_e4m3fn_row_f32 runs fp8's register MMA: without an entry of its own it takes the fp8
    // (row BF16) A16 entry of its shape; an entry of its own wins.
    {
        const std::string fp8 = table(
            gpu, R"({"format": "fp8", "compute": "a16", "n": 4096, "k": 12288,
                     "bands": [[1, 16777216, "rmma.m1.n1.w1x4"]], "prefill": "rmma.m2.n8.w4x1"},
                    {"format": "fp8", "compute": "a16", "n": 4096, "k": 4096,
                     "bands": [[1, 16777216, "rmma.m1.n1.w1x4"]]},
                    {"format": "fp8r32", "compute": "a16", "n": 4096, "k": 4096,
                     "bands": [[1, 16777216, "rmma.m1.n1.w2x2"]]})");
        const auto loaded = ninferno::ops::load_route_table(write_file("fp8", fp8));
        failures += check(loaded.loaded && loaded.generic_entries == 3,
                          "the fp8 table loads: " + loaded.message);
        const auto f32 = [](std::int32_t k, ninferno::ops::LinearPolicy policy) {
            return generic::Problem{ninferno::QType::FP8_E4M3FN_ROW_F32, policy, 4096, k,
                                    generic::Epilogue::Store};
        };
        const auto a16 = ninferno::ops::LinearPolicy::A16Only;
        failures += check(route_or_none(generic::table_route(f32(12288, a16), 1, false)) ==
                                  "rmma.m1.n1.w1x4" &&
                              route_or_none(generic::table_route(f32(12288, a16), 512, true)) ==
                                  "rmma.m2.n8.w4x1",
                          "row_f32 takes the fp8 entry of its shape");
        failures += check(route_or_none(generic::table_route(
                              f32(12288, ninferno::ops::LinearPolicy::AllowA8), 1, false)) ==
                              "rmma.m1.n1.w1x4",
                          "row_f32 stays A16 under AllowA8 and takes the fp8 A16 entry");
        failures += check(route_or_none(generic::table_route(f32(4096, a16), 1, false)) ==
                              "rmma.m1.n1.w2x2",
                          "an fp8r32 entry of its own wins");
        failures += check(!generic::table_route(f32(2048, a16), 1, false),
                          "row_f32 of a shape fp8 lacks has no route");
    }

    // "claim": a generic entry of a shape a shape-owned Linear table holds takes the call only at
    // the widths it covers (bands, or its prefill route under the prefill profile); an entry
    // without it never does; a claim on a fused epilogue is refused.
    {
        const std::string claims = table(
            gpu, R"({"format": "fp8", "compute": "a16", "n": 14336, "k": 5120, "claim": true,
                     "bands": [[64, 16777216, "rmma.m1.n4.w4x1"]], "prefill": "rmma.m1.n8.w4x1"},
                    {"format": "fp8", "compute": "a16", "n": 16384, "k": 5120,
                     "bands": [[1, 16777216, "rmma.m1.n1.w1x8"]]})");
        const auto loaded = ninferno::ops::load_route_table(write_file("claims", claims));
        failures += check(loaded.loaded && loaded.generic_entries == 2,
                          "the claim table loads: " + loaded.message);
        const auto fp8 = [](std::int32_t n, ninferno::ops::LinearPolicy policy) {
            return generic::Problem{ninferno::QType::FP8_E4M3FN_ROW_BF16, policy, n, 5120,
                                    generic::Epilogue::Store};
        };
        const auto a16 = ninferno::ops::LinearPolicy::A16Only;
        failures += check(!generic::table_claims(fp8(14336, a16), 16, false) &&
                              generic::table_claims(fp8(14336, a16), 64, false) &&
                              generic::table_claims(fp8(14336, a16), 512, false) &&
                              generic::table_claims(fp8(14336, a16), 17, true),
                          "a claiming entry takes the widths it covers");
        failures += check(!generic::table_claims(fp8(14336, ninferno::ops::LinearPolicy::AllowA8),
                                                 512, false),
                          "a claim is A16Only: a policy that allows quantized activations keeps "
                          "the shape's own table (its A8 kernels)");
        failures += check(!generic::table_claims(fp8(16384, a16), 1, false),
                          "an entry without claim does not take an owned shape");
        failures += check(refused(gpu, "claim-swiglu",
                                  R"({"format": "fp8", "compute": "a16", "epilogue": "swiglu",
                                      "n": 14336, "k": 5120, "claim": true,
                                      "bands": [[1, 16777216, "rmma.m1.n1.w1x8"]]})"),
                          "a claim on a fused epilogue is refused");
    }

    // A binary's shipped routes: <bin>/routes, else <bin>/../share/ninferno/routes, each only with
    // a table in it.
    {
        const auto pkg    = scratch_dir() / "pkg";
        const auto exe    = (pkg / "bin" / "ninferno-serve").string();
        const auto beside = pkg / "bin" / "routes", share = pkg / "share" / "ninferno" / "routes";
        std::filesystem::create_directories(beside);
        std::filesystem::create_directories(share);
        failures += check(ninferno::ops::route_dir_beside(exe).empty(),
                          "routes beside the binary: none without a table");
        std::ofstream(share / "ad102.json") << "{}";
        failures += check(ninferno::ops::route_dir_beside(exe) == share.string(),
                          "routes beside the binary: share/ninferno/routes");
        std::ofstream(beside / "ad102.json") << "{}";
        failures += check(ninferno::ops::route_dir_beside(exe) == beside.string(),
                          "routes beside the binary: bin/routes first");
    }

    ninferno::ops::clear_route_table();
    failures +=
        check(!ninferno::ops::route_table_status().loaded, "clear_route_table drops the table");
    failures += check(!generic::table_route(q8, 1, false) && detail::route_tokens(a4, 48) == 48,
                      "no table: no tuned route, no override");

    std::error_code ec;
    std::filesystem::remove_all(scratch_dir(), ec);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " route_table\n";
    return failures == 0 ? 0 : 1;
}
