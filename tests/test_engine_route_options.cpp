// The Engine refuses a --route-dir or --route-table that cannot be read before it touches the
// device: Engine::Impl admits its options (runtime::admit_engine_options) before the
// process-options lease and the device, and admission checks them (ops/route_options.h).
// Read as no table, they would run the heuristic routes. CPU-only.

#include "runtime/engine/model_instance.h"
#include "runtime/engine/served_bytes.h"
#include "support/temp_dir.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

void write_file(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << text;
}

// Whether admitting `options` is refused with a message that holds `expected`.
bool refused(const ninferno::EngineOptions& options, const std::string& expected) {
    try {
        (void)ninferno::runtime::admit_engine_options(options);
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find(expected) != std::string::npos) { return true; }
        std::cerr << "  refused with: " << error.what() << '\n';
    }
    return false;
}

} // namespace

int main() {
    const TempDir scratch("ninferno_engine_route_options_test");
    const auto& root = scratch.path();
    write_file(root / "routes" / "gb202.json",
               R"({"schema":"ninferno-route-table/1","gpu":{"class":"gb202"},"entries":[]})");
    write_file(root / "bad.json", R"({"schema":"other"})");

    ninferno::EngineOptions options;
    options.max_concurrency = 1;
    {
        auto o                 = options;
        o.maintainer.route_dir = root / "none";
        check(refused(o, "--route-dir"), "a missing --route-dir was accepted");
        o.maintainer.route_dir = root / "bad.json";
        check(refused(o, "not a directory"), "a file as --route-dir was accepted");
    }
    {
        auto o                   = options;
        o.maintainer.route_table = root / "none.json";
        check(refused(o, "--route-table"), "a missing --route-table was accepted");
        o.maintainer.route_table = root / "bad.json";
        check(refused(o, "is not ninferno-route-table/1"),
              "a malformed --route-table was accepted");
    }
    {
        auto o                   = options;
        o.maintainer.route_dir   = root / "routes";
        o.maintainer.route_table = root / "routes" / "gb202.json";
        try {
            const auto normalized = ninferno::runtime::admit_engine_options(o);
            check(normalized.maintainer.route_dir == o.maintainer.route_dir &&
                      normalized.maintainer.route_table == o.maintainer.route_table,
                  "readable route options did not pass through");
        } catch (const std::exception& error) {
            check(false, std::string("readable route options were refused: ") + error.what());
        }
    }
    // The byte class's activation bits, as the Engine's policy rule gives them: exact under Exact,
    // A4 only where a use records AllowA4, Fast keeps it and the device runs W4A4.
    {
        using ninferno::ActivationMode;
        using ninferno::runtime::activation_bits;
        check(activation_bits(ActivationMode::Exact, true, true) == "exact" &&
                  activation_bits(ActivationMode::Fast, true, true) == "fast-a4" &&
                  activation_bits(ActivationMode::Fast, true, false) == "fast-a8" &&
                  activation_bits(ActivationMode::Fast, false, true) == "fast-a8" &&
                  activation_bits(ActivationMode::Int8, true, true) == "fast-a8",
              "activation_bits follows effective_policy and the W4A4 device features");
        // No artifact and no device here: never "fast-a4".
        auto none_here = options;
        check(ninferno::runtime::served_activations(none_here) == "fast-a8",
              "an unreadable artifact labels fast-a8");
        auto o               = options;
        o.kv_cache           = ninferno::KvCacheStorage::Int8Group64;
        o.kv_cache_automatic = false;
        check(ninferno::runtime::resolved_kv_cache(o) == ninferno::KvCacheStorage::Int8Group64,
              "an explicit KV storage resolves to itself");
    }
    std::cout << (failures ? "FAIL" : "OK") << " engine_route_options\n";
    return failures ? 1 : 0;
}
