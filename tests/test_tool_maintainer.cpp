// The family tools' maintainer options (product/tool_maintainer.h) parse as the Engine's products
// parse them and refuse --moe-tune; a measurement names route data by content
// (product/route_labels.h): the same tables carry one label wherever they are, another table or
// one more file another.

#include "product/route_labels.h"
#include "product/tool_maintainer.h"

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

} // namespace

int main() {
    using ninferno::product::ToolArguments;
    using ninferno::product::tool_maintainer_options;
    ToolArguments args{"a.ninferno", "score", {}};
    const auto none = tool_maintainer_options(args, "ninferno-dense");
    check(none == ninferno::MaintainerOptions{}, "no flag gave maintainer options");
    args.options   = {{"route-table", "t.json"}, {"route-dir", "routes"},
                      {"dense-tune", "retune"},  {"tune-cache", "cache"},
                      {"debug-poison", "nan"},   {"debug-poison-skip", "kv,state"},
                      {"tokens", "x.bin"}};
    const auto lab = tool_maintainer_options(args, "ninferno-dense");
    check(lab.route_table == "t.json" && lab.route_dir == "routes" &&
              lab.dense_tune == ninferno::TuneMode::Retune && lab.tune_cache == "cache" &&
              lab.debug_poison == std::uint8_t{0xFF} &&
              lab.debug_poison_skip == std::vector<std::string>{"kv", "state"} &&
              lab.moe_tune == ninferno::TuneMode::Off,
          "the tool's maintainer options were not parsed");
    const auto refuses = [](ToolArguments a, const std::string& expected) {
        try {
            (void)tool_maintainer_options(a, "ninferno-gemma4");
        } catch (const std::invalid_argument& error) {
            return std::string(error.what()).find(expected) != std::string::npos;
        }
        return false;
    };
    check(
        refuses({"a", "score", {{"moe-tune", "on"}}}, "--moe-tune does not apply: ninferno-gemma4"),
        "--moe-tune was accepted by a tool without an MoE tuner");
    check(refuses({"a", "score", {{"dense-tune", "yes"}}}, "--dense-tune must be"),
          "a bad --dense-tune was accepted");
    check(refuses({"a", "score", {{"debug-poison", "300"}}}, "--debug-poison must be"),
          "a bad --debug-poison was accepted");

    // Route data by content.
    const auto root = std::filesystem::temp_directory_path() / "ninferno_tool_maintainer_test";
    std::filesystem::remove_all(root);
    write_file(root / "a" / "gb202.json", "{\"a\":1}");
    write_file(root / "a" / "tuner" / "dense_tune.json", "{}");
    write_file(root / "b" / "gb202.json", "{\"a\":1}");
    write_file(root / "b" / "tuner" / "dense_tune.json", "{}");
    const std::string a = ninferno::product::route_dir_label(root / "a");
    check(a.starts_with("sha256:") && a.size() == 7 + 64, "a route directory label: " + a);
    check(a == ninferno::product::route_dir_label(root / "b"),
          "two directories with the same tables carry different labels");
    write_file(root / "b" / "gb203.json", "{}");
    check(a != ninferno::product::route_dir_label(root / "b"), "one more table kept the label");
    write_file(root / "a" / "gb202.json", "{\"a\":2}");
    check(a != ninferno::product::route_dir_label(root / "a"), "a changed table kept the label");
    check(ninferno::product::route_dir_label(root / "missing") == "unreadable",
          "a missing directory has a label");
    const std::string table = ninferno::product::route_table_label(root / "b" / "gb202.json");
    check(table == "sha256:" + ninferno::sha256_hex(std::string_view("{\"a\":1}")),
          "a route table is not labelled by the SHA-256 of its bytes: " + table);
    check(ninferno::product::route_table_label(root / "none.json") == "unreadable",
          "a missing route table has a label");
    std::filesystem::remove_all(root);

    std::cout << (failures ? "FAIL" : "OK") << " tool_maintainer\n";
    return failures ? 1 : 0;
}
