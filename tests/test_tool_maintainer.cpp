// The family tools' maintainer options (product/tool_maintainer.h) parse as the Engine's products
// parse them, refuse --moe-tune and refuse route data that cannot be read (ops/route_options.h); a
// measurement names its data by content (product/route_labels.h): the same tables carry one label
// wherever they are, another table or one more table another, a file the engine never reads none.

#include "ops/route_options.h"
#include "product/route_labels.h"
#include "product/tool_maintainer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

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

// Whether `call` throws std::invalid_argument whose message holds `expected`.
template <typename Call>
bool refuses(Call&& call, const std::string& expected) {
    try {
        (void)call();
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find(expected) != std::string::npos) { return true; }
        std::cerr << "  refused with: " << error.what() << '\n';
    }
    return false;
}

constexpr const char* kTable =
    R"({"schema":"ninferno-route-table/1","gpu":{"class":"gb202"},"entries":[]})";

} // namespace

int main() {
    using ninferno::product::route_dir_label;
    using ninferno::product::ToolArguments;
    using ninferno::product::tool_maintainer_options;
    const auto root = std::filesystem::temp_directory_path() / "ninferno_tool_maintainer_test";
    std::filesystem::remove_all(root);
    write_file(root / "lab" / "t.json", kTable);
    std::filesystem::create_directories(root / "lab" / "routes");

    ToolArguments args{"a.ninferno", "score", {}};
    const auto none = tool_maintainer_options(args, "ninferno-dense");
    check(none == ninferno::MaintainerOptions{}, "no flag gave maintainer options");
    const std::string table_path = (root / "lab" / "t.json").string();
    const std::string dir_path   = (root / "lab" / "routes").string();
    args.options                 = {{"route-table", table_path}, {"route-dir", dir_path},
                                    {"dense-tune", "retune"},    {"tune-cache", "cache"},
                                    {"debug-poison", "nan"},     {"debug-poison-skip", "kv,state"},
                                    {"tokens", "x.bin"}};
    const auto lab = tool_maintainer_options(args, "ninferno-dense");
    check(lab.route_table == table_path && lab.route_dir == dir_path &&
              lab.dense_tune == ninferno::TuneMode::Retune && lab.tune_cache == "cache" &&
              lab.debug_poison == std::uint8_t{0xFF} &&
              lab.debug_poison_skip == std::vector<std::string>{"kv", "state"} &&
              lab.moe_tune == ninferno::TuneMode::Off,
          "the tool's maintainer options were not parsed");
    const auto tool_refuses = [](ToolArguments a, const std::string& expected) {
        return refuses([&] { return tool_maintainer_options(a, "ninferno-gemma4"); }, expected);
    };
    check(tool_refuses({"a", "score", {{"moe-tune", "on"}}},
                       "--moe-tune does not apply: ninferno-gemma4"),
          "--moe-tune was accepted by a tool without an MoE tuner");
    check(tool_refuses({"a", "score", {{"dense-tune", "yes"}}}, "--dense-tune must be"),
          "a bad --dense-tune was accepted");
    check(tool_refuses({"a", "score", {{"debug-poison", "300"}}}, "--debug-poison must be"),
          "a bad --debug-poison was accepted");
    // Route data that cannot be read is refused before anything loads, never run as no table.
    check(tool_refuses({"a", "score", {{"route-dir", (root / "none").string()}}}, "--route-dir"),
          "a missing --route-dir was accepted");
    check(tool_refuses({"a", "score", {{"route-dir", table_path}}}, "not a directory"),
          "a file as --route-dir was accepted");
    check(tool_refuses({"a", "score", {{"route-table", (root / "none.json").string()}}},
                       "--route-table"),
          "a missing --route-table was accepted");
    write_file(root / "lab" / "bad.json", "{\"schema\":\"other\"}");
    check(tool_refuses({"a", "score", {{"route-table", (root / "lab" / "bad.json").string()}}},
                       "is not ninferno-route-table/1"),
          "a malformed --route-table was accepted");

    // Route data by content: the files the engine reads (DIR/*.json, DIR/tuner/dense_tune.json).
    write_file(root / "a" / "gb202.json", "{\"a\":1}");
    write_file(root / "a" / "tuner" / "dense_tune.json", "{}");
    write_file(root / "b" / "gb202.json", "{\"a\":1}");
    write_file(root / "b" / "tuner" / "dense_tune.json", "{}");
    const std::string a = route_dir_label(root / "a");
    check(a.starts_with("sha256:") && a.size() == 7 + 64, "a route directory label: " + a);
    check(a == route_dir_label(root / "b"),
          "two directories with the same tables carry different labels");
    const auto files = ninferno::ops::route_dir_files(root / "a");
    check(files.size() == 2 && files[0].relative == "gb202.json" &&
              files[1].relative == "tuner/dense_tune.json",
          "the route directory's files are not the tables and the tuner's measurements");
    // Files the engine never reads change nothing.
    write_file(root / "b" / "README.md", "notes");
    write_file(root / "b" / "tuner" / "moe_tune.json", "{}");
    write_file(root / "b" / "old" / "gb203.json", "{}");
    check(a == route_dir_label(root / "b"), "a file the engine does not read changed the label");
    write_file(root / "b" / "gb203.json", "{}");
    check(a != route_dir_label(root / "b"), "one more table kept the label");
    write_file(root / "a" / "tuner" / "dense_tune.json", "{\"x\":1}");
    check(a != route_dir_label(root / "a"), "changed tuner measurements kept the label");
    const std::string a2 = route_dir_label(root / "a");
    write_file(root / "a" / "gb202.json", "{\"a\":2}");
    check(a2 != route_dir_label(root / "a"), "a changed table kept the label");
    check(refuses([&] { return route_dir_label(root / "missing"); }, "--route-dir"),
          "a missing directory has a label");
    // Symlinks are followed: a linked directory or table reads as its target; a dangling one is
    // refused.
    std::error_code link_error;
    std::filesystem::create_directory_symlink(root / "a", root / "link", link_error);
    if (!link_error) {
        check(route_dir_label(root / "link") == route_dir_label(root / "a"),
              "a linked route directory has another label than its target");
        std::filesystem::create_directories(root / "c");
        std::filesystem::create_symlink(root / "a" / "gb202.json", root / "c" / "gb202.json");
        write_file(root / "c" / "tuner" / "dense_tune.json", "{\"x\":1}");
        check(route_dir_label(root / "c") == route_dir_label(root / "a"),
              "a linked table has another label than its target");
        std::filesystem::create_symlink(root / "gone.json", root / "c" / "gb203.json");
        check(refuses([&] { return route_dir_label(root / "c"); }, "dangling symlink"),
              "a dangling table symlink was labelled");
        check(refuses([&] { return ninferno::ops::route_dir_files(root / "c"); },
                      "gb203.json is a dangling symlink"),
              "a dangling table symlink was accepted as --route-dir");
        // A link the file system cannot resolve for another reason (a directory on its way that
        // may not be searched) is refused with that reason, not called dangling.
        std::filesystem::create_directories(root / "locked");
        write_file(root / "locked" / "gb202.json", kTable);
        std::filesystem::create_directories(root / "d");
        std::filesystem::create_symlink(root / "locked" / "gb202.json", root / "d" / "gb202.json");
        std::filesystem::permissions(root / "locked", std::filesystem::perms::none);
        std::error_code probe;
        (void)std::filesystem::status(root / "d" / "gb202.json", probe);
        if (probe && probe != std::errc::no_such_file_or_directory) {
            check(refuses([&] { return ninferno::ops::route_dir_files(root / "d"); },
                          "gb202.json: " + probe.message()),
                  "an unresolvable table symlink was not refused with its reason");
        } else {
            std::cout << "a directory without permissions is searchable here: not checked\n";
        }
        std::filesystem::permissions(root / "locked", std::filesystem::perms::owner_all);
    } else {
        std::cout << "symlinks unavailable here (" << link_error.message() << "): not checked\n";
    }

    const std::string table = ninferno::product::route_table_label(root / "b" / "gb202.json");
    check(table == "sha256:" + ninferno::sha256_hex(std::string_view("{\"a\":1}")),
          "a route table is not labelled by the SHA-256 of its bytes: " + table);
    check(refuses([&] { return ninferno::product::route_table_label(root / "none.json"); },
                  "--route-table"),
          "a missing route table has a label");
    check(refuses([&] { return ninferno::product::drafter_registry_label(root / "none.json"); },
                  "--drafter-registry"),
          "a missing drafter registry has a label");
    std::filesystem::remove_all(root);

    std::cout << (failures ? "FAIL" : "OK") << " tool_maintainer\n";
    return failures ? 1 : 0;
}
