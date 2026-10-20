// The release check: which versions count as releases, their order, what a manifest says for a
// version, and check()'s cache and switches against a manifest served on loopback. No network.

#include "product/update/update_check.h"

#include <httplib.h>
#include "support/posix.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << std::endl;
        ++failures;
    }
}

using namespace ninferno::update;

void test_versions() {
    check(is_release_version("0.1.0"), "0.1.0");
    check(is_release_version("v1.2.3"), "v prefix");
    check(is_release_version("1.2.3-rc1"), "pre-release");
    check(!is_release_version("dev"), "dev");
    check(!is_release_version("5c5095317"), "a commit");
    check(!is_release_version("0.1.0-12-g5c50953"), "git describe past a tag");
    check(!is_release_version("0.1.0-dirty-x"), "describe with a dirty mark... not a plain pre-release");
    check(!is_release_version("1"), "one number");
    check(!is_release_version(""), "empty");
    check(compare_versions("0.2.0", "0.1.9") > 0, "minor beats patch");
    check(compare_versions("0.10.0", "0.9.0") > 0, "numbers, not text");
    check(compare_versions("v0.2.0", "0.2.0") == 0, "v prefix ignored");
    check(compare_versions("0.2", "0.2.0") == 0, "missing parts are 0");
    check(compare_versions("0.2.0-rc1", "0.2.0") < 0, "a pre-release is older than its release");
    check(compare_versions("0.2.0-rc2", "0.2.0-rc10") < 0, "pre-release numbers, not text");
    check(compare_versions("0.2.0-beta", "0.2.0-rc1") < 0, "pre-release words in order");
    check(compare_versions("0.2.0-rc2", "0.2.0-rc1") > 0, "pre-releases in order");
    check(compare_versions("0.2.0-rc1", "0.1.5") > 0, "a pre-release is newer than the last release");
}

void test_manifest() {
    const auto r = parse_manifest(
        R"({"version":"0.3.0","notes":"https://x/notes","min_version":"0.2.0","assets":{}})",
        "0.1.0");
    check(r.newer && r.required, "older than the minimum: newer and required");
    check(r.latest == "0.3.0" && r.notes == "https://x/notes", "fields");
    check(notice(r).find("no longer supported") != std::string::npos, "required notice");
    const auto same = parse_manifest(R"({"version":"0.3.0"})", "0.3.0");
    check(!same.newer && !same.required && notice(same).empty(), "up to date: nothing to say");
    const auto plain = parse_manifest(R"({"version":"0.3.0","min_version":"0.1.0"})", "0.2.0");
    check(plain.newer && !plain.required && notice(plain).find("available") != std::string::npos,
          "newer, not required");
    bool threw = false;
    try {
        (void)parse_manifest(R"({"name":"x"})", "0.1.0");
    } catch (const std::exception&) { threw = true; }
    check(threw, "no version: not a manifest");
    threw = false;
    try {
        (void)parse_manifest("<html>", "0.1.0");
    } catch (const std::exception&) { threw = true; }
    check(threw, "not JSON");

    // A manifest's other fields are read only when they have their type.
    const auto typed = parse_manifest(R"({"version":"0.3.0","notes":5,"min_version":3})", "0.1.0");
    check(typed.newer && typed.notes.empty() && !typed.required,
          "mistyped optional fields ignored");
    const auto controls =
        parse_manifest(R"({"version":"0.3.0","notes":"a\nb\u0001c\u007f"})", "0.1.0");
    check(controls.notes == "a b c ", "control characters in notes made spaces");
}

void test_check() {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / ("ninferno_update_test_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    std::atomic<int> hits{0};
    std::string body = R"({"version":"0.5.0","notes":"n"})";
    httplib::Server server;
    server.Get("/latest.json", [&](const httplib::Request&, httplib::Response& res) {
        ++hits;
        res.set_content(body, "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    std::thread thread([&] { server.listen_after_bind(); });
    server.wait_until_ready();
    const std::string url = "http://127.0.0.1:" + std::to_string(port) + "/latest.json";

    CheckOptions options{.current = "0.4.0", .manifest_url = url, .cache_file = dir / "update.json"};
    auto first = check(options);
    check(first && first->newer && first->latest == "0.5.0", "read from the server");
    auto second = check(options);
    check(second && second->latest == "0.5.0" && hits == 1, "second answer from the cache");
    body    = R"({"version":"0.6.0"})";
    options.max_age = std::chrono::seconds(0);
    auto third = check(options);
    check(third && third->latest == "0.6.0" && hits == 2, "a stale cache reads again");

    options.max_age = std::chrono::hours(24);
    std::ofstream(options.cache_file) << R"({"url":5,"checked":"x","manifest":1})";
    auto damaged = check(options);
    check(damaged && damaged->latest == "0.6.0" && hits == 3, "a damaged cache reads again");

    options.current = "dev";
    check(!check(options) && hits == 3, "a source build never asks");
    options.current = "0.4.0";
    ::setenv("NINFERNO_UPDATE_CHECK", "off", 1);
    check(!check(options) && hits == 3, "switched off");
    ::unsetenv("NINFERNO_UPDATE_CHECK");

    options.manifest_url = "http://127.0.0.1:" + std::to_string(port) + "/missing.json";
    options.cache_file   = dir / "other.json";
    check(!check(options), "a manifest that is not there: nothing said");

    server.stop();
    thread.join();
    fs::remove_all(dir);
}

} // namespace

int main() {
    test_versions();
    test_manifest();
    test_check();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
