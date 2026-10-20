// The release check: which versions count as releases, their order, what a manifest says for a
// version, its signature (only a manifest the key signed counts, from the server or the cache),
// check()'s cache and switches against a manifest served on loopback, and that every copy of the
// release key (installer, updater, Windows setup) is tools/release/signing.pub.pem. No network.

#include "product/update/update_check.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include "support/posix.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
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

// A key pair for these tests only (the release key's private half is never in the repository).
constexpr const char* kTestPrivate = R"(-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgnRORtkKYU4sOKxkl
HPg9ywPgYEPAE2NvXHXuldkzJG6hRANCAAT73bToeuO17cXdIyeRl4UZf0J/wgAr
ATIFJURFMCTT0aiVWMoE9s7AxQCho126V/MN5exfe+3pM16HZiht9Fg/
-----END PRIVATE KEY-----
)";
constexpr const char* kTestPublic  = R"(-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE+9206Hrjte3F3SMnkZeFGX9Cf8IA
KwEyBSVERTAk09GolVjKBPbOwMUAoaNdulfzDeXsX3vt6TNeh2YobfRYPw==
-----END PUBLIC KEY-----
)";

// What tools/release/sign.sh writes: base64 of the DER ECDSA P-256 / SHA-256 signature.
std::string sign(const std::string& text) {
    const std::unique_ptr<BIO, decltype(&BIO_free)> bio(BIO_new_mem_buf(kTestPrivate, -1),
                                                        BIO_free);
    const std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> key(
        PEM_read_bio_PrivateKey(bio.get(), nullptr, nullptr, nullptr), EVP_PKEY_free);
    const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),
                                                                      EVP_MD_CTX_free);
    std::size_t size = 0;
    EVP_DigestSignInit(ctx.get(), nullptr, EVP_sha256(), nullptr, key.get());
    EVP_DigestSign(ctx.get(), nullptr, &size, reinterpret_cast<const unsigned char*>(text.data()),
                   text.size());
    std::string der(size, '\0');
    EVP_DigestSign(ctx.get(), reinterpret_cast<unsigned char*>(der.data()), &size,
                   reinterpret_cast<const unsigned char*>(text.data()), text.size());
    der.resize(size);
    std::string out(4 * ((der.size() + 2) / 3), '\0');
    EVP_EncodeBlock(reinterpret_cast<unsigned char*>(out.data()),
                    reinterpret_cast<const unsigned char*>(der.data()),
                    static_cast<int>(der.size()));
    return out;
}

std::string read(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

void test_signatures() {
    const std::string text = R"({"version":"0.5.0"})";
    const std::string good = sign(text);
    check(verify_signature(text, good, kTestPublic), "a signature of the key verifies");
    check(!verify_signature(text + " ", good, kTestPublic), "one byte more: refused");
    check(!verify_signature(text, good, release_key()), "another key: refused");
    check(!verify_signature(text, "", kTestPublic), "no signature: refused");
    check(!verify_signature(text, "not base64!", kTestPublic), "garbage: refused");
    check(!verify_signature(text, good, "not a key"), "no key: refused");

    // Every copy of the release key is the one tools/release/sign.sh checks against.
    const std::filesystem::path source = NINFERNO_SOURCE_DIR;
    std::string pem                    = read(source / "tools/release/signing.pub.pem");
    while (!pem.empty() && (pem.back() == '\n' || pem.back() == '\r')) { pem.pop_back(); }
    check(release_key().find(pem) == 0, "update_check carries tools/release/signing.pub.pem");
    check(read(source / "tools/release/install.sh").find("RELEASE_KEY='" + pem + "'") !=
              std::string::npos,
          "install.sh carries tools/release/signing.pub.pem");
    std::string body = pem.substr(pem.find('\n') + 1);
    body             = body.substr(0, body.rfind('\n'));
    body.erase(std::remove(body.begin(), body.end(), '\n'), body.end());
    // There the key is a string literal, which the formatter may split: compared without quotes
    // and white space.
    std::string setup = read(source / "tools/release/windows/engine.cpp");
    setup.erase(std::remove_if(
                    setup.begin(), setup.end(),
                    [](char c) { return c == '"' || std::isspace(static_cast<unsigned char>(c)); }),
                setup.end());
    check(setup.find("kReleaseKey=" + body + ";") != std::string::npos,
          "the Windows setup carries tools/release/signing.pub.pem");
}

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
    std::string signature = sign(body);
    httplib::Server server;
    server.Get("/latest.json", [&](const httplib::Request&, httplib::Response& res) {
        ++hits;
        res.set_content(body, "application/json");
    });
    server.Get("/latest.json.sig", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(signature, "text/plain");
    });
    server.Get("/unsigned.json", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(body, "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    std::thread thread([&] { server.listen_after_bind(); });
    server.wait_until_ready();
    const std::string url = "http://127.0.0.1:" + std::to_string(port) + "/latest.json";

    CheckOptions options{.current      = "0.4.0",
                         .manifest_url = url,
                         .cache_file   = dir / "update.json",
                         .public_key   = kTestPublic};
    auto first = check(options);
    check(first && first->newer && first->latest == "0.5.0", "read from the server");
    auto second = check(options);
    check(second && second->latest == "0.5.0" && hits == 1, "second answer from the cache");
    body            = R"({"version":"0.6.0"})";
    signature       = sign(body);
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

    // A cache whose manifest was changed after it was signed is read again from the server.
    std::ofstream(options.cache_file) << nlohmann::json{
        {"url", url},
        {"checked", 0x7fffffffLL},
        {"manifest", R"({"version":"9.9.9"})"},
        {"signature", signature}}.dump();
    options.max_age = std::chrono::hours(24 * 365 * 100);
    auto forged     = check(options);
    check(forged && forged->latest == "0.6.0" && hits == 4, "a forged cache is not believed");
    options.max_age = std::chrono::hours(24);

    body            = R"({"version":"7.0.0"})"; // served, but the signature is still 0.6.0's
    options.max_age = std::chrono::seconds(0);
    check(!check(options), "a manifest its signature does not match: nothing said");
    options.max_age    = std::chrono::hours(24);
    options.public_key = release_key();
    body               = R"({"version":"0.6.0"})";
    signature          = sign(body);
    options.max_age    = std::chrono::seconds(0);
    check(!check(options), "signed by another key: nothing said");
    options.public_key = kTestPublic;
    options.max_age    = std::chrono::hours(24);

    options.manifest_url = "http://127.0.0.1:" + std::to_string(port) + "/unsigned.json";
    options.cache_file   = dir / "unsigned.json";
    check(!check(options), "a manifest without a signature: nothing said");

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
    test_signatures();
    test_check();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
