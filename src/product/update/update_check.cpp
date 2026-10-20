#include "product/update/update_check.h"
#include "core/ascii.h"
#include "core/cache_file.h"
#include "core/sys.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/pem.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninferno::update {

namespace {

constexpr const char* kManifestUrl = "https://ninferno.brontir.com/releases/latest.json";

// tools/release/signing.pub.pem (a test checks this copy against it).
constexpr const char* kReleaseKey = R"(-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE1R5JganVYknHjhX9nN2fYLqxwe2G
R9Zeg0QL/WoCKgTm4fpmwOM4ASQ+0Av3ASV+RPHQH1rTaXBSHyWCPZokZg==
-----END PUBLIC KEY-----
)";

// "v0.2.0-rc1" -> numbers {0, 2, 0}, pre-release "rc1". False when it is not that shape.
bool split_version(std::string version, std::vector<long>& numbers, std::string& pre) {
    numbers.clear();
    pre.clear();
    if (!version.empty() && (version[0] == 'v' || version[0] == 'V')) { version.erase(0, 1); }
    if (const auto dash = version.find('-'); dash != std::string::npos) {
        pre = version.substr(dash + 1);
        version.resize(dash);
        if (pre.empty()) { return false; }
        // `git describe` past a tag: <tag>-<commits>-g<hash>, a build between releases.
        if (pre.find("-g") != std::string::npos) { return false; }
        for (const char c : pre) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '.') { return false; }
        }
    }
    std::stringstream parts(version);
    std::string part;
    while (std::getline(parts, part, '.')) {
        if (part.empty() || part.size() > 9) { return false; }
        for (const char c : part) {
            if (!std::isdigit(static_cast<unsigned char>(c))) { return false; }
        }
        numbers.push_back(std::stol(part));
    }
    return numbers.size() >= 2 && numbers.size() <= 4;
}

size_t append(char* data, size_t size, size_t count, void* out) {
    auto* text = static_cast<std::string*>(out);
    if (text->size() + size * count > (1U << 20)) { return 0; } // a manifest is small
    text->append(data, size * count);
    return size * count;
}

std::string fetch(const std::string& url, const std::string& current, std::chrono::seconds timeout) {
    static const bool initialized = curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
    if (!initialized) { throw std::runtime_error("curl cannot start"); }
    CURL* handle = curl_easy_init();
    if (handle == nullptr) { throw std::runtime_error("curl cannot start"); }
    std::string body;
    const std::string agent = "ninferno/" + current;
    curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(handle, CURLOPT_USERAGENT, agent.c_str());
    curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(handle, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(handle, CURLOPT_TIMEOUT, static_cast<long>(timeout.count()));
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, append);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, &body);
    const CURLcode code = curl_easy_perform(handle);
    long status         = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_cleanup(handle);
    if (code != CURLE_OK) { throw std::runtime_error(curl_easy_strerror(code)); }
    if (status != 200) { throw std::runtime_error("HTTP " + std::to_string(status)); }
    return body;
}

// Base64 (whitespace ignored); "" when it is not base64.
std::string base64_decode(const std::string& text) {
    std::string clean;
    for (const char c : text) {
        if (!std::isspace(static_cast<unsigned char>(c))) { clean.push_back(c); }
    }
    if (clean.empty() || clean.size() % 4 != 0) { return ""; }
    std::string out(clean.size() / 4 * 3, '\0');
    const int n = EVP_DecodeBlock(reinterpret_cast<unsigned char*>(out.data()),
                                  reinterpret_cast<const unsigned char*>(clean.data()),
                                  static_cast<int>(clean.size()));
    if (n < 0) { return ""; }
    const auto padding = static_cast<std::size_t>(clean.end()[-1] == '=') +
                         static_cast<std::size_t>(clean.end()[-2] == '=');
    out.resize(static_cast<std::size_t>(n) - padding);
    return out;
}

} // namespace

const std::string& release_key() {
    static const std::string key = kReleaseKey;
    return key;
}

bool verify_signature(const std::string& text, const std::string& signature,
                      const std::string& public_key) {
    const std::unique_ptr<BIO, decltype(&BIO_free)> bio(
        BIO_new_mem_buf(public_key.data(), static_cast<int>(public_key.size())), BIO_free);
    if (!bio) { return false; }
    const std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> key(
        PEM_read_bio_PUBKEY(bio.get(), nullptr, nullptr, nullptr), EVP_PKEY_free);
    const std::string der = base64_decode(signature);
    if (!key || der.empty()) { return false; }
    const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),
                                                                      EVP_MD_CTX_free);
    return ctx && EVP_DigestVerifyInit(ctx.get(), nullptr, EVP_sha256(), nullptr, key.get()) == 1 &&
           EVP_DigestVerify(ctx.get(), reinterpret_cast<const unsigned char*>(der.data()),
                            der.size(), reinterpret_cast<const unsigned char*>(text.data()),
                            text.size()) == 1;
}

std::string default_manifest_url() {
    const auto url = sys::env("NINFERNO_UPDATE_URL");
    return url.empty() ? kManifestUrl : url;
}

std::filesystem::path default_cache_file() {
    const auto dir = sys::cache_dir();
    return dir.empty() ? dir : dir / "update.json";
}

bool checks_enabled() {
    const auto value = ascii_lower(sys::env("NINFERNO_UPDATE_CHECK"));
    return value != "0" && value != "off" && value != "no" && value != "false";
}

bool is_release_version(const std::string& version) {
    std::vector<long> numbers;
    std::string pre;
    return split_version(version, numbers, pre);
}

// Pre-release labels as semantic versioning orders them: dot-separated parts left to right, a
// numeric part by value (rc2 < rc10 as rc.2 < rc.10; digits trailing a word compare as a number
// too), a numeric part before a word, and a shorter label first when one is a prefix.
static int compare_prerelease(const std::string& a, const std::string& b) {
    const auto split = [](const std::string& label) {
        std::vector<std::pair<std::string, long>> parts; // a word and its trailing number
        std::size_t at = 0;
        while (at <= label.size()) {
            const std::size_t dot  = std::min(label.find('.', at), label.size());
            const std::string part = label.substr(at, dot - at);
            std::size_t digits     = part.size();
            while (digits > 0 && std::isdigit(static_cast<unsigned char>(part[digits - 1]))) {
                --digits;
            }
            const bool numbered = digits < part.size() && part.size() - digits <= 9;
            parts.emplace_back(numbered ? part.substr(0, digits) : part,
                               numbered ? std::stol(part.substr(digits)) : -1);
            at = dot + 1;
        }
        return parts;
    };
    const auto pa = split(a);
    const auto pb = split(b);
    for (std::size_t i = 0; i < std::min(pa.size(), pb.size()); ++i) {
        if (pa[i].first != pb[i].first) { return pa[i].first < pb[i].first ? -1 : 1; }
        if (pa[i].second != pb[i].second) { return pa[i].second < pb[i].second ? -1 : 1; }
    }
    if (pa.size() == pb.size()) { return a < b ? -1 : a == b ? 0 : 1; }
    return pa.size() < pb.size() ? -1 : 1;
}

int compare_versions(const std::string& a, const std::string& b) {
    std::vector<long> na, nb;
    std::string pa, pb;
    if (!split_version(a, na, pa) || !split_version(b, nb, pb)) {
        throw std::invalid_argument("not a release version: " + (is_release_version(a) ? b : a));
    }
    na.resize(4, 0);
    nb.resize(4, 0);
    for (size_t i = 0; i < 4; ++i) {
        if (na[i] != nb[i]) { return na[i] < nb[i] ? -1 : 1; }
    }
    if (pa == pb) { return 0; }
    if (pa.empty()) { return 1; } // a release is newer than its pre-releases
    if (pb.empty()) { return -1; }
    return compare_prerelease(pa, pb);
}

Release parse_manifest(const std::string& text, const std::string& current) {
    const auto json = nlohmann::json::parse(text, nullptr, false);
    if (!json.is_object() || !json.contains("version") || !json["version"].is_string()) {
        throw std::runtime_error("not a release manifest (no \"version\")");
    }
    Release release;
    release.current = current;
    release.latest  = json["version"].get<std::string>();
    // Notes go into a log line and a link: a string, its control characters made spaces.
    if (json.contains("notes") && json["notes"].is_string()) {
        release.notes = json["notes"].get<std::string>();
        for (char& c : release.notes) {
            if (static_cast<unsigned char>(c) < 0x20 || c == 0x7f) { c = ' '; }
        }
    }
    if (!is_release_version(release.latest)) {
        throw std::runtime_error("the manifest's version is not a release: " + release.latest);
    }
    release.newer = compare_versions(release.latest, current) > 0;
    if (json.contains("min_version") && json["min_version"].is_string()) {
        if (const auto minimum = json["min_version"].get<std::string>();
            is_release_version(minimum)) {
            release.required = compare_versions(current, minimum) < 0;
        }
    }
    return release;
}

std::optional<Release> check(const CheckOptions& options) {
    if (!checks_enabled() || !is_release_version(options.current)) { return std::nullopt; }
    const auto url   = options.manifest_url.empty() ? default_manifest_url() : options.manifest_url;
    const auto cache = options.cache_file.empty() ? default_cache_file() : options.cache_file;
    const auto& key  = options.public_key.empty() ? release_key() : options.public_key;
    const auto now   = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    if (!cache.empty()) {
        std::ifstream in(cache);
        const auto saved = nlohmann::json::parse(in, nullptr, false);
        // Typed reads only: a damaged cache is read again from the server, never thrown.
        const auto field = [&](const char* key) -> const nlohmann::json* {
            return saved.is_object() && saved.contains(key) ? &saved[key] : nullptr;
        };
        const auto* saved_url     = field("url");
        const auto* saved_checked = field("checked");
        const auto* saved_text    = field("manifest");
        const auto* saved_sig     = field("signature");
        // The cache is checked like the server's answer: only a signed manifest counts.
        if (saved_url != nullptr && saved_url->is_string() &&
            saved_url->get<std::string>() == url && saved_checked != nullptr &&
            saved_checked->is_number_integer() &&
            now - saved_checked->get<std::int64_t>() < options.max_age.count() &&
            saved_text != nullptr && saved_text->is_string() && saved_sig != nullptr &&
            saved_sig->is_string() &&
            verify_signature(saved_text->get<std::string>(), saved_sig->get<std::string>(), key)) {
            try {
                return parse_manifest(saved_text->get<std::string>(), options.current);
            } catch (const std::exception&) {
                // a damaged cache: read the manifest again
            }
        }
    }
    try {
        const auto text      = fetch(url, options.current, options.timeout);
        const auto signature = fetch(url + ".sig", options.current, options.timeout);
        if (!verify_signature(text, signature, key)) { return std::nullopt; }
        const auto release = parse_manifest(text, options.current);
        if (!cache.empty()) {
            // Best effort: a cache that cannot be written only means asking the server again.
            try {
                create_parent_directories(cache);
                write_file_atomic(cache, nlohmann::json{{"url", url},
                                                        {"checked", now},
                                                        {"manifest", text},
                                                        {"signature", signature}}
                                             .dump());
            } catch (const std::exception&) {}
        }
        return release;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::string notice(const Release& release) {
    if (release.required) {
        return "NInferno " + release.current + " is no longer supported: update to " +
               release.latest + " (ninferno-update, or Update in ninferno-ui)" +
               (release.notes.empty() ? "" : "; what changed: " + release.notes);
    }
    if (release.newer) {
        return "NInferno " + release.latest + " is available (this is " + release.current +
               "): run ninferno-update, or Update in ninferno-ui" +
               (release.notes.empty() ? "" : "; what changed: " + release.notes);
    }
    return "";
}

} // namespace ninferno::update
