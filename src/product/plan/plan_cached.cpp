#include "product/plan/plan_cached.h"

#include "convert/acquire.h"
#include "convert/fetch.h"
#include "convert/gguf.h"
#include "core/hub_revision.h"
#include "core/quant_label.h"
#include "product/plan/plan_facts.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <mutex>
#include <regex>
#include <set>
#include <stdexcept>
#include <tuple>

namespace ninferno::product {

namespace fs = std::filesystem;

namespace {

std::string ascii_lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// A path as UTF-8 (path::string() throws on Windows for a name outside the ANSI code page).
std::string utf8(const fs::path& path) {
    const auto text = path.u8string();
    return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

} // namespace

// A GGUF header from the first bytes of a file, read longer until its metadata fits.
convert::GgufFile read_local_head(const fs::path& path) {
    const std::uint64_t size = fs::file_size(path);
    for (std::uint64_t want : {8ULL << 20, 64ULL << 20, 512ULL << 20}) {
        const std::uint64_t n = std::min(want, size);
        std::string bytes(static_cast<std::size_t>(n), '\0');
        std::ifstream in(path, std::ios::binary);
        in.read(bytes.data(), static_cast<std::streamsize>(n));
        try {
            return convert::parse_gguf_head(reinterpret_cast<const std::byte*>(bytes.data()),
                                            bytes.size(), size, utf8(path));
        } catch (const convert::GgufTruncated&) {
            if (n == size) { throw; }
        }
    }
    throw std::runtime_error(utf8(path) + ": the GGUF metadata runs past 512 MiB");
}

// Every part of a split GGUF beside `path` ("x-00001-of-00003.gguf"), or the file alone.
std::uint64_t local_gguf_bytes(const fs::path& path) {
    static const std::regex split(R"((.*)-\d{5}-of-(\d{5})\.gguf)", std::regex::icase);
    std::smatch match;
    const std::string name = utf8(path.filename());
    if (!std::regex_match(name, match, split)) { return fs::file_size(path); }
    std::uint64_t total = 0;
    for (const auto& entry : fs::directory_iterator(path.parent_path())) {
        std::smatch other;
        const std::string n = utf8(entry.path().filename());
        if (std::regex_match(n, other, split) && other[1] == match[1]) {
            total += entry.file_size();
        }
    }
    return total;
}

// The GGUF quantizations of a hub repository already in the local hub cache, each with its file:
// a downloaded GGUF (the first part of a split one) in the revision's snapshot; where the snapshot
// was removed, the artifact a start serves in its place (convert::kept_artifact: the conversion in
// <repo>/ninferno/<commit>/, convert::converted_artifact's pick per quantization).
std::map<std::string, fs::path> cached_quants(const fs::path& hub_root, const std::string& hub,
                                              const std::string& repo,
                                              const std::string& revision) {
    std::map<std::string, fs::path> out;
    std::error_code error;
    // The requested revision's snapshot alone (a commit, or the commit its ref names: "main" when
    // none was asked), never another revision's files; the hub's own directory.
    const fs::path repo_dir = convert::is_modelscope_hub(hub)
                                  ? hub_root / "modelscope" / hub_repo_directory(repo)
                                  : hub_root / hub_repo_directory(repo);
    const auto commit       = convert::local_commit(repo_dir, revision);
    if (!commit) { return out; }
    const fs::path root = repo_dir / "snapshots" / *commit;
    if (!fs::is_directory(root, error)) {
        // The download removed, its conversion kept: what a start of each quantization loads.
        const fs::path converted = repo_dir / "ninferno" / *commit;
        std::set<std::string> labels;
        for (const auto& entry : fs::directory_iterator(converted, error)) {
            if (const auto name = convert::converted_name(entry.path()); name && !name->label.empty()) {
                labels.insert(name->label);
            }
        }
        for (const auto& label : labels) {
            try {
                if (auto path = convert::converted_artifact(converted, label, "")) {
                    out[label] = *path;
                }
            } catch (const std::exception&) {}
        }
        return out;
    }
    for (auto it = fs::recursive_directory_iterator(root, error);
         !error && it != fs::recursive_directory_iterator(); it.increment(error)) {
        const std::string name  = utf8(it->path().filename());
        const std::string lower = ascii_lower(name);
        std::string gguf;
        bool artifact = false;
        if (lower.ends_with(".gguf")) {
            static const std::regex later(
                R"(.*-0000[2-9]-of-\d{5}\.gguf|.*-000[1-9]\d-of-\d{5}\.gguf)", std::regex::icase);
            if (std::regex_match(name, later)) { continue; } // a later part of a split file
            gguf = name;
        } else if (lower.ends_with(".ninferno") &&
                   it->path().parent_path().filename() == ".ninferno-cache") {
            // <stem>.<recipe>.ninferno: the stem is the GGUF's.
            const std::string stem = utf8(it->path().stem()); // <stem>.<recipe>
            const auto dot         = stem.rfind('.');
            if (dot == std::string::npos) { continue; }
            gguf     = stem.substr(0, dot) + ".gguf";
            artifact = true;
        } else {
            continue;
        }
        if (ascii_lower(gguf).starts_with("mmproj") || ascii_lower(gguf).find("mtp") == 0) {
            continue;
        }
        if (const std::string label = convert::gguf_quant_label(gguf); !label.empty()) {
            // A downloaded file before its conversion: the header is the GGUF's.
            if (!artifact || !out.contains(label)) { out[label] = it->path(); }
        }
    }
    return out;
}

// A quantization here: its own tensor table (nullopt: unreadable, a split part). A GGUF's from
// its header; a converted artifact kept in its place (its GGUF removed) from its tensors with its
// own drafter and MTP layers, as the artifact's own plan counts them (the hub's attached drafter
// may be unknown offline: the artifact's own copy is the one a start loads).
namespace {

std::optional<GgufTensorTable> read_tensor_table(const fs::path& path) {
    try {
        if (ascii_lower(path.extension().string()) == ".gguf") {
            return gguf_tensor_table(
                plan_facts_of_gguf(read_local_head(path), local_gguf_bytes(path)));
        }
        const PlanSourceFacts facts = plan_facts_of_artifact(path);
        if (facts.model.weight_basis != runtime::WeightBasis::Artifact ||
            facts.model.weight_bytes == 0) {
            return std::nullopt;
        }
        return GgufTensorTable{.tensor_bytes      = facts.model.weight_bytes,
                                        .mtp_bytes         = facts.model.mtp_bytes,
                                        .own_drafter_bytes = facts.model.own_drafter_bytes,
                                        .basis             = runtime::WeightBasis::Artifact};
    } catch (const std::exception&) { return std::nullopt; }
}

} // namespace

std::optional<GgufTensorTable> local_tensor_table(const fs::path& path) {
    // Read once per file as it is (its size and time): a plan is made often, a file rarely moves.
    using Key = std::tuple<std::string, std::uintmax_t, fs::file_time_type>;
    static std::mutex mutex;
    static std::map<Key, std::optional<GgufTensorTable>> kept;
    std::error_code error;
    const auto size = fs::file_size(path, error);
    if (error) { return std::nullopt; }
    const auto time = fs::last_write_time(path, error);
    if (error) { return std::nullopt; }
    const Key key{utf8(path), size, time};
    {
        std::lock_guard lock(mutex);
        if (const auto it = kept.find(key); it != kept.end()) { return it->second; }
    }
    auto table = read_tensor_table(path); // outside the lock: a header read
    std::lock_guard lock(mutex);
    if (kept.size() >= 256) { kept.clear(); }
    kept[key] = table;
    return table;
}

std::string cached_stamp(const std::map<std::string, fs::path>& cached) {
    std::string out;
    for (const auto& [label, path] : cached) {
        std::error_code error;
        const auto size = fs::file_size(path, error);
        const auto time = fs::last_write_time(path, error);
        out += label + "=" + utf8(path) + ":" + std::to_string(size) + ":" +
               std::to_string(time.time_since_epoch().count()) + ";";
    }
    return out;
}

// The quantizations already here price their own tensor tables (read from their files, over a
// hub header's: the file a start loads): the cached one a start keeps is planned as its explicit
// --quant would be.
void tables_of_cached(std::vector<QuantListing>& listing,
                      const std::map<std::string, fs::path>& cached) {
    for (auto& q : listing) {
        if (q.drafter || q.format != "gguf") { continue; }
        if (const auto it = cached.find(q.quant); it != cached.end()) {
            if (auto local = local_tensor_table(it->second)) { q.table = std::move(local); }
        }
    }
}

} // namespace ninferno::product
