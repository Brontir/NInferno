#include "product/plan/plan_cached.h"

#include "convert/fetch.h"
#include "convert/gguf.h"
#include "core/hub_revision.h"
#include "core/quant_label.h"
#include "product/plan/plan_facts.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace ninferno::product {

namespace fs = std::filesystem;

namespace {

std::string ascii_lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
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
                                            bytes.size(), size, path.string());
        } catch (const convert::GgufTruncated&) {
            if (n == size) { throw; }
        }
    }
    throw std::runtime_error(path.string() + ": the GGUF metadata runs past 512 MiB");
}

// Every part of a split GGUF beside `path` ("x-00001-of-00003.gguf"), or the file alone.
std::uint64_t local_gguf_bytes(const fs::path& path) {
    static const std::regex split(R"((.*)-\d{5}-of-(\d{5})\.gguf)", std::regex::icase);
    std::smatch match;
    const std::string name = path.filename().string();
    if (!std::regex_match(name, match, split)) { return fs::file_size(path); }
    std::uint64_t total = 0;
    for (const auto& entry : fs::directory_iterator(path.parent_path())) {
        std::smatch other;
        const std::string n = entry.path().filename().string();
        if (std::regex_match(n, other, split) && other[1] == match[1]) {
            total += entry.file_size();
        }
    }
    return total;
}

// The GGUF quantizations of a hub repository already in the local hub cache, each with its file:
// a downloaded GGUF (the first part of a split one), or the artifact converted from one
// (<snapshot>/.ninferno-cache/<gguf stem>.<recipe>.ninferno) where the download was removed.
std::map<std::string, fs::path> cached_quants(const fs::path& hub_root, const std::string& repo,
                                              const std::string& revision) {
    std::map<std::string, fs::path> out;
    std::error_code error;
    // The requested revision's snapshot alone (a commit, or the commit its ref names: "main" when
    // none was asked), never another revision's files.
    const fs::path repo_dir = hub_root / hub_repo_directory(repo);
    const auto commit       = convert::local_commit(repo_dir, revision);
    if (!commit) { return out; }
    const fs::path root = repo_dir / "snapshots" / *commit;
    if (!fs::is_directory(root, error)) { return out; }
    for (auto it = fs::recursive_directory_iterator(root, error);
         !error && it != fs::recursive_directory_iterator(); it.increment(error)) {
        const std::string name  = it->path().filename().string();
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
            const std::string stem = it->path().stem().string(); // <stem>.<recipe>
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
std::optional<GgufTensorTable> local_tensor_table(const fs::path& path) {
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
