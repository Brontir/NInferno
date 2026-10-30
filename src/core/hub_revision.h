#pragma once

#include "core/ascii.h"

#include <string>
#include <string_view>

namespace ninferno {

// A hub repository at a revision, "org/name@revision" (a commit, a prefix of one, or a ref): the
// identity a lab pair's target, an attached drafter and a conversion in the hub cache are named by.
struct HubRevision {
    std::string repo;     // org/name
    std::string revision; // "" any

    // "org/name@revision", or "org/name" for any revision.
    [[nodiscard]] static HubRevision parse(std::string_view id) {
        const auto at = id.find('@');
        return {std::string(id.substr(0, at)),
                at == std::string_view::npos ? std::string() : std::string(id.substr(at + 1))};
    }
    [[nodiscard]] std::string id() const { return revision.empty() ? repo : repo + "@" + revision; }
    [[nodiscard]] bool empty() const noexcept { return repo.empty(); }

    // Whether the two name one revision: the same repository (case aside), and the same revision
    // where both name one (either may be a prefix of the other). True when either is unknown.
    [[nodiscard]] bool same_as(const HubRevision& other) const {
        if (empty() || other.empty()) { return true; }
        if (ascii_lower(repo) != ascii_lower(other.repo)) { return false; }
        if (revision.empty() || other.revision.empty()) { return true; }
        const auto a = ascii_lower(revision), b = ascii_lower(other.revision);
        return a.starts_with(b) || b.starts_with(a);
    }
};

// A repository's directory in a hub cache, as Hugging Face's own tools name it: org/name ->
// models--org--name.
[[nodiscard]] inline std::string hub_repo_directory(std::string_view repo) {
    std::string name = "models--" + std::string(repo);
    if (const auto slash = name.find('/'); slash != std::string::npos) { name.replace(slash, 1, "--"); }
    return name;
}

// And back: models--org--name -> org/name (the first "--" the slash; a name may hold more); "" for a
// directory that is no model repository's.
[[nodiscard]] inline std::string hub_repo_of_directory(std::string_view directory) {
    if (!directory.starts_with("models--")) { return {}; }
    std::string repo(directory.substr(8));
    if (const auto separator = repo.find("--"); separator != std::string::npos) {
        repo.replace(separator, 2, "/");
    }
    return repo;
}

} // namespace ninferno
