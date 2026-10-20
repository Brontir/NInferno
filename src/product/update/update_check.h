#pragma once

// Whether a newer NInferno release is out: one small JSON file (the release manifest) read over
// https at most once a day, never in the way of the work. ninferno-serve logs one line when there
// is a newer release; ninferno-ui shows it with an Update button; ninferno-update (Linux) and
// NInferno.exe (Windows) download and install it.
//
// The manifest (tools/release/manifest.py writes it beside the release's files):
//
//   {"version": "0.2.0",                       the newest release
//    "date": "2026-10-20",
//    "notes": "https://.../releases/v0.2.0",   what changed (a page)
//    "min_version": "0.1.2",                   older installs must update (they get a warning
//                                               at every start instead of a quiet note)
//    "assets": {"<name>": {"url": ..., "sha256": ..., "size": ...}, ...}}
//
// Turned off by NINFERNO_UPDATE_CHECK=0 (off, no, false) and for builds that carry no release
// version (a source build reports "dev" or a commit): those never ask.

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>

namespace ninferno::update {

struct Release {
    std::string current;     // this build's version
    std::string latest;      // the manifest's
    std::string notes;       // the manifest's notes URL ("" when it gives none)
    bool newer    = false;   // latest > current
    bool required = false;   // current < min_version: this version should not be used further
};

struct CheckOptions {
    std::string current;                 // this build's version (NINFERNO_VERSION_STRING)
    std::string manifest_url;            // default_manifest_url() when empty
    std::filesystem::path cache_file;    // default_cache_file() when empty
    std::chrono::seconds max_age{24 * 3600}; // a cached answer younger than this is used as is
    std::chrono::seconds timeout{8};     // the whole request
};

// $NINFERNO_UPDATE_URL, else the release server's latest.json.
[[nodiscard]] std::string default_manifest_url();
// $XDG_CACHE_HOME/ninferno/update.json, else ~/.cache/ninferno/update.json (on Windows
// %LOCALAPPDATA%\NInferno\update.json).
[[nodiscard]] std::filesystem::path default_cache_file();
// False when NINFERNO_UPDATE_CHECK says 0, off, no or false.
[[nodiscard]] bool checks_enabled();

// A release version: "0.2.0", "v0.2.0", "0.2.0-rc1" (a pre-release sorts below its release).
// "dev", a commit hash or a `git describe` past a tag ("0.1.0-12-gabc1234") are not: such builds
// were not installed from a release and are never told to update.
[[nodiscard]] bool is_release_version(const std::string& version);
// <0, 0, >0 as a is older than, the same as, newer than b. Both must be release versions.
[[nodiscard]] int compare_versions(const std::string& a, const std::string& b);

// What the manifest's text says for `current`. Throws std::runtime_error when it is not a
// manifest (no "version").
[[nodiscard]] Release parse_manifest(const std::string& text, const std::string& current);

// The answer for options.current: the cached one while younger than max_age, else the manifest
// read now (and cached). nullopt when checks are off, the version is not a release, or the
// manifest cannot be read (no network: nothing is said).
[[nodiscard]] std::optional<Release> check(const CheckOptions& options);

// The one line ninferno-serve logs for a release, "" when there is nothing to say.
[[nodiscard]] std::string notice(const Release& release);

} // namespace ninferno::update
