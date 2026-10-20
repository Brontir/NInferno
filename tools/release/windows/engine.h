#pragma once

// What NInferno-Setup.exe installs and runs, apart from its window: the GPU check, the release
// manifest, the engine, the Start menu, and the entry in Windows' installed apps.
//
// The engine is the native Windows build for the GPU's architecture when the release has one
// (ninferno-windows-x86_64-sm<arch>.zip, unpacked under the program's folder), else the Linux build
// in NInferno's own WSL 2 distribution ("NInferno", not the user's Ubuntu: no Linux account,
// nothing from the Store).
//
// Layout (per user, no administrator rights except to turn WSL on):
//   %LOCALAPPDATA%\Programs\NInferno\NInferno.exe   this program: launcher, updater, uninstaller
//   %LOCALAPPDATA%\Programs\NInferno\engine\<release>\  the native engine (current.txt names it)
//   %LOCALAPPDATA%\NInferno\wsl\                    the distribution's disk (engine and models)
//   %LOCALAPPDATA%\NInferno\setup.log               what every run did
// Inside the distribution: /opt/ninferno (release.json's versions, `current`), /usr/local/bin.

#include "winutil.h"

#include <functional>
#include <string>

namespace setup {

constexpr const wchar_t* kDistro = L"NInferno";
constexpr int kUiPort            = 8090;

struct Gpu {
    bool has_driver = false; // nvidia-smi ran: the NVIDIA driver is installed
    bool found      = false; // it listed a GPU
    std::wstring name, compute_cap, driver;
    std::uint64_t memory_mib = 0;
    std::string arch;   // the release's build for it: 120a, 89, 86, 80, 75, 70; "" unsupported
    int min_driver = 0; // what its CUDA asks for
};

struct Asset {
    std::wstring url;
    std::string sha256;
    std::uint64_t size = 0;
};

struct Manifest {
    std::string version, min_version, notes;
    std::map<std::string, Asset> assets;
    [[nodiscard]] const Asset* asset(const std::string& name) const;
};

std::wstring app_dir();  // %LOCALAPPDATA%\Programs\NInferno
std::wstring data_dir(); // %LOCALAPPDATA%\NInferno
std::wstring installed_exe();
void log_line(const std::string& text);
std::wstring log_path();

Gpu detect_gpu();
std::wstring manifest_url(); // %NINFERNO_UPDATE_URL% or the release server's latest.json
Manifest read_manifest();    // throws
// The installed version ("" when not installed), from the uninstall entry.
std::string installed_version();

// Reports a step's progress: a line of text and a share done (-1: unknown).
using Report = std::function<void(const std::wstring&, double)>;

// WSL: present and working; turning it on (elevated) returns false when the user declined.
// Whether the release carries a native Windows engine for the GPU (else WSL 2 runs the Linux one).
bool native_engine(const Manifest& manifest, const Gpu& gpu);
// The engine this installation runs: true native, false WSL (what the last install chose).
bool installed_native();

bool wsl_ready();
bool wsl_turn_on();
bool distro_exists();

// Downloads `name` into the data directory's downloads and checks its SHA-256 (kept when it is
// already there and matches). The local path.
std::wstring fetch_asset(const Manifest& manifest, const std::string& name, const Report& report,
                         const std::function<bool()>& cancelled);

// The engine for `gpu` into the distribution (created from the release's root file system when
// missing). Throws std::runtime_error naming what failed.
void install_engine(const Manifest& manifest, const Gpu& gpu, const Report& report,
                    const std::function<bool()>& cancelled);

// This program into app_dir(), the Start menu and desktop shortcuts, and the uninstall entry.
void register_app(const std::string& version);
void unregister_app(); // and removes the distribution and the files

// Stops the engine (the page and every model in the distribution).
void stop_engine();
// The page's process (ninferno-ui in the distribution); null when it cannot start.
HANDLE start_ui();
bool ui_listening();
void open_page();

} // namespace setup
