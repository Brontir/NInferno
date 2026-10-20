#include "engine.h"
#include "strings.h"

#include <objbase.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#ifndef NINFERNO_VERSION
#define NINFERNO_VERSION "0.0.0-dev"
#endif

namespace setup {

namespace {

constexpr const wchar_t* kManifestUrl   = L"https://ninferno.brontir.com/releases/latest.json";
// The release key (tools/release/signing.pub.pem, its SubjectPublicKeyInfo): only a manifest it
// signed is acted on, so a release server that is broken into cannot hand this setup a file.
constexpr const char* kReleaseKey =
    "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE1R5JganVYknHjhX9nN2fYLqxwe2GR9Zeg0QL/"
    "WoCKgTm4fpmwOM4ASQ+0Av3ASV+RPHQH1rTaXBSHyWCPZokZg==";
constexpr const wchar_t* kUninstallKey  = L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\NInferno";
constexpr const char* kRootfs           = "ninferno-wsl-rootfs.tar.gz";
constexpr const wchar_t* kSettingsKey   = L"Software\\NInferno";
constexpr const wchar_t* kLinuxPath     = L"PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

std::wstring wsl_exe() {
    // A 32-bit process would see SysWOW64 without wsl.exe; this program is 64-bit.
    return known_folder(CSIDL_SYSTEM) + L"\\wsl.exe";
}

// C:\Users\me\x -> /mnt/c/Users/me/x (the distribution mounts drives under /mnt, wsl.conf).
std::wstring linux_path(const std::wstring& windows) {
    if (windows.size() < 3 || windows[1] != L':') { throw std::runtime_error("not a drive path: " + narrow(windows)); }
    std::wstring out = L"/mnt/";
    out += static_cast<wchar_t>(towlower(windows[0]));
    for (std::size_t i = 2; i < windows.size(); ++i) { out += windows[i] == L'\\' ? L'/' : windows[i]; }
    return out;
}

// A command in the distribution as root, its output to the log and to `on_line`.
DWORD in_distro(const std::wstring& command, std::string* output = nullptr,
                const std::function<void(const std::string&)>& on_line = {}) {
    const std::wstring line = quote(wsl_exe()) + L" -d " + kDistro + L" -u root --cd / -- " + command;
    log_line("$ " + narrow(line));
    std::string text;
    const DWORD code = run(line, &text, [&](const std::string& l) {
        log_line("  " + l);
        if (on_line) { on_line(l); }
    });
    if (output) { *output = text; }
    return code;
}

std::wstring trim(std::wstring s) {
    while (!s.empty() && iswspace(s.back())) { s.pop_back(); }
    while (!s.empty() && iswspace(s.front())) { s.erase(0, 1); }
    return s;
}

void make_dirs(const std::wstring& path) { SHCreateDirectoryExW(nullptr, path.c_str(), nullptr); }

void shortcut(const std::wstring& link, const std::wstring& target, const std::wstring& description) {
    IShellLinkW* shell = nullptr;
    if (FAILED(CoCreateInstance(CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER, IID_IShellLinkW,
                                reinterpret_cast<void**>(&shell)))) {
        return;
    }
    shell->SetPath(target.c_str());
    shell->SetDescription(description.c_str());
    shell->SetIconLocation(target.c_str(), 0);
    IPersistFile* file = nullptr;
    if (SUCCEEDED(shell->QueryInterface(IID_IPersistFile, reinterpret_cast<void**>(&file)))) {
        file->Save(link.c_str(), TRUE);
        file->Release();
    }
    shell->Release();
}

std::wstring start_menu_link() { return known_folder(CSIDL_PROGRAMS) + L"\\NInferno.lnk"; }
std::wstring desktop_link() { return known_folder(CSIDL_DESKTOPDIRECTORY) + L"\\NInferno.lnk"; }

} // namespace

const Asset* Manifest::asset(const std::string& name) const {
    const auto it = assets.find(name);
    return it == assets.end() ? nullptr : &it->second;
}

std::wstring app_dir() { return known_folder(CSIDL_LOCAL_APPDATA) + L"\\Programs\\NInferno"; }
std::wstring data_dir() { return known_folder(CSIDL_LOCAL_APPDATA) + L"\\NInferno"; }
std::wstring installed_exe() { return app_dir() + L"\\NInferno.exe"; }
std::wstring log_path() { return data_dir() + L"\\setup.log"; }

void log_line(const std::string& text) {
    static CRITICAL_SECTION lock = [] { CRITICAL_SECTION c; InitializeCriticalSection(&c); return c; }();
    EnterCriticalSection(&lock);
    make_dirs(data_dir());
    if (FILE* file = _wfopen(log_path().c_str(), L"ab")) {
        char stamp[32];
        const std::time_t now = std::time(nullptr);
        std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S ", std::localtime(&now));
        std::fputs(stamp, file);
        std::fputs(text.c_str(), file);
        std::fputs("\r\n", file);
        std::fclose(file);
    }
    LeaveCriticalSection(&lock);
}

Gpu detect_gpu() {
    Gpu gpu;
    std::string out;
    DWORD code = 1;
    try {
        code = run(L"nvidia-smi.exe --query-gpu=name,compute_cap,memory.total,driver_version "
                   L"--format=csv,noheader,nounits",
                   &out);
    } catch (const std::exception&) {
        return gpu; // no driver: no nvidia-smi
    }
    log_line("nvidia-smi: " + out);
    if (code != 0) { return gpu; } // ran, but could not talk to the driver
    gpu.has_driver = true;
    // One line per GPU: "NVIDIA GeForce RTX 5090, 12.0, 32607, 616.64". The native release runs
    // on every GPU from compute capability 7.0 (an engine per generation, PTX for later ones), so
    // a machine qualifies when one of its GPUs does; the page then offers each GPU its models.
    std::size_t at = 0;
    while (at < out.size()) {
        std::size_t end = out.find('\n', at);
        if (end == std::string::npos) { end = out.size(); }
        const std::wstring line = trim(widen(out.substr(at, end - at)));
        at                      = end + 1;
        std::vector<std::wstring> fields;
        std::size_t start = 0;
        for (std::size_t comma; (comma = line.find(L',', start)) != std::wstring::npos; start = comma + 1) {
            fields.push_back(trim(line.substr(start, comma - start)));
        }
        fields.push_back(trim(line.substr(start)));
        if (fields.size() < 4) { continue; }
        const double cc = _wtof(fields[1].c_str());
        if (!gpu.found) {
            gpu.compute_cap = fields[1];
            gpu.driver      = fields[3];
        }
        gpu.name += (gpu.found ? L" + " : L"") + fields[0];
        gpu.found = true;
        gpu.memory_mib = std::max<std::uint64_t>(gpu.memory_mib, _wcstoui64(fields[2].c_str(), nullptr, 10));
        if (cc >= 7.0 && gpu.arch.empty()) {
            // The Linux engine WSL would run (one generation per archive there).
            const std::string cap = narrow(fields[1]);
            gpu.arch = cap == "12.0" ? "120a" : cap == "8.9" ? "89" : cap == "8.6" ? "86"
                     : cap == "8.0"  ? "80"   : cap == "7.5" ? "75" : cap == "7.0" ? "70" : "89";
            gpu.compute_cap = fields[1];
        }
        // CUDA 13 (driver 580) for every engine but Volta's, CUDA 12.8 (570).
        gpu.min_driver = std::max(gpu.min_driver, cc >= 7.0 && cc < 7.5 ? 570 : 580);
    }
    return gpu;
}

std::wstring manifest_url() {
    const std::wstring url = env(L"NINFERNO_UPDATE_URL");
    return url.empty() ? kManifestUrl : url;
}

Manifest read_manifest() {
    const std::string text = http_get(manifest_url());
    if (!verify_signature(text, http_get(manifest_url() + L".sig"), kReleaseKey)) {
        throw std::runtime_error("the release manifest is not signed by the NInferno release key");
    }
    const auto json = Json::parse(text);
    Manifest manifest;
    manifest.version     = json.str("version");
    manifest.min_version = json.str("min_version");
    manifest.notes       = json.str("notes");
    if (manifest.version.empty()) { throw std::runtime_error("the release manifest names no version"); }
    if (const Json* assets = json.get("assets"); assets && assets->kind == Json::Kind::Object) {
        for (const auto& [name, value] : assets->object) {
            Asset asset;
            asset.url    = widen(value.str("url"));
            asset.sha256 = value.str("sha256");
            if (const Json* size = value.get("size")) { asset.size = static_cast<std::uint64_t>(size->number); }
            manifest.assets[name] = asset;
        }
    }
    return manifest;
}

std::string installed_version() {
    wchar_t value[64];
    DWORD size = sizeof(value);
    if (RegGetValueW(HKEY_CURRENT_USER, kUninstallKey, L"DisplayVersion", RRF_RT_REG_SZ, nullptr, value,
                     &size) != ERROR_SUCCESS) {
        return "";
    }
    return narrow(value);
}

bool wsl_ready() {
    std::string out;
    try {
        // --status answers 0 once WSL (the Store package or the Windows feature) works.
        return run(quote(wsl_exe()) + L" --status", &out) == 0;
    } catch (const std::exception&) { return false; }
}

bool wsl_turn_on() {
    SHELLEXECUTEINFOW info{sizeof(info)};
    info.fMask        = SEE_MASK_NOCLOSEPROCESS;
    info.lpVerb       = L"runas";
    const std::wstring exe = wsl_exe();
    info.lpFile       = exe.c_str();
    info.lpParameters = L"--install --no-distribution";
    info.nShow        = SW_SHOWNORMAL;
    log_line("wsl --install --no-distribution (elevated)");
    if (!ShellExecuteExW(&info)) { return false; } // declined
    WaitForSingleObject(info.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(info.hProcess, &code);
    CloseHandle(info.hProcess);
    log_line("  exit " + std::to_string(code));
    return true;
}

bool distro_exists() {
    std::string out;
    try {
        if (run(quote(wsl_exe()) + L" --list --quiet", &out) != 0) { return false; }
    } catch (const std::exception&) { return false; }
    std::size_t start = 0;
    for (std::size_t nl; start < out.size(); start = nl + 1) {
        nl = out.find('\n', start);
        if (nl == std::string::npos) { nl = out.size(); }
        std::wstring name = trim(widen(out.substr(start, nl - start)));
        if (_wcsicmp(name.c_str(), kDistro) == 0) { return true; }
    }
    return false;
}

std::wstring fetch_asset(const Manifest& manifest, const std::string& name, const Report& report,
                         const std::function<bool()>& cancelled) {
    const Asset* asset = manifest.asset(name);
    if (!asset) { throw std::runtime_error("the release has no " + name); }
    const std::wstring dir = data_dir() + L"\\downloads";
    make_dirs(dir);
    const std::wstring path = dir + L"\\" + widen(name);
    if (GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES && sha256_file(path) == asset->sha256) {
        return path; // downloaded before, intact
    }
    log_line("download " + narrow(asset->url));
    DWORD last = 0;
    http_download(asset->url, path, [&](std::uint64_t done, std::uint64_t total) {
        if (total == 0) { total = asset->size; }
        if (GetTickCount() - last > 200 || done == total) {
            last = GetTickCount();
            report(fmt(word(Word::Downloading), {widen(name), format_bytes(done),
                                                 total ? format_bytes(total) : std::wstring(L"?")}),
                   total ? static_cast<double>(done) / static_cast<double>(total) : -1);
        }
        return !cancelled();
    });
    report(word(Word::Verifying), -1);
    const std::string got = sha256_file(path);
    if (got != asset->sha256) {
        DeleteFileW(path.c_str());
        throw std::runtime_error(name + ": checksum mismatch (expected " + asset->sha256 + ", got " + got + ")");
    }
    return path;
}

namespace {

// One archive for every GPU (package-native.ps1): the engines of all generations beside the dispatcher.
std::string native_asset(const Gpu&) { return "ninferno-windows-x86_64.zip"; }
std::wstring engine_root() { return app_dir() + L"\\engine"; }

// The engine folder in use (engine\current.txt), "" when none.
std::wstring current_engine() {
    std::ifstream in(std::filesystem::path(engine_root()) / "current.txt");
    std::string name;
    std::getline(in, name);
    return name.empty() ? std::wstring() : engine_root() + L"\\" + widen(name);
}

void remember_backend(bool native) {
    HKEY key = nullptr;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, kSettingsKey, 0, nullptr, 0, KEY_WRITE, nullptr, &key, nullptr) ==
        ERROR_SUCCESS) {
        const DWORD value = native ? 1 : 0;
        RegSetValueExW(key, L"Native", 0, REG_DWORD, reinterpret_cast<const BYTE*>(&value), sizeof(value));
        RegCloseKey(key);
    }
}

// The native engine: the release's zip unpacked beside the older ones (a running page keeps its
// files), checked with ninferno-serve --version, then made current; the one before stays.
void install_native(const Manifest& manifest, const Gpu& gpu, const Report& report,
                    const std::function<bool()>& cancelled) {
    const std::wstring zip = fetch_asset(manifest, native_asset(gpu), report, cancelled);
    report(word(Word::Unpacking), -1);
    make_dirs(engine_root());
    // tar.exe (bsdtar) reads zip and ships with Windows 10 1803 and later.
    std::string out;
    const std::wstring tar = known_folder(CSIDL_SYSTEM) + L"\\tar.exe";
    if (run(quote(tar) + L" -xf " + quote(zip) + L" -C " + quote(engine_root()), &out) != 0) {
        throw std::runtime_error("cannot unpack " + native_asset(gpu) + ": " + out);
    }
    const std::string top = "ninferno-" + manifest.version + "-windows-x86_64";
    const std::wstring dir = engine_root() + L"\\" + widen(top);
    report(fmt(word(Word::Installing), {widen(manifest.version)}), -1);
    out.clear();
    const DWORD code = run(quote(dir + L"\\bin\\ninferno-serve.exe") + L" --version", &out);
    log_line("ninferno-serve --version: " + out);
    if (code != 0) {
        throw std::runtime_error("this build cannot run here: " + out.substr(0, 400) +
                                 " (a newer NVIDIA driver usually fixes it)");
    }
    const std::wstring previous = current_engine();
    {
        std::ofstream current(std::filesystem::path(engine_root()) / "current.txt", std::ios::trunc);
        current << top << "\n";
    }
    // Keep the new engine and the one before it.
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(engine_root(), ec)) {
        if (!entry.is_directory()) { continue; }
        const std::wstring path = entry.path().wstring();
        if (_wcsicmp(path.c_str(), dir.c_str()) != 0 && _wcsicmp(path.c_str(), previous.c_str()) != 0) {
            std::filesystem::remove_all(entry.path(), ec);
        }
    }
    DeleteFileW(zip.c_str());
    remember_backend(true);
}

} // namespace

bool native_engine(const Manifest& manifest, const Gpu& gpu) {
    if (env(L"NINFERNO_FORCE_WSL") == L"1") { return false; }
    return manifest.asset(native_asset(gpu)) != nullptr;
}

bool installed_native() {
    DWORD value = 0, size = sizeof(value);
    return RegGetValueW(HKEY_CURRENT_USER, kSettingsKey, L"Native", RRF_RT_REG_DWORD, nullptr, &value, &size) ==
               ERROR_SUCCESS &&
           value == 1;
}

void install_engine(const Manifest& manifest, const Gpu& gpu, const Report& report,
                    const std::function<bool()>& cancelled) {
    if (native_engine(manifest, gpu)) {
        install_native(manifest, gpu, report, cancelled);
        return;
    }
    if (!distro_exists()) {
        const std::wstring rootfs = fetch_asset(manifest, kRootfs, report, cancelled);
        report(word(Word::Unpacking), -1);
        const std::wstring disk = data_dir() + L"\\wsl";
        make_dirs(disk);
        std::string out;
        const std::wstring line = quote(wsl_exe()) + L" --import " + kDistro + L" " + quote(disk) + L" " +
                                  quote(rootfs) + L" --version 2";
        log_line("$ " + narrow(line));
        if (run(line, &out) != 0) {
            log_line(out);
            throw std::runtime_error("wsl --import: " + out);
        }
        DeleteFileW(rootfs.c_str());
    }
    const std::string archive_name = "ninferno-linux-x86_64-sm" + gpu.arch + ".tar.gz";
    const std::wstring installer   = fetch_asset(manifest, "install.sh", report, cancelled);
    const std::wstring archive     = fetch_asset(manifest, archive_name, report, cancelled);
    report(fmt(word(Word::Installing), {widen(manifest.version)}), -1);
    std::string out;
    const std::wstring command = std::wstring(L"env ") + kLinuxPath + L" NINFERNO_ARCHIVE=" +
                                 quote(linux_path(archive)) +
                                 L" NINFERNO_HOME=/opt/ninferno NINFERNO_BIN=/usr/local/bin sh " +
                                 quote(linux_path(installer));
    const DWORD code = in_distro(command, &out, [&](const std::string& l) { report(widen(l), -1); });
    if (code != 0) {
        // install.sh's last line says why (no GPU in WSL, a driver too old to run this build).
        std::string last = out;
        while (!last.empty() && (last.back() == '\n' || last.back() == '\r')) { last.pop_back(); }
        last = last.substr(last.rfind('\n') == std::string::npos ? 0 : last.rfind('\n') + 1);
        throw std::runtime_error(last.empty() ? "install.sh failed" : last);
    }
    DeleteFileW(archive.c_str()); // the engine is unpacked; the archive is only taking space
    remember_backend(false);
}

void register_app(const std::string& version) {
    make_dirs(app_dir());
    wchar_t self[MAX_PATH];
    GetModuleFileNameW(nullptr, self, MAX_PATH);
    const std::wstring target = installed_exe();
    if (_wcsicmp(self, target.c_str()) != 0) {
        // The running launcher holds the old file: move it aside, the copy takes its name.
        const std::wstring old = target + L".old";
        DeleteFileW(old.c_str());
        MoveFileExW(target.c_str(), old.c_str(), MOVEFILE_REPLACE_EXISTING);
        if (!CopyFileW(self, target.c_str(), FALSE)) {
            throw std::runtime_error("cannot copy NInferno.exe into " + narrow(app_dir()));
        }
    }
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    shortcut(start_menu_link(), target, word(Word::Subtitle));
    shortcut(desktop_link(), target, word(Word::Subtitle));
    CoUninitialize();
    HKEY key = nullptr;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, kUninstallKey, 0, nullptr, 0, KEY_WRITE, nullptr, &key, nullptr) ==
        ERROR_SUCCESS) {
        const auto set = [&](const wchar_t* name, const std::wstring& value) {
            RegSetValueExW(key, name, 0, REG_SZ, reinterpret_cast<const BYTE*>(value.c_str()),
                           static_cast<DWORD>((value.size() + 1) * sizeof(wchar_t)));
        };
        const auto set_dword = [&](const wchar_t* name, DWORD value) {
            RegSetValueExW(key, name, 0, REG_DWORD, reinterpret_cast<const BYTE*>(&value), sizeof(value));
        };
        set(L"DisplayName", L"NInferno");
        set(L"DisplayVersion", widen(version));
        set(L"Publisher", L"Brontir");
        set(L"DisplayIcon", target);
        set(L"InstallLocation", app_dir());
        set(L"UninstallString", quote(target) + L" --uninstall");
        set(L"QuietUninstallString", quote(target) + L" --uninstall --quiet");
        set(L"URLInfoAbout", L"https://ninferno.brontir.com");
        set_dword(L"NoModify", 1);
        set_dword(L"NoRepair", 1);
        set_dword(L"EstimatedSize", 1200 * 1024); // KB: the engine; models come on top
        RegCloseKey(key);
    }
    log_line("registered NInferno " + version);
}

void unregister_app() {
    stop_engine();
    if (distro_exists()) {
        std::string out;
        run(quote(wsl_exe()) + L" --unregister " + kDistro, &out);
        log_line("wsl --unregister: " + out);
    }
    RegDeleteKeyW(HKEY_CURRENT_USER, kSettingsKey);
    DeleteFileW(start_menu_link().c_str());
    DeleteFileW(desktop_link().c_str());
    RegDeleteKeyW(HKEY_CURRENT_USER, kUninstallKey);
    // The running program cannot delete itself: a detached cmd removes the folders after it exits.
    const std::wstring command = L"cmd.exe /c ping -n 3 127.0.0.1 >nul & rmdir /s /q " + quote(app_dir()) +
                                 L" & rmdir /s /q " + quote(data_dir());
    if (HANDLE process = spawn(command, DETACHED_PROCESS)) { CloseHandle(process); }
}

void stop_engine() {
    if (installed_native()) { return; } // the page ends with its launcher (and its models with it)
    std::string out;
    try {
        run(quote(wsl_exe()) + L" --terminate " + kDistro, &out);
    } catch (const std::exception&) {}
}

HANDLE start_ui() {
    if (installed_native()) {
        // The page finds the updater in NINFERNO_UPDATER; its console stays hidden (its children
        // share it, so Ctrl+Break stops a model gracefully).
        SetEnvironmentVariableW(L"NINFERNO_UPDATER", installed_exe().c_str());
        const std::wstring line = quote(current_engine() + L"\\bin\\ninferno-ui.exe") + L" --no-browser";
        log_line("$ " + narrow(line));
        return spawn(line);
    }
    // The page asks this program to update (NINFERNO_UPDATER): it reaches it through WSL's
    // interop as /mnt/c/.../NInferno.exe.
    const std::wstring line = quote(wsl_exe()) + L" -d " + kDistro + L" -u root --cd /root -- env " +
                              kLinuxPath + L" NINFERNO_UPDATER=" + quote(linux_path(installed_exe())) +
                              L" /usr/local/bin/ninferno-ui --no-browser";
    log_line("$ " + narrow(line));
    return spawn(line);
}

bool ui_listening() {
    try {
        (void)http_get(L"http://localhost:" + std::to_wstring(kUiPort) + L"/api/state", 8u << 20);
        return true;
    } catch (const std::exception&) { return false; }
}

void open_page() {
    const std::wstring url = L"http://localhost:" + std::to_wstring(kUiPort) + L"/";
    ShellExecuteW(nullptr, L"open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

} // namespace setup
