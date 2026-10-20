#pragma once

// Small Windows helpers for NInferno-Setup.exe: text conversion, processes, https downloads
// (WinHTTP), SHA-256 (CNG), a minimal JSON reader for the release manifest, and paths.

#ifndef UNICODE
#define UNICODE
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace setup {

std::wstring widen(const std::string& text);
std::string narrow(const std::wstring& text);
// wsl.exe writes its own messages in UTF-16LE; Linux programs it runs write UTF-8.
std::string from_wsl_output(const std::string& bytes);
std::wstring env(const wchar_t* name);
std::wstring known_folder(int csidl);
std::wstring format_bytes(std::uint64_t bytes);
// Each %s or %d of `pattern` replaced by the next of `args`, in order.
std::wstring fmt(const wchar_t* pattern, std::initializer_list<std::wstring> args);

// Runs a command line (no window) and returns its exit code; its output (stdout and stderr) is
// appended to `output` when given, and each line goes to `on_line` as it comes.
DWORD run(const std::wstring& command_line, std::string* output = nullptr,
          const std::function<void(const std::string&)>& on_line = {});
// Starts a command line with no window and does not wait; the process handle (close it).
HANDLE spawn(const std::wstring& command_line, DWORD flags = 0);
// Quotes one argument for CreateProcess.
std::wstring quote(const std::wstring& arg);

// Progress of a download: bytes so far, total (0 when unknown). Return false to cancel.
using Progress = std::function<bool(std::uint64_t, std::uint64_t)>;
// GET over https (redirects followed). Throws std::runtime_error naming what failed.
std::string http_get(const std::wstring& url, std::size_t max_bytes = 4u << 20);
void http_download(const std::wstring& url, const std::wstring& path, const Progress& progress);
// Lower-case hex SHA-256 of a file.
std::string sha256_file(const std::wstring& path);

// A JSON value: just enough for the release manifest.
struct Json {
    enum class Kind { Null, Bool, Number, String, Array, Object } kind = Kind::Null;
    bool boolean   = false;
    double number  = 0;
    std::string string;
    std::vector<Json> array;
    std::map<std::string, Json> object;

    static Json parse(const std::string& text); // throws std::runtime_error
    [[nodiscard]] const Json* get(const std::string& key) const;
    [[nodiscard]] std::string str(const std::string& key) const; // "" when absent
};

// "v0.2.0-rc1" order; false when either is not a release version.
bool version_less(const std::string& a, const std::string& b, bool& comparable);

} // namespace setup
