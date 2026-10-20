#include "winutil.h"

#include <bcrypt.h>
#include <shlobj.h>
#include <winhttp.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace setup {

std::wstring widen(const std::string& text) {
    if (text.empty()) { return {}; }
    const int n = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring out(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), n);
    return out;
}

std::string narrow(const std::wstring& text) {
    if (text.empty()) { return {}; }
    const int n = WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0,
                                      nullptr, nullptr);
    std::string out(static_cast<std::size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), n, nullptr,
                        nullptr);
    return out;
}

std::string from_wsl_output(const std::string& bytes) {
    // UTF-16LE text has a zero byte after most ASCII characters.
    std::size_t zeros = 0;
    for (std::size_t i = 1; i < bytes.size(); i += 2) { zeros += bytes[i] == '\0'; }
    if (bytes.size() >= 4 && zeros * 4 >= bytes.size()) {
        std::wstring wide(bytes.size() / 2, L'\0');
        std::memcpy(wide.data(), bytes.data(), wide.size() * 2);
        if (!wide.empty() && wide[0] == 0xFEFF) { wide.erase(0, 1); }
        return narrow(wide);
    }
    return bytes;
}

std::wstring env(const wchar_t* name) {
    wchar_t buffer[32768];
    const DWORD n = GetEnvironmentVariableW(name, buffer, 32768);
    return n > 0 && n < 32768 ? std::wstring(buffer, n) : std::wstring();
}

std::wstring known_folder(int csidl) {
    wchar_t path[MAX_PATH];
    if (SHGetFolderPathW(nullptr, csidl, nullptr, SHGFP_TYPE_CURRENT, path) != S_OK) { return {}; }
    return path;
}

std::wstring format_bytes(std::uint64_t bytes) {
    wchar_t text[64];
    if (bytes >= (1ull << 30)) {
        swprintf(text, 64, L"%.1f GB", static_cast<double>(bytes) / (1ull << 30));
    } else {
        swprintf(text, 64, L"%.0f MB", static_cast<double>(bytes) / (1ull << 20));
    }
    return text;
}

std::wstring fmt(const wchar_t* pattern, std::initializer_list<std::wstring> args) {
    std::wstring out;
    auto next = args.begin();
    for (const wchar_t* p = pattern; *p; ++p) {
        if (p[0] == L'%' && (p[1] == L's' || p[1] == L'd') && next != args.end()) {
            out += *next++;
            ++p;
        } else {
            out += *p;
        }
    }
    return out;
}

// As src/core/command_line.h (this program builds without the source tree's headers): a run of
// backslashes doubles only where a quote follows it.
std::wstring quote(const std::wstring& arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\"") == std::wstring::npos) { return arg; }
    std::wstring out = L"\"";
    std::size_t slashes = 0;
    for (const wchar_t c : arg) {
        if (c == L'\\') {
            ++slashes;
            continue;
        }
        out.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\');
        out += c;
        slashes = 0;
    }
    out.append(slashes * 2, L'\\');
    return out + L"\"";
}

HANDLE spawn(const std::wstring& command_line, DWORD flags) {
    STARTUPINFOW si{sizeof(si)};
    PROCESS_INFORMATION pi{};
    std::wstring line = command_line;
    if (!CreateProcessW(nullptr, line.data(), nullptr, nullptr, FALSE, flags | CREATE_NO_WINDOW,
                        nullptr, nullptr, &si, &pi)) {
        return nullptr;
    }
    CloseHandle(pi.hThread);
    return pi.hProcess;
}

DWORD run(const std::wstring& command_line, std::string* output,
          const std::function<void(const std::string&)>& on_line) {
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE read = nullptr, write = nullptr;
    if (!CreatePipe(&read, &write, &sa, 0)) { throw std::runtime_error("CreatePipe failed"); }
    SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{sizeof(si)};
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = write;
    si.hStdError  = write;
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::wstring line = command_line;
    if (!CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                        nullptr, &si, &pi)) {
        CloseHandle(read);
        CloseHandle(write);
        throw std::runtime_error("cannot start: " + narrow(command_line));
    }
    CloseHandle(write);
    std::string all, partial;
    char buffer[4096];
    DWORD n = 0;
    while (ReadFile(read, buffer, sizeof(buffer), &n, nullptr) && n > 0) {
        all.append(buffer, n);
        if (on_line) {
            partial.append(buffer, n);
            // wsl.exe's own UTF-16 messages are converted once whole; Linux output line by line.
            for (std::size_t nl; (nl = partial.find('\n')) != std::string::npos;) {
                std::string text = partial.substr(0, nl);
                partial.erase(0, nl + 1);
                if (!text.empty() && text.back() == '\r') { text.pop_back(); }
                if (text.find('\0') == std::string::npos) { on_line(text); }
            }
        }
    }
    CloseHandle(read);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    if (output) { *output += from_wsl_output(all); }
    return code;
}

namespace {

struct Internet {
    HINTERNET handle = nullptr;
    explicit Internet(HINTERNET h) : handle(h) {}
    ~Internet() { if (handle) { WinHttpCloseHandle(handle); } }
    Internet(const Internet&)            = delete;
    Internet& operator=(const Internet&) = delete;
};

std::string last_error(const char* what) {
    return std::string(what) + " failed (error " + std::to_string(GetLastError()) + ")";
}

// Sends a GET and returns the request; follows redirects itself (WinHTTP's default policy).
void open_request(const std::wstring& url, Internet& session, std::unique_ptr<Internet>& connection,
                  std::unique_ptr<Internet>& request) {
    URL_COMPONENTS parts{sizeof(parts)};
    wchar_t host[256], path[4096];
    parts.lpszHostName     = host;
    parts.dwHostNameLength = 256;
    parts.lpszUrlPath      = path;
    parts.dwUrlPathLength  = 4096;
    wchar_t extra[4096];
    parts.lpszExtraInfo     = extra;
    parts.dwExtraInfoLength = 4096;
    if (!WinHttpCrackUrl(url.c_str(), 0, 0, &parts)) { throw std::runtime_error("bad URL " + narrow(url)); }
    connection = std::make_unique<Internet>(WinHttpConnect(session.handle, host, parts.nPort, 0));
    if (!connection->handle) { throw std::runtime_error(last_error("connect")); }
    const std::wstring object = std::wstring(path) + extra;
    const DWORD secure        = parts.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0;
    request = std::make_unique<Internet>(WinHttpOpenRequest(connection->handle, L"GET", object.c_str(),
                                                            nullptr, WINHTTP_NO_REFERER,
                                                            WINHTTP_DEFAULT_ACCEPT_TYPES, secure));
    if (!request->handle) { throw std::runtime_error(last_error("request")); }
    if (!WinHttpSendRequest(request->handle, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA,
                            0, 0, 0) ||
        !WinHttpReceiveResponse(request->handle, nullptr)) {
        throw std::runtime_error(last_error(("download of " + narrow(url)).c_str()));
    }
    DWORD status = 0, size = sizeof(status);
    WinHttpQueryHeaders(request->handle, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &status, &size, WINHTTP_NO_HEADER_INDEX);
    if (status != 200) {
        throw std::runtime_error(narrow(url) + ": HTTP " + std::to_string(status));
    }
}

std::unique_ptr<Internet> open_session() {
    auto session = std::make_unique<Internet>(WinHttpOpen(L"NInferno-Setup/1", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                                 WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
    if (!session->handle) {
        // Windows before 8.1 has no automatic proxy: the system's settings instead.
        session->handle = WinHttpOpen(L"NInferno-Setup/1", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                     WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    }
    if (!session->handle) { throw std::runtime_error(last_error("WinHttpOpen")); }
    WinHttpSetTimeouts(session->handle, 15000, 15000, 30000, 60000);
    return session;
}

} // namespace

std::string http_get(const std::wstring& url, std::size_t max_bytes) {
    auto session = open_session();
    std::unique_ptr<Internet> connection, request;
    open_request(url, *session, connection, request);
    std::string body;
    char buffer[16384];
    DWORD n = 0;
    while (WinHttpReadData(request->handle, buffer, sizeof(buffer), &n) && n > 0) {
        body.append(buffer, n);
        if (body.size() > max_bytes) { throw std::runtime_error(narrow(url) + ": too large"); }
    }
    return body;
}

void http_download(const std::wstring& url, const std::wstring& path, const Progress& progress) {
    auto session = open_session();
    std::unique_ptr<Internet> connection, request;
    open_request(url, *session, connection, request);
    std::uint64_t total = 0;
    wchar_t length[32];
    DWORD size = sizeof(length);
    if (WinHttpQueryHeaders(request->handle, WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_HEADER_NAME_BY_INDEX,
                            length, &size, WINHTTP_NO_HEADER_INDEX)) {
        total = _wcstoui64(length, nullptr, 10);
    }
    const std::wstring partial = path + L".partial";
    HANDLE file = CreateFileW(partial.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) { throw std::runtime_error("cannot write " + narrow(partial)); }
    std::vector<char> buffer(1 << 20);
    std::uint64_t done = 0;
    DWORD n = 0, written = 0;
    bool ok = true;
    while ((ok = WinHttpReadData(request->handle, buffer.data(), static_cast<DWORD>(buffer.size()), &n)) &&
           n > 0) {
        if (!WriteFile(file, buffer.data(), n, &written, nullptr) || written != n) {
            CloseHandle(file);
            DeleteFileW(partial.c_str());
            throw std::runtime_error("cannot write " + narrow(partial) + " (disk full?)");
        }
        done += n;
        if (progress && !progress(done, total)) {
            CloseHandle(file);
            DeleteFileW(partial.c_str());
            throw std::runtime_error("cancelled");
        }
    }
    CloseHandle(file);
    if (!ok || (total != 0 && done != total)) {
        DeleteFileW(partial.c_str());
        throw std::runtime_error("download of " + narrow(url) + " broke off");
    }
    if (!MoveFileExW(partial.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        throw std::runtime_error("cannot write " + narrow(path));
    }
}

std::string sha256_file(const std::wstring& path) {
    HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (file == INVALID_HANDLE_VALUE) { throw std::runtime_error("cannot read " + narrow(path)); }
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash     = nullptr;
    BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
    BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0);
    std::vector<unsigned char> buffer(1 << 20);
    DWORD n = 0;
    while (ReadFile(file, buffer.data(), static_cast<DWORD>(buffer.size()), &n, nullptr) && n > 0) {
        BCryptHashData(hash, buffer.data(), n, 0);
    }
    CloseHandle(file);
    unsigned char digest[32];
    BCryptFinishHash(hash, digest, sizeof(digest), 0);
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
    std::string hex;
    char two[3];
    for (const unsigned char byte : digest) {
        std::snprintf(two, sizeof(two), "%02x", byte);
        hex += two;
    }
    return hex;
}

// ---- JSON -------------------------------------------------------------------------------------

namespace {

struct Parser {
    const std::string& s;
    std::size_t i = 0;

    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("bad JSON: ") + what + " at " + std::to_string(i));
    }
    void space() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) { ++i; } }
    bool take(char c) { space(); if (i < s.size() && s[i] == c) { ++i; return true; } return false; }

    std::string string() {
        if (!take('"')) { fail("string expected"); }
        std::string out;
        while (i < s.size() && s[i] != '"') {
            char c = s[i++];
            if (c != '\\') { out += c; continue; }
            if (i >= s.size()) { fail("escape"); }
            c = s[i++];
            switch (c) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'u': {
                if (i + 4 > s.size()) { fail("\\u"); }
                const unsigned code = static_cast<unsigned>(std::stoul(s.substr(i, 4), nullptr, 16));
                i += 4;
                out += narrow(std::wstring(1, static_cast<wchar_t>(code)));
                break;
            }
            default: out += c; break; // \" \\ \/
            }
        }
        if (i >= s.size()) { fail("unterminated string"); }
        ++i;
        return out;
    }

    Json value() {
        space();
        if (i >= s.size()) { fail("value expected"); }
        Json v;
        const char c = s[i];
        if (c == '{') {
            ++i;
            v.kind = Json::Kind::Object;
            if (take('}')) { return v; }
            do {
                std::string key = string();
                if (!take(':')) { fail("':' expected"); }
                v.object[key] = value();
            } while (take(','));
            if (!take('}')) { fail("'}' expected"); }
        } else if (c == '[') {
            ++i;
            v.kind = Json::Kind::Array;
            if (take(']')) { return v; }
            do { v.array.push_back(value()); } while (take(','));
            if (!take(']')) { fail("']' expected"); }
        } else if (c == '"') {
            v.kind   = Json::Kind::String;
            v.string = string();
        } else if (s.compare(i, 4, "true") == 0) {
            i += 4; v.kind = Json::Kind::Bool; v.boolean = true;
        } else if (s.compare(i, 5, "false") == 0) {
            i += 5; v.kind = Json::Kind::Bool;
        } else if (s.compare(i, 4, "null") == 0) {
            i += 4;
        } else {
            std::size_t used = 0;
            try { v.number = std::stod(s.substr(i, 32), &used); } catch (...) { fail("value"); }
            i += used;
            v.kind = Json::Kind::Number;
        }
        return v;
    }
};

bool split(std::string v, std::vector<long>& numbers, std::string& pre) {
    numbers.clear();
    pre.clear();
    if (!v.empty() && (v[0] == 'v' || v[0] == 'V')) { v.erase(0, 1); }
    if (const auto dash = v.find('-'); dash != std::string::npos) {
        pre = v.substr(dash + 1);
        v.resize(dash);
        if (pre.empty() || pre.find('-') != std::string::npos) { return false; }
    }
    std::stringstream parts(v);
    std::string part;
    while (std::getline(parts, part, '.')) {
        if (part.empty() || part.size() > 9 || part.find_first_not_of("0123456789") != std::string::npos) {
            return false;
        }
        numbers.push_back(std::stol(part));
    }
    return numbers.size() >= 2 && numbers.size() <= 4;
}

} // namespace

Json Json::parse(const std::string& text) {
    Parser parser{text};
    Json v = parser.value();
    parser.space();
    if (parser.i != text.size()) { parser.fail("trailing text"); }
    return v;
}

const Json* Json::get(const std::string& key) const {
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

std::string Json::str(const std::string& key) const {
    const Json* v = get(key);
    return v && v->kind == Kind::String ? v->string : std::string();
}

bool version_less(const std::string& a, const std::string& b, bool& comparable) {
    std::vector<long> na, nb;
    std::string pa, pb;
    comparable = split(a, na, pa) && split(b, nb, pb);
    if (!comparable) { return false; }
    na.resize(4, 0);
    nb.resize(4, 0);
    if (na != nb) { return na < nb; }
    if (pa == pb) { return false; }
    if (pa.empty()) { return false; }
    if (pb.empty()) { return true; }
    return pa < pb;
}

} // namespace setup
