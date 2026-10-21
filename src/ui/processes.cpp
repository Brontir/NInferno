#include "ui/processes.h"

#include <regex>
#include <vector>

#include "core/clock.h"
#include "core/command_line.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <cerrno>
#include <cstring>
#include <set>
#include <stdexcept>

#ifndef _WIN32
extern char** environ;
#endif

namespace ninferno::ui {

namespace {

// "" for a time never set.
std::string iso(std::chrono::system_clock::time_point t) {
    return t.time_since_epoch().count() == 0 ? std::string() : utc_iso8601(t);
}

#ifdef _WIN32
bool port_free(int port) {
    static const bool started = [] {
        WSADATA data;
        return WSAStartup(MAKEWORD(2, 2), &data) == 0;
    }();
    if (!started) { return false; }
    const SOCKET fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd == INVALID_SOCKET) { return false; }
    // Exclusive, as a listener would bind it (Windows otherwise lets a second bind share it).
    BOOL exclusive = TRUE;
    ::setsockopt(fd, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&exclusive),
                 sizeof(exclusive));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(static_cast<std::uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const bool ok        = ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    ::closesocket(fd);
    return ok;
}

std::wstring wide(const std::string& text) {
    if (text.empty()) { return {}; }
    const int n = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring out(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), n);
    return out;
}

#else
bool port_free(int port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { return false; }
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(static_cast<std::uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const bool ok        = ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    ::close(fd);
    return ok;
}
#endif

// ninferno-serve's start-up lines the UI shows beside the process.
void note_fact(ProcessInfo& info, const std::string& line) {
    const auto take = [&](const char* tag, std::string& slot, bool first_only) {
        const auto at = line.find(tag);
        if (at == std::string::npos || (first_only && !slot.empty())) { return; }
        slot = line.substr(at + std::char_traits<char>::length(tag));
    };
    take(" context | ", info.context, false);
    take(" speculation | ", info.speculation, true);
    static const std::regex download(R"((\S+): (\d{1,3})% \((.+ of [^)]+)\)(.*)$)");
    std::smatch match;
    if (std::regex_search(line, match, download)) {
        info.progress_file = match[1].str();
        info.percent       = std::stoi(match[2].str());
        info.progress      = match[2].str() + "% (" + match[3].str() + ")" + match[4].str();
    }
    for (const char* tag : {"cached artifact ", "converted artifact "}) {
        if (const auto at = line.find(tag); at != std::string::npos) {
            const auto begin = at + std::char_traits<char>::length(tag);
            const auto end   = line.find(" (", begin);
            info.artifact    = line.substr(begin, end == std::string::npos ? end : end - begin);
            info.converted   = std::string_view(tag) == "converted artifact ";
        }
    }
}

} // namespace

nlohmann::json ProcessInfo::to_json() const {
    return {{"id", id},
            {"kind", spec.kind},
            {"model", spec.model},
            {"gpu", spec.gpu},
            {"port", spec.port},
            {"pid", pid},
            {"state", state},
            {"exit_code", exit_code},
            {"started", iso(started)},
            {"ended", iso(ended)},
            {"last_line", last_line},
            {"context", context},
            {"speculation", speculation},
            {"artifact", artifact},
            {"percent", percent},
            {"progress", progress},
            {"progress_file", progress_file},
            {"command", [&] {
                 std::string text;
                 for (const auto& arg : spec.argv) { text += (text.empty() ? "" : " ") + arg; }
                 return text;
             }()}};
}

ProcessPool::ProcessPool(std::size_t log_lines) : log_lines_(log_lines) {}

ProcessPool::~ProcessPool() {
    std::vector<std::uint64_t> ids;
    {
        std::lock_guard lock(mutex_);
        for (const auto& [id, child] : children_) { ids.push_back(id); }
    }
    for (const auto id : ids) { stop(id, std::chrono::seconds(5)); }
    std::lock_guard lock(mutex_);
    for (auto& [id, child] : children_) {
        if (child->reader.joinable()) { child->reader.join(); }
    }
}

#ifdef _WIN32
ProcessPool::Child::~Child() {
    if (process) { CloseHandle(static_cast<HANDLE>(process)); }
    if (job) { CloseHandle(static_cast<HANDLE>(job)); }
}

std::uint64_t ProcessPool::start(LaunchSpec spec) {
    if (spec.argv.empty()) { throw std::runtime_error("empty command"); }
    std::wstring command;
    for (const auto& arg : spec.argv) {
        command +=
            (command.empty() ? L"" : L" ") + windows_quoted_argument(std::wstring_view(wide(arg)));
    }
    // The environment block: the UI's own with spec.env on top.
    std::map<std::wstring, std::wstring> variables;
    if (wchar_t* block = GetEnvironmentStringsW()) {
        for (const wchar_t* entry = block; *entry; entry += wcslen(entry) + 1) {
            const std::wstring text(entry);
            const auto eq = text.find(L'=', 1); // "=C:=C:\..." entries start with '='
            if (eq != std::wstring::npos) { variables[text.substr(0, eq)] = text.substr(eq + 1); }
        }
        FreeEnvironmentStringsW(block);
    }
    for (const auto& [key, value] : spec.env) { variables[wide(key)] = wide(value); }
    std::wstring environment;
    for (const auto& [key, value] : variables) { environment += key + L"=" + value + L'\0'; }
    environment += L'\0';

    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    HANDLE read = nullptr, write = nullptr;
    if (!CreatePipe(&read, &write, &inherit, 0)) { throw std::runtime_error("CreatePipe failed"); }
    SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING, 0, nullptr);
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb         = sizeof(startup);
    startup.StartupInfo.dwFlags    = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput  = nul;
    startup.StartupInfo.hStdOutput = write;
    startup.StartupInfo.hStdError  = write;
    // The child inherits its own standard handles only: with every inheritable handle it also
    // took another child's pipe end (whose output then never reaches end of file) and the page's
    // listening socket (which then outlives ninferno-ui).
    std::vector<HANDLE> inherited{write};
    if (nul != INVALID_HANDLE_VALUE) { inherited.push_back(nul); }
    SIZE_T list_bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &list_bytes);
    std::vector<unsigned char> list_storage(list_bytes);
    auto* attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(list_storage.data());
    const bool listed =
        InitializeProcThreadAttributeList(attributes, 1, 0, &list_bytes) &&
        UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited.data(),
                                  inherited.size() * sizeof(HANDLE), nullptr, nullptr);
    if (listed) { startup.lpAttributeList = attributes; }
    PROCESS_INFORMATION info{};
    // Its own console process group (Ctrl+Break reaches it alone), suspended until it is in its job.
    const DWORD flags = CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT |
                        (listed ? EXTENDED_STARTUPINFO_PRESENT : 0);
    const BOOL created = CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, flags,
                                        environment.data(), nullptr, &startup.StartupInfo, &info);
    const DWORD error = GetLastError();
    if (listed) { DeleteProcThreadAttributeList(attributes); }
    CloseHandle(write);
    if (nul != INVALID_HANDLE_VALUE) { CloseHandle(nul); }
    if (!created) {
        CloseHandle(read);
        throw std::runtime_error("cannot start " + spec.argv[0] + " (error " + std::to_string(error) + ")");
    }
    // The job ends the child's whole tree with it, and with ninferno-ui (its last handle closes).
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
    AssignProcessToJobObject(job, info.hProcess);
    ResumeThread(info.hThread);
    CloseHandle(info.hThread);

    auto child          = std::make_unique<Child>();
    child->info.spec    = std::move(spec);
    child->info.pid     = static_cast<int>(info.dwProcessId);
    child->info.state   = child->info.spec.kind == "serve" ? "starting" : "running";
    child->info.started = std::chrono::system_clock::now();
    for (const auto& note : child->info.spec.notes) {
        child->lines.push_back("ninferno-ui | " + note);
    }
    child->process      = info.hProcess;
    child->job          = job;
    child->output       = read;
    Child* raw          = child.get();
    std::uint64_t id    = 0;
    {
        std::lock_guard lock(mutex_);
        id            = next_id_++;
        raw->info.id  = id;
        children_[id] = std::move(child);
    }
    raw->reader = std::thread([this, raw] { read_loop(raw); });
    return id;
}

void ProcessPool::read_loop(Child* child) {
    char buffer[4096];
    DWORD n = 0;
    while (ReadFile(static_cast<HANDLE>(child->output), buffer, sizeof(buffer), &n, nullptr) && n > 0) {
        std::lock_guard lock(mutex_);
        child->partial.append(buffer, n);
        std::size_t start = 0;
        for (std::size_t nl; (nl = child->partial.find('\n', start)) != std::string::npos;) {
            std::string line = child->partial.substr(start, nl - start);
            if (!line.empty() && line.back() == '\r') { line.pop_back(); }
            note_fact(child->info, line);
            child->info.last_line = line;
            child->lines.push_back(std::move(line));
            if (child->lines.size() > log_lines_) { child->lines.pop_front(); }
            start = nl + 1;
        }
        child->partial.erase(0, start);
    }
    CloseHandle(static_cast<HANDLE>(child->output));
    WaitForSingleObject(static_cast<HANDLE>(child->process), INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(static_cast<HANDLE>(child->process), &code);
    std::lock_guard lock(mutex_);
    if (!child->partial.empty()) {
        child->lines.push_back(child->partial);
        child->info.last_line = child->partial;
        child->partial.clear();
    }
    child->info.ended     = std::chrono::system_clock::now();
    child->info.exit_code = static_cast<int>(code);
    const bool asked      = child->info.state == "stopping";
    child->info.state     = (child->info.exit_code == 0 || asked) ? "exited" : "failed";
}

bool ProcessPool::stop(std::uint64_t id, std::chrono::milliseconds grace) {
    int pid    = 0;
    void* job  = nullptr;
    {
        std::lock_guard lock(mutex_);
        const auto it = children_.find(id);
        if (it == children_.end()) { return false; }
        auto& info = it->second->info;
        if (info.state == "exited" || info.state == "failed") { return false; }
        info.state = "stopping";
        pid        = info.pid;
        job        = it->second->job;
    }
    // Ctrl+Break to its process group: ninferno-serve drains as on SIGTERM. Without a console to
    // share (the UI started detached) it cannot arrive, and the grace period runs out instead.
    GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, static_cast<DWORD>(pid));
    const auto deadline = std::chrono::steady_clock::now() + grace;
    for (;;) {
        {
            std::lock_guard lock(mutex_);
            const auto& info = children_.at(id)->info;
            if (info.state == "exited" || info.state == "failed") { return true; }
        }
        if (std::chrono::steady_clock::now() >= deadline) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    TerminateJobObject(static_cast<HANDLE>(job), 1);
    return true;
}
#else
std::uint64_t ProcessPool::start(LaunchSpec spec) {
    if (spec.argv.empty()) { throw std::runtime_error("empty command"); }
    int pipe_fds[2];
    if (::pipe2(pipe_fds, O_CLOEXEC) != 0) {
        throw std::runtime_error(std::string("pipe: ") + std::strerror(errno));
    }
    // Build argv and envp before fork: the child only calls async-signal-safe functions.
    std::vector<std::string> env_storage;
    std::set<std::string> overridden;
    for (const auto& [key, value] : spec.env) {
        env_storage.push_back(key + "=" + value);
        overridden.insert(key);
    }
    for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
        const std::string entry(*e);
        const auto eq = entry.find('=');
        if (eq != std::string::npos && overridden.count(entry.substr(0, eq))) { continue; }
        env_storage.push_back(entry);
    }
    std::vector<char*> argv, envp;
    for (auto& arg : spec.argv) { argv.push_back(arg.data()); }
    argv.push_back(nullptr);
    for (auto& entry : env_storage) { envp.push_back(entry.data()); }
    envp.push_back(nullptr);

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipe_fds[0]);
        ::close(pipe_fds[1]);
        throw std::runtime_error(std::string("fork: ") + std::strerror(errno));
    }
    if (pid == 0) {
        ::setpgid(0, 0);
        ::dup2(pipe_fds[1], STDOUT_FILENO);
        ::dup2(pipe_fds[1], STDERR_FILENO);
        const int devnull = ::open("/dev/null", O_RDONLY);
        if (devnull >= 0) { ::dup2(devnull, STDIN_FILENO); }
        ::execve(argv[0], argv.data(), envp.data());
        const char message[]          = "ninferno-ui: exec failed\n";
        [[maybe_unused]] auto ignored = ::write(STDERR_FILENO, message, sizeof(message) - 1);
        ::_exit(127);
    }
    ::setpgid(pid, pid); // also from the parent, so a stop right after start finds the group
    ::close(pipe_fds[1]);

    auto child          = std::make_unique<Child>();
    child->info.spec    = std::move(spec);
    child->info.pid     = pid;
    child->info.state   = child->info.spec.kind == "serve" ? "starting" : "running";
    child->info.started = std::chrono::system_clock::now();
    for (const auto& note : child->info.spec.notes) {
        child->lines.push_back("ninferno-ui | " + note);
    }
    child->out_fd       = pipe_fds[0];
    Child* raw          = child.get();
    std::uint64_t id    = 0;
    {
        std::lock_guard lock(mutex_);
        id            = next_id_++;
        raw->info.id  = id;
        children_[id] = std::move(child);
    }
    raw->reader = std::thread([this, raw] { read_loop(raw); });
    return id;
}

void ProcessPool::read_loop(Child* child) {
    char buffer[4096];
    for (;;) {
        const ssize_t n = ::read(child->out_fd, buffer, sizeof(buffer));
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) { break; }
        std::lock_guard lock(mutex_);
        child->partial.append(buffer, static_cast<std::size_t>(n));
        std::size_t start = 0;
        for (std::size_t nl; (nl = child->partial.find('\n', start)) != std::string::npos;) {
            std::string line = child->partial.substr(start, nl - start);
            if (!line.empty() && line.back() == '\r') { line.pop_back(); }
            note_fact(child->info, line);
            child->info.last_line = line;
            child->lines.push_back(std::move(line));
            if (child->lines.size() > log_lines_) { child->lines.pop_front(); }
            start = nl + 1;
        }
        child->partial.erase(0, start);
    }
    ::close(child->out_fd);
    int status = 0;
    while (::waitpid(child->info.pid, &status, 0) < 0 && errno == EINTR) {}
    std::lock_guard lock(mutex_);
    if (!child->partial.empty()) {
        child->lines.push_back(child->partial);
        child->info.last_line = child->partial;
        child->partial.clear();
    }
    child->info.ended = std::chrono::system_clock::now();
    if (WIFEXITED(status)) {
        child->info.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        child->info.exit_code = 128 + WTERMSIG(status);
    }
    const bool asked  = child->info.state == "stopping";
    child->info.state = (child->info.exit_code == 0 || asked) ? "exited" : "failed";
}

bool ProcessPool::stop(std::uint64_t id, std::chrono::milliseconds grace) {
    pid_t pid = 0;
    {
        std::lock_guard lock(mutex_);
        const auto it = children_.find(id);
        if (it == children_.end()) { return false; }
        auto& info = it->second->info;
        if (info.state == "exited" || info.state == "failed") { return false; }
        info.state = "stopping";
        pid        = info.pid;
    }
    ::kill(-pid, SIGTERM);
    const auto deadline = std::chrono::steady_clock::now() + grace;
    for (;;) {
        {
            std::lock_guard lock(mutex_);
            const auto& info = children_.at(id)->info;
            if (info.state == "exited" || info.state == "failed") { return true; }
        }
        if (std::chrono::steady_clock::now() >= deadline) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ::kill(-pid, SIGKILL);
    return true;
}
#endif

void ProcessPool::mark_ready(std::uint64_t id) {
    std::lock_guard lock(mutex_);
    if (const auto it = children_.find(id); it != children_.end()) {
        if (it->second->info.state == "starting") { it->second->info.state = "ready"; }
    }
}

std::vector<ProcessInfo> ProcessPool::list() const {
    std::lock_guard lock(mutex_);
    std::vector<ProcessInfo> out;
    for (const auto& [id, child] : children_) { out.push_back(child->info); }
    return out;
}

bool ProcessPool::find(std::uint64_t id, ProcessInfo& out) const {
    std::lock_guard lock(mutex_);
    const auto it = children_.find(id);
    if (it == children_.end()) { return false; }
    out = it->second->info;
    return true;
}

std::vector<std::string> ProcessPool::log(std::uint64_t id, std::size_t max) const {
    std::lock_guard lock(mutex_);
    const auto it = children_.find(id);
    if (it == children_.end()) { return {}; }
    const auto& lines      = it->second->lines;
    const std::size_t skip = (max == 0 || lines.size() <= max) ? 0 : lines.size() - max;
    return {lines.begin() + static_cast<std::ptrdiff_t>(skip), lines.end()};
}

void ProcessPool::forget_ended(std::chrono::seconds age) {
    const auto now = std::chrono::system_clock::now();
    std::vector<std::unique_ptr<Child>> dropped;
    {
        std::lock_guard lock(mutex_);
        for (auto it = children_.begin(); it != children_.end();) {
            const auto& info = it->second->info;
            const bool ended = info.state == "exited" || info.state == "failed";
            if (ended && now - info.ended > age) {
                dropped.push_back(std::move(it->second));
                it = children_.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (auto& child : dropped) {
        if (child->reader.joinable()) { child->reader.join(); }
    }
}

int ProcessPool::free_port(int first) const {
    std::set<int> taken;
    {
        std::lock_guard lock(mutex_);
        for (const auto& [id, child] : children_) {
            const auto& info = child->info;
            if (info.state != "exited" && info.state != "failed") { taken.insert(info.spec.port); }
        }
    }
    for (int port = first; port < first + 1000; ++port) {
        if (!taken.count(port) && port_free(port)) { return port; }
    }
    throw std::runtime_error("no free port from " + std::to_string(first));
}

} // namespace ninferno::ui
