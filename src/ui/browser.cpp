#include "ui/browser.h"

#include "core/sys.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <shellapi.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>

#ifndef _WIN32
extern char** environ;
#endif

namespace ninferno::ui {
namespace {

constexpr const char* kWindowsCmd = "/mnt/c/Windows/System32/cmd.exe";

#ifdef _WIN32
bool loopback_answers(int port) {
    static const bool started = [] {
        WSADATA data;
        return WSAStartup(MAKEWORD(2, 2), &data) == 0;
    }();
    if (!started) { return false; }
    const SOCKET fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd == INVALID_SOCKET) { return false; }
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(static_cast<std::uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const bool ok = ::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0;
    ::closesocket(fd);
    return ok;
}
#else
bool loopback_answers(int port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) { return false; }
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(static_cast<std::uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const bool ok = ::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0;
    ::close(fd);
    return ok;
}

// Runs `argv` detached from the terminal (its output would land in the server's log) and reaps it.
void run_quietly(const std::vector<std::string>& argv) {
    std::vector<char*> args;
    for (const auto& a : argv) { args.push_back(const_cast<char*>(a.c_str())); }
    args.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t pid = 0;
    if (posix_spawnp(&pid, args[0], &actions, nullptr, args.data(), environ) == 0) {
        int status = 0;
        (void)::waitpid(pid, &status, 0);
    }
    posix_spawn_file_actions_destroy(&actions);
}
#endif

} // namespace

std::vector<std::string> browser_command(const BrowserHost& host, const std::string& url) {
    if (host.wsl) {
        if (host.has_program("wslview")) { return {"wslview", url}; }
        // start's first quoted argument is the window title; the URL holds no shell characters.
        if (host.has_program(kWindowsCmd)) { return {kWindowsCmd, "/c", "start", "", url}; }
    }
    if ((!host.env("DISPLAY").empty() || !host.env("WAYLAND_DISPLAY").empty()) &&
        host.has_program("xdg-open")) {
        return {"xdg-open", url};
    }
    return {};
}

#ifdef _WIN32
BrowserHost current_browser_host() {
    BrowserHost host; // Windows opens the default browser itself (open_browser_when_listening)
    host.env         = [](const char*) { return std::string(); };
    host.has_program = [](const std::string&) { return false; };
    return host;
}

bool open_browser_when_listening(const std::string& url, int port) {
    std::thread([url, port] {
        for (int i = 0; i < 300 && !loopback_answers(port); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    }).detach();
    return true;
}
#else
BrowserHost current_browser_host() {
    BrowserHost host;
    host.env         = [](const char* name) { return sys::env(name); };
    host.has_program = [](const std::string& name) {
        if (name.find('/') != std::string::npos) { return ::access(name.c_str(), X_OK) == 0; }
        const char* path = std::getenv("PATH");
        std::stringstream dirs(path != nullptr ? path : "");
        for (std::string dir; std::getline(dirs, dir, ':');) {
            if (!dir.empty() && ::access((dir + "/" + name).c_str(), X_OK) == 0) { return true; }
        }
        return false;
    };
    std::ifstream version("/proc/version");
    std::string text;
    std::getline(version, text);
    host.wsl =
        text.find("microsoft") != std::string::npos || text.find("Microsoft") != std::string::npos;
    return host;
}

bool open_browser_when_listening(const std::string& url, int port) {
    const auto command = browser_command(current_browser_host(), url);
    if (command.empty()) { return false; }
    std::thread([command, port] {
        for (int i = 0; i < 300 && !loopback_answers(port); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        run_quietly(command);
    }).detach();
    return true;
}
#endif

} // namespace ninferno::ui
