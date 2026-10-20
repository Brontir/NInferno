#pragma once

// Opening the page in the desktop's browser once ninferno-ui listens: WSL hands the URL to
// Windows (wslview, else cmd.exe's start), a Linux desktop to xdg-open; a machine with no display
// (a server over SSH) opens nothing and the URL is printed for the user to open.

#include <functional>
#include <string>
#include <vector>

namespace ninferno::ui {

struct BrowserHost {
    std::function<std::string(const char*)> env;         // getenv, "" when unset
    std::function<bool(const std::string&)> has_program; // on PATH (or an absolute path)
    bool wsl = false;                                    // /proc/version names Microsoft
};

// The command that opens `url` on this host; empty when it has no display to open it on.
[[nodiscard]] std::vector<std::string> browser_command(const BrowserHost& host,
                                                       const std::string& url);

// The real host: the process environment, PATH and /proc/version.
[[nodiscard]] BrowserHost current_browser_host();

// Waits (in the background) until `port` on the loopback answers, then runs the command. Returns
// at once; false when there is nothing to run.
bool open_browser_when_listening(const std::string& url, int port);

} // namespace ninferno::ui
