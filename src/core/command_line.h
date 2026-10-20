#pragma once

// Command-line helpers the binaries share.
//
// windows_quoted_argument: one argument of a Windows command line (CreateProcess, _spawnv), quoted
// so that the C runtime and CommandLineToArgvW split it back unchanged: a run of backslashes
// doubles only where a quote follows it (an escaped quote of the argument's own, or the closing
// one), and stays as it is elsewhere. Portable, so its tests run on every platform.

#include <string>
#include <string_view>

namespace ninferno {

// --help or -h anywhere among the arguments, as every binary answers it.
[[nodiscard]] inline bool help_requested(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") { return true; }
    }
    return false;
}

// --version as the first argument: every binary prints its version and exits.
[[nodiscard]] inline bool version_requested(int argc, char** argv) {
    return argc >= 2 && std::string_view(argv[1]) == "--version";
}

template <class Char>
[[nodiscard]] std::basic_string<Char> windows_quoted_argument(std::basic_string_view<Char> arg) {
    const Char special[] = {Char(' '), Char('\t'), Char('"'), Char(0)};
    if (!arg.empty() && arg.find_first_of(special) == std::basic_string_view<Char>::npos) {
        return std::basic_string<Char>(arg);
    }
    std::basic_string<Char> out(1, Char('"'));
    std::size_t backslashes = 0;
    for (const Char c : arg) {
        if (c == Char('\\')) {
            ++backslashes;
            continue;
        }
        out.append(c == Char('"') ? backslashes * 2 + 1 : backslashes, Char('\\'));
        out += c;
        backslashes = 0;
    }
    out.append(backslashes * 2, Char('\\'));
    out += Char('"');
    return out;
}

} // namespace ninferno
