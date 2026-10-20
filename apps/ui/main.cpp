// ninferno-ui: a local web page to choose, download and run models on this machine's GPUs.
//
//   ninferno-ui [--listen ADDR] [--port N] [--access-key-file F] [--serve PATH] [--hub-cache DIR]
//               [--offline] [--no-browser]
//
// It listens on 127.0.0.1:8090 by default. Whoever reaches it can start processes and spend the
// GPUs, so another address is only taken from --listen, and then other machines need the access
// key (ui/access.h): --access-key-file, else $NINFERNO_UI_KEY, else one made once and kept in
// <hub cache>/ninferno-ui/access-key; the log prints the URL with it. Once it listens, the page
// opens in the desktop's browser (on WSL, Windows' default browser) unless --no-browser; with no
// display the URL is only printed.

#include "convert/fetch.h"
#include "core/command_line.h"
#include "product/numeric_options.h"
#include "ui/app.h"
#include "ui/browser.h"
#include "ui/chat_ui.h"

#include "core/sys.h"

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <cerrno>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

ninferno::ui::App* g_app = nullptr;

void on_signal(int) {
    if (g_app != nullptr) { g_app->stop(); }
}

const char* kUsage =
    "usage: ninferno-ui --version\n"
    "       ninferno-ui [--listen ADDR] [--port N] [--access-key-file F] [--serve PATH]\n"
    "                 [--hub-cache DIR]\n"
    "                 [--models-dir DIR]... [--offline] [--chat-ui DIR]\n"
    "                 [--keep-originals] [--catalog-url URL] [--no-browser]\n"
    "  --listen ADDR   address to bind (default 127.0.0.1). On another address, other machines\n"
    "                  need the access key (plain HTTP: a trusted network, or TLS in front)\n"
    "  --port N        port (default 8090)\n"
    "  --access-key-file F  the access key, F's first line, 16+ characters (default:\n"
    "                  $NINFERNO_UI_KEY, else made once into <hub cache>/ninferno-ui/access-key)\n"
    "  --serve PATH    the ninferno-serve binary (default: beside ninferno-ui)\n"
    "  --hub-cache DIR Hugging Face cache for downloads (default: $HF_HUB_CACHE, $HF_HOME/hub,\n"
    "                  $XDG_CACHE_HOME/huggingface/hub, ~/.cache/huggingface/hub)\n"
    "  --models-dir DIR  also list the .ninferno artifacts in DIR (two levels deep); repeatable\n"
    "  --offline       no hub listings (sizes come from estimates)\n"
    "  --no-browser    do not open the page in the desktop's browser once it listens\n"
    "  --keep-originals  keep a hub model's download after its conversion (default: removed\n"
    "                  once the converted model loaded; nothing reads it again)\n"
    "  --catalog-url URL  the recommended models from a catalog.json at URL (your server, a raw\n"
    "                  GitHub file; the built-in one's schema), read at start and every 6 hours\n"
    "                  and kept in the hub cache for when it is unreachable (default:\n"
    "                  $NINFERNO_CATALOG_URL, else the built-in catalog)\n"
    "  --chat-ui DIR   llama-ui's bundle for the chat window (default: beside ninferno-ui, else "
    "the\n"
    "                  build's copy; without one the page has no chat window)\n";

// A program of the installation (ninferno-serve, ...) beside this one; ".exe" on Windows.
std::filesystem::path beside_self(const char* name) {
#ifdef _WIN32
    const std::string file = std::string(name) + (std::string_view(name).find('.') == std::string_view::npos ? ".exe" : "");
#else
    const std::string file = name;
#endif
    const auto self = ninferno::sys::self_exe();
    return self.empty() ? std::filesystem::path(file) : self.parent_path() / file;
}

bool executable(const std::filesystem::path& path) {
#ifdef _WIN32
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
#else
    return ::access(path.c_str(), X_OK) == 0;
#endif
}

#ifdef _WIN32
// _spawnv joins its arguments with spaces and quotes nothing, so each is quoted here
// (core/command_line.h).
#endif

// After an update: the installed version in place of this process, with the same arguments and
// no second browser window (the open page reconnects by itself). A release install lives in
// <home>/ninferno-<version>-.../bin with <home>/current pointing at the newest one.
[[noreturn]] void restart_installed(int argc, char** argv) {
    namespace fs = std::filesystem;
#ifdef _WIN32
    constexpr const char* kProgram = "ninferno-ui.exe";
#else
    constexpr const char* kProgram = "ninferno-ui";
#endif
    auto next = beside_self(kProgram);
    std::error_code ec;
    if (const auto current = next.parent_path().parent_path().parent_path() / "current" / "bin" /
                             kProgram;
        fs::exists(current, ec)) {
        next = current;
    }
    std::vector<std::string> args{next.string()};
    bool quiet = false;
    for (int i = 1; i < argc; ++i) {
        args.emplace_back(argv[i]);
        quiet = quiet || args.back() == "--no-browser";
    }
    if (!quiet) { args.emplace_back("--no-browser"); }
#ifdef _WIN32
    for (auto& arg : args) { arg = ninferno::windows_quoted_argument(std::string_view(arg)); }
#endif
    std::vector<char*> raw;
    for (auto& arg : args) { raw.push_back(arg.data()); }
    raw.push_back(nullptr);
    std::cout << "ninferno-ui: restarting into " << next << std::endl;
#ifdef _WIN32
    // No exec on Windows: the new process starts, this one leaves.
    const std::string program = next.string(); // the path itself, not its quoted form
    if (_spawnv(_P_NOWAIT, program.c_str(), raw.data()) != -1) { std::exit(0); }
#else
    ::execv(raw[0], raw.data());
#endif
    std::cerr << "ninferno-ui: cannot start " << next << ": " << std::strerror(errno)
              << "; start ninferno-ui again\n";
    std::exit(1);
}

// The first line of `path` without surrounding blanks; empty when unreadable.
std::string first_line(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    const auto begin = line.find_first_not_of(" \t\r");
    if (begin == std::string::npos) { return {}; }
    return line.substr(begin, line.find_last_not_of(" \t\r") - begin + 1);
}

// The key kept in `path`, made there (readable by its owner only) when there is none yet.
std::string kept_access_key(const std::filesystem::path& path) {
    namespace fs = std::filesystem;
    if (auto key = first_line(path); key.size() >= 16) { return key; }
    std::string key = ninferno::ui::new_access_key();
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::ofstream(path, std::ios::trunc) << key << '\n';
    fs::permissions(path, fs::perms::owner_read | fs::perms::owner_write, ec);
    return key;
}

} // namespace

int main(int argc, char** argv) {
    if (ninferno::version_requested(argc, argv)) {
        std::cout << "NInferno " << NINFERNO_VERSION_STRING << '\n';
        return 0;
    }
    ninferno::ui::UiOptions options;
    bool open_browser = true;
    std::filesystem::path key_file;
    if (const char* url = std::getenv("NINFERNO_CATALOG_URL"); url != nullptr && *url != '\0') {
        options.catalog_url = url;
    }
    options.serve_binary = beside_self("ninferno-serve");
    options.version      = NINFERNO_VERSION_STRING;
    // The installer's updater: NInferno-Setup.exe on Windows hands its own command line over
    // NINFERNO_UPDATER; a Linux install carries ninferno-update beside the binaries.
    if (const char* updater = std::getenv("NINFERNO_UPDATER"); updater != nullptr && *updater) {
        options.updater = {updater, "--update"};
    } else if (const auto script = beside_self("ninferno-update"); executable(script)) {
        options.updater = {script.string()};
    }
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value      = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "ninferno-ui: " << arg << " needs a value\n" << kUsage;
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--listen") {
            options.access.listen = value();
        } else if (arg == "--port") {
            const std::string text = value();
            try {
                options.access.port =
                    ninferno::product::parse_nonnegative_int(text.c_str(), "port");
            } catch (const std::invalid_argument& error) {
                std::cerr << "ninferno-ui: " << error.what() << '\n';
                return 2;
            }
            if (options.access.port <= 0 || options.access.port > 65535) {
                std::cerr << "ninferno-ui: --port must be in [1,65535]\n";
                return 2;
            }
        } else if (arg == "--access-key-file") {
            key_file = value();
        } else if (arg == "--serve") {
            options.serve_binary = value();
        } else if (arg == "--hub-cache") {
            options.hub_cache = value();
        } else if (arg == "--models-dir") {
            options.model_dirs.emplace_back(value());
        } else if (arg == "--offline") {
            options.offline = true;
        } else if (arg == "--no-browser") {
            open_browser = false;
        } else if (arg == "--keep-originals") {
            options.keep_originals = true;
        } else if (arg == "--catalog-url") {
            options.catalog_url = value();
            if (!options.catalog_url.starts_with("https://") &&
                !options.catalog_url.starts_with("http://")) {
                std::cerr << "ninferno-ui: --catalog-url takes an http(s) URL\n";
                return 2;
            }
        } else if (arg == "--chat-ui") {
            options.chat_ui = value();
        } else if (arg == "-h" || arg == "--help") {
            std::cout << kUsage;
            return 0;
        } else {
            std::cerr << "ninferno-ui: unknown argument " << arg << "\n" << kUsage;
            return 2;
        }
    }
    try {
        const auto found =
            ninferno::ui::find_chat_ui(options.chat_ui, beside_self("x").parent_path());
        options.chat_ui = found.value_or(std::filesystem::path());
    } catch (const std::exception& error) {
        std::cerr << "ninferno-ui: " << error.what() << '\n';
        return 1;
    }
    if (!executable(options.serve_binary)) {
        std::cerr << "ninferno-ui: warning: " << options.serve_binary
                  << " is not executable; models cannot be started (--serve PATH)\n";
    }
    const bool exposed =
        !ninferno::ui::is_loopback_host(ninferno::ui::host_of_authority(options.access.listen));
    if (!key_file.empty()) {
        options.access.key = first_line(key_file);
        if (options.access.key.size() < 16) {
            std::cerr << "ninferno-ui: " << key_file << ": needs a key of 16+ characters\n";
            return 2;
        }
    } else if (const char* key = std::getenv("NINFERNO_UI_KEY"); key != nullptr && *key) {
        options.access.key = key;
    } else if (exposed) {
        const auto hub = options.hub_cache.empty() ? ninferno::convert::default_hub_cache()
                                                   : options.hub_cache;
        options.access.key = kept_access_key(hub / "ninferno-ui" / "access-key");
    }
    if (exposed && options.access.key.empty()) {
        std::cerr << "ninferno-ui: no access key for " << options.access.listen
                  << " (--access-key-file F)\n";
        return 1;
    }
    ninferno::ui::App app(options, ninferno::ui::make_nvml_source());
    g_app = &app;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
#ifndef _WIN32
    std::signal(SIGPIPE, SIG_IGN);
#endif
    app.prefetch_listings();
    const std::string host = options.access.listen.find(':') != std::string::npos
                                 ? "[" + options.access.listen + "]"
                                 : options.access.listen;
    const std::string url  = "http://" + (exposed ? host : std::string("localhost")) + ":" +
                            std::to_string(options.access.port) + "/";
    std::cout << "ninferno-ui: " << url << std::endl;
    if (open_browser &&
        !ninferno::ui::open_browser_when_listening(
            "http://localhost:" + std::to_string(options.access.port) + "/", options.access.port)) {
        std::cout << "ninferno-ui: no desktop browser here; open " << url << " in yours"
                  << std::endl;
    }
    if (exposed) {
        std::cout << "ninferno-ui: from other machines: " << url << "?key=" << options.access.key
                  << " (plain HTTP: a trusted network, or TLS in front)" << std::endl;
    }
    if (!app.run()) {
        std::cerr << "ninferno-ui: cannot listen on " << options.access.listen << ":"
                  << options.access.port << " (in use?)\n";
        return 1;
    }
    if (app.restart_requested()) { restart_installed(argc, argv); }
    return 0;
}
