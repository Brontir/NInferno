// The universal entry point of a release: installed as bin/ninferno-serve (and under the name of
// every other GPU program), it starts the engine built for the GPU the command will run on.
//
// A release carries one engine per GPU generation beside it, bin/<name>-sm<tag>, and
// share/ninferno/engines.json saying where each runs:
//
//   {"engines": [{"tag": "120a", "arch": 120, "specific": true,  "ptx": false},
//                {"tag": "89",   "arch": 89,  "specific": false, "ptx": true}, ...]}
//
// An engine runs on a GPU of compute capability cc when its target is architecture-specific and
// equal to cc (120a: 12.0 only), or plain with the same major version and arch <= cc (86: 8.6,
// 8.7, 8.9), or it carries PTX and arch <= cc (89 with PTX: Hopper, datacenter Blackwell, any later
// generation, the driver compiling each kernel once). The most specific, newest match wins.
//
// The GPU is the one the command names (--device N, pci:BUSID or GPU-UUID; without it the first),
// found as the engine finds it: among the GPUs the CUDA driver enumerates, so CUDA_VISIBLE_DEVICES
// agrees, by PCI bus order and identity, never the enumeration order (core/device_identity.h).
// Every argument goes to the engine unchanged. On Linux the engine replaces this process; on
// Windows it runs as a child in this console (Ctrl+C and Ctrl+Break reach it) and its exit code is
// returned.

#include "core/command_line.h"
#include "core/device_identity.h"
#include "core/sys.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifndef NINFERNO_VERSION_STRING
#define NINFERNO_VERSION_STRING "dev"
#endif

namespace {

namespace fs = std::filesystem;

struct Engine {
    std::string tag; // "120a", "89", ...
    int arch      = 0;
    bool specific = false;
    bool ptx      = false;
};

struct Gpu {
    std::string name;
    int cc = 0; // major * 10 + minor; 0 when the driver cannot report the device
    ninferno::DeviceIdentity identity;
};

// The CUDA driver's entry points, loaded at run time (the dispatcher links no CUDA library).
struct Driver {
    using Init      = int (*)(unsigned);
    using Count     = int (*)(int*);
    using Get       = int (*)(int*, int);
    using Attribute = int (*)(int*, int, int);
    using Name      = int (*)(char*, int, int);
    using BusId     = int (*)(char*, int, int);
    using Uuid      = int (*)(unsigned char*, int); // CUuuid: 16 bytes
    Init init           = nullptr;
    Count count         = nullptr;
    Get get             = nullptr;
    Attribute attribute = nullptr;
    Name name           = nullptr;
    BusId bus_id        = nullptr;
    Uuid uuid           = nullptr;

    bool load() {
        void* library = ninferno::sys::load_library("libcuda.so.1", "nvcuda.dll");
        if (library == nullptr) { return false; }
        init      = reinterpret_cast<Init>(ninferno::sys::symbol(library, "cuInit"));
        count     = reinterpret_cast<Count>(ninferno::sys::symbol(library, "cuDeviceGetCount"));
        get       = reinterpret_cast<Get>(ninferno::sys::symbol(library, "cuDeviceGet"));
        attribute = reinterpret_cast<Attribute>(ninferno::sys::symbol(library, "cuDeviceGetAttribute"));
        name      = reinterpret_cast<Name>(ninferno::sys::symbol(library, "cuDeviceGetName"));
        bus_id    = reinterpret_cast<BusId>(ninferno::sys::symbol(library, "cuDeviceGetPCIBusId"));
        uuid      = reinterpret_cast<Uuid>(ninferno::sys::symbol(library, "cuDeviceGetUuid_v2"));
        return init && count && get && attribute && name && bus_id && uuid && init(0) == 0;
    }

    // Entry i is ordinal i. A device the driver cannot report keeps its place, with cc 0 and an
    // identity sorted last.
    std::vector<Gpu> gpus() const {
        constexpr int kMajor = 75; // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        constexpr int kMinor = 76; // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        int n = 0;
        std::vector<Gpu> out;
        if (count(&n) != 0) { return out; }
        for (int i = 0; i < n; ++i) {
            Gpu gpu;
            gpu.identity.pci_domain = 0xFFFFFFFFu;
            int device = 0, major = 0, minor = 0;
            char text[256] = {};
            char bus[32]   = {};
            if (get(&device, i) == 0 && attribute(&major, kMajor, device) == 0 &&
                attribute(&minor, kMinor, device) == 0) {
                (void)name(text, sizeof(text), device);
                gpu.name = text;
                gpu.cc   = major * 10 + minor;
                if (bus_id(bus, sizeof(bus), device) == 0) {
                    if (const auto identity = ninferno::parse_pci_bus_id(bus)) {
                        gpu.identity = *identity;
                        (void)uuid(gpu.identity.uuid.data(), device);
                    }
                }
            }
            out.push_back(std::move(gpu));
        }
        return out;
    }
};

std::string exe_suffix() {
#ifdef _WIN32
    return ".exe";
#else
    return "";
#endif
}

// This program's name without directory and .exe: the engine's family ("ninferno-serve").
std::string program_name(const fs::path& self) {
    std::string name = self.filename().string();
    const std::string suffix = exe_suffix();
    if (!suffix.empty() && name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        name.resize(name.size() - suffix.size());
    }
    return name;
}

std::vector<Engine> read_engines(const fs::path& bin, const std::string& program) {
    std::vector<Engine> engines;
    std::ifstream in(bin.parent_path() / "share" / "ninferno" / "engines.json");
    if (in) {
        try {
            const auto json = nlohmann::json::parse(in);
            for (const auto& e : json.at("engines")) {
                engines.push_back({e.at("tag").get<std::string>(), e.at("arch").get<int>(),
                                   e.value("specific", false), e.value("ptx", false)});
            }
        } catch (const std::exception& error) {
            std::cerr << "ninferno: engines.json: " << error.what() << '\n';
        }
    }
    // Only the engines this release carries for this program.
    std::erase_if(engines, [&](const Engine& e) {
        return !fs::exists(bin / (program + "-sm" + e.tag + exe_suffix()));
    });
    return engines;
}

// The engine for a GPU of compute capability cc, if any (see the top of this file).
std::optional<Engine> choose(const std::vector<Engine>& engines, int cc) {
    const auto pick = [&](auto&& runs) -> std::optional<Engine> {
        std::optional<Engine> best;
        for (const auto& e : engines) {
            if (runs(e) && (!best || e.arch > best->arch)) { best = e; }
        }
        return best;
    };
    if (auto e = pick([&](const Engine& e) { return e.specific && e.arch == cc; })) { return e; }
    if (auto e = pick([&](const Engine& e) { return !e.specific && e.arch / 10 == cc / 10 && e.arch <= cc; })) {
        return e;
    }
    return pick([&](const Engine& e) { return e.ptx && e.arch <= cc; });
}

// The ordinal of the GPU --device names among the arguments (the spelling every engine parses),
// as the engine selects it; -1 when it names none or does not parse, which the engine then
// refuses itself (the newest engine runs to say so).
int device_argument(int argc, char** argv, const std::vector<Gpu>& gpus) {
    ninferno::DeviceSelector selector;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string_view(argv[i]) != "--device") { continue; }
        const auto parsed = ninferno::parse_device_selector(argv[i + 1]);
        if (!parsed) { return -1; }
        selector = *parsed;
        break;
    }
    std::vector<ninferno::DeviceIdentity> identities;
    for (const Gpu& gpu : gpus) { identities.push_back(gpu.identity); }
    return ninferno::select_device(selector, identities).value_or(-1);
}

std::string engines_text(const std::vector<Engine>& engines) {
    std::string text;
    for (const auto& e : engines) {
        text += (text.empty() ? "" : ", ") + std::string("sm_") + e.tag + (e.ptx ? " (+PTX)" : "");
    }
    return text.empty() ? "none" : text;
}

int print_version(const std::vector<Engine>& engines, const Driver* driver) {
    std::cout << "NInferno " << NINFERNO_VERSION_STRING << " (engines: " << engines_text(engines) << ")\n";
    if (driver == nullptr) {
        std::cout << "no NVIDIA driver found\n";
        return 1;
    }
    const auto gpus = driver->gpus();
    int runnable    = 0;
    // Numbered as --device counts them: in PCI bus order.
    std::vector<ninferno::DeviceIdentity> identities;
    for (const Gpu& gpu : gpus) { identities.push_back(gpu.identity); }
    const std::vector<int> order = ninferno::canonical_device_order(identities);
    for (std::size_t n = 0; n < order.size(); ++n) {
        const Gpu& gpu = gpus[static_cast<std::size_t>(order[n])];
        if (gpu.cc == 0) { continue; }
        const auto engine = choose(engines, gpu.cc);
        runnable += engine ? 1 : 0;
        std::cout << "GPU " << n << " (" << ninferno::pci_bus_id_text(gpu.identity)
                  << "): " << gpu.name << ", sm_" << gpu.cc << " -> "
                  << (engine ? "engine sm_" + engine->tag
                             : std::string("no engine in this release"))
                  << '\n';
    }
    if (gpus.empty()) { std::cout << "no CUDA GPU found\n"; }
    return runnable > 0 ? 0 : 1;
}

#ifdef _WIN32
// The child gets Ctrl+C / Ctrl+Break itself (same console); this process waits for it to drain.
BOOL WINAPI keep_waiting(DWORD) { return TRUE; }

int run(const fs::path& engine) {
    // This command line as it is: the arguments stay byte for byte, and the program name the user
    // typed (ninferno-serve, not the engine's file) is the one the engine's usage shows.
    std::wstring line = GetCommandLineW();
    SetConsoleCtrlHandler(keep_waiting, TRUE);
    // The engine ends with this process however it ends (a job that kills on close).
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (job) { SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)); }
    STARTUPINFOW si{sizeof(si)};
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(engine.wstring().c_str(), line.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED,
                        nullptr, nullptr, &si, &pi)) {
        std::cerr << "ninferno: cannot start " << engine.string() << " (error " << GetLastError() << ")\n";
        return 127;
    }
    if (job) { AssignProcessToJobObject(job, pi.hProcess); }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    return static_cast<int>(code);
}
#else
int run(const fs::path& engine, char** argv) {
    // argv[0] stays the name the user typed (ninferno-serve), which the engine's usage shows.
    const std::string path = engine.string();
    ::execv(path.c_str(), argv);
    std::cerr << "ninferno: cannot start " << path << ": " << std::strerror(errno) << '\n';
    return 127;
}
#endif

} // namespace

int main(int argc, char** argv) {
    const fs::path self = ninferno::sys::self_exe();
    if (self.empty()) {
        std::cerr << "ninferno: cannot find this program's own path\n";
        return 127;
    }
    const fs::path bin          = self.parent_path();
    const std::string program   = program_name(self);
    const auto engines          = read_engines(bin, program);
    Driver driver;
    const bool have_driver = driver.load();

    if (ninferno::version_requested(argc, argv)) {
        return print_version(engines, have_driver ? &driver : nullptr);
    }
    if (engines.empty()) {
        std::cerr << program << ": this installation carries no engine (" << bin.string() << "/" << program
                  << "-sm*); reinstall NInferno\n";
        return 127;
    }
    // Without a driver or a GPU the newest engine still runs what needs none (--help, --prepare-only)
    // and says itself what is missing.
    const Engine* fallback = &*std::max_element(engines.begin(), engines.end(),
                                                [](const Engine& a, const Engine& b) { return a.arch < b.arch; });
    std::optional<Engine> engine;
    if (have_driver) {
        const auto gpus  = driver.gpus();
        const int device = device_argument(argc, argv, gpus);
        if (device >= 0 && device < static_cast<int>(gpus.size()) &&
            gpus[static_cast<std::size_t>(device)].cc != 0) {
            engine = choose(engines, gpus[static_cast<std::size_t>(device)].cc);
            if (!engine) {
                std::cerr << program << ": GPU " << device << " (" << gpus[static_cast<std::size_t>(device)].name
                          << ", sm_" << gpus[static_cast<std::size_t>(device)].cc
                          << ") is older than this release supports (engines: " << engines_text(engines) << ")\n";
                return 1;
            }
        }
    }
    if (!engine) { engine = *fallback; }
    // A GPU no engine names runs the PTX one: the driver compiles its kernels at the first start
    // (minutes) and keeps them in its cache, whose default 256 MiB an engine outgrows; without
    // room every start compiles again. The user's own setting stays.
    if (engine->ptx && have_driver) {
        const auto gpus  = driver.gpus();
        const int device = device_argument(argc, argv, gpus);
        if (device >= 0 && device < static_cast<int>(gpus.size()) &&
            gpus[static_cast<std::size_t>(device)].cc != 0 &&
            gpus[static_cast<std::size_t>(device)].cc / 10 != engine->arch / 10) {
            std::cerr << program << ": GPU " << device << " (" << gpus[static_cast<std::size_t>(device)].name
                      << ", sm_" << gpus[static_cast<std::size_t>(device)].cc
                      << ") is newer than this release's engines: the driver compiles one for it now (up to "
                         "half an hour at the first start, minutes after that). A newer NInferno release "
                         "usually carries an engine of its own for it.\n";
        }
    }
    if (engine->ptx && std::getenv("CUDA_CACHE_MAXSIZE") == nullptr) {
#ifdef _WIN32
        _putenv_s("CUDA_CACHE_MAXSIZE", "4294967296");
#else
        ::setenv("CUDA_CACHE_MAXSIZE", "4294967296", 1);
#endif
    }
    const fs::path path = bin / (program + "-sm" + engine->tag + exe_suffix());
#ifdef _WIN32
    (void)argv;
    return run(path);
#else
    return run(path, argv);
#endif
}
