// NInferno-Setup.exe: the Windows installer, launcher, updater and uninstaller of NInferno, one
// small program with no installer framework behind it.
//
//   NInferno-Setup.exe                 the installer's window
//   NInferno.exe (installed copy)      starts NInferno: the page opens in the browser, an icon in
//                                      the notification area (open, check for updates, quit)
//   --install [--quiet] [--relaunch]   install or update without the window (--relaunch starts
//                                      NInferno afterwards; --no-browser: without opening the page)
//   --launch [--no-browser]            start NInferno (--no-browser: the icon only)
//   --update [--yes]                   install the newest release (the page's Update button runs
//                                      this through NINFERNO_UPDATER and shows its lines)
//   --uninstall [--quiet]              remove NInferno, its models and its entries
//
// Updates: the launcher reads the release manifest at start and daily. A newer release is offered
// in a notification; a release that the manifest's min_version makes obsolete asks at every start.
// Updating downloads the new NInferno-Setup.exe (SHA-256 from the manifest), which installs the new
// engine beside the old one, replaces this program and starts NInferno again.

#include "engine.h"
#include "strings.h"

#include <commctrl.h>
#include <shellapi.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

#ifndef NINFERNO_VERSION
#define NINFERNO_VERSION "0.0.0-dev"
#endif

using namespace setup;

namespace {

constexpr const wchar_t* kLauncherClass = L"NInferno.Launcher";
constexpr const wchar_t* kLauncherMutex = L"Local\\NInferno.Launcher";
constexpr const wchar_t* kSetupClass    = L"NInferno.Setup";
constexpr UINT kTrayMessage             = WM_APP + 1;
constexpr UINT kSetupEvent              = WM_APP + 2;
constexpr COLORREF kAccent              = RGB(0x4b, 0x3d, 0xf5);

HINSTANCE g_instance = nullptr;

bool has_arg(int argc, wchar_t** argv, const wchar_t* flag) {
    for (int i = 1; i < argc; ++i) {
        if (_wcsicmp(argv[i], flag) == 0) { return true; }
    }
    return false;
}

std::wstring self_path() {
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    return path;
}

int message(HWND owner, const std::wstring& text, UINT flags) {
    return MessageBoxW(owner, text.c_str(), L"NInferno", flags);
}

// ---- the update's status, for the page: one line per step in update.status ------------------

std::wstring status_path() { return data_dir() + L"\\update.status"; }

void status(const std::string& line) {
    log_line("update: " + line);
    std::ofstream out(std::filesystem::path(status_path()), std::ios::app | std::ios::binary);
    out << line << "\n";
}

// ---- install --------------------------------------------------------------------------------

struct InstallSink {
    std::function<void(int step, int state)> step; // state: 0 waiting, 1 running, 2 done, 3 failed
    Report report;
    std::function<bool()> cancelled;
    HWND owner = nullptr;
};

enum Step { kGpu, kWsl, kDownload, kInstall, kShortcuts, kSteps };

// Downloads a newer installer than this one and runs it in place of this run: what it installs
// matches what it is. True when it was started.
bool hand_over_to_newer(const Manifest& manifest, const InstallSink& sink, const std::wstring& args) {
    bool comparable = false;
    if (!version_less(NINFERNO_VERSION, manifest.version, comparable) || !comparable) { return false; }
    sink.report(fmt(word(Word::NewerSetup), {widen(manifest.version)}), -1);
    const std::wstring setup = fetch_asset(manifest, "NInferno-Setup.exe", sink.report, sink.cancelled);
    HANDLE process = spawn(quote(setup) + L" " + args);
    if (!process) { throw std::runtime_error("cannot start the new installer"); }
    CloseHandle(process);
    return true;
}

// The whole installation; throws naming what failed. False when it stopped for a restart of
// Windows (WSL) or handed over to a newer installer.
bool install(const InstallSink& sink, bool quiet, const std::wstring& resume_args) {
    sink.step(kGpu, 1);
    const Gpu gpu = detect_gpu();
    if (!gpu.found) {
        throw std::runtime_error(narrow(word(gpu.has_driver ? Word::NoGpu : Word::NoDriver)));
    }
    if (gpu.arch.empty()) { throw std::runtime_error(narrow(fmt(word(Word::GpuUnsupported), {gpu.compute_cap}))); }
    if (_wtoi(gpu.driver.c_str()) < gpu.min_driver) {
        sink.report(fmt(word(Word::DriverOld), {gpu.driver, std::to_wstring(gpu.min_driver)}), -1);
    }
    const Manifest manifest = read_manifest();
    log_line("manifest " + manifest.version + " for sm_" + gpu.arch);
    if (hand_over_to_newer(manifest, sink, resume_args)) { return false; }
    sink.step(kGpu, 2);

    sink.step(kWsl, 1);
    if (!native_engine(manifest, gpu) && !wsl_ready()) {
        if (quiet || message(sink.owner, word(Word::WslNeeded), MB_YESNO | MB_ICONINFORMATION) != IDYES) {
            throw std::runtime_error(narrow(word(Word::WslAdmin)));
        }
        if (!wsl_turn_on()) { throw std::runtime_error(narrow(word(Word::WslAdmin))); }
        if (!wsl_ready()) {
            // The Windows feature needs a restart: continue from here after it.
            HKEY key = nullptr;
            if (RegCreateKeyExW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce", 0,
                                nullptr, 0, KEY_WRITE, nullptr, &key, nullptr) == ERROR_SUCCESS) {
                const std::wstring value = quote(self_path()) + L" " + resume_args;
                RegSetValueExW(key, L"NInfernoSetup", 0, REG_SZ, reinterpret_cast<const BYTE*>(value.c_str()),
                               static_cast<DWORD>((value.size() + 1) * sizeof(wchar_t)));
                RegCloseKey(key);
            }
            if (message(sink.owner, word(Word::WslRestart), MB_YESNO | MB_ICONINFORMATION) == IDYES) {
                if (HANDLE p = spawn(L"shutdown.exe /r /t 5")) { CloseHandle(p); }
            }
            return false;
        }
    }
    sink.step(kWsl, 2);

    sink.step(kDownload, 1);
    // install_engine downloads, then installs: the step moves on at its first installing line.
    bool installing = false;
    install_engine(
        manifest, gpu,
        [&](const std::wstring& text, double share) {
            if (!installing && text.rfind(fmt(word(Word::Installing), {widen(manifest.version)}), 0) == 0) {
                installing = true;
                sink.step(kDownload, 2);
                sink.step(kInstall, 1);
            }
            sink.report(text, share);
        },
        sink.cancelled);
    sink.step(kDownload, 2);
    sink.step(kInstall, 2);

    sink.step(kShortcuts, 1);
    register_app(manifest.version);
    sink.step(kShortcuts, 2);
    return true;
}

// ---- the installer's window ------------------------------------------------------------------

struct SetupWindow {
    HWND hwnd = nullptr, gpu = nullptr, folder = nullptr, status = nullptr, progress = nullptr;
    HWND cancel = nullptr, action = nullptr;
    HWND steps[kSteps]{};
    int step_state[kSteps]{};
    HFONT title_font = nullptr, text_font = nullptr, bold_font = nullptr;
    int dpi = 96;
    std::atomic<bool> cancelled{false};
    bool running = false, finished = false;
    std::wstring status_text;
    std::thread worker;
};

SetupWindow g_setup;

int scale(int value) { return MulDiv(value, g_setup.dpi, 96); }

HFONT make_font(int points, int weight) {
    return CreateFontW(-MulDiv(points, g_setup.dpi, 72), 0, 0, 0, weight, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                       OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
}

const wchar_t* step_name(int step) {
    static const Word words[kSteps] = {Word::StepGpu, Word::StepWsl, Word::StepDownload, Word::StepInstall,
                                       Word::StepShortcuts};
    return word(words[step]);
}

void paint_step(int step) {
    static const wchar_t* marks[] = {L"○  ", L"●  ", L"✓  ", L"✕  "};
    const std::wstring text = marks[g_setup.step_state[step]] + std::wstring(step_name(step));
    SetWindowTextW(g_setup.steps[step], text.c_str());
    SendMessageW(g_setup.steps[step], WM_SETFONT,
                 reinterpret_cast<WPARAM>(g_setup.step_state[step] == 1 ? g_setup.bold_font : g_setup.text_font), TRUE);
}

// Messages from the worker thread: kSetupEvent with wParam = kind.
enum Event { kStepEvent, kReportEvent, kDoneEvent, kFailEvent, kStoppedEvent };

struct EventData {
    int step = 0, state = 0;
    std::wstring text;
    double share = -1;
};

void post(Event kind, EventData* data) { PostMessageW(g_setup.hwnd, kSetupEvent, kind, reinterpret_cast<LPARAM>(data)); }

void start_install() {
    g_setup.running = true;
    EnableWindow(g_setup.action, FALSE);
    g_setup.worker = std::thread([] {
        InstallSink sink;
        sink.owner     = g_setup.hwnd;
        sink.step      = [](int step, int state) { post(kStepEvent, new EventData{step, state}); };
        sink.report    = [](const std::wstring& text, double share) { post(kReportEvent, new EventData{0, 0, text, share}); };
        sink.cancelled = [] { return g_setup.cancelled.load(); };
        try {
            const bool done = install(sink, false, L"--install");
            post(done ? kDoneEvent : kStoppedEvent, new EventData{});
        } catch (const std::exception& error) {
            log_line(std::string("failed: ") + error.what());
            post(kFailEvent, new EventData{0, 0, widen(error.what())});
        }
    });
}

LRESULT CALLBACK setup_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    switch (msg) {
    case WM_CTLCOLORSTATIC: {
        HDC dc = reinterpret_cast<HDC>(wparam);
        SetBkColor(dc, RGB(255, 255, 255));
        const HWND control = reinterpret_cast<HWND>(lparam);
        SetTextColor(dc, control == g_setup.status || control == g_setup.folder ? RGB(0x64, 0x6d, 0x78)
                                                                                 : RGB(0x1a, 0x1d, 0x21));
        return reinterpret_cast<LRESULT>(GetStockObject(WHITE_BRUSH));
    }
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(hwnd, &ps);
        RECT client;
        GetClientRect(hwnd, &client);
        RECT band = client;
        band.bottom = scale(92);
        HBRUSH accent = CreateSolidBrush(kAccent);
        FillRect(dc, &band, accent);
        DeleteObject(accent);
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, RGB(255, 255, 255));
        HGDIOBJ old = SelectObject(dc, g_setup.title_font);
        RECT title{scale(24), scale(14), client.right - scale(24), scale(54)};
        DrawTextW(dc, L"NInferno", -1, &title, DT_LEFT | DT_SINGLELINE);
        SelectObject(dc, g_setup.text_font);
        RECT subtitle{scale(24), scale(56), client.right - scale(24), scale(84)};
        SetTextColor(dc, RGB(0xdc, 0xd9, 0xff));
        DrawTextW(dc, word(Word::Subtitle), -1, &subtitle, DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS);
        SelectObject(dc, old);
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_COMMAND:
        if (reinterpret_cast<HWND>(lparam) == g_setup.action) {
            if (g_setup.finished) {
                if (HANDLE p = spawn(quote(installed_exe()) + L" --launch")) { CloseHandle(p); }
                DestroyWindow(hwnd);
            } else if (!g_setup.running) {
                start_install();
            }
        } else if (reinterpret_cast<HWND>(lparam) == g_setup.cancel) {
            SendMessageW(hwnd, WM_CLOSE, 0, 0);
        }
        return 0;
    case kSetupEvent: {
        std::unique_ptr<EventData> data(reinterpret_cast<EventData*>(lparam));
        switch (static_cast<Event>(wparam)) {
        case kStepEvent:
            g_setup.step_state[data->step] = data->state;
            paint_step(data->step);
            break;
        case kReportEvent: {
            SetWindowTextW(g_setup.status, data->text.c_str());
            const bool known = data->share >= 0;
            LONG_PTR style   = GetWindowLongPtrW(g_setup.progress, GWL_STYLE);
            if (known == ((style & PBS_MARQUEE) != 0)) {
                SetWindowLongPtrW(g_setup.progress, GWL_STYLE, known ? style & ~PBS_MARQUEE : style | PBS_MARQUEE);
                SendMessageW(g_setup.progress, PBM_SETMARQUEE, known ? FALSE : TRUE, 30);
            }
            if (known) { SendMessageW(g_setup.progress, PBM_SETPOS, static_cast<WPARAM>(data->share * 1000), 0); }
            break;
        }
        case kDoneEvent: {
            g_setup.running  = false;
            g_setup.finished = true;
            SendMessageW(g_setup.progress, PBM_SETMARQUEE, FALSE, 0);
            SetWindowLongPtrW(g_setup.progress, GWL_STYLE, GetWindowLongPtrW(g_setup.progress, GWL_STYLE) & ~PBS_MARQUEE);
            SendMessageW(g_setup.progress, PBM_SETPOS, 1000, 0);
            SetWindowTextW(g_setup.status, fmt(word(Word::Done), {widen(installed_version())}).c_str());
            SetWindowTextW(g_setup.action, word(Word::Open));
            SetWindowTextW(g_setup.cancel, word(Word::Close));
            EnableWindow(g_setup.action, TRUE);
            break;
        }
        case kStoppedEvent:
            DestroyWindow(hwnd);
            break;
        case kFailEvent:
            g_setup.running = false;
            for (int s = 0; s < kSteps; ++s) {
                if (g_setup.step_state[s] == 1) { g_setup.step_state[s] = 3; paint_step(s); }
            }
            SendMessageW(g_setup.progress, PBM_SETMARQUEE, FALSE, 0);
            SetWindowTextW(g_setup.status, fmt(word(Word::Failed), {data->text}).c_str());
            message(hwnd, fmt(word(Word::Failed), {data->text}) + L"\n\n" + fmt(word(Word::LogHint), {log_path()}),
                    MB_OK | MB_ICONERROR);
            EnableWindow(g_setup.action, TRUE);
            break;
        }
        return 0;
    }
    case WM_CLOSE:
        if (g_setup.running) {
            g_setup.cancelled = true; // the download stops at its next block; the worker then fails
            return 0;
        }
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

int run_setup_window() {
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc   = setup_proc;
    wc.hInstance     = g_instance;
    wc.hCursor       = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(GetStockObject(WHITE_BRUSH));
    wc.lpszClassName = kSetupClass;
    wc.hIcon         = LoadIconW(g_instance, MAKEINTRESOURCEW(1));
    RegisterClassExW(&wc);

    g_setup.dpi = static_cast<int>(GetDpiForSystem());
    g_setup.title_font = make_font(22, FW_BOLD);
    g_setup.text_font  = make_font(10, FW_NORMAL);
    g_setup.bold_font  = make_font(10, FW_SEMIBOLD);
    const DWORD style  = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    RECT frame{0, 0, scale(600), scale(440)};
    AdjustWindowRectExForDpi(&frame, style, FALSE, 0, static_cast<UINT>(g_setup.dpi));
    const int w = frame.right - frame.left, h = frame.bottom - frame.top;
    g_setup.hwnd = CreateWindowExW(0, kSetupClass, word(Word::Title), style, (GetSystemMetrics(SM_CXSCREEN) - w) / 2,
                                   (GetSystemMetrics(SM_CYSCREEN) - h) / 2, w, h, nullptr, nullptr, g_instance, nullptr);
    const auto label = [](int x, int y, int width, int height, const std::wstring& text) {
        HWND control = CreateWindowExW(0, L"STATIC", text.c_str(), WS_CHILD | WS_VISIBLE | SS_ENDELLIPSIS | SS_NOPREFIX,
                                       scale(x), scale(y), scale(width), scale(height), g_setup.hwnd, nullptr,
                                       g_instance, nullptr);
        SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(g_setup.text_font), TRUE);
        return control;
    };
    const Gpu gpu = detect_gpu();
    std::wstring gpu_text =
        !gpu.found ? word(gpu.has_driver ? Word::NoGpu : Word::NoDriver)
        : gpu.arch.empty()
            ? fmt(word(Word::GpuUnsupported), {gpu.compute_cap})
            : fmt(word(Word::Gpu), {gpu.name, format_bytes(gpu.memory_mib << 20), gpu.driver});
    g_setup.gpu = label(24, 108, 552, 22, gpu_text);
    SendMessageW(g_setup.gpu, WM_SETFONT, reinterpret_cast<WPARAM>(g_setup.bold_font), TRUE);
    ULARGE_INTEGER free_bytes{};
    const std::wstring drive = data_dir().substr(0, 3);
    GetDiskFreeSpaceExW(drive.c_str(), &free_bytes, nullptr, nullptr);
    g_setup.folder = label(24, 132, 552, 22,
                           fmt(word(Word::DiskSpace), {format_bytes(free_bytes.QuadPart), drive.substr(0, 2)}));
    for (int s = 0; s < kSteps; ++s) {
        g_setup.steps[s] = label(36, 170 + s * 28, 520, 24, L"");
        paint_step(s);
    }
    g_setup.progress = CreateWindowExW(0, PROGRESS_CLASSW, nullptr, WS_CHILD | WS_VISIBLE | PBS_SMOOTH, scale(24),
                                       scale(322), scale(552), scale(8), g_setup.hwnd, nullptr, g_instance, nullptr);
    SendMessageW(g_setup.progress, PBM_SETRANGE32, 0, 1000);
    g_setup.status = label(24, 338, 552, 22, installed_version().empty() ? L"" : fmt(word(Word::Folder), {app_dir()}));
    const auto button = [](int x, const wchar_t* text, bool primary) {
        HWND control = CreateWindowExW(0, L"BUTTON", text, WS_CHILD | WS_VISIBLE | WS_TABSTOP |
                                                               (primary ? BS_DEFPUSHBUTTON : BS_PUSHBUTTON),
                                       scale(x), scale(386), scale(120), scale(32), g_setup.hwnd, nullptr, g_instance,
                                       nullptr);
        SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(g_setup.text_font), TRUE);
        return control;
    };
    g_setup.cancel = button(320, word(Word::Cancel), false);
    g_setup.action = button(456, word(Word::Install), true);
    if (!gpu.found || gpu.arch.empty()) { EnableWindow(g_setup.action, FALSE); }
    ShowWindow(g_setup.hwnd, SW_SHOW);
    SetFocus(g_setup.action);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        if (!IsDialogMessageW(g_setup.hwnd, &msg)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
    g_setup.cancelled = true;
    if (g_setup.worker.joinable()) { g_setup.worker.join(); }
    return 0;
}

// ---- quiet install (updates) -------------------------------------------------------------------

void wait_for_launcher_exit() {
    for (int i = 0; i < 300; ++i) {
        HANDLE mutex = OpenMutexW(SYNCHRONIZE, FALSE, kLauncherMutex);
        if (!mutex) { return; }
        CloseHandle(mutex);
        Sleep(100);
    }
}

// relaunch: start NInferno afterwards; browser: and open the page (not when the page asked for the
// update: it reloads itself).
int run_quiet_install(bool relaunch, bool browser) {
    InstallSink sink;
    int last_percent = -1;
    sink.step   = [](int step, int state) { if (state == 1) { status(narrow(step_name(step))); } };
    sink.report = [&](const std::wstring& text, double share) {
        // Downloads report often: every 5 %.
        const int percent = share >= 0 ? static_cast<int>(share * 100) : -1;
        if (percent >= 0 && percent / 5 == last_percent / 5) { return; }
        last_percent = percent;
        status(narrow(text));
    };
    sink.cancelled = [] { return false; };
    wait_for_launcher_exit();
    try {
        if (!install(sink, true, L"--install --quiet" + std::wstring(relaunch ? L" --relaunch" : L"") + (browser ? L"" : L" --no-browser"))) { return 0; }
    } catch (const std::exception& error) {
        status(std::string("failed: ") + error.what());
        if (relaunch) {
            if (HANDLE p = spawn(quote(installed_exe()) + L" --launch" + (browser ? L"" : L" --no-browser"))) { CloseHandle(p); }
        }
        return 1;
    }
    status("done " + installed_version());
    if (relaunch) {
        stop_engine(); // the page and models of the old version
        if (HANDLE p = spawn(quote(installed_exe()) + L" --launch" + (browser ? L"" : L" --no-browser"))) { CloseHandle(p); }
    }
    return 0;
}

// ---- launcher: the page, the notification icon, updates -----------------------------------------

struct Launcher {
    HWND hwnd = nullptr;
    NOTIFYICONDATAW icon{};
    HANDLE ui = nullptr;
    std::atomic<bool> updating{false};
    std::string offered; // the version the last notification named
    bool open_browser = true; // false with --no-browser (a start at sign-in: the icon only)
};

Launcher g_launcher;

void tray_balloon(const std::wstring& text) {
    g_launcher.icon.uFlags      = NIF_INFO;
    g_launcher.icon.dwInfoFlags = NIIF_INFO;
    wcsncpy(g_launcher.icon.szInfoTitle, L"NInferno", 63);
    wcsncpy(g_launcher.icon.szInfo, text.c_str(), 255);
    Shell_NotifyIconW(NIM_MODIFY, &g_launcher.icon);
}

// Downloads the newest installer and hands over to it (it stops this launcher's engine, installs,
// and starts NInferno again). Runs on its own thread. from_page: the page asked (its Update button),
// so the new version opens no second tab.
void update_now(bool from_page) {
    if (g_launcher.updating.exchange(true)) { return; }
    std::thread([from_page] {
        DeleteFileW(status_path().c_str());
        try {
            const Manifest manifest = read_manifest();
            bool comparable = false;
            if (!version_less(NINFERNO_VERSION, manifest.version, comparable)) {
                status("done " + std::string(NINFERNO_VERSION) + " (the newest)");
                g_launcher.updating = false;
                return;
            }
            status(narrow(fmt(word(Word::Updating), {widen(manifest.version)})));
            int last = -1;
            const std::wstring setup = fetch_asset(manifest, "NInferno-Setup.exe",
                                                   [&](const std::wstring& text, double share) {
                                                       const int p = share >= 0 ? static_cast<int>(share * 20) : -1;
                                                       if (p != last) { last = p; status(narrow(text)); }
                                                   },
                                                   [] { return false; });
            HANDLE process = spawn(quote(setup) + L" --install --quiet --relaunch" + (from_page ? L" --no-browser" : L""));
            if (!process) { throw std::runtime_error("cannot start the new installer"); }
            CloseHandle(process);
            // The new installer waits for this launcher to go before it replaces it.
            PostMessageW(g_launcher.hwnd, WM_CLOSE, 0, 0);
        } catch (const std::exception& error) {
            status(std::string("failed: ") + error.what());
            g_launcher.updating = false;
        }
    }).detach();
}

void check_updates(bool said_by_user) {
    std::thread([said_by_user] {
        try {
            const Manifest manifest = read_manifest();
            bool comparable = false;
            const bool newer = version_less(NINFERNO_VERSION, manifest.version, comparable) && comparable;
            bool required    = false;
            if (!manifest.min_version.empty()) {
                required = version_less(NINFERNO_VERSION, manifest.min_version, comparable) && comparable;
            }
            if (required) {
                if (message(nullptr, fmt(word(Word::UpdateRequired), {widen(NINFERNO_VERSION), widen(manifest.version)}),
                            MB_YESNO | MB_ICONWARNING | MB_SETFOREGROUND) == IDYES) {
                    update_now(false);
                }
            } else if (newer) {
                g_launcher.offered = manifest.version;
                tray_balloon(fmt(word(Word::UpdateAvailable), {widen(manifest.version), widen(NINFERNO_VERSION)}));
            } else if (said_by_user) {
                tray_balloon(fmt(word(Word::UpToDate), {widen(NINFERNO_VERSION)}));
            }
        } catch (const std::exception& error) {
            log_line(std::string("update check: ") + error.what());
        }
    }).detach();
}

enum MenuItem { kOpen = 1, kCheck, kQuit };

LRESULT CALLBACK launcher_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    switch (msg) {
    case kTrayMessage:
        switch (LOWORD(lparam)) {
        case WM_LBUTTONUP:
        case NIN_SELECT:
            open_page();
            break;
        case NIN_BALLOONUSERCLICK:
            if (!g_launcher.offered.empty()) { update_now(false); }
            break;
        case WM_CONTEXTMENU:
        case WM_RBUTTONUP: {
            HMENU menu = CreatePopupMenu();
            AppendMenuW(menu, MF_STRING, kOpen, word(Word::TrayOpen));
            AppendMenuW(menu, MF_STRING, kCheck, word(Word::TrayUpdate));
            AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
            AppendMenuW(menu, MF_STRING, kQuit, word(Word::TrayQuit));
            SetMenuDefaultItem(menu, kOpen, FALSE);
            POINT at;
            GetCursorPos(&at);
            SetForegroundWindow(hwnd);
            const int chosen = TrackPopupMenu(menu, TPM_RETURNCMD | TPM_RIGHTBUTTON, at.x, at.y, 0, hwnd, nullptr);
            DestroyMenu(menu);
            if (chosen == kOpen) { open_page(); }
            if (chosen == kCheck) { check_updates(true); }
            if (chosen == kQuit) {
                stop_engine();
                DestroyWindow(hwnd);
            }
            break;
        }
        }
        return 0;
    case WM_COPYDATA: {
        // NInferno.exe --update (the page's Update button) asks this launcher to update.
        const auto* data = reinterpret_cast<const COPYDATASTRUCT*>(lparam);
        if (data && data->dwData == 1) { update_now(true); }
        return TRUE;
    }
    case WM_CLOSE:
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        // The native page and its models end with the launcher (WSL's end with the distribution).
        if (g_launcher.ui && installed_native()) { TerminateProcess(g_launcher.ui, 0); }
        Shell_NotifyIconW(NIM_DELETE, &g_launcher.icon);
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

int run_launcher() {
    HANDLE mutex = CreateMutexW(nullptr, TRUE, kLauncherMutex);
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        open_page(); // already running: its page
        return 0;
    }
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc   = launcher_proc;
    wc.hInstance     = g_instance;
    wc.lpszClassName = kLauncherClass;
    RegisterClassExW(&wc);
    g_launcher.hwnd = CreateWindowExW(0, kLauncherClass, L"NInferno", 0, 0, 0, 0, 0, nullptr, nullptr, g_instance, nullptr);

    g_launcher.icon.cbSize           = sizeof(g_launcher.icon);
    g_launcher.icon.hWnd             = g_launcher.hwnd;
    g_launcher.icon.uID              = 1;
    g_launcher.icon.uFlags           = NIF_ICON | NIF_TIP | NIF_MESSAGE;
    g_launcher.icon.uCallbackMessage = kTrayMessage;
    g_launcher.icon.hIcon            = LoadIconW(g_instance, MAKEINTRESOURCEW(1));
    wcsncpy(g_launcher.icon.szTip, word(Word::Starting), 127);
    Shell_NotifyIconW(NIM_ADD, &g_launcher.icon);
    g_launcher.icon.uVersion = NOTIFYICON_VERSION_4;
    Shell_NotifyIconW(NIM_SETVERSION, &g_launcher.icon);

    g_launcher.ui = start_ui();
    std::thread([] {
        // The page answers once ninferno-ui listens (the distribution starts in a few seconds).
        for (int i = 0; i < 1200; ++i) {
            if (ui_listening()) {
                g_launcher.icon.uFlags = NIF_TIP;
                wcsncpy(g_launcher.icon.szTip, L"NInferno", 127);
                Shell_NotifyIconW(NIM_MODIFY, &g_launcher.icon);
                if (g_launcher.open_browser) { open_page(); }
                return;
            }
            if (!g_launcher.ui || WaitForSingleObject(g_launcher.ui, 0) == WAIT_OBJECT_0) { break; }
            Sleep(100);
        }
        message(nullptr, fmt(word(Word::StartFailed), {fmt(word(Word::LogHint), {log_path()})}), MB_OK | MB_ICONERROR);
        PostMessageW(g_launcher.hwnd, WM_CLOSE, 0, 0);
    }).detach();
    check_updates(false);
    SetTimer(g_launcher.hwnd, 1, 24 * 3600 * 1000, [](HWND, UINT, UINT_PTR, DWORD) { check_updates(false); });

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    if (g_launcher.ui) { CloseHandle(g_launcher.ui); }
    ReleaseMutex(mutex);
    CloseHandle(mutex);
    return 0;
}

// ---- --update: asked by the page (through WSL) or from a terminal --------------------------------

int run_update_request() {
    AttachConsole(ATTACH_PARENT_PROCESS);
    std::FILE* out = _wfopen(L"CONOUT$", L"w");
    const auto say = [&](const std::string& line) {
        std::fputs((line + "\n").c_str(), out ? out : stdout);
        std::fflush(out ? out : stdout);
        std::fputs((line + "\n").c_str(), stdout);
        std::fflush(stdout);
    };
    HWND launcher = FindWindowW(kLauncherClass, nullptr);
    DeleteFileW(status_path().c_str());
    if (launcher) {
        COPYDATASTRUCT data{1, 0, nullptr};
        SendMessageW(launcher, WM_COPYDATA, 0, reinterpret_cast<LPARAM>(&data));
    } else {
        // No launcher: this process updates (it is not inside a distribution that stops).
        std::thread([] { run_quiet_install(false, false); }).detach();
    }
    // Follow update.status until it says done or failed (or until this process ends with the
    // engine the update restarts).
    std::size_t shown = 0;
    for (int i = 0; i < 2 * 3600 * 4; ++i) {
        std::ifstream in(std::filesystem::path(status_path()), std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        while (shown < text.size()) {
            const auto nl = text.find('\n', shown);
            if (nl == std::string::npos) { break; }
            const std::string line = text.substr(shown, nl - shown);
            shown                  = nl + 1;
            say(line);
            if (line.rfind("done", 0) == 0) { return 0; }
            if (line.rfind("failed", 0) == 0) { return 1; }
        }
        Sleep(250);
    }
    return 1;
}

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int) {
    g_instance = instance;
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    INITCOMMONCONTROLSEX controls{sizeof(controls), ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES};
    InitCommonControlsEx(&controls);
    int argc         = 0;
    wchar_t** argv   = CommandLineToArgvW(GetCommandLineW(), &argc);
    const bool quiet = has_arg(argc, argv, L"--quiet");
    log_line("NInferno-Setup " NINFERNO_VERSION ": " + narrow(GetCommandLineW()));

    if (has_arg(argc, argv, L"--uninstall")) {
        ULARGE_INTEGER used{};
        // The distribution's disk holds the engine and every downloaded model.
        WIN32_FILE_ATTRIBUTE_DATA disk{};
        if (GetFileAttributesExW((data_dir() + L"\\wsl\\ext4.vhdx").c_str(), GetFileExInfoStandard, &disk)) {
            used.LowPart  = disk.nFileSizeLow;
            used.HighPart = disk.nFileSizeHigh;
        }
        const std::wstring question =
            installed_native() ? fmt(word(Word::UninstallAskNative), {env(L"USERPROFILE") + L"\\.cache\\huggingface"})
                               : fmt(word(Word::UninstallAsk), {format_bytes(used.QuadPart)});
        if (!quiet && message(nullptr, question, MB_YESNO | MB_ICONQUESTION) != IDYES) {
            return 1;
        }
        if (HWND launcher = FindWindowW(kLauncherClass, nullptr)) { PostMessageW(launcher, WM_CLOSE, 0, 0); }
        wait_for_launcher_exit();
        unregister_app();
        if (!quiet) { message(nullptr, word(Word::UninstallDone), MB_OK | MB_ICONINFORMATION); }
        return 0;
    }
    if (has_arg(argc, argv, L"--update")) { return run_update_request(); }
    if (has_arg(argc, argv, L"--install") && quiet) {
        return run_quiet_install(has_arg(argc, argv, L"--relaunch"), !has_arg(argc, argv, L"--no-browser"));
    }
    const bool installed_copy = _wcsicmp(self_path().c_str(), installed_exe().c_str()) == 0;
    if (has_arg(argc, argv, L"--launch") || (installed_copy && !has_arg(argc, argv, L"--install"))) {
        g_launcher.open_browser = !has_arg(argc, argv, L"--no-browser");
        return run_launcher();
    }
    return run_setup_window();
}
