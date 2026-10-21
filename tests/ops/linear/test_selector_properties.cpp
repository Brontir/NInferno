// Properties of the generic projection path's choices (docs/maintainer/deterministic-decisions.md),
// checked on a CPU over many substituted boards (core/device_class.h): the golden boards and
// synthetic ones of every compute capability from 7.0 to 12.1 at the smallest, a typical and the
// largest SM count of their generation (selector_boards.h). Every (format, policy, epilogue,
// shape, profile, T) of selector_boards.h is asked; no GPU is needed.
//
//   P1 determinism: a second evaluation in the same process, after one under other process-global
//      settings (the prefill route profile, a four-lane round) and a dense-tuner reset, gives the
//      same choices; separate processes give the same choices (P2's runs and the digests, P6).
//   P2 no hidden inputs:
//      - env: the choices do not move when every NINFERNO_* variable the tree names is set (removed
//        ones included) and HOME / XDG_CACHE_HOME point elsewhere. No variable is read by a rule
//        any more, so this leg is a regression guard: a new knob that moves a choice fails here
//        (and the source scan requires every NINFERNO_* name in kEnvVars).
//      - tune: a dense-tune cache naming the scalar route at every decode width for every A16
//        problem of the board, with the tuner enabled: under --dense-tune off the choices must not
//        move; under on they may (a tuning run, tainted in the fingerprint), which is reported.
//        On a CPU both legs only show that nothing but the tuner reaches the file: the tuner reads
//        it for a weight in device memory alone (cudaPointerGetAttributes), which no CPU run has.
//      - clock: zero clocks (MIG, WSL) or skewed ones, the facts derived from them as the query
//        does. The invariant profile must not move at all.
//   P3 capability safety: every choice is one route_admits accepts and needs no feature the board
//      and this build lack (route_requires); W4A4 and the NVFP4 tiled routes only where compiled;
//      the substituted class carries no feature the build lacks.
//   P4 exact profile: under the invariant route profile a problem sums in one order at every width
//      (one summation class from T = 1 to 2048) with BF16 activations; under the prefill profile a
//      call of 17 tokens or more takes token-invariant routes of one summation class.
//   P5 plan levels: no plan-level API exists yet (the pillar's gap list); nothing is checked.
//   P6 digest: each board's choices hash to the lines checked in under selector_digest/<set>.txt
//      (the golden text's FNV-1a, whole and per format: a board's whole-text digest is the hash of
//      its selector_golden file), the compact form meant to replace the golden files. A board
//      without lines fails.
//
// A violation that exists today is listed in selector_properties_known.txt with its gap, the route
// sets (@set) and the boards (%board, %group: "group <name> <board>...", or the built-in known-pci
// and unknown-pci) it is seen on; the test fails on a violation of a board no entry covers and on
// an entry no board shows any more (then the entry is removed and the pillar's gap list updated).
// --update rewrites the digest file of this build's route set.
//
// The route table and several heuristics cache what they read of the device on first use, so one
// process evaluates one board: without --board the test runs itself per board and variant, in
// parallel.
#include "ops/linear/selector_boards.h"

#include "ops/common/native_nvfp4.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#    include <process.h>
#    define popen  _popen
#    define pclose _pclose
#else
#    include <unistd.h>
#endif

namespace {

using namespace ninferno;
using namespace ninferno::test::selector;

// Every NINFERNO_* variable the tree names, with a value other than its default. "{tmp}" is the
// run's scratch directory. The source scan (tools/audit/selection_static_check.py) fails when src/,
// apps/ or include/ names one this list lacks.
struct EnvVar {
    const char* name;
    const char* value;
};

constexpr EnvVar kEnvVars[] = {
    {"NINFERNO_API_KEY", "selector-properties"},
    {"NINFERNO_ARTIFACT_DIRECT_IO", "0"},
    {"NINFERNO_CATALOG_URL", "http://127.0.0.1:9/catalog"},
    {"NINFERNO_CUDA_SYNC", "1"},
    {"NINFERNO_DEBUG_CUDA_ERROR_FILE", "{tmp}/cuda_error"},
    {"NINFERNO_DEBUG_FREEZE_FILE", "{tmp}/freeze"},
    {"NINFERNO_DEBUG_NAN_FILE", "{tmp}/nan"},
    {"NINFERNO_DEBUG_POISON", "1"},
    {"NINFERNO_DEBUG_POISON_SKIP", "linear"},
    {"NINFERNO_DEBUG_TRAP_FILE", "{tmp}/trap"},
    {"NINFERNO_DECODE_STEPS", "3"},
    {"NINFERNO_DECODE_WAIT", "1"},
    {"NINFERNO_DENSE_DUMP", "{tmp}/dense_dump"},
    {"NINFERNO_DENSE_DUMP_ROWS", "4"},
    {"NINFERNO_DENSE_TUNE", "on"},
    {"NINFERNO_DENSE_TUNE_CACHE", "{tmp}/cache/ninferno/dense_tune.json"},
    {"NINFERNO_DRAFTER_REGISTRY", "{tmp}/registry.json"},
    {"NINFERNO_GEMMA4_DUMP", "{tmp}/gemma4_dump"},
    {"NINFERNO_GEMMA4_DUMP_ROWS", "4"},
    {"NINFERNO_MMQ_VERIFY", "0"},
    {"NINFERNO_MOE_TUNE", "on"},
    {"NINFERNO_MOE_TUNE_CACHE", "{tmp}/moe_tune.json"},
    {"NINFERNO_PDFIUM", "{tmp}/libpdfium.so"},
    {"NINFERNO_QWEN_DUMP", "{tmp}/qwen_dump"},
    {"NINFERNO_QWEN_DUMP_ROWS", "4"},
    {"NINFERNO_QWEN_DUMP_KINDS", "all"},
    {"NINFERNO_QWEN_PREFILL_CUTS", "64"},
    {"NINFERNO_RANKING_DIR", "{tmp}"},
    {"NINFERNO_ROUTE_DIR", "{tmp}/routes"},
    {"NINFERNO_ROUTE_RECORD", "{tmp}/route_record.txt"},
    {"NINFERNO_ROUTE_TABLE", "{tmp}/route_table.json"},
    {"NINFERNO_SCORE_ARGMAX", "1"},
    {"NINFERNO_SCORE_LOGITS", "1"},
    {"NINFERNO_SPEC_CORPUS", "{tmp}/corpus"},
    {"NINFERNO_UI_KEY", "selector-properties"},
    {"NINFERNO_UPDATER", "{tmp}/updater"},
    {"NINFERNO_UPDATE_CHECK", "0"},
    {"NINFERNO_UPDATE_URL", "http://127.0.0.1:9/update"},
    // Where NInferno's cache directory (the dense tuner's default user cache) lives.
    {"HOME", "{tmp}"},
    {"XDG_CACHE_HOME", "{tmp}/cache"},
    {"LOCALAPPDATA", "{tmp}/cache"},
};

// ---------------------------------------------------------------------------------------------
// The child: one board, one variant.

struct Violations {
    std::map<std::string, std::pair<int, std::vector<std::string>>> by_key; // "<prop> <subject>"

    void add(const std::string& property, const std::string& subject, const std::string& detail) {
        auto& [count, details] = by_key[property + " " + subject];
        if (++count <= 3) details.push_back(detail);
    }

    void print() const {
        for (const auto& [key, v] : by_key) {
            for (const auto& d : v.second) std::cout << "V " << key << "\t" << d << '\n';
            if (v.first > 3) std::cout << "V " << key << "\t(" << v.first << " in all)\n";
        }
    }
};

const char* policy_key(LinearPolicy p) {
    return p == LinearPolicy::AllowA4 ? "a4" : p == LinearPolicy::AllowA8 ? "a8" : "a16";
}

struct SafetyCheck final : ChoiceCheck {
    Violations* out;
    DeviceFeatures available;
    std::size_t checked = 0;
    // The tune leg: the dense tuner sees every problem before its choices are asked, and the A16
    // ones get a cache entry naming the scalar route at every decode width.
    bool prime_tuner          = false;
    std::ostringstream* cache = nullptr;
    bool first_entry          = true;

    explicit SafetyCheck(Violations* v)
        : out(v), available(common_features(device_class().features, build_features())) {}

    void problem(const generic::Problem& p, const Weight& w) override {
        if (cache != nullptr && p.policy == LinearPolicy::A16Only) {
            // generic_tune.cpp's key for an A16 problem (no format or policy suffix applies to it).
            *cache << (first_entry ? "" : ",\n") << "\"" << device_class().id() << '|'
                   << generic::qtype_name(p.qtype) << '|' << policy_key(p.policy) << '|'
                   << generic::epilogue_name(p.epilogue) << "|n" << p.n << "|k" << p.k << "|s"
                   << generic::rmma_schedule_count() << '.' << generic::staged_schedule_count()
                   << '.' << generic::nvfp4_tiled_schedule_count() << '.'
                   << generic::r8_schedule_count() << '.' << generic::w4a8_schedule_count()
                   << (device_class().features.mma_tensor ? "" : "|ll4") << "\": {\"routes\": {";
            for (int t = 1; t <= generic::kDenseTuneMaxTokens; ++t) {
                *cache << (t == 1 ? "" : ", ") << '"' << t << "\": \"scalar\"";
            }
            *cache << "}}";
            first_entry = false;
        }
        if (prime_tuner) {
            generic::Call call;
            call.weight = &w;
            call.rows   = p.n;
            generic::tune_dense(p, call, nullptr);
        }
    }

    void check(const Row& row, std::int32_t t, const Weight& w, const Outcome& o) override {
        ++checked;
        const std::string where =
            row_key(row) + " T=" + std::to_string(t) + " " + generic::route_name(o.route);
        const std::string fmt = generic::qtype_name(row.qtype);
        if (!generic::route_admits(o.route, w, row.policy))
            out->add("P3", fmt, where + ": not admitted");
        if (!has_features(available, generic::route_requires(o.route))) {
            out->add("P3", fmt, where + ": needs a feature the board or the build lacks");
        }
        if (!ops::detail::kNativeNvfp4 && (o.route.family == generic::Family::W4a4Mma ||
                                           o.route.family == generic::Family::Nvfp4TiledMma)) {
            out->add("P3", fmt, where + ": absent from this build");
        }
    }
};

// The order a route sums an output element in, as far as the routes document it: the register
// MMA's order depends on its K split alone (WarpsK, and the cluster's CTAs), the staged routes sum
// as WarpsK = 1 and the NVFP4 tiled routes as WarpsK = classes; the routes that quantize or read
// each token on its own (W4A4, W4A8, R8, I8, ggml's vector kernel) and the BF16 tiled kernel give a
// token the same bytes whatever their tile; ggml's matrix kernel splits K by the call's tiles.
std::string summation_class(generic::Route r) {
    using generic::Family;
    switch (r.family) {
    case Family::RegisterMma: {
        const auto s = generic::rmma_schedule_info(r.schedule);
        return "rmma.k" + std::to_string(s.warps_k) + ".c" + std::to_string(s.cluster);
    }
    case Family::StagedMma:
        return "rmma.k1.c1";
    case Family::Nvfp4TiledMma:
        return "rmma.k" + std::to_string(generic::nvfp4_tiled_schedule_info(r.schedule).classes) +
               ".c1";
    case Family::Scalar:
        return "scalar";
    case Family::W4a4Mma:
        return "w4a4";
    case Family::W4a8Mma:
        return "w4a8";
    case Family::R8Mma:
        return "r8";
    case Family::I8Mma:
        return "i8";
    case Family::GgmlVec:
        return "vec";
    case Family::TiledMma:
        return "tiled";
    case Family::GgmlMmq:
        return generic::route_name(r); // not invariant (route_is_token_invariant)
    }
    return generic::route_name(r);
}

bool bf16_activations(generic::Route r) {
    using generic::Family;
    return r.family == Family::RegisterMma || r.family == Family::StagedMma ||
           r.family == Family::Nvfp4TiledMma || r.family == Family::Scalar ||
           r.family == Family::TiledMma;
}

// P4. The invariant profile: one summation class at every T, BF16 activations. The prefill
// profile: one summation class from T = 17 on, and no route the selector itself calls width-
// dependent (route_is_token_invariant: ggml's matrix kernel, whose stream-k splits K by the
// call's tiles). A register MMA of WarpsK > 1 that a problem keeps at every width is one order.
void check_profiles(const Choices& c, Violations& v) {
    for (const Item& item : c.items) {
        if (!item.note.empty()) continue;
        const Row& row = item.row;
        if (row.profile == Profile::Speed) continue;
        const bool exact = row.profile == Profile::Invariant;
        const std::string subject =
            std::string(generic::qtype_name(row.qtype)) + " " + profile_name(row.profile);
        std::string first, classes;
        bool split = false;
        for (const auto& [t, o] : row.by_tokens) {
            if (o.threw) continue;
            const std::string where =
                row_key(row) + " T=" + std::to_string(t) + " " + generic::route_name(o.route);
            if (exact && !bf16_activations(o.route)) {
                v.add("P4", subject + " activations", where + ": quantized activations");
            }
            if (!exact && o.route.family == generic::Family::GgmlMmq) {
                v.add("P4", subject + " mmq", where + ": stream-k, not token-invariant");
            }
            const std::string cls = summation_class(o.route);
            if (first.empty()) first = cls;
            if (cls != first) split = true;
            if (classes.find(cls) == std::string::npos)
                classes += (classes.empty() ? "" : ",") + cls;
        }
        if (split) v.add("P4", subject + " order", row_key(row) + ": summation classes " + classes);
    }
}

// Per (format, profile) hashes of the text, for the parent's comparisons between processes.
std::map<std::string, std::string> part_hashes(const Choices& c) {
    std::map<std::string, std::string> text;
    for (const Item& item : c.items) {
        const std::string fmt = generic::qtype_name(item.row.qtype);
        const char* profile   = item.note.empty() ? profile_name(item.row.profile) : "speed";
        text[fmt + " " + profile] += render_item(item);
    }
    std::map<std::string, std::string> out;
    char hex[17];
    for (const auto& [key, t] : text) {
        std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(fnv1a(t)));
        out[key] = hex;
    }
    return out;
}

const TestBoard* find_board(std::string_view name) {
    for (const TestBoard& b : kGoldenBoards)
        if (name == b.name) return &b;
    for (const TestBoard& b : kSyntheticBoards)
        if (name == b.name) return &b;
    return nullptr;
}

struct ChildOptions {
    std::string clock;    // "", "zero", "skew"
    std::string tune;     // "", "off", "on"
    std::string tune_dir; // the tune leg's cache directory
    bool repeat = true;
};

int run_child(const TestBoard& b, const ChildOptions& opt) {
    DeviceClass d = device_class_of(b);
    if (opt.clock == "zero") {
        d.sm_clock_khz = d.memory_clock_khz = 0;
    } else if (opt.clock == "skew") {
        d.sm_clock_khz     = d.sm_clock_khz * 4 / 5;
        d.memory_clock_khz = d.memory_clock_khz * 5 / 4;
    }
    // The facts as the query derives them from the varied driver values (a known PCI id keeps the
    // table's; an unknown one takes the driver's, labelled extrapolated).
    d.facts = machine_facts_of(d);
    const ScopedDeviceClass scoped(d);
    Violations v;
    // The substituted class carries only what this build compiled, as the device query does
    // (common_features with build_features): a board newer than the build must not offer the
    // selector a kernel family the build lacks (an sm_75 build once offered the I8 tiles).
    if (!has_features(build_features(), device_class().features)) {
        v.add("P3", "model", "the substituted class has a feature this build lacks");
    }
    SafetyCheck safety(&v);
    if (!opt.tune.empty()) {
        // The cache first (keys need only the board and the build), then the tuner on it.
        std::ostringstream cache;
        safety.cache = &cache;
        (void)evaluate(b.name, &safety);
        safety.cache = nullptr;
        std::filesystem::create_directories(opt.tune_dir);
        std::ofstream(std::filesystem::path(opt.tune_dir) / "dense_tune.json")
            << "{\"schema\": \"ninferno-dense-tune/4\", \"problems\": {\n"
            << cache.str() << "\n}}\n";
        generic::set_dense_tune(opt.tune == "on" ? TuneMode::On : TuneMode::Off, opt.tune_dir);
        generic::enable_dense_tune(true);
        safety.prime_tuner = true;
        safety.checked     = 0;
    }
    const Choices first    = evaluate(b.name, &safety);
    const std::string text = render(first);
    check_profiles(first, v);
    // P1: the same choices after an evaluation under other process-global settings and a reset of
    // the dense tuner's state (the plain run only: the variants compare their first evaluation).
    if (opt.repeat) {
        {
            const ops::ScopedPrefillRoutes prefill(true);
            const ops::ScopedRoundLanes lanes(4);
            (void)evaluate(b.name);
        }
        generic::clear_dense_tune();
        const std::string again = render(evaluate(b.name));
        if (again != text) {
            std::istringstream a(text), c(again);
            std::string la, lc;
            while (std::getline(a, la) && std::getline(c, lc)) {
                if (la != lc) v.add("P1", "repeat", "first: " + la + " / again: " + lc);
            }
            v.add("P1", "repeat", "the second evaluation differs");
        }
    }
    std::cout << "C " << (d.facts.source == FactSource::Verified ? "known-pci" : "unknown-pci")
              << '\n';
    for (const auto& line : digest_lines(b.name, first)) std::cout << "D " << line << '\n';
    for (const auto& [key, hash] : part_hashes(first))
        std::cout << "P " << key << " " << hash << '\n';
    v.print();
    std::cout << "N " << safety.checked << '\n';
    return 0;
}

// ---------------------------------------------------------------------------------------------
// The parent: every board and variant in its own process.

struct Job {
    const TestBoard* board;
    std::string variant; // "plain", "env", "tune:off", "tune:on", "clock:zero", "clock:skew"
    std::string command;
    std::vector<std::pair<std::string, std::string>> env; // set for this run only
    // Results.
    int status = -1;
    std::string board_class;
    std::vector<std::string> digest;
    std::map<std::string, std::string> parts;
    std::vector<std::string> violations;
    std::string checked;
    std::string other; // unparsed output
};

std::string shell_quote(const std::string& s) {
#ifdef _WIN32
    return "\"" + s + "\"";
#else
    std::string out = "'";
    for (const char c : s) out += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return out + "'";
#endif
}

void run_job(Job& job) {
    std::string command = job.command;
#ifndef _WIN32
    if (!job.env.empty()) {
        std::string prefix = "env";
        for (const auto& [name, value] : job.env) prefix += " " + name + "=" + shell_quote(value);
        command = prefix + " " + command;
    }
#endif
    std::FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) return;
    std::string out;
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) out += buffer;
    job.status = pclose(pipe);
    std::istringstream lines(out);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.rfind("D ", 0) == 0) {
            job.digest.push_back(line.substr(2));
        } else if (line.rfind("C ", 0) == 0) {
            job.board_class = line.substr(2);
        } else if (line.rfind("P ", 0) == 0) {
            const auto cut                     = line.rfind(' ');
            job.parts[line.substr(2, cut - 2)] = line.substr(cut + 1);
        } else if (line.rfind("V ", 0) == 0) {
            job.violations.push_back(line.substr(2));
        } else if (line.rfind("N ", 0) == 0) {
            job.checked = line.substr(2);
        } else if (job.other.size() < 2000) {
            job.other += line + '\n';
        }
    }
}

std::string expand(std::string value, const std::string& tmp) {
    for (std::size_t at; (at = value.find("{tmp}")) != std::string::npos;)
        value.replace(at, 5, tmp);
    return value;
}

// A known violation: "<property> <subject...>", the route sets it holds in ("@full"; none: every
// set) and the boards it is seen on ("%board" or "%group"; none: every board).
struct Known {
    std::string key;
    std::set<std::string> sets;
    std::vector<std::string> boards;
    bool seen = false;
};

struct KnownList {
    std::vector<Known> entries;
    std::map<std::string, std::set<std::string>> groups; // "group <name> <board>..."
};

KnownList read_known(const std::string& path) {
    KnownList out;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::string line;
    while (std::getline(in, line)) {
        if (const auto hash = line.find('#'); hash != std::string::npos) line.resize(hash);
        std::istringstream words(line);
        std::string word, key;
        if (!(words >> word)) continue;
        if (word == "group") {
            std::string name;
            words >> name;
            auto& members = out.groups[name];
            while (words >> word) members.insert(word);
            continue;
        }
        Known k;
        do {
            if (word[0] == '@') {
                k.sets.insert(word.substr(1));
            } else if (word[0] == '%') {
                k.boards.push_back(word.substr(1));
            } else {
                key += (key.empty() ? "" : " ") + word;
            }
        } while (words >> word);
        k.key = key;
        out.entries.push_back(std::move(k));
    }
    return out;
}

bool applies(const Known& k) { return k.sets.empty() || k.sets.count(kBuildSet) != 0; }

bool covers(const KnownList& list, const Known& k, const std::string& board,
            const std::string& board_class) {
    if (k.boards.empty()) return true;
    for (const auto& b : k.boards) {
        if (b == board || b == board_class) return true;
        const auto g = list.groups.find(b);
        if (g != list.groups.end() && g->second.count(board) != 0) return true;
    }
    return false;
}

void run_jobs(std::vector<Job>& jobs) {
    std::vector<Job*> parallel, serial;
    for (Job& j : jobs) {
#ifdef _WIN32
        // No `env` command: a run with variables sets them in this process, one run at a time.
        (j.env.empty() ? parallel : serial).push_back(&j);
#else
        parallel.push_back(&j);
#endif
    }
    std::atomic<std::size_t> next{0};
    const unsigned workers = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < workers; ++i) {
        threads.emplace_back([&] {
            for (std::size_t j; (j = next.fetch_add(1)) < parallel.size();) run_job(*parallel[j]);
        });
    }
    for (auto& t : threads) t.join();
#ifdef _WIN32
    for (Job* j : serial) {
        std::vector<std::pair<std::string, std::optional<std::string>>> saved;
        for (const auto& [name, value] : j->env) {
            const char* was = std::getenv(name.c_str());
            saved.emplace_back(name, was ? std::optional<std::string>(was) : std::nullopt);
            _putenv_s(name.c_str(), value.c_str());
        }
        run_job(*j);
        for (const auto& [name, was] : saved) _putenv_s(name.c_str(), was ? was->c_str() : "");
    }
#endif
}

std::string differing_parts(const Job& a, const Job& b, std::set<std::string>* profiles = nullptr) {
    std::string out;
    int shown = 0;
    for (const auto& [key, hash] : a.parts) {
        const auto it = b.parts.find(key);
        if (it != b.parts.end() && it->second == hash) continue;
        if (profiles != nullptr) profiles->insert(key.substr(key.rfind(' ') + 1));
        if (++shown <= 6) out += (out.empty() ? "" : ", ") + key;
    }
    if (shown > 6) out += ", ... (" + std::to_string(shown) + " parts)";
    return out;
}

int process_id() {
#ifdef _WIN32
    return _getpid();
#else
    return static_cast<int>(::getpid());
#endif
}

int run_parent(const std::string& exe, bool update, const std::string& only) {
    KnownList known = read_known(NINFERNO_SELECTOR_PROPERTIES_KNOWN);

    // Scratch for the variables' files: an empty routes/ directory, an unreadable route table,
    // recorded shapes, and a dense-tune cache directory per board for the tune leg.
    const auto tmp = std::filesystem::temp_directory_path() /
                     ("ninferno_selector_properties_" + std::to_string(process_id()));
    std::filesystem::create_directories(tmp / "routes");
    std::filesystem::create_directories(tmp / "cache" / "ninferno");
    std::ofstream(tmp / "route_table.json") << "{\"schema\": \"ninferno-route-table/1\"}\n";
    const std::string tmp_dir = tmp.string();
    std::vector<std::pair<std::string, std::string>> all_env;
    for (const EnvVar& e : kEnvVars) all_env.emplace_back(e.name, expand(e.value, tmp_dir));

    // --only <text>: the boards whose name contains it (a focused run: a known violation it does
    // not see is not stale, and boards it leaves out need no digest).
    std::vector<const TestBoard*> boards;
    for (const TestBoard& b : kGoldenBoards)
        if (std::string_view(b.name).find(only) != std::string_view::npos) boards.push_back(&b);
    for (const TestBoard& b : kSyntheticBoards)
        if (std::string_view(b.name).find(only) != std::string_view::npos) boards.push_back(&b);
    const bool focused = !only.empty();

    std::vector<Job> jobs;
    for (const TestBoard* b : boards) {
        const std::string self = shell_quote(exe) + " --board " + b->name;
        const std::string tune = (tmp / "tune" / b->name).string();
        jobs.push_back({b, "plain", self + " 2>&1"});
        jobs.push_back({b, "clock:zero", self + " --no-repeat --clock zero 2>&1"});
        jobs.push_back({b, "clock:skew", self + " --no-repeat --clock skew 2>&1"});
        jobs.push_back({b, "env", self + " --no-repeat 2>&1", all_env});
        jobs.push_back(
            {b, "tune:off",
             self + " --no-repeat --tune off --tune-dir " + shell_quote(tune + "_off") + " 2>&1"});
        jobs.push_back(
            {b, "tune:on",
             self + " --no-repeat --tune on --tune-dir " + shell_quote(tune + "_on") + " 2>&1"});
    }
    run_jobs(jobs);

    // Observed violations: key -> board -> detail.
    std::map<std::string, std::map<std::string, std::string>> observed;
    std::map<std::string, std::string> board_class;
    int failures    = 0;
    const auto fail = [&](const std::string& what) {
        ++failures;
        std::cerr << "FAIL " << what << '\n';
    };
    std::map<std::string, const Job*> plain;
    std::size_t choices = 0;
    for (const Job& j : jobs) {
        if (j.status != 0 || j.checked.empty()) {
            fail(std::string(j.board->name) + " " + j.variant + ": the run failed (status " +
                 std::to_string(j.status) + ")\n" + j.other);
            continue;
        }
        if (j.variant == "plain") {
            plain[j.board->name]       = &j;
            board_class[j.board->name] = j.board_class;
            choices += std::stoull(j.checked);
        }
    }
    int tune_on_moved = 0;
    for (const Job& j : jobs) {
        const auto it = plain.find(j.board->name);
        if (j.status != 0 || it == plain.end()) continue;
        const Job& base         = *it->second;
        const std::string board = j.board->name;
        const auto note         = [&](const std::string& key, const std::string& detail) {
            auto& at = observed[key][board];
            at += (at.empty() ? "" : "; ") + detail;
        };
        if (j.variant == "plain") {
            for (const auto& v : j.violations) {
                const auto tab        = v.find('\t');
                const std::string key = v.substr(0, tab);
                const std::string d   = v.substr(tab + 1);
                // A board's first detail and the count, if any.
                if (observed[key][board].empty() || d.find(" in all)") != std::string::npos)
                    note(key, d);
            }
        } else if (j.variant == "env" || j.variant == "tune:off") {
            if (j.parts != base.parts) note("P2-" + j.variant, differing_parts(base, j));
        } else if (j.variant == "tune:on") {
            if (j.parts != base.parts) ++tune_on_moved; // allowed: a tuning run, tainted
        } else if (j.variant.rfind("clock:", 0) == 0) {
            std::set<std::string> profiles;
            const std::string parts = differing_parts(base, j, &profiles);
            for (const auto& p : profiles)
                note("P2-clock " + p, "(" + j.variant.substr(6) + ") " + parts);
        }
    }

    // P6: the digests checked in for this route set; every board evaluated needs its lines.
    const auto digest_path =
        std::filesystem::path(NINFERNO_SELECTOR_DIGEST_DIR) / (std::string(kBuildSet) + ".txt");
    if (update && focused) {
        fail("--update rewrites every board's digest: run it without --only");
    } else if (update) {
        std::ofstream out(digest_path, std::ios::binary);
        out << "# Selector choice digests, route set " << kBuildSet
            << " (test_selector_properties --update).\n"
            << "# <board> <format | *> <fnv1a-64 of the golden text> <lines>\n";
        for (const TestBoard* b : boards) {
            const auto it = plain.find(b->name);
            if (it == plain.end()) continue;
            for (const auto& line : it->second->digest) out << line << '\n';
        }
        if (!out.good()) fail("cannot write " + digest_path.string());
        std::cout << "wrote " << digest_path.string() << '\n';
    } else {
        std::map<std::string, std::vector<std::string>> want;
        std::ifstream in(digest_path);
        if (!in) fail("no digest file " + digest_path.string() + " (run with --update)");
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            want[line.substr(0, line.find(' '))].push_back(line);
        }
        for (const TestBoard* b : boards) {
            const auto it = plain.find(b->name);
            if (it == plain.end()) continue;
            const auto w = want.find(b->name);
            if (w == want.end()) {
                fail(std::string("P6 ") + b->name + ": no digest in " + digest_path.string() +
                     " (--update in a tree of this route set writes it)");
                continue;
            }
            if (w->second != it->second->digest) {
                std::string diff;
                for (const auto& l : it->second->digest) {
                    if (std::find(w->second.begin(), w->second.end(), l) == w->second.end()) {
                        diff += "\n  now: " + l;
                    }
                }
                fail(std::string("P6 ") + b->name + ": choices differ from " +
                     digest_path.string() + diff);
            }
        }
    }

    // Observed violations against the known list: each (violation, board) must be covered by an
    // entry of this route set, and each such entry must still cover one.
    for (const auto& [key, by_board] : observed) {
        std::string uncovered, all;
        int shown = 0;
        for (const auto& [board, detail] : by_board) {
            all += (all.empty() ? "" : " ") + board;
            bool covered = false;
            for (Known& k : known.entries) {
                if (k.key == key && applies(k) && covers(known, k, board, board_class[board])) {
                    k.seen  = true;
                    covered = true;
                }
            }
            if (!covered && ++shown <= 6) uncovered += "\n  " + board + ": " + detail;
        }
        std::cout << "seen " << key << " on " << by_board.size() << " boards: " << all << '\n';
        if (shown > 0) {
            fail("violation " + key + " on " + std::to_string(shown) + " boards no entry covers" +
                 uncovered);
        }
    }
    for (const Known& k : known.entries) {
        if (!focused && applies(k) && !k.seen) {
            fail("listed in " + std::string(NINFERNO_SELECTOR_PROPERTIES_KNOWN) +
                 " but not seen: " + k.key +
                 " (fixed: remove the entry and update the pillar's gap list)");
        }
    }
    std::cout << "P2-tune on: " << tune_on_moved
              << " boards moved (allowed: a tuning run is tainted in the fingerprint)\n";
    std::cout << "P5: no plan-level API yet; not checked\n";
    std::error_code ignored;
    std::filesystem::remove_all(tmp, ignored);
    std::cout << "selector properties: " << boards.size() << " boards, " << jobs.size() << " runs, "
              << choices << " choices per variant, route set " << kBuildSet << ": "
              << (failures == 0 ? "PASS" : "FAIL") << '\n';
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool update = false;
        ChildOptions child;
        std::string board, only;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--update") update = true;
            if (arg == "--no-repeat") child.repeat = false;
            if (arg == "--board" && i + 1 < argc) board = argv[++i];
            if (arg == "--clock" && i + 1 < argc) child.clock = argv[++i];
            if (arg == "--tune" && i + 1 < argc) child.tune = argv[++i];
            if (arg == "--tune-dir" && i + 1 < argc) child.tune_dir = argv[++i];
            if (arg == "--only" && i + 1 < argc) only = argv[++i];
        }
        if (!board.empty()) {
            const TestBoard* b = find_board(board);
            if (b == nullptr) {
                std::cerr << "unknown board " << board << '\n';
                return 2;
            }
            return run_child(*b, child);
        }
        return run_parent(argv[0], update, only);
    } catch (const std::exception& error) {
        std::cerr << "selector properties: " << error.what() << '\n';
        return 1;
    }
}
