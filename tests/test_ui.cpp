// ninferno-ui without a GPU or a network: who may talk to it, the launch command it builds, the
// catalog's estimates, the process pool, and the HTTP routes behind the access checks.

#include "artifact/fixture.h"
#include "convert/acquire.h"
#include "core/sys.h"
#include "spec/fingerprint.h"
#include "spec/profile.h"
#include "ui/access.h"
#include "ui/app.h"
#include "ui/browser.h"
#include "ui/chat_ui.h"
#include "ui/catalog.h"
#include "ui/processes.h"
#include "support/posix.h"
#include "support/temp_dir.h"

#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << std::endl;
        ++failures;
    }
}

// Paths built with std::filesystem (/ is \ between the parts on Windows), compared as paths.
bool same_path(const std::string& got, const std::string& want) {
    return std::filesystem::path(got) == std::filesystem::path(want);
}

using namespace ninferno::ui;

void test_access() {
    check(is_loopback_host("localhost"), "localhost is loopback");
    check(is_loopback_host("127.0.0.1"), "127.0.0.1");
    check(is_loopback_host("127.1.2.3"), "127/8");
    check(is_loopback_host("[::1]") && is_loopback_host("::1"), "::1");
    check(!is_loopback_host("127.0.0.1.evil.com"), "rebinding name is not loopback");
    check(!is_loopback_host("127.0.0"), "three parts");
    check(!is_loopback_host("0.0.0.0"), "any address");
    check(host_of_authority("[::1]:8090") == "::1", "bracketed v6");
    check(host_of_authority("LocalHost:8090") == "localhost", "lower-cased, port dropped");
    check(host_of_authority("::1") == "::1" && is_loopback_host(host_of_authority("::1")) &&
              host_of_authority("[::1]:8090") == "::1" && host_of_authority("[::1]") == "::1" &&
              host_of_authority("FE80::1") == "fe80::1" &&
              !is_loopback_host(host_of_authority("fe80::1")) &&
              host_of_authority("127.0.0.1:8090") == "127.0.0.1",
          "bare v6 listen address is a host, not host:port");

    AccessPolicy local;
    check(host_allowed(local, "localhost:8090"), "local: localhost");
    check(host_allowed(local, "127.0.0.1:8090"), "local: 127.0.0.1");
    check(!host_allowed(local, "evil.example:8090"), "local: rebinding refused");
    check(!host_allowed(local, ""), "local: no Host refused");
    const AccessPolicy keyed_local{.listen = "127.0.0.1", .port = 8090, .key = "0123456789abcdef"};
    check(host_allowed(keyed_local, "ui.example.com"),
          "keyed loopback behind a proxy: the public Host passes on to the key check");
    check(check_key(keyed_local, "127.0.0.1", "ui.example.com", false, "", "", "") ==
              KeyCheck::refused,
          "keyed loopback: a rebound name still needs the key");
    AccessPolicy open{.listen = "0.0.0.0", .port = 8090};
    check(host_allowed(open, "192.168.1.5:8090"), "exposed: any name");

    check(origin_allowed("", "localhost:8090"), "no Origin (curl) allowed");
    check(origin_allowed("http://localhost:8090", "localhost:8090"), "same origin");
    check(!origin_allowed("http://localhost:8091", "localhost:8090"), "other port");
    check(!origin_allowed("https://evil.example", "localhost:8090"), "other site");
    check(!origin_allowed("null", "localhost:8090"), "opaque origin");
    check(origin_refusal("https://ui.example", "127.0.0.1:8090", true)
                      .find("the proxy must forward the browser's Host header") !=
                  std::string::npos &&
              origin_refusal("https://ui.example", "127.0.0.1:8090", false) ==
                  "cross-origin request refused" &&
              origin_refusal("null", "127.0.0.1:8090", true) == "cross-origin request refused",
          "a proxy that rewrote Host is named in the refusal");

    const std::string key = new_access_key();
    check(key.size() == 32 && key.find_first_not_of("0123456789abcdef") == std::string::npos,
          "key: 32 hex digits");
    check(new_access_key() != key, "key: a new one each time");
    check(usable_key(key) && usable_key("A-b_c.d~e0123456789") && !usable_key("short-key") &&
              !usable_key("sixteen chars;x&y#z") && !usable_key("0123456789abcdef\r") &&
              !usable_key(std::string(257, 'k')),
          "key: URL- and cookie-safe characters only");
    check(forwarded_https("", "for=1.2.3.4;proto=\"https\" ;by=proxy") &&
              forwarded_https("", "proto=\"https\"\t, for=x"),
          "Forwarded: whitespace after a quoted proto");
    check(forwarded_https("https", "") && forwarded_https("HTTPS, http", "") &&
              forwarded_https("", "for=1.2.3.4;proto=https") && !forwarded_https("http", "") &&
              !forwarded_https("", ""),
          "key: HTTPS seen through the proxy");
    check(forwarded_https(" https ", "") && forwarded_https("", "for=1.2.3.4;proto=\"https\"") &&
              forwarded_https("", "for=\"[2001:db8::1]:443\"; Proto=HTTPS;by=x") &&
              forwarded_https("", "PROTO=\"HTTPS\", proto=http") &&
              forwarded_https("", "for=\"a,b;proto=http\";proto=https") &&
              forwarded_https("", "proto=\"ht\\tps\""),
          "Forwarded: quoted values, any case, separators inside quotes");
    check(!forwarded_https("", "proto=http, proto=https") &&
              !forwarded_https("", "for=\"x;proto=https\"") &&
              !forwarded_https("", "for=1.2.3.4;proto=httpsx") &&
              !forwarded_https("", "xproto=https") && !forwarded_https("", "for=proto=https"),
          "Forwarded: only the first element's proto parameter");
    check(cookie_value("a=1; ninferno_ui_key=k1; b=2", "ninferno_ui_key") == "k1", "cookie found");
    // The kept key: made once, kept, an unusable one replaced, and one key for starters at once.
    {
        namespace fs = std::filesystem;
        const auto dir =
            fs::temp_directory_path() / ("ninferno_ui_key_" + std::to_string(::getpid()));
        fs::remove_all(dir);
        const auto path        = dir / "ninferno-ui" / "access-key";
        bool kept              = false;
        const std::string made = kept_access_key(path, &kept);
        check(usable_key(made) && kept && key_file_line(path) == made, "kept key: made and kept");
        check(kept_access_key(path) == made, "kept key: the same at the next start");
        std::ofstream(path, std::ios::trunc) << "0123456789abcdef0123\n";
        check(kept_access_key(path) == "0123456789abcdef0123", "kept key: another one written");
        std::ofstream(path, std::ios::trunc) << "short\n";
        const std::string replaced = kept_access_key(path, &kept);
        check(usable_key(replaced) && kept && replaced != made && kept_access_key(path) == replaced,
              "kept key: an unusable one replaced once");
        for (const char* start : {"", "bad;key\n"}) {
            fs::remove(path);
            if (*start != '\0') { std::ofstream(path) << start; }
            std::vector<std::string> keys(8);
            std::vector<std::thread> starters;
            for (auto& key : keys) {
                starters.emplace_back([&key, &path] { key = kept_access_key(path); });
            }
            for (auto& starter : starters) { starter.join(); }
            check(usable_key(keys[0]) && std::count(keys.begin(), keys.end(), keys[0]) == 8 &&
                      key_file_line(path) == keys[0],
                  std::string("kept key: starters at once agree, ") +
                      (*start != '\0' ? "an unusable file" : "no file"));
        }
        fs::remove_all(dir);
    }
    check(cookie_value("xninferno_ui_key=k1", "ninferno_ui_key").empty(), "cookie name exact");
    AccessPolicy keyed{.listen = "0.0.0.0", .port = 8090, .key = key};
    const std::string cookie = "ninferno_ui_key_8090=" + key, bearer = "Bearer " + key;
    const auto keyed_check = [&](std::string_view peer, std::string_view host, bool proxied,
                                 std::string_view c, std::string_view a, std::string_view q) {
        return check_key(keyed, peer, host, proxied, c, a, q);
    };
    check(check_key(local, "10.0.0.2", "x:1", false, "", "", "") == KeyCheck::pass, "no key: all pass");
    check(check_key(local, "127.0.0.1", "ui.example", true, "", "", "") == KeyCheck::keyless_proxy,
          "no key behind a proxy: refused, the proxy would publish the page");
    check(keyed_check("127.0.0.1", "localhost:8090", false, "", "", "") == KeyCheck::pass,
          "this machine passes");
    check(keyed_check("::ffff:127.0.0.1", "127.0.0.1:8090", false, "", "", "") == KeyCheck::pass,
          "mapped loopback");
    check(keyed_check("127.0.0.1", "evil.example:8090", false, "", "", "") == KeyCheck::refused,
          "rebound name refused");
    check(keyed_check("127.0.0.1", "127.0.0.1:8090", true, "", "", "") == KeyCheck::refused,
          "local proxy refused");
    check(keyed_check("10.0.0.2", "box:8090", false, "", "", "") == KeyCheck::refused,
          "other machine refused");
    check(keyed_check("10.0.0.2", "box:8090", false, cookie, "", "") == KeyCheck::pass, "cookie");
    check(keyed_check("10.0.0.2", "box:8090", false, "ninferno_ui_key=" + key, "", "") ==
              KeyCheck::refused,
          "another port's or an unported cookie is not this UI's");
    check(key_cookie(8091) == "ninferno_ui_key_8091", "cookie named by port");
    check(keyed_check("10.0.0.2", "box:8090", false, "", bearer, "") == KeyCheck::pass, "bearer");
    check(keyed_check("10.0.0.2", "box:8090", false, "", "BEARER " + key, "") == KeyCheck::pass,
          "bearer: the scheme in any case");
    check(keyed_check("10.0.0.2", "box:8090", false, "", "Bearer   " + key, "") == KeyCheck::pass,
          "bearer: several spaces before the credential");
    check(keyed_check("10.0.0.2", "box:8090", false, "", "", key) == KeyCheck::pass_set_cookie,
          "?key= sets it");
    check(keyed_check("10.0.0.2", "box:8090", false, "ninferno_ui_key_8090=x", "Bearer x", "x") ==
              KeyCheck::refused,
          "wrong key refused");
    check(keyed_check("", "box:8090", false, "", "", "") == KeyCheck::refused,
          "no peer address refused");
}

// The hub cache's directory names, both ways, as the page lists and deletes converted models.
void test_hub_directories() {
    using ninferno::convert::hub_repo_directory;
    using ninferno::convert::hub_repo_of_directory;
    check(hub_repo_directory("Qwen/Qwen3.6-27B") == "models--Qwen--Qwen3.6-27B",
          "repo to directory");
    check(hub_repo_of_directory("models--Qwen--Qwen3.6-27B") == "Qwen/Qwen3.6-27B",
          "directory to repo");
    check(hub_repo_of_directory(hub_repo_directory("org/a--b")) == "org/a--b",
          "a name with -- comes back whole");
    check(hub_repo_of_directory("datasets--org--name").empty(), "another kind is no repository");
}

void test_launch() {
    check(split_model_tag("unsloth/Qwen3.5-9B-GGUF:Q4_K_M") ==
              std::pair<std::string, std::string>{"unsloth/Qwen3.5-9B-GGUF", "Q4_K_M"},
          "repo:quant split");
    check(split_model_tag("Qwen/Qwen3.5-9B").second.empty(), "no tag");
    check(split_model_tag("/models/a:b.ninferno").second.empty(), "absolute path kept whole");

    UiOptions options;
    options.serve_binary = "/opt/ninferno/bin/ninferno-serve";
    LaunchRequest request;
    request.model                       = "unsloth/Qwen3.5-9B-GGUF:Q4_K_M";
    request.gpu                         = 1;
    request.port                        = 8101;
    request.max_context                 = 32768;
    request.spec                        = "mtp";
    request.precision                   = "exact";
    {
        // Under Auto the planned own context is the automatic context's ceiling.
        LaunchRequest capped   = request;
        capped.max_context     = 0;
        capped.max_context_cap = 49152;
        const auto capped_argv = make_launch(options, capped).argv;
        const auto at = std::find(capped_argv.begin(), capped_argv.end(), "--max-context-cap");
        check(at != capped_argv.end() && at + 1 != capped_argv.end() && *(at + 1) == "49152" &&
                  std::find(capped_argv.begin(), capped_argv.end(), "--max-context") ==
                      capped_argv.end(),
              "a planned context launches as --max-context-cap");
        LaunchRequest both   = request;
        both.max_context_cap = 49152;
        bool refused         = false;
        try {
            (void)make_launch(options, both);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "max_context with max_context_cap is refused");
    }
    const auto spec                     = make_launch(options, request);
    const std::vector<std::string> want = {"/opt/ninferno/bin/ninferno-serve",
                                           "unsloth/Qwen3.5-9B-GGUF",
                                           "--quant",
                                           "Q4_K_M",
                                           "--host",
                                           "127.0.0.1",
                                           "--port",
                                           "8101",
                                           "--max-context",
                                           "32768",
                                           "--kv-dtype",
                                           "bf16",
                                           "--activations",
                                           "exact",
                                           "--spec",
                                           "mtp"};
    check(spec.argv == want, "launch argv");
    check(spec.env.at("CUDA_VISIBLE_DEVICES") == "1" &&
              spec.env.at("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
          "GPU binding by PCI order");
    check(spec.model == "unsloth/Qwen3.5-9B-GGUF:Q4_K_M", "display name");

    // No speculation named: the published profile's choice, --spec auto (ninferno-serve's default
    // too), named on the command line.
    {
        LaunchRequest unnamed = request;
        unnamed.spec          = "";
        const auto argv       = make_launch(options, unnamed).argv;
        const auto at         = std::find(argv.begin(), argv.end(), "--spec");
        check(at != argv.end() && at + 1 != argv.end() && *(at + 1) == "auto",
              "a launch without a named speculation runs --spec auto");
    }

    // A tune beside a model started by its artifact path is a measurement of this machine: never
    // named by itself, even one that would serve (a note says so); named when the start asks for
    // it (spec_profile), and then only one that serves on the request's GPU: one that does not
    // refuses the start, as ninferno-serve refuses a named tune that does not serve. On a GPU
    // whose class NVML does not name, only the tune itself is checked here; the server judges it.
    {
        const ninferno::test::artifact_fixture::Fixture fixture;
        fixture.write();
        const auto tune = ninferno::spec::default_profile_path(fixture.entry);
        ninferno::spec::Profile tuned;
        tuned.key.artifact_id = ninferno::spec::artifact_id_hex(fixture.entry);
        tuned.key.gpu         = "gb202:170sm";
        ninferno::spec::write_profile(tune, tuned);
        GpuStat rtx5090;
        rtx5090.index         = 1;
        rtx5090.cc_major      = 12;
        rtx5090.cc_minor      = 0;
        rtx5090.pci_device_id = 0x2B85;
        rtx5090.sms           = 170;

        LaunchRequest local = request;
        local.model         = ninferno::utf8_path(fixture.entry);
        local.prepare_only  = false;

        const auto named = [](const LaunchSpec& launch) {
            const auto at = std::find(launch.argv.begin(), launch.argv.end(), "--spec-profile");
            if (at == launch.argv.end() || at + 1 == launch.argv.end()) { return std::string(); }
            return *(at + 1);
        };
        const auto refusal = [&](const LaunchRequest& asked, const GpuStat* on) {
            try {
                (void)make_launch(options, asked, on);
            } catch (const std::invalid_argument& error) { return std::string(error.what()); }
            return std::string();
        };
        const auto unasked = make_launch(options, local, &rtx5090);
        check(named(unasked).empty() && unasked.notes.size() == 1 &&
                  unasked.notes[0].find("\"spec_profile\": true") != std::string::npos,
              "a tune that would serve is not named by itself: a note says how a start asks");
        local.spec_profile = true;
        const auto asked   = make_launch(options, local, &rtx5090);
        check(same_path(named(asked), ninferno::utf8_path(tune)) && asked.notes.empty(),
              "a start that asks for the tune names it");
        GpuStat other = rtx5090;
        other.sms     = 188;
        check(refusal(local, &other).find("another") != std::string::npos,
              "asked for, a tune of another GPU class refuses the start");
        GpuStat mig = rtx5090;
        mig.sms     = 0;
        check(same_path(named(make_launch(options, local, nullptr)), ninferno::utf8_path(tune)) &&
                  same_path(named(make_launch(options, local, &mig)), ninferno::utf8_path(tune)),
              "asked for on a GPU whose class is unknown here, the server judges the tune");
        LaunchRequest hub = local;
        hub.model         = "org/m";
        check(refusal(hub, &rtx5090).find(".ninferno path") != std::string::npos,
              "asked for, a model not started by its artifact path refuses the start");
        std::ofstream(tune) << "{}";
        check(!refusal(local, &rtx5090).empty() && !refusal(local, nullptr).empty(),
              "asked for, a stale tune refuses the start, on an unknown GPU too");
        std::filesystem::remove(tune);
        check(refusal(local, &rtx5090).find("no profile") != std::string::npos,
              "asked for, a missing tune refuses the start");
        local.spec_profile = false;
        check(make_launch(options, local, &rtx5090).notes.empty(),
              "no tune beside the model: no note");
    }

    // A tune beside the model serves only on the GPU class it was made on, as the server keys it
    // ("<chip>:<sms>sm"); on a GPU unknown here or without the SM count (a MIG slice), only the
    // tune itself is checked: the server names its GPU's class from CUDA and judges the rest.
    {
        namespace fs   = std::filesystem;
        const TempDir scratch("ninferno_ui_tune_gpu");
        const auto& dir = scratch.path();
        const auto tune = dir / "m.ninferno.spec_profile.json";
        ninferno::spec::Profile tuned;
        tuned.key.artifact_id = "aaaa";
        tuned.key.gpu         = "gb202:170sm";
        ninferno::spec::write_profile(tune, tuned);
        GpuStat rtx5090;
        rtx5090.index         = 1;
        rtx5090.cc_major      = 12;
        rtx5090.cc_minor      = 0;
        rtx5090.pci_device_id = 0x2B85;
        rtx5090.sms           = 170;
        check(tune_refusal_on(tune, "aaaa", &rtx5090).empty(), "a tune made on this GPU's class");
        check(!tune_refusal_on(tune, "bbbb", &rtx5090).empty(), "a tune of another artifact");
        GpuStat other = rtx5090;
        other.sms     = 188;
        check(tune_refusal_on(tune, "aaaa", &other).find("another") != std::string::npos,
              "a tune of another SM count of the chip is refused");
        other.pci_device_id = 0x2C02; // RTX 5080: gb203
        other.sms           = 0;
        check(tune_refusal_on(tune, "aaaa", &other).empty() &&
                  !tune_refusal_on(tune, "bbbb", &other).empty(),
              "without the SM count, only the tune itself is checked here");
        other.sms      = 84;
        other.cc_major = 0;
        check(tune_refusal_on(tune, "aaaa", nullptr).empty() &&
                  tune_refusal_on(tune, "aaaa", &other).empty() &&
                  !tune_refusal_on(tune, "bbbb", nullptr).empty(),
              "on a GPU unknown here, only the tune itself is checked here");
        // The GPU a launch runs on: by NVML's index; none named, the only one, else none.
        std::vector<GpuStat> stats(2);
        stats[0].index = 0;
        stats[1].index = 1;
        check(launch_gpu(stats, 1) == &stats[1] && launch_gpu(stats, 2) == nullptr,
              "a launch's GPU by its index");
        check(launch_gpu(stats, -1) == nullptr, "several GPUs and none named: none known");
        stats.resize(1);
        check(launch_gpu(stats, -1) == &stats[0], "one GPU and none named: that one");
    }

    request.prepare_only = true;
    const auto prepare   = make_launch(options, request);
    check(prepare.argv.back() == "--prepare-only" && prepare.port == 0, "prepare-only");

    const auto refused = [&](LaunchRequest bad, const char* what) {
        try {
            (void)make_launch(options, bad);
            check(false, what);
        } catch (const std::invalid_argument&) {}
    };
    // Calibration: the engine family picks the binary beside ninferno-serve.
    {
        const TempDir scratch("ninferno_ui_test");
        const auto artifact = scratch.path() / "ninferno_ui_test.ninferno";
        std::ofstream(artifact) << "x";
        const auto qwen = make_tune(
            options, {.artifact = ninferno::utf8_path(artifact), .engine = "qwen3_5", .gpu = 0});
        check(qwen.argv.size() >= 3 && same_path(qwen.argv[0], "/opt/ninferno/bin/ninferno") &&
                  qwen.argv[1] == "spec-tune" && qwen.argv[2] == ninferno::utf8_path(artifact) &&
                  qwen.kind == "tune",
              "tune: ninferno spec-tune");
        const auto gemma = make_tune(
            options, {.artifact = ninferno::utf8_path(artifact), .engine = "gemma4", .gpu = 0});
        check(same_path(gemma.argv[0], "/opt/ninferno/bin/ninferno-gemma4") &&
                  gemma.argv[2] == "spec-tune",
              "tune: ninferno-gemma4 <artifact> spec-tune");
        try {
            (void)make_tune(options, {.artifact = "/etc/passwd", .engine = "", .gpu = 0});
            check(false, "tune: a non-artifact accepted");
        } catch (const std::invalid_argument&) {}
    }

    LaunchRequest bad = request;
    bad.model         = "--help";
    refused(bad, "option injection as model");
    bad.model = "org/m\" --host 0.0.0.0";
    refused(bad, "a quote in the model name");
    bad.model = "org/m\n";
    refused(bad, "a newline in the model name");
    bad       = request;
    bad.quant = "Q4;rm -rf";
    refused(bad, "bad quant");
    bad      = request;
    bad.spec = "eagle9";
    refused(bad, "bad spec");
    bad           = request;
    bad.precision = "fp2";
    refused(bad, "bad precision");
}

void test_catalog() {
    const auto& catalog = Catalog::builtin();
    check(catalog.models().size() >= 7, "catalog has the target models");
    for (const auto& model : catalog.models()) {
        for (const auto& source : model.sources) {
            check(source.bits <= 8, model.id + ": at most 8 bits");
        }
    }
    const CatalogModel* qwen = nullptr;
    for (const auto& model : catalog.models()) {
        if (model.id == "qwen3.6-35b-a3b") { qwen = &model; }
    }
    check(qwen != nullptr, "Qwen3.6 35B listed");
    if (qwen == nullptr) { return; }

    // RTX 4090 as NVML reports it: measured numbers, GGUF preferred (no FP4).
    TargetGpu ada{.name          = "NVIDIA GeForce RTX 4090",
                  .memory_gib    = 23.99,
                  .bandwidth_gbs = 1008,
                  .fp4           = false};
    SizeTable sizes{{size_key("unsloth/Qwen3.6-35B-A3B-GGUF", "UD-Q4_K_M"), 41ULL << 29},
                    {size_key("RedHatAI/Qwen3.6-35B-A3B-NVFP4", ""), 21ULL << 30}};
    auto e = estimate(catalog, *qwen, ada, sizes);
    check(e.preferred.has_value(), "4090: something fits");
    if (e.preferred) {
        const auto& s = e.sources[*e.preferred];
        check(qwen->sources[s.source].format == "gguf", "4090 prefers GGUF");
        check(s.tps_basis == "measured" && s.tps > 300, "4090 speed is the measurement");
        check(s.size_known && s.fit != "no", "size from listing");
    }
    // RTX 5090: NVFP4 preferred.
    TargetGpu blackwell{
        .name = "NVIDIA GeForce RTX 5090", .memory_gib = 31.4, .bandwidth_gbs = 1792, .fp4 = true};
    e = estimate(catalog, *qwen, blackwell, sizes);
    check(e.preferred && qwen->sources[e.sources[*e.preferred].source].format == "nvfp4",
          "5090 prefers NVFP4");
    check(e.recommended, "5090: recommended");
    // No listing: each size is estimated from the parameters and bits (35B at 2 to 8 bits).
    e = estimate(catalog, *qwen, blackwell, SizeTable{});
    for (const auto& s : e.sources) {
        check(!s.size_known && s.size_gib > 5 && s.size_gib < 45, "size estimated from parameters");
    }
    // An unmeasured card: scaled from the nearest measured bandwidth.
    TargetGpu other{
        .name = "NVIDIA RTX A6000", .memory_gib = 47.5, .bandwidth_gbs = 768, .fp4 = false};
    e = estimate(catalog, *qwen, other, sizes);
    check(e.preferred && e.sources[*e.preferred].tps_basis.rfind("scaled from", 0) == 0,
          "scaled estimate");
    // Speculation: the 5090 (gb202) has its own profiles; the 3090 (ga102) had none of its own in
    // the registry when this was written (a later own profile makes it "own", also right).
    TargetGpu gb202     = blackwell;
    gb202.chip          = "gb202";
    gb202.profile_chain = {"gb202:170sm", "gb202", "sm120", "*"};
    const auto own      = speculation_plan("qwen3.6-35b-a3b", gb202);
    check(own.profile == "own" && own.profile_class == "gb202", "5090: own profile");
    check(std::find(own.drafters.begin(), own.drafters.end(), "dflash") != own.drafters.end(),
          "Qwen3.6 drafts with DFlash");
    TargetGpu ga102{.name = "NVIDIA GeForce RTX 3090", .memory_gib = 24, .bandwidth_gbs = 936};
    ga102.chip          = "ga102";
    ga102.profile_chain = {"ga102:82sm", "ga102", "sm86", "*", "ad102", "gb202"};
    const auto borrowed = speculation_plan("qwen3.6-35b-a3b", ga102);
    check(borrowed.profile != "none", "3090: some profile");
    check(speculation_plan("no-such-model", ga102).profile == "none", "unknown model: none");
    // The profile chain takes the board's mma_slow as the device query does: a GTX 1660 SUPER on
    // TU106 (0x1F09, no tensor cores by the PCI-ID table) borrows only TU116 / TU117, an RTX 2070
    // on the same chip (0x1F07) only the tensor-core chips.
    const auto has = [](const std::vector<std::string>& chain, const char* key) {
        return std::find(chain.begin(), chain.end(), key) != chain.end();
    };
    GpuStat gtx_stat;
    gtx_stat.cc_major      = 7;
    gtx_stat.cc_minor      = 5;
    gtx_stat.pci_device_id = 0x1F09;
    gtx_stat.sms           = 22;
    const TargetGpu gtx    = target_of(gtx_stat);
    check(gtx.chip == "tu106" && has(gtx.profile_chain, "tu116") &&
              has(gtx.profile_chain, "tu117") && !has(gtx.profile_chain, "tu104") &&
              !has(gtx.profile_chain, "tu102"),
          "GTX on TU106: the profile chain borrows no tensor-core chip");
    GpuStat rtx_stat       = gtx_stat;
    rtx_stat.pci_device_id = 0x1F07;
    rtx_stat.sms           = 36;
    const TargetGpu rtx    = target_of(rtx_stat);
    check(rtx.chip == "tu106" && has(rtx.profile_chain, "tu104") &&
              !has(rtx.profile_chain, "tu116"),
          "RTX on TU106: the profile chain borrows no chip without tensor cores");

    // A 16 GB card: no listed source of Qwen3.6 35B fits, the listing's largest 2-4 bit quant
    // that does is added, and the model becomes recommended.
    TargetGpu gb203{
        .name = "NVIDIA GeForce RTX 5080", .memory_gib = 15.9, .bandwidth_gbs = 960, .fp4 = true};
    std::map<std::string, nlohmann::json> listings;
    listings["unsloth/Qwen3.6-35B-A3B-GGUF"] = nlohmann::json::array(
        {{{"quant", "UD-Q4_K_M"}, {"bytes", 22.0 * (1ULL << 30)}, {"bits", 4}, {"drafter", false}},
         {{"quant", "UD-IQ3_XXS"}, {"bytes", 12.3 * (1ULL << 30)}, {"bits", 3}, {"drafter", false}},
         {{"quant", "UD-IQ2_M"}, {"bytes", 10.7 * (1ULL << 30)}, {"bits", 2}, {"drafter", false}},
         {{"quant", "mtp-x"}, {"bytes", 1.0 * (1ULL << 30)}, {"bits", 4}, {"drafter", true}}});
    const auto fitted = with_fitting_quant(*qwen, gb203, listings);
    check(fitted.sources.size() == qwen->sources.size() + 1 &&
              fitted.sources.back().quant == "UD-IQ3_XXS",
          "16 GB: the largest fitting quant is added");
    SizeTable small_sizes = sizes;
    small_sizes[size_key("unsloth/Qwen3.6-35B-A3B-GGUF", "UD-IQ3_XXS")] =
        static_cast<std::uint64_t>(12.3 * (1ULL << 30));
    small_sizes[size_key("unsloth/Qwen3.6-35B-A3B-GGUF", "UD-Q4_K_M")] = 22ULL << 30;
    check(estimate(catalog, fitted, gb203, small_sizes).recommended, "16 GB: recommended with IQ3");
    check(with_fitting_quant(*qwen, blackwell, listings).sources.size() == qwen->sources.size(),
          "32 GB: nothing added when a source fits");
    // The plan without the weights decides where it can say: it refuses IQ3 and keeps IQ2 (the
    // weights against the budget would take IQ3), and where it cannot (nullopt) the budget stands.
    const QuantFits planned = [](const std::string&,
                                 const nlohmann::json& q) -> std::optional<bool> {
        const std::string label = q.value("quant", std::string());
        if (label == "UD-IQ3_XXS") { return false; }
        if (label == "UD-IQ2_M") { return true; }
        return std::nullopt;
    };
    const auto by_plan = with_fitting_quant(*qwen, gb203, listings, planned);
    check(by_plan.sources.size() == qwen->sources.size() + 1 &&
              by_plan.sources.back().quant == "UD-IQ2_M",
          "16 GB: the plan's fit decides which quant is added");
    const QuantFits unknown = [](const std::string&, const nlohmann::json&) -> std::optional<bool> {
        return std::nullopt;
    };
    check(with_fitting_quant(*qwen, gb203, listings, unknown).sources.back().quant == "UD-IQ3_XXS",
          "16 GB: without the plan's word the weights against the budget decide");
    // The rule's pick, where the repository is planned, is the quant the catalog adds.
    const QuantPickOf rule = [](const std::string&) -> std::optional<std::string> {
        return std::string("UD-IQ2_M");
    };
    check(with_fitting_quant(*qwen, gb203, listings, unknown, rule).sources.back().quant ==
              "UD-IQ2_M",
          "16 GB: the rule's pick is added");

    // The chat proxy names the served model when the page sends none, and keeps one it sends.
    check(first_model_id(R"({"object":"list","data":[{"id":"qwen3.5-4b","object":"model"}]})") ==
              "qwen3.5-4b",
          "first model id read");
    check(first_model_id("not json").empty() && first_model_id(R"({"data":[]})").empty(),
          "no model id from a bad or empty list");
    check(nlohmann::json::parse(with_model(R"({"messages":[]})", "m1"))["model"] == "m1",
          "the served model is named");
    check(nlohmann::json::parse(with_model(R"({"model":"x","messages":[]})", "m1"))["model"] == "x",
          "a named model is kept");
    bool not_object = false;
    try {
        (void)with_model("[1]", "m1");
    } catch (const std::invalid_argument&) { not_object = true; }
    check(not_object, "a non-object chat body refused");

    // The chat window: bundle paths, ./props from /v1/models, the artifact a serve process runs.
    check(safe_bundle_path("") && safe_bundle_path("_app/immutable/bundle.x.js") &&
              safe_bundle_path("favicon.svg"),
          "bundle paths accepted");
    check(!safe_bundle_path("../etc/passwd") && !safe_bundle_path("a/../../b") &&
              !safe_bundle_path("/etc/passwd") && !safe_bundle_path("a\\b") &&
              !safe_bundle_path("./x") && !safe_bundle_path("c:x"),
          "bundle paths outside refused");
    check(bundle_mime("a/b.js").starts_with("text/javascript") &&
              bundle_mime("i.png") == "image/png" &&
              bundle_mime("x.bin") == "application/octet-stream",
          "bundle types");
    const auto served =
        chat_model_of(R"({"object":"list","data":[{"id":"qwen3.5-4b","max_model_len":32768}]})");
    check(served.id == "qwen3.5-4b" && served.context == 32768, "chat model from /v1/models");
    ChatModel chat{.id = "qwen3.5-4b", .context = 32768, .slots = 4, .vision = true};
    const auto props = chat_props(chat);
    check(props["default_generation_settings"]["n_ctx"] == 32768 && props["total_slots"] == 4 &&
              props["modalities"]["vision"] == true && props["role"] == "model" &&
              props["model_path"] == "qwen3.5-4b",
          "chat props");
    check(!artifact_has_vision("/no/such.ninferno"), "an unreadable artifact takes no images");

    // The browser: WSL hands the page to Windows, a Linux desktop to xdg-open, no display opens none.
    {
        using ninferno::ui::BrowserHost;
        using ninferno::ui::browser_command;
        const auto host = [](std::string display, std::vector<std::string> programs, bool wsl) {
            BrowserHost h;
            h.env = [display](const char* name) {
                return std::string(name) == "DISPLAY" ? display : std::string();
            };
            h.has_program = [programs](const std::string& name) {
                return std::find(programs.begin(), programs.end(), name) != programs.end();
            };
            h.wsl = wsl;
            return h;
        };
        const std::string url = "http://localhost:8090/";
        check(browser_command(host("", {"wslview"}, true), url) ==
                      std::vector<std::string>{"wslview", url} &&
                  browser_command(host("", {"/mnt/c/Windows/System32/cmd.exe"}, true), url).at(0) ==
                      "/mnt/c/Windows/System32/cmd.exe" &&
                  browser_command(host(":0", {"xdg-open"}, false), url) ==
                      std::vector<std::string>{"xdg-open", url} &&
                  browser_command(host("", {"xdg-open"}, false), url).empty() &&
                  browser_command(host(":0", {}, false), url).empty(),
              "the browser command by host");
    }

    // My Models: names, sizes, architectures.
    {
        const auto gguf = artifact_name("Qwen3.6-35B-A3B-UD-Q4_K_M.qwen3_5_gguf.ninferno");
        check(gguf.stem == "Qwen3.6-35B-A3B-UD-Q4_K_M" && gguf.recipe == "qwen3_5_gguf" &&
                  gguf.quant == "UD-Q4_K_M",
              "artifact name: GGUF conversion");
        const auto native = artifact_name("qwen3_5_4b_nvfp4.ninferno");
        check(native.stem == "qwen3_5_4b_nvfp4" && native.recipe.empty() && native.quant == "NVFP4",
              "artifact name: a native checkpoint's");
        const auto dotted = artifact_name("Qwen3.5-9B-Q8_0.ninferno");
        check(dotted.stem == "Qwen3.5-9B-Q8_0" && dotted.recipe.empty() && dotted.quant == "Q8_0",
              "artifact name: a dot inside the stem is not a recipe");
        check(artifact_name("qwen3_8_27b_nvfp4_dflash2_v3.ninferno").quant == "NVFP4" &&
                  artifact_name("qwen3_6_35b_a3b_v3.ninferno").quant.empty(),
              "artifact name: a native artifact's format by name");
        check(params_label("unsloth/Qwen3.6-35B-A3B-GGUF") == "35B-A3B" &&
                  params_label("gemma-4-26B-A4B-it-qat") == "26B-A4B" &&
                  params_label("Qwen3.5-9B-Q8_0") == "9B" &&
                  params_label("gpt-oss-20b-MXFP4") == "20B" && params_label("Qwen3.5").empty() &&
                  params_label("qwen3_6_35b_a3b_v3") == "35B-A3B",
              "params labels");
        check(short_architecture("Qwen3_5MoeForCausalLM") == "qwen3_5moe" &&
                  short_architecture("Gemma4ForConditionalGeneration") == "gemma4" &&
                  short_architecture("NemotronHForCausalLM") == "nemotronh",
              "short architectures");
    }
    // Only converted files under the hub cache's ninferno/ or a models dir may be deleted.
    {
        namespace fs = std::filesystem;
        const auto root =
            fs::temp_directory_path() / ("ninferno_ui_del_" + std::to_string(::getpid()));
        fs::create_directories(root / "hub/models--org--m/ninferno/abc");
        fs::create_directories(root / "hub/models--org--m/snapshots/abc");
        fs::create_directories(root / "models/x");
        for (const auto& f :
             {root / "hub/models--org--m/ninferno/abc/m.ninferno",
              root / "hub/models--org--m/snapshots/abc/m.ninferno", root / "models/x/a.ninferno",
              root / "outside.ninferno", root / "models/x/a.gguf"}) {
            std::ofstream(f) << "x";
        }
        const std::vector<fs::path> dirs{root / "models"};
        check(deletable_artifact(root / "hub/models--org--m/ninferno/abc/m.ninferno", root / "hub",
                                 dirs) &&
                  deletable_artifact(root / "models/x/a.ninferno", root / "hub", dirs),
              "converted files deletable");
        check(!deletable_artifact(root / "hub/models--org--m/snapshots/abc/m.ninferno", root / "hub",
                                  dirs) &&
                  !deletable_artifact(root / "outside.ninferno", root / "hub", dirs) &&
                  !deletable_artifact(root / "models/x/a.gguf", root / "hub", dirs) &&
                  !deletable_artifact(root / "models/x/missing.ninferno", root / "hub", dirs),
              "anything else is not");
        fs::remove_all(root);
    }
    // A converted model whose download was removed is found at its ref's commit, by quantization.
    {
        namespace fs = std::filesystem;
        const auto root =
            fs::temp_directory_path() / ("ninferno_ui_kept_" + std::to_string(::getpid()));
        const auto repo = root / "models--org--m-GGUF";
        fs::create_directories(repo / "refs");
        fs::create_directories(repo / "ninferno/c0ffee");
        std::ofstream(repo / "refs/main") << "c0ffee\n";
        for (const char* name : {"m-Q4_K_M.qwen3_5_bf16.ninferno", "m-Q8_0.qwen3_5_bf16.ninferno",
                                 "m-UD-Q4_K_M.qwen3_5_bf16.ninferno"}) {
            std::ofstream(repo / "ninferno/c0ffee" / name) << "x";
            std::ofstream(repo / "ninferno/c0ffee" / (std::string(name) + ".conversion.json"))
                << "{}";
        }
        std::ofstream(repo / "ninferno/c0ffee/m-IQ2_XXS.qwen3_5_bf16.ninferno") << "x"; // unfinished
        using ninferno::convert::kept_artifact;
        const auto q4 = kept_artifact(root, "org/m-GGUF", "", "q4_k_m", "");
        check(q4 && q4->filename() == "m-Q4_K_M.qwen3_5_bf16.ninferno" &&
                  kept_artifact(root, "org/m-GGUF", "main", "8bit", "qwen3_5_bf16") &&
                  kept_artifact(root, "org/m-GGUF", "", "8bit", "")->filename() ==
                      "m-Q8_0.qwen3_5_bf16.ninferno",
              "a kept artifact by label and by width");
        // A label inside another (Q4_K_M of UD-Q4_K_M): the conversion whose own label it is.
        const auto ud = kept_artifact(root, "org/m-GGUF", "", "UD-Q4_K_M", "");
        check(ud && ud->filename() == "m-UD-Q4_K_M.qwen3_5_bf16.ninferno",
              "a kept artifact by its exact label beside a longer one");
        // The same of --vision conversions (<gguf-stem>.vision.<recipe>).
        const auto vision = root / "models--org--v-GGUF";
        fs::create_directories(vision / "refs");
        fs::create_directories(vision / "ninferno/c0ffee");
        std::ofstream(vision / "refs/main") << "c0ffee
";
        for (const char* name : {"v-Q4_K_M.vision.qwen3_5_bf16.ninferno",
                                 "v-UD-Q4_K_M.vision.qwen3_5_bf16.ninferno"}) {
            std::ofstream(vision / "ninferno/c0ffee" / name) << "x";
            std::ofstream(vision / "ninferno/c0ffee" / (std::string(name) + ".conversion.json"))
                << "{}";
        }
        const auto vq4 = kept_artifact(root, "org/v-GGUF", "", "Q4_K_M", "");
        check(vq4 && vq4->filename() == "v-Q4_K_M.vision.qwen3_5_bf16.ninferno",
              "a kept --vision artifact by its exact label beside a longer one");
        check(!kept_artifact(root, "org/m-GGUF", "", "", "") &&
                  !kept_artifact(root, "org/m-GGUF", "", "IQ2_XXS", "") &&
                  !kept_artifact(root, "org/m-GGUF", "", "Q4_K_M", "other_recipe") &&
                  !kept_artifact(root, "org/m-GGUF", "dev", "Q4_K_M", "") &&
                  !kept_artifact(root, "org/other", "", "Q4_K_M", ""),
              "no kept artifact: two match, unfinished, another recipe, ref or repository");
        // The user kept this conversion while the hub named another commit: until it moves again.
        using ninferno::convert::accept_converted;
        using ninferno::convert::converted_accepted;
        const auto commit_dir = repo / "ninferno/c0ffee";
        check(!converted_accepted(commit_dir, "main", "d00d"), "nothing kept yet");
        accept_converted(commit_dir, "main", "d00d");
        check(converted_accepted(commit_dir, "main", "d00d") &&
                  !converted_accepted(commit_dir, "main", "beef") &&
                  !converted_accepted(commit_dir, "dev", "d00d"),
              "kept for that ref and hub commit only");
        fs::create_directories(repo / "snapshots/c0ffee");
        check(!kept_artifact(root, "org/m-GGUF", "", "Q4_K_M", ""),
              "with its download still there, the usual path asks the hub");
        fs::remove_all(root);
    }
    // A prepare that downloads a moved branch's new commit and removes the old conversion; only a
    // prepare does.
    {
        UiOptions refetch_options;
        refetch_options.serve_binary = "ninferno-serve";
        LaunchRequest refetch;
        refetch.model        = "org/m-GGUF";
        refetch.quant        = "Q4_K_M";
        refetch.prepare_only = true;
        refetch.refetch      = true;
        refetch.gpu          = -1;
        const auto argv      = make_launch(refetch_options, refetch).argv;
        check(std::find(argv.begin(), argv.end(), "--refetch") != argv.end() &&
                  std::find(argv.begin(), argv.end(), "--prepare-only") != argv.end(),
              "a refetch prepare");
        refetch.recipe          = "qwen3_5_q8";
        const auto with_recipe  = make_launch(refetch_options, refetch).argv;
        const auto recipe_flag  = std::find(with_recipe.begin(), with_recipe.end(), "--recipe");
        check(recipe_flag != with_recipe.end() && recipe_flag + 1 != with_recipe.end() &&
                  recipe_flag[1] == "qwen3_5_q8",
              "a refetch keeps the replaced artifact's recipe");
        refetch.prepare_only = false;
        bool refused         = false;
        try {
            (void)make_launch(refetch_options, refetch);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "a serve start does not refetch");
    }
    // The Vision projector a start fetches and the plan charges: one choice from the listing, every
    // part of a split one.
    {
        using ninferno::convert::best_projector;
        using Names = std::vector<std::string>;
        check(best_projector({"m-Q4_K_M.gguf", "mmproj-F16.gguf", "mmproj-model-bf16.gguf",
                              "mmproj-F32.gguf"}) == Names{"mmproj-model-bf16.gguf"} &&
                  best_projector({"mmproj-F16.gguf", "mmproj-F32.gguf"}) ==
                      Names{"mmproj-F32.gguf"} &&
                  best_projector({"sub/mmproj-b.gguf", "sub/mmproj-a.gguf"}) ==
                      Names{"sub/mmproj-a.gguf"} &&
                  best_projector({"mmproj-BF16-00002-of-00002.gguf", "mmproj-F16.gguf",
                                  "mmproj-BF16-00001-of-00002.gguf"}) ==
                      Names{"mmproj-BF16-00001-of-00002.gguf", "mmproj-BF16-00002-of-00002.gguf"} &&
                  best_projector({"m-Q4_K_M.gguf", "README.md"}).empty(),
              "the projector: BF16, F32, F16, then any; ties to the first name; all its parts");
    }
    // KV per token from GGUF metadata: a plain stack, a hybrid one, a sliding-window one.
    {
        const nlohmann::json dense{
            {"general.architecture", "llama"},  {"llama.block_count", 32},
            {"llama.context_length", 131072},   {"llama.embedding_length", 4096},
            {"llama.attention.head_count", 32}, {"llama.attention.head_count_kv", 8}};
        const auto kv = kv_model_of_gguf(dense);
        check(kv.known && kv.full_bytes_per_token == 2 * 32 * 8 * 256 &&
                  kv.native_context == 131072,
              "KV of a dense GQA stack (INT8: 64 KiB per token)");
        check(kv_bytes(kv, 1000) == 2 * 1000.0 * 32 * 8 * 256, "KV bytes scale with the context");
        const nlohmann::json hybrid{
            {"general.architecture", "qwen35"},    {"qwen35.block_count", 32},
            {"qwen35.full_attention_interval", 4}, {"qwen35.attention.head_count", 16},
            {"qwen35.attention.head_count_kv", 4}, {"qwen35.attention.key_length", 256},
            {"qwen35.attention.value_length", 256}};
        check(kv_model_of_gguf(hybrid).full_bytes_per_token == 2 * 8 * 4 * 512,
              "KV of a hybrid stack: its attention layers only");
        const nlohmann::json swa{
            {"general.architecture", "gemma4"},
            {"gemma4.block_count", 4},
            {"gemma4.attention.head_count", 8},
            {"gemma4.attention.head_count_kv", 2},
            {"gemma4.attention.key_length", 128},
            {"gemma4.attention.sliding_window", 1024},
            {"gemma4.attention.sliding_window_pattern", {true, true, true, false}}};
        const auto s = kv_model_of_gguf(swa);
        check(s.full_bytes_per_token == 2 * 2 * 256 &&
                  s.window_bytes_per_token == 2 * 3 * 2 * 256 &&
                  kv_bytes(s, 4096) == 2 * (4096.0 * 512 + 1024.0 * 1536),
              "KV of a sliding-window stack stops at the window");
        check(!kv_model_of_gguf(nlohmann::json{{"general.architecture", "x"}}).known,
              "unknown geometry");

        // The same from config.json: a hybrid's layer_types under text_config (Qwen3.5: one full
        // layer in four), gpt-oss's sliding layers, NemotronH's pattern, a dense model's
        // hidden_size / heads, a global head geometry; latent attention unknown.
        const nlohmann::json qwen{
            {"text_config",
             {{"num_hidden_layers", 8},
              {"num_attention_heads", 16},
              {"num_key_value_heads", 4},
              {"head_dim", 256},
              {"max_position_embeddings", 262144},
              {"layer_types",
               {"linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"}}}}};
        const auto q = kv_model_of_config(qwen);
        check(q.known && q.full_bytes_per_token == 2 * 2 * 4 * 2 * 256 &&
                  q.window_bytes_per_token == 0 && q.native_context == 262144,
              "config.json KV: full layers of a hybrid under text_config");
        const auto g = kv_model_of_config(
            {{"num_hidden_layers", 4},
             {"num_attention_heads", 64},
             {"num_key_value_heads", 8},
             {"head_dim", 64},
             {"sliding_window", 128},
             {"layer_types",
              {"sliding_attention", "full_attention", "sliding_attention", "full_attention"}}});
        check(g.full_bytes_per_token == 2 * 2 * 8 * 2 * 64 &&
                  g.window_bytes_per_token == 2 * 2 * 8 * 2 * 64 && g.window == 128 &&
                  kv_bytes(g, 4096) == 2 * (4096.0 * 2048 + 128.0 * 2048),
              "config.json KV: sliding layers stop at the window");
        const auto n = kv_model_of_config({{"num_hidden_layers", 6},
                                           {"num_attention_heads", 32},
                                           {"num_key_value_heads", 2},
                                           {"head_dim", 128},
                                           {"hybrid_override_pattern", "M-M*-M"}});
        check(n.full_bytes_per_token == 2 * 2 * 2 * 128,
              "config.json KV: NemotronH's attention layer");
        const auto d = kv_model_of_config({{"num_hidden_layers", 2},
                                           {"num_attention_heads", 32},
                                           {"num_key_value_heads", 8},
                                           {"hidden_size", 4096}});
        check(d.full_bytes_per_token == 2 * 2 * 8 * 2 * 128,
              "config.json KV: head_dim from the width");
        const auto w =
            kv_model_of_config({{"num_hidden_layers", 2},
                                {"num_attention_heads", 16},
                                {"num_key_value_heads", 8},
                                {"head_dim", 256},
                                {"global_head_dim", 512},
                                {"num_global_key_value_heads", 2},
                                {"sliding_window", 1024},
                                {"layer_types", {"sliding_attention", "full_attention"}}});
        check(w.full_bytes_per_token == 2 * 2 * 2 * 512 &&
                  w.window_bytes_per_token == 2 * 8 * 2 * 256,
              "config.json KV: the full layers' own geometry");
        check(!kv_model_of_config({{"num_hidden_layers", 2},
                                   {"num_attention_heads", 16},
                                   {"head_dim", 128},
                                   {"kv_lora_rank", 512}})
                   .known,
              "config.json KV: latent attention is not estimated");
        // gpt-oss-20b: 20.9B parameters, 3.6B a token reads (4 of 32 experts).
        const auto o = kv_model_of_config({{"num_hidden_layers", 24},
                                           {"num_attention_heads", 64},
                                           {"num_key_value_heads", 8},
                                           {"head_dim", 64},
                                           {"hidden_size", 2880},
                                           {"intermediate_size", 2880},
                                           {"num_local_experts", 32},
                                           {"num_experts_per_tok", 4},
                                           {"vocab_size", 201088}});
        check(o.params > 20.5e9 && o.params < 21.5e9 && o.active_params > 3.3e9 &&
                  o.active_params < 3.9e9 && o.experts_used == 0.125 && o.routed_share > 0.85,
              "config.json shape: a mixture of experts' total and active parameters");
        const auto t       = kv_model_of_config({{"num_hidden_layers", 2},
                                                 {"num_attention_heads", 32},
                                                 {"num_key_value_heads", 8},
                                                 {"hidden_size", 4096},
                                                 {"intermediate_size", 14336},
                                                 {"vocab_size", 1000},
                                                 {"tie_word_embeddings", true}});
        const double layer = 4096.0 * 128 * (64 + 16) + 3.0 * 4096 * 14336;
        check(t.params == 2 * layer + 1000.0 * 4096 && t.active_params == t.params &&
                  t.experts_used == 0,
              "config.json shape: a dense model reads all of it");
        // Qwen3.6 35B's GGUF: 10 attention layers keep bf16 KV, 30 GatedDeltaNet layers a 2 MiB
        // fp32 state each (what serve's capacity line showed on the 4090: 60 MiB a sequence).
        const auto h = kv_model_of_gguf({{"general.architecture", "qwen35moe"},
                                         {"qwen35moe.block_count", 40},
                                         {"qwen35moe.embedding_length", 2048},
                                         {"qwen35moe.attention.head_count", 16},
                                         {"qwen35moe.attention.head_count_kv", 2},
                                         {"qwen35moe.attention.key_length", 256},
                                         {"qwen35moe.attention.value_length", 256},
                                         {"qwen35moe.full_attention_interval", 4},
                                         {"qwen35moe.expert_count", 256},
                                         {"qwen35moe.expert_used_count", 8},
                                         {"qwen35moe.expert_feed_forward_length", 512},
                                         {"qwen35moe.expert_shared_feed_forward_length", 512},
                                         {"qwen35moe.ssm.conv_kernel", 4},
                                         {"qwen35moe.ssm.state_size", 128},
                                         {"qwen35moe.ssm.group_count", 16},
                                         {"qwen35moe.ssm.time_step_rank", 32},
                                         {"qwen35moe.ssm.inner_size", 4096}});
        check(h.full_layers == 10 && h.full_bytes_per_token == 10 * 2 * 512 * 2 &&
                  h.state_bytes == 30 * (4096.0 * 128 * 4 + 3 * (4096 + 2 * 16 * 128) * 2) &&
                  h.workspace_bytes == 1024.0 * (12 * 2048 + 3 * (8 * 512 + 512)) * 2,
              "GGUF memory: a hybrid's KV layers, recurrent states and workspace");
        // NemotronH: attention, Mamba2 and MoE layers apart (6 + 24 + 23 of 53), not every layer
        // with experts.
        std::vector<int> kv_heads(53, 0), ffn(53, 0);
        for (int i : {5, 12, 19, 26, 33, 42}) { kv_heads[static_cast<std::size_t>(i)] = 2; }
        for (int i = 1, n = 0; i < 53 && n < 23; i += 2) {
            if (kv_heads[static_cast<std::size_t>(i)] == 0) {
                ffn[static_cast<std::size_t>(i)] = 1856;
                ++n;
            }
        }
        const auto nh =
            kv_model_of_gguf({{"general.architecture", "nemotron_h_moe"},
                              {"nemotron_h_moe.block_count", 53},
                              {"nemotron_h_moe.embedding_length", 2688},
                              {"nemotron_h_moe.attention.head_count", 32},
                              {"nemotron_h_moe.attention.head_count_kv", kv_heads},
                              {"nemotron_h_moe.attention.key_length", 128},
                              {"nemotron_h_moe.attention.value_length", 128},
                              {"nemotron_h_moe.feed_forward_length", ffn},
                              {"nemotron_h_moe.expert_count", 128},
                              {"nemotron_h_moe.expert_used_count", 6},
                              {"nemotron_h_moe.expert_feed_forward_length", 1856},
                              {"nemotron_h_moe.expert_shared_feed_forward_length", 3712},
                              {"nemotron_h_moe.vocab_size", 131072},
                              {"nemotron_h_moe.ssm.conv_kernel", 4},
                              {"nemotron_h_moe.ssm.state_size", 128},
                              {"nemotron_h_moe.ssm.group_count", 8},
                              {"nemotron_h_moe.ssm.inner_size", 4096},
                              {"nemotron_h_moe.ssm.time_step_rank", 64}});
        check(nh.full_layers == 6 && nh.params > 28e9 && nh.params < 34e9 &&
                  nh.active_params > 2.5e9 && nh.active_params < 4.5e9 &&
                  nh.state_bytes > 24 * 2.0 * 1024 * 1024,
              "GGUF shape: NemotronH counts each layer by its kind");
    }

    // Parallel requests reach ninferno-serve as --max-concurrency.
    {
        UiOptions launch_options;
        launch_options.serve_binary = "/opt/ninferno/ninferno-serve";
        LaunchRequest conc;
        conc.model           = "org/name";
        conc.port            = 8101;
        conc.max_concurrency = 4;
        const auto argv      = make_launch(launch_options, conc).argv;
        const auto at        = std::find(argv.begin(), argv.end(), "--max-concurrency");
        check(at != argv.end() && at + 1 != argv.end() && *(at + 1) == "4",
              "--max-concurrency passed");
        conc.max_concurrency = 65;
        bool refused         = false;
        try {
            (void)make_launch(launch_options, conc);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "max_concurrency above 64 refused");
    }

    // A small card: does not fit, not recommended.
    TargetGpu small{
        .name = "NVIDIA GeForce RTX 4060", .memory_gib = 7.6, .bandwidth_gbs = 272, .fp4 = false};
    e = estimate(catalog, *qwen, small, sizes);
    check(!e.recommended && !e.preferred.has_value(), "8 GB: 35B does not fit");
}

#ifdef _WIN32
// The children of test_processes on Windows, as the /bin/sh lines print them on Linux: "prepare"
// writes serve's lines and a download's progress and exits 3; "serve" runs until Ctrl+Break
// (ProcessPool::stop) and then exits 0.
std::atomic<bool> child_stop{false};

int child(const std::string& mode) {
    if (mode == "prepare") {
        std::cout << "one\ntwo\n" << std::getenv("NINFERNO_UI_TEST") << std::endl;
        std::cerr << "12:00 INFO  context | 32 768 tokens per request (auto)" << std::endl;
        std::cout << "12:00 INFO  speculation | registry x | default mtp:3\n"
                     "12:00 INFO  speculation | family code -> mtp:5\n"
                     "12:00 WARN  speculation | first warning\n"
                     "12:00 WARN  speculation | fit fallback: x; running mtp:3\n"
                     "  m-Q4_K_M.gguf: 42% (1.2 GB of 2.9 GB), 61 MB/s, about 1 min left"
                  << std::endl;
        return 3;
    }
    SetConsoleCtrlHandler(
        [](DWORD) -> BOOL {
            child_stop = true;
            return TRUE;
        },
        TRUE);
    while (!child_stop) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
    return 0;
}
#endif

void test_processes([[maybe_unused]] const std::string& self) {
    ProcessPool pool(100);
    LaunchSpec spec;
    spec.kind = "prepare";
#ifdef _WIN32
    spec.argv = {self, "--child", "prepare"}; // no /bin/sh: this program prints the same (child())
#else
    spec.argv = {
        "/bin/sh", "-c",
        "echo one; echo two; echo $NINFERNO_UI_TEST; "
        "echo '12:00 INFO  context | 32 768 tokens per request (auto)' >&2; "
        "echo '12:00 INFO  speculation | registry x | default mtp:3'; "
        "echo '12:00 INFO  speculation | family code -> mtp:5'; "
        "echo '12:00 WARN  speculation | first warning'; "
        "echo '12:00 WARN  speculation | fit fallback: x; running mtp:3'; "
        "echo '  m-Q4_K_M.gguf: 42% (1.2 GB of 2.9 GB), 61 MB/s, about 1 min left'; exit 3"};
#endif
    spec.env["NINFERNO_UI_TEST"] = "env-ok";
    const auto id              = pool.start(spec);
    ProcessInfo info;
    for (int i = 0; i < 100; ++i) {
        check(pool.find(id, info), "process listed");
        if (info.state == "failed" || info.state == "exited") { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    check(info.state == "failed" && info.exit_code == 3, "exit code kept");
    const auto lines = pool.log(id);
    check(lines.size() == 9 && lines[2] == "env-ok", "output and environment captured");
    check(info.percent == 42 && info.progress_file == "m-Q4_K_M.gguf" &&
              info.progress == "42% (1.2 GB of 2.9 GB), 61 MB/s, about 1 min left",
          "download progress noted");
    check(info.context == "32 768 tokens per request (auto)", "serve's context noted");
    check(info.speculation == "registry x | default mtp:3", "serve's first speculation line noted");
    check(info.speculation_fallback == "fit fallback: x; running mtp:3",
          "serve's last speculation warning noted");

    LaunchSpec sleeper;
    sleeper.kind   = "serve";
#ifdef _WIN32
    sleeper.argv = {self, "--child", "serve"};
#else
    sleeper.argv   = {"/bin/sh", "-c", "trap 'exit 0' TERM; while :; do sleep 0.05; done"};
#endif
    const auto sid = pool.start(sleeper);
    check(pool.find(sid, info) && info.state == "starting", "serve starts as starting");
    pool.mark_ready(sid);
    check(pool.find(sid, info) && info.state == "ready", "marked ready");
    check(pool.stop(sid, std::chrono::seconds(5)), "stop");
    check(pool.find(sid, info) && info.state == "exited", "stopped on request: exited");
    check(pool.free_port(18100) >= 18100, "a free port");
}

void test_http() {
    UiOptions options;
    options.offline      = true;
    options.serve_binary = "/bin/false";

    struct NoGpu final : GpuSource {
        std::vector<GpuStat> snapshot() override {
            GpuStat g;
            g.name         = "NVIDIA GeForce RTX 4090";
            g.memory_total = 24ULL << 30;
            g.cc_major     = 8;
            g.cc_minor     = 9;
            return {g};
        }

        std::string error() const override { return {}; }
    };

    App app(options, std::make_unique<NoGpu>());
    httplib::Server server;
    app.install(server);
    const int port = server.bind_to_any_port("127.0.0.1");
    std::thread thread([&] { server.listen_after_bind(); });
    server.wait_until_ready();

    httplib::Client client("127.0.0.1", port);
    auto res = client.Get("/");
    check(res && res->status == 200 &&
              res->body.find("<title>NInferno</title>") != std::string::npos,
          "page served");
    check(res && res->has_header("Content-Security-Policy"), "page has a CSP");
    res = client.Get("/api/state");
    check(res && res->status == 200 && nlohmann::json::parse(res->body)["gpus"].size() == 1,
          "state");
    res = client.Get("/api/catalog?gpu=0");
    check(res && res->status == 200 && nlohmann::json::parse(res->body)["models"].size() >= 7,
          "catalog");

    // DNS rebinding: a foreign Host is refused even on GET.
    res = client.Get("/api/state",
                     httplib::Headers{{"Host", "evil.example:" + std::to_string(port)}});
    check(res && res->status == 403, "foreign Host refused");
    // CSRF: a POST from another origin, or not JSON, is refused before it runs.
    res = client.Post("/api/stop", httplib::Headers{{"Origin", "https://evil.example"}},
                      "{\"id\":1}", "application/json");
    check(res && res->status == 403, "cross-origin POST refused");
    res = client.Post("/api/stop", "id=1", "application/x-www-form-urlencoded");
    check(res && res->status == 415, "form POST refused");
    res = client.Post("/api/launch", "{\"model\":\"--help\"}", "application/json");
    check(res && res->status == 400, "bad model refused");
    // The tune beside a model only by request, and only for a converted model's artifact.
    res = client.Post("/api/launch", R"({"model":"org/m","spec_profile":true})",
                      "application/json");
    check(res && res->status == 400 &&
              res->body.find("started by its .ninferno path") != std::string::npos,
          "spec_profile for a model not converted here answered 400");
    res = client.Post("/api/launch", R"({"model":"org/m","spec_profile":"yes"})",
                      "application/json");
    check(res && res->status == 400 && res->body.find("bad request body") != std::string::npos,
          "spec_profile is a boolean");
    res = client.Post("/api/tune", "{", "application/json");
    check(res && res->status == 400 && res->body.find("bad request body") != std::string::npos,
          "a broken JSON body answered 400");
    // The plan without the weights: its parameters as ninferno-serve takes them, refused with 400;
    // offline the repository's shape cannot be read (502 with the reason, never a guess).
    for (const char* query : {"/api/plan?repo=a/b&ctx=abc", "/api/plan?repo=a/b&par=9",
                              "/api/plan?repo=a/b&par=0", "/api/plan?repo=a/b&kv=fp16",
                              "/api/plan?repo=a/b&spec=eagle", "/api/quants?repo=a/b&gpu=0&ctx=-5",
                              "/api/catalog?gpu=0&ctx=abc", "/api/catalog?gpu=0&par=12"}) {
        res = client.Get(query);
        check(res && res->status == 400 && res->body.find("\"error\"") != std::string::npos,
              std::string("a bad plan parameter answered 400: ") + query);
    }
    res = client.Get("/api/plan?repo=a/b&ctx=auto&par=4&kv=auto&spec=auto");
    check(res && res->status == 502 && res->body.find("\"error\"") != std::string::npos,
          "offline: the plan says it cannot read the repository");
    for (const char* query :
         {"/api/arch?repo=..%2Fx", "/api/quants?repo=..%2Fx", "/api/plan?repo=..%2Fx"}) {
        res = client.Get(query);
        check(res && res->status == 400 &&
                  res->body.find("repo must be org/name") != std::string::npos,
              std::string("a repo outside org/name answered 400: ") + query);
    }
    res = client.Post("/chat/99/v1/chat/completions", "{}", "application/json");
    check(res && res->status == 409, "chat to nothing");

    // A keyless UI reached by another name: through a proxy the missing key is named, else the
    // Host is.
    res = client.Get("/api/state",
                     httplib::Headers{{"Host", "ui.example.com"}, {"X-Forwarded-For", "10.0.0.2"}});
    check(res && res->status == 403 && res->body.find("NINFERNO_UI_KEY") != std::string::npos,
          "keyless UI behind a proxy: 403 naming NINFERNO_UI_KEY");
    res = client.Get("/api/state", httplib::Headers{{"Host", "ui.example.com"}});
    check(res && res->status == 403 && res->body.find("Host not allowed") != std::string::npos &&
              res->body.find("NINFERNO_UI_KEY") == std::string::npos,
          "keyless UI, another Host without a proxy: 403 Host not allowed");

    server.stop();
    thread.join();

    // The access key, through the handler: a request marked as proxied is not this machine's.
    {
        UiOptions keyed_options = options;
        keyed_options.access.key = std::string(32, 'k');
        App keyed_app(keyed_options, std::make_unique<NoGpu>());
        httplib::Server keyed_server;
        keyed_app.install(keyed_server);
        const int keyed_port = keyed_server.bind_to_any_port("127.0.0.1");
        std::thread keyed_thread([&] { keyed_server.listen_after_bind(); });
        keyed_server.wait_until_ready();
        httplib::Client keyed_client("127.0.0.1", keyed_port);
        const httplib::Headers far{{"X-Forwarded-For", "10.0.0.2"}};
        res = keyed_client.Get("/api/state");
        check(res && res->status == 200, "key: this machine passes");
        res = keyed_client.Get("/api/state", far);
        check(res && res->status == 401 && res->body.find("access key") != std::string::npos,
              "key: 401 JSON off the page");
        res = keyed_client.Get("/", far);
        check(res && res->status == 401 && res->body.find("name=key") != std::string::npos,
              "key: the key form on /");
        res = keyed_client.Get("/?key=" + keyed_options.access.key, far);
        check(res && res->status == 303 && res->get_header_value("Location") == "/" &&
                  res->get_header_value("Set-Cookie").find("HttpOnly") != std::string::npos,
              "key: ?key= sets the cookie and redirects");
        auto tls = far;
        tls.emplace("X-Forwarded-Proto", "https");
        res = keyed_client.Get("/?key=" + keyed_options.access.key, tls);
        check(res && res->get_header_value("Set-Cookie").find("; Secure") != std::string::npos,
              "key: Secure behind TLS");
        res = keyed_client.Get("//evil.example/?key=" + keyed_options.access.key, far);
        check(res && res->status == 303 && res->get_header_value("Location") == "/",
              "key: no redirect off the site");
        // The 303 that drops ?key= stays on this site whatever the target: the targets go out
        // as written (no client-side encoding), absolute-form included.
        {
            httplib::Client raw("127.0.0.1", keyed_port);
            raw.set_path_encode(false);
            const std::string key  = "key=" + keyed_options.access.key;
            const auto same_origin = [](const std::string& location) {
                if (location.empty() || location[0] != '/') { return false; }
                if (location.size() > 1 && (location[1] == '/' || location[1] == '\\')) {
                    return false;
                }
                return std::all_of(location.begin(), location.end(), [](unsigned char c) {
                    return c > 0x20 && c < 0x7f && c != '\\';
                });
            };
            // target -> the Location it must get
            const std::pair<std::string, std::string> targets[] = {
                {"/\\evil?", "/"},
                {"/%09/evil.example/?", "/%09/evil.example/"},
                {"/%2F%2Fevil.example/?", "/%2F%2Fevil.example/"},
                {"/%5Cevil?", "/%5Cevil"},
                {"//evil.example/?", "/"},
                {"/\\/evil.example/?", "/"},
                {"http://evil.example/x?", "/"},
            };
            for (const auto& [target, want] : targets) {
                res                        = raw.Get(target + key, far);
                const std::string location = res ? res->get_header_value("Location") : "";
                check(res && res->status == 303 && same_origin(location) && location == want,
                      "key: the redirect of " + target + key.substr(0, 4) +
                          "... stays on the site (Location " + location + ")");
            }
            // The other parameters stay, re-encoded and in the order sent; every key goes, a
            // repeated one too. A name without a value comes back as name= (httplib's parse).
            // The second key is spelled k%65y: an identical raw pair would be merged by httplib's
            // query parse before the handler sees it.
            res = raw.Get("/api/x?a=1&" + key + "&b=two%20words&k%65y=" + keyed_options.access.key +
                              "&flag",
                          far);
            const std::string kept = res ? res->get_header_value("Location") : "";
            check(res && res->status == 303 && kept == "/api/x?a=1&b=two%20words&flag=",
                  "key: other parameters kept, every key dropped (Location " + kept + ")");
            res                       = raw.Get("/api/x?z=1&" + key + "&a=%2Fx%26y%3D&a=2", far);
            const std::string ordered = res ? res->get_header_value("Location") : "";
            check(res && res->status == 303 && ordered == "/api/x?z=1&a=%2Fx%26y%3D&a=2",
                  "key: reserved bytes re-encoded, the order sent kept (Location " + ordered + ")");
        }
        auto bearer_headers = far;
        bearer_headers.emplace("Authorization", "Bearer " + keyed_options.access.key);
        res = keyed_client.Get("/api/state", bearer_headers);
        check(res && res->status == 200, "key: bearer passes");
        keyed_server.stop();
        keyed_thread.join();
    }

    // The catalog's feed: read from a URL, used, and kept for the next start; a broken feed
    // leaves the catalog in use.
    {
        namespace fs = std::filesystem;
        const auto cache =
            fs::temp_directory_path() / ("ninferno_ui_feed_" + std::to_string(::getpid()));
        std::string feed = R"({"schema": 1, "gpus": [{"key": "RTX 4090", "match": ["RTX 4090"],
            "memory_gib": 24, "bandwidth_gbs": 1008}], "models": [{"id": "feed-model",
            "name": "Feed Model 9B", "params_b": 9, "sources": [{"repo": "org/feed-9B-GGUF",
            "quant": "Q4_K_M", "format": "gguf", "bits": 4}]}]})";
        httplib::Server origin;
        origin.Get("/catalog.json", [&](const httplib::Request&, httplib::Response& res) {
            res.set_content(feed, "application/json");
        });
        const int origin_port = origin.bind_to_any_port("127.0.0.1");
        std::thread origin_thread([&] { origin.listen_after_bind(); });
        origin.wait_until_ready();

        UiOptions fed;
        fed.offline      = true; // no background reads: refresh_catalog() reads it here
        fed.serve_binary = "/bin/false";
        fed.hub_cache    = cache;
        fed.catalog_url  = "http://127.0.0.1:" + std::to_string(origin_port) + "/catalog.json";
        {
            App fed_app(fed, std::make_unique<NoGpu>());
            check(fed_app.catalog()->models().size() >= 7, "built-in catalog before the feed");
            check(fed_app.refresh_catalog() && fed_app.catalog()->models().size() == 1 &&
                      fed_app.catalog()->models()[0].id == "feed-model",
                  "the feed's catalog in use");
            httplib::Server fed_server;
            fed_app.install(fed_server);
            const int fed_port = fed_server.bind_to_any_port("127.0.0.1");
            std::thread fed_thread([&] { fed_server.listen_after_bind(); });
            fed_server.wait_until_ready();
            httplib::Client fed_client("127.0.0.1", fed_port);
            const auto listed = fed_client.Get("/api/catalog?gpu=0");
            const auto body   = listed ? nlohmann::json::parse(listed->body) : nlohmann::json{};
            check(listed && listed->status == 200 && body["models"].size() == 1 &&
                      body.value("catalog_source", std::string()).starts_with(fed.catalog_url),
                  "/api/catalog lists the feed's models and names the feed");
            fed_server.stop();
            fed_thread.join();
            feed = "{\"schema\": 1, \"models\": [";
            check(!fed_app.refresh_catalog() && fed_app.catalog()->models()[0].id == "feed-model",
                  "a broken feed leaves the catalog in use");
        }
        origin.stop();
        origin_thread.join();
        App restarted(fed, std::make_unique<NoGpu>());
        check(restarted.catalog()->models().size() == 1, "the feed's last copy serves at start");
        fs::remove_all(cache);
    }
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    if (argc == 3 && std::string(argv[1]) == "--child") { return child(argv[2]); }
#endif
    (void)argc;
    test_access();
    test_hub_directories();
    test_launch();
    test_catalog();
    test_processes(ninferno::sys::self_exe().string());
    test_http();
    if (failures != 0) {
        std::cerr << failures << " ui check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "ui: all checks passed" << std::endl;
    return 0;
}
