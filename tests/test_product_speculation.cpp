// The drafter a product finds for an explicit --spec without --drafter: the separate drafter
// artifact named in <artifact>.speculation.json whose kind drafts through that backend
// (product::model_drafter). No GPU; only the manifest is read. Also where load bounds move when
// their own graphs do not fit (product::snap_load_to_captured).

#include "models/qwen3_5/program/planning/copy_proposal.h"
#include "models/qwen3_5/program/planning/speculation_load.h"
#include "product/speculation/speculation.h"
#include "spec/registry.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// A manifest beside `artifact` with one attached separate drafter of `kind` at `drafter`.
void write_manifest(const std::filesystem::path& artifact, const std::string& kind,
                    const std::string& drafter, const std::string& status = "attached") {
    const nlohmann::json doc = {
        {"schema", 1},
        {"target", {{"artifact_id", "0"}, {"engine", "nemotron_h"}}},
        {"source", "none"},
        {"drafters",
         nlohmann::json::array({{{"name", "mtp"},
                                 {"kind", kind},
                                 {"artifact", drafter},
                                 {"status", status}}})},
        {"profiles", nlohmann::json::array()}};
    std::ofstream(ninferno::spec::manifest_path(artifact)) << doc.dump(1) << '\n';
}

} // namespace

int main() {
    using ninferno::SpeculativeBackend;
    using ninferno::product::model_drafter;
    int failures   = 0;
    const auto dir =
        std::filesystem::temp_directory_path() /
        ("ninferno-product-speculation-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    const auto target = dir / "target.ninferno";
    const auto mtp    = [&] { return model_drafter(target, SpeculativeBackend::Mtp); };
    const auto dflash = [&] { return model_drafter(target, SpeculativeBackend::DFlash); };

    failures += check(mtp().empty(), "no manifest: no drafter");
    // Nothing shallower than one draft, nothing for plain decode, nothing without a profile.
    {
        using ninferno::product::served_shallower;
        const auto dflash_at = [](std::int32_t k) {
            ninferno::spec::SpecConfig c;
            c.mode = ninferno::spec::Mode::DFlash;
            c.k    = k;
            return c;
        };
        const auto subject = ninferno::product::speculation_subject(target, 0);
        failures += check(!served_shallower(subject, true, dflash_at(1), 1.5) &&
                              !served_shallower(subject, true, ninferno::spec::SpecConfig{}, 0) &&
                              !served_shallower(subject, true, dflash_at(3), 1.5),
                          "served_shallower ran below depth 1, for plain, or without a profile");
    }
    // A server whose choice does not fit runs the deepest depth that fits, with the same drafter:
    // one draft shallower, then depth 1, then halving between them, and the depth it returns is the
    // one built last; nothing when depth 1 does not fit.
    {
        using ninferno::product::deepest_fitting_shallower;
        using ninferno::product::ServedSpeculation;
        std::vector<std::int32_t> offered;
        const auto shallower = [](const ninferno::spec::SpecConfig& config, double floor) {
            std::optional<ServedSpeculation> out;
            if (config.k <= 1) { return out; }
            out.emplace();
            out->menu.primary.config   = config;
            out->menu.primary.config.k = config.k - 1;
            out->menu.primary.floor    = floor;
            return out;
        };
        const auto fits_at_most = [&](std::int32_t depth) {
            return [&offered, depth](const ServedSpeculation& candidate) {
                offered.push_back(candidate.menu.primary.config.k);
                return candidate.menu.primary.config.k <= depth;
            };
        };
        ninferno::spec::MenuChoice tried;
        tried.config.mode = ninferno::spec::Mode::Mtp;
        tried.config.k    = 6;
        tried.floor       = 1.4;

        const auto three = deepest_fitting_shallower(tried, shallower, fits_at_most(3));
        failures += check(three && three->menu.primary.config.k == 3 &&
                              three->menu.primary.config.mode == ninferno::spec::Mode::Mtp &&
                              offered == std::vector<std::int32_t>{5, 1, 3, 4, 3},
                          "deepest_fitting_shallower: 5 no, 1 and 3 fit, 4 no, 3 built again");
        offered.clear();
        const auto five = deepest_fitting_shallower(tried, shallower, fits_at_most(5));
        failures += check(five && five->menu.primary.config.k == 5 &&
                              offered == std::vector<std::int32_t>{5},
                          "deepest_fitting_shallower: a near miss takes one build");
        offered.clear();
        failures += check(!deepest_fitting_shallower(tried, shallower, fits_at_most(0)) &&
                              offered == std::vector<std::int32_t>{5, 1},
                          "deepest_fitting_shallower: depth 1 refused, then nothing");
        // From a pick at depth 16 (mtp:15 + 1): a handful of builds, not one per depth.
        tried.config.k = 16;
        for (std::int32_t fitting = 1; fitting <= 15; ++fitting) {
            offered.clear();
            const auto deepest = deepest_fitting_shallower(tried, shallower, fits_at_most(fitting));
            failures += check(deepest && deepest->menu.primary.config.k == fitting &&
                                  offered.back() == fitting && offered.size() <= 7,
                              "deepest_fitting_shallower from depth 16: depth " +
                                  std::to_string(fitting) + " in at most 7 builds, built last");
        }
    }
    // One draft shallower than a depth that does not fit: the profile's drafter for that mode, and
    // a watchdog floor for the shallower depth from the floors measured around it.
    {
        using ninferno::product::shallower_floor;
        using ninferno::product::shallower_profile;
        const auto near = [](double a, double b) { return a > b - 1e-9 && a < b + 1e-9; };
        failures += check(near(shallower_floor({{2, 1.52}, {7, 2.14}}, 6), 1.52 + 0.8 * 0.62) &&
                              near(shallower_floor({{7, 2.14}, {6, 1.9}}, 6), 1.9) &&
                              near(shallower_floor({{7, 2.14}}, 6), 1.0 + 1.14 * 6.0 / 7.0) &&
                              near(shallower_floor({{2, 1.5}}, 4), 2.0) &&
                              shallower_floor({{7, 0.0}}, 6) == 0.0 &&
                              shallower_floor({{7, 1.0}}, 6) == 0.0 &&
                              near(shallower_floor({{2, 0.9}, {7, 2.14}}, 6),
                                   1.0 + 1.14 * 6.0 / 7.0) &&
                              shallower_floor({}, 6) == 0.0,
                          "shallower_floor: measured, interpolated, scaled, none; floors of 1 or "
                          "less are no measurement");

        const auto choice = [](ninferno::spec::Mode mode, std::int32_t k, const char* drafter,
                               double floor) {
            ninferno::spec::Choice c;
            c.config.mode            = mode;
            c.config.k               = k;
            c.drafter                = drafter;
            c.floor_tokens_per_round = floor;
            return c;
        };
        using ninferno::spec::Mode;
        ninferno::spec::Profile profile;
        profile.overall = profile.non_thinking = choice(Mode::DFlash, 7, "dflash", 2.14);
        profile.thinking                       = choice(Mode::DFlash, 5, "dflash", 1.9);
        profile.fit_fallback                   = choice(Mode::Mtp, 2, "mtp", 1.3);
        profile.served                         = ninferno::spec::ServedChoices{};

        const ninferno::spec::SpecConfig tried = profile.overall.config;
        const auto dflash6                     = shallower_profile(profile, tried, 2.14);
        failures +=
            check(dflash6 && dflash6->overall.config.k == 6 &&
                      dflash6->overall.drafter == "dflash" &&
                      near(dflash6->overall.floor_tokens_per_round, 1.9 + 0.5 * 0.24) &&
                      dflash6->thinking.config.k == 6 && dflash6->non_thinking.config.k == 6 &&
                      !dflash6->fit_fallback && !dflash6->served,
                  "DFlash k7 serves k6 everywhere with the floor between k5 and k7");

        // The fit fallback's MTP at k3 serves k2 with the fallback's drafter and its measured
        // floor.
        const ninferno::spec::SpecConfig mtp3 = choice(Mode::Mtp, 3, "", 0).config;
        const auto mtp2                       = shallower_profile(profile, mtp3, 1.6);
        failures += check(mtp2 && mtp2->overall.config.mode == Mode::Mtp &&
                              mtp2->overall.config.k == 2 &&
                              mtp2->overall.drafter == "mtp" &&
                              near(mtp2->overall.floor_tokens_per_round, 1.3),
                          "MTP k3 serves k2 with the fallback's drafter and floor");
        failures += check(!shallower_profile(profile, choice(Mode::DFlash2, 4, "", 0).config, 2) &&
                              !shallower_profile(profile, choice(Mode::DFlash, 1, "", 0).config,
                                                 1.5) &&
                              !shallower_profile(profile, ninferno::spec::SpecConfig{}, 0),
                          "no shallower profile for a mode it lacks, depth 1, or plain decode");
    }

    for (const char* kind : {"nemotron_h_mtp", "gemma4_assistant"}) {
        write_manifest(target, kind, "/drafters/head.ninferno");
        failures += check(mtp() == "/drafters/head.ninferno",
                          std::string("an attached ") + kind + " drafter drafts for --spec mtp");
        failures += check(dflash().empty(),
                          std::string("an attached ") + kind + " drafter is no DFlash drafter");
    }
    write_manifest(target, "nemotron_h_mtp", "/drafters/head.ninferno", "incompatible: vocabulary");
    failures += check(mtp().empty(), "an incompatible drafter is not used");
    write_manifest(target, "dflash", "/drafters/dflash.ninferno");
    failures += check(dflash() == "/drafters/dflash.ninferno" && mtp().empty(),
                      "a dflash drafter drafts for --spec dflash only");

    // Load bounds at depths the menu does not capture move to captured ones when the Engine
    // cannot afford their graphs: the deepest within the bound, else the shallowest captured.
    {
        ninferno::SpeculativeOptions options;
        options.backend              = SpeculativeBackend::DFlash;
        options.draft_tokens         = 7;
        options.request_draft_tokens = {2, 5, 11};
        options.load                 = {{.lanes = 2, .draft_tokens = 3},
                                        {.lanes = 3, .draft_tokens = 0},
                                        {.lanes = 4, .draft_tokens = 1},
                                        {.lanes = 5, .draft_tokens = 7}};
        failures +=
            check(ninferno::product::snap_load_to_captured(options, 8), "uncaptured bounds move");
        const auto& load = options.load;
        failures += check(load[0].draft_tokens == 2 && load[1].draft_tokens == 0 &&
                              load[2].draft_tokens == 2 && load[3].draft_tokens == 7,
                          "bounds move to 2, stay plain, move up to 2, stay at 7");
        failures += check(!ninferno::product::snap_load_to_captured(options, 8),
                          "captured bounds do not move");

        // A bound wider than the server's concurrency plans no graphs: it stays, and nothing moved
        // means no retry of the Engine.
        options.load = {{.lanes = 2, .draft_tokens = 5}, {.lanes = 5, .draft_tokens = 3}};
        failures += check(!ninferno::product::snap_load_to_captured(options, 4) &&
                              options.load[1].draft_tokens == 3,
                          "a bound beyond max_concurrency moved");
        failures += check(ninferno::product::snap_load_to_captured(options, 5) &&
                              options.load[1].draft_tokens == 2,
                          "a bound within max_concurrency did not move");
    }

    // A profile's table on the Engine: depths clamped to the engine's window, none on a plain one.
    {
        using ninferno::product::engine_load;
        const std::vector<ninferno::spec::LoadBound> table{{2, 7}, {3, 3}, {5, 0}};
        const auto clamped = engine_load(table, 5);
        failures += check(clamped.size() == 3 && clamped[0].lanes == 2 &&
                              clamped[0].draft_tokens == 5 && clamped[1].draft_tokens == 3 &&
                              clamped[2].lanes == 5 && clamped[2].draft_tokens == 0,
                          "engine_load clamps depths to the window and keeps lanes");
        failures += check(engine_load(table, 0).empty(), "a plain engine takes no bounds");
        const auto widths = engine_load(ninferno::spec::parse_load("2:7:block,3:3:drafts,5:0"), 7);
        failures += check(widths.size() == 3 &&
                              widths[0].draft_width == ninferno::DFlashDraftWidth::Block &&
                              widths[1].draft_width == ninferno::DFlashDraftWidth::Drafts &&
                              widths[2].draft_width == ninferno::DFlashDraftWidth::Default,
                          "engine_load carries a bound's draft width");

        // --spec auto drops a width its speculation does not run, with a note, rather than refuse
        // (the Engine refuses a width on an MTP round, and Qwen 3.5 refuses block).
        using ninferno::DFlashDraftWidth;
        using ninferno::product::fit_draft_widths;
        const auto named = [](SpeculativeBackend backend) {
            ninferno::SpeculativeOptions options;
            options.backend     = backend;
            options.draft_width = DFlashDraftWidth::Block;
            options.load = engine_load(ninferno::spec::parse_load("2:3:block,4:2:drafts"), 7);
            return options;
        };
        using ninferno::spec::engine_support;
        const ninferno::spec::EngineSupport dense_family = engine_support("dense_decoder");
        const ninferno::spec::EngineSupport qwen_family  = engine_support("qwen3_5_text");

        auto kept = named(SpeculativeBackend::DFlash);
        failures += check(dense_family.dflash_block_width &&
                              fit_draft_widths(kept, dense_family).empty() &&
                              kept.draft_width == DFlashDraftWidth::Block &&
                              kept.load[0].draft_width == DFlashDraftWidth::Block &&
                              kept.load[1].draft_width == DFlashDraftWidth::Drafts,
                          "a dense DFlash drafter runs both widths");
        auto qwen_block       = named(SpeculativeBackend::DFlash2);
        const auto qwen_notes = fit_draft_widths(qwen_block, qwen_family);
        failures += check(!qwen_family.dflash_block_width && qwen_notes.size() == 2 &&
                              qwen_block.draft_width == DFlashDraftWidth::Default &&
                              qwen_block.load[0].draft_width == DFlashDraftWidth::Default &&
                              qwen_block.load[1].draft_width == DFlashDraftWidth::Drafts &&
                              qwen_block.load[0].draft_tokens == 3,
                          "Qwen 3.5 drops block and keeps drafts, with notes");
        auto mtp_options = named(SpeculativeBackend::Mtp);
        failures += check(fit_draft_widths(mtp_options, dense_family).size() == 2 &&
                              mtp_options.draft_width == DFlashDraftWidth::Default &&
                              !ninferno::models::qwen3_5::detail::names_draft_width(mtp_options),
                          "an MTP pick or fit fallback drops every width");

        // A published table plain from three requests (Muse Glimmer 30B's gb203 table until
        // 2026-09-29) reaches SpeculativeOptions::load: two rows speculate at the profile's depth,
        // three and more decode plainly, and no speculative launch is planned wider than two rows.
        const std::vector<ninferno::spec::LoadBound> muse{{.lanes = 3, .depth = 0}};
        ninferno::SpeculativeOptions options;
        options.backend      = SpeculativeBackend::DFlash;
        options.draft_tokens = 3;
        options.load         = engine_load(muse, ninferno::product::speculative_window(options));
        namespace plan       = ninferno::models::qwen3_5::detail;
        failures += check(options.load.size() == 1 && options.load[0].lanes == 3 &&
                              options.load[0].draft_tokens == 0 &&
                              plan::speculation_load_valid(options),
                          "the Muse table reaches SpeculativeOptions::load as 3:0");
        constexpr auto unbounded = std::numeric_limits<std::uint32_t>::max();
        failures += check(plan::speculation_load_bound(options.load, 1) == unbounded &&
                              plan::speculation_load_bound(options.load, 2) == unbounded &&
                              plan::speculation_load_bound(options.load, 3) == 0 &&
                              plan::speculation_load_bound(options.load, 8) == 0,
                          "Muse: below the first entry no bound, from three rows plain");
        failures += check(plan::speculative_batch_limit(options.load, 8) == 2 &&
                              plan::speculative_batch_limit(options.load, 2) == 2 &&
                              plan::speculative_batch_limit({}, 8) == 8,
                          "no speculative launch from a bound of 0's lanes on");
        options.load = {{.lanes = 2, .draft_tokens = 3}, {.lanes = 4, .draft_tokens = 0}};
        failures += check(plan::speculation_load_bound(options.load, 3) == 3 &&
                              plan::speculation_load_valid(options),
                          "a step between two entries takes the lower one's bound");
        // The DFlash drafter's width per round: the whole block for one row, K + 1 columns from
        // two, unless a load bound or the options name one.
        const auto width = [&](DFlashDraftWidth forced, std::size_t rows) {
            return plan::dflash_draft_width(forced, options.load, rows);
        };
        failures += check(width(DFlashDraftWidth::Default, 1) == DFlashDraftWidth::Block &&
                              width(DFlashDraftWidth::Default, 2) == DFlashDraftWidth::Drafts &&
                              width(DFlashDraftWidth::Default, 8) == DFlashDraftWidth::Drafts &&
                              !plan::names_draft_width(options),
                          "the default draft width: block at one row, drafts from two");
        options.load = {{.lanes = 2, .draft_tokens = 3, .draft_width = DFlashDraftWidth::Block},
                        {.lanes = 4, .draft_tokens = 3}};
        failures +=
            check(width(DFlashDraftWidth::Default, 1) == DFlashDraftWidth::Block &&
                      width(DFlashDraftWidth::Default, 3) == DFlashDraftWidth::Block &&
                      width(DFlashDraftWidth::Default, 4) == DFlashDraftWidth::Drafts &&
                      width(DFlashDraftWidth::Drafts, 1) == DFlashDraftWidth::Drafts &&
                      width(DFlashDraftWidth::Block, 8) == DFlashDraftWidth::Block &&
                      plan::names_draft_width(options) && plan::speculation_load_valid(options),
                  "a bound's draft width applies to its rows; the options' to every round");
        options.load        = {};
        options.draft_width = DFlashDraftWidth::Drafts;
        failures += check(plan::names_draft_width(options), "a forced draft width is named");
        options.draft_width = DFlashDraftWidth::Default;
        for (const std::vector<ninferno::SpeculationLoad>& bad :
             {std::vector<ninferno::SpeculationLoad>{{.lanes = 1, .draft_tokens = 2}},
              std::vector<ninferno::SpeculationLoad>{{.lanes = 2, .draft_tokens = 0},
                                                   {.lanes = 3, .draft_tokens = 2}},
              std::vector<ninferno::SpeculationLoad>{{.lanes = 2, .draft_tokens = 4}}}) {
            options.load = bad;
            failures += check(!plan::speculation_load_valid(options),
                              "one-lane, after-plain and wider-than-window bounds are refused");
        }

        // The batches each launch window runs: Qwen3.6 on gb203, menu 2,5,7,11 with 2:3 at C=8
        // (depth 3 from two lanes). The deeper windows run one row only; 2 every batch, 3 from 2.
        const auto batches = [](std::initializer_list<std::uint32_t> list) {
            std::uint32_t mask = 0;
            for (const std::uint32_t batch : list) { mask |= 1U << (batch - 1U); }
            return mask;
        };
        const std::vector<std::uint32_t> qwen =
            plan::launch_window_batches({2, 3, 5, 7, 11}, {1, 2, 1, 1, 1},
                                        {{.lanes = 2, .draft_tokens = 3}}, 8, 8);
        failures += check(qwen == std::vector<std::uint32_t>{batches({1, 2, 3, 4, 5, 6, 7, 8}),
                                                             batches({2, 3, 4, 5, 6, 7, 8}),
                                                             batches({1}), batches({1}),
                                                             batches({1})},
                          "2:3 at C=8: windows 5, 7 and 11 run one row only");
        // Unbounded: every window every batch from its own first.
        const std::vector<std::uint32_t> unbounded_masks =
            plan::launch_window_batches({2, 5}, {1, 1}, {}, 4, 4);
        failures += check(unbounded_masks == std::vector<std::uint32_t>{batches({1, 2, 3, 4}),
                                                                        batches({1, 2, 3, 4})},
                          "without bounds every window runs every batch");
        // Gemma 4 31B on gb202 (2:7,3:3,5:5, menu 3,7): a launch of 3 or 4 rows in a step of 5
        // rows or more runs depth 5, whose graphs start at 5 rows, so it takes window 7; a bound
        // of 0 at 7 lanes ends every window at 6 rows.
        const std::vector<std::uint32_t> gemma = plan::launch_window_batches(
            {3, 5, 7}, {1, 5, 1},
            {{.lanes = 2, .draft_tokens = 7}, {.lanes = 3, .draft_tokens = 3},
             {.lanes = 5, .draft_tokens = 5}, {.lanes = 7, .draft_tokens = 0}},
            8, 6);
        failures += check(gemma == std::vector<std::uint32_t>{batches({1, 2, 3, 4, 5, 6}),
                                                              batches({5, 6}),
                                                              batches({1, 2, 3, 4})},
                          "a non-monotone table keeps the windows its bounds reach at each batch");
        failures += check(plan::window_runs_batch(batches({2}), 2) &&
                              !plan::window_runs_batch(batches({2}), 1) &&
                              !plan::window_runs_batch(batches({2}), 0),
                          "a window mask holds its batches only");

        // Every speculative launch the decode step can make finds its window, and that window's
        // mask holds the launch's batch (launch_mtp throws otherwise): menus x load tables x
        // concurrency, each step of `rows` rows launching `batch` of them (the rest plain) at
        // every depth up to the step's bound (the engine's window when unbounded). The window is
        // checked against its definition, not against launch_window_of's own loop: windows
        // ascend, it covers the depth and the batch, and no narrower one does. The layouts do not
        // depend on the backend (the options below say MTP), so the launches up to the bound hold
        // for MTP and DFlash alike. Deeper launches come from DFlash only: a row still appending
        // the features of a wider round launches deeper than the bound (up to the engine's window,
        // ProgramImpl::launch_depth); such a launch still finds a window, whose graphs may leave
        // it out (it then runs without one, ungraphed_rounds). MTP rows never exceed the bound.
        const auto covers = [](const plan::LaunchWindowShape& shape, std::uint32_t depth,
                               std::uint32_t batch) {
            return shape.drafts >= depth && shape.min_batch <= batch;
        };
        const auto launches_hold = [&](const ninferno::SpeculativeOptions& speculative,
                                       std::uint32_t concurrency) {
            const auto shapes = plan::launch_window_shapes(speculative, concurrency);
            std::vector<std::uint32_t> depths;
            std::vector<std::uint32_t> firsts;
            for (const plan::LaunchWindowShape& shape : shapes) {
                if (!depths.empty() && shape.drafts <= depths.back()) { return -1; }
                depths.push_back(shape.drafts);
                firsts.push_back(shape.min_batch);
            }
            // The window a launch takes, by its definition.
            const auto narrowest = [&](std::size_t w, std::uint32_t depth, std::uint32_t batch) {
                bool ok = covers(shapes[w], depth, batch);
                for (std::size_t v = 0; v < w; ++v) { ok = ok && !covers(shapes[v], depth, batch); }
                return ok;
            };
            const std::uint32_t limit =
                shapes.empty() ? 0U : plan::speculative_batch_limit(speculative.load, concurrency);
            const auto masks =
                plan::launch_window_batches(depths, firsts, speculative.load, concurrency, limit);
            std::uint32_t launches = 0;
            for (std::uint32_t rows = 1; rows <= concurrency; ++rows) {
                const std::uint32_t bound =
                    shapes.empty()
                        ? 0U
                        : std::min(plan::speculation_load_bound(speculative.load, rows),
                                   depths.back());
                for (std::uint32_t batch = 1; batch <= std::min(rows, limit); ++batch) {
                    for (std::uint32_t depth = 1; depth <= bound; ++depth) {
                        const auto w = plan::launch_window_of(shapes, depth, batch);
                        if (!w || !narrowest(*w, depth, batch) ||
                            !plan::window_runs_batch(masks.at(*w), batch)) {
                            return -1;
                        }
                        ++launches;
                    }
                    // DFlash rows appending a wider round's features (not MTP): past the bound.
                    for (std::uint32_t depth = bound + 1; bound > 0 && depth <= depths.back();
                         ++depth) {
                        const auto w = plan::launch_window_of(shapes, depth, batch);
                        if (!w || !narrowest(*w, depth, batch)) { return -1; }
                    }
                }
            }
            return static_cast<int>(launches);
        };
        using Load                    = std::vector<ninferno::SpeculationLoad>;
        const std::vector<Load> loads = {
            {},
            {{.lanes = 2, .draft_tokens = 3}},
            {{.lanes = 2, .draft_tokens = 3}, {.lanes = 3, .draft_tokens = 0}},
            {{.lanes = 2, .draft_tokens = 7},
             {.lanes = 3, .draft_tokens = 3},
             {.lanes = 5, .draft_tokens = 5},
             {.lanes = 7, .draft_tokens = 0}},
            {{.lanes = 3, .draft_tokens = 1},
             {.lanes = 4, .draft_tokens = 2},
             {.lanes = 6, .draft_tokens = 0}},
            {{.lanes = 2, .draft_tokens = 0}},
            {{.lanes = 2, .draft_tokens = 5}, {.lanes = 4, .draft_tokens = 2}}};
        const std::vector<std::vector<std::uint32_t>> menus = {{7}, {2, 5, 7, 11}, {3, 7}, {1, 4}};
        int walked                                          = 0;
        bool held                                           = true;
        for (const auto& menu : menus) {
            for (const Load& load : loads) {
                for (std::uint32_t concurrency = 1; concurrency <= 8; ++concurrency) {
                    ninferno::SpeculativeOptions speculative;
                    speculative.request_draft_tokens.assign(menu.begin(), menu.end() - 1);
                    speculative.backend      = SpeculativeBackend::Mtp;
                    speculative.draft_tokens = menu.back();
                    speculative.load         = load;
                    if (!plan::speculation_load_valid(speculative)) { continue; }
                    const int launches = launches_hold(speculative, concurrency);
                    held               = held && launches >= 0;
                    walked += launches > 0 ? 1 : 0;
                }
            }
        }
        failures += check(held && walked > 100,
                          "a launch's window does not run its batch (" + std::to_string(walked) +
                              " layouts walked)");
        // One request: every depth from one row, every window runs batch 1 only.
        ninferno::SpeculativeOptions single;
        single.backend              = SpeculativeBackend::DFlash;
        single.draft_tokens         = 7;
        single.request_draft_tokens = {3};
        single.load                 = {{.lanes = 2, .draft_tokens = 3}};
        failures += check(launches_hold(single, 1) == 7 &&
                              plan::launch_window_shapes(single, 1).size() == 2 &&
                              plan::launch_window_batches({3, 7}, {1, 1}, single.load, 1, 1) ==
                                  std::vector<std::uint32_t>{batches({1}), batches({1})},
                          "C=1: both windows run one row, every depth finds one");
        // Plain decode: no windows, no masks, no launch.
        const ninferno::SpeculativeOptions none;
        failures += check(plan::launch_window_shapes(none, 4).empty() &&
                              plan::launch_window_batches({}, {}, {}, 4, 4).empty() &&
                              !plan::launch_window_of(std::vector<plan::LaunchWindowShape>{},
                                                      1, 1) &&
                              launches_hold(none, 4) == 0,
                          "an empty window list plans and finds nothing");

        // Stopped rows ride at the width of the window the launch takes: menu 3,7 with 4:5 at C=8
        // (depth 5 captured from four rows). A drafting row at depth 5 in a launch of three rows
        // takes window 7 (width 8), in one of four window 5 (width 6).
        ninferno::SpeculativeOptions riding;
        riding.backend              = SpeculativeBackend::Mtp;
        riding.draft_tokens         = 7;
        riding.request_draft_tokens = {3};
        riding.load                 = {{.lanes = 4, .draft_tokens = 5}};
        const auto shapes           = plan::launch_window_shapes(riding, 8);
        const auto three            = plan::launch_window_of(shapes, 5, 3);
        const auto four             = plan::launch_window_of(shapes, 5, 4);
        failures += check(shapes.size() == 3 && shapes[1].drafts == 5 &&
                              shapes[1].min_batch == 4 && three == 2U && four == 1U &&
                              !plan::launch_window_of(shapes, 8, 4),
                          "depth 5 takes window 7 below four rows and window 5 from four");
        failures += check(plan::stopped_rows_ride(shapes, *three, 2, 16) &&
                              !plan::stopped_rows_ride(shapes, *three, 2, 15) &&
                              plan::stopped_rows_ride(shapes, *four, 2, 12) &&
                              !plan::stopped_rows_ride(shapes, *four, 3, 16),
                          "stopped rows ride within the columns of the launch's own window");

        // Copy rounds (ngram_draft_tokens): MTP 3 at C=4 with 2:2 adds the copy window 15 last,
        // from one row; a depth-3 row still takes window 3, a 15-token copy the copy window. Under
        // the bound of 2 from two rows only batch 1 runs the copy window. DFlash has no copy
        // window.
        ninferno::SpeculativeOptions copies;
        copies.backend            = SpeculativeBackend::Mtp;
        copies.draft_tokens       = 3;
        copies.load               = {{.lanes = 2, .draft_tokens = 2}};
        copies.ngram_draft_tokens = 15;
        const auto copy_shapes    = plan::launch_window_shapes(copies, 4);
        failures += check(copy_shapes.size() == 3 && copy_shapes[0].drafts == 2 &&
                              copy_shapes[1].drafts == 3 && copy_shapes[2].drafts == 15 &&
                              copy_shapes[2].min_batch == 1 &&
                              plan::launch_window_of(copy_shapes, 3, 1) == 1U &&
                              plan::launch_window_of(copy_shapes, 15, 1) == 2U,
                          "the copy window comes last, from one row");
        failures +=
            check(plan::launch_window_batches({2, 3, 15}, {2, 1, 1}, copies.load, 4, 4).back() ==
                      batches({1}),
                  "a load bound below the copy window keeps it to the rows it allows");
        ninferno::SpeculativeOptions dflash_copies = copies;
        dflash_copies.backend                    = SpeculativeBackend::DFlash;
        failures += check(plan::launch_window_shapes(dflash_copies, 4).back().drafts == 3,
                          "a DFlash engine has no copy window");

        // The copy proposal: the continuation of the most recent occurrence of the longest
        // recurring suffix, at most `limit` tokens; nothing below the minimum match.
        {
            using ninferno::models::qwen3_5::detail::copy_proposal;
            std::int32_t out[16] = {};
            // 1 2 3 4 5 6 | 9 | 1 2 3 4 5 6 | 7 8 | 3 4 5 6: suffix 3 4 5 6 recurs twice; the
            // most recent occurrence of the longest match (4) continues with 7 8 3 4 5 6.
            const std::vector<std::int32_t> h{1, 2, 3, 4, 5, 6, 9, 1, 2, 3,
                                              4, 5, 6, 7, 8, 3, 4, 5, 6};
            const std::size_t n = copy_proposal(h, 3, 4, out);
            failures += check(n == 4 && out[0] == 7 && out[1] == 8 && out[2] == 3 && out[3] == 4,
                              "a copy proposal follows the most recent longest match");
            // A longer match earlier wins over a shorter recent one: x 1 2 3 4 a | 4 b | 1 2 3 4.
            const std::vector<std::int32_t> l{0, 1, 2, 3, 4, 50, 4, 60, 1, 2, 3, 4};
            failures += check(copy_proposal(l, 2, 3, out) == 3 && out[0] == 50 && out[1] == 4 &&
                                  out[2] == 60,
                              "the longest recurring suffix decides the copy");
            failures +=
                check(copy_proposal(l, 5, 3, out) == 0 && copy_proposal(l, 2, 0, out) == 0 &&
                          copy_proposal(std::vector<std::int32_t>{7}, 1, 3, out) == 0,
                      "no copy below the minimum match, without a limit or a history");
        }
    }

    std::filesystem::remove_all(dir);
    if (failures == 0) { std::cout << "product speculation: ok\n"; }
    return failures == 0 ? 0 : 1;
}
