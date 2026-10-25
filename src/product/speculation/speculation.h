#pragma once

// The speculation an artifact came with, as the products run it: the CLI (one request, one
// engine: the profile's default for the request's thinking flag) and the server (one
// engine for every request: spec/menu.h lays the profile onto it, and each request picks its
// entry). The profile comes from spec/manifest.h's resolve_speculation() for this GPU; each entry
// point's `local_profile` is a maintainer's own tune named by --spec-profile (none: not read). Both run
// only what the Engine's family implements (spec/engine_support.h, by the artifact's model_type):
// a profile tuned where more runs (Gemma 4's Session: DFlash, adaptive depths, MTP windows past
// 5) carries its served choices, the best the tune measured among those, and they are taken; a
// choice the family cannot run is never handed to the Engine.

#include "ninferno/types.h"
#include "spec/config.h"
#include "spec/engine_support.h"
#include "spec/manifest.h"
#include "spec/menu.h"
#include "spec/profile.h"
#include "spec/registry.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninferno::product {

[[nodiscard]] SpeculativeBackend backend_of(spec::Mode mode) noexcept;

// Whether the artifact carries the proposal head that "+head" configurations draft with.
[[nodiscard]] bool has_proposal_head(const std::filesystem::path& artifact);

// The watchdog floor a speculative choice runs with: its measured break-even point, or 1.2 tokens
// per round when the profile recorded none.
[[nodiscard]] double watchdog_floor(double measured) noexcept;

// The speculation an artifact came with for this GPU (a maintainer's local tune, else the
// published profile in its <artifact>.speculation.json), as one engine configuration: nullopt when
// there is none, its drafter is not in the artifact or the Engine cannot run the choice (a note
// says which): the profile's thinking or non-thinking default. `source` names where it came from;
// `notes` explain passed-over sources. A "+head" choice on an artifact converted without the
// proposal head drafts with the full head (same drafter and K), and a note says so.
[[nodiscard]] std::optional<SpeculativeOptions>
profile_speculation(const std::filesystem::path& artifact, int device, std::optional<bool> thinking,
                    std::string* source, std::vector<std::string>* notes,
                    const std::filesystem::path& local_profile = {});

// The same for one choice of an already resolved profile (with the profile's context bound).
[[nodiscard]] std::optional<SpeculativeOptions>
speculation_of(const spec::Choice& choice, const spec::ResolvedSpeculation& resolved,
               bool proposal_head, std::vector<std::string>* notes);

// A choice a product took from the profile (or one of its fallbacks) with the settings the
// operator gives every configuration: the watchdog's window, minimum rounds and shared trip floor,
// and the copy rounds that ride MTP (--ngram-*). The profile picks what drafts and how deep; these
// stay as given, in the CLI as in the server.
[[nodiscard]] SpeculativeOptions with_given_settings(SpeculativeOptions picked,
                                                     const SpeculativeOptions& given);

// A server's speculation: one engine for every request, each request at its own entry, the
// engine bounded under load by the profile's table (spec::Profile::load).
struct ServedSpeculation {
    spec::SpeculationMenu menu;
    SpeculativeOptions engine; // backend, default depth, every depth of the menu, head, floor
    std::string source;        // where the profile came from
    std::string basis; // the same as data: own | similar | generic | builtin (ResolvedSpeculation)
    std::vector<std::string> notes;
    // Set when the profile's choice did not fit the requested context on this device and its fit
    // fallback runs instead: why (the Engine's refusal). Logged as a warning.
    std::string fit_fallback;
};

// What the served functions read of a target, resolved once: an artifact on disk and the device's
// profile key (speculation_subject), or what a dry run without the artifact (ninferno-serve --plan
// --no-load, ninferno-ui /api/plan) builds from the drafter plan the converter would write
// (product/plan/plan_speculation.h). The functions below that take a subject are the ones the
// artifact-and-device ones call: one code path for a start and a dry run.
struct SpeculationSubject {
    std::filesystem::path artifact; // empty: none yet (a dry run)
    // The resolved profile (spec::resolve_speculation), with the built-in drafter's default when
    // none was found; nullopt: the artifact's id is unreadable.
    std::optional<spec::ResolvedSpeculation> resolved;
    std::set<std::string> components; // the artifact's (text, mtp, dflash, dflash2, ...)
    bool proposal_head = false;       // it carries Qwen's proposal head
    spec::EngineSupport support;      // what the Engine's family runs
    // A dry run's drafters the converter's drafter plan takes from a repository of their own
    // (uploaded beside the target, or embedded in the artifact as a component: either way their
    // weights are not the target's and are counted); empty for an artifact.
    std::vector<spec::DrafterDecl> attached_drafters;
};

[[nodiscard]] SpeculationSubject
speculation_subject(const std::filesystem::path& artifact, int device,
                    const std::filesystem::path& local_profile = {});
[[nodiscard]] SpeculationSubject
speculation_subject(const std::filesystem::path& artifact, const spec::GpuKey& gpu,
                    const std::filesystem::path& local_profile = {});
// A subject whose profile was not found takes its built-in drafter's default (MTP 3, else
// DFlash2 or DFlash 7), as speculation_subject does for an artifact.
void apply_builtin_default(SpeculationSubject& subject);

// The subject with its profile chosen again for one speculation mode (the configuration choice's
// drafter axis, product/choice/choice.h): over the profile's measurements of that mode alone (a lab
// pair's, a local tune's), served and portable choices dropped. A profile without measurements is
// kept as it is (its own choice is its only one).
[[nodiscard]] SpeculationSubject subject_of_mode(SpeculationSubject subject, spec::Mode mode);

[[nodiscard]] std::optional<ServedSpeculation> served_speculation(const SpeculationSubject& subject,
                                                                  bool thinking, bool per_request);

// Whether each request should run at its own thinking flag's choice (--spec-select thinking) rather
// than every request at the server's default (`thinking`'s): where the profile's measurements say
// the requests of the other flag gain more than kPerRequestMargin (spec::per_request_gain over the
// configurations the Engine runs). A start without --spec-select, and the plan of one, take it.
// The margin is above the measurement's spread: tokens per round of one pair agree within 1.8 %
// across cards and 2.3 % across quantizations of 4 bits or more (choice/SPECULATION.md v4).
inline constexpr double kPerRequestMargin = 0.03;
[[nodiscard]] bool per_request_pays(const SpeculationSubject& subject, bool thinking);
// The served configuration one draft shallower (`config` with k - 1, its watchdog at `floor`),
// for every default: what a start runs when neither the profile's choice nor its fit fallback fits
// (a narrower round takes less workspace and graph memory). nullopt at depth 1, for plain decode,
// or when the subject has no profile with a drafter for the configuration's mode.
[[nodiscard]] std::optional<ServedSpeculation> served_shallower(const SpeculationSubject& subject,
                                                                bool per_request,
                                                                const spec::SpecConfig& config,
                                                                double floor);
// The profile's fit fallback (spec::Profile::fit_fallback): what a start runs when the Engine
// refuses the profile's choice with CapacityError (its drafter does not fit the requested context
// on this device), as a menu whose every entry is the fallback. nullopt when the profile has none,
// or the Engine cannot run it. Never for an explicit --spec.
[[nodiscard]] std::optional<ServedSpeculation>
served_fit_fallback(const SpeculationSubject& subject, bool per_request);
// The model's own MTP head in place of a separate drafter (a start's last step before plain
// decode, and the pool rule's): the built-in MTP default (as for a model with no profile), with the
// registry's watchdog floors (spec::raise_foreign_floors). nullopt when the profile's choice
// already drafts with MTP, the model carries no MTP head, or the Engine cannot run it.
[[nodiscard]] std::optional<ServedSpeculation>
served_head_fallback(const SpeculationSubject& subject, bool per_request);

// nullopt when the artifact came with no speculation profile for this GPU. `thinking` is the
// server's thinking default: the profile's default for it is the engine's default (primary)
// configuration. With `per_request` false every request runs the primary (a fixed configuration),
// and the engine captures its depth and the shallower menu depths it steps down to; with it true a
// request runs the default for its own thinking flag.
[[nodiscard]] std::optional<ServedSpeculation>
served_speculation(const std::filesystem::path& artifact, int device, bool thinking,
                   bool per_request, const std::filesystem::path& local_profile = {});

// The deepest configuration shallower than `tried` that `fits` accepts, with the same drafter: a
// server whose profile's choice does not fit the requested context speculates at the deepest depth
// that does before it decodes plainly. `shallower` is the server's served_shallower (asked for the
// depth below a configuration); `fits` builds the Engine for a candidate, replacing any Engine an
// earlier call built. Fit is monotone in depth (a depth that fits means every shallower one does),
// so it tries one draft shallower first (a near miss), then depth 1 (when that does not fit,
// nothing does), then bisects between them: a handful of builds from any depth, not one per
// depth. The candidate returned is the one `fits` built last; nullopt when none fits.
[[nodiscard]] std::optional<ServedSpeculation> deepest_fitting_shallower(
    spec::MenuChoice tried,
    const std::function<std::optional<ServedSpeculation>(const spec::SpecConfig&, double)>&
        shallower,
    const std::function<bool(const ServedSpeculation&)>& fits);

// The start's order for --spec auto, one code for every place a start is judged: the server
// (GenerationService builds the Engine), the CLI (ninferno builds it) and the plan without the
// weights (product/plan/plan_speculation.h plans it). Each passes only its own `fits`.
//
// After the profile's choice `refused` does not fit: the profile's fit fallback, then the deepest
// shallower draft (deepest_fitting_shallower) from the last choice tried (the fit fallback when
// the profile has one, else `refused`), then the model's own MTP head. `fits` builds or plans a
// candidate at its step and says whether it fits, replacing whatever an earlier call built; the
// candidate returned is the one `fits` accepted last. nullopt: none fits, and the caller decodes
// plainly.
enum class FallbackStep { FitFallback, Shallower, Head };
[[nodiscard]] const char* fallback_step_name(FallbackStep step) noexcept; // fit_fallback, ...
[[nodiscard]] std::optional<ServedSpeculation>
fit_fallbacks(const SpeculationSubject& subject, const ServedSpeculation& refused, bool per_request,
              const std::function<bool(const ServedSpeculation&, FallbackStep)>& fits);

// The pool rule (--spec auto with an automatic KV pool): when the served drafter leaves the pool
// `with_drafter` whole contexts, fewer than `concurrency`, and plain decode holds more
// (`plain_contexts`, 0 when plain decode does not fit), requests waiting for pages cost more than
// drafting saves. The model's own MTP head runs then when it holds at least four fifths of the
// contexts plain decode does (or of every one served; `head_contexts`, nullopt when it does not
// fit), else plain decode. One context always fits a drafter, so a server of one request never
// loses it. `plain_contexts` is called only inside the rule's range, `head_contexts` only when
// plain decode holds more; each builds or plans its configuration as `fits` above does.
enum class PoolKeep { Drafter, Head, Plain };
struct PoolDecision {
    PoolKeep keep = PoolKeep::Drafter;
    std::optional<ServedSpeculation> head; // PoolKeep::Head: the head served
    std::uint32_t plain = 0;               // the contexts plain decode holds (0: not asked)
};
[[nodiscard]] PoolDecision
pool_rule(const SpeculationSubject& subject, bool per_request, std::uint32_t with_drafter,
          std::uint32_t concurrency, const std::function<std::uint32_t()>& plain_contexts,
          const std::function<std::optional<std::uint32_t>(const ServedSpeculation&)>& head_contexts);

// The profile served_shallower serves: every choice `config` one draft shallower, with the drafter
// the profile runs `config`'s mode with and the floor shallower_floor finds from `floor` (measured
// at `config`) and the profile's own floors for that mode; no fit fallback or served choices.
// None for plain decode, depth 1, or a mode the profile does not run.
[[nodiscard]] std::optional<spec::Profile>
shallower_profile(const spec::Profile& profile, const spec::SpecConfig& config, double floor);

// The watchdog floor (a round's cost in plain-decode tokens, spec::Choice) at `depth` from floors
// measured at other depths of the same configuration ({depth, floor}): the measured one at
// `depth`, else linear between the nearest depths around it, else scaled from the nearest as
// 1 + (floor - 1) * depth / its depth (a round of no drafts costs one token), so always above 1.
// A floor of 1 or less is no measurement (every round emits a token; watchdog_floor serves such a
// floor as its default) and is ignored. 0 when none is measured, which watchdog_floor also serves
// as its default.
[[nodiscard]] double shallower_floor(const std::vector<std::pair<int, double>>& floors, int depth);

// A tune named by --spec-profile (`local_profile`) that does not serve this artifact on this GPU
// (unreadable, tuned for another artifact, GPU or drafters, or a drafter replaced since) refuses to
// start: throws std::runtime_error naming it and why (spec::require_named_tune), whatever --spec
// says. Every function here that takes `local_profile` throws the same. Nothing when none is named.
void require_named_tune(const std::filesystem::path& artifact, int device,
                        const std::filesystem::path& local_profile);

// Why the tune at `tune` would not serve `artifact` on `device` (spec::tune_refusal with the
// device's class); empty when it would. For a tune found beside the artifact rather than named.
[[nodiscard]] std::string tune_refusal(const std::filesystem::path& artifact, int device,
                                       const std::filesystem::path& tune);

// Why the artifact came with no speculation profile for this GPU (each source passed over, such
// as no manifest beside it); empty when served_speculation finds one.
[[nodiscard]] std::vector<std::string> speculation_notes(const std::filesystem::path& artifact,
                                                         int device, const std::filesystem::path& local_profile = {});


// What the Engine runs of the artifact's speculation (none when its model_type is unreadable).
[[nodiscard]] spec::EngineSupport engine_support(const std::filesystem::path& artifact);

// The drafter artifact that came with the model for `backend` when it ships separately from the
// target (Gemma 4's MTP assistant, the NemotronH MTP head; named in <artifact>.speculation.json);
// empty when the model came with none or carries its drafter inside the artifact (Qwen). Nothing
// is searched for elsewhere.
[[nodiscard]] std::filesystem::path model_drafter(const std::filesystem::path& artifact,
                                                  SpeculativeBackend backend);

// What a request runs for an entry of `menu`: its depth (0: plain decode), watchdog floor and the
// shallower depths a trip steps it down to (SpeculationMenu::step_down).
[[nodiscard]] RequestSpeculation request_speculation(const spec::SpeculationMenu& menu,
                                                     const spec::MenuChoice& choice);

// The Engine's load bounds (SpeculativeOptions::load) for a profile's table on an engine whose
// deepest captured depth is `window` (SpeculativeOptions: the largest of draft_tokens and
// request_draft_tokens): each depth at most the window. Empty for a plain engine (window 0).
[[nodiscard]] std::vector<SpeculationLoad> engine_load(const std::vector<spec::LoadBound>& load,
                                                       std::uint32_t window);
// The DFlash draft widths `options` names (SpeculativeOptions::draft_width and its load bounds')
// that its speculation runs: none unless it drafts with DFlash or DFlash2 (a profile's table read
// by an MTP fit fallback, an MTP pick), and no Block where the family drafts only the anchor and
// the drafts (spec::EngineSupport::dflash_block_width). What it drops falls back to the Engine's
// own choice (DFlashDraftWidth::Default) and a note says so: --spec auto never fails to start over
// a width. Returns the notes.
[[nodiscard]] std::vector<std::string> fit_draft_widths(SpeculativeOptions& options,
                                                        const spec::EngineSupport& support);
// That window of `options` (0: a plain engine).
[[nodiscard]] std::uint32_t speculative_window(const SpeculativeOptions& options) noexcept;
// A bound depth the menu does not capture costs the Engine a set of speculative graphs of its own
// (about 130 MiB for DFlash at C=8). When that does not fit, a product moves each such bound to the
// deepest captured depth within it (the shallowest captured depth when none is) and retries. A
// bound whose lanes exceed `max_concurrency` plans nothing (no step is that wide) and stays. True
// when a bound moved, so a retry can plan differently.
[[nodiscard]] bool snap_load_to_captured(SpeculativeOptions& options,
                                         std::uint32_t max_concurrency);

} // namespace ninferno::product
