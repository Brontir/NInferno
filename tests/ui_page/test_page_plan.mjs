// ninferno-ui's page: the plan helpers (the block between "// ---- plan helpers (tested) ----" and
// "// ---- end plan helpers ----" in src/ui/page.html) over synthetic ninferno-dry-plan/1 documents
// (round 2: states, the pick, codes), the page's English and Russian words for every code, and the
// whole page script parses.
// Usage: node test_page_plan.mjs [path/to/page.html]
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {dirname, join} from "node:path";
import vm from "node:vm";
import assert from "node:assert/strict";

const page = process.argv[2] || join(dirname(fileURLToPath(import.meta.url)), "../../src/ui/page.html");
const html = readFileSync(page, "utf8");

// Every inline script of the page parses.
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
assert.ok(scripts.length >= 1, "the page has its script");
for (const body of scripts) new vm.Script(body, {filename: "page.html <script>"});

const BEGIN = "// ---- plan helpers (tested) ----", END = "// ---- end plan helpers ----";
const from = html.indexOf(BEGIN), to = html.indexOf(END);
assert.ok(from >= 0 && to > from, "the plan helpers block is marked in the page");
const NAMES = ["parseCtxText", "planTok", "ctxInput", "ctxNoteText", "planSpecOf", "planQuery", "catalogQuery", "planStateOf",
               "planRowOf", "launchContextOf", "reasonText", "unsupportedText", "cardGbOf", "cardGbText", "ruleText", "planRowState",
               "runGate", "rowEstimate", "choiceFor", "choiceRows", "quantChangeNotice", "busyGpuModels", "planMemory", "planSpecSummary", "specVariantsLine", "specBasisText", "launchSettingsOf",
               "REASON_CODES", "UNSUPPORTED_CODES", "RULE_TIERS", "SPEC_BASES"];
const ctx = vm.createContext({});
vm.runInContext(html.slice(from, to) + `\n;globalThis.H = {${NAMES.join(", ")}};`, ctx);
const H = ctx.H;

// The page's words: the dictionary block (T, from "const T = {" to "const LANGS = ["), run alone.
const wFrom = html.indexOf("const T = {"), wTo = html.indexOf("const LANGS = [");
assert.ok(wFrom >= 0 && wTo > wFrom, "the words block is in the page");
const wctx = vm.createContext({});
vm.runInContext(html.slice(wFrom, wTo).replace("const T = {", "globalThis.T = {"), wctx);
const T = wctx.T;
const fill = (s, v) => s.replace(/\{(\w+)\}/g, (m, k) => v[k] ?? m);
const words = lang => (k, v) => {
  const s = T[lang][k] ?? T.en[k];
  assert.ok(typeof s === "string", `a word for ${k}`);
  return fill(s, v || {});
};
const EN = words("en"), RU = words("ru");
// Keys only: a test sees which text and with what.
const tr = (k, v) => v ? k + JSON.stringify(v) : k;
let checks = 0;
const plain = v => v === undefined ? v : JSON.parse(JSON.stringify(v));
const eq = (a, b, what) => { assert.deepEqual(plain(a), plain(b), what); checks++; };
const ok = (v, what) => { assert.ok(v, what); checks++; };
// A rendered text is whole: no {placeholder} left, no key in place of words, no letter x for ×.
const whole = (s, what) => {
  ok(typeof s === "string" && s.length > 0, what + ": not empty");
  ok(!/\{\w+\}/.test(s), what + ": no placeholder left: " + s);
  ok(!/\d x \d|\bauto x\b/.test(s), what + ": × not x: " + s);
};
const cyr = s => /[а-яё]/i.test(s);

// ---- parseCtxText, ctxInput ----
eq(H.parseCtxText(""), 0, "empty is auto");
eq(H.parseCtxText("auto"), 0, "auto");
eq(H.parseCtxText(" Auto "), 0, "Auto with spaces");
eq(H.parseCtxText("авто"), 0, "Russian auto");
eq(H.parseCtxText("32k"), 32768, "32k");
eq(H.parseCtxText("128K"), 131072, "128K");
eq(H.parseCtxText("1M"), 1048576, "1M");
eq(H.parseCtxText("1.5k"), 1536, "1.5k");
eq(H.parseCtxText("32 768"), 32768, "grouped by a space");
eq(H.parseCtxText("32,768"), 32768, "grouped by a comma");
eq(H.parseCtxText("64к"), 65536, "Cyrillic k");
eq(H.parseCtxText("abc"), null, "not a context");
eq(H.parseCtxText("32kk"), null, "two units");
eq(H.parseCtxText("-5"), null, "negative");
eq(H.parseCtxText("99M"), null, "above the server's 16M");
// Typed context against a 262K model: above native is taken down and said, below 1K raised, nonsense refused.
eq(H.ctxInput("1M", 262144), {ctx: 262144, note: "max"}, "1M on a 262K model");
eq(H.ctxInput("100", 262144), {ctx: 1024, note: "min"}, "100 is raised to 1K");
eq(H.ctxInput("abc", 262144), {ctx: null, note: "bad"}, "nonsense");
eq(H.ctxInput("", 262144), {ctx: 0, note: ""}, "empty is Auto");
eq(H.ctxInput("128k", 262144), {ctx: 131072, note: ""}, "a plain value");
eq(H.ctxNoteText("max", 262144, tr), 'ctxMaxNote{"max":"256K"}', "the max note");
eq(H.ctxNoteText("min", 262144, tr), 'ctxMinNote{"min":"1K"}', "the min note");
eq(H.ctxNoteText("bad", 262144, tr), "ctxTypeBad", "the bad note");
eq(H.ctxNoteText("", 262144, tr), "", "no note");
eq(H.ctxNoteText("max", 262144, RU), "для этой модели максимум 256K: взято 256K", "RU max note");
eq(H.ctxNoteText("max", 262144, EN), "max 256K for this model: taken as 256K", "EN max note");
whole(H.ctxNoteText("min", 262144, RU), "RU min note");
whole(H.ctxNoteText("bad", 262144, RU), "RU bad note");

// ---- planTok, spec, queries ----
eq(H.planTok(32768), "32K", "32K");
eq(H.planTok(1048576), "1M", "1M");
eq(H.planTok(500), "500", "below 1K");
// The page sends auto; the server decides (no page-side plain verdict).
eq(H.planSpecOf("off"), "none", "off is none");
eq(H.planSpecOf("auto"), "auto", "auto stays auto");
eq(H.planSpecOf("dflash2"), "dflash2", "manual kind");
eq(H.planQuery({repo: "unsloth/Qwen3.6-27B-GGUF", hub: "huggingface", gpu: 1, ctx: 0, par: 2, kv: "auto", spec: "auto"}),
   "/api/plan?repo=unsloth%2FQwen3.6-27B-GGUF&hub=huggingface&gpu=1&ctx=auto&par=2&kv=auto&spec=auto", "auto context");
eq(H.planQuery({repo: "a/b", gpu: 0, quant: "UD-Q4_K_XL", ctx: 32768, par: 16, kv: "fp8", spec: "none", precision: "exact"}),
   "/api/plan?repo=a%2Fb&hub=huggingface&gpu=0&quant=UD-Q4_K_XL&ctx=32768&par=8&kv=fp8&spec=none&precision=exact",
   "explicit context, par clamped to 8, quant and precision");
eq(H.catalogQuery(1, 0, 4), "/api/catalog?gpu=1&ctx=auto&par=4", "catalog at the user's auto × 4");
eq(H.catalogQuery(0, 131072, 20), "/api/catalog?gpu=0&ctx=131072&par=8", "catalog at 128K × 8 (clamped)");

// ---- a synthetic round-2 document: a 24 GB card, Auto × 4 ----
const spec = {served: "mtp:3", backend: "mtp", depth: 3, basis: "own", depth_at_concurrency: 2};
const chosen = {context: 49152, kv_tokens: 200000, survival: false, needed_bytes: 6e9,
                bytes: {kv: 4e9, states: 5e8, rings: 1e8, workspace: 3e8, graphs: 3.2e7, fixed: 0, total: 4.932e9}};
const plan = o => ({planned: true, fits: true, kv_dtype: "int8", context: 49152, pool_contexts: 4.1, private_contexts: true,
  description: "48K x 4", weights_bytes: 16e9, free_bytes: 23e9, free_after_weights_bytes: 7e9, class_reserve_bytes: 3e9,
  chosen, reason: "", reason_code: "", reason_params: {}, fits_instead: [], speculation: spec, ...o});
const row = o => ({supported: true, why_unsupported: "", unsupported_code: "", largest: false, below_floor: false, low_quality: false,
                   own_context: 0, launch_context: 0, drafter_bytes: 0, weights_basis: "gguf_tensors", ...o});
const shortP = {weights_bytes: 21e9, needed_bytes: 3.5e9, free_after_weights_bytes: 2.1e9, short_bytes: 1.4e9, context: 8192, concurrency: 4};
const doc = {schema: "ninferno-dry-plan/1", ask: {context: "auto", concurrency: 4},
  device: {name: "NVIDIA GeForce RTX 4090", card_gb: 24, free_basis: "idle_card"},
  pick: {quant: "Q4_K_M", tier: "own", target_context: 32768, context: 49152, concurrency: 4, card_gb: 24, below_floor: false, low_quality: false},
  rule: "quant | Q4_K_M: the largest that fits auto x 4 on NVIDIA GeForce RTX 4090 (23.75 GiB class)",
  rule_code: "largest_fits",
  rule_params: {quant: "Q4_K_M", tier: "own", context: 49152, concurrency: 4, card_gb: 24, automatic: true},
  quants: [
    row({quant: "UD-IQ2_M", state: "own", own_context: 98304, launch_context: 98304, below_floor: true, low_quality: true,
         fits: true, plan: plan({context: 98304})}),
    row({quant: "Q4_K_M", state: "own", own_context: 49152, launch_context: 49152, fits: true, largest: false, plan: plan({})}),
    row({quant: "Q6_K", state: "shared", fits: false, fits_shared: true,
         plan: plan({private_contexts: false, pool_contexts: 1.2, context: 32768, chosen: {...chosen, kv_tokens: 9216}})}),
    row({quant: "Q8_0", state: "desktop", fits: false,
         plan: plan({chosen: {...chosen, survival: true}, reason_code: "survival_only",
                     reason_params: {context: 16384, reserve_bytes: 3221225472}, reason: "fits only on a survival rung"})}),
    row({quant: "UD-Q8_K_XL", state: "no", fits: false,
         plan: plan({fits: false, context: 0, chosen: null, reason_code: "smallest_plan_short", reason_params: shortP,
                     reason: "the weights leave 2.1 GiB, a context needs 3.5 GiB",
                     fits_instead: [{concurrency: 1, context: 65536}, {concurrency: 2, context: 32768}, {concurrency: 2, context: 32768},
                                    {concurrency: 16, context: 4096}, {concurrency: 4, context: 0}]})}),
    row({quant: "Q2_K", state: "unknown", fits: false,
         plan: {planned: false, fits: false, context: 0, chosen: null, reason_code: "shape_unknown", reason_params: {},
                reason: "the model's shape is unknown"}}),
    row({quant: "BF16", state: "unsupported", supported: false, unsupported_code: "above_8_bits", why_unsupported: "above 8 bits per weight",
         fits: false, plan: null}),
  ],
  chosen: "Q4_K_M", plan: plan({})};

// planRowOf: the pick decides the star (the row's own `largest` is overridden by doc.pick).
eq(H.planRowOf(doc, "Q4_K_M").largest, true, "the pick is starred");
eq(H.planRowOf(doc, "UD-IQ2_M").largest, false, "only the pick");
eq(H.planRowOf(doc, "Q2_K_XL"), null, "no row for an unlisted label");
eq(H.planRowOf(null, "Q4_K_M"), null, "no document");
eq(H.planRowOf({...doc, pick: null}, "Q4_K_M").largest, false, "nothing fits: no star");
// An older document (no pick key) keeps the row's own largest.
eq(H.planRowOf({quants: [{quant: "A", largest: true, fits: true}]}, "A").largest, true, "an older document");

// Each state.
const ask4 = {ctx: 0, par: 4};
const S = q => H.planRowState(H.planRowOf(doc, q), ask4, tr);
let s = S("Q4_K_M");
eq([s.state, s.cls, s.star, s.run, s.label, s.ctxText, s.ctxOut],
   ["own", "full", true, true, "fitsYes", 'ctxAutoWord → ctxEach{"ctx":"48K"}', 'ctxAuto → ctxEach{"ctx":"48K"}'], "own, Auto × 4");
eq(s.tip, 'fitsYesTip{"ctx":"48K","par":4}', "own tooltip");
eq(s.ctxTip, 'poolHolds{"n":4.1,"ctx":"48K"}', "the cache holds, not the C++ description");
eq(H.planRowState(H.planRowOf(doc, "Q4_K_M"), {ctx: 0, par: 1}, tr).tip, 'fitsYesTip1{"ctx":"48K","par":1}', "one request");
s = S("Q6_K");
eq([s.state, s.cls, s.run, s.label, s.ctxText], ["shared", "tight", true, "fitsShared", 'ctxAutoWord → ctxShared{"ctx":"9K","n":4}'],
   "shared: 9K shared by 4, never ×4");
eq(s.warn, 'fitsSharedTip{"n":4,"pool":"9K"}', "shared warning");
s = S("Q8_0");
eq([s.state, s.cls, s.run, s.label], ["desktop", "desk", true, "fitsDesktop"], "desktop: yellow, Run enabled");
eq(s.tip, 'fitsDesktopTip{"reserve":"3.0 GiB"}', "desktop: the reserve given up");
s = S("UD-Q8_K_XL");
eq([s.state, s.cls, s.run, s.star, s.ctxText], ["no", "no", true, false, "—"], "no: red, Run enabled (M3)");
eq(s.warn, "noFitRun", "no: the start may fail");
eq(s.instead, [{ctx: 65536, par: 1, text: "64K × 1"}, {ctx: 32768, par: 2, text: "32K × 2"}], "fits instead");
ok(s.why.startsWith("reason_smallest_plan_short"), "no: the reason from its code");
eq(H.planRowState(H.planRowOf(doc, "UD-Q8_K_XL"), {...ask4, picked: true}, tr).warn, "pickedNoFit", "an explicit pick that no longer fits warns");
s = S("Q2_K");
eq([s.state, s.cls, s.run, s.label, s.ctxText], ["unknown", "unknown", true, "fitUnknown", "ctxAutoWord × 4"], "unknown: not planned, Run enabled");
ok(s.why.startsWith("reason_shape_unknown"), "unknown: why from its code");
s = S("BF16");
eq([s.state, s.cls, s.run, s.star], ["unsupported", "off", false, false], "unsupported: Run off");
eq(s.why, "unsup_above_8_bits", "unsupported: the code's words");
eq(H.planRowState(null, ask4, tr).state, "unknown", "no row: not planned");
// No fit without a reason code or text: the asked context × streams.
eq(H.planRowState({supported: true, state: "no", plan: {fits: false}}, {ctx: 131072, par: 3}, tr).why,
   'noFitAt{"ctx":"128K","par":3}', "no reason at all");
// An unknown code: the server's text is the fallback.
eq(H.planRowState({supported: true, state: "no", plan: {fits: false, reason_code: "new_thing", reason: "server words"}}, ask4, tr).why,
   "server words", "unknown reason code");
// A pinned context, own: "128K each"; shared: the pool shared.
s = H.planRowState(row({state: "own", plan: plan({context: 131072})}), {ctx: 131072, par: 2}, tr);
eq([s.ctxText, s.ctxOut], ['ctxEach{"ctx":"128K"}', 'ctxEach{"ctx":"128K"}'], "pinned, own");
eq(H.planRowState(row({state: "own", plan: plan({context: 131072})}), {ctx: 131072, par: 1}, tr).ctxText, "128K", "pinned, one request");
// Pick tiers and floors: low_quality warns on the row.
s = S("UD-IQ2_M");
eq([s.lowq, s.lowTip], [true, "lowQuality"], "below 3 bits warns");
eq(S("Q4_K_M").lowq, false, "no warning above 3 bits");
// The stale flag rides along (the page greys the star and the dots).
eq(H.planRowState(H.planRowOf(doc, "Q4_K_M"), {...ask4, stale: true}, tr).stale, true, "stale");

// Older documents and one model: the state from fits / fits_shared / the plan.
eq(H.planStateOf({supported: true, fits: true, plan: {}}), "own", "legacy fits");
eq(H.planStateOf({supported: true, fits_shared: true, plan: {}}), "shared", "legacy shared");
eq(H.planStateOf({supported: true, plan: {planned: true, fits: true, chosen: {survival: true}}}), "desktop", "legacy survival");
eq(H.planStateOf({supported: true, plan: {planned: false}}), "unknown", "legacy not planned");
eq(H.planStateOf({supported: false}), "unsupported", "legacy unsupported");
let r = H.planRowOf({quants: [], plan: plan({}), pick: {quant: "", tier: "own", low_quality: true}}, "");
eq([r.state, r.largest, r.low_quality], ["own", true, true], "one model, own");
eq(H.planRowOf({quants: [], plan: plan({private_contexts: false})}, "").state, "shared", "one model, shared");
eq(H.planRowOf({quants: [], plan: plan({chosen: {...chosen, survival: true}})}, "").state, "desktop", "one model, desktop");
eq(H.planRowOf({quants: [], plan: plan({fits: false, chosen: null})}, "").state, "no", "one model, no");
eq(H.planRowOf({quants: [], plan: {planned: false}}, "").state, "unknown", "one model, not planned");

// launch_context under Auto.
eq(H.launchContextOf(H.planRowOf(doc, "Q4_K_M"), {ctx: 0, par: 4}), 49152, "Auto × 4: the row's launch_context");
eq(H.launchContextOf(row({state: "own", launch_context: 0, plan: plan({})}), {ctx: 0, par: 1}), 0, "Auto × 1: automatic");
eq(H.launchContextOf(H.planRowOf(doc, "Q6_K"), {ctx: 0, par: 4}), 0, "shared: automatic");
eq(H.launchContextOf(H.planRowOf(doc, "Q4_K_M"), {ctx: 65536, par: 4}), 65536, "typed context wins");
eq(H.launchContextOf({state: "own", plan: plan({context: 40960})}, {ctx: 0, par: 3}), 40960, "no launch_context: own context at C > 1");
eq(H.launchContextOf({state: "own", plan: plan({context: 40960})}, {ctx: 0, par: 1}), 0, "no launch_context: automatic at 1");
eq(H.launchContextOf(null, {ctx: 0, par: 4}), 0, "no plan: automatic");

// Run: never disabled on an estimate; only unsupported, a bad typed context, or a plan in flight.
const gate = (q, f) => H.runGate(S(q), f, tr);
eq(gate("UD-Q8_K_XL", {}), {disabled: false, why: "noFitRun"}, "no: Run enabled, warned");
eq(gate("Q8_0", {}).disabled, false, "desktop: Run enabled");
eq(gate("Q2_K", {}).disabled, false, "unknown: Run enabled");
eq(gate("Q4_K_M", {}), {disabled: false, why: ""}, "own: Run enabled");
eq(gate("BF16", {}), {disabled: true, why: "unsup_above_8_bits"}, "unsupported: Run off");
eq(gate("Q4_K_M", {unsupported: true}), {disabled: true, why: "unsupportedRun"}, "the architecture is not run");
eq(gate("Q4_K_M", {ctxBad: true}), {disabled: true, why: "ctxTypeBad"}, "invalid typed context: Run off");
eq(gate("Q4_K_M", {busy: true}), {disabled: true, why: "planning"}, "a stale plan: Run off until the plan for the ask is in");

// The quant-changed notice.
const prev = {quant: "Q5_K_M", bytes: 19.1}, next = {quant: "IQ4_XS", bytes: 14.9};
eq(H.quantChangeNotice(prev, next, {ctx: 131072, par: 2}, tr),
   'quantChanged{"from":"Q5_K_M","to":"IQ4_XS","why":"quantWhyMore{\\"ask\\":\\"2 × 128K\\"}"}', "smaller: needs more memory");
eq(H.quantChangeNotice(next, prev, {ctx: 0, par: 1}, tr),
   'quantChanged{"from":"IQ4_XS","to":"Q5_K_M","why":"quantWhyRoom{\\"ask\\":\\"ctxAutoWord\\"}"}', "larger: room for a larger one");
eq(H.quantChangeNotice(prev, next, {ctx: 131072, par: 2, picked: true}, tr), "", "never for an explicit pick");
eq(H.quantChangeNotice(prev, next, {ctx: 131072, par: 2, stale: true}, tr), "", "never from a stale plan");
eq(H.quantChangeNotice(prev, {...prev}, {ctx: 0, par: 2}, tr), "", "no change");
eq(H.quantChangeNotice(null, next, {ctx: 0, par: 2}, tr), "", "nothing before");
eq(H.quantChangeNotice(prev, next, {ctx: 131072, par: 2}, RU), "Выбран другой квант: Q5_K_M → IQ4_XS (для 2 × 128K нужно больше памяти)",
   "RU notice");
eq(H.quantChangeNotice(prev, next, {ctx: 131072, par: 2}, EN), "A different quantization was picked: Q5_K_M → IQ4_XS (2 × 128K needs more memory)",
   "EN notice");

// card_gb and the rule line.
eq(H.cardGbOf(doc), 24, "card_gb from device");
eq(H.cardGbOf({pick: {card_gb: 16}}), 16, "card_gb from the pick");
eq(H.cardGbOf({}), 0, "no card_gb");
eq(H.cardGbText(24, EN), "24 GB", "EN 24 GB");
eq(H.cardGbText(24, RU), "24 ГБ", "RU 24 ГБ");
const aCtx = {ctx: 0, par: 4, native: 262144};
eq(H.ruleText(doc, aCtx, EN), "Q4_K_M, the largest that fits this 24 GB card at auto context (≈ 48K) × 4: each request has its own 48K", "EN rule");
eq(H.ruleText(doc, aCtx, RU), "Q4_K_M — самый большой квант, который влезает на эту карту на 24 ГБ при авто-контексте (≈ 48K) × 4: у каждого запроса свои 48K",
   "RU rule");
ok(!H.ruleText(doc, aCtx, RU).includes("GiB class"), "RU rule: not the C++ text");
eq(H.ruleText({...doc, rule_code: "brand_new"}, aCtx, RU), doc.rule, "an unknown rule code: the server's text");
eq(H.ruleText(null, aCtx, EN), "", "no document");

// Every rule code, tier, reason code and unsupported code, in both languages: whole words, RU in Russian.
const tierDoc = tier => ({...doc, pick: {...doc.pick, tier, target_context: 16384},
                          rule_params: {...doc.rule_params, tier, context: tier.startsWith("shared") || tier === "desktop" ? 9216 : 16384}});
for (const tier of H.RULE_TIERS) {
  for (const [name, W] of [["EN", EN], ["RU", RU]]) {
    const text = H.ruleText(tierDoc(tier), aCtx, W);
    whole(text, `${name} rule, tier ${tier}`);
    ok(text.includes(":"), `${name} rule, tier ${tier}: the tier's words`);
    if (name === "RU") ok(cyr(text), `RU rule, tier ${tier}: Russian`);
  }
}
const noneDoc = {...doc, pick: null, rule_code: "none_fits",
  rule_params: {context: 0, concurrency: 4, card_gb: 24, automatic: true, smallest: "UD-Q8_K_XL", reason_code: "smallest_plan_short", fits_instead: []}};
for (const [name, W] of [["EN", EN], ["RU", RU]]) {
  const text = H.ruleText(noneDoc, aCtx, W);
  whole(text, name + " none_fits");
  if (name === "RU") ok(cyr(text) && !text.includes("the weights leave"), "RU none_fits in Russian");
}
eq(H.ruleText(noneDoc, aCtx, EN),
   "Nothing fits this 24 GB card at auto context × 4 (not even UD-Q8_K_XL): after the weights 2.0 GiB is left, and even 8K × 4 needs 3.3 GiB (1.3 GiB short)",
   "EN none_fits");
eq(H.ruleText({...doc, rule_code: "none_served", rule_params: {}}, aCtx, RU), "Движок не запускает ни один квант этого репозитория", "RU none_served");
whole(H.ruleText({...doc, rule_code: "none_served", rule_params: {}}, aCtx, EN), "EN none_served");
// A pinned ask: "128K × 2".
eq(H.ruleText({...doc, rule_params: {...doc.rule_params, automatic: false, context: 131072, concurrency: 2}}, {ctx: 131072, par: 2}, EN),
   "Q4_K_M, the largest that fits this 24 GB card at 128K × 2: each request has its own 128K", "EN pinned rule");
const reasonParams = {shape_unknown: {}, weights_too_big: {weights_bytes: 30e9, free_bytes: 22e9}, smallest_plan_short: shortP,
                      survival_only: {context: 16384, reserve_bytes: 3221225472}};
for (const code of H.REASON_CODES) {
  for (const [name, W] of [["EN", EN], ["RU", RU]]) {
    const text = H.reasonText({reason_code: code, reason_params: reasonParams[code], reason: "C++ english"}, W);
    whole(text, `${name} reason ${code}`);
    ok(!text.includes("C++ english"), `${name} reason ${code}: from the code`);
    if (name === "RU") ok(cyr(text), `RU reason ${code}: Russian`);
  }
}
eq(H.reasonText({reason_code: "weights_too_big", reason_params: reasonParams.weights_too_big}, RU),
   "одни веса (27.9 GiB) больше, чем свободно на карте (20.5 GiB)", "RU weights_too_big");
eq(H.reasonText({reason_code: "", reason: "older server text"}, RU), "older server text", "no code: the server's text");
for (const code of H.UNSUPPORTED_CODES) {
  for (const [name, W] of [["EN", EN], ["RU", RU]]) {
    const text = H.unsupportedText({unsupported_code: code, why_unsupported: "C++ english"}, W);
    whole(text, `${name} unsupported ${code}`);
    ok(!text.includes("C++ english"), `${name} unsupported ${code}: from the code`);
    if (name === "RU") ok(cyr(text), `RU unsupported ${code}: Russian`);
  }
}
eq(H.unsupportedText({unsupported_code: "", why_unsupported: "server words"}, RU), "server words", "no code: the server's text");
eq(H.unsupportedText({}, tr), "unsupportedRun", "nothing at all");
// The row words of every state, both languages.
for (const q of ["UD-IQ2_M", "Q4_K_M", "Q6_K", "Q8_0", "UD-Q8_K_XL", "Q2_K", "BF16"]) {
  for (const [name, W] of [["EN", EN], ["RU", RU]]) {
    const st = H.planRowState(H.planRowOf(doc, q), ask4, W);
    for (const k of ["label", "tip", "ctxText", "ctxOut", "why", "warn", "lowTip", "ctxTip"]) {
      if (st[k] && st[k] !== "—") whole(st[k], `${name} ${q} ${k}`);
    }
    if (name === "RU") ok(cyr(st.label), `RU ${q}: Russian label`);
  }
}
eq(H.planRowState(H.planRowOf(doc, "Q6_K"), ask4, RU).ctxOut, "Авто → 9K на всех 4", "RU shared box");
eq(H.planRowState(H.planRowOf(doc, "Q6_K"), ask4, EN).ctxOut, "Auto → 9K shared by 4", "EN shared box");
eq(H.planRowState(H.planRowOf(doc, "Q4_K_M"), ask4, RU).ctxOut, "Авто → 48K на каждый", "RU own box");
eq(H.planRowState(H.planRowOf(doc, "Q4_K_M"), ask4, EN).ctxOut, "Auto → 48K each", "EN own box");
eq(H.planRowState(H.planRowOf(doc, "Q8_0"), ask4, RU).label, "Влезает, но рабочий стол может подтормаживать", "RU desktop label");
eq(H.planRowState(H.planRowOf(doc, "UD-Q8_K_XL"), ask4, RU).warn, "По расчёту не влезает; запуск может не стартовать", "RU no warning");
eq(H.planRowState(H.planRowOf(doc, "UD-IQ2_M"), ask4, EN).lowTip, "⚠ quality drops below 3 bits", "EN low quality");
// The words that changed meaning: Auto is "each its own at least 32K", the streams do not promise own caches.
ok(T.en.ctxAutoTip.includes("32K") && T.ru.ctxAutoTip.includes("32K"), "ctxAutoTip names the 32K target");
ok(!/each takes its own cache/.test(T.en.parTip) && !/у каждого свой кеш/.test(T.ru.parTip) && !/у кожного свій кеш/.test(T.uk.parTip),
   "parTip does not promise each its own cache");
for (const code of ["de", "fr", "es", "it", "pl", "zh", "ja"]) ok(T[code].parTip === undefined, code + " parTip falls back to English");
// No jargon in the words this page shows for the plan.
for (const lang of ["en", "ru"]) {
  for (const [k, v] of Object.entries(T[lang])) {
    if (/survival rung|graphs CUDA|графы CUDA/i.test(v)) assert.fail(`${lang}.${k}: jargon: ${v}`);
  }
  const shared = H.planRowState(H.planRowOf(doc, "Q6_K"), ask4, words(lang));
  ok(!/pool/i.test(shared.label + shared.tip + shared.ctxText), lang + ": no 'pool' in the shared words");
}
checks++;

// The busy GPU.
const procs = [{kind: "serve", state: "ready", gpu: 0, model: "/m/hub/Qwen3.6-27B-Q4_K_M.ninferno"},
               {kind: "serve", state: "exited", gpu: 0, model: "old"}, {kind: "prepare", state: "running", gpu: 0, model: "dl"},
               {kind: "serve", state: "starting", gpu: 1, model: "other/repo"}];
eq(H.busyGpuModels(procs, 0), ["Qwen3.6-27B-Q4_K_M"], "the model running on GPU 0");
eq(H.busyGpuModels(procs, 1), ["repo"], "starting counts");
eq(H.busyGpuModels(procs, 2), [], "an idle GPU");

// planMemory
const m = H.planMemory(H.planRowOf(doc, "Q4_K_M").plan);
eq([m.weights, m.kv, m.states, m.rings, m.workspace, m.graphs, m.need, m.free, m.freeAfter, m.reserve],
   [16e9, 4e9, 5e8, 1e8, 3e8, 3.2e7, 22e9, 23e9, 7e9, 3e9], "memory parts");
eq(H.planMemory(H.planRowOf(doc, "UD-Q8_K_XL").plan), null, "no step, no memory");

// planSpecSummary and the basis words.
const sp = H.planSpecSummary(spec);
// The variants by the thinking flag: both, with the second's distance from the first; none when one.
{
  const both = H.planSpecSummary({served: "dflash:15", backend: "dflash", basis: "measured",
    by_thinking: {thinking: {config: "dflash:15", speedup: 3.86}, non_thinking: {config: "dflash:7", speedup: 2.22}, per_request: true}});
  const line = H.specVariantsLine(both, EN);
  ok(line.includes("dflash:15 ×3.86") && line.includes("dflash:7 ×2.22 (−1.64)"), "both variants with the delta: " + line);
  const same = H.planSpecSummary({served: "mtp:4", backend: "mtp", basis: "measured",
    by_thinking: {thinking: {config: "mtp:4", speedup: 2}, non_thinking: {config: "mtp:4", speedup: 1.8}, per_request: false}});
  eq(H.specVariantsLine(same, EN), "", "one configuration for both: no line");
  eq(H.specVariantsLine(H.planSpecSummary({served: "plain", backend: "none", basis: "none"}), EN), "", "no profile: no line");
}
eq([sp.served, sp.kind, sp.basis, sp.depth], ["mtp:3", "mtp", "own", 2], "served speculation");
eq(H.planSpecSummary({served: "plain", backend: "none", depth: 0, basis: "none"}).kind, "", "plain decode");
eq(H.planSpecSummary(null), null, "no SPEC");
const plainMeasured = H.planSpecSummary({served: "plain", backend: "none", depth: 0, basis: "measured_plain"});
eq(H.specBasisText(plainMeasured, "auto", EN), "measured faster without it here", "EN measured_plain");
eq(H.specBasisText(plainMeasured, "auto", RU), "здесь без ускорения замерено быстрее", "RU measured_plain");
eq(H.specBasisText(plainMeasured, "off", tr), "specOffChosen", "Off chosen");
ok(!/profile/i.test(H.specBasisText(H.planSpecSummary({served: "plain", basis: "none"}), "auto", EN)), "no 'no profile' mislabel");
for (const b of H.SPEC_BASES) {
  whole(H.specBasisText({basis: b}, "auto", EN), "EN basis " + b);
  ok(cyr(H.specBasisText({basis: b}, "auto", RU)), "RU basis " + b);
}
ok(T.en.specAutoRuns === "Auto runs this" && T.ru.specAutoRuns === "Запустится в Авто", "the served-kind badge words");

// My models → Run: the model's own saved settings, with defaults.
eq(H.launchSettingsOf(null), {ctx: 0, cap: 0, par: 1, kv: "", spec: "auto"}, "defaults");
eq(H.launchSettingsOf({ctx: 49152, par: 4, kv: "fp8", spec: "none"}), {ctx: 49152, cap: 0, par: 4, kv: "fp8", spec: "none"}, "saved");
eq(H.launchSettingsOf({ctx: -1, par: 99, kv: 5, spec: ""}), {ctx: 0, cap: 0, par: 8, kv: "", spec: "auto"}, "bad saved values");
// A saved Auto x 4 launch runs again with its planned context as the ceiling (max_context_cap),
// never as a pool the streams silently share; a typed context wins over a stale cap.
eq(H.launchSettingsOf({ctx: 0, cap: 49152, par: 4, kv: "", spec: "auto"}), {ctx: 0, cap: 49152, par: 4, kv: "", spec: "auto"}, "saved Auto x 4 keeps its cap");
eq(H.launchSettingsOf({ctx: 65536, cap: 49152, par: 4}), {ctx: 65536, cap: 0, par: 4, kv: "", spec: "auto"}, "a typed context drops the cap");

// The page no longer decides speculation itself (P7): no autoVerdict, no spec=none from it.
ok(!/autoVerdict/.test(html), "autoVerdict is gone");
ok(!/sp\.config\)\}\$\{sp\.speedup/.test(html), "the old registry auto line is gone");

// The speeds are the server's choice model's (src/product/choice): the page keeps no model of its own.
ok(!/function genTps|KIND_LOAD|function calibOf|function prefillTps/.test(html), "the page's own speed model is gone");
eq(H.rowEstimate({estimate: {decode_tok_s: 80.5, prefill_tok_s: 5700, depth: 8192}}), {decode_tok_s: 80.5, prefill_tok_s: 5700, depth: 8192}, "a row's estimate");
eq(H.rowEstimate({estimate: null}), null, "no estimate");
eq(H.rowEstimate(null), null, "no row");
// The choice's alternatives as deltas against the chosen.
const choiceDoc = {choice: {chosen: {kv: "int8", activations: "fast", own_context: 65536, decode_tok_s: 80, prefill_tok_s: 5700, request_seconds: 6.6, kld: 0.0044},
  alternatives: [{axis: "kv", value: "bf16", fits: true, own_context: 40960, decode_tok_s: 79.2, prefill_tok_s: 5700, request_seconds: 6.65, kld: 0.0043, why_code: "lower_worth"},
                 {axis: "activations", value: "exact", fits: true, own_context: 61440, decode_tok_s: 70, prefill_tok_s: 1567, request_seconds: 8.3, kld: 0, why_code: "lower_worth"},
                 {axis: "kv", value: "nvfp4", fits: false, own_context: 0, why_code: "no_fit"}]}};
const cr = H.choiceRows(choiceDoc);
eq(cr.length, 3, "one row an alternative");
eq(cr[0].dctx, -24576, "BF16 KV holds less context each");
ok(Math.abs(cr[0].ddecode + 0.01) < 1e-9 && Math.abs(cr[1].dtime - (8.3 / 6.6 - 1)) < 1e-9, "decode and time as ratios");
ok(Math.abs(cr[1].dkld + 0.0044) < 1e-12, "exact: the KLD difference");
eq([cr[2].fits, cr[2].ddecode, cr[2].dkld, cr[2].why], [false, null, null, "no_fit"], "an alternative that does not fit has no speeds");
eq(H.choiceRows({choice: {chosen: null}}), [], "no choice: no rows");
eq(H.choiceRows(null), [], "no document");
// The worth's pick within the rule's tier: its own words, naming the largest that fits.
{
  const doc = {rule_code: "best_worth", rule_params: {quant: "Q6_K", largest: "Q8_0", tier: "own", automatic: true, concurrency: 1, card_gb: 24},
               pick: {quant: "Q6_K", tier: "own", target_context: 32768}};
  const en = H.ruleText(doc, {ctx: 0, par: 1, native: 262144}, words("en"));
  ok(en.startsWith("Q6_K, the best balance of speed and accuracy") && en.includes("the largest that fits is Q8_0"), "best_worth in words");
}
// The choice belongs to the quantization it was made at: the panel and the pick's speed only there.
const atQ4 = {choice: {chosen: {quant: "Q4_K_M"}}};
ok(H.choiceFor(atQ4, "Q4_K_M") && !H.choiceFor(atQ4, "Q8_0") && H.choiceFor(atQ4, ""), "a choice is for its quantization");
ok(!H.choiceFor({choice: {chosen: null}}, "Q4_K_M") && !H.choiceFor(null, "Q4_K_M"), "no choice: for none");
for (const lang of ["en", "ru"]) for (const k of ["choiceHead", "choiceLine", "choiceAlt", "estBasis", "why_no_fit", "why_lower_worth", "why_quant_rule", "ax_kv", "ax_activations", "ax_quant", "ax_drafter", "choiceDrafter", "why_default_drafter"])
  ok(typeof T[lang][k] === "string" && T[lang][k].length > 0, `${lang}: ${k}`);

console.log(`test_page_plan: ${checks} checks passed`);
