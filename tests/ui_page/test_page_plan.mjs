// ninferno-ui's page: the plan helpers (the block between "// ---- plan helpers (tested) ----" and
// "// ---- end plan helpers ----" in src/ui/page.html) over synthetic ninferno-dry-plan/1 documents,
// and the whole page script parses.
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
const ctx = vm.createContext({});
vm.runInContext(html.slice(from, to) +
  "\n;globalThis.H = {parseCtxText, planTok, planSpecOf, planQuery, planRowOf, planRowState, planMemory, planSpecSummary};", ctx);
const H = ctx.H;
// The words: the key and its values, so a test sees which text and with what.
const tr = (k, v) => v ? k + JSON.stringify(v) : k;
let checks = 0;
// Values made in the vm context have its prototypes: compare them as plain data.
const plain = v => v === undefined ? v : JSON.parse(JSON.stringify(v));
const eq = (a, b, what) => { assert.deepEqual(plain(a), plain(b), what); checks++; };

// parseCtxText
eq(H.parseCtxText(""), 0, "empty is auto");
eq(H.parseCtxText("auto"), 0, "auto");
eq(H.parseCtxText(" Auto "), 0, "Auto with spaces");
eq(H.parseCtxText("авто"), 0, "Russian auto");
eq(H.parseCtxText("32k"), 32768, "32k");
eq(H.parseCtxText("128K"), 131072, "128K");
eq(H.parseCtxText("1M"), 1048576, "1M");
eq(H.parseCtxText("1.5k"), 1536, "1.5k");
eq(H.parseCtxText("32768"), 32768, "tokens");
eq(H.parseCtxText("32 768"), 32768, "grouped by a space");
eq(H.parseCtxText("32,768"), 32768, "grouped by a comma");
eq(H.parseCtxText("64к"), 65536, "Cyrillic k");
eq(H.parseCtxText("abc"), null, "not a context");
eq(H.parseCtxText("-5"), null, "negative");
eq(H.parseCtxText("99M"), null, "above the server's 16M");

// planTok
eq(H.planTok(32768), "32K", "32K");
eq(H.planTok(1048576), "1M", "1M");
eq(H.planTok(100352), "98K", "98K");
eq(H.planTok(500), "500", "below 1K");

// planSpecOf
eq(H.planSpecOf("off", false), "none", "off is none");
eq(H.planSpecOf("auto", false), "auto", "auto");
eq(H.planSpecOf("auto", true), "none", "auto, measured plain");
eq(H.planSpecOf("dflash2", true), "dflash2", "manual kind");

// planQuery
eq(H.planQuery({repo: "unsloth/Qwen3.6-27B-GGUF", hub: "huggingface", gpu: 1, ctx: 0, par: 2, kv: "auto", spec: "auto"}),
   "/api/plan?repo=unsloth%2FQwen3.6-27B-GGUF&hub=huggingface&gpu=1&ctx=auto&par=2&kv=auto&spec=auto", "auto context");
eq(H.planQuery({repo: "a/b", gpu: 0, quant: "UD-Q4_K_XL", ctx: 32768, par: 16, kv: "fp8", spec: "none", precision: "exact"}),
   "/api/plan?repo=a%2Fb&hub=huggingface&gpu=0&quant=UD-Q4_K_XL&ctx=32768&par=8&kv=fp8&spec=none&precision=exact",
   "explicit context, par clamped to 8, quant and precision");
eq(H.planQuery({repo: "a/b", ctx: 4096, par: 0}), "/api/plan?repo=a%2Fb&hub=huggingface&gpu=0&ctx=4096&par=1&kv=auto&spec=auto",
   "defaults");

// A synthetic /api/plan document: a fit, a shared fit, no fit with what fits instead, an unsupported row.
const spec = {served: "mtp:3", backend: "mtp", depth: 3, basis: "own", depth_at_concurrency: 2,
              classes: [{class: "short", below_tokens: 1024, depth: 3, basis: "modeled"},
                        {class: "huge", below_tokens: null, depth: 0, basis: "no estimate"}]};
const plan = (o) => ({planned: true, fits: true, kv_dtype: "int8", context: 100352, pool_contexts: 2.4, private_contexts: true,
  description: "98K x 2", weights_bytes: 16e9, free_bytes: 23e9, free_after_weights_bytes: 7e9, class_reserve_bytes: 1.5e9,
  chosen: {context: 100352, kv_tokens: 240000, survival: false, needed_bytes: 6e9,
           bytes: {kv: 4e9, states: 5e8, rings: 1e8, workspace: 3e8, graphs: 3.2e7, fixed: 0, total: 4.932e9}},
  reason: "", fits_instead: [], speculation: spec, ...o});
const doc = {schema: "ninferno-dry-plan/1", ask: {context: "auto", concurrency: 2},
  quants: [
    {quant: "Q4_K_M", supported: true, why_unsupported: "", fits: true, fits_shared: false, largest: false, plan: plan({})},
    {quant: "Q6_K", supported: true, why_unsupported: "", fits: true, fits_shared: false, largest: true, plan: plan({context: 65536})},
    {quant: "Q8_0", supported: true, why_unsupported: "", fits: false, fits_shared: true, largest: false,
     plan: plan({private_contexts: false, pool_contexts: 1.2, chosen: {...plan({}).chosen, kv_tokens: 81920}})},
    {quant: "UD-Q8_K_XL", supported: true, why_unsupported: "", fits: false, fits_shared: false, largest: false,
     plan: plan({fits: false, context: 0, chosen: null, reason: "the weights leave 2.1 GiB, a context needs 3.0 GiB",
                 fits_instead: [{concurrency: 1, context: 65536}, {concurrency: 2, context: 32768}, {concurrency: 2, context: 32768},
                                {concurrency: 16, context: 4096}, {concurrency: 4, context: 0}]})},
    {quant: "BF16", supported: false, why_unsupported: "above 8 bits per weight", fits: false, fits_shared: false, largest: false, plan: null},
  ], chosen: "Q6_K", rule: "quant | Q6_K: the largest that fits auto x 2", plan: plan({})};

eq(H.planRowOf(doc, "Q6_K").largest, true, "row by label");
eq(H.planRowOf(doc, "Q2_K"), null, "no row for an unlisted label");
eq(H.planRowOf(null, "Q6_K"), null, "no document");

// Auto context: "auto → 64K ×2", the star.
let s = H.planRowState(H.planRowOf(doc, "Q6_K"), {ctx: 0, par: 2}, tr);
eq([s.cls, s.star, s.run, s.ctxText, s.label], ["full", true, true, "ctxAutoWord → 64K ×2", "fitsYes"], "fits, auto context");
eq(s.tip, 'fitsYesTip{"ctx":"64K","par":2}', "fits tooltip");
eq(s.ctxTip.split("\n"), ["98K x 2", 'poolHolds{"n":2.4,"ctx":"64K"}'], "context tooltip");
// An explicit context: no "auto →".
s = H.planRowState(H.planRowOf(doc, "Q4_K_M"), {ctx: 100352, par: 1}, tr);
eq([s.cls, s.star, s.ctxText], ["full", false, "98K"], "fits, explicit context, one request");
// Shared: runs, with a warning naming the streams and the pool.
s = H.planRowState(H.planRowOf(doc, "Q8_0"), {ctx: 32768, par: 4}, tr);
eq([s.cls, s.run, s.label], ["tight", true, 'fitsShared{"n":4}'], "fits shared");
eq(s.warn, 'fitsSharedTip{"n":4,"pool":"80K"}', "shared warning: 4 streams, 80K tokens");
// No fit: the reason, what fits instead (deduplicated, 1..8 streams, a context), Run off.
s = H.planRowState(H.planRowOf(doc, "UD-Q8_K_XL"), {ctx: 0, par: 2}, tr);
eq([s.cls, s.run, s.star, s.ctxText, s.why], ["no", false, false, "—", "the weights leave 2.1 GiB, a context needs 3.0 GiB"], "does not fit");
eq(s.instead, [{ctx: 65536, par: 1, text: "64K × 1"}, {ctx: 32768, par: 2, text: "32K × 2"}], "fits instead");
eq(s.runWhy, 'fitsNotTip{"reason":"the weights leave 2.1 GiB, a context needs 3.0 GiB"}', "Run's reason");
// No fit without a reason: the asked context x streams.
s = H.planRowState({supported: true, fits: false, fits_shared: false, plan: {fits: false, reason: ""}}, {ctx: 131072, par: 3}, tr);
eq(s.why, 'noFitAt{"ctx":"128K","par":3}', "no fit, no reason");
// Unsupported.
s = H.planRowState(H.planRowOf(doc, "BF16"), {ctx: 0, par: 1}, tr);
eq([s.cls, s.run, s.why], ["off", false, "above 8 bits per weight"], "unsupported");
// No plan for the row: shown without the fit, the server decides.
s = H.planRowState(null, {ctx: 0, par: 2}, tr);
eq([s.cls, s.run, s.ctxText, s.star], ["unknown", true, "ctxAutoWord ×2", false], "no plan");

// One model (no rows): the model's plan stands for its quantization.
const single = {quants: [], plan: plan({}), rule: ""};
let r = H.planRowOf(single, "");
eq([r.fits, r.fits_shared, r.largest], [true, false, true], "single model fits");
r = H.planRowOf({quants: [], plan: plan({private_contexts: false})}, "");
eq([r.fits, r.fits_shared, r.largest], [false, true, false], "single model shared");
r = H.planRowOf({quants: [], plan: plan({fits: false, chosen: null, context: 0})}, "");
eq([r.fits, r.fits_shared], [false, false], "single model does not fit");

// planMemory
const m = H.planMemory(H.planRowOf(doc, "Q4_K_M").plan);
eq([m.weights, m.kv, m.states, m.rings, m.workspace, m.graphs, m.need, m.free, m.freeAfter, m.reserve],
   [16e9, 4e9, 5e8, 1e8, 3e8, 3.2e7, 22e9, 23e9, 7e9, 1.5e9], "memory parts");
eq(H.planMemory(H.planRowOf(doc, "UD-Q8_K_XL").plan), null, "no step, no memory");

// planSpecSummary
const sp = H.planSpecSummary(spec);
eq([sp.served, sp.kind, sp.basis, sp.depth], ["mtp:3", "mtp", "own", 2], "served speculation");
eq(sp.classes, [{cls: "short", below: 1024, depth: 3, basis: "modeled"}, {cls: "huge", below: 0, depth: 0, basis: "no estimate"}], "classes");
eq(H.planSpecSummary({served: "plain", backend: "none", depth: 0, basis: "none"}).kind, "", "plain decode");
eq(H.planSpecSummary({served: "dflash2:7", backend: "dflash2", depth: 7, depth_at_concurrency: 0, basis: "similar"}).depth, 7,
   "depth without a load table");
eq(H.planSpecSummary(null), null, "no SPEC");

console.log(`test_page_plan: ${checks} checks passed`);
