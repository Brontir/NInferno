#!/usr/bin/env python3
"""Generate NInferno's performance pages from measured data.

Three levels, each one deeper than the last:
  README.md                                the summary line against llama.cpp and the GPU x model table
                                           (single stream / all streams), between the perf markers
  docs/performance/README.md               the same table for every model measured
  docs/performance/gpus/<gpu>.md           level 2: every model on one GPU, its best setting
  docs/performance/models/<model>.md       level 2: one model on every GPU, its best setting
  docs/performance/runs/<gpu>/<model>.md   level 3: every setting measured for the pair, the commands
                                           that reproduce them, and the quality measurements

Inputs (docs/performance/data/):
  bench.jsonl   one measurement per line: machine {gpu_name, sm, vram_gb, power_limit_w, driver,
                cuda}, model (Hugging Face id or name), recipe ("org/repo --quant LABEL"), spec,
                draft_tokens, kv, metric, prompt_len, prompt_set, concurrency, context, repeats,
                value, spread, unit, date, version, flags, engine (default "ninferno")
                metric: prefill | decode (one stream after a long prompt) | decode_aggregate
                (short chats, all streams together) | perplexity | acceptance | greedy_match
  gpus.json     {"<slug>": {"name", "chip", "sm", "vram_gb", "match": [nvidia-smi names]}}
  models.json   {"<slug>": {"name", "hf", "family", "params", "label": [line, line],
                "readme": true when the project README's table shows it}}, in table order

Rows whose engine is not "ninferno" (llama.cpp on the same weights and GPU) are references: they
feed the summary line and the level-3 comparison, never the speed tables.

--check exits 1 (printing a diff) when a page on disk differs from what the data generates.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "performance" / "data"
OUT = ROOT / "docs" / "performance"
README = ROOT / "README.md"
MARKERS = {"perf": ("<!-- perf:begin -->", "<!-- perf:end -->"),
           "summary": ("<!-- perf-summary:begin -->", "<!-- perf-summary:end -->")}
REQUIRED = ("machine", "model", "recipe", "metric", "value", "date")
SPEED = ("prefill", "decode", "decode_aggregate")
QUALITY = ("perplexity", "kld", "acceptance", "greedy_match")
OURS = "ninferno"


# --- loading ------------------------------------------------------------------------------------

def load():
    gpus = json.loads((DATA / "gpus.json").read_text(encoding="utf-8"))
    models = json.loads((DATA / "models.json").read_text(encoding="utf-8"))
    by_name = {n: g for g, info in gpus.items() for n in info["match"]}
    by_hf = {info["hf"].lower(): m for m, info in models.items()}
    by_hf.update({info["hf"].split("/")[-1].lower(): m for m, info in models.items()})
    rows, problems = [], []
    for n, line in enumerate((DATA / "bench.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        missing = [k for k in REQUIRED if k not in r]
        gpu = by_name.get(r.get("machine", {}).get("gpu_name"))
        if missing:
            problems.append(f"bench.jsonl:{n}: missing {missing}")
        elif gpu is None:
            problems.append(f"bench.jsonl:{n}: no gpus.json entry matches {r['machine'].get('gpu_name')!r}")
        elif r["metric"] == "decode_per_request":
            continue  # the per-request view of a chat wave; the pages show the aggregate
        elif r["model"].lower() not in by_hf:
            problems.append(f"bench.jsonl:{n}: no models.json entry for {r['model']!r}")
        elif r["metric"] not in SPEED + QUALITY:
            problems.append(f"bench.jsonl:{n}: unknown metric {r['metric']!r}")
        elif r["metric"] == "prefill" and not r.get("prompt_len"):
            problems.append(f"bench.jsonl:{n}: prefill without prompt_len")
        else:
            r["_gpu"], r["_model"] = gpu, by_hf[r["model"].lower()]
            r.setdefault("engine", OURS)
            r.setdefault("spec", "none")
            r.setdefault("kv", "auto")
            r.setdefault("concurrency", 1)
            rows.append(r)
    return gpus, models, rows, problems


# --- formatting ---------------------------------------------------------------------------------

HUB_REPO = re.compile(r"^([A-Za-z0-9][\w.-]*/[\w.-]+)(\s+.*)?$")


def weights(recipe, link=True):
    """A recipe that starts with a Hugging Face repository links it: "org/name --quant LABEL"."""
    m = HUB_REPO.match(recipe)
    if not m:
        return recipe
    label = (m.group(2) or "").replace("--quant ", "").strip()
    repo = f"[`{m.group(1)}`](https://huggingface.co/{m.group(1)})" if link else f"`{m.group(1)}`"
    return f"{repo} {label}".rstrip()


def weights_short(recipe):
    """A narrow column's weights: the quantization, linked to its repository, the repository in the
    title: [UD-Q4_K_XL](https://huggingface.co/unsloth/... "unsloth/Qwen3.8-27B-GGUF")."""
    m = HUB_REPO.match(recipe)
    if not m:
        return recipe
    # No --quant: the publisher's own checkpoint (gpt-oss ships MXFP4 itself).
    label = quant(recipe) if "--quant" in recipe else "original"
    return f"[{label}](https://huggingface.co/{m.group(1)}{hint(m.group(1))})"


def repos(recipes):
    """The Hugging Face repositories of these recipes, linked, in first-seen order."""
    seen = []
    for r in recipes:
        m = HUB_REPO.match(r)
        if m and m.group(1) not in seen:
            seen.append(m.group(1))
    return ", ".join(f"[`{s}`](https://huggingface.co/{s})" for s in seen)


def gpu_short(g, info, base):
    """A GPU on one line: the label's first line, spaces unbreakable, the full name, memory and chip
    in the title."""
    short = (info["label"][0] if info.get("label") else info["name"]).replace(" ", "&nbsp;")
    size = "" if info["name"].endswith(" GB") else f", {info['vram_gb']} GB"
    return f"[{short}]({base}gpus/{g}.md{hint(info['name'] + size + ' (' + info['chip'] + ', sm_' + str(info['sm']) + ')')})"


def quant(recipe):
    """The quantization alone, for the narrow columns: "UD-Q4_K_XL", or the repository name."""
    m = re.search(r"--quant\s+(\S+)", recipe)
    return m.group(1) if m else recipe.split()[0].split("/")[-1]


def fmt(v):
    if v is None:
        return "–"
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"


def spec_label(r):
    if r.get("spec") in ("none", "", None):
        return "off"
    return f"{r['spec']} ×{r['draft_tokens']}" if r.get("draft_tokens") else r["spec"]


def prompt_k(r):
    m = re.match(r"long(\d+)k", r.get("prompt_set") or "")
    if m:
        return int(m.group(1))
    return round(r["prompt_len"] / 1000) if r.get("prompt_len") else 0


def day(r):
    return r["date"][:10]


def command(r, concurrency):
    """The server command line that reproduces a setting (the defaults choose everything else)."""
    parts = ["ninferno-serve", r["recipe"]]
    if r.get("spec") not in ("none", "", None):  # the drafters the measured setting used
        parts.append("--with-drafters")
    if r.get("spec") in ("none", "", None):
        parts.append("--spec none")
    elif r["spec"] != "auto" and r.get("draft_tokens"):
        parts.append(f"--spec {r['spec']} --draft-tokens {r['draft_tokens']}")
    if r.get("kv") not in ("auto", None):
        parts.append(f"--kv-dtype {r['kv']}")
    if concurrency > 1:
        parts.append(f"--max-concurrency {concurrency}")
    return " ".join(parts)


# --- collecting ---------------------------------------------------------------------------------

def setting(r):
    return (r["recipe"], spec_label(r), r["kv"])


def collect(rows):
    """{(gpu, model): {setting: {cell: record}}} for our speed rows; references and quality apart."""
    speed = defaultdict(lambda: defaultdict(dict))
    refs = defaultdict(list)
    quality = defaultdict(list)
    for r in rows:
        key = (r["_gpu"], r["_model"])
        if r["metric"] in QUALITY:
            quality[key].append(r)
        elif r["engine"] != OURS:
            refs[key].append(r)
        else:
            cell = ("prefill", prompt_k(r)) if r["metric"] == "prefill" else \
                ("long", prompt_k(r)) if r["metric"] == "decode" else ("chat", r["concurrency"])
            old = speed[key][setting(r)].get(cell)
            if old is None or old["date"] <= r["date"]:  # the latest measurement of a cell wins
                speed[key][setting(r)][cell] = r
    return speed, refs, quality


def best_single(settings):
    """(record, setting) of the fastest single stream: one short chat at a time."""
    cands = [(c[("chat", 1)], s) for s, c in settings.items() if ("chat", 1) in c]
    return max(cands, key=lambda x: x[0]["value"]) if cands else (None, None)


def best_multi(settings):
    """(record, setting) of the most tokens per second over all streams, at any concurrency > 1."""
    cands = [(r, s) for s, c in settings.items() for k, r in c.items() if k[0] == "chat" and k[1] > 1]
    return max(cands, key=lambda x: x[0]["value"]) if cands else (None, None)


def best_long(settings):
    """(prefill, decode) records of the setting with the fastest decode after a long prompt."""
    cands = [(c[k], s) for s, c in settings.items() for k in c if k[0] == "long"]
    if not cands:
        return None, None
    r, s = max(cands, key=lambda x: x[0]["value"])
    pre = settings[s].get(("prefill", prompt_k(r)))
    return pre, r


def summary_ratio(speed, refs):
    """[(gpu, model, single ratio, multi ratio)] against the reference engine on the same weights."""
    out = []
    for key, rs in refs.items():
        if key not in speed:
            continue
        for kind, pick, conc in (("single", best_single, lambda c: c == 1), ("multi", best_multi, lambda c: c > 1)):
            ours, s = pick(speed[key])
            if ours is None:
                continue
            same = [r for r in rs if r["metric"] == "decode_aggregate" and conc(r.get("concurrency", 1))
                    and quant(r["recipe"]) == quant(s[0])]
            if same:
                ref = max(same, key=lambda r: r["value"])
                if ref["value"] > 0:
                    out.append((key, kind, ours["value"] / ref["value"]))
    return out


# --- level 1: README and the matrix ---------------------------------------------------------------

def sort_present(order, present):
    return [k for k in order if k in present]




def hint(text):
    """A link title: GitHub shows it when the pointer rests on the link. Double quotes would end it."""
    return ' "' + text.replace('"', "'") + '"'


def cell(settings, url):
    """A matrix cell: "one stream / all streams", each number its own link to the run with its own
    title (what it was measured with), the slash plain text and the concurrency of the latter a
    superscript: [58.9] / [140<sup>8</sup>]."""
    parts = []
    for name, (rec, s) in (("one stream", best_single(settings)), ("all streams", best_multi(settings))):
        if rec is None:
            parts.append("–")
            continue
        users = f" at {rec['concurrency']} streams" if name == "all streams" else ""
        spec = "no speculation" if s[1] == "off" else s[1]
        title = f"{name}: {fmt(rec['value'])} tok/s{users}, {quant(s[0])}, {spec}, measured {day(rec)}"
        number = fmt(rec["value"]) + (f"<sup>{rec['concurrency']}</sup>" if name == "all streams" else "")
        parts.append(f"[{number}]({url}{hint(title)})")
    return " / ".join(parts)


def label(info):
    """A table heading on two lines where the data gives one ("label": ["Qwen3.6", "35B-A3B"]), so
    more columns fit a normal window."""
    return "<br>".join(info["label"]) if info.get("label") else info["name"]


def matrix(gpus, models, speed, base, readme_only=False):
    """GPU rows x model columns; readme_only: the models marked "readme" in models.json (the current
    ones the project README shows; the performance index shows every model measured)."""
    present_g = {k[0] for k in speed}
    cols = [m for m in sort_present(models, {k[1] for k in speed})
            if not readme_only or models[m].get("readme")]
    lines = ["| GPU | " + " | ".join(
        f"[{label(models[m])}]({base}models/{m}.md{hint(models[m]['family'] + ', ' + models[m]['params'])})"
        for m in cols) + " |",
             "|---|" + "---:|" * len(cols)]
    for g in sort_present(gpus, present_g):
        if not any((g, m) in speed for m in cols):
            continue
        info = gpus[g]
        row = [gpu_short(g, info, base).replace("[", "[**", 1).replace("](", "**](", 1)]
        for m in cols:
            row.append(cell(speed[(g, m)], f"{base}runs/{g}/{m}.md") if (g, m) in speed else "")
        lines.append("| " + " | ".join(row) + " |")
    return lines, cols


def stamp(recs, base):
    dates = sorted({day(r) for r in recs})
    versions = sorted({r["version"] for r in recs if r.get("version")})
    when = dates[-1] if len(dates) == 1 else f"{dates[0]} – {dates[-1]}"
    ver = f" · NInferno {', '.join(versions)}" if versions else ""
    return f"<sub>Measured {when}{ver} · [methodology]({base}methodology.md)</sub>"


def summary_line(gpus, models, speed, refs, base):
    ratios = summary_ratio(speed, refs)
    if not ratios:
        return []
    out = []
    for kind, what in (("single", "one stream"), ("multi", "all streams under load")):
        vals = [v for _, k, v in ratios if k == kind]
        if vals:
            lo, hi = min(vals), max(vals)
            span = f"×{lo:.1f}" if abs(hi - lo) < 0.05 else f"×{lo:.1f}–{hi:.1f}"
            out.append(f"{span} {what}")
    pairs = len({k for k, _, _ in ratios})
    return [f"**Against llama.cpp on the same GGUF and GPU:** {', '.join(out)} "
            f"({pairs} GPU × model pairs, [details]({base}README.md#against-llamacpp))."]


def readme_blocks(gpus, models, speed, refs):
    table, _ = matrix(gpus, models, speed, "docs/performance/", readme_only=True)
    recs = [r for s in speed.values() for c in s.values() for r in c.values()]
    perf = ["Tokens per second, best settings: **one stream / all streams**, superscript = concurrent "
            "streams. Click a GPU, model or number for details "
            "([all GPUs and models](docs/performance/README.md)).", "",
            *table, "", stamp(recs, "docs/performance/")]
    return {"perf": perf, "summary": summary_line(gpus, models, speed, refs, "docs/performance/")}


def page_index(gpus, models, speed, refs):
    table, _ = matrix(gpus, models, speed, "")
    recs = [r for s in speed.values() for c in s.values() for r in c.values()]
    lines = ["# NInferno performance", "",
             "Tokens per second, models converted by `ninferno-serve` from Hugging Face as published. "
             "Cell: **one stream / all streams**, superscript = concurrent streams, each at its best "
             "setting. A GPU or model opens its page; a cell opens every setting measured, with "
             "commands ([methodology](methodology.md)).", "",
             *table, "", stamp(recs, "")]
    ratios = summary_ratio(speed, refs)
    if ratios:
        lines += ["", "## Against llama.cpp", "",
                  "NInferno's tokens per second over llama.cpp's, on the same GGUF file and GPU.", "",
                  "| GPU | Model | One stream | All streams |", "|---|---|---:|---:|"]
        by = defaultdict(dict)
        for key, kind, v in ratios:
            by[key][kind] = v
        for g in sort_present(gpus, {k[0] for k in by}):
            for m in sort_present(models, {k[1] for k in by if k[0] == g}):
                d = by[(g, m)]
                lines.append(f"| {gpus[g]['name']} | [{models[m]['name']}](runs/{g}/{m}.md) | "
                             + " | ".join(f"×{d[k]:.1f}" if k in d else "–" for k in ("single", "multi")) + " |")
    return "\n".join(lines + [""])


# --- level 2: one GPU, one model ------------------------------------------------------------------

LEVEL2_HEAD = ["Weights", "One stream", "All streams", "Prefill 24K", "Decode @24K"]


def level2_row(first, settings, link):
    s1, set1 = best_single(settings)
    sm, _ = best_multi(settings)
    pre, long = best_long(settings)
    w = weights_short(set1[0] if set1 else next(iter(settings))[0])
    multi = f"{fmt(sm['value'])}<sup>{sm['concurrency']}</sup>" if sm else "–"
    return "| " + " | ".join([first, w, fmt(s1["value"] if s1 else None), multi,
                              fmt(pre["value"] if pre else None), fmt(long["value"] if long else None),
                              f"[details]({link})"]) + " |"


def footer(recs):
    dates = sorted({day(r) for r in recs})
    versions = sorted({r["version"] for r in recs if r.get("version")})
    machines = sorted({(r["machine"]["gpu_name"], r["machine"].get("driver", "?"),
                        r["machine"].get("power_limit_w")) for r in recs}, key=str)
    parts = [f"Measured {dates[0]}" + (f" to {dates[-1]}" if len(dates) > 1 else "")]
    if versions:
        parts.append(f"NInferno {', '.join(versions)}")
    parts += [f"{n}, driver {d}" + (f", {w:g} W" if w else "") for n, d, w in machines]
    return f"<sub>{' · '.join(parts)} · [methodology](../methodology.md)</sub>"


def page_gpu(g, gpus, models, speed):
    info = gpus[g]
    lines = [f"# {info['name']}", "",
             f"{info['chip']}, sm_{info['sm']}, {info['vram_gb']} GB · [all GPUs and models](../README.md)", "",
             "Each model on this GPU, tokens per second, each column at its best setting: one stream, all "
             "streams (superscript = concurrent streams), prefill of a 24K-token prompt, one stream after it.", "",
             "| Model | " + " | ".join(LEVEL2_HEAD) + " | |", "|---|---|" + "---:|" * 4 + "---|"]
    recs = []
    for m in sort_present(models, {k[1] for k in speed if k[0] == g}):
        name = models[m]["name"].replace(" ", "&nbsp;")
        lines.append(level2_row(f"[{name}](../models/{m}.md)", speed[(g, m)], f"../runs/{g}/{m}.md"))
        recs += [r for c in speed[(g, m)].values() for r in c.values()]
    return "\n".join(lines + ["", footer(recs), ""])


def page_model(m, gpus, models, speed):
    info = models[m]
    present = sort_present(gpus, {k[0] for k in speed if k[1] == m})
    lines = [f"# {info['name']}", "",
             f"[`{info['hf']}`](https://huggingface.co/{info['hf']}) · {info['family']}, {info['params']} · "
             "[all GPUs and models](../README.md)", "",
             f"Measured with {repos(s[0] for g in present for s in speed[(g, m)])}. Its fine-tunes and other "
             f"checkpoints of the same architecture run the same way.", "",
             "Tokens per second, each column at its best setting: one stream, all streams (superscript = "
             "concurrent streams), prefill of a 24K-token prompt, one stream after it.", "",
             "| GPU | " + " | ".join(LEVEL2_HEAD) + " | |", "|---|---|" + "---:|" * 4 + "---|"]
    recs = []
    for g in present:
        lines.append(level2_row(gpu_short(g, gpus[g], "../"), speed[(g, m)], f"../runs/{g}/{m}.md"))
        recs += [r for c in speed[(g, m)].values() for r in c.values()]
    return "\n".join(lines + ["", footer(recs), ""])


# --- level 3: one GPU x one model -----------------------------------------------------------------

def page_run(g, m, gpus, models, speed, refs, quality):
    settings = speed[(g, m)]
    gi, mi = gpus[g], models[m]
    lines = [f"# {mi['name']} on the {gi['name']}", "",
             f"[{gi['name']}](../../gpus/{g}.md) · [{mi['name']}](../../models/{m}.md) · "
             "[all GPUs and models](../../README.md)", ""]
    s1, set1 = best_single(settings)
    sm, setm = best_multi(settings)
    if s1 or sm:
        lines += ["## Best launch", ""]
        if s1:
            lines += [f"One stream, {fmt(s1['value'])} tok/s:", "", "```bash", command(s1, 1), "```", ""]
        if sm:
            lines += [f"{sm['concurrency']} concurrent chats, {fmt(sm['value'])} tok/s over all streams:", "",
                      "```bash", command(sm, sm["concurrency"]), "```", ""]
    order = {"prefill": 0, "long": 1, "chat": 2}
    cells = sorted({c for s in settings.values() for c in s}, key=lambda c: (order[c[0]], c[1]))
    head = {"prefill": "Prefill {}K", "long": "Decode @{}K", "chat": "C={}"}
    lines += ["## Every setting measured", "",
              "Tokens per second. **Prefill** and **Decode** follow a long prompt; **C=n** is *n* "
              "concurrent short chats, all streams together.", "",
              "| Weights | Speculation | KV | " + " | ".join(head[c[0]].format(c[1]) for c in cells) + " | Measured |",
              "|---|---|---|" + "---:|" * len(cells) + "---|"]
    for s in sorted(settings):
        recs = settings[s].values()
        lines.append("| " + " | ".join([weights(s[0], link=False), s[1], s[2]]
                                        + [fmt(settings[s][c]["value"]) if c in settings[s] else "–" for c in cells]
                                        + [max(day(r) for r in recs)]) + " |")
    ref = refs.get((g, m), [])
    if ref:
        lines += ["", "## Against llama.cpp", "",
                  "| Engine | Weights | Concurrency | tok/s | Measured |", "|---|---|---:|---:|---|"]
        for r in sorted(ref, key=lambda r: (r["recipe"], r.get("concurrency", 1))):
            lines.append(f"| {r['engine']} {r.get('version', '')}".rstrip() + f" | {quant(r['recipe'])} | "
                         f"{r.get('concurrency', 1)} | {fmt(r['value'])} | {day(r)} |")
    q = quality.get((g, m), [])
    if q:
        lines += ["", "## Quality", ""]
        ppl = [r for r in q if r["metric"] in ("perplexity", "kld")]
        if ppl:
            # One row per weights and --activations mode: its perplexity, how far its token
            # distributions move from the reference mode's (exact), and llama.cpp's perplexity
            # on the same file where measured.
            by = defaultdict(dict)
            for r in ppl:
                kind = r["metric"] if r["engine"] == OURS else r["engine"]
                by[(quant(r["recipe"]), r.get("activations", "fast"))][kind] = r
            with_ref = any("llama.cpp" in d for d in by.values())
            refs_of = sorted({r.get("reference", "exact") for r in ppl if r["metric"] == "kld"})
            ref_name = refs_of[0] if refs_of else "exact"
            corpus = sorted({r.get("corpus") for r in ppl if r.get("corpus")})
            lines += [f"Perplexity (lower is better) on {', '.join(f'`{c}`' for c in corpus) or 'the fixed corpus'}, "
                      f"and how far each `--activations` mode moves the next-token distribution from "
                      f"`{ref_name}`: mean KL divergence and the share of positions with the same top token.", "",
                      "| Weights | Mode | Perplexity | KL from " + ref_name + " | Same top token |"
                      + (" llama.cpp |" if with_ref else "") + " Measured |",
                      "|---|---|---:|---:|---:|" + ("---:|" if with_ref else "") + "---|"]
            order = {"exact": 0, "fast": 1, "int8": 2}
            for (w, mode), d in sorted(by.items(), key=lambda x: (x[0][0], order.get(x[0][1], 9))):
                p, k, o = d.get("perplexity"), d.get("kld"), d.get("llama.cpp")
                row = [w, mode, f"{p['value']:.4f}" if p else "–",
                       "reference" if mode == ref_name else (f"{k['value']:.5f}" if k else "–"),
                       "" if mode == ref_name else (f"{k['top1_agreement'] * 100:.2f} %" if k else "–")]
                if with_ref:
                    row.append(f"{o['value']:.4f}" if o else "–")
                row.append(max(day(r) for r in d.values()))
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        spec = [r for r in q if r["metric"] in ("acceptance", "greedy_match")]
        if spec:
            lines += ["Speculation: the share of drafted tokens accepted, and whether greedy output with it is "
                      "identical to plain decoding.", "",
                      "| Weights | Speculation | Accepted | Greedy output identical | Measured |", "|---|---|---:|---|---|"]
            by = defaultdict(dict)
            for r in spec:
                by[(quant(r["recipe"]), spec_label(r))][r["metric"]] = r
            for (w, sp), d in sorted(by.items()):
                acc = f"{d['acceptance']['value'] * 100:.0f} %" if "acceptance" in d else "–"
                gm = ("yes" if d["greedy_match"]["value"] >= 1 else f"{d['greedy_match']['value'] * 100:.0f} % of prompts") \
                    if "greedy_match" in d else "–"
                lines.append(f"| {w} | {sp} | {acc} | {gm} | {max(day(r) for r in d.values())} |")
    recs = [r for c in settings.values() for r in c.values()]
    lines += ["", "## Measurement", ""]
    mach = sorted({json.dumps(r["machine"], sort_keys=True) for r in recs})
    for mj in mach:
        mm = json.loads(mj)
        w = mm.get("power_limit_w")
        lines.append(f"- {mm['gpu_name']}: driver {mm.get('driver', '?')}, CUDA {mm.get('cuda', '?')}"
                     + (f", power limit {w:g} W" if w else "") + ".")
    versions = sorted({r["version"] for r in recs if r.get("version")})
    if versions:
        lines.append(f"- NInferno {', '.join(versions)}.")
    lines.append("- How each number is taken: [methodology](../../methodology.md).")
    flags = sorted({r["flags"] for r in recs if r.get("flags")})
    if flags:
        lines += ["", "Server flags of the measured settings:", "", "```", *flags, "```"]
    return "\n".join(lines + [""])


# --- output -------------------------------------------------------------------------------------

def replace_block(text, name, body, problems):
    begin, end = MARKERS[name]
    if begin not in text or end not in text:
        problems.append(f"README.md: no {begin} ... {end} block")
        return text
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{begin}\n" + "\n".join(body) + (f"\n{end}" if body else end) + tail


def render():
    gpus, models, rows, problems = load()
    speed, refs, quality = collect(rows)
    pages = {OUT / "README.md": page_index(gpus, models, speed, refs)}
    for g in {k[0] for k in speed}:
        pages[OUT / "gpus" / f"{g}.md"] = page_gpu(g, gpus, models, speed)
    for m in {k[1] for k in speed}:
        pages[OUT / "models" / f"{m}.md"] = page_model(m, gpus, models, speed)
    for g, m in speed:
        pages[OUT / "runs" / g / f"{m}.md"] = page_run(g, m, gpus, models, speed, refs, quality)
    text = README.read_text(encoding="utf-8")
    for name, body in readme_blocks(gpus, models, speed, refs).items():
        text = replace_block(text, name, body, problems)
    pages[README] = text
    return pages, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail when a page differs from the data")
    args = ap.parse_args()
    pages, problems = render()
    for p in problems:
        print(p, file=sys.stderr)
    stale = []
    for path, text in sorted(pages.items()):
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if old == text:
            continue
        stale.append(path)
        if args.check:
            sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), text.splitlines(True),
                                                       str(path.relative_to(ROOT)), "generated"))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    # Pages the data no longer has go too.
    orphans = [f for d in ("gpus", "models", "runs") for f in sorted((OUT / d).rglob("*.md")) if f not in pages]
    for f in orphans:
        if args.check:
            print(f"{f.relative_to(ROOT)}: no data left for this page")
        else:
            f.unlink()
    stale += orphans
    if args.check:
        sys.exit(1 if stale or problems else 0)
    print(f"{len(pages)} pages, {len(stale)} written, {len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
