"""Maintainer tool: put a spec-tune result into the drafter registry.

    python tools/spec/registry_add.py MODEL_ID PUBLISHED.json [--registry PATH] [--requires a,b]
                                      [--fallback]
    python tools/spec/registry_add.py MODEL_ID LOAD.json --load [--registry PATH] [--requires a,b]
                                      [--fallback] [--create]

PUBLISHED.json is what `ninferno-gemma4 <artifact> spec-tune --publish FILE` or
`ninferno spec-tune <artifact> --publish FILE` wrote: one profile for one GPU class (the chip, e.g.
gb202, unless spec-tune was given --gpu-class). It replaces the model entry's profile with the
same gpu_class, applies_to and requires (or is appended), so re-running a tune on the same
hardware updates the entry in place. --requires names the drafters the profile needs (a Qwen entry
has one profile with the DFlash2 companion and one without). A profile whose choices run a
drafter variant ("assistant.q8": NInferno's own quantization the converter builds beside a drafter
that declares "variants") requires it, whether or not --requires says so, so that a model without
the variant attached takes the class's profile of the published drafters.

--fallback also writes the "*" profile (GPUs nobody measured) with the same requires from it: its
"portable" choices when spec-tune wrote them (per family and default, the shallowest
configuration within 2% of the best here, over what a serving Engine runs; src/spec/profile.h),
else the same configurations; each with the watchdog floor measured here (tokens per round at
which a round costs what plain decode of its tokens does: a ratio, so the best guess for another
GPU, and without it an unmeasured GPU runs with no working watchdog), without the measured speeds.
Without portable choices, a profile's "served" choices (what a serving Engine runs,
src/spec/engine_support.h) are carried the same way; its "fit_fallback" (src/spec/profile.h) is
carried either way. A tune published before spec-tune wrote portable choices is brought up to date
from its saved profile (spec-tune --out) without measuring again: `ninferno spec-reselect SAVED.json
--publish PUBLISHED.json --gpu-class ... --gpu ... --applies-to ... --measured-on ...` (with
`--engine gemma4_text` for a Gemma 4 tune), then this tool with --fallback.

--load merges instead what `ninferno spec-tune <artifact> --load C,... --publish FILE` wrote: the
bounds under decode load (src/spec/profile.h, "load") measured for one GPU class. They go into
every profile of the entry with that gpu_class and requires (and applies_to, when the file names
repositories), replacing an earlier table; --fallback also puts them into the "*" profile with the
same requires. A table spec-tune marked "load_partial" (a concurrency it could not decide) is
merged with a warning, and the profiles keep the mark (so does a re-tune, and the "*" profile
--fallback derives). Before merging a table for a class that borrows one today, A/B the server's
defaults on a short chat against the release: spec-tune --load needs a 3 % gain over plain on
sampled thinking prompts, yet a short greedy chat can still gain where it picks plain (gpt-oss-20b,
RTX 3090: a ga102 table going plain from 3 lanes lost 8 % at C=4 to the borrowed ga100 table).
--create makes the profile when the class has none: the "*" profile's
choices (what serve --spec auto ran there, so what the bounds were measured with) for the file's
applies_to.

Every write stamps the registry (write_registry): a profile that is new or changed gets
"measured_at" (today, UTC), every profile its "revision" (a hash of its content), and the registry,
when anything changed, the next "version" and today's "published". The version orders registries
(a newer one downloaded later can override the compiled-in one, as --drafter-registry FILE does
from a file today); a revision names the table a server runs. After a hand edit,

    python tools/spec/registry_add.py --restamp [--registry PATH]

stamps the registry the same way.

--pair puts a lab measurement of one target with its drafters into the registry's "pairs"
(src/spec/pairs.h: a start resolves a pair by the target's weights before any published profile):

    python tools/spec/registry_add.py --pair TUNE.json --manifest MANIFEST.json
                                      --target REPO@COMMIT --quant LABEL --weights SHA256[,...]

TUNE.json is what `ninferno spec-tune <artifact> --out FILE` saved (its measurements), MANIFEST.json
the drafter manifest beside the artifact (<artifact>.speculation.json: the engine family and the
drafters attached, each separate one by repo@revision, recorded as what was measured), --weights the
SHA-256 of the target's weight files as the hub lists them (the pair's key; order does not matter),
--quant the quantization label ("Q4_K_M"; "" for a checkpoint's own weights: its bit class). The
card's generation comes from the tune's GPU class through the PCI facts table
(src/core/data/gpu_facts.json: "ad104" is sm89). A pair of the same target, generation and GPU class
is replaced.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

CHOICE_KEYS = ("config", "drafter", "floor_tokens_per_round")


def revision_of(profile: dict) -> str:
    """12 hex digits of the SHA-256 of the profile's canonical JSON, its own "revision" left out."""
    body = {k: v for k, v in profile.items() if k != "revision"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def stamped(profile: dict, measured_at: str) -> dict:
    """`profile` with "measured_at" after "measured_on" (else last) and "revision" last."""
    out = {}
    for k, v in profile.items():
        if k in ("measured_at", "revision"):
            continue
        out[k] = v
        if k == "measured_on":
            out["measured_at"] = measured_at
    out.setdefault("measured_at", measured_at)
    out["revision"] = revision_of(out)
    return out


def stamp_registry(registry: dict, before: dict | None, today: str) -> bool:
    """Stamps `registry` against `before` (its content when read; None: nothing to compare, so only
    missing stamps are added). Returns whether anything changed."""
    old = {}
    for m in (before or {}).get("models", []):
        for p in m.get("profiles", []):
            old.setdefault(m.get("id"), set()).add(revision_of({k: v for k, v in p.items() if k != "measured_at"}))
    for m in registry.get("models", []):
        profiles = []
        for p in m.get("profiles", []):
            unchanged = revision_of({k: v for k, v in p.items() if k != "measured_at"}) in old.get(m.get("id"), set())
            date = p.get("measured_at") if (unchanged or before is None) and p.get("measured_at") else today
            profiles.append(stamped(p, date))
        m["profiles"] = profiles
    strip = lambda r: {k: v for k, v in r.items() if k not in ("version", "published")}
    changed = before is not None and strip(registry) != strip(before)
    if changed:
        registry["version"] = int(before.get("version", 0)) + 1
        registry["published"] = today
    return changed


def write_registry(registry: dict, before: dict | None, path: Path) -> None:
    """Stamps `registry` (stamp_registry) and writes it to `path` (LF, two-space indent)."""
    stamp_registry(registry, before, datetime.datetime.now(datetime.timezone.utc).date().isoformat())
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")


def fallback_of(profile: dict) -> dict:
    """The "*" profile carrying `profile`'s portable choices (else its choices) and floors."""
    strip = lambda c: {k: c[k] for k in CHOICE_KEYS if k in c}
    portable = profile.get("portable")
    which = "the shallowest choices within 2% of the best" if portable else "the choices"
    out = {
        "gpu_class": "*",
        "applies_to": [],
        "measured_on": f"fallback for unmeasured GPUs: {which} measured on {profile.get('gpu', '?')} "
                       f"({profile.get('gpu_class')}), {profile.get('measured_on', '')}",
        "source": profile.get("source", ""),
    }
    if profile.get("requires"):
        out["requires"] = profile["requires"]
    choices = portable or profile
    for key in ("default", "non_thinking", "thinking"):
        if key in choices:
            out[key] = strip(choices[key])
    # The portable choices are over what a serving Engine runs: no "served" beside them.
    if "served" in profile and not portable:  # what a serving Engine runs, when the choices need more
        served = profile["served"]
        out["served"] = {key: strip(served[key]) for key in ("default", "non_thinking", "thinking") if key in served}
    if "fit_fallback" in profile:  # what runs when the chosen drafter does not fit the context
        out["fit_fallback"] = strip(profile["fit_fallback"])
    if "load" in profile:  # the bounds under decode load
        out["load"] = profile["load"]
        # Worded as merge_load words the table --load --fallback puts into the "*" profile.
        out["load_measured_on"] = (f"fallback for unmeasured GPUs: {profile.get('gpu', '?')} "
                                   f"({profile.get('gpu_class')}), {profile.get('load_measured_on', '')}")
        if profile.get("load_partial"):
            out["load_partial"] = True
    return out


LOAD_KEYS = ("load", "load_measured_on", "load_partial")


def kind(p: dict) -> str:
    """The drafter kind of a profile's default ("mtp:5+head" -> "mtp"), as the runtime's
    select_load_fallback (src/spec/profile.cpp) compares it."""
    return p.get("default", {}).get("config", "").split(":")[0]


def load_borrowed(profiles: list[dict], profile: dict) -> dict | None:
    """The profile whose load table `profile` (without one) runs under decode load, as
    src/spec/profile.cpp select_load_fallback picks it: another profile of the same GPU class,
    drafter kind and requires with a table, else the "*" profile (any variant) of the kind and
    requires; None: unbounded."""
    requires = sorted(profile.get("requires") or [])
    star = None
    for p in profiles:
        if p is profile or "load" not in p or kind(p) != kind(profile):
            continue
        if sorted(p.get("requires") or []) != requires:
            continue
        if p.get("gpu_class") == profile.get("gpu_class") != "*":
            return p
        if star is None and p.get("gpu_class") == "*" and not p.get("applies_to"):
            star = p
    return star


def check_load(table: list[dict]) -> None:
    """Refuses a table the Engine refuses (src/spec/profile.cpp parse_load): lanes ascending from 2,
    depths 0..15, no speculating bound after a bound of 0, a draft_width (block or drafts) only on a
    speculating bound."""
    lanes, plain = 1, False
    for bound in table:
        width = bound.get("draft_width")
        if (bound["lanes"] <= lanes or not 0 <= bound["depth"] <= 15 or (plain and bound["depth"] != 0)
                or (width is not None and (width not in ("block", "drafts") or bound["depth"] == 0))):
            raise SystemExit(f"load table {table} is invalid: lanes ascending from 2, depths 0..15, "
                             "no nonzero depth after a 0, draft_width block or drafts on a nonzero "
                             "depth only")
        lanes, plain = bound["lanes"], bound["depth"] == 0


def merge_load(profiles: list[dict], load: dict, fallback: bool) -> int:
    """Puts `load`'s bounds into the matching profiles; returns how many it changed."""
    check_load(load["load"])
    requires = sorted(load.get("requires") or [])
    applies = sorted(load.get("applies_to") or [])
    changed = 0
    for p in profiles:
        cls = p.get("gpu_class")
        if sorted(p.get("requires") or []) != requires:
            continue
        if cls == load["gpu_class"] and (not applies or sorted(p.get("applies_to") or []) == applies):
            p["load"] = load["load"]
            p["load_measured_on"] = load.get("load_measured_on", "")
        elif fallback and cls == "*":
            p["load"] = load["load"]
            p["load_measured_on"] = (f"fallback for unmeasured GPUs: {load.get('gpu', '?')} "
                                     f"({load['gpu_class']}), {load.get('load_measured_on', '')}")
        else:
            continue
        # A table with concurrencies spec-tune could not decide says so, until a full tune
        # replaces it.
        if load.get("load_partial"):
            p["load_partial"] = True
        else:
            p.pop("load_partial", None)
        changed += 1
    return changed


def put(profiles: list[dict], profile: dict) -> list[dict]:
    same = lambda p: (p.get("gpu_class"), sorted(p.get("applies_to") or []), sorted(p.get("requires") or [])) == \
        (profile.get("gpu_class"), sorted(profile.get("applies_to") or []), sorted(profile.get("requires") or []))
    # A profile it replaces keeps its place (a re-tune changes one entry, not the order).
    at = next((i for i, p in enumerate(profiles) if same(p)), None)
    profiles = [p for p in profiles if not same(p)]
    if at is None:
        # Measured classes before the "*" fallback, so an exact class is found first when read in order.
        at = len(profiles) if profile.get("gpu_class") == "*" else sum(1 for p in profiles if p.get("gpu_class") != "*")
    profiles.insert(at, profile)
    return profiles


FACTS = Path(__file__).resolve().parents[2] / "src/core/data/gpu_facts.json"


def label_bits(label: str) -> float:
    """A quantization label's bits per weight as spec::label_bits reads it (src/spec/pairs.cpp)."""
    text = label.lower()
    if not text or "bf16" in text or "f16" in text:
        return 16
    if "f32" in text:
        return 32
    if "fp4" in text:
        return 4
    match = re.search(r"(?:^|[-_])[it]?q([0-9])", text)
    return float(match.group(1)) if match else 16


def generation_of(gpu_class: str) -> str:
    """The architecture generation of a GPU class ("ad104:60sm" -> "sm89"), from the facts table."""
    chip = gpu_class.split(":")[0].lower()
    for row in json.loads(FACTS.read_text(encoding="utf-8"))["gpus"]:
        if row.get("chip", "").lower() == chip and row.get("cc"):
            return f"sm{row['cc']}"
    raise SystemExit(f"no generation for GPU class {gpu_class!r} in {FACTS.name}")


# The lab's boards by GPU class (the tunes' key.gpu): their PCI product ids in
# src/core/data/gpu_facts.json.
LAB_BOARDS = {
    "ad104:60sm": "0x2782",   # GeForce RTX 4070 Ti
    "ad102:128sm": "0x2684",  # GeForce RTX 4090
    "ad106:34sm": "0x2805",   # GeForce RTX 4060 Ti 16GB
    "gb203:70sm": "0x2C05",   # GeForce RTX 5070 Ti
    "gb202:170sm": "0x2B85",  # GeForce RTX 5090
    "ga106:28sm": "0x2503",   # GeForce RTX 3060
    "tu102:68sm": "0x1E04",   # GeForce RTX 2080 Ti
    "gv100:80sm": "0x1DBA",   # Quadro GV100
}


def pair_of(tune: dict, manifest: dict, target: str, quant: str, weights: list[str]) -> dict:
    """The registry's pair for one tune (src/spec/pairs.h MeasuredPair)."""
    shas = sorted(w.strip().lower() for w in weights if w.strip())
    if not shas or any(len(s) != 64 or any(c not in "0123456789abcdef" for c in s) for s in shas):
        raise SystemExit("--weights: the SHA-256 (64 hex) of every weight file")
    gpu = (tune.get("key") or {}).get("gpu", "")
    if not gpu:
        raise SystemExit("the tune names no GPU class (key.gpu)")
    # A tune that measured no drafter (none fitted beside the target, or the budget ran out before
    # one) says nothing of the pair: a start would take it for unmeasured anyway (pair_choice).
    if not any(m.get("config", "plain") != "plain" for m in tune.get("measurements", [])):
        raise SystemExit("the tune measured no drafter configuration: nothing to record")
    engine = (manifest.get("target") or {}).get("engine") or manifest.get("engine", "")
    if not engine:
        raise SystemExit("the manifest names no engine family")
    drafters = {}
    for d in manifest.get("drafters", []):
        if d.get("repo") and d.get("status") in ("embedded", "converted", "cached", "attached"):
            drafters[d["kind"]] = d["repo"] + ("@" + d["revision"] if d.get("revision") else "")
    # The lab's local paths (the artifact, a separate drafter's) are no part of what was measured.
    tune = json.loads(json.dumps(tune))
    tune.pop("artifact", None)
    for d in tune.get("drafters", []):
        d.pop("artifact", None)
    pair = {"target": ",".join(shas), "target_repo": f"{target} {quant}".strip(), "engine": engine,
            "target_bits": label_bits(quant), "generation": generation_of(gpu), "tune": tune}
    if drafters:
        pair["drafters"] = drafters
    if gpu in LAB_BOARDS:  # the board whose rates price the pair's rounds (spec/round_cost.h)
        pair["lab_board"] = LAB_BOARDS[gpu]
    return pair


def put_pair(pairs: list[dict], pair: dict) -> list[dict]:
    """`pairs` with `pair` in the place of one of the same target, generation and GPU class."""
    same = lambda p: (p.get("target") == pair["target"] and p.get("generation") == pair["generation"]
                      and (p.get("tune") or {}).get("key", {}).get("gpu") == pair["tune"]["key"]["gpu"])
    kept = [p for p in pairs if not same(p)]
    # The weights' round facts are the target's and its drafters' (tools/spec/round_facts.py): a
    # new tune of the same pair keeps them; the lab board stays with its GPU class.
    # Only facts that price every mode the new tune measured: stale ones would drop its other
    # configurations off the lab card (spec/pairs.cpp price_by_facts); then none, and a warning.
    modes = {m.get("config", "plain").split(":")[0] for m in pair["tune"].get("measurements", [])} - {"plain"}
    for p in pairs:
        if (p.get("target") == pair["target"] and p.get("drafters") == pair.get("drafters")
                and "round_facts" in p and "round_facts" not in pair):
            if modes <= set(p["round_facts"].get("drafters", {})):
                pair["round_facts"] = p["round_facts"]
            else:
                print(f"{pair['target_repo']}: the earlier round facts do not price "
                      f"{sorted(modes - set(p['round_facts'].get('drafters', {})))}: none kept "
                      "(run tools/spec/round_facts.py)", file=sys.stderr)
        if same(p) and "lab_board" in p and "lab_board" not in pair:
            pair["lab_board"] = p["lab_board"]
    return kept + [pair]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?")
    parser.add_argument("published", type=Path, nargs="?")
    parser.add_argument("--registry", type=Path,
                        default=Path(__file__).resolve().parents[2] / "src/spec/data/drafter_registry.json")
    parser.add_argument("--requires", default="")
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--restamp", action="store_true")
    parser.add_argument("--pair", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--target", default="")
    parser.add_argument("--quant", default="")
    parser.add_argument("--weights", default="")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    before = copy.deepcopy(registry)
    if args.restamp:
        write_registry(registry, before, args.registry)
        print(f"{args.registry}: version {registry.get('version')}")
        return 0
    if args.pair is not None:
        if args.manifest is None or not args.target or not args.weights:
            parser.error("--pair needs --manifest, --target and --weights")
        pair = pair_of(json.loads(args.pair.read_text(encoding="utf-8")),
                       json.loads(args.manifest.read_text(encoding="utf-8")), args.target, args.quant,
                       args.weights.split(","))
        registry["pairs"] = put_pair(registry.get("pairs", []), pair)
        write_registry(registry, before, args.registry)
        print(f"{pair['target_repo']}: pair on {pair['generation']} ({pair['tune']['key']['gpu']}), "
              f"{len(registry['pairs'])} pair(s), version {registry.get('version')}")
        return 0
    if args.model is None or args.published is None:
        parser.error("MODEL_ID and PUBLISHED.json are required (or --restamp, --pair)")
    profile = json.loads(args.published.read_text(encoding="utf-8"))
    if args.load:
        entry = next((m for m in registry["models"] if m["id"] == args.model), None)
        if entry is None:
            raise SystemExit(f"no registry entry {args.model}")
        if args.requires:
            profile["requires"] = [r for r in args.requires.split(",") if r]
        profiles = entry.get("profiles", [])
        if args.create and not any(
                p.get("gpu_class") == profile["gpu_class"] and
                sorted(p.get("requires") or []) == sorted(profile.get("requires") or []) and
                (not profile.get("applies_to") or
                 sorted(p.get("applies_to") or []) == sorted(profile["applies_to"]))
                for p in profiles):
            star = next((p for p in profiles if p.get("gpu_class") == "*" and
                         sorted(p.get("requires") or []) == sorted(profile.get("requires") or [])), None)
            if star is None:
                raise SystemExit(f"{args.model}: no * profile with requires {profile.get('requires') or []}")
            made = json.loads(json.dumps(star))
            made.update({"gpu_class": profile["gpu_class"], "gpu": profile.get("gpu", ""),
                         "applies_to": profile.get("applies_to") or [],
                         "measured_on": "choices: the * fallback's (" + star.get("measured_on", "") +
                                        "); load measured here"})
            entry["profiles"] = profiles = put(profiles, made)
        if profile.get("load_partial"):
            print(f"warning: {args.published} is a partial load table (a concurrency spec-tune "
                  f"could not decide: {profile.get('load_measured_on', '')}); the profiles keep "
                  f"\"load_partial\": true", file=sys.stderr)
        changed = merge_load(profiles, profile, args.fallback)
        if changed == 0:
            raise SystemExit(f"{args.model}: no profile for {profile['gpu_class']} with requires "
                             f"{profile.get('requires') or []}")
        write_registry(registry, before, args.registry)
        print(f"{args.model}: load {profile['load']} in {changed} profile(s)")
        return 0
    if args.requires:
        profile["requires"] = [r for r in args.requires.split(",") if r]
    entry = next((m for m in registry["models"] if m["id"] == args.model), None)
    if entry is None:
        raise SystemExit(f"no registry entry {args.model}")
    names = {d["name"] for d in entry.get("drafters", [])}
    names |= {f"{d['name']}.{v}" for d in entry.get("drafters", []) for v in d.get("variants", [])}
    served = profile.get("served", {})
    checked = [(k, profile.get(k, {})) for k in ("default", "non_thinking", "thinking")]
    checked += [("served " + k, served[k]) for k in ("default", "non_thinking", "thinking") if k in served]
    portable = profile.get("portable", {})
    checked += [("portable " + k, portable[k]) for k in ("default", "non_thinking", "thinking") if k in portable]
    checked += [("fit_fallback", profile["fit_fallback"])] if "fit_fallback" in profile else []
    for key, choice in checked:
        if choice.get("config", "plain") != "plain" and choice.get("drafter") not in names | {"ngram"}:
            raise SystemExit(f"{key} runs drafter {choice.get('drafter')!r}, which {args.model} does not declare")
    variants = sorted({c["drafter"] for _, c in checked
                       if c.get("config", "plain") != "plain" and "." in c.get("drafter", "")})
    if variants:
        profile["requires"] = sorted(set(profile.get("requires") or []) | set(variants))
    # A re-tune keeps the bounds measured under load (src/spec/profile.h, "load") of the profile
    # it replaces: spec-tune --publish does not measure them. Only a table measured with the same
    # drafter kind: another kind's bounds would sit on a drafter they were not measured for (and
    # --fallback would spread them to "*").
    key = lambda p: (p.get("gpu_class"), sorted(p.get("applies_to") or []),
                     sorted(p.get("requires") or []))
    for old in entry.get("profiles", []):
        if "load" in old and "load" not in profile and key(old) == key(profile):
            if kind(old) == kind(profile):
                profile.update({k: old[k] for k in LOAD_KEYS if k in old})
            else:
                print(f"warning: the replaced {old.get('gpu_class')} profile's load table "
                      f"{old['load']} was measured with {kind(old)}, the new default runs "
                      f"{kind(profile)}: dropped (measure one with ninferno spec-tune --load and "
                      "merge it with --load)", file=sys.stderr)
    profiles = put(entry.get("profiles", []), profile)
    if args.fallback:
        star = fallback_of(profile)
        # A source without a load table keeps the one the "*" profile it replaces had, when that
        # was measured with the same drafter kind (a "*" profile without bounds runs its C=1
        # choices unbounded under load, and every class without a table borrows none: gb203
        # Gemma 4 31B at C=4, 132 tok/s against 164 plain).
        old = next((p for p in profiles if key(p) == key(star)), None)
        if "load" not in star and old is not None and "load" in old:
            if kind(old) == kind(star):
                star.update({k: old[k] for k in LOAD_KEYS if k in old})
            else:
                print(f"warning: the * profile's load table {old['load']} was measured with "
                      f"{kind(old)}, the new * default runs {kind(star)}: dropped (measure one with "
                      "ninferno spec-tune --load and merge it with --load --fallback)", file=sys.stderr)
        profiles = put(profiles, star)
    entry["profiles"] = profiles
    if "load" not in profile and profile.get("gpu_class") != "*":
        # Under decode load a profile without bounds borrows another's of the same drafter kind
        # and requires (src/spec/profile.cpp select_load_fallback: its own class first, then "*"),
        # else runs its C=1 choices unbounded (gb203 Gemma 4 31B at C=4 with a registry whose "*"
        # had none: 132 tok/s against 164 plain).
        lender = load_borrowed(profiles, profile)
        print(f"warning: {args.model} {profile.get('gpu_class')} has no load table: under load it "
              + (f"borrows the {lender.get('gpu_class')} {lender.get('applies_to') or ''} profile's "
                 f"{lender['load']}" if lender else "runs unbounded")
              + "; measure one with ninferno spec-tune --load and merge it with --load",
              file=sys.stderr)
    write_registry(registry, before, args.registry)
    print(f"{args.model}: {len(profiles)} profile(s); {profile.get('gpu_class')} default "
          f"{profile.get('default', {}).get('config')}" + (" (and the * fallback)" if args.fallback else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
