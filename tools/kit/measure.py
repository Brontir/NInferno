#!/usr/bin/env python3
"""The user measurement kit (docs/maintainer/user-measurement-kit.md): on your own card, what our
stands measure, beside what the plan predicts for the same start, in one file you can send us.

    python tools/kit/measure.py --bin DIR [--budget-minutes 30] [--models auto|ID,...] [--device N]
        [--passes 2] [--hard50] [--spec-tune SECONDS] [--out FILE] [--yes]

DIR holds ninferno-serve and ninferno (a release or a build). Standard library only; Windows and
Linux. The models are fetched first through `ninferno-serve --prepare-only` into your hub cache,
after the bytes are stated (--yes skips the question); the budget counts the GPU time only.
Nothing is sent anywhere: the kit writes FILE.json and FILE.md (a ready PR or issue body) and
prints how to submit them yourself.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import gpu  # noqa: E402
import validate  # noqa: E402

KIT_VERSION = "1"
SCHEMA = "ninferno-user-run/1.0"
HAYSTACK = ROOT / "examples/cli/messages/long_niah_256k.json"  # perf_matrix.py's long text
QUESTION = "\n\nSummarize the text above in detail, section by section."
START_TIMEOUT_S = 900


def exe(bin_dir, name):
    path = Path(bin_dir) / (name + (".exe" if os.name == "nt" else ""))
    if not path.exists():
        sys.exit(f"{path} not found (--bin is the directory holding ninferno-serve and ninferno)")
    return str(path)


def scrub(text):
    """An error without the user's paths (home and drive paths become <path>)."""
    return re.sub(r"(?:[A-Za-z]:)?[\\/][^\s:'\"]+", "<path>", str(text))[:400]


class Progress:
    def __init__(self, budget_s, steps):
        self.budget_s, self.steps, self.i = budget_s, steps, 0
        self.used = 0.0
        self.skipped = []

    def fits(self, cost_s, what):
        if self.used + cost_s > self.budget_s:
            self.skipped.append(what)
            print(f"[skip] {what}: about {cost_s / 60:.0f} min would pass the budget", flush=True)
            return False
        return True

    def step(self, what):
        self.i += 1
        print(f"[step {self.i}/{self.steps}, {self.used / 60:4.1f} of {self.budget_s / 60:.0f} min] "
              f"{what}", flush=True)

    def spend(self, seconds):
        self.used += seconds


def plan_json(serve, model, flags):
    out = subprocess.run([serve, model["repo"], "--quant", model["quant"], "--plan", "--no-load",
                          "--json"] + flags, capture_output=True, text=True, timeout=600)
    try:
        return json.loads(out.stdout), None
    except json.JSONDecodeError:
        return None, scrub((out.stderr or out.stdout).strip().splitlines()[-1:] or "no plan")


def run_flags(model, device, spec="none"):
    return ["--max-context", str(model["context"]), "--max-concurrency", "1", "--spec", spec,
            "--device", str(device)]


def request(port, prompt, max_tokens):
    body = {"model": "m", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return (out.get("usage") or {}), (out.get("timings") or {})


class Server:
    def __init__(self, serve, model, flags, port, log):
        self.port, self.log = port, log
        self.proc = subprocess.Popen([serve, model["repo"], "--quant", model["quant"], "--port",
                                      str(port), "--model-id", "m"] + flags,
                                     stdout=open(log, "w", encoding="utf-8", errors="replace"),
                                     stderr=subprocess.STDOUT)

    def wait(self):
        start = time.time()
        while time.time() - start < START_TIMEOUT_S:
            if self.proc.poll() is not None:
                raise RuntimeError("the server stopped: " + scrub(self.tail(2)))
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5):
                    return time.time() - start
            except OSError:
                time.sleep(1)
        raise RuntimeError("the server did not start in time")

    def meta(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=30) as r:
            model = json.load(r)["data"][0]
        meta = model.get("meta") or {}
        return {"fingerprint": meta.get("decision_fingerprint"), "speculation": meta.get("speculation")}

    def tail(self, n):
        try:
            return "\n".join(Path(self.log).read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def choice_line(self):
        try:
            for line in Path(self.log).read_text(encoding="utf-8", errors="replace").splitlines():
                if "choice |" in line:
                    return line.split("choice |", 1)[1].strip()
        except OSError:
            pass
        return None

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def prompt_of(blob, chars, i):
    offset = (i * 7919 * 13) % max(1, len(blob) - chars)
    return blob[offset:offset + chars] + QUESTION


def measure_pass(serve, model, flags, port, depth, prompts, device, workdir, tag):
    """One start: an idle window, a warm-up, one probe for the characters per token, then `prompts`
    requests at the plan's depth, each timed by the server (timings: prefill = prompt_n / prompt_ms,
    decode = (predicted_n - 1) / predicted_ms), under telemetry."""
    out = {"telemetry": None}
    server = Server(serve, model, flags, port, str(workdir / f"{tag}.serve.log"))
    sampler = None
    try:
        out["start_s"] = round(server.wait(), 1)
        out["served"] = server.meta()
        out["served"]["choice"] = server.choice_line()
        bus = gpu.query(["pci.bus_id"], device).get("pci.bus_id")
        apps = gpu.compute_apps(bus) or []
        out["other_processes"] = [a for a in apps if a.get("pid") != server.proc.pid]
        sampler = gpu.Sampler(device).start()
        t0 = sampler.now()
        time.sleep(5)  # the idle window (R2)
        sampler.mark("idle", t0, sampler.now())
        blob = json.loads(HAYSTACK.read_text(encoding="utf-8"))[-1]["content"]
        blob = blob.split("<document>\n", 1)[1].split("\n</document>", 1)[0]
        t0 = sampler.now()
        request(port, "Say hello in one short sentence.", 16)
        usage, _ = request(port, prompt_of(blob, 4000, 99), 1)  # characters per token
        sampler.mark("warmup", t0, sampler.now())
        per_token = 4000 / max(1, usage.get("prompt_tokens", 1000))
        chars = int(depth * per_token)
        out_tokens = max(32, min(256, depth // 8))
        rows = []
        for i in range(prompts):
            t0 = sampler.now()
            usage, timings = request(port, prompt_of(blob, chars, i), out_tokens)
            t1 = sampler.now()
            prompt_ms = timings.get("prompt_ms") or 0
            predicted_ms = timings.get("predicted_ms") or 0
            # The prefill window ends where the first token is out: by the server's own split.
            split = t0 + (t1 - t0) * prompt_ms / max(1e-9, prompt_ms + predicted_ms)
            sampler.mark(f"prefill {i}", t0, split)
            sampler.mark(f"decode {i}", split, t1)
            rows.append({"prompt_tokens": usage.get("prompt_tokens"), "cache_n": timings.get("cache_n"),
                         "prompt_n": timings.get("prompt_n"), "prompt_ms": prompt_ms,
                         "predicted_n": timings.get("predicted_n"), "predicted_ms": predicted_ms})
        out["requests"] = rows
        prefill = [r["prompt_n"] / (r["prompt_ms"] / 1000) for r in rows if r["prompt_n"] and r["prompt_ms"]]
        decode = [(r["predicted_n"] - 1) / (r["predicted_ms"] / 1000) for r in rows
                  if r["predicted_n"] and r["predicted_n"] > 1 and r["predicted_ms"]]
        tokens = sorted(r["prompt_tokens"] for r in rows if r["prompt_tokens"])

        def summary(values):
            values = sorted(values)
            return {"median": round(values[len(values) // 2], 2), "min": round(values[0], 2),
                    "max": round(values[-1], 2), "n": len(values),
                    "prompt_tokens": tokens[len(tokens) // 2] if tokens else None} if values else {}
        out["measured"] = {"prefill": summary(prefill), "decode": summary(decode)}
    except Exception as e:  # noqa: BLE001 - the file records the failure, the other runs go on
        out["error"] = scrub(e)
    finally:
        if sampler:
            out["telemetry"] = sampler.stop()
        server.stop()
    return out


def prepare(serve, model, with_drafters=False):
    """Fetch and convert (the start's own cache); the artifact path from its log."""
    cmd = [serve, model["repo"], "--quant", model["quant"], "--prepare-only"]
    if with_drafters:
        cmd.append("--with-drafters")
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    paths = [p for p in re.findall(r"(\S+\.ninferno)\b", text) if "drafter" not in p and ".speculation" not in p]
    return (paths[-1] if paths else None), (None if out.returncode == 0 else scrub(text.strip().splitlines()[-1:]))


def submission_md(doc, report_text, out_json):
    card = doc["card"]["smi"].get("name")
    name = Path(out_json).name
    return f"""# User measurement: {card}

Kit {doc['kit']['version']} ({doc['schema']}), NInferno {doc['ninferno'].get('version')}, {doc['created']}.

## What the validator says on this machine

```
{report_text}
```

## How to send it (you send it; the kit never posts)

- As a pull request: fork https://github.com/Brontir/NInferno, add `{name}` under
  `docs/performance/user-runs/`, and open a PR with this text as its description:

      git checkout -b user-run-{doc['created'][:10]}
      git add docs/performance/user-runs/{name}
      git commit -m "user run: {card}"
      gh pr create --title "user run: {card}" --body-file {Path(out_json).with_suffix('.md').name}

- Or as an issue: open https://github.com/Brontir/NInferno/issues/new with this text and attach
  `{name}`.

The file holds your card's facts (name, driver, clocks, limits), the OS, the CPU and the RAM, the
plan and the measurements; no user name, host name, path or GPU serial.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="the directory holding ninferno-serve and ninferno")
    ap.add_argument("--budget-minutes", type=float, default=30, help="GPU time (downloads apart)")
    ap.add_argument("--models", default="auto", help="auto, or ids from tools/kit/models.json")
    ap.add_argument("--max-models", type=int, default=2)
    ap.add_argument("--device", type=int, default=0, help="the GPU (CUDA ordinal = PCI bus order)")
    ap.add_argument("--passes", type=int, default=2, help="starts per model (a repeat each)")
    ap.add_argument("--prompts", type=int, default=3, help="timed requests per pass")
    ap.add_argument("--hard50", action="store_true", help="also the hard question set (quality)")
    ap.add_argument("--spec-tune", type=int, default=0, metavar="SECONDS",
                    help="also `ninferno spec-tune` per model for SECONDS (0: no)")
    ap.add_argument("--port", type=int, default=18390)
    ap.add_argument("--out", default=None, help="result file (default user-run-<gpu>-<date>.json)")
    ap.add_argument("--yes", action="store_true", help="fetch without asking")
    args = ap.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # the CUDA ordinal is nvidia-smi's index

    serve, cli = exe(args.bin, "ninferno-serve"), exe(args.bin, "ninferno")
    models_file = HERE / "models.json"
    models_doc = json.loads(models_file.read_text(encoding="utf-8"))
    wanted = models_doc["models"] if args.models == "auto" else [
        m for m in models_doc["models"] if m["id"] in args.models.split(",")]

    created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    version = subprocess.run([serve, "--version"], capture_output=True, text=True).stdout.strip()
    card = {"smi": gpu.card_facts(args.device)}
    doc = {"schema": SCHEMA, "kit": {"version": KIT_VERSION, "models_version": models_doc["version"],
                                     "models_sha256": hashlib.sha256(models_file.read_bytes()).hexdigest()[:16]},
           "created": created, "ninferno": {"version": version.splitlines()[0] if version else None,
                                            "build": version.splitlines()[1:3]},
           "host": gpu.host_facts(), "card": card, "runs": []}
    print(f"card: {card['smi'].get('name')}, driver {card['smi'].get('driver_version')}, "
          f"{card['smi'].get('memory.total')} MiB", flush=True)

    # The models whose plan fits this card, in the set's order (the plan's own answer, no guess).
    chosen = []
    for model in wanted:
        plan, error = plan_json(serve, model, run_flags(model, args.device))
        if plan is None:
            print(f"  {model['id']}: no plan ({error})", flush=True)
            continue
        card.setdefault("plan_device", plan.get("device"))
        fits = (plan.get("plan") or {}).get("fits")
        print(f"  {model['id']}: {'fits' if fits else 'does not fit'}", flush=True)
        if fits and len(chosen) < args.max_models:
            chosen.append((model, plan))
    if not chosen:
        print("No model of the kit's set fits or runs on this card (a card the engine does not support "
              "says so above); nothing to measure.")
    out_json = Path(args.out or f"user-run-{re.sub(r'[^A-Za-z0-9]+', '-', str(card['smi'].get('name')))}-"
                    f"{created[:10]}.json")
    workdir = out_json.with_suffix(".logs")
    workdir.mkdir(parents=True, exist_ok=True)

    # Fetch first (not GPU time): the bytes are said before anything is downloaded.
    if chosen:
        listed = []
        for model, plan in chosen:
            rows = [r for r in plan.get("quants") or [] if r.get("quant") == model["quant"]]
            listed.append(f"{model['repo']} {model['quant']} ({(rows[0].get('weights_bytes') or 0) / 2**30:.1f} GiB)"
                          if rows else f"{model['repo']} {model['quant']}")
        print("The kit fetches into your hub cache (skipped where already there): " + "; ".join(listed))
        if not args.yes and input("Fetch now? [y/N] ").strip().lower() != "y":
            sys.exit("Nothing fetched; nothing measured.")
        for model, _ in chosen:
            print(f"fetching {model['repo']} {model['quant']} ...", flush=True)
            _, error = prepare(serve, model, with_drafters=args.spec_tune > 0)
            if error:
                print(f"  failed: {error}", flush=True)

    per_pass = 120 + args.prompts * 60
    steps = len(chosen) * (args.passes + (1 if args.hard50 else 0) + (1 if args.spec_tune else 0))
    progress = Progress(args.budget_minutes * 60, max(1, steps))
    for model, plan in chosen:
        flags = run_flags(model, args.device)
        chosen_plan = (plan.get("choice") or {}).get("chosen") or {}
        depth = int(chosen_plan.get("depth") or model["context"] // 4)
        run = {"id": model["id"], "repo": model["repo"], "quant": model["quant"], "mode": "plain",
               "flags": flags, "plan": {"chosen": chosen_plan, "document": plan}, "passes": []}
        doc["runs"].append(run)
        for p in range(args.passes):
            if not progress.fits(per_pass, f"{model['id']} pass {p + 1}"):
                break
            progress.step(f"{model['id']}: plain start, pass {p + 1} of {args.passes} at depth {depth}")
            started = time.time()
            result = measure_pass(serve, model, flags, args.port, depth, args.prompts, args.device,
                                  workdir, f"{model['id']}.plain.{p + 1}")
            progress.spend(time.time() - started)
            run["passes"].append(result)
            m = result.get("measured") or {}
            print(f"    prefill {m.get('prefill', {}).get('median')} tok/s, decode "
                  f"{m.get('decode', {}).get('median')} tok/s (plan: prefill "
                  f"{chosen_plan.get('prefill_tok_s')}, decode {chosen_plan.get('decode_tok_s_at_depth')})"
                  + (f"; error: {result['error']}" if result.get("error") else ""), flush=True)
        run["served"] = next((p.get("served") for p in run["passes"] if p.get("served")), None)
        if args.hard50 and progress.fits(900, f"{model['id']} hard50"):
            progress.step(f"{model['id']}: the hard question set (plain)")
            started = time.time()
            doc["runs"].append(hard50(serve, model, flags, args.port, workdir))
            progress.spend(time.time() - started)
        if args.spec_tune and progress.fits(args.spec_tune + 300, f"{model['id']} spec-tune"):
            progress.step(f"{model['id']}: ninferno spec-tune for {args.spec_tune} s")
            started = time.time()
            doc["runs"].append(spec_tune(serve, cli, model, args.device, args.spec_tune, card, workdir))
            progress.spend(time.time() - started)
    doc["budget"] = {"asked_s": progress.budget_s, "used_s": round(progress.used), "skipped": progress.skipped}

    out_json.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    report = validate.render(validate.validate(doc))
    out_json.with_suffix(".md").write_text(submission_md(doc, report, out_json), encoding="utf-8")
    print("\n" + report)
    print(f"\nWrote {out_json} and {out_json.with_suffix('.md')} (how to send them is in the .md; "
          "nothing was sent).")
    return 0


def hard50(serve, model, flags, port, workdir):
    run = {"id": model["id"], "repo": model["repo"], "quant": model["quant"], "mode": "hard50"}
    server = Server(serve, model, flags, port, str(workdir / f"{model['id']}.hard50.serve.log"))
    out = workdir / f"{model['id']}.hard50.json"
    try:
        server.wait()
        subprocess.run([sys.executable, str(ROOT / "tools/bench/hard_set.py"), "--url",
                        f"http://127.0.0.1:{port}/v1", "--set", str(ROOT / "eval/corpora/hard50/hard50.json"),
                        "--tag", model["id"], "--out", str(out)], capture_output=True, text=True)
        result = json.loads(out.read_text(encoding="utf-8"))
        run["hard50"] = result.get("summary", result)
    except Exception as e:  # noqa: BLE001
        run["error"] = scrub(e)
    finally:
        server.stop()
    return run


def spec_tune(serve, cli, model, device, seconds, card, workdir):
    """`ninferno spec-tune` as the stands publish registry entries from, beside the plan's own
    speculation prediction (--spec auto)."""
    run = {"id": model["id"], "repo": model["repo"], "quant": model["quant"], "mode": "spec"}
    try:
        flags = run_flags(model, device, spec="auto")
        plan, error = plan_json(serve, model, flags)
        run["plan"] = {"chosen": ((plan or {}).get("choice") or {}).get("chosen"), "error": error}
        artifact, error = prepare(serve, model, with_drafters=True)
        if not artifact:
            raise RuntimeError(error or "no artifact")
        profile, publish = workdir / f"{model['id']}.spec_profile.json", workdir / f"{model['id']}.publish.json"
        out = subprocess.run([cli, "spec-tune", artifact, "--budget-seconds", str(seconds), "--device",
                              str(device), "--out", str(profile), "--publish", str(publish),
                              "--applies-to", model["repo"], "--measured-on",
                              f"{card['smi'].get('name')} user-kit"], capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError((out.stderr or out.stdout).strip().splitlines()[-1:])
        run["profile"] = json.loads(profile.read_text(encoding="utf-8"))
        run["publish"] = json.loads(publish.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        run["error"] = scrub(e)
    return run


if __name__ == "__main__":
    sys.exit(main())
