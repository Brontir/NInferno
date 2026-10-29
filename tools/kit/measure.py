#!/usr/bin/env python3
"""The user measurement kit (docs/maintainer/user-measurement-kit.md): on your own card, what our
stands measure, beside what the plan predicts for the same start, in one file you can send us.

    python tools/kit/measure.py --bin DIR [--budget-minutes 30] [--models auto|ID,...] [--device N]
        [--passes 2] [--hard50] [--spec-tune SECONDS] [--out FILE] [--yes]

DIR holds ninferno-serve (and ninferno, for --spec-tune): a release or a build. Standard library
only; Windows and Linux. The models are fetched first through `ninferno-serve --prepare-only` into
your hub cache, after the bytes are stated (--yes skips the question); the budget counts the GPU
time only. Nothing is sent anywhere: the kit writes FILE.json and FILE.md (a ready PR or issue
body) and prints how to submit them yourself.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import socket
import statistics
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
SCHEMA = "ninferno-user-run/1.1"
HAYSTACK = ROOT / "examples/cli/messages/long_niah_256k.json"  # perf_matrix.py's long text
QUESTION = "\n\nSummarize the text above in detail, section by section."
START_TIMEOUT_S = 900
# The plan document's parts kept in the file (the rest may name local files).
PLAN_KEEP = ("schema", "ask", "device", "choice", "estimate", "pick", "rule", "rule_code")
PLAN_PLAN_KEEP = ("fits", "context", "rung", "rungs", "planned", "private_contexts", "kv_dtype",
                  "state_dtype", "weights_bytes", "free_bytes", "reason_code", "estimate_basis")


def exe(bin_dir, name):
    path = Path(bin_dir) / (name + (".exe" if os.name == "nt" else ""))
    if not path.exists():
        sys.exit(f"{path} not found (--bin is the directory holding ninferno-serve and ninferno)")
    return str(path)


def private_strings():
    """The local strings a result must not carry: the home (it holds the user name), the working
    directory and the hub cache, longest first."""
    out = {str(Path.home()), os.getcwd(), str(ROOT)}
    for var in ("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA"):
        if os.environ.get(var):
            out.add(os.environ[var])
    return sorted((s for s in out if s and len(s) > 2), key=len, reverse=True)


PRIVATE = private_strings()


def scrub(value):
    """A string without the user's paths or name: the known local strings, then any remaining
    absolute path."""
    text = str(value)
    for s in PRIVATE:
        text = text.replace(s, "<local>").replace(s.replace("\\", "/"), "<local>")
    # Any other absolute path, to the end of its line (a path may hold spaces and quotes, so what
    # follows it on the line goes too): a drive path with either separator, a UNC share with
    # either separator, or a POSIX path (a slash after a space, a quote or the start, then a name:
    # not a URL's //host, and not tok/s, N/A or org/repo, whose slash follows a letter).
    text = re.sub(r"(?<![\w.])[A-Za-z]:[\\/][^\n]*", "<path>", text)
    text = re.sub(r"\\\\[^\n]+", "<path>", text)
    text = re.sub(r"(?<![\w.:/])//[^/\s][^\n]*", "<path>", text)
    text = re.sub(r"(?<![\w.:/<>])/[^\s/][^\n]*", "<path>", text)
    return text


def scrub_doc(obj):
    """Every string of a document scrubbed; keys that hold file paths dropped."""
    if isinstance(obj, dict):
        return {k: scrub_doc(v) for k, v in obj.items() if k not in ("artifact", "path", "file", "files")}
    if isinstance(obj, list):
        return [scrub_doc(v) for v in obj]
    if isinstance(obj, str):
        return scrub(obj)
    return obj


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
                          "--json"] + flags, capture_output=True, text=True, timeout=900)
    try:
        return json.loads(out.stdout), None
    except json.JSONDecodeError:
        lines = (out.stderr or out.stdout).strip().splitlines()
        return None, scrub(lines[-1] if lines else "no plan")


def kept_plan(doc):
    out = {k: doc[k] for k in PLAN_KEEP if k in doc}
    out["plan"] = {k: v for k, v in (doc.get("plan") or {}).items() if k in PLAN_PLAN_KEEP}
    out["model"] = {k: v for k, v in (doc.get("model") or {}).items() if k != "source"}
    return out


def run_flags(model, device, spec="none"):
    return ["--max-context", str(model["context"]), "--max-concurrency", "1", "--spec", spec,
            "--device", str(device)]


def port_free(port):
    """Nothing listens on the port (a closed socket in TIME_WAIT from the last start is free)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        return s.connect_ex(("127.0.0.1", port)) != 0


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
        if not port_free(port):
            raise RuntimeError(f"port {port} is in use (--port takes another)")
        self.port, self.log = port, log
        self.log_file = open(log, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen([serve, model["repo"], "--quant", model["quant"], "--port",
                                      str(port), "--model-id", "m"] + flags,
                                     stdout=self.log_file, stderr=subprocess.STDOUT)

    def wait(self):
        start = time.time()
        while time.time() - start < START_TIMEOUT_S:
            if self.proc.poll() is not None:
                raise RuntimeError("the server stopped: " + scrub(self.tail(2)))
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5):
                    pass
            except OSError:
                time.sleep(1)
                continue
            # Ours: still running, and serving the id it was given.
            if self.proc.poll() is None and self.meta().get("id") == "m":
                return time.time() - start
            raise RuntimeError("another server answers on the port")
        raise RuntimeError("the server did not start in time")

    def meta(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=30) as r:
            model = json.load(r)["data"][0]
        meta = model.get("meta") or {}
        return {"id": model.get("id"), "fingerprint": meta.get("decision_fingerprint"),
                "speculation": meta.get("speculation")}

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
            self.proc.wait(timeout=30)
        self.log_file.close()


def text_blob():
    blob = json.loads(HAYSTACK.read_text(encoding="utf-8"))[-1]["content"]
    return blob.split("<document>\n", 1)[1].split("\n</document>", 1)[0]


def prompt_of(blob, chars, i, nonce):
    """A slice of the long text at its own offset, after a nonce line (a probe and its timed
    request share no reusable prefix beyond the chat template's head), and the question."""
    # The offset is the slice's own, whatever its length (a probe and its timed request read the
    # same text): within the first half of the text, the second half left for the length.
    offset = (i * 7919 * 13 + i * 104729) % max(1, len(blob) // 2)
    return f"[{nonce}]\n" + blob[offset:offset + chars] + QUESTION


def apps_info(bus, wddm, server_pid):
    apps = gpu.compute_apps(bus)
    if apps is None:
        return {"listed": None}
    ours = gpu.process_tree(server_pid)  # the server and its engine child (Windows dispatch)
    others = [a for a in apps if a.get("pid") not in ours]
    # Under WDDM every process holding a context is listed: only the count is kept.
    return {"listed": len(apps), "others": len(others), "informative": not wddm}


def measure_pass(serve, model, flags, port, depth, prompt_len, prompts, smi_id, workdir, tag, wddm):
    """One start: an idle window, a warm-up and the template's overhead, then `prompts` decode
    requests at the plan's depth and `prompts` prefill requests at the choice's prompt length,
    each timed by the server's own timings (prefill = prompt_n / prompt_ms; decode = (predicted_n
    - 1) / predicted_ms, serving.md), under telemetry."""
    out = {"telemetry": None}
    server = sampler = None
    try:
        server = Server(serve, model, flags, port, str(workdir / f"{tag}.serve.log"))
        out["start_s"] = round(server.wait(), 1)
        out["served"] = server.meta()
        out["served"]["choice"] = server.choice_line()
        bus = smi_id
        out["processes_before"] = apps_info(bus, wddm, server.proc.pid)
        sampler = gpu.Sampler(smi_id).start()
        t0 = sampler.now()
        time.sleep(5)  # the idle window (R2)
        sampler.mark("idle", t0, sampler.now())
        blob = text_blob()
        t0 = sampler.now()
        request(port, "Say hello in one short sentence.", 16)
        base, _ = request(port, "[w]\n" + QUESTION.strip(), 1)  # the template, a nonce, the question
        overhead = base.get("prompt_tokens", 0)
        # Each timed prompt is sized on its own slice: a probe of that slice (its own nonce: nothing
        # of it is reused) gives the slice's characters per token.
        sizes = {}

        def size(kind, i, tokens):
            key = (kind, i)
            index = i + (0 if kind == "decode" else 50)
            chars = int(max(16, tokens - overhead) * 4)
            for _ in range(2):
                usage, _ = request(port, prompt_of(blob, chars, index, f"probe {kind} {i} {chars}"), 1)
                got = max(1, usage.get("prompt_tokens", tokens) - overhead)
                chars = int(chars * max(16, tokens - overhead) / got)
            sizes[key] = chars
        for i in range(prompts):
            size("decode", i, depth)
            size("prefill", i, prompt_len)
        sampler.mark("warmup", t0, sampler.now())
        out_tokens = max(32, min(256, depth // 8))
        rows = []

        def timed(kind, i, tokens, max_tokens):
            chars = sizes[(kind, i)]
            t_a = sampler.now()
            usage, timings = request(port, prompt_of(blob, chars, i + (0 if kind == "decode" else 50),
                                                     f"run {kind} {i}"), max_tokens)
            t_b = sampler.now()
            prompt_ms = timings.get("prompt_ms") or 0
            predicted_ms = timings.get("predicted_ms") or 0
            split = t_a + (t_b - t_a) * prompt_ms / max(1e-9, prompt_ms + predicted_ms)
            sampler.mark(f"{kind} {i}", t_a if kind == "prefill" else split, split if kind == "prefill" else t_b)
            row = {"kind": kind, "prompt_tokens": usage.get("prompt_tokens"), "cache_n": timings.get("cache_n"),
                   "prompt_n": timings.get("prompt_n"), "prompt_ms": prompt_ms,
                   "predicted_n": timings.get("predicted_n"), "predicted_ms": predicted_ms}
            # A row that cannot be timed as asked: a reused prefix, an early end of the answer.
            if timings.get("cache_n"):
                row["excluded"] = "a reused prefix (cache_n > 0)"
            elif kind == "decode" and (timings.get("predicted_n") or 0) < out_tokens / 2:
                row["excluded"] = "the answer ended early"
            rows.append(row)

        for i in range(prompts):
            timed("decode", i, depth, out_tokens)
        for i in range(prompts):
            timed("prefill", i, prompt_len, 1)
        out["requests"] = rows
        out["processes_after"] = apps_info(bus, wddm, server.proc.pid)

        def summary(kind, values):
            used = [r for r in rows if r["kind"] == kind and "excluded" not in r]
            tokens = sorted(r["prompt_tokens"] for r in used if r["prompt_tokens"])
            values = sorted(values)
            return {"median": round(statistics.median(values), 2), "min": round(values[0], 2),
                    "max": round(values[-1], 2), "n": len(values),
                    "prompt_tokens": statistics.median(tokens) if tokens else None,
                    "prompt_tokens_range": [tokens[0], tokens[-1]] if tokens else None} if values else {}
        prefill = [r["prompt_n"] / (r["prompt_ms"] / 1000) for r in rows
                   if r["kind"] == "prefill" and "excluded" not in r and r["prompt_n"] and r["prompt_ms"]]
        decode = [(r["predicted_n"] - 1) / (r["predicted_ms"] / 1000) for r in rows
                  if r["kind"] == "decode" and "excluded" not in r and (r["predicted_n"] or 0) > 1
                  and r["predicted_ms"]]
        out["measured"] = {"prefill": summary("prefill", prefill), "decode": summary("decode", decode)}
    except Exception as e:  # noqa: BLE001 - the file records the failure, the other runs go on
        out["error"] = scrub(e)
    finally:
        if sampler:
            out["telemetry"] = sampler.stop()
        if server:
            server.stop()
    return out


def prepare(serve, model, with_drafters=False):
    """Fetch and convert (the start's own cache): the artifact path from `model | prepared <path>`
    (apps/serve/main.cpp), and the error when it failed."""
    cmd = [serve, model["repo"], "--quant", model["quant"], "--prepare-only"]
    if with_drafters:
        cmd.append("--with-drafters")
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    found = re.findall(r"model \| prepared (.+?)\s*$", text, re.M)
    if out.returncode != 0:
        lines = text.strip().splitlines()
        return None, scrub(lines[-1] if lines else "failed")
    return (found[-1] if found else None), None


def submission_md(doc, report_text, out_json):
    card = doc["card"]["smi"].get("name")
    name = Path(out_json).name
    return f"""# User measurement: {card}

Kit {doc['kit']['version']} ({doc['schema']}), {doc['ninferno'].get('version')}, {doc['created']}.

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
plan and the measurements. Local paths and your user name are replaced by `<local>`; no host name
or GPU serial is collected. Read it before you send it.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="the directory holding ninferno-serve (and ninferno)")
    ap.add_argument("--budget-minutes", type=float, default=30, help="GPU time (downloads apart)")
    ap.add_argument("--models", default="auto", help="auto, or ids from tools/kit/models.json")
    ap.add_argument("--max-models", type=int, default=2)
    ap.add_argument("--device", type=int, default=0, help="the GPU (CUDA ordinal = PCI bus order)")
    ap.add_argument("--passes", type=int, default=2, help="starts per model (a repeat each)")
    ap.add_argument("--prompts", type=int, default=3, help="timed requests per kind per pass")
    ap.add_argument("--hard50", action="store_true", help="also the hard question set (quality)")
    ap.add_argument("--spec-tune", type=int, default=0, metavar="SECONDS",
                    help="also `ninferno spec-tune` per model for SECONDS (0: no)")
    ap.add_argument("--port", type=int, default=18390)
    ap.add_argument("--out", default=None, help="result file (default user-run-<gpu>-<date>.json)")
    ap.add_argument("--yes", action="store_true", help="fetch without asking")
    args = ap.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # the CUDA ordinal is nvidia-smi's index
    # A visibility remapping would make the engine's --device another card than nvidia-smi's
    # --id: --device alone picks the card.
    if os.environ.pop("CUDA_VISIBLE_DEVICES", None) is not None:
        print("CUDA_VISIBLE_DEVICES is ignored by the kit: --device picks the card", flush=True)

    serve = exe(args.bin, "ninferno-serve")
    cli = exe(args.bin, "ninferno") if args.spec_tune else None
    models_file = HERE / "models.json"
    models_doc = json.loads(models_file.read_text(encoding="utf-8"))
    wanted = models_doc["models"] if args.models == "auto" else [
        m for m in models_doc["models"] if m["id"] in args.models.split(",")]

    created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    version = subprocess.run([serve, "--version"], capture_output=True, text=True).stdout.strip()
    smi_id = gpu.smi_id_of_cuda(args.device)  # the same card for nvidia-smi as for the engine
    card = {"smi": gpu.card_facts(smi_id)}
    wddm = str(card["smi"].get("driver_model.current") or "").upper() == "WDDM"
    doc = {"schema": SCHEMA, "kit": {"version": KIT_VERSION, "models_version": models_doc["version"],
                                     "models_sha256": hashlib.sha256(models_file.read_bytes()).hexdigest()[:16]},
           "created": created, "ninferno": {"version": version.splitlines()[0] if version else None},
           "host": gpu.host_facts(), "card": card, "runs": []}
    print(f"card: {card['smi'].get('name')}, driver {card['smi'].get('driver_version')}, "
          f"{card['smi'].get('memory.total')} MiB{', WDDM' if wddm else ''}", flush=True)

    out_json = Path(args.out or f"user-run-{re.sub(r'[^A-Za-z0-9]+', '-', str(card['smi'].get('name')))}-"
                    f"{created[:10]}.json")
    workdir = out_json.with_suffix(".logs")
    workdir.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.budget_minutes * 60, 1)
    try:
        # The models whose plan fits this card, in the set's order (the plan's own answer).
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
            print("No model of the kit's set fits or runs on this card (a card the engine does not "
                  "support says so above); nothing to measure.")

        # Fetch first (not GPU time): the bytes are said before anything is downloaded.
        if chosen:
            listed = []
            for model, plan in chosen:
                size = ((plan.get("model") or {}).get("weights_bytes") or 0) / 2**30
                listed.append(f"{model['repo']} {model['quant']} ({size:.1f} GiB"
                              + (" and its drafters" if args.spec_tune else "") + ")")
            print("The kit fetches into your hub cache (what is already there is not fetched again): "
                  + "; ".join(listed))
            if not args.yes and input("Fetch now? [y/N] ").strip().lower() != "y":
                sys.exit("Nothing fetched; nothing measured.")
            fetched = []
            for model, plan in chosen:
                print(f"fetching {model['repo']} {model['quant']} ...", flush=True)
                _, error = prepare(serve, model, with_drafters=args.spec_tune > 0)
                if error:
                    print(f"  failed: {error}; not measured", flush=True)
                    doc["runs"].append({"id": model["id"], "repo": model["repo"], "quant": model["quant"],
                                        "mode": "plain", "error": "fetch: " + error})
                    continue
                fetched.append((model, plan))
            chosen = fetched

        per_pass = 150 + args.prompts * 90
        progress.steps = max(1, len(chosen) * (args.passes + (1 if args.hard50 else 0)
                                               + (1 if args.spec_tune else 0)))
        for model, plan in chosen:
            flags = run_flags(model, args.device)
            choice = plan.get("choice") or {}
            chosen_plan = choice.get("chosen") or {}
            depth = int(chosen_plan.get("depth") or model["context"] // 4)
            prompt_len = int((choice.get("task") or {}).get("prompt") or 2048)
            run = {"id": model["id"], "repo": model["repo"], "quant": model["quant"], "mode": "plain",
                   "flags": flags, "plan": {"chosen": chosen_plan, "prompt": prompt_len,
                                            "document": kept_plan(plan)}, "passes": []}
            doc["runs"].append(run)
            # A model's passes are measured whole or not at all (one pass cannot be validated).
            if not progress.fits(per_pass * args.passes, f"{model['id']} ({args.passes} passes)"):
                run["error"] = "skipped: the budget"
                continue
            for p in range(args.passes):
                progress.step(f"{model['id']}: plain start, pass {p + 1} of {args.passes} (decode at "
                              f"{depth}, prefill at {prompt_len} tokens)")
                started = time.time()
                result = measure_pass(serve, model, flags, args.port, depth, prompt_len, args.prompts,
                                      smi_id, workdir, f"{model['id']}.plain.{p + 1}", wddm)
                progress.spend(time.time() - started)
                run["passes"].append(result)
                m = result.get("measured") or {}
                print(f"    prefill {m.get('prefill', {}).get('median')} tok/s, decode "
                      f"{m.get('decode', {}).get('median')} tok/s (plan: prefill "
                      f"{chosen_plan.get('prefill_tok_s')}, decode {chosen_plan.get('decode_tok_s_at_depth')})"
                      + (f"; error: {result['error']}" if result.get("error") else ""), flush=True)
            # 50 questions of about 1100 answer tokens at the plan's decode, and a start.
            hard_cost = 150 + 50 * 1100 / max(1.0, float(chosen_plan.get("decode_tok_s") or 30))
            if args.hard50 and progress.fits(hard_cost, f"{model['id']} hard50"):
                progress.step(f"{model['id']}: the hard question set (plain)")
                started = time.time()
                doc["runs"].append(hard50(serve, model, flags, args.port, workdir))
                progress.spend(time.time() - started)
            if args.spec_tune and progress.fits(args.spec_tune + 300, f"{model['id']} spec-tune"):
                progress.step(f"{model['id']}: ninferno spec-tune for {args.spec_tune} s")
                started = time.time()
                doc["runs"].append(spec_tune(serve, cli, model, args.device, args.spec_tune, card, workdir,
                                              smi_id, wddm))
                progress.spend(time.time() - started)
    finally:
        # Written whatever happened (an interrupted run keeps what it measured).
        doc["budget"] = {"asked_s": progress.budget_s, "used_s": round(progress.used),
                         "skipped": progress.skipped}
        doc = scrub_doc(doc)
        out_json.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        report = validate.render(validate.validate(doc))
        out_json.with_suffix(".md").write_text(submission_md(doc, report, out_json), encoding="utf-8")
        print("\n" + report)
        print(f"\nWrote {out_json} and {out_json.with_suffix('.md')} (how to send them is in the .md; "
              "nothing was sent).")
    return 0


HARD_SET = ROOT / "eval/corpora/hard50/hard50.json"
REGISTRY = ROOT / "src/spec/data/drafter_registry.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def hard50(serve, model, flags, port, workdir):
    run = {"id": model["id"], "repo": model["repo"], "quant": model["quant"], "mode": "hard50"}
    out = workdir / f"{model['id']}.hard50.json"
    out.unlink(missing_ok=True)  # never an earlier run's file
    server = None
    try:
        run["set_sha256"] = sha256_of(HARD_SET)
        server = Server(serve, model, flags, port, str(workdir / f"{model['id']}.hard50.serve.log"))
        server.wait()
        run["served"] = {"choice": server.choice_line()}
        done = subprocess.run([sys.executable, str(ROOT / "tools/bench/hard_set.py"), "--url",
                               f"http://127.0.0.1:{port}/v1", "--set", str(HARD_SET), "--tag", model["id"],
                               "--out", str(out)], capture_output=True, text=True)
        if done.returncode != 0:
            lines = (done.stderr or done.stdout).strip().splitlines()
            raise RuntimeError("hard_set.py failed: " + (lines[-1] if lines else str(done.returncode)))
        result = json.loads(out.read_text(encoding="utf-8"))
        run["hard50"] = result.get("summary", {k: v for k, v in result.items() if k != "questions"})
        # The ids answered wrong (the set's own ids, no answer text): what the lab's runs are
        # compared with item by item (tools/kit/hard50_reference.json).
        run["hard50"]["wrong"] = sorted(r["id"] for r in result.get("rows", []) if not r.get("correct"))
    except Exception as e:  # noqa: BLE001
        run["error"] = scrub(e)
    finally:
        if server:
            server.stop()
    return run


def declared_drafter_repos():
    """The drafter repositories the registry names (its models' and its pairs'): a manifest's other
    drafters are recorded as "other" (a private or local repository names nothing of ours)."""
    try:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    repos = {d.get("repo") for m in registry.get("models", []) for d in m.get("drafters", [])}
    repos |= {v.split("@")[0] for p in registry.get("pairs", []) for v in (p.get("drafters") or {}).values()}
    return {r for r in repos if r}


def manifest_of(artifact):
    """The drafter manifest beside the artifact, without its profiles and paths."""
    path = Path(artifact + ".speculation.json")
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    known = declared_drafter_repos()
    target = doc.get("target") or {}
    out = {"schema": doc.get("schema"), "entry": doc.get("entry"), "matched_by": doc.get("matched_by"),
           "target": {k: target.get(k) for k in ("engine", "repo", "architecture", "builtin_mtp",
                                                 "vocab_fingerprint") if k in target},
           "drafters": []}
    for d in doc.get("drafters", []):
        kept = {k: d.get(k) for k in ("name", "kind", "component", "status", "compatible") if k in d}
        if d.get("repo"):
            if d["repo"] in known:
                kept["repo"], kept["revision"] = d["repo"], d.get("revision", "")
            else:
                kept["repo"] = "other"
        out["drafters"].append(kept)
    return out


def target_identity(artifact):
    """The target's hub revision and its weight files' SHA-256 (the registry pairs' key): the GGUF
    files of the snapshot the artifact was converted from, hashed (a blob's name is what the hub
    promised, not what is on disk). None when the artifact is not in a hub cache's layout."""
    path = Path(artifact)
    revision = path.parent.name  # <hub>/models--org--name/ninferno/<revision>/<artifact>
    if path.parent.parent.name != "ninferno" or not re.fullmatch(r"[0-9a-f]{40}", revision):
        return None
    snapshot = path.parent.parent.parent / "snapshots" / revision
    weights = []
    for f in sorted(snapshot.glob("*.gguf")):
        base = re.sub(r"-\d{5}-of-\d{5}$", "", f.stem)
        # A split model's artifact is named after its first part or after the model.
        first = re.sub(r"-\d{5}-of-(\d{5})$", r"-00001-of-\1", f.stem)
        named = path.name.startswith(base + ".") or path.name.startswith(first + ".")
        if not named or "mmproj" in f.name.lower():
            continue
        entry = {"sha256": sha256_of(f), "bytes": f.stat().st_size}
        try:
            blob = Path(os.readlink(f)).name
        except OSError:
            blob = ""
        if re.fullmatch(r"[0-9a-f]{64}", blob):
            entry["blob_matches"] = blob == entry["sha256"]
        weights.append(entry)
    return {"revision": revision, "weights": weights}


def lab_configs(model):
    """The configurations the registry's lab pairs of this repository and quantization measured
    (by name: the weights are checked by their SHA-256 when judged), in the registry's order; empty
    when the lab measured none (spec-tune then runs its default menu). A user's tune of the same
    menu is what the pairs' acceptance and the plan's round costs can be held to: the default menu
    drafts through the proposal head where the artifact has one, which no lab pair measured."""
    try:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    for p in registry.get("pairs", []):
        repo, _, quant = str(p.get("target_repo", "")).partition(" ")
        if repo.split("@")[0] == model["repo"] and quant == model["quant"] and not p.get("user_kit"):
            for m in (p.get("tune") or {}).get("measurements", []):
                if m.get("config") not in out and m.get("config") != "plain":
                    out.append(m["config"])
    return out


SPEC_KEEP = ("ask", "served", "basis", "source", "round_cost", "by_thinking", "fallbacks", "notes")


def spec_tune(serve, cli, model, device, seconds, card, workdir, smi_id=None, wddm=False):
    """`ninferno spec-tune` as the stands publish registry entries from, under telemetry, beside
    the plan's own speculation (--spec auto: its basis, source and round costs), the drafter
    manifest and the target's identity (validate.py judges it; registry_add.py --from-user-run
    takes an accepted one)."""
    run = {"id": model["id"], "repo": model["repo"], "quant": model["quant"], "mode": "spec"}
    sampler = proc = None
    try:
        artifact, error = prepare(serve, model, with_drafters=True)
        if not artifact:
            raise RuntimeError(error or "no artifact")
        # Planned after the drafters are fetched: the plan resolves the profile from their manifest.
        plan, error = plan_json(serve, model, run_flags(model, device, spec="auto"))
        run["plan"] = {"chosen": ((plan or {}).get("choice") or {}).get("chosen"), "error": error,
                       "speculation": {k: v for k, v in (((plan or {}).get("plan") or {})
                                                         .get("speculation") or {}).items() if k in SPEC_KEEP}}
        run["manifest"] = manifest_of(artifact)
        print("    hashing the target's weight files (the registry's key) ...", flush=True)
        run["target"] = target_identity(artifact)
        profile, publish = workdir / f"{model['id']}.spec_profile.json", workdir / f"{model['id']}.publish.json"
        for f in (profile, publish):
            f.unlink(missing_ok=True)  # never an earlier run's file
        if smi_id:
            sampler = gpu.Sampler(smi_id).start()
        t0 = sampler.now() if sampler else 0
        run["configs"] = lab_configs(model)
        menu = ["--configs", ",".join(run["configs"])] if run["configs"] else []
        proc = subprocess.Popen([cli, "spec-tune", artifact, "--budget-seconds", str(seconds), "--device",
                                 str(device), "--out", str(profile), "--publish", str(publish),
                                 "--applies-to", model["repo"], "--measured-on",
                                 f"{card['smi'].get('name')} user-kit"] + menu,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if smi_id:
            time.sleep(5)  # the tune has loaded its first model by then
            run["processes_before"] = apps_info(smi_id, wddm, proc.pid)
        stdout, stderr = proc.communicate(timeout=3 * seconds + 900)
        if sampler:
            sampler.mark("spec-tune", t0, sampler.now())
            run["processes_after"] = apps_info(smi_id, wddm, proc.pid)
        if proc.returncode != 0:
            lines = (stderr or stdout).strip().splitlines()
            raise RuntimeError(lines[-1] if lines else "spec-tune failed")
        run["profile"] = json.loads(profile.read_text(encoding="utf-8"))
        run["publish"] = json.loads(publish.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        run["error"] = scrub(e)
    finally:
        # Never a tune left running on the card (an error, a timeout, Ctrl-C).
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        if sampler:
            run["telemetry"] = sampler.stop()
    return run


if __name__ == "__main__":
    sys.exit(main())
