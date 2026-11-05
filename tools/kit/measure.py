#!/usr/bin/env python3
"""The user measurement kit (docs/maintainer/user-measurement-kit.md): on your own card, what our
stands measure, beside what the plan predicts for the same start, in one file you can send us.

    python tools/kit/measure.py --bin DIR [--full] [--budget-minutes 30] [--models auto|ID,...]
        [--device N] [--passes 2] [--hard50] [--spec-tune SECONDS] [--routes] [--out FILE] [--yes]

DIR holds ninferno-serve: a release or a build (`ninferno-serve spec-tune` and `route-tune` run
there; a build's own `ninferno` is used for --spec-tune where it is beside it). Standard library
only; Windows and Linux. The models are fetched through `ninferno-serve --prepare-only` into your
hub cache, after the bytes are stated (--yes skips the question): the next model downloads while
the kit measures the one before (--no-prefetch: all of them first), where the disk holds it. The
budget counts the GPU time only. --full measures everything for this card, for hours: every model
of the set that fits, plain passes, hard50, a speculation tune and the route sweep (the sweeps run
last, once nothing downloads beside them). Nothing is sent anywhere: the kit writes
FILE.json and FILE.md (a ready PR or issue body) and prints how to submit them yourself.
"""
import argparse
import collections
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import gpu  # noqa: E402
import validate  # noqa: E402

KIT_VERSION = "2"
SCHEMA = "ninferno-user-run/1.3"
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


def models_path(args):
    return Path(args.models_file) if args.models_file else HERE / "models.json"


def load_models(path):
    """The model set at `path`, or the run stops with what it lacks (before any GPU work)."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "version" not in doc:
            raise KeyError("version")
        for m in doc["models"]:
            for field in ("id", "repo", "quant", "context"):
                if field not in m:
                    raise KeyError(f"{field} of {m}")
            # The id names the run's log and temporary files: one plain file name component.
            if not isinstance(m["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", m["id"]):
                raise ValueError(f"id {m['id']!r}: letters, digits, '.', '_' and '-' only")
            # Windows reserves these names whatever the extension (CON.plain.1.server.log is the console).
            if re.fullmatch(r"(con|prn|aux|nul|com[1-9]|lpt[1-9])", m["id"].split(".")[0].lower()):
                raise ValueError(f"id {m['id']!r}: a Windows device name")
            # Room in a 255-byte file name for the longest suffix the run adds.
            if len(m["id"]) > 120:
                raise ValueError(f"id {m['id'][:20]!r}...: at most 120 characters")
            for field in ("repo", "quant"):
                if not isinstance(m[field], str) or not m[field]:
                    raise ValueError(f"{field} of {m['id']}: a string")
            if not isinstance(m["context"], int) or isinstance(m["context"], bool) or m["context"] <= 0:
                raise ValueError(f"context of {m['id']}: a positive integer")
        # Two models of one id would write one set of files (and case alone does not tell files
        # apart on Windows).
        ids = [m["id"].lower() for m in doc["models"]]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"ids used twice: {', '.join(dup)}")
        return doc
    except (OSError, ValueError, KeyError, TypeError) as e:
        sys.exit(f"the model set {path}: {e!r} (a version, and id, repo, quant, context a model; an id is one file name)")


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
    def __init__(self, serve, model, flags, port, log, env=None):
        if not port_free(port):
            raise RuntimeError(f"port {port} is in use (--port takes another)")
        self.port, self.log = port, log
        self.log_file = open(log, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen([serve, model["repo"], "--quant", model["quant"], "--port",
                                      str(port), "--model-id", "m"] + flags,
                                     stdout=self.log_file, stderr=subprocess.STDOUT, env=env)

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


# The fetches (main's background one, spec_tune's for drafters): `stop` set when the run ends, so
# no fetch starts and every running one is killed rather than waited for.
FETCH = {"stop": False, "procs": set()}
FETCH_LOCK = threading.Lock()
# Free disk a fetch needs beyond the bytes it writes (the artifacts' headers, the logs).
FETCH_MARGIN_BYTES = 4 << 30


def hub_dir():
    """The hub cache the fetch writes to (huggingface_hub's own order)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface" / "hub"


def free_bytes(path):
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


def cached_parts(model):
    """Whether the hub cache already holds the model's source weights (a snapshot file naming the
    quant, or safetensors) and its converted artifact (a .ninferno naming it)."""
    root = hub_dir() / ("models--" + model["repo"].replace("/", "--"))
    quant = model["quant"].lower()
    source = any(quant in f.name.lower() or f.suffix == ".safetensors"
                 for f in root.glob("snapshots/*/*"))
    converted = any(quant in f.name.lower() for f in root.glob("ninferno/*/*.ninferno"))
    return source, converted


def gguf_cached(model):
    """Whether the hub cache holds every part of the model's GGUF of its quant (a split
    name-0000i-of-0000n.gguf needs all n parts), for the measuring order: a sidecar or one part
    of a split is not the checkpoint."""
    root = hub_dir() / ("models--" + model["repo"].replace("/", "--"))
    quant = model["quant"].lower()
    parts = collections.defaultdict(set)
    for f in root.glob("snapshots/*/**/*.gguf"):
        name = f.name.lower()
        # A repository's drafter and projector files are not its checkpoint.
        if name.startswith(("mtp-", "dflash-", "mmproj")):
            continue
        # The quant as the name's last label (UD-Q6_K is not Q6_K, Q4_K_M is not Q4_K).
        if not f.exists() or not re.search(r"(^|[-.])" + re.escape(quant) + r"(-\d{5}-of-\d{5})?\.gguf$", name):
            continue
        split = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", name)
        if not split:
            return True
        # Parts of one split count together only inside one snapshot (one revision).
        snapshot = f.relative_to(root / "snapshots").parts[0]
        parts[(snapshot, name[:split.start()], int(split.group(2)))].add(int(split.group(1)))
    return any(have == set(range(1, total + 1)) for (_, _, total), have in parts.items())


def fetch_need_bytes(model, weights_bytes):
    """The disk a fetch writes: the source weights where they are not cached, and the converted
    artifact (about the weights' bytes again) where it is not, plus FETCH_MARGIN_BYTES."""
    source, converted = cached_parts(model)
    return weights_bytes * ((0 if source else 1) + (0 if converted else 1)) + FETCH_MARGIN_BYTES


def prepare(serve, model, with_drafters=False, weights_bytes=0, device=0):
    """Fetch and convert (the start's own cache) on the measured card: the artifact path from
    `model | prepared <path>` (apps/serve/main.cpp), and the error when it failed, when the disk
    has not what the fetch writes free (fetch_need_bytes: a full disk would break the runs beside
    it), or when the run stopped."""
    if FETCH["stop"]:
        return None, "not fetched: the run stopped"
    free = free_bytes(hub_dir())
    need = fetch_need_bytes(model, weights_bytes) if weights_bytes else 0
    if free is not None and need and free < need:
        return None, (f"not fetched: {free / 2**30:.0f} GiB free on the hub cache's disk, about "
                      f"{need / 2**30:.0f} GiB needed")
    cmd = [serve, model["repo"], "--quant", model["quant"], "--prepare-only", "--device", str(device)]
    if with_drafters:
        cmd.append("--with-drafters")
    try:
        with FETCH_LOCK:
            if FETCH["stop"]:
                return None, "not fetched: the run stopped"
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace")
            FETCH["procs"].add(proc)
        try:
            text, _ = proc.communicate()
        finally:
            with FETCH_LOCK:
                FETCH["procs"].discard(proc)
    except Exception as error:  # a fetch's failure is that model's, not the run's
        return None, scrub(f"{type(error).__name__}: {error}")
    found = re.findall(r"model \| prepared (.+?)\s*$", text, re.M)
    if proc.returncode != 0:
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
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)  # --full reads the options as spelt
    ap.add_argument("--bin", required=True, help="the directory holding ninferno-serve (and ninferno)")
    ap.add_argument("--budget-minutes", type=float, default=30, help="GPU time (downloads apart)")
    ap.add_argument("--models", default="auto", help="auto, or ids from the models file")
    ap.add_argument("--models-file", default=None,
                    help="the model set (default tools/kit/models.json): a lab's own list, e.g. the "
                         "largest checkpoints and their small quantizations on one card")
    ap.add_argument("--max-models", type=int, default=2)
    ap.add_argument("--device", type=int, default=0, help="the GPU (CUDA ordinal = PCI bus order)")
    ap.add_argument("--passes", type=int, default=2, help="starts per model (a repeat each)")
    ap.add_argument("--prompts", type=int, default=3, help="timed requests per kind per pass")
    ap.add_argument("--hard50", action="store_true", help="also the hard question set (quality)")
    ap.add_argument("--spec-tune", type=int, default=0, metavar="SECONDS",
                    help="also `ninferno spec-tune` per model for SECONDS (0: no)")
    ap.add_argument("--routes", action="store_true",
                    help="also the route sweep of each model's projections (ninferno-serve "
                         "route-tune; about an hour a model, after every model's other runs)")
    ap.add_argument("--route-passes", type=int, default=2, help="route sweeps per model (a repeat each)")
    ap.add_argument("--full", action="store_true",
                    help="everything for this card: every fitting model, hard50, spec-tune, routes")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="fetch every model before measuring (default: the next while one is measured)")
    ap.add_argument("--port", type=int, default=18390)
    ap.add_argument("--out", default=None, help="result file (default user-run-<gpu>-<date>.json)")
    ap.add_argument("--yes", action="store_true", help="fetch without asking")
    args = ap.parse_args()
    if args.full:
        # Hours, not minutes: the budget is a guard against a run that never ends, not a plan.
        given = {a.split("=")[0] for a in sys.argv[1:]}
        if "--budget-minutes" not in given:
            args.budget_minutes = 24 * 60
        if "--max-models" not in given:
            args.max_models = len(load_models(models_path(args))["models"])
        if "--spec-tune" not in given:
            args.spec_tune = 1200
        args.hard50 = args.routes = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # the CUDA ordinal is nvidia-smi's index
    # A visibility remapping would make the engine's --device another card than nvidia-smi's
    # --id: --device alone picks the card.
    if os.environ.pop("CUDA_VISIBLE_DEVICES", None) is not None:
        print("CUDA_VISIBLE_DEVICES is ignored by the kit: --device picks the card", flush=True)

    # The kit's data, resolved from the source tree's layout (a release installs that layout under
    # share/ninferno/kit): a copy of tools/kit alone would time nothing, hours later.
    needed = [HAYSTACK] + ([HARD_SET, ROOT / "tools/bench/hard_set.py"] if args.hard50 else [])
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        sys.exit(f"the kit's data is missing: {', '.join(missing)} (run measure.py from the source "
                 "tree or the release's bin/ninferno-kit)")

    serve = exe(args.bin, "ninferno-serve")
    # A build's own CLI where it is beside ninferno-serve, else the release's `ninferno-serve
    # spec-tune` (the same code, through the dispatcher's engine for this GPU).
    cli_path = Path(args.bin) / ("ninferno" + (".exe" if os.name == "nt" else ""))
    tune_cmd = [str(cli_path), "spec-tune"] if cli_path.exists() else [serve, "spec-tune"]
    models_file = models_path(args)
    models_doc = load_models(models_file)
    wanted = models_doc["models"] if args.models == "auto" else [
        m for m in models_doc["models"] if m["id"] in args.models.split(",")]

    created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    version = subprocess.run([serve, "--version"], capture_output=True, text=True).stdout.strip()
    smi_id = gpu.smi_id_of_cuda(args.device)  # the same card for nvidia-smi as for the engine
    card = {"smi": gpu.card_facts(smi_id)}
    wddm = str(card["smi"].get("driver_model.current") or "").upper() == "WDDM"
    doc = {"schema": SCHEMA, "kit": {"version": KIT_VERSION, "models_version": models_doc["version"],
                                     "models_sha256": hashlib.sha256(models_file.read_bytes()).hexdigest()[:16],
                                     "models_file": models_file.name if args.models_file else None},
           "created": created, "ninferno": {"version": version.splitlines()[0] if version else None},
           "host": gpu.host_facts(), "card": card, "runs": []}
    print(f"card: {card['smi'].get('name')}, driver {card['smi'].get('driver_version')}, "
          f"{card['smi'].get('memory.total')} MiB{', WDDM' if wddm else ''}", flush=True)

    out_json = Path(args.out or f"user-run-{re.sub(r'[^A-Za-z0-9]+', '-', str(card['smi'].get('name')))}-"
                    f"{created[:10]}.json")
    workdir = out_json.with_suffix(".logs")
    workdir.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.budget_minutes * 60, 1)
    fetcher = None
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
        # The fetches run one after another in the background, in that order: a model is
        # measured once its own fetch ended, while the next ones download (--no-prefetch: all
        # first). A conversion is the host's work: each run records whether a fetch ran beside it
        # (validate.py notes it), and the route sweeps, whose small widths time the host's
        # launches, run after the last fetch.
        # The models whose GGUF the cache already holds whole go first (in the set's order among
        # themselves): the card measures them while the others download, instead of waiting on
        # the first fetch (a lab set of a dozen checkpoints is hours of download).
        chosen.sort(key=lambda mp: not gguf_cached(mp[0]))
        fetcher = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fetches = []
        for model, plan in chosen:
            weights = int(((plan.get("model") or {}).get("weights_bytes")) or 0)
            fetches.append(fetcher.submit(prepare, serve, model, args.spec_tune > 0, weights,
                                          args.device))
        if args.no_prefetch:
            concurrent.futures.wait(fetches)
        fetching = lambda: any(not f.done() for f in fetches)  # noqa: E731
        sweeps = []

        per_pass = 150 + args.prompts * 90
        progress.steps = max(1, len(chosen) * (args.passes + (1 if args.hard50 else 0)
                                               + (1 if args.spec_tune else 0) + (1 if args.routes else 0)))
        for index, (model, plan) in enumerate(chosen):
            if not fetches[index].done():
                print(f"waiting for the fetch of {model['repo']} {model['quant']} ...", flush=True)
            _, error = fetches[index].result()
            if error:
                print(f"  {model['id']}: fetch failed: {error}; not measured", flush=True)
                doc["runs"].append({"id": model["id"], "repo": model["repo"], "quant": model["quant"],
                                    "mode": "plain", "error": "fetch: " + error})
                continue
            beside = fetching()
            flags = run_flags(model, args.device)
            choice = plan.get("choice") or {}
            chosen_plan = choice.get("chosen") or {}
            depth = int(chosen_plan.get("depth") or model["context"] // 4)
            prompt_len = int((choice.get("task") or {}).get("prompt") or 2048)
            run = {"id": model["id"], "repo": model["repo"], "quant": model["quant"], "mode": "plain",
                   "flags": flags, "plan": {"chosen": chosen_plan, "prompt": prompt_len,
                                            "document": kept_plan(plan)}, "passes": [],
                   "fetch_beside": beside}
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
                run["fetch_beside"] = run["fetch_beside"] or fetching()
                m = result.get("measured") or {}
                print(f"    prefill {m.get('prefill', {}).get('median')} tok/s, decode "
                      f"{m.get('decode', {}).get('median')} tok/s (plan: prefill "
                      f"{chosen_plan.get('prefill_tok_s')}, decode {chosen_plan.get('decode_tok_s_at_depth')})"
                      + (f"; error: {result['error']}" if result.get("error") else ""), flush=True)
            # 50 questions of about 1100 answer tokens at the plan's decode, and a start.
            hard_cost = 150 + 50 * 1100 / max(1.0, float(chosen_plan.get("decode_tok_s") or 30))
            if args.hard50 and progress.fits(hard_cost, f"{model['id']} hard50"):
                progress.step(f"{model['id']}: the hard question set (plain)")
                started, beside = time.time(), fetching()
                doc["runs"].append(hard50(serve, model, flags, args.port, workdir))
                doc["runs"][-1]["fetch_beside"] = beside or fetching()
                progress.spend(time.time() - started)
            if args.spec_tune and progress.fits(args.spec_tune + 300, f"{model['id']} spec-tune"):
                progress.step(f"{model['id']}: spec-tune for {args.spec_tune} s")
                started, beside = time.time(), fetching()
                doc["runs"].append(spec_tune(serve, tune_cmd, model, args.device, args.spec_tune, card,
                                              workdir, smi_id, wddm))
                doc["runs"][-1]["fetch_beside"] = beside or fetching()
                progress.spend(time.time() - started)
            # A route sweep pass is about two minutes a recorded problem; a model records ten to
            # twenty (its distinct projection shapes).
            if args.routes:
                sweeps.append((model, flags))
        # The route sweeps once nothing downloads beside them.
        concurrent.futures.wait(fetches)
        route_cost = 180 + args.route_passes * 15 * 120
        for model, flags in sweeps:
            if progress.fits(route_cost, f"{model['id']} route sweep"):
                progress.step(f"{model['id']}: the route sweep ({args.route_passes} passes)")
                started = time.time()
                doc["runs"].append(route_sweep(serve, model, flags, args.device, args.port,
                                               args.route_passes, workdir, smi_id, wddm))
                progress.spend(time.time() - started)
    finally:
        # No fetch outlives the run: the queued ones are cancelled, every running one killed.
        with FETCH_LOCK:
            FETCH["stop"] = True
            running = list(FETCH["procs"])
        if fetcher is not None:
            fetcher.shutdown(wait=False, cancel_futures=True)
        for proc in running:
            if proc.poll() is None:
                proc.kill()
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
    (by name: the weights are checked by their SHA-256 when judged), the newest pairs' first; empty
    when the lab measured none (spec-tune then runs its default menu). A user's tune of the same
    menu is what the pairs' acceptance and the plan's round costs can be held to; the lab pairs
    measured on master from 30.10 draft through the proposal head (`+head`) as the default menu
    does, the older ones without it, so the menu carries both where the lab has both."""
    try:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    # The newest pairs' configurations first (what the current engine serves, `+head` on master
    # from 30.10): a finite --budget-seconds stops starting configurations when it runs out, so the
    # older pairs' come last.
    pairs = sorted(registry.get("pairs", []), key=lambda p: str((p.get("tune") or {}).get("created", "")),
                   reverse=True)
    for p in pairs:
        repo, _, quant = str(p.get("target_repo", "")).partition(" ")
        if repo.split("@")[0] == model["repo"] and quant == model["quant"] and not p.get("user_kit"):
            for m in (p.get("tune") or {}).get("measurements", []):
                if m.get("config") not in out and m.get("config") != "plain":
                    out.append(m["config"])
    return out


SPEC_KEEP = ("ask", "served", "basis", "source", "round_cost", "by_thinking", "fallbacks", "notes")


def spec_tune(serve, tune_cmd, model, device, seconds, card, workdir, smi_id=None, wddm=False):
    """`ninferno spec-tune` as the stands publish registry entries from, under telemetry, beside
    the plan's own speculation (--spec auto: its basis, source and round costs), the drafter
    manifest and the target's identity (validate.py judges it; registry_add.py --from-user-run
    takes an accepted one)."""
    run = {"id": model["id"], "repo": model["repo"], "quant": model["quant"], "mode": "spec"}
    sampler = proc = None
    try:
        artifact, error = prepare(serve, model, with_drafters=True, device=device)
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
            # The tune runs for minutes and has no short window: 500 ms keeps its file small.
            sampler = gpu.Sampler(smi_id, period_ms=500).start()
        t0 = sampler.now() if sampler else 0
        run["configs"] = lab_configs(model)
        menu = ["--configs", ",".join(run["configs"])] if run["configs"] else []
        proc = subprocess.Popen(tune_cmd + [artifact, "--budget-seconds", str(seconds), "--device",
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


ROUTE_COLUMNS = ("format", "compute", "epilogue", "n", "k", "T")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def route_sweep(serve, model, flags, device, port, passes, workdir, smi_id=None, wddm=False):
    """This card's own timings of the routes the model's projections can take: the problems the
    engine runs for it, recorded by a start (NINFERNO_ROUTE_RECORD: every problem that reaches
    the generic path, once), then `ninferno-serve route-tune` over them `passes` times. Each pass
    keeps every timed (problem, T, route), the route the production path selects there and the
    route table it would write (validate.py: the selection's regret; a maintainer: a table)."""
    run = {"id": model["id"], "repo": model["repo"], "quant": model["quant"], "mode": "routes",
           "passes": []}
    shapes = workdir / f"{model['id']}.shapes"
    shapes.unlink(missing_ok=True)
    server = None
    try:
        env = dict(os.environ, NINFERNO_ROUTE_RECORD=str(shapes))
        server = Server(serve, model, flags, port, str(workdir / f"{model['id']}.routes.serve.log"), env)
        server.wait()
        # A decode and a prefill: the record holds the problems, whatever their width.
        request(port, "Say hello in one short sentence.", 32)
        request(port, prompt_of(text_blob(), 6000, 7, "routes"), 1)
        server.stop()
        server = None
        lines = sorted({line.strip() for line in shapes.read_text(encoding="utf-8").splitlines()
                        if line.strip() and not line.startswith("#")}) if shapes.exists() else []
        run["shapes"] = lines
        if not lines:
            raise RuntimeError("the start recorded no problem of the generic path")
        print(f"    {len(lines)} problems recorded", flush=True)
        # The tune runs on this card alone: its first CUDA device is the one in bus order.
        tune_env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(device))
        for p in range(passes):
            tag = f"{model['id']}.routes.{p + 1}"
            table, routes, selected = (workdir / f"{tag}.table.json", workdir / f"{tag}.csv",
                                       workdir / f"{tag}.selected.csv")
            for f in (table, routes, selected):
                f.unlink(missing_ok=True)
            sampler = gpu.Sampler(smi_id, period_ms=500).start() if smi_id else None
            t0 = sampler.now() if sampler else 0
            started = time.time()
            out = {}
            try:
                done = subprocess.run([serve, "route-tune", "--shapes", str(shapes), "--out", str(table),
                                       "--routes-out", str(routes), "--selected-out", str(selected)],
                                      env=tune_env, capture_output=True, text=True,
                                      timeout=len(lines) * 1800 + 600)
                out["seconds"] = round(time.time() - started)
                if sampler:
                    sampler.mark("route-tune", t0, sampler.now())
                    out["processes_after"] = apps_info(smi_id, wddm, os.getpid())
                if done.returncode != 0:
                    tail = (done.stderr or done.stdout).strip().splitlines()
                    raise RuntimeError("route-tune failed: " + (tail[-1] if tail else str(done.returncode)))
                out["rows"] = [[r[c] for c in ROUTE_COLUMNS] + [r["route"], float(r["median_us"])]
                               for r in read_csv(routes)]
                out["selected"] = [[r[c] for c in ROUTE_COLUMNS] + [r["asked"], r["selected"]]
                                   for r in read_csv(selected)]
                out["table"] = json.loads(table.read_text(encoding="utf-8"))
                print(f"    pass {p + 1}: {len(out['rows'])} timings in {out['seconds'] / 60:.0f} min",
                      flush=True)
            except Exception as e:  # noqa: BLE001 - the next pass and the other runs go on
                out["error"] = scrub(e)
            finally:
                if sampler:
                    out["telemetry"] = sampler.stop()
            run["passes"].append(out)
    except Exception as e:  # noqa: BLE001
        run["error"] = scrub(e)
    finally:
        if server:
            server.stop()
    return run


if __name__ == "__main__":
    sys.exit(main())
