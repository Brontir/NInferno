"""Soak and chaos driver for a running ninferno-serve.

soak:  mixed traffic for --seconds from --workers threads: plain and streaming chat, agentic
       multi-turn tool conversations over one long shared system prompt, structured output,
       max_tokens=1, seeded sampling, long prompts, Anthropic streams, cancellations mid-stream and
       client disconnects before the first byte. Every --canary-seconds the traffic pauses and a
       fixed greedy request runs alone and is compared with the first canary. A changed canary
       is a note, not a failure: the prefill plan of a cold request depends on the cache state
       (capture splits), which the contract classes as D; a failed canary request is a failure,
       and corruption is judged by the witnesses. Every
       --sample-seconds the server's RSS and VRAM and the counters are sampled.
chaos: storms with idle checkpoints between them: disconnects at 0/1/5/20 ms and after 1/3/10
       stream chunks, a cancel storm during long prefills, saturation beyond
       max-concurrency + max-pending, and (--storms shared) --shared-requests requests over
       one ~10k-token system prompt with 30 % cancelled at random moments, after which a fixed
       greedy request over that prompt must repeat its pre-storm answer. Two witness workers run
       a copy task (exact expected output) throughout every storm: a witness that does not return
       its line is a corrupted neighbour.
       After each storm the server must be healthy, the solo canary is compared (as above), and
       RSS/VRAM are recorded. Overload must be an honest 429/503 JSON answer, never a 5xx other
       than 503, a dropped connection or a hang.

The JSON summary goes to stdout, samples and events to --out (JSONL). Exit 1 on any failure.
Standard library only.
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
          "required": ["name", "age"], "additionalProperties": False}
WITNESS_LINES = [
    "amber falcon 4417 quiet river",
    "seven copper lanterns 9021 north",
    "velvet orbit 3380 paper harbor",
    "granite tulip 7154 silver canyon",
]
NO_THINK = {"enable_thinking": False}


@dataclass
class Result:
    status: int = 0
    body: bytes = b""
    error: str = ""
    chunks: int = 0
    seconds: float = 0.0


@dataclass
class Stats:
    lock: threading.Lock = field(default_factory=threading.Lock)
    counts: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)

    def note(self, what: str, detail: Any) -> None:
        """A contract-D observation: recorded and counted, not a failure."""
        with self.lock:
            self.counts["notes"] = self.counts.get("notes", 0) + 1
            if len(self.notes) < 200:
                self.notes.append({"t": time.time(), "what": what, "detail": detail})
        print(f"NOTE {what}: {str(detail)[:300]}", file=sys.stderr, flush=True)

    def count(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + n

    def fail(self, what: str, detail: Any) -> None:
        with self.lock:
            self.counts["failures"] = self.counts.get("failures", 0) + 1
            if len(self.failures) < 200:
                self.failures.append({"t": time.time(), "what": what, "detail": detail})
        print(f"FAIL {what}: {str(detail)[:300]}", file=sys.stderr, flush=True)


class Server:
    def __init__(self, base_url: str, pid: int | None) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("", "http"):
            raise SystemExit(f"{base_url}: the server speaks plain HTTP (http://host:port)")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80
        self.pid = pid

    def request(self, path: str, payload: Any, *, stream: bool = False, stop_after_chunks: int = -1,
                disconnect_after_s: float | None = None, timeout: float = 900.0) -> Result:
        """POSTs payload. stop_after_chunks >= 0 closes the socket after that many SSE chunks;
        disconnect_after_s closes it that long after sending, before reading anything."""
        result = Result()
        start = time.monotonic()
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
            if disconnect_after_s is not None:
                time.sleep(disconnect_after_s)
                result.error = "client-disconnect"
                return result
            response = conn.getresponse()
            result.status = response.status
            if stream and response.status == 200 and stop_after_chunks >= 0:
                buffer = b""
                while result.chunks < stop_after_chunks:
                    piece = response.read1(4096) if hasattr(response, "read1") else response.read(1)
                    if not piece:
                        break
                    buffer += piece
                    result.chunks = buffer.count(b"\n\n")
                result.body = buffer
                result.error = "client-cancel"
                return result
            result.body = response.read()
            if stream:
                result.chunks = result.body.count(b"\n\n")
        except (OSError, http.client.HTTPException) as error:
            result.error = f"{type(error).__name__}: {error}"
        finally:
            try:
                if conn.sock is not None:
                    conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()
            result.seconds = time.monotonic() - start
        return result

    def get(self, path: str, timeout: float = 10.0) -> Result:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            return Result(status=response.status, body=response.read())
        except (OSError, http.client.HTTPException) as error:
            return Result(error=str(error))
        finally:
            conn.close()

    def rss_kib(self) -> int | None:
        if self.pid is None:
            return None
        try:
            with open(f"/proc/{self.pid}/status", encoding="utf-8") as status:
                for line in status:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except OSError:
            return None
        return None

    def vram_mib(self) -> int | None:
        if self.pid is None:
            return None
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == str(self.pid) and parts[1].isdigit():
                return int(parts[1])
        # WSL often hides per-process usage; fall back to the device total.
        try:
            total = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True,
                                   text=True, timeout=20).stdout.split()
            return int(total[0]) if total else None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None


def chat(model: str, content: str, **extra: Any) -> dict[str, Any]:
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "chat_template_kwargs": NO_THINK}
    payload.update(extra)
    return payload


def content_of(result: Result) -> str | None:
    try:
        return json.loads(result.body)["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def stream_complete(result: Result) -> bool:
    return b"data: [DONE]" in result.body or b"event: message_stop" in result.body


class Traffic:
    def __init__(self, server: Server, model: str, stats: Stats, seed: int) -> None:
        self.server = server
        self.model = model
        self.stats = stats
        self.rng = random.Random(seed)
        self.system = ("You are a meticulous operations assistant. " * 120).strip()
        self.long_text = " ".join(f"Record {i}: sensor {i % 17} reported value {i * 7 % 101}."
                                  for i in range(700))

    def classify(self, scenario: str, result: Result, expect_stream: bool = False) -> None:
        self.stats.count(f"{scenario}.requests")
        if result.error in ("client-cancel", "client-disconnect"):
            self.stats.count(f"{scenario}.cancelled")
            return
        if result.error:
            self.stats.fail(f"{scenario}: transport error", result.error)
            return
        if result.status in (429, 503):
            self.stats.count(f"{scenario}.overload_{result.status}")
            time.sleep(0.2)  # back off: an overloaded or failed server must not be hammered
            try:
                json.loads(result.body)
            except ValueError:
                self.stats.fail(f"{scenario}: {result.status} without a JSON body", result.body[:200])
            return
        if result.status >= 500:
            self.stats.fail(f"{scenario}: HTTP {result.status}", result.body[:300])
            return
        if result.status != 200:
            self.stats.fail(f"{scenario}: HTTP {result.status}", result.body[:300])
            return
        if expect_stream and not stream_complete(result):
            self.stats.fail(f"{scenario}: stream ended without its terminal event",
                            result.body[-300:])
            return
        self.stats.count(f"{scenario}.ok")

    def one(self) -> None:
        r = self.rng.random()
        m = self.model
        if r < 0.12:
            res = self.server.request("/v1/chat/completions", chat(
                m, "Explain photosynthesis briefly.", max_tokens=self.rng.choice([32, 96, 200]),
                temperature=0))
            self.classify("plain", res)
        elif r < 0.30:
            res = self.server.request("/v1/chat/completions", chat(
                m, "Write a short poem about the sea.", max_tokens=self.rng.choice([64, 160]),
                stream=True, stream_options={"include_usage": True}), stream=True)
            self.classify("stream", res, True)
        elif r < 0.46:
            turns = self.rng.randint(1, 4)
            messages: list[dict[str, Any]] = [{"role": "system", "content": self.system}]
            for t in range(turns):
                messages.append({"role": "user", "content": f"Check the weather in city {t}."})
                if t < turns - 1:
                    messages.append({"role": "assistant", "content": None, "tool_calls": [{
                        "id": f"call_{t}", "type": "function", "function": {
                            "name": "get_weather", "arguments": json.dumps({"city": f"c{t}"})}}]})
                    messages.append({"role": "tool", "tool_call_id": f"call_{t}",
                                     "content": json.dumps({"temp_c": t})})
            res = self.server.request("/v1/chat/completions", {
                "model": m, "messages": messages, "tools": [WEATHER_TOOL], "max_tokens": 128,
                "stream": True, "chat_template_kwargs": NO_THINK}, stream=True)
            self.classify("agentic", res, True)
        elif r < 0.54:
            res = self.server.request("/v1/chat/completions", chat(
                m, "Invent a person.", max_tokens=96, response_format={
                    "type": "json_schema", "json_schema": {"name": "p", "schema": SCHEMA}}))
            self.classify("structured", res)
            text = content_of(res)
            if res.status == 200 and text is not None:
                try:
                    value = json.loads(text)
                    if not (isinstance(value, dict) and set(value) == {"name", "age"}):
                        self.stats.fail("structured: schema violated", text)
                except ValueError:
                    if json.loads(res.body)["choices"][0]["finish_reason"] != "length":
                        self.stats.fail("structured: output is not JSON", text)
        elif r < 0.62:
            res = self.server.request("/v1/chat/completions", chat(m, "Say hi.", max_tokens=1))
            self.classify("max_tokens_1", res)
        elif r < 0.68:
            res = self.server.request("/v1/chat/completions", chat(
                m, "Tell a one-line joke.", max_tokens=48, temperature=0.9,
                seed=self.rng.randint(0, 10 ** 6)))
            self.classify("seeded", res)
        elif r < 0.74:
            res = self.server.request("/v1/chat/completions", chat(
                m, self.long_text + "\nWhich sensor reported the largest value?", max_tokens=64,
                temperature=0))
            self.classify("long_context", res)
        elif r < 0.80:
            res = self.server.request("/v1/messages", {
                "model": m, "max_tokens": 96, "stream": True,
                "messages": [{"role": "user", "content": "Name three planets."}]}, stream=True)
            self.classify("anthropic_stream", res, True)
        elif r < 0.92:
            res = self.server.request("/v1/chat/completions", chat(
                m, "Count slowly from one to one hundred in words.", max_tokens=400, stream=True),
                stream=True, stop_after_chunks=self.rng.choice([0, 1, 2, 5, 20]))
            self.classify("cancel_mid_stream", res)
        else:
            res = self.server.request("/v1/chat/completions", chat(
                m, self.long_text + "\nSummarize.", max_tokens=64, stream=True), stream=True,
                disconnect_after_s=self.rng.choice([0.0, 0.001, 0.005, 0.02, 0.2]))
            self.classify("disconnect_early", res)


def canary(server: Server, model: str) -> tuple[str | None, Result]:
    res = server.request("/v1/chat/completions", chat(
        model, "List the first twelve prime numbers and add them up.", max_tokens=160,
        temperature=0))
    return content_of(res), res


def witness_reference(server: Server, model: str, stats: Stats) -> dict[str, str]:
    refs = {}
    for line in WITNESS_LINES:
        res = server.request("/v1/chat/completions", chat(
            model, f"Repeat exactly the following line and nothing else:\n{line}", max_tokens=32,
            temperature=0))
        text = content_of(res)
        if text is None:
            stats.fail("witness reference failed", res.body[:200])
            continue
        refs[line] = text
        if text.strip() != line:
            print(f"note: witness solo output differs from its line: {text!r}", file=sys.stderr)
    return refs


class Sampler(threading.Thread):
    def __init__(self, server: Server, stats: Stats, out, period: float) -> None:
        super().__init__(daemon=True)
        self.server, self.stats, self.out, self.period = server, stats, out, period
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()

    def run(self) -> None:
        start = time.time()
        while not self.stop_event.is_set():
            sample = {"kind": "sample", "t": round(time.time() - start, 1),
                      "rss_kib": self.server.rss_kib(), "vram_mib": self.server.vram_mib(),
                      "counts": dict(self.stats.counts)}
            self.samples.append(sample)
            self.out.write(json.dumps(sample) + "\n")
            self.out.flush()
            self.stop_event.wait(self.period)


def slope_per_hour(samples: list[dict[str, Any]], key: str, skip_fraction: float) -> float | None:
    points = [(s["t"], s[key]) for s in samples if s.get(key) is not None]
    points = points[int(len(points) * skip_fraction):]
    if len(points) < 3:
        return None
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx == 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in points) / sxx * 3600.0


def run_soak(args, server: Server, stats: Stats, out) -> dict[str, Any]:
    first_canary, res = canary(server, args.model)
    if first_canary is None:
        stats.fail("first canary failed", res.body[:200])
    sampler = Sampler(server, stats, out, args.sample_seconds)
    sampler.start()
    pause = threading.Event()
    idle = threading.Semaphore(0)
    deadline = time.monotonic() + args.seconds
    started = time.monotonic()
    stop = threading.Event()

    def worker(index: int) -> None:
        traffic = Traffic(server, args.model, stats, args.seed + index)
        while time.monotonic() < deadline and not stop.is_set():
            if pause.is_set():
                idle.release()
                while pause.is_set():
                    time.sleep(0.2)
                continue
            traffic.one()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.workers)]
    for t in threads:
        t.start()
    canaries = 0
    next_canary = time.monotonic() + args.canary_seconds
    next_health = time.monotonic() + 10
    unhealthy = 0
    engine_failed_after = None
    while time.monotonic() < deadline and not stop.is_set():
        time.sleep(1)
        if time.monotonic() >= next_health:
            next_health = time.monotonic() + 10
            unhealthy = unhealthy + 1 if server.get("/health").status != 200 else 0
            if unhealthy >= 2:
                # The engine latched failed (fail-stop): end the segment, record when.
                engine_failed_after = round(time.monotonic() - started, 1)
                stats.fail("engine unavailable during the soak",
                           {"seconds_into_soak": engine_failed_after})
                stop.set()
                break
        if time.monotonic() >= next_canary and time.monotonic() < deadline - 30:
            pause.set()
            for _ in threads:
                idle.acquire(timeout=600)
            text, res = canary(server, args.model)
            canaries += 1
            same = text == first_canary
            out.write(json.dumps({"kind": "canary", "t": time.time(), "identical": same}) + "\n")
            if text is None:
                stats.fail("solo canary failed during the soak", res.body[:200])
            elif not same:
                stats.note("solo canary changed during the soak", {"first": first_canary,
                                                                   "now": text})
            pause.clear()
            next_canary = time.monotonic() + args.canary_seconds
    for t in threads:
        t.join(timeout=900)
    sampler.stop_event.set()
    sampler.join(timeout=30)
    time.sleep(5)
    final = sampler.samples[-1] if sampler.samples else {}
    health = server.get("/health")
    if health.status != 200:
        stats.fail("health after soak", health.status or health.error)
    last_text, last_res = canary(server, args.model)
    if last_text is None:
        stats.fail("final solo canary failed", last_res.body[:200])
    elif last_text != first_canary:
        stats.note("final solo canary differs from the first", {"first": first_canary,
                                                                "last": last_text})
    return {"canaries": canaries + 1, "final_canary_identical": last_text == first_canary,
            "engine_failed_after_seconds": engine_failed_after,
            "soak_seconds": round(time.monotonic() - started, 1),
            "rss_kib_first": sampler.samples[0]["rss_kib"] if sampler.samples else None,
            "rss_kib_last": final.get("rss_kib"),
            "vram_mib_first": sampler.samples[0]["vram_mib"] if sampler.samples else None,
            "vram_mib_last": final.get("vram_mib"),
            "rss_slope_kib_per_hour_after_warmup": slope_per_hour(sampler.samples, "rss_kib", 0.3),
            "vram_slope_mib_per_hour_after_warmup": slope_per_hour(sampler.samples, "vram_mib",
                                                                   0.3)}


def run_chaos(args, server: Server, stats: Stats, out) -> dict[str, Any]:
    base_canary, res = canary(server, args.model)
    if base_canary is None:
        stats.fail("baseline canary failed", res.body[:200])
    refs = witness_reference(server, args.model, stats)
    storms: list[dict[str, Any]] = []

    def checkpoint(name: str) -> None:
        time.sleep(3)
        health = server.get("/health")
        # After the storm and 3 s idle every load gauge /health reports is back to
        # 0 (a server without gauges in /health reports none, which is recorded, not failed).
        gauges: dict[str, Any] = {}
        try:
            gauges = {k: v for k, v in json.loads(health.body or b"{}").items() if k != "status"}
        except ValueError:
            pass
        text, res = canary(server, args.model)
        entry = {"kind": "checkpoint", "after": name, "health": health.status, "gauges": gauges,
                 "canary_identical": text == base_canary, "rss_kib": server.rss_kib(),
                 "vram_mib": server.vram_mib(), "counts": dict(stats.counts)}
        if any(v != 0 for v in gauges.values()):
            stats.fail(f"load gauges not back to 0 after {name}", gauges)
        storms.append(entry)
        out.write(json.dumps(entry) + "\n")
        out.flush()
        if health.status != 200:
            stats.fail(f"health after {name}", health.status or health.error)
        if text is None:
            stats.fail(f"solo canary failed after {name}", res.body[:200])
        elif text != base_canary:
            stats.note(f"solo canary changed after {name}", {"base": base_canary, "now": text})

    def with_witnesses(name: str, storm) -> None:
        stop = threading.Event()

        def witness(index: int) -> None:
            line = WITNESS_LINES[index % len(WITNESS_LINES)]
            while not stop.is_set():
                res = server.request("/v1/chat/completions", chat(
                    args.model, f"Repeat exactly the following line and nothing else:\n{line}",
                    max_tokens=32, temperature=0))
                stats.count("witness.requests")
                if res.status in (429, 503):
                    stats.count("witness.overload")
                    time.sleep(0.2)
                    continue
                text = content_of(res)
                if text is None or (line in refs and text != refs[line]):
                    if text is not None and text.strip() == line:
                        stats.count("witness.batch_variation_exact_line")
                    else:
                        stats.fail(f"{name}: corrupted neighbour output",
                                   {"expected": refs.get(line), "got": text,
                                    "status": res.status, "error": res.error,
                                    "finished_at": time.time(), "seconds": res.seconds})
                else:
                    stats.count("witness.ok")

        threads = [threading.Thread(target=witness, args=(i,), daemon=True) for i in range(2)]
        for t in threads:
            t.start()
        try:
            storm()
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=600)
        checkpoint(name)

    long_prompt = " ".join(f"Entry {i}: value {i * 13 % 97}." for i in range(2500))

    def parallel(jobs, width: int) -> None:
        lock = threading.Lock()
        queue = list(jobs)

        def run() -> None:
            while True:
                with lock:
                    if not queue:
                        return
                    job = queue.pop()
                job()

        threads = [threading.Thread(target=run, daemon=True) for _ in range(width)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=1800)

    def disconnect_storm() -> None:
        jobs = []
        for _ in range(args.repeats):
            for delay in (0.0, 0.001, 0.005, 0.02):
                jobs.append(lambda d=delay: stats.count("storm.disconnect_early") or server.request(
                    "/v1/chat/completions", chat(args.model, "Describe a forest.", max_tokens=200,
                                                 stream=True), stream=True, disconnect_after_s=d))
            for chunks in (1, 3, 10):
                jobs.append(lambda c=chunks: stats.count("storm.cancel_mid_stream") or
                            server.request("/v1/chat/completions", chat(
                                args.model, "Describe a desert.", max_tokens=200, stream=True),
                                stream=True, stop_after_chunks=c))
        parallel(jobs, 8)

    def prefill_cancel_storm() -> None:
        jobs = [lambda d=d: stats.count("storm.prefill_cancel") or server.request(
            "/v1/chat/completions", chat(args.model, long_prompt + f"\nQuestion {d}: sum?",
                                         max_tokens=64, stream=True), stream=True,
            disconnect_after_s=d) for d in [0.05, 0.1, 0.2, 0.3, 0.5] * max(1, args.repeats // 10)]
        parallel(jobs, 6)

    def saturation() -> None:
        results: list[Result] = []
        lock = threading.Lock()

        def fire(i: int) -> None:
            res = server.request("/v1/chat/completions", chat(
                args.model, f"Write a haiku number {i}.", max_tokens=64, stream=(i % 2 == 0)),
                stream=(i % 2 == 0), timeout=900)
            with lock:
                results.append(res)

        threads = [threading.Thread(target=fire, args=(i,), daemon=True)
                   for i in range(args.saturation)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=900)
        hung = sum(1 for t in threads if t.is_alive())
        statuses: dict[str, int] = {}
        for res in results:
            key = str(res.status) if not res.error else "transport-error"
            statuses[key] = statuses.get(key, 0) + 1
            if res.error or (res.status >= 500 and res.status != 503) or res.status not in (
                    200, 429, 503):
                stats.fail("saturation: dishonest answer", {"status": res.status,
                                                             "error": res.error,
                                                             "body": res.body[:200]})
            elif res.status in (429, 503):
                try:
                    json.loads(res.body)
                except ValueError:
                    stats.fail("saturation: overload without a JSON body", res.body[:200])
            elif res.status == 200 and res.body.startswith((b"data:", b":")) and \
                    not stream_complete(res):
                stats.fail("saturation: stream without terminal event", res.body[-200:])
        if hung:
            stats.fail("saturation: hung requests", hung)
        stats.counts["saturation.statuses"] = statuses  # type: ignore[assignment]

    # A ~10k-token shared system prompt, many requests over it with ~30 % cancelled at a
    # random moment (before admission, during the prefix lookup and materialization, mid-prefill,
    # mid-decode) and more clients than the queue holds. Afterwards the same greedy request must
    # repeat its pre-storm answer exactly.
    shared_system = " ".join(f"Rule {i}: item {i} maps to code {i * 7 % 101}." for i in range(900))

    def shared_body(question: str, **extra: Any) -> dict[str, Any]:
        body = chat(args.model, question, **extra)
        body["messages"].insert(0, {"role": "system", "content": shared_system})
        return body

    shared_reference: dict[str, str | None] = {}

    def shared_prefix_storm() -> None:
        probe = shared_body("Which code does item 42 map to? Answer with the number only.",
                            max_tokens=16, temperature=0)
        if "text" not in shared_reference:
            shared_reference["text"] = content_of(server.request("/v1/chat/completions", probe))
        rng = random.Random(args.seed)
        jobs = []
        for i in range(args.shared_requests):
            cancel = rng.random() < 0.3
            delay = rng.choice([0.0, 0.001, 0.005, 0.02, 0.05, 0.1, 0.3, 0.8, 2.0])
            stream = cancel or rng.random() < 0.5
            body = shared_body(f"Question {i}: which code does item {i % 900} map to?",
                               max_tokens=rng.choice([8, 32, 96]), temperature=0, stream=stream)

            def job(body=body, cancel=cancel, delay=delay, stream=stream) -> None:
                stats.count("storm.shared_prefix")
                if cancel:
                    stats.count("storm.shared_prefix_cancel")
                res = server.request("/v1/chat/completions", body, stream=stream,
                                     disconnect_after_s=delay if cancel else None)
                if not cancel and res.status not in (200, 429, 503):
                    stats.fail("shared prefix storm: dishonest answer",
                               {"status": res.status, "error": res.error, "body": res.body[:200]})
                elif not cancel and res.status in (429, 503):
                    stats.count("storm.shared_prefix_overload")
            jobs.append(job)
        parallel(jobs, 12)
        time.sleep(2)
        again = content_of(server.request("/v1/chat/completions", probe))
        if again != shared_reference["text"]:
            stats.fail("shared prefix storm: repeated request differs",
                       {"before": shared_reference["text"], "after": again})

    # Cancellation of waiting requests (meant for --max-concurrency 1). A heavy request
    # runs; four more heavy requests queue behind it and their clients leave while they wait. None
    # of the four may start (the request log, --request-log, shows them cancelled with no prefill
    # time and no output), the first finishes, and the gauges return to 0. Repeated
    # --pending-rounds times.
    def pending_cancel_storm() -> None:
        # ~9k tokens (Qwen tokenizer): heavy enough to queue behind, and inside a 16K context
        # with the 400-token budget (1500 clauses, ~16.5k tokens, exceeded it).
        heavy = " ".join(f"Clause {i}: section {i * 11 % 89} applies." for i in range(800))
        for round_index in range(args.pending_rounds):
            first: dict[str, Result] = {}

            def run_first() -> None:
                first["r"] = server.request("/v1/chat/completions", chat(
                    args.model, heavy + f"\nRound {round_index}: summarize the clauses.",
                    max_tokens=400, stream=True, temperature=0), stream=True)

            runner = threading.Thread(target=run_first, daemon=True)
            runner.start()
            time.sleep(1.0)  # the first request is admitted and prefilling
            waiters = [threading.Thread(target=lambda i=i: stats.count("storm.pending_cancel") or
                                        server.request("/v1/chat/completions", chat(
                                            args.model, heavy + f"\nRound {round_index} waiter {i}.",
                                            max_tokens=400, stream=True), stream=True,
                                            disconnect_after_s=0.3), daemon=True)
                       for i in range(4)]
            for w in waiters:
                w.start()
            for w in waiters:
                w.join(timeout=60)
            runner.join(timeout=600)
            res = first.get("r")
            if res is None or res.status != 200 or not stream_complete(res):
                stats.fail("pending cancel: the running request did not finish",
                           {"status": None if res is None else res.status,
                            "error": None if res is None else res.error})
            else:
                stats.count("storm.pending_first_ok")
        if not args.request_log:
            return
        time.sleep(1.0)
        started, cancelled = 0, 0
        with open(args.request_log, encoding="utf-8") as log:
            for line in log:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("event") != "request_done":
                    continue
                result = record.get("result", {})
                if result.get("finish_reason") != "cancelled":
                    continue
                timings = record.get("timings_seconds", {})
                if result.get("prompt_tokens", 0) < 1000:
                    continue  # other storms' short requests
                cancelled += 1
                if result.get("completion_tokens", 0) != 0 or (timings.get("prefill") or 0) > 0:
                    started += 1
        stats.count("storm.pending_cancelled_logged", cancelled)
        if started:
            stats.fail("pending cancel: cancelled waiting requests were started", started)

    kinds = {"disconnect": ("disconnect storm", disconnect_storm),
             "prefill": ("prefill cancel storm", prefill_cancel_storm),
             "saturation": ("saturation", saturation),
             "shared": ("shared prefix cancel storm", shared_prefix_storm),
             "pending": ("pending cancel storm", pending_cancel_storm)}
    for _ in range(args.iterations):
        for key in args.storms.split(","):
            name, storm = kinds[key]
            with_witnesses(name, storm)
    return {"checkpoints": storms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["soak", "chaos"])
    parser.add_argument("--base-url", default="http://127.0.0.1:18931")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--canary-seconds", type=float, default=180)
    parser.add_argument("--sample-seconds", type=float, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--saturation", type=int, default=64)
    parser.add_argument("--storms", default="disconnect,prefill,saturation")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--shared-requests", type=int, default=500)
    parser.add_argument("--pending-rounds", type=int, default=5)
    parser.add_argument("--request-log", help="the server's --request-log-jsonl file (pending storm)")
    args = parser.parse_args()
    server = Server(args.base_url, args.pid)
    stats = Stats()
    with open(args.out, "a", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps({"kind": "start", "mode": args.mode, "t": time.time(),
                              "args": vars(args)}) + "\n")
        summary = (run_soak if args.mode == "soak" else run_chaos)(args, server, stats, out)
        summary.update({"mode": args.mode, "counts": stats.counts,
                        "failures": stats.failures[:50], "failure_count": len(stats.failures),
                        "notes": stats.notes[:50]})
        out.write(json.dumps({"kind": "summary", **summary}, default=str) + "\n")
    print(json.dumps(summary, indent=1, default=str))
    sys.exit(1 if stats.failures else 0)


if __name__ == "__main__":
    main()
