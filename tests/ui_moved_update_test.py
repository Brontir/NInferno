"""ninferno-ui's update of a moved hub model (DECISIONS 29), end to end against a stand-in hub and a
stand-in ninferno-serve: the user's "Download new" prepares the new commit beside the old
conversion (--refetch --keep-replaced); a model serving the old conversion restarts once on the
new one, only when it has no request in flight; then the old conversion goes. A new conversion
that does not start sends the model back to the old one once, and nothing starts again after.

usage: ui_moved_update_test.py NINFERNO_UI CONVERT_NATIVE_TESTS_DIR
"""

import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

OLD = "c0ffee0123456789abcdef0123456789abcdef01"
NEW = "e" * 40
NEW_BAD = "b" * 40
NAME = "model-Q4_K_M.qwen3_5_bf16.ninferno"

# The stand-in ninferno-serve. A prepare (--prepare-only --refetch --keep-replaced) writes the
# hub's current commit's conversion and names it; a serve names its artifact, refuses one under a
# commit FAKE_BAD names, else answers /health with the requests FAKE_ACTIVE holds.
FAKE_SERVE = r'''#!/usr/bin/env python3
import http.server, json, os, sys, time, urllib.request
args = sys.argv[1:]
model = args[0]
log = open(os.environ["FAKE_LOG"], "a")
if "--prepare-only" in args:
    assert "--refetch" in args and "--keep-replaced" in args, args
    quant = args[args.index("--quant") + 1]
    recipe = args[args.index("--recipe") + 1]
    org, name = model.split("@")[0].split("/")
    meta = json.load(urllib.request.urlopen(
        os.environ["HF_ENDPOINT"] + "/api/models/%s/%s/revision/main" % (org, name)))
    repo = os.path.join(os.environ["FAKE_CACHE"], "models--%s--%s" % (org, name))
    out = os.path.join(repo, "ninferno", meta["sha"])
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "model-%s.%s.ninferno" % (quant, recipe))
    time.sleep(2)
    open(path, "wb").write(b"new conversion")
    open(os.path.join(repo, "refs", "main"), "w").write(meta["sha"])
    log.write("prepare %s\n" % model); log.flush()
    print("converted artifact %s (0.0 GiB)" % path, flush=True)
    sys.exit(0)
port = int(args[args.index("--port") + 1])
log.write("serve %s\n" % model); log.flush()
print("cached artifact %s (0.0 GiB)" % model, flush=True)
if os.environ.get("FAKE_BAD") and os.environ["FAKE_BAD"] in model:
    print("error: this conversion does not start", flush=True)
    sys.exit(3)
class Health(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        active = int(open(os.environ["FAKE_ACTIVE"]).read() or 0)
        body = json.dumps({"status": "ok", "active_requests": active}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
http.server.HTTPServer(("127.0.0.1", port), Health).serve_forever()
'''

failures = []


def case(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else "  " + str(detail)))
    if not ok:
        failures.append(name)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    if os.name == "nt":
        print("skipped: the stand-in ninferno-serve is a POSIX script")
        return 77
    ui, convert_tests = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(convert_tests))
    import fetch_server

    hub = fetch_server.Hub()
    hub.add_repo("org/m-GGUF", {"README.md": b"m"}, commit=NEW)
    hub.add_repo("org/f-GGUF", {"README.md": b"f"}, commit=NEW_BAD)
    server, base = fetch_server.start(hub)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache = tmp / "hub"
        olds = {}
        for repo in ("m", "f"):
            repo_dir = cache / f"models--org--{repo}-GGUF"
            (repo_dir / "refs").mkdir(parents=True)
            (repo_dir / "refs" / "main").write_text(OLD)
            (repo_dir / "ninferno" / OLD).mkdir(parents=True)
            olds[repo] = repo_dir / "ninferno" / OLD / NAME
            olds[repo].write_bytes(b"old conversion")
        fake = tmp / "fake-serve"
        fake.write_text(FAKE_SERVE)
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        log, active = tmp / "serve.log", tmp / "active"
        log.write_text("")
        active.write_text("1")  # busy: no restart yet
        port = free_port()
        env = {**os.environ, "HF_ENDPOINT": base, "FAKE_LOG": str(log), "FAKE_ACTIVE": str(active),
               "FAKE_CACHE": str(cache), "FAKE_BAD": NEW_BAD}
        process = subprocess.Popen([str(ui), "--serve", str(fake), "--hub-cache", str(cache),
                                    "--port", str(port), "--no-browser",
                                    "--catalog-url", "http://127.0.0.1:9/catalog.json"],
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        url = f"http://127.0.0.1:{port}"

        def call(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(url + path, data=data,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as res:
                    return res.status, json.loads(res.read() or b"{}")
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read() or b"{}")

        def wait(predicate, seconds=40):
            deadline = time.time() + seconds
            while time.time() < deadline:
                try:
                    value = predicate()
                    if value:
                        return value
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(0.3)
            return None

        def update(repo):
            for u in call("/api/moved")[1].get("updates", []):
                if u["repo"] == f"org/{repo}-GGUF":
                    return u
            return None

        def serves():
            return [line.split(" ", 1)[1] for line in log.read_text().splitlines()
                    if line.startswith("serve ")]

        try:
            case("ninferno-ui answers", wait(lambda: call("/api/state")[0] == 200))
            for repo in ("m", "f"):
                status, body = call("/api/launch", {"model": str(olds[repo]), "gpu": -1})
                case(f"{repo}: the model on its old conversion starts", status == 200, body)
            moved = call("/api/moved")[1]
            listed = {m["repo"]: m for m in moved.get("models", [])}
            case("both moved models are listed, with their conversions",
                 set(listed) == {"org/m-GGUF", "org/f-GGUF"}
                 and listed["org/m-GGUF"]["artifacts"] == [str(olds["m"])], moved)

            # The good update: downloaded beside, then held while the model is busy.
            status, body = call("/api/moved", {"repo": "org/m-GGUF", "ref": "main", "commit": OLD,
                                               "action": "refetch"})
            case("the update starts on the user's answer", status == 200 and len(body["ids"]) == 1, body)
            status, body = call("/api/moved", {"repo": "org/m-GGUF", "ref": "main", "commit": OLD,
                                               "action": "refetch"})
            case("a second answer while it runs is refused", status == 400, body)
            new = cache / "models--org--m-GGUF" / "ninferno" / NEW / NAME
            case("held while the model has a request in flight",
                 wait(lambda: (update("m") or {}).get("state") == "switching")
                 and new.exists() and olds["m"].exists())
            time.sleep(3)
            case("no restart while busy", serves().count(str(new)) == 0, serves())
            active.write_text("0")
            case("restarted once on the new conversion when idle, the old one removed",
                 wait(lambda: (update("m") or {}).get("state") == "done")
                 and serves().count(str(new)) == 1 and not olds["m"].exists() and new.exists(),
                 [update("m"), serves()])

            # The failing update: the new conversion does not start; back on the old one, once.
            status, body = call("/api/moved", {"repo": "org/f-GGUF", "ref": "main", "commit": OLD,
                                               "action": "refetch"})
            case("the second model's update starts", status == 200, body)
            bad = cache / "models--org--f-GGUF" / "ninferno" / NEW_BAD / NAME
            failed = wait(lambda: (u := update("f")) and u["state"] == "failed" and u)
            case("a new conversion that does not start ends the update as failed, said",
                 bool(failed) and "did not start on the new conversion" in failed.get("error", ""),
                 failed)
            time.sleep(4)
            starts = serves()
            case("back on the old conversion once, nothing started again",
                 starts.count(str(bad)) == 1 and starts.count(str(olds["f"])) == 2
                 and olds["f"].exists(), starts)
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            server.shutdown()
            if failures:
                print(process.stdout.read().decode(errors="replace")[-4000:])
    print("FAILED: " + ", ".join(failures) if failures else "all passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
