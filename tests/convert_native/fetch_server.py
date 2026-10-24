"""A local stand-in for a model hub, for the fetcher's tests.

Serves Hugging Face's API and resolve/302 flow (X-Linked-Etag names the LFS sha256; the CDN
URL is signed and carries no credentials), a ModelScope-shaped API, and plain files, all with
HTTP ranges. Blobs can misbehave on purpose: stall mid-transfer, drop the connection, serve
corrupted bytes, answer 403 once (an expired signed URL), or ignore ranges. /stats reports the
bytes served per blob and every request's path, range and whether it carried credentials.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

TOKEN = "hf_test_secret_token_0123456789"


def git_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Hub:
    def __init__(self):
        self.repos = {}      # repo -> {"commit", "gated", "files": {path: bytes}, "lfs": set}
        self.blobs = {}      # sha256 -> bytes
        self.behavior = {}   # sha256 -> set of behaviors
        self.served = {}     # sha256 -> bytes served
        self.requests = []   # (path, range, authorized)
        self.fired = set()   # one-shot behaviors already used
        self.lock = threading.Lock()

    def add_repo(self, repo, files, commit="c0ffee0123456789abcdef0123456789abcdef01", gated=False,
                 lfs_threshold=1024, xet_small=False):
        """xet_small: git-stored files are redirected to the CDN too, their X-Linked-Etag the git
        blob sha1 (as Xet-backed repositories answer)."""
        entry = {"commit": commit, "gated": gated, "files": dict(files), "lfs": set(),
                 "xet_small": xet_small}
        for path, data in files.items():
            if len(data) >= lfs_threshold:
                sha = hashlib.sha256(data).hexdigest()
                entry["lfs"].add(path)
                self.blobs[sha] = data
            elif xet_small:
                self.blobs[git_sha1(data)] = data
        self.repos[repo] = entry

    def sha(self, data):
        return hashlib.sha256(data).hexdigest()


def make_handler(hub: Hub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def authorized(self):
            return self.headers.get("Authorization") == f"Bearer {TOKEN}"

        def record(self, path):
            with hub.lock:
                hub.requests.append((path, self.headers.get("Range"), "Authorization" in self.headers))

        def reply(self, status, body=b"", headers=None, head=False):
            self.send_response(status)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head and body:
                self.wfile.write(body)

        def serve_bytes(self, data, head, key=None, behavior=()):
            """Bytes with Range support and optional misbehavior."""
            size = len(data)
            start, end = 0, size - 1
            ranged = False
            spec = self.headers.get("Range")
            if spec and "no_range" not in behavior and spec.startswith("bytes="):
                first, _, last = spec[6:].partition("-")
                start = int(first)
                end = int(last) if last else size - 1
                ranged = True
            if start >= size:
                return self.reply(416, headers={"Content-Range": f"bytes */{size}"}, head=head)
            body = data[start:end + 1]
            if "corrupt" in behavior and body:
                body = bytes([body[0] ^ 0xFF]) + body[1:]
            self.send_response(206 if ranged else 200)
            self.send_header("Accept-Ranges", "bytes")
            if ranged:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", '"cdn-etag-not-the-sha"')
            self.end_headers()
            if head:
                return
            once = None
            for name in ("stall_once", "drop_once"):
                if name in behavior:
                    with hub.lock:
                        if (key, name) not in hub.fired and len(body) > (1 << 20):
                            hub.fired.add((key, name))
                            once = name
            if once:
                self.wfile.write(body[: 1 << 20])
                self.wfile.flush()
                with hub.lock:
                    hub.served[key] = hub.served.get(key, 0) + (1 << 20)
                if once == "stall_once":
                    time.sleep(30)
                self.close_connection = True
                return
            if "slow" in behavior:
                # Long enough for a second downloader to arrive while the first still writes.
                for at in range(0, len(body), 1 << 18):
                    self.wfile.write(body[at:at + (1 << 18)])
                    self.wfile.flush()
                    time.sleep(0.05)
            else:
                self.wfile.write(body)
            if key is not None:
                with hub.lock:
                    hub.served[key] = hub.served.get(key, 0) + len(body)

        def handle_any(self, head):
            url = urlparse(self.path)
            path = unquote(url.path)
            self.record(path)
            parts = path.strip("/").split("/")
            if path == "/stats":
                with hub.lock:
                    body = json.dumps({"served": hub.served, "requests": hub.requests}).encode()
                return self.reply(200, body, {"Content-Type": "application/json"}, head)
            if path == "/reset":
                with hub.lock:
                    hub.served.clear()
                    hub.requests.clear()
                return self.reply(200, b"{}", head=head)
            # Hugging Face: /api/models/<org>/<name>/revision/<rev>
            if len(parts) >= 6 and parts[0] == "api" and parts[1] == "models" and parts[4] == "revision":
                repo = hub.repos.get(f"{parts[2]}/{parts[3]}")
                if repo is None:
                    return self.reply(404, b'{"error":"Repository not found"}', head=head)
                if repo["gated"] and not self.authorized():
                    return self.reply(401, b'{"error":"Access to model is restricted"}', head=head)
                siblings = []
                for name, data in repo["files"].items():
                    item = {"rfilename": name, "size": len(data), "blobId": git_sha1(data)}
                    if name in repo["lfs"]:
                        item["lfs"] = {"sha256": hub.sha(data), "size": len(data), "pointerSize": 134}
                    if repo.get("no_sizes"):  # a hub that does not say how large a file is
                        item.pop("size")
                        item.get("lfs", {}).pop("size", None)
                    siblings.append(item)
                # A commit the repository had before (its "history", as a moved branch leaves it)
                # still lists as itself; any other revision lists as the current commit.
                revision = "/".join(parts[5:])
                sha = revision if revision in repo.get("history", ()) else repo["commit"]
                body = json.dumps({"id": f"{parts[2]}/{parts[3]}", "sha": sha,
                                   "siblings": siblings}).encode()
                return self.reply(200, body, {"Content-Type": "application/json"}, head)
            # Hugging Face: /<org>/<name>/resolve/<commit>/<path>
            if len(parts) >= 5 and parts[2] == "resolve":
                repo = hub.repos.get(f"{parts[0]}/{parts[1]}")
                name = "/".join(parts[4:])
                if repo is None or name not in repo["files"]:
                    return self.reply(404, head=head)
                if repo["gated"] and not self.authorized():
                    return self.reply(401, head=head)
                if repo.get("busy_once"):  # a hub too busy to answer, once per file
                    with hub.lock:
                        first = ("busy", path) not in hub.fired
                        hub.fired.add(("busy", path))
                    if first:
                        return self.reply(503, head=head)
                data = repo["files"][name]
                if name in repo["lfs"]:
                    sha = hub.sha(data)
                    host = self.headers.get("Host")
                    linked = {"Location": f"http://{host}/cdn/{sha}?X-Signature=abc&Expires=9999",
                              "X-Linked-Etag": f'"{sha}"', "X-Linked-Size": str(len(data)),
                              "ETag": f'"{sha}"', "X-Repo-Commit": repo["commit"]}
                    if repo.get("no_sizes"):
                        linked.pop("X-Linked-Size")
                    return self.reply(302, headers=linked, head=head)
                if repo["xet_small"]:
                    oid = git_sha1(data)
                    host = self.headers.get("Host")
                    return self.reply(302, headers={
                        "Location": f"http://{host}/cdn/{oid}?X-Signature=abc",
                        "X-Linked-Etag": f'"{oid}"', "X-Linked-Size": str(len(data))}, head=head)
                return self._small(data, head)
            # CDN: /cdn/<sha256>
            if len(parts) == 2 and parts[0] == "cdn":
                sha = parts[1]
                if "Authorization" in self.headers:
                    with hub.lock:
                        hub.requests.append(("CREDENTIALS SENT TO CDN", None, True))
                behavior = hub.behavior.get(sha, set())
                if "expire_once" in behavior:
                    with hub.lock:
                        fire = (sha, "expire_once") not in hub.fired
                        hub.fired.add((sha, "expire_once"))
                    if fire:
                        return self.reply(403, b"expired", head=head)
                if sha not in hub.blobs:
                    return self.reply(404, head=head)
                return self.serve_bytes(hub.blobs[sha], head, key=sha, behavior=behavior)
            # ModelScope: /api/v1/models/<org>/<name>/repo/files
            if len(parts) >= 7 and parts[:3] == ["api", "v1", "models"] and parts[5:7] == ["repo", "files"]:
                repo = hub.repos.get(f"{parts[3]}/{parts[4]}")
                if repo is None:
                    return self.reply(404, head=head)
                revision = parse_qs(url.query).get("Revision", ["master"])[0]
                files = [{"Name": name.split("/")[-1], "Path": name, "Type": "blob", "Size": len(data),
                          "Sha256": hub.sha(data), "Revision": repo.get("file_revision", revision)}
                         for name, data in repo["files"].items()]
                body = json.dumps({"Code": 200, "Data": {"Files": files}}).encode()
                return self.reply(200, body, {"Content-Type": "application/json"}, head)
            # ModelScope: /models/<org>/<name>/resolve/<rev>/<path>
            if len(parts) >= 6 and parts[0] == "models" and parts[3] == "resolve":
                repo = hub.repos.get(f"{parts[1]}/{parts[2]}")
                name = "/".join(parts[5:])
                if repo is None or name not in repo["files"]:
                    return self.reply(404, head=head)
                data = repo["files"][name]
                return self.serve_bytes(data, head, key=hub.sha(data))
            # Plain files: /files/<name>
            if len(parts) == 2 and parts[0] == "files":
                data = hub.repos.get("files", {}).get("files", {}).get(parts[1])
                if data is None:
                    return self.reply(404, head=head)
                return self.serve_bytes(data, head, key=hub.sha(data))
            return self.reply(404, head=head)

        def _small(self, data, head):
            return self.reply(200, data, {"ETag": f'"{git_sha1(data)}"'}, head)

        def do_GET(self):
            self.handle_any(False)

        def do_HEAD(self):
            self.handle_any(True)

    return Handler


class Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass  # clients abandoning a transfer (stalls, ignored ranges) are part of the tests


def start(hub: Hub):
    server = Server(("127.0.0.1", 0), make_handler(hub))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"
