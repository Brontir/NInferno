"""Image cache key probe against a running ninferno-serve --vision.

Image A + text T, then image B of the same size + T, then A again, a third image and B again.
B must not be served from A's image blocks: its answer names B's colour and its cached prompt
tokens stop before the image. Images are generated here (solid colours with a small marker,
standard library PNG writer); nothing is downloaded. Exit 1 on any wrong answer.
"""
import argparse
import base64
import json
import struct
import sys
import urllib.error
import urllib.request
import zlib


def png(width, height, rgb, marker):
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            if marker[0] <= x < marker[0] + 24 and marker[1] <= y < marker[1] + 24:
                row += bytes((255, 255, 255))
            else:
                row += bytes(rgb)
        rows.append(bytes(row))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


def ask(base, model, image, question):
    body = {"model": model, "max_tokens": 16, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()}},
                {"type": "text", "text": question}]}]}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return None, e.code, e.read()[:300]
    u = out.get("usage", {})
    return (out["choices"][0]["message"]["content"], u.get("prompt_tokens"),
            (u.get("prompt_tokens_details") or {}).get("cached_tokens"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    question = "What is the main colour of this picture? Answer with one word."
    red = png(448, 448, (220, 30, 30), (40, 40))
    blue = png(448, 448, (30, 40, 220), (40, 40))
    green = png(448, 448, (30, 200, 40), (300, 300))
    bad = 0
    plan = [("A red", red, "red"), ("B blue", blue, "blue"), ("A red again", red, "red"),
            ("C green, marker moved", green, "green"), ("B blue again", blue, "blue")]
    for label, image, expected in plan:
        text, prompt_tokens, cached = ask(a.base_url, a.model, image, question)
        ok = text is not None and expected in text.lower()
        print(json.dumps({"request": label, "answer": text, "prompt_tokens": prompt_tokens,
                          "cached_tokens": cached, "ok": ok}))
        bad += 0 if ok else 1
    print("image cache key probe: bad =", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
