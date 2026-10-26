import json, subprocess, sys
A, B = sys.argv[1], sys.argv[2]
DEVS = ["RTX 4060", "RTX 4070 Ti", "RTX 4090", "RTX 5090"]
REPOS = ["unsloth/Qwen3.5-9B-MTP-GGUF", "unsloth/Qwen3.5-9B-GGUF", "bartowski/Qwen_Qwen3.5-9B-GGUF",
         "unsloth/gemma-4-12b-it-GGUF", "unsloth/Qwen3.8-27B-GGUF", "unsloth/Qwen3.6-35B-A3B-GGUF", "unsloth/gpt-oss-20b-GGUF"]
def plan(binary, repo, dev):
    out = subprocess.run([binary, repo, "--max-context", "32768", "--plan", "--json", "--no-load", "--plan-device", dev],
                         capture_output=True, text=True, timeout=300).stdout
    try: return json.loads(out)
    except Exception: return None
def rows(d):
    out = {}
    for r in d.get("quants", []):
        s = (r.get("plan") or {}).get("speculation") or {}
        out[r["quant"]] = (s.get("served"), s.get("basis"), round(r.get("bits_per_weight") or 0, 2), r.get("below_floor"))
    return out
spec_changes = bits_changes = 0
for dev in DEVS:
    for repo in REPOS:
        a, b = plan(A, repo, dev), plan(B, repo, dev)
        if a is None or b is None:
            print("NOPLAN", dev, repo, a is None, b is None); continue
        pa, pb = (a.get("pick") or {}).get("quant"), (b.get("pick") or {}).get("quant")
        ra, rb = rows(a), rows(b)
        lines = []
        if pa != pb: lines.append(f"  pick {pa} -> {pb}")
        for q in sorted(set(ra) | set(rb)):
            x, y = ra.get(q), rb.get(q)
            if x is None or y is None or x[:2] != y[:2]:
                spec_changes += 1
                lines.append(f"  {q}: spec {x and x[:2]} -> {y and y[:2]}")
            elif x[2:] != y[2:]:
                bits_changes += 1
        if lines: print(dev, repo); print("\n".join(lines))
print(f"spec/basis row changes {spec_changes}, rows with only bits/floor changed {bits_changes}")
