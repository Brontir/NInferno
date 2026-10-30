"""Counted parameters (the plan's model.params) of two ninferno-serve builds against the weights'
element count (weights_bytes x 8 / the row's bits per weight from the GGUF's tensor table)."""
import json, subprocess, sys

CASES = [("unsloth/gemma-4-12b-it-GGUF", "Q8_0"), ("unsloth/gemma-4-31B-it-GGUF", "Q4_K_M"),
         ("unsloth/gemma-4-26B-A4B-it-GGUF", "UD-Q4_K_XL"), ("unsloth/gemma-4-E4B-it-GGUF", "Q8_0"),
         ("unsloth/Qwen3.5-9B-GGUF", "Q8_0"), ("unsloth/Qwen3.8-27B-GGUF", "Q8_0"),
         ("unsloth/Qwen3.6-35B-A3B-GGUF", "Q8_0")]


def read(serve, repo, quant):
    out = subprocess.run([serve, repo, "--quant", quant, "--plan", "--json", "--no-load",
                          "--plan-device", "RTX 4090"], capture_output=True, text=True).stdout
    d = json.loads(out)
    m = d.get("model") or {}
    rows = [r for r in d.get("quants", []) if r.get("quant") == quant] or d.get("quants", [])[:1]
    bpw = (rows[0].get("bits_per_weight") if rows else 0) or 0
    return m.get("params", 0), (m.get("weights_bytes", 0) * 8 / bpw if bpw else 0)


for repo, quant in CASES:
    for name, serve in (("master", sys.argv[1]), ("branch", sys.argv[2])):
        try:
            params, elements = read(serve, repo, quant)
            ratio = params / elements if elements else 0
            print(f"{repo:36s} {quant:11s} {name:6s} params {params/1e9:7.3f} B  tensors {elements/1e9:7.3f} B  ratio {ratio:.3f}", flush=True)
        except Exception as e:
            print(f"{repo} {quant} {name}: {e}", flush=True)
