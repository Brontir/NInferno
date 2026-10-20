"""Gemma 4 checkpoints as published: gemma4_published quantizes nothing.

    python synthetic_gemma4_published.py NINFERNO_CONVERT [REPOSITORY]

synthetic_gemma4's text checkpoints converted with gemma4_published:

  dense-ct   compressed-tensors NVFP4 attention and MLP projections: every projection is an
             nvfp4 object imported as published (AllowA4 with its activation divisor), the
             embedding table and the untied head stay bf16, no q8/q6 object is written;
  moe-ct     the same with NVFP4 expert banks (tied table): banks stay nvfp4, the router and the
             table and head bf16 (the head is unscaled, the table carries sqrt(hidden), so two
             objects); its dense MLP gate/up have 192 rows, not whole 128-row NVFP4 scale tiles:
             kept nvfp4 all the same (the layout completes a last tile), as in dense_published;
  dense-bf16 (modelopt layout with BF16 attention and MLP): every projection bf16.

The official recipes (gemma4_31b_nvfp4 and gemma4_26b_a4b_nvfp4) write the table as q8 and the
head as q6 and requantize the BF16 projections to q8 - a quantization of our own that a published
checkpoint does not carry. Exits 77 (skipped) without numpy/torch.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import numpy  # noqa: F401
    import torch  # noqa: F401
except ImportError:
    sys.exit(77)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic_gemma4 as sg  # noqa: E402
import synthetic_formats as sf  # noqa: E402


def formats(value, objects):
    out = {}
    for name, binding in value["bindings"].items():
        ids = [binding["object"]] if "object" in binding else [p["object"] for p in binding["parts"]]
        out[name] = {objects[i]["format"] for i in ids}
    return out


def assistant_cases(work: Path, native: Path, repository: Path) -> list[str]:
    """Google's assistant (BF16): without --recipe the converter chooses gemma4_assistant_published
    and every object stays bf16 (norms and layer scalars keep their own form, nothing is q8);
    NInferno's Q8 of it is written only when gemma4_assistant_q8 is named."""
    errors = []
    root = work / "assistant"
    sg.assistant_checkpoint(root)
    template = work / "template.jinja"
    template.write_text("{% for m in messages %}{{ m['content'] }}{% endfor %}\n", encoding="utf-8")
    runs = {"assistant-default": [], "assistant-q8": ["--recipe=gemma4_assistant_q8"]}
    for name, extra in runs.items():
        out = work / f"{name}.ninferno"
        result = subprocess.run([str(native), f"--model={root}", f"--resource=chat_template.jinja={template}",
                                 *extra, f"--out={out}"], cwd=repository, capture_output=True, text=True)
        if result.returncode:
            errors.append(f"{name}: conversion failed: {result.stderr.strip()[-600:]}")
            continue
        value, objects, _, _ = sf.directory(out)
        kinds = formats(value, objects)
        projections = {n: f for n, f in kinds.items()
                       if n.split("/")[-1] in ("query", "output", "gate", "up", "down", "gate_up",
                                                "pre_projection", "post_projection", "output_head")}
        if name == "assistant-default":
            if "recipe gemma4_assistant_published (chosen" not in result.stdout:
                errors.append(f"{name}: the chosen recipe is not gemma4_assistant_published\n{result.stdout[-800:]}")
            quantized = sorted({f for fs in kinds.values() for f in fs if f.startswith(("q8", "q6", "q4"))})
            if quantized or not projections or any(f != {"bf16"} for f in projections.values()):
                errors.append(f"{name}: not as published: quantized {quantized}, projections {projections}")
        elif not projections or any(f != {"q8_g32_fp16"} for f in projections.values()):
            errors.append(f"{name}: named Q8 recipe wrote {projections}")
        print(f"{name}: {len(projections)} projections {sorted({f for fs in projections.values() for f in fs})}",
              flush=True)
    return errors


def dflash_cases(work: Path, native: Path, repository: Path, resources_from: Path) -> list[str]:
    """z-lab's DFlash drafter (BF16): without --recipe the converter chooses gemma4_dflash_published
    and nothing is quantized; NInferno's Q8 of it is written only when gemma4_dflash_q8 is named."""
    errors = []
    root = work / "dflash"
    sg.dflash_checkpoint(root)
    overrides = [f"--resource={role}={resources_from / role}" for role in
                 ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")]
    runs = {"dflash-default": [], "dflash-q8": ["--recipe=gemma4_dflash_q8"]}
    for name, extra in runs.items():
        out = work / f"{name}.ninferno"
        result = subprocess.run([str(native), f"--model={root}", *overrides, *extra, f"--out={out}"],
                                cwd=repository, capture_output=True, text=True)
        if result.returncode:
            errors.append(f"{name}: conversion failed: {result.stderr.strip()[-600:]}")
            continue
        value, objects, _, _ = sf.directory(out)
        kinds = formats(value, objects)
        quantized = sorted({f for fs in kinds.values() for f in fs if f.startswith(("q8", "q6", "q4"))})
        if name == "dflash-default":
            if "recipe gemma4_dflash_published (chosen" not in result.stdout:
                errors.append(f"{name}: the chosen recipe is not gemma4_dflash_published\n{result.stdout[-800:]}")
            if quantized:
                errors.append(f"{name}: not as published: quantized {quantized}")
        elif quantized != ["q8_g32_fp16"]:
            errors.append(f"{name}: named Q8 recipe wrote {quantized}")
        print(f"{name}: formats {sorted({f for fs in kinds.values() for f in fs})}", flush=True)
    return errors


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2])
    errors = []
    with tempfile.TemporaryDirectory(prefix="ninferno-convert-g4pub-") as tmp:
        work = Path(tmp)
        cases = {"dense-ct": dict(moe=False, layout="ct", nested=False, tied=False, seed=3),
                 "moe-ct": dict(moe=True, layout="ct", nested=True, tied=True, seed=1),
                 "dense-bf16": dict(moe=False, layout="modelopt", nested=False, tied=False, seed=4)}
        for name, spec in cases.items():
            root = work / name
            sg.text_checkpoint(root, **spec)
            out = work / f"{name}.ninferno"
            result = subprocess.run([str(native), f"--model={root}", "--recipe=gemma4_published", "--quiet",
                                     f"--out={out}"], cwd=repository, capture_output=True, text=True)
            if result.returncode:
                errors.append(f"{name}: conversion failed: {result.stderr.strip()[:400]}")
                continue
            value, objects, _, _ = sf.directory(out)
            kinds = formats(value, objects)
            own = sorted({f for fs in kinds.values() for f in fs if f.startswith(("q8", "q6", "q4"))})
            if own:
                errors.append(f"{name}: requantized objects {own}")
            for table in ("text/token_embedding", "text/output_head"):
                if kinds.get(table) != {"bf16"}:
                    errors.append(f"{name}: {table} is {kinds.get(table)}, expected bf16")
            projections = {n: f for n, f in kinds.items()
                           if n.startswith("text/layers/") and n.split("/")[-1] in
                           ("query", "key", "value", "output", "gate", "up", "down", "gate_up")}
            expected = "bf16" if spec["layout"] == "modelopt" else "nvfp4"
            wrong = {n: f for n, f in projections.items() if f != {expected}}
            if not projections or wrong:
                errors.append(f"{name}: projections not {expected}: {dict(list(wrong.items())[:4])}")
            if spec["moe"]:
                experts = {n: f for n, f in kinds.items() if "/moe/experts/" in n}
                if not experts or any(f != {"nvfp4"} for f in experts.values()):
                    errors.append(f"{name}: expert banks {set().union(*experts.values()) if experts else None}")
                router = [f for n, f in kinds.items() if n.endswith("/moe/router")]
                if not router or any(f != {"bf16"} for f in router):
                    errors.append(f"{name}: router {router}")
            if expected == "nvfp4":
                policies = {u["activation_policy"] for u in value["uses"]
                            if u["parameter"] in projections}
                if policies != {"AllowA4"}:
                    errors.append(f"{name}: projection policies {policies}")
            print(f"{name}: {len(projections)} projections {expected}, table and head bf16", flush=True)
        errors += assistant_cases(work, native, repository)
        errors += dflash_cases(work, native, repository, work / "dense-ct")
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
