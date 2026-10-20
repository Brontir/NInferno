"""NVFP4 scales through the native converter: fused projections and MoE experts.

    python synthetic_scales.py NINFERNO_CONVERT [REPOSITORY]

The oracle is FP32 arithmetic on the checkpoint's own words (code x E4M3 block scale / divisor)
against the same arithmetic on the artifact's words (its own block scales and divisors), in
both NVFP4 layouts (compressed-tensors divisors, Model Optimizer multipliers):

  fused      q/k/v and gate/up with per-tensor divisors that differ (2x and 4x), and MoE
             experts whose up divisor is 2x its gate divisor (the stacked plane adopts gate's
             divisor, the expert's down compensates): every binding decodes to the checkpoint's
             values, every block scale word is the checkpoint's (nothing is recomputed, so
             nothing saturates at 448), and each expert's GeGLU FFN on random inputs matches
             the checkpoint's to rel < 1e-2.
  experts    per-expert activation scales that differ 4x: every expert of a bank keeps the
             smallest calibrated divisor (one activation tensor feeds the bank; the widest
             range saturates nothing), not expert 0's or the largest.
  dead       a dead expert (amax 0: zero codes and block scales) whose scalar scales are what a
             calibrator writes for amax 0 or ~0 (ModelOpt 0, 1e-23; compressed-tensors inf, 1e23),
             NaN, or the corrupt opposite converts, decodes to exact zeros, carries finite
             positive divisors, leaves the bank's activation divisor to its calibrated peers, and
             the log and report say what was substituted; a never-activated expert (live weights,
             the same activation scales) converts with its weights exact; a NaN weight scale over
             live block scales is still refused by name.

Sabotage arms: the oracle run on artifacts with a doubled divisor, an uncompensated down, a
swapped expert and a per-expert activation divisor must fail, and so must reading the stacked
plane as [up; gate] instead of the [gate; up] the kernels read. Exits 77 (skipped)
without torch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

try:
    import torch
except ImportError:
    sys.exit(77)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synthetic_gemma4 as base  # noqa: E402

E4M3 = torch.arange(256, dtype=torch.int64).to(torch.uint8).view(torch.float8_e4m3fn).float()
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
EXPERTS, LAYERS = 4, 2


def f32_word(value: float) -> bytes:
    return struct.pack("<f", value)


def read_checkpoint(root: Path) -> dict:
    tensors = {}
    for file in sorted(root.glob("*.safetensors")):
        raw = file.read_bytes()
        n = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + n])
        data = raw[8 + n:]
        for name, item in header.items():
            if name != "__metadata__":
                begin, end = item["data_offsets"]
                tensors[name] = (item["dtype"], item["shape"], data[begin:end])
    return tensors


def rewrite_checkpoint(root: Path, change) -> None:
    tensors = read_checkpoint(root)
    for file in root.glob("*.safetensors"):
        file.unlink()
    index = root / "model.safetensors.index.json"
    if index.exists():
        index.unlink()
    change(tensors)
    writer = base.Writer(0)
    for name, (dtype, shape, data) in tensors.items():
        writer.raw(name, dtype, shape, data)
    writer.save(root)


def fields(layout: str) -> tuple[str, str, str]:
    """(codes suffix, weight scalar suffix, activation scalar suffix)."""
    if layout == "ct":
        return ".weight_packed", ".weight_global_scale", ".input_global_scale"
    return ".weight", ".weight_scale_2", ".input_scale"


def scalar_divisor(layout: str, value: float) -> float:
    """The divisor a stored scalar means: compressed-tensors stores it, ModelOpt its binary32
    reciprocal."""
    if layout == "ct":
        return value
    with torch.no_grad():
        return (torch.tensor(1.0, dtype=torch.float32) / torch.tensor(value, dtype=torch.float32)).item()


def decode(codes: bytes, scales: torch.Tensor, divisors: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """FP32 values of NVFP4 words: e2m1(code) * e4m3(scale) / divisor(row)."""
    raw = torch.frombuffer(bytearray(codes), dtype=torch.uint8).reshape(n, k // 2).long()
    nibbles = torch.stack((raw & 15, raw >> 4), -1).reshape(n, k)
    block = E4M3[scales.long()].repeat_interleave(16, 1)
    return (E2M1[nibbles] * block) / divisors.reshape(n, 1)


def source_values(tensors: dict, prefix: str, layout: str) -> tuple[torch.Tensor, torch.Tensor]:
    codes_suffix, weight_suffix, _ = fields(layout)
    _, shape, codes = tensors[prefix + codes_suffix]
    n, k = shape[0], shape[1] * 2
    scales = torch.frombuffer(bytearray(tensors[prefix + ".weight_scale"][2]), dtype=torch.uint8).reshape(n, k // 16)
    divisor = scalar_divisor(layout, struct.unpack("<f", tensors[prefix + weight_suffix][2])[0])
    divisors = torch.full((n,), divisor, dtype=torch.float32)
    return decode(codes, scales, divisors, n, k), scales


class Artifact:
    def __init__(self, path: Path) -> None:
        data = path.read_bytes()
        length = struct.unpack("<Q", data[8:16])[0]
        self.directory = json.loads(data[32:32 + length].decode())
        self.payload = bytearray(data[-(-(32 + length) // 4096) * 4096:])
        self.objects = {o["id"]: o for o in self.directory["objects"]}
        self.bindings = self.directory["bindings"]
        self.uses = {u["parameter"]: u for u in self.directory["uses"]}

    def words(self, oid: str) -> bytearray:
        o = self.objects[oid]
        return self.payload[o["offset"]:o["offset"] + o["bytes"]]

    def nvfp4_object(self, oid: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(values [N,K], natural scale words [N,K/16], per-row divisors [N])."""
        from tools.artifact.codecs.nvfp4 import unswizzle_nvfp4_scales
        from tools.artifact.layouts import block_scale_geometry

        o = self.objects[oid]
        n, k = o["shape"]
        words = self.words(oid)
        g = block_scale_geometry("nvfp4", (n, k))
        count = (len(words) - g.weight_divisor_offset) // 4
        divisors = torch.tensor(struct.unpack(f"<{count}f", bytes(words[g.weight_divisor_offset:])))
        stored = torch.frombuffer(bytearray(words[g.scale_plane_offset:g.scale_plane_offset + g.scale_plane_bytes]),
                                  dtype=torch.uint8)
        scales = unswizzle_nvfp4_scales(stored, (n, k))
        rows = divisors.repeat_interleave(n // count)
        return decode(bytes(words[:g.code_plane_bytes]), scales, rows, n, k), scales, rows

    def parameter(self, name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        binding = self.bindings[name]
        parts = binding["parts"] if "parts" in binding else [{"object": binding["object"], "range": None}]
        values, scales, divisors = [], [], []
        for part in parts:
            v, s, d = self.nvfp4_object(part["object"])
            k = v.shape[1]
            begin, end = part["range"] if part["range"] else (0, v.numel())
            rows = slice(begin // k, end // k)
            values.append(v[rows]); scales.append(s[rows]); divisors.append(d[rows])
        return torch.cat(values), torch.cat(scales), torch.cat(divisors)

    def activation_divisor(self, name: str) -> float | None:
        aux = (self.uses.get(name, {}).get("auxiliaries") or {}).get("activation_input_divisor")
        if aux is None:
            return None
        return struct.unpack("<f", bytes(self.words(aux["object"])[:4]))[0]


def convert(native: Path, repository: Path, model: Path, out: Path, recipe: str,
            *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(native), f"--model={model}", f"--recipe={recipe}", f"--out={out}", *extra],
                          cwd=repository, capture_output=True, text=True)


# ------------------------------------------------------------------------------------------------
# Oracles: each returns a list of failures.


def same_values(artifact: Artifact, tensors: dict, layout: str, pairs: dict[str, str], root: str) -> list[str]:
    """Every binding decodes to its checkpoint matrix bit for bit and keeps its block scales."""
    errors = []
    for parameter, prefix in pairs.items():
        expected, source_scales = source_values(tensors, root + prefix, layout)
        values, scales, _ = artifact.parameter(parameter)
        if not torch.equal(scales, source_scales):
            errors.append(f"{parameter}: block scale words differ from the checkpoint's")
        if not torch.equal(values, expected):
            bad = (values != expected).sum().item()
            errors.append(f"{parameter}: {bad} values differ from the checkpoint's")
    return errors


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def expert_ffn(gate, up, down, x):
    return (gelu_tanh(x @ gate.T) * (x @ up.T)) @ down.T


def expert_functions(artifact: Artifact, tensors: dict, layout: str, root: str, layer: int,
                     skip=(), swap_halves: bool = False) -> tuple[list[str], list[float]]:
    """Each expert's FFN from the artifact (stacked gate_up, compensated down) against the
    checkpoint's gate, up and down; rel L2 < 1e-2 per expert. The artifact's gate_up plane is
    read as [gate; up], the order the MoE kernels read it; `swap_halves` reads it as [up; gate]
    (the control: a swapped plane must fail)."""
    errors, rels = [], []
    generator = torch.Generator().manual_seed(layer)
    for e in range(EXPERTS):
        if e in skip:
            continue
        p = f"{root}layers.{layer}.experts.{e}."
        gate = source_values(tensors, p + "gate_proj", layout)[0]
        up = source_values(tensors, p + "up_proj", layout)[0]
        down = source_values(tensors, p + "down_proj", layout)[0]
        a = f"text/layers/{layer}/moe/experts/{e}/"
        gate_up = artifact.parameter(a + "gate_up")[0]
        adown = artifact.parameter(a + "down")[0]
        rows = gate_up.shape[0] // 2
        x = torch.randn(7, gate.shape[1], generator=generator)
        want = expert_ffn(gate.double(), up.double(), down.double(), x.double())
        first, second = (gate_up[rows:], gate_up[:rows]) if swap_halves else (gate_up[:rows], gate_up[rows:])
        got = expert_ffn(first.double(), second.double(), adown.double(), x.double())
        rel = ((got - want).norm() / want.norm()).item()
        rels.append(rel)
        if not math.isfinite(rel) or rel >= 1e-2:
            errors.append(f"layer {layer} expert {e}: FFN rel {rel:.3e} against the checkpoint")
    return errors, rels


def bank_activation(artifact: Artifact, tensors: dict, layout: str, layer: int) -> list[str]:
    """Every expert's gate_up activation divisor is the smallest finite positive calibrated
    gate divisor of the bank; the down divisors are one finite positive value."""
    _, _, input_suffix = fields(layout)
    calibrated = []
    for e in range(EXPERTS):
        stored = struct.unpack("<f", tensors[f"model.language_model.layers.{layer}.experts.{e}.gate_proj{input_suffix}"][2])[0]
        if not math.isfinite(stored) or stored <= 0:
            continue  # never calibrated: takes the bank's value, does not set it
        divisor = scalar_divisor(layout, stored)
        if math.isfinite(divisor) and divisor > 0:
            calibrated.append(divisor)
    want = min(calibrated)
    errors = []
    downs = set()
    for e in range(EXPERTS):
        got = artifact.activation_divisor(f"text/layers/{layer}/moe/experts/{e}/gate_up")
        if got != want:
            errors.append(f"layer {layer} expert {e}: gate_up activation divisor {got}, bank minimum {want}")
        down = artifact.activation_divisor(f"text/layers/{layer}/moe/experts/{e}/down")
        downs.add(down)
        if down is None or not (math.isfinite(down) and down > 0):
            errors.append(f"layer {layer} expert {e}: down activation divisor {down}")
    if len(downs) != 1:
        errors.append(f"layer {layer}: down activation divisors differ across the bank: {sorted(map(str, downs))}")
    return errors


# ------------------------------------------------------------------------------------------------


class Results:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def case(self, name: str, errors: list[str], detail: str = "") -> None:
        print(f"{name}: {'ok' if not errors else 'FAILED'}{('  ' + detail) if detail else ''}", flush=True)
        self.failed += [f"{name}: {e}" for e in errors]

    def control(self, name: str, errors: list[str]) -> None:
        print(f"{name} (sabotage arm): {'caught, ' + str(len(errors)) + ' failures' if errors else 'NOT CAUGHT'}",
              flush=True)
        if not errors:
            self.failed.append(f"{name}: the oracle passes the sabotaged artifact")


def moe_checkpoint(root: Path, layout: str, seed: int, change=None) -> dict:
    base.text_checkpoint(root, moe=True, layout=layout, nested=True, tied=True, seed=seed)

    def fused_scales(t):
        _, weight_suffix, input_suffix = fields(layout)
        for layer in range(LAYERS):
            for e in range(EXPERTS):
                p = f"model.language_model.layers.{layer}.experts.{e}."
                gate = 0.2 + 0.01 * e
                # up's divisor is twice gate's; activation divisors 4x apart across experts.
                for role, divisor in (("gate_proj", gate), ("up_proj", 2 * gate)):
                    stored = divisor if layout == "ct" else 1.0 / divisor
                    t[p + role + weight_suffix] = (t[p + role + weight_suffix][0], t[p + role + weight_suffix][1],
                                                   f32_word(stored))
                for role in ("gate_proj", "up_proj", "down_proj"):
                    activation = 0.25 * (4.0 ** (e / (EXPERTS - 1)))  # 0.25 .. 1.0
                    stored = activation if layout == "ct" else 1.0 / activation
                    t[p + role + input_suffix] = (t[p + role + input_suffix][0], t[p + role + input_suffix][1],
                                                  f32_word(stored))
        if change:
            change(t)

    rewrite_checkpoint(root, fused_scales)
    return read_checkpoint(root)


def fused_case(results: Results, native: Path, repository: Path, work: Path) -> None:
    for layout in ("ct", "modelopt"):
        # Dense projections: q/k/v divisors 0.5/1.0/0.25, gate/up 0.3/0.6 (layer 0 of each).
        root = work / f"dense-{layout}"
        base.text_checkpoint(root, moe=False, layout="ct", nested=False, tied=False, seed=31)
        if layout == "modelopt":
            def to_modelopt(t):
                for name in [n for n in t if n.endswith(".weight_packed")]:
                    prefix = name[: -len(".weight_packed")]
                    t[prefix + ".weight"] = t.pop(name)
                    g = struct.unpack("<f", t.pop(prefix + ".weight_global_scale")[2])[0]
                    i = struct.unpack("<f", t.pop(prefix + ".input_global_scale")[2])[0]
                    t[prefix + ".weight_scale_2"] = ("F32", [], f32_word(1.0 / g))
                    t[prefix + ".input_scale"] = ("F32", [], f32_word(1.0 / i))
            rewrite_checkpoint(root, to_modelopt)

        def dense_scales(t):
            _, weight_suffix, _ = fields(layout)
            for role, divisor in (("self_attn.q_proj", 0.5), ("self_attn.k_proj", 1.0), ("self_attn.v_proj", 0.25),
                                  ("mlp.gate_proj", 0.3), ("mlp.up_proj", 0.6)):
                name = f"model.layers.0.{role}{weight_suffix}"
                stored = divisor if layout == "ct" else 1.0 / divisor
                t[name] = (t[name][0], t[name][1], f32_word(stored))
        rewrite_checkpoint(root, dense_scales)
        tensors = read_checkpoint(root)
        out = work / f"dense-{layout}.ninferno"
        run = convert(native, repository, root, out, "gemma4_31b_nvfp4")
        if run.returncode:
            results.case(f"fused[dense {layout}]", [f"conversion failed: {run.stderr[-800:]}"])
            continue
        artifact = Artifact(out)
        pairs = {f"text/layers/0/attention/{r}": f"layers.0.self_attn.{f}"
                 for r, f in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj"))}
        pairs.update({f"text/layers/0/mlp/{r}": f"layers.0.mlp.{r}_proj" for r in ("gate", "up")})
        results.case(f"fused[dense {layout}]", same_values(artifact, tensors, layout, pairs, "model."))
        if layout == "ct":
            sabotaged = Artifact(out)
            oid = sabotaged.bindings["text/layers/0/mlp/up"]["parts"][0]["object"] \
                if "parts" in sabotaged.bindings["text/layers/0/mlp/up"] else sabotaged.bindings["text/layers/0/mlp/up"]["object"]
            o = sabotaged.objects[oid]
            end = o["offset"] + o["bytes"]
            last = struct.unpack("<f", bytes(sabotaged.payload[end - 4:end]))[0]
            sabotaged.payload[end - 4:end] = f32_word(last * 2)  # doubled weight divisor
            results.control("fused[dense doubled divisor]", same_values(sabotaged, tensors, layout, pairs, "model."))

        # MoE experts: up divisor = 2 x gate divisor, activation divisors 4x apart.
        root = work / f"moe-{layout}"
        tensors = moe_checkpoint(root, layout, seed=32)
        out = work / f"moe-{layout}.ninferno"
        run = convert(native, repository, root, out, "gemma4_26b_a4b_nvfp4")
        if run.returncode:
            results.case(f"fused[experts {layout}]", [f"conversion failed: {run.stderr[-800:]}"])
            continue
        artifact = Artifact(out)
        errors, rels = [], []
        for layer in range(LAYERS):
            e, r = expert_functions(artifact, tensors, layout, "model.language_model.", layer)
            errors += e
            rels += r
            for expert in range(EXPERTS):
                p = f"model.language_model.layers.{layer}.experts.{expert}."
                stored = artifact.parameter(f"text/layers/{layer}/moe/experts/{expert}/gate_up")[1]
                gate_s = source_values(tensors, p + "gate_proj", layout)[1]
                up_s = source_values(tensors, p + "up_proj", layout)[1]
                if not torch.equal(stored, torch.cat((gate_s, up_s))):
                    errors.append(f"layer {layer} expert {expert}: gate_up block scales were recomputed")
        results.case(f"fused[experts {layout}]", errors, f"max FFN rel {max(rels):.3e} over {len(rels)} experts")
        if layout == "ct":
            sabotaged = Artifact(out)
            oid = sabotaged.bindings["text/layers/0/moe/experts/1/down"]["parts"][0]["object"]
            o = sabotaged.objects[oid]
            from tools.artifact.layouts import block_scale_geometry
            g = block_scale_geometry("nvfp4", o["shape"])
            at = o["offset"] + g.weight_divisor_offset + 4 * 1  # expert 1's down divisor
            value = struct.unpack("<f", bytes(sabotaged.payload[at:at + 4]))[0]
            sabotaged.payload[at:at + 4] = f32_word(value / 2)  # the uncompensated divisor
            results.control("fused[experts uncompensated down]",
                            expert_functions(sabotaged, tensors, layout, "model.language_model.", 0)[0])
            swapped = Artifact(out)
            a, b = (swapped.bindings[f"text/layers/0/moe/experts/{e}/gate_up"] for e in (0, 1))
            a["parts"], b["parts"] = b["parts"], a["parts"]
            results.control("fused[experts swapped planes]",
                            expert_functions(swapped, tensors, layout, "model.language_model.", 0)[0])
            results.control("fused[experts gate_up read as up;gate]",
                            expert_functions(artifact, tensors, layout, "model.language_model.", 0,
                                             swap_halves=True)[0])

        # Per-expert activation divisors 4x apart (0.25 .. 1.0): one activation tensor
        # feeds the bank, so every expert keeps the smallest calibrated divisor (the widest
        # range, nothing saturates) rather than expert 0's or the largest.
        errors = bank_activation(artifact, tensors, layout, layer=0)
        results.case(f"experts[bank activation divisor {layout}]", errors,
                     f"checkpoint {', '.join(f'{0.25 * 4.0 ** (e / 3):.4g}' for e in range(EXPERTS))}")
        if layout == "ct":
            sabotaged = Artifact(out)  # expert 2 keeps its own calibration (0.63), not the bank's
            aux = sabotaged.uses["text/layers/0/moe/experts/2/gate_up"]["auxiliaries"]["activation_input_divisor"]
            o = sabotaged.objects[aux["object"]]
            sabotaged.payload[o["offset"]:o["offset"] + 4] = f32_word(0.25 * 4.0 ** (2 / 3))
            results.control("experts[bank activation divisor]", bank_activation(sabotaged, tensors, layout, layer=0))


def dead_case(results: Results, native: Path, repository: Path, work: Path) -> None:
    dead = 3
    for layout in ("ct", "modelopt"):
        codes_suffix, weight_suffix, input_suffix = fields(layout)
        # What a calibrator writes for amax 0, a clamped amax ~2.7e-20, a NaN, and the corrupt
        # opposite: ModelOpt stores amax/2688 (0, 1e-23, nan, inf); compressed-tensors stores
        # 2688/amax (inf, 1e23, nan, 0).
        stored = (0.0, 1e-23, float("nan"), float("inf")) if layout == "modelopt" else (
            float("inf"), 1e23, float("nan"), 0.0)
        variants = [("amax0", value) for value in stored]
        variants += [("never-activated", value) for value in stored]
        variants += [("live-nan-weight", float("nan"))]
        for kind, value in variants:
            def change(t, kind=kind, value=value):
                for layer in range(LAYERS):
                    for role in ("gate_proj", "up_proj", "down_proj"):
                        p = f"model.language_model.layers.{layer}.experts.{dead}.{role}"
                        if kind == "amax0":
                            for suffix in (codes_suffix, ".weight_scale"):
                                t[p + suffix] = (t[p + suffix][0], t[p + suffix][1], bytes(len(t[p + suffix][2])))
                            t[p + weight_suffix] = (t[p + weight_suffix][0], t[p + weight_suffix][1], f32_word(value))
                            t[p + input_suffix] = (t[p + input_suffix][0], t[p + input_suffix][1], f32_word(value))
                        elif kind == "never-activated":
                            t[p + input_suffix] = (t[p + input_suffix][0], t[p + input_suffix][1], f32_word(value))
                        else:
                            t[p + weight_suffix] = (t[p + weight_suffix][0], t[p + weight_suffix][1], f32_word(value))
            name = f"dead[{layout} {kind} {value}]"
            root = work / f"dead-{layout}-{kind}-{value}"
            tensors = moe_checkpoint(root, layout, seed=33, change=change)
            out = work / f"dead-{layout}-{kind}-{value}.ninferno"
            run = convert(native, repository, root, out, "gemma4_26b_a4b_nvfp4")
            if kind == "live-nan-weight":
                errors = [] if run.returncode and f"experts.{dead}.gate_proj{weight_suffix}" in run.stderr else [
                    f"a NaN weight scale over live block scales was not refused by name (rc {run.returncode})"]
                results.case(name, errors, "refused" if not errors else "")
                continue
            if run.returncode:
                results.case(name, [f"conversion failed: {run.stderr.strip().splitlines()[0][:300]}"])
                continue
            artifact = Artifact(out)
            report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
            errors = []
            for layer in range(LAYERS):
                live = expert_functions(artifact, tensors, layout, "model.language_model.", layer, skip=(dead,))[0]
                errors += live
                a = f"text/layers/{layer}/moe/experts/{dead}/"
                for part in ("gate_up", "down"):
                    values, _, divisors = artifact.parameter(a + part)
                    activation = artifact.activation_divisor(a + part)
                    if kind == "amax0" and bool((values != 0).any()):
                        errors.append(f"layer {layer} {part}: the dead expert decodes to nonzero values")
                    if not bool(torch.isfinite(values).all()):
                        errors.append(f"layer {layer} {part}: non-finite values")
                    if not bool(torch.isfinite(divisors).all() and (divisors > 0).all()):
                        errors.append(f"layer {layer} {part}: divisor not finite positive")
                    if activation is None or not (math.isfinite(activation) and activation > 0):
                        errors.append(f"layer {layer} {part}: activation divisor {activation}")
                if kind == "never-activated":
                    errors += expert_functions(artifact, tensors, layout, "model.language_model.", layer,
                                               skip=tuple(e for e in range(EXPERTS) if e != dead))[0]
                errors += bank_activation(artifact, tensors, layout, layer)
            # A substituted scale is never silent: the report's native notes and the log say so.
            invalid = not (math.isfinite(value) and value > 0)
            mentioned = any(("zero matrix" in n["encoding"]) or ("uncalibrated" in n["encoding"])
                            for n in report.get("native_notes", []))
            if invalid and not mentioned:
                errors.append("the report does not note the substituted scale")
            if invalid and "zero matrix" not in run.stdout + run.stderr and "uncalibrated" not in run.stdout + run.stderr:
                errors.append("the log does not note the substituted scale")
            results.case(name, errors)


def kv_scales_case(results: Results, native: Path, repository: Path, work: Path) -> None:
    """Static KV-cache scales (k_scale/v_scale) are never applied - the KV codecs scale every token
    and group at append - so a checkpoint that carries them converts with a warning and a
    report entry naming them; tensors of towers the text converter does not read are reported
    too. Control: the same checkpoint without them has neither."""
    for extra in (True, False):
        root = work / f"kv-scales-{extra}"
        base.text_checkpoint(root, moe=False, layout="modelopt", nested=False, tied=True, seed=41)
        names = [f"model.layers.{layer}.self_attn.{kind}" for layer in range(LAYERS)
                 for kind in ("k_scale", "v_scale")]
        if extra:
            def add(t):
                for name in names:
                    t[name] = ("F32", [], f32_word(0.0125))
                t["model.vision_tower.encoder.weight"] = ("BF16", [2, 2], bytes(8))
            rewrite_checkpoint(root, add)
        out = work / f"kv-scales-{extra}.ninferno"
        # An all-BF16 Gemma checkpoint: the NVFP4 recipe quantizes it, a maintainer conversion.
        run = convert(native, repository, root, out, "gemma4_31b_nvfp4", "--maintainer-quantize")
        name = f"kv_scales[{'carried' if extra else 'control: absent'}]"
        if run.returncode:
            results.case(name, [f"conversion failed: {run.stderr[-600:]}"])
            continue
        report = json.loads(Path(str(out) + ".conversion.json").read_text(encoding="utf-8"))
        artifact = Artifact(out)
        errors = []
        listed = report.get("kv_cache_scales_ignored")
        log = run.stdout + run.stderr
        warned = "static KV-cache scale tensors not used" in log
        others = "checkpoint tensors not converted: model.vision_tower.encoder.* (1)" in log
        if extra:
            if sorted(listed or []) != sorted(names):
                errors.append(f"report lists {listed}, expected {names}")
            if not warned:
                errors.append("no warning in the log")
            if not others:
                errors.append("the log does not list the unconverted tower tensor")
            if any("scale" in b for b in artifact.bindings if "attention" in b):
                errors.append("a KV scale reached the artifact")
        elif listed is not None or others or warned or "not converted" in log:
            errors.append(f"a clean checkpoint reported {listed} / {others} (warned {warned})")
        results.case(name, errors)


def main() -> int:
    native = Path(sys.argv[1]).resolve()
    repository = Path(sys.argv[2] if len(sys.argv) > 2 else HERE.parents[1]).resolve()
    sys.path.insert(0, str(repository))
    results = Results()
    with tempfile.TemporaryDirectory(prefix="ninferno-convert-scales-") as tmp:
        work = Path(tmp)
        fused_case(results, native, repository, work)
        dead_case(results, native, repository, work)
        kv_scales_case(results, native, repository, work)
    for failure in results.failed:
        print(failure, file=sys.stderr)
    return 1 if results.failed else 0


if __name__ == "__main__":
    sys.exit(main())
