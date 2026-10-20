"""HF transformers reference for the MoE quality gates (CPU, FP32 by default).

usage: python3 tools/moe_quality/hf_reference.py <checkpoint dir> <token dir> <tag> <out dir> [--dtype fp32|bf16] [--decode N]
                 [--nvfp4-dequant [--a4 MODE[,MODE...]]]

For every <tag>_<name>.i32 in the token dir except <tag>_chat (a generation prompt):
  <out>/<tag>_<name>.ref   two raw planes like `ninferno-dense score --dump`: FP32 log p(token i+1)
                           for i < T-1, then I32 argmax per position; followed by a third plane,
                           FP32 top-1 minus top-2 logit gap per position (near-tie analysis)
For <tag>_chat: greedy generation of N tokens,
  <out>/<tag>_chat.greedy.json   {"tokens": [...], "gaps": [...]} (gap: top-1 minus top-2 logit)
gpt-oss is loaded from its MXFP4 checkpoint dequantized (Mxfp4Config(dequantize=True)).
--nvfp4-dequant: a modelopt NVFP4 checkpoint (weight / weight_scale / weight_scale_2) is
dequantized to the model dtype before transformers sees it (transformers has no NVFP4 loader).
--a4 (with --nvfp4-dequant): emulate the checkpoint's NVFP4 input activations (modelopt W4A4,
input_activations num_bits 4) by fake-quantizing the input of every projection that carries an
input_scale: blocks of 16 along K, block scale amax/6 in units of input_scale rounded to E4M3,
values rounded to E2M1 (nearest, ties to even, saturating at 6). MODE is one of
  off        no emulation (the plain reference, written to <out>)
  attention  q/k/v/o only: the projections an engine runs A4 while its MoE experts run BF16
  declared   every projection the checkpoint declares A4 (attention and every expert's
             gate/up/down)
Several modes run over one load; each mode but off writes to <out>/a4_<mode>/.
"""
import argparse, glob, json, os, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('tokens'); ap.add_argument('tag'); ap.add_argument('out')
ap.add_argument('--dtype', default='fp32', choices=('fp32', 'bf16'))
ap.add_argument('--decode', type=int, default=64)
ap.add_argument('--nvfp4-dequant', action='store_true')
ap.add_argument('--a4', default='off')
args = ap.parse_args()
modes = args.a4.split(',')
assert all(m in ('off', 'attention', 'declared') for m in modes), modes
assert modes == ['off'] or args.nvfp4_dequant, '--a4 needs --nvfp4-dequant'
os.makedirs(args.out, exist_ok=True)
dtype = torch.float32 if args.dtype == 'fp32' else torch.bfloat16
torch.set_num_threads(int(os.environ.get('THREADS', os.cpu_count())))
t0 = time.time()

from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained(args.ckpt)
kw = dict(dtype=dtype, device_map='cpu', attn_implementation='eager')
if cfg.model_type == 'gpt_oss':
    from transformers import Mxfp4Config
    kw['quantization_config'] = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, **kw)
elif args.nvfp4_dequant:
    from safetensors import safe_open
    E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    if hasattr(cfg, 'quantization_config'):
        del cfg.quantization_config
    model = AutoModelForCausalLM.from_config(cfg, dtype=dtype, attn_implementation='eager')
    state = {}
    for f in sorted(glob.glob(args.ckpt + '/*.safetensors')):
        with safe_open(f, 'pt') as h:
            for k in h.keys():
                state[k] = (f, k)
    def raw(k):
        with safe_open(state[k][0], 'pt') as h:
            return h.get_tensor(k)

    def weight(name):  # a checkpoint weight by name, NVFP4 words dequantized in FP32
        t = raw(name)
        base = name[:-len('weight')]
        if t.dtype == torch.uint8 and base + 'weight_scale' in state:
            s1 = raw(base + 'weight_scale').float()
            s2 = raw(base + 'weight_scale_2').float()
            lo, hi = (t & 15).long(), (t >> 4).long()
            codes = torch.stack([E2M1[lo], E2M1[hi]], dim=-1).reshape(t.shape[0], -1)
            w = codes.reshape(codes.shape[0], -1, 16) * s1.reshape(codes.shape[0], -1, 1) * s2
            t = w.reshape(codes.shape)
        return t

    params = dict(model.named_parameters())
    loaded, fused, missing = 0, 0, []
    for name, p in params.items():
        if name in state:
            t = weight(name)
            loaded += 1
        elif name.endswith('mlp.experts.gate_up_proj') or name.endswith('mlp.experts.down_proj'):
            # transformers >= 5 keeps a layer's experts as one 3D tensor per projection:
            # gate_up_proj[e] = [gate_e; up_e] ([2F, H]), down_proj[e] = down_e ([H, F]).
            pre = name[:-len('gate_up_proj')] if name.endswith('gate_up_proj') else name[:-len('down_proj')]
            n_exp = p.shape[0]
            if name.endswith('gate_up_proj'):
                t = torch.stack([torch.cat([weight(f'{pre}{e}.gate_proj.weight'),
                                            weight(f'{pre}{e}.up_proj.weight')], dim=0)
                                 for e in range(n_exp)])
            else:
                t = torch.stack([weight(f'{pre}{e}.down_proj.weight') for e in range(n_exp)])
            fused += 1
        else:
            missing.append(name)
            continue
        assert tuple(t.shape) == tuple(p.shape), (name, t.shape, p.shape)
        with torch.no_grad():
            p.copy_(t.to(p.dtype))
    print('loaded', loaded, 'fused expert banks', fused, 'missing', missing[:5], len(missing), flush=True)
    if cfg.tie_word_embeddings:
        model.tie_weights()
    if modes != ['off']:
        import types
        A4 = {'attention': False, 'experts': False}  # what the running mode fake-quantizes
        MIDS = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
        GRID = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])

        def fq(x, s):  # NVFP4 fake-quant of x's last dim (blocks of 16) under global scale s
            shape = x.shape
            b = x.float().reshape(-1, shape[-1] // 16, 16)
            bs = (b.abs().amax(-1, keepdim=True) / 6 / s).clamp(max=448)
            bs = bs.to(torch.float8_e4m3fn).float() * s
            y = torch.where(bs > 0, b / torch.where(bs > 0, bs, 1), 0).clamp(-6, 6)
            a = y.abs()
            lo = torch.bucketize(a, MIDS, right=False)
            hi = torch.bucketize(a, MIDS, right=True)
            idx = torch.where((lo != hi) & (lo % 2 == 1), hi, lo)
            return (GRID[idx] * y.sign() * bs).reshape(shape).to(x.dtype)

        def scale(key):
            return float(raw(key).float()) if key in state else None

        attn_hooked, experts_hooked = 0, 0
        for i, layer in enumerate(model.model.layers):
            for proj in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                s_in = scale(f'model.layers.{i}.self_attn.{proj}.input_scale')
                if s_in is None:
                    continue
                getattr(layer.self_attn, proj).register_forward_pre_hook(
                    lambda m, a, s_in=s_in: (fq(a[0], s_in),) + tuple(a[1:]) if A4['attention'] else None)
                attn_hooked += 1
            ex = getattr(layer.mlp, 'experts', None)
            if ex is None or not hasattr(ex, 'gate_up_proj'):
                continue
            pre = f'model.layers.{i}.mlp.experts.'
            sc = [[scale(f'{pre}{e}.{p}_proj.input_scale') for p in ('gate', 'up', 'down')]
                  for e in range(ex.gate_up_proj.shape[0])]
            if any(v is None for row in sc for v in row):
                continue
            ex.a4_scales = sc
            ex.eager_forward = ex.forward

            def experts_forward(self, hidden_states, top_k_index, top_k_weights):
                # The eager expert loop of transformers' Qwen3MoeExperts, with fake-quantized
                # gate, up and down inputs (each expert its own input_scale).
                if not A4['experts']:
                    return self.eager_forward(hidden_states, top_k_index, top_k_weights)
                final = torch.zeros_like(hidden_states)
                mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0].tolist():
                    top_k_pos, token_idx = torch.where(mask[e])
                    x = hidden_states[token_idx]
                    w = self.gate_up_proj[e]
                    f = w.shape[0] // 2
                    sg, su, sd = self.a4_scales[e]
                    gate = torch.nn.functional.linear(fq(x, sg), w[:f])
                    up = torch.nn.functional.linear(fq(x, su), w[f:])
                    h = torch.nn.functional.linear(fq(self.act_fn(gate) * up, sd), self.down_proj[e])
                    h = h * top_k_weights[token_idx, top_k_pos, None]
                    final.index_add_(0, token_idx, h.to(final.dtype))
                return final

            ex.forward = types.MethodType(experts_forward, ex)
            experts_hooked += 1
        print('a4 emulation: attention projections', attn_hooked, 'expert banks', experts_hooked, flush=True)
else:
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, **kw)
model.eval()
# Mxfp4Config(dequantize=True) lands the experts in BF16 (exact: an E2M1 value times a power of two
# fits BF16), and model.to() leaves the dequantized expert banks alone, so every tensor is cast
# here, one module at a time.
with torch.no_grad():
    for m in model.modules():
        for n, p in m.named_parameters(recurse=False):
            if p.is_floating_point() and p.dtype != dtype:
                p.data = p.data.to(dtype)
hist = {}
for n, t in list(model.named_parameters()) + list(model.named_buffers()):
    hist.setdefault(str(t.dtype), [0, 0])
    hist[str(t.dtype)][0] += 1
    hist[str(t.dtype)][1] += t.numel()
print(f'load {time.time() - t0:.1f}s dtype {model.dtype} tensors {hist}', flush=True)
print('experts module', type(model.model.layers[0].mlp.experts).__name__,
      {n: (str(t.dtype), tuple(t.shape)) for n, t in model.model.layers[0].mlp.experts.__dict__.items()
       if torch.is_tensor(t)} if cfg.model_type == 'gpt_oss' else '', flush=True)

# <tag>_long is the speed benches' prompt, not a quality text.
names = sorted(os.path.basename(p)[len(args.tag) + 1:-4]
               for p in glob.glob(f'{args.tokens}/{args.tag}_*.i32'))
names = [n for n in names if n != 'long']
root = args.out
for mode in modes:
    if mode != 'off':
        A4['attention'], A4['experts'] = True, mode == 'declared'
        args.out = f'{root}/a4_{mode}'
        os.makedirs(args.out, exist_ok=True)
    elif modes != ['off']:
        A4['attention'] = A4['experts'] = False
        args.out = root
    print('mode', mode, flush=True)
    summary = {}
    with torch.no_grad():
        for name in names:
            ids = np.fromfile(f'{args.tokens}/{args.tag}_{name}.i32', dtype='<i4').astype(np.int64)
            t1 = time.time()
            if name == 'chat':
                x = torch.tensor(ids)[None]
                out = model.generate(x, do_sample=False, max_new_tokens=args.decode,
                                     output_scores=True, return_dict_in_generate=True,
                                     pad_token_id=0)
                toks = out.sequences[0, len(ids):].tolist()
                gaps = []
                for s in out.scores:
                    v = torch.topk(s[0].float(), 2).values
                    gaps.append(float(v[0] - v[1]))
                with open(f'{args.out}/{args.tag}_chat.greedy.json', 'w', encoding="utf-8", newline="\n") as f:
                    json.dump({'tokens': toks, 'gaps': gaps}, f)
                summary[name] = {'generated': len(toks), 'seconds': time.time() - t1}
                print(name, toks[:20], flush=True)
                continue
            logits = model(torch.tensor(ids)[None]).logits[0].float()
            lp = torch.log_softmax(logits, dim=-1)
            target = lp[:-1].gather(1, torch.tensor(ids[1:])[:, None])[:, 0]
            top = torch.topk(logits, 2, dim=-1).values
            arg = logits.argmax(dim=-1).to(torch.int32)
            with open(f'{args.out}/{args.tag}_{name}.ref', 'wb') as f:
                f.write(target.numpy().astype('<f4').tobytes())
                f.write(arg[:-1].numpy().astype('<i4').tobytes())
                f.write((top[:-1, 0] - top[:-1, 1]).numpy().astype('<f4').tobytes())
            nll = float(-target.mean())
            summary[name] = {'tokens': int(len(ids)), 'mean_nll': nll, 'seconds': time.time() - t1}
            print(name, summary[name], flush=True)
    with open(f'{args.out}/{args.tag}_summary.json', 'w', encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
print('REF_DONE', f'{time.time() - t0:.1f}s')
