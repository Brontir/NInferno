# The local page: ninferno-ui

`ninferno-ui` is a small web page, served from your machine, to pick, download and run models on
its GPUs. It starts `ninferno-serve` processes for you, one per model and GPU, with the settings
measured for that GPU.

```bash
ninferno-ui
```

Once it listens, it opens `http://localhost:8090` in the desktop's default browser:

| Where it runs | What opens the page |
|---|---|
| WSL 2 on Windows | the Windows default browser (`wslview`, else `cmd.exe /c start`) |
| A Linux desktop (`DISPLAY` or `WAYLAND_DISPLAY` set) | `xdg-open` |
| A server with no display | nothing: the address is printed; over SSH, forward the port |

## What the page does

- **GPUs:** each card's memory, load, temperature and power, and the models running on it.
- **My models:** the models downloaded or converted on this machine, ready to start.
- **Find:** Hugging Face search by downloads, likes, update or trend, with a mark for what
  NInferno runs; each repository's quantizations with their size and the context, concurrency and
  speed they leave on your GPU.
- **Downloads:** progress of every download and conversion.
- **Start:** a model with the measured speculation, its context and parallel requests (1 to 8,
  the server's limit), its KV cache and its precision, then a chat window on it, or its
  OpenAI-compatible address for other apps.

## Precision

The model drawer's **Activation precision** control picks `ninferno-serve --activations`
([serving](serving.md#activation-precision)): **Auto** (what the converter recorded in the model),
**16** (16-bit everywhere: BF16, f16 on RTX 20 and V100; the answer does not depend on the load),
**8** or **4** bits for the values the compressed weights are multiplied with. The **KV cache**
beside it is a separate choice under every precision: under 16 a narrower KV cache stays narrower,
and the drawer says so (pick KV BF16 for full precision).

- **The server decides which modes run.** For a model converted on this machine the page asks
  `/api/precision`, which runs `ninferno-serve MODEL --activations-preflight CC` (the resolver
  alone: on the CPU, nothing loaded or downloaded) for the selected GPU's compute capability.
  Each mode is enabled or disabled by that answer, with the server's own refusal as the reason,
  and its tooltip counts the matrices at each width and names those left wider. A model not
  downloaded yet is checked after its download: until then the page judges from the weights'
  format (4 needs NVFP4 weights and an RTX 50, sm_120/121; 8 has no route on BF16, F16 or ternary
  GGUF weights; FP8 weights need compute capability 8.9: RTX 40/50, L4/L40 or H100) and marks the
  modes the name cannot settle *checked after download*.
- **A mode equal to Auto says so.** Where a mode resolves to the same kernels as Auto (as the
  server reports it; before the download, 8 on GGUF integer, MXFP4 and FP8 weights), it is marked
  *= Auto (same kernels)* and shows Auto's numbers. Where Auto is slower than 8 (NVFP4 weights
  without a calibrated divisor), the server's note says so above the table.
- **Numbers are lab measurements only.** A row per mode shows its quality against 16 bits (how
  often the next word matches; KL divergence and perplexity in the tooltip) and the decode and
  prefill of one request, from a lab measurement of the same board (matched by its PCI device id,
  never by name) and quantization, labelled *verified* with the card and the date. Without such a
  measurement the table is replaced by one line.
- **Memory follows the mode.** Under 16 the recurrent state of Qwen 3.5 models is FP32 (F16 under
  the other modes), so less context fits; the drawer gives both figures.
- A running model's row shows what its precision resolved to (`/v1/models` `meta.activations`):
  the matrices at each width, with a warning mark when some are kept wider.
- **Run** in *My models* starts a model with what the drawer last showed for it (precision, KV,
  context, parallel requests, speculation); a model never opened in the drawer starts with Auto
  and the defaults, as its tooltip says.
- The lab rows come from `src/ui/data/precision.json`, generated from
  `docs/performance/data/bench.jsonl` (`tools/bench/quality_rows.py` for quality,
  `tools/bench/perf_matrix.py --activations MODE` for speed) by
  `python3 tools/docs/gen_ui_precision.py` (`--check` verifies it), which adds each board's PCI
  ids from `src/core/data/gpu_facts.json`. A catalog from `--catalog-url` may carry its own
  `precision` list per model.

`/api/launch` takes the old names `exact`, `fast` and `int8` as `16`, `auto` and `8`, and the
model's log says they are deprecated.

The page has no on-machine calibration: speculation and routes come from the profiles and tables
measured in the lab, the same on every machine with the same card. A speculation tune calibrated on
this machine (`ninferno spec-tune`, or an older version of this page) is never used by the page:
when one lies beside a model, the model's log says so. Calibration is a command-line step:
`ninferno spec-tune MODEL` writes the profile
([maintainers](cli.md#measuring-speculation-profiles-maintainers)), and
`ninferno-serve MODEL --spec-profile PATH` runs with it.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--version` | | Print the version and exit. |
| `--listen ADDR` | `127.0.0.1` | Address to bind. On another address, other machines need the access key: the log prints `http://ADDR:PORT/?key=KEY`, which opens the page and keeps the key in a cookie (`ninferno_ui_key_PORT`: UIs on other ports keep their own); scripts send `Authorization: Bearer KEY`. Plain HTTP: use a trusted network, or put TLS in front; a proxy on this machine must send `X-Forwarded-For` (else its requests count as local), and the cookie reaches other services on the same host name (marked `Secure` when the proxy sends `X-Forwarded-Proto: https`). |
| `--port N` | `8090` | Port. |
| `--access-key-file F` | `$NINFERNO_UI_KEY`, else made once into `<hub cache>/ninferno-ui/access-key` | The access key: F's first line, 16–256 characters of `A-Z a-z 0-9 . _ ~ -`. |
| `--no-browser` | off | Do not open the page in a browser; only print its address. |
| `--serve PATH` | beside `ninferno-ui` | The `ninferno-serve` binary to start. |
| `--hub-cache DIR` | `$HF_HUB_CACHE`, `$HF_HOME/hub`, `$XDG_CACHE_HOME/huggingface/hub`, `~/.cache/huggingface/hub` | Where downloads go. |
| `--models-dir DIR` | none | Also list the `.ninferno` models in DIR (two levels deep); repeatable. |
| `--offline` | off | No Hugging Face listings; sizes come from estimates. |
| `--keep-originals` | off | Keep a model's download after it was converted (by default it is removed once the converted model loaded; nothing reads it again). |
| `--catalog-url URL` | `$NINFERNO_CATALOG_URL`, else built in | The recommended models from a `catalog.json` at URL, read at start and every 6 hours. |
| `--chat-ui DIR` | the bundled one | The chat window's files (llama.cpp's llama-ui). |

Behind a TLS proxy, give the UI a key (`NINFERNO_UI_KEY` or `--access-key-file F`): without one
it refuses every proxied request, since its default 127.0.0.1 listener makes no key of its own.
The proxy must forward the browser's `Host` header (the page refuses a
change whose `Origin` names another host than `Host`; `X-Forwarded-Host` is not trusted, since
any client can send it), and send `X-Forwarded-For` and `X-Forwarded-Proto`. nginx:

```nginx
location / {
    proxy_pass http://127.0.0.1:8090;
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

Caddy's `reverse_proxy` keeps `Host` and sends both headers by default
(`ui.example.com { reverse_proxy 127.0.0.1:8090 }`; spelled out: `header_up Host {hostport}`).
With `--listen 0.0.0.0` (or `::`) the log prints the URL with this machine's host name; where
that name does not resolve, use the machine's address.

The models the page starts inherit its environment: `NINFERNO_API_KEY` set for `ninferno-ui`
makes them ask for that key, and the page's proxy to them sends it. The page checks the release
manifest at start, at most once a day (`NINFERNO_UPDATE_CHECK=0` turns it off,
`NINFERNO_UPDATE_URL` names another manifest); an installer's updater is found as
`NINFERNO_UPDATER` or `ninferno-update` beside the binary.

The page talks only to its own server: requests must name its host and, for changes, come from
its own origin as JSON, so another web page in your browser cannot drive it.
