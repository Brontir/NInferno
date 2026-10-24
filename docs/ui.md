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
- **My models:** the models downloaded or converted on this machine, ready to start. A
  converted hub model whose branch has moved on the hub since (its download removed or still in
  the cache) is marked "newer weights on the hub", and a popup asks once per new commit. It still
  starts from its converted weights; nothing downloads by itself. The choices:
  - **Keep these:** treat them as valid from now on (no mark until the branch moves again).
  - **Download new:** download and convert the new commit in the background, beside the old
    conversion. A model running on the old conversion restarts on the new one once, when it has no
    request in flight; then the old conversion is deleted. If the new one does not start, the
    model starts on the old one once more and the update ends, saying why; nothing is retried.
  - **Later:** ask again at the next new commit.

  The hub is asked at most hourly per model, never with `--offline`.
- **Find:** Hugging Face search by downloads, likes, update or trend, with a mark for what
  NInferno runs; each repository's quantizations with their size and the context, concurrency and
  speed they leave on your GPU. The speeds are the engine's estimates from the card's and the
  model's facts ([Configuration choice](maintainer/configuration-choice.md)), not measurements.
  Under the pick, "Engine's choice" says which KV cache, activations and state a start would take
  when you leave them automatic, and "What another value would change" lists, for every other
  value, the context each stream keeps, decode, prefill, a typical request's time, the accuracy
  (KLD) and why it was not chosen.
- **Downloads:** progress of every download and conversion.
- **Start:** a model with the measured speculation and its precision (exact output or the
  faster default), then a chat window on it, or its OpenAI-compatible address for other apps.
- **Calibrate (⚙):** measures speculation for a converted model on the chosen GPU
  (`ninferno spec-tune`, about two minutes) and writes `<model>.ninferno.spec_profile.json` beside
  it. A measurement of your machine is never used by itself: the page's Start does not use it,
  and a start names it (`--spec-profile`) only when it asks for it (`POST /api/launch` with
  `"spec_profile": true`). Asked for, a tune that does not serve on that GPU (made for another
  artifact, GPU class or drafters) refuses the start: the UI answers with the reason, or, on a GPU
  whose class it cannot name (a MIG slice), `ninferno-serve` refuses to start and its log says why.

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
