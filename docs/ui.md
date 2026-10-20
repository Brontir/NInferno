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
- **Start:** a model with the measured speculation and its precision (exact output or the
  faster default), then a chat window on it, or its OpenAI-compatible address for other apps.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--version` | | Print the version and exit. |
| `--listen ADDR` | `127.0.0.1` | Address to bind. On another address, other machines need the access key: the log prints `http://ADDR:PORT/?key=KEY`, which opens the page and keeps the key in a cookie; scripts send `Authorization: Bearer KEY`. Plain HTTP: use a trusted network, or put TLS in front. |
| `--port N` | `8090` | Port. |
| `--access-key-file F` | `$NINFERNO_UI_KEY`, else made once into `<hub cache>/ninferno-ui/access-key` | The access key: F's first line, 16+ characters. |
| `--no-browser` | off | Do not open the page in a browser; only print its address. |
| `--serve PATH` | beside `ninferno-ui` | The `ninferno-serve` binary to start. |
| `--hub-cache DIR` | `$HF_HUB_CACHE`, `$HF_HOME/hub`, `$XDG_CACHE_HOME/huggingface/hub`, `~/.cache/huggingface/hub` | Where downloads go. |
| `--models-dir DIR` | none | Also list the `.ninferno` models in DIR (two levels deep); repeatable. |
| `--offline` | off | No Hugging Face listings; sizes come from estimates. |
| `--keep-originals` | off | Keep a model's download after it was converted (by default it is removed once the converted model loaded; nothing reads it again). |
| `--catalog-url URL` | `$NINFERNO_CATALOG_URL`, else built in | The recommended models from a `catalog.json` at URL, read at start and every 6 hours. |
| `--chat-ui DIR` | the bundled one | The chat window's files (llama.cpp's llama-ui). |

The models the page starts inherit its environment: `NINFERNO_API_KEY` set for `ninferno-ui`
makes them ask for that key, and the page's proxy to them sends it. The page checks the release
manifest at start, at most once a day (`NINFERNO_UPDATE_CHECK=0` turns it off,
`NINFERNO_UPDATE_URL` names another manifest); an installer's updater is found as
`NINFERNO_UPDATER` or `ninferno-update` beside the binary.

The page talks only to its own server: requests must name its host and, for changes, come from
its own origin as JSON, so another web page in your browser cannot drive it.
