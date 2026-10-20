# NInferno on Windows

## Install

Download [**NInferno-Setup.exe**](https://github.com/Brontir/NInferno/releases/latest/download/NInferno-Setup.exe)
and run it. It needs no administrator rights and builds nothing:

1. It reads your GPUs (`nvidia-smi`) and shows them. An NVIDIA driver must be installed
   ([nvidia.com/drivers](https://www.nvidia.com/drivers)); every GPU from compute capability 7.0 runs:
   GeForce RTX 20 to 50, the workstation (Quadro, RTX A/Ada/PRO) and datacenter (Tesla V100, T4, A100,
   A10, L4, L40, H100, B200, ...) cards, several of them in one machine.
2. It downloads NInferno (one archive for every GPU), checks it against the signed release
   manifest, and installs it under
   `%LOCALAPPDATA%\Programs\NInferno`. The engine carries the Visual C++ runtime it needs: no
   redistributable to install.
3. It adds NInferno to the Start menu and the desktop, and to *Settings → Apps → Installed apps*.

Start **NInferno** from the Start menu: the page opens in your browser (`http://localhost:8090`), and an
icon in the notification area opens it again, checks for updates, or quits (which stops running models).
`NInferno.exe --launch --no-browser` starts the icon and the page without opening the browser (a start
at sign-in, say).
Models download to `%USERPROFILE%\.cache\huggingface\hub`, the cache Hugging Face's own tools use.

### Native engine or WSL 2

The release carries the native Windows engine (`ninferno-windows-x86_64.zip`): an engine per GPU
generation behind one `ninferno-serve.exe`, which starts the one for the GPU a model runs on: RTX 20
to 50 and their workstation cards, V100, A100, and H100/H200 and B200/B300 (these two built, not
measured) have their own; a GPU newer than the release runs Ada's PTX, which the driver compiles at
its first start (up to half an hour, minutes after that; NInferno sets the driver's compile cache,
`CUDA_CACHE_MAXSIZE`, to 4 GiB unless it is set, so the result is kept). A release without the
native archive runs the Linux engine in WSL 2 instead, in a distribution of its own named `NInferno`
(not your Ubuntu: no Linux account to create, nothing from the Microsoft Store). Turning WSL on is
the one step that asks for administrator rights, and Windows may want a restart; the installation
continues by itself after it. `NINFERNO_FORCE_WSL=1` makes the installer take WSL even where a
native build exists.

## Updates

NInferno checks the release manifest (`https://ninferno.brontir.com/releases/latest.json`) when it
starts and once a day, and never waits for it:

- a newer release shows a notification, and a bar on the page with **Update**; one click downloads the
  new installer (its SHA-256 checked against the manifest), which installs the new engine beside the
  old one and starts NInferno again; your models stay;
- a release older than the manifest's `min_version` is out of date: NInferno asks to update at every
  start.

`NINFERNO_UPDATE_CHECK=0` (or `off`, `no`, `false`) turns the check off; `NINFERNO_UPDATE_URL`
points it at another manifest.

## Uninstall

*Settings → Apps → Installed apps → NInferno → Uninstall* (or `NInferno.exe --uninstall`). The native
engine leaves the downloaded models in the Hugging Face cache, which other tools may share; delete
`%USERPROFILE%\.cache\huggingface\hub` yourself to free that space. With WSL, the `NInferno`
distribution goes, and the models in it with it.

## Command line

The native engine's programs are in `%LOCALAPPDATA%\Programs\NInferno\engine\<release>\bin`:
`ninferno-serve.exe`, `ninferno-convert.exe` and `ninferno-ui.exe` take the same options as on Linux
(see [serving](serving.md)). `ninferno-serve.exe unsloth/Qwen3.6-35B-A3B-GGUF --quant UD-Q4_K_XL`
downloads, converts and serves a model on `http://localhost:8080`.

## Build from source

Needs Visual Studio 2022 Build Tools (the C++ workload), the CUDA Toolkit 13.x (12.8 for V100), CMake,
Ninja and [vcpkg](https://github.com/microsoft/vcpkg) (the libraries Linux takes from the distribution:
FFmpeg, curl, OpenSSL; `vcpkg.json` names them). Then, in PowerShell:

```powershell
tools\release\windows\package-native.ps1 -Vcpkg C:\vcpkg -Version 0.1.0
```

builds an engine per GPU generation (`-Archs`, default `120a,100,90,89,86,80,75`: Blackwell RTX,
datacenter Blackwell, Hopper, Ada with PTX, Ampere, Turing; `-Cuda70` with CUDA 12.8 adds Volta), each in its own tree, and packs them with
the dispatcher into `dist\ninferno-windows-x86_64.zip`. The installer itself
(`tools/release/windows`) is a small Win32 program cross-compiled on Linux with mingw-w64:
`tools/release/windows/build.sh VERSION`.

### What differs from Linux

- Files: positional reads and writes, `O_DIRECT` (`FILE_FLAG_NO_BUFFERING`, the same 4096-byte
  alignment), file views for mappings, and locks go through `src/core/sys.h`; page-cache advice is a
  no-op, and Windows journals directory entries itself.
- Processes (the page's models): each runs in a job object, so its whole tree ends with it and with the
  page; *Stop* sends Ctrl+Break, which `ninferno-serve` takes as SIGTERM (it drains), and ends the job
  after the grace period.
- Executables carry a manifest that makes the process code page UTF-8 (non-ASCII user names and paths).
- Hugging Face cache links: Windows allows symbolic links only in Developer Mode; without it the cache
  holds hard links to its blobs. A cache that WSL downloaded holds Linux symbolic links, which Windows
  programs cannot open: give the native engine a cache of its own (the default) or the model file.
