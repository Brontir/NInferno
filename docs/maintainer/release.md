# Making a release

A release is five files, each with a `.sha256`. They are built on two machines, published to the
release server, and the Windows files are also attached to the GitHub release:

| file | made by | where |
|---|---|---|
| `ninferno-linux-x86_64.tar.gz` | `tools/release/package.sh` | the Linux build machine |
| `ninferno-wsl-rootfs.tar.gz` | `tools/release/windows/make_rootfs.sh` | the Linux build machine (root) |
| `NInferno-Setup.exe` | `tools/release/windows/build.sh` | the Linux build machine (mingw-w64) |
| `ninferno-windows-x86_64.zip` | `tools/release/windows/package-native.ps1` | a Windows machine |
| `install.sh` | `tools/release/package.sh` (a copy of `tools/release/install.sh` at the tag) | the Linux build machine |

## The Linux build machine

The archive runs on the glibc it was built against or a newer one. `install.sh`'s `GLIBC_MIN` is
that version, and `package.sh` stops on a machine whose glibc is newer. The build machine is
therefore pinned:

- **Ubuntu 24.04** (glibc 2.39, `GLIBC_MIN`). `make_rootfs.sh` builds the WSL distribution on the
  same base (`UBUNTU=24.04`).
- **CUDA 13.4, nvcc V13.4.92**, for every engine but Volta's. The kernel-facts snapshots
  (`src/ops/generic/data/kernel_facts/`) record the nvcc they were made with, and `package.sh`
  stops on any other version.
- **CUDA 12.8, nvcc V12.8.93**, for the Volta engine (CUDA 13 dropped sm_70), at
  `/usr/local/cuda-12.8` or `NINFERNO_CUDA70`.

Install both toolkits from NVIDIA's apt repository for Ubuntu 24.04. Their component packages carry
nvcc's version, so the patch level can be pinned:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install cuda-toolkit-13-4 cuda-toolkit-12-8
/usr/local/cuda-13.4/bin/nvcc --version   # V13.4.92
/usr/local/cuda-12.8/bin/nvcc --version   # V12.8.93
```

Beside them: the build dependencies [building](../building.md) lists, `patchelf` (`package.sh`),
`g++-mingw-w64-x86-64-posix` (`windows/build.sh`), and `gh` and `openssl`
(`github_release.sh`). If the meta package has moved on to a newer patch, pin the compiler package:
`sudo apt install cuda-nvcc-13-4=13.4.92-1`.

**Never copy a toolkit from another machine.** A CUDA 12.8 taken from a distribution with glibc
2.41 or newer (Ubuntu 26.04, for one) carries that distribution's patch of
`crt/math_functions.h`: `sinpi`, `cospi` and `rsqrt` are declared `noexcept (true)`, as glibc 2.41
declares them. Against 24.04's glibc these declarations clash, and the Volta tree fails to compile
hours into the build. `package.sh` checks for this before it starts building.

`package.sh` downloads the FFmpeg and PDFium sources that `tools/release/ffmpeg.pin` and
`pdfium.pin` name, so the build needs network access.

## The Windows machine

Visual Studio 2022 Build Tools (C++), the CUDA toolkits of the same patch levels (13.4.92, and 12.8.93
for Volta: `-Cuda70`), CMake, Ninja and vcpkg; `package-native.ps1`'s header has the details.

## The checklist

1. **Master is green.** Every PR is merged with its full stand run. Then run one full
   `ctest` on master at the commit you will tag, on at least one stand. A batch of PRs that are
   each green can still break master together. The stands build sm_70, sm_75, sm_86, sm_89 and
   sm_120a, and each checks its own kernel-facts snapshot. For the other release architectures
   run `tools/dev/kernel_facts_archs.sh` on a build machine with CUDA 13.4.92: from a work
   directory (`KFA_ROOT`, default the current one) with `KFA_REPO` naming a checkout of this
   repository (`KFA_BRANCH` the branch, master by default), or `KFA_SOURCE=.` to check a source
   tree as it is. It builds sm_80, sm_90 and sm_100 with the release flags
   and checks each against its snapshot, without a GPU. Run it nightly too, and on a PR that
   changes `src/ops/**/*.cu*`, so that a stale snapshot shows before `package.sh` stops on it
   (#758).
2. **Release notes.** Write `docs/release-notes/v<VERSION>.md` and merge it.
3. **Tag.** Push the tag: `git tag -a v<VERSION> -m "NInferno <VERSION>" && git push origin v<VERSION>`.
   Everything below is built from a checkout of that tag.
4. **The Linux archive.** On the Linux build machine:
   `CUDACXX=/usr/local/cuda-13.4/bin/nvcc NINFERNO_VERSION=<VERSION> NINFERNO_PACKAGE_CLEAN=1 tools/release/package.sh dist`.
   It builds every generation and checks each engine against its kernel-facts snapshot. Where a
   GPU of a generation is present, it also checks the running kernels. It also writes `install.sh`
   and its `.sha256` into `dist/`.
5. **The WSL distribution.** `sudo tools/release/windows/make_rootfs.sh dist`.
6. **The installer.** `tools/release/windows/build.sh <VERSION> dist`.
7. **The Windows archive.** On the Windows machine:
   `tools\release\windows\package-native.ps1 -Version <VERSION> -OutDir dist -Cuda70 "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"`.
   Without `-Cuda70` the zip has no Volta engine; the script only warns. Copy
   `ninferno-windows-x86_64.zip` and its `.sha256` into the Linux machine's `dist/`.
8. **Publish to the release server.**
   `tools/release/server/publish.sh <VERSION> dist --notes https://github.com/Brontir/NInferno/blob/v<VERSION>/docs/release-notes/v<VERSION>.md`.
   The notes link is the notes file at the tag, so it works from the moment clients see the update
   (the GitHub release page stays a draft until step 9).
   - On a machine without the release key, it writes `dist/latest.json` and stops. Take that
     file to the machine that holds the key, sign it there (`tools/release/sign.sh latest.json`),
     and put the `latest.json.sig` it writes back beside `dist/latest.json`.
   - Run `publish.sh` again: it uploads the files, checks them on the server, and switches
     `latest.json` last ([server/README.md](../../tools/release/server/README.md)).
9. **The GitHub release.** `tools/release/github_release.sh <VERSION> dist` (`gh` signed in with
   write access to the repository).
   - It checks `latest.json`'s signature against the release key, the three Windows files against
     their `.sha256` and against that manifest, and that the tag is on origin.
   - It makes a draft release with the notes and the three Windows files plus their `.sha256`.
   - Look the draft over, then publish it: `gh release edit v<VERSION> --draft=false`.
   - A version with a label (0.2.0-rc1) is marked a pre-release.
10. **Check both destinations on clean machines.**
    - Linux: `curl -fsSL https://raw.githubusercontent.com/Brontir/NInferno/master/tools/release/install.sh | sh`
      installs `<VERSION>`, and `ninferno-update --check` on the release before says it is available.
    - Windows: the GitHub release's `NInferno-Setup.exe` installs it.

A published version is never changed; to fix a release, make a new version. For a broken or unsafe
release, also pass `--min-version` to `publish.sh`. `github_release.sh` refuses a release that is
already there. `publish.sh` refuses only the version `latest` already points at: run again for an
older version, it switches `latest` back to that one.
