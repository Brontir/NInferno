# The release server (ninferno.brontir.com)

Installers and updaters read one static site:

```
/releases/latest.json                    the manifest (tools/release/manifest.py): newest version,
                                         min_version, notes, every file with its URL, size, SHA-256
/releases/v0.1.0/<file>                  a release's files, never changed once published
/releases/latest/download/<file>         the newest release's file (install.sh, the README's download
                                         links), through the releases/latest link to v<newest>
```

The files of a release: `NInferno-Setup.exe`, `install.sh`, `ninferno-wsl-rootfs.tar.gz`,
`ninferno-linux-x86_64.tar.gz` and `ninferno-windows-x86_64.zip` (each with an engine for every GPU
generation), each with its `.sha256`.

## Signing

Installers and updaters act on a manifest only with a valid signature of the release key:
`latest.json.sig` beside `latest.json` holds base64 of the DER ECDSA P-256 / SHA-256 signature of its
exact bytes, and every file is checked against the SHA-256 the signed manifest gives (`.sha256`
files on the server are not trusted). The public key is `tools/release/signing.pub.pem`, built into
`install.sh`, `ninferno-update` (through the installed `install.sh`), the release check of
`ninferno-serve` and `ninferno-ui`, and `NInferno-Setup.exe`; `ninferno_update_check_test` checks every
copy against it.

The private key never reaches this server or the repository: `tools/release/sign.sh` signs on the
machine that holds it. A server that is broken into can serve old or broken files, but it cannot
sign, so no installation acts on what it serves. The installer itself comes from GitHub
(`tools/release/install.sh` on master, `NInferno-Setup.exe` from the GitHub release), not from here.

## Publishing

```bash
tools/release/server/publish.sh 0.1.0 dist/ --notes https://github.com/Brontir/NInferno/releases/tag/v0.1.0 \
    [--min-version 0.1.2] [--host releases@ninferno.brontir.com]
```

uploads the release's files to `/releases/v0.1.0/`, checks them there, then switches `latest.json`, its
signature and the `releases/latest` link last (each by an atomic rename): a client that reads the new
manifest finds every file it names. Where the release key is not (a build machine), it writes
`latest.json` and stops; signing it with `sign.sh` where the key is and putting `latest.json.sig` beside
it lets the next run publish. Raising `--min-version` makes older installations ask to update at every start; use
it for a broken or unsafe release only.

## nginx

`nginx.conf` here is the site: static files, long caching for versioned files, no caching for the
manifest, and `latest/download` served through the link.
