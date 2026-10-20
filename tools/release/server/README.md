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

## Publishing

```bash
tools/release/server/publish.sh 0.1.0 dist/ --notes https://github.com/Brontir/NInferno/releases/tag/v0.1.0 \
    [--min-version 0.1.2] [--host releases@ninferno.brontir.com]
```

uploads the release's files to `/releases/v0.1.0/`, checks them there, then switches `latest.json` and the
`releases/latest` link last (each by an atomic rename): a client that reads the new manifest finds every
file it names. Raising `--min-version` makes older installations ask to update at every start; use
it for a broken or unsafe release only.

## nginx

`nginx.conf` here is the site: static files, long caching for versioned files, no caching for the
manifest, and `latest/download` served through the link.
