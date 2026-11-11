#!/usr/bin/env bash
# github_release_test.sh REPO: tools/release/github_release.sh makes a GitHub release only of the
# files the signed manifest names, only from a tag already on origin, never over a release that is
# there, as a draft unless --publish, a pre-release for a version with a label. gh and git are
# stand-ins that record their arguments; nothing reaches GitHub. Needs python3; skipped (77)
# without it.
set -euo pipefail
REPO=$(cd "${1:?usage: github_release_test.sh REPO}" && pwd)
command -v python3 > /dev/null && command -v openssl > /dev/null || exit 77
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
fail() { echo "FAIL: $*"; failures=$((failures + 1)); }
expect() { # what  status-wanted  command...
  local what=$1 want=$2 got=0
  shift 2
  : > "$work/gh.log"
  "$@" > "$work/out.txt" 2>&1 < /dev/null || got=$?
  if { [ "$want" = 0 ] && [ "$got" != 0 ]; } || { [ "$want" != 0 ] && [ "$got" = 0 ]; }; then
    fail "$what (exit $got)"; sed 's/^/  /' "$work/out.txt"
  fi
}
said() { grep -qF -- "$1" "$work/out.txt" || { fail "the output does not say '$1'"; sed 's/^/  /' "$work/out.txt"; }; }
created() { grep -q '^release create' "$work/gh.log"; }

# A copy of the repository's release tools with its own notes, so the notes' presence is the test's.
root=$work/repo
mkdir -p "$root/tools/release" "$root/docs/release-notes"
cp "$REPO/tools/release/github_release.sh" "$root/tools/release/"
# The copy's release key is one made here (the release key's private half is never in the
# repository).
openssl ecparam -name prime256v1 -genkey -noout -out "$work/key.pem" 2> /dev/null
openssl ec -in "$work/key.pem" -pubout -out "$root/tools/release/signing.pub.pem" 2> /dev/null
sign() { openssl dgst -sha256 -sign "$work/key.pem" "$1" | base64 -w0 > "$1.sig"; }
echo "# notes" > "$root/docs/release-notes/v0.2.0.md"
echo "# notes" > "$root/docs/release-notes/v0.3.0-rc1.md"
mkdir -p "$work/bin"
# gh: `release view` finds the releases listed in $work/existing; `release create` is recorded.
cat > "$work/bin/gh" << EOF
#!/bin/sh
echo "\$*" >> "$work/gh.log"
[ "\$(pwd)" = "$work/repo" ] || { echo "gh run outside the checkout: \$(pwd)" >&2; exit 9; }
if [ "\$1 \$2" = "release view" ]; then
  [ -f "$work/gh-down" ] && { echo "error connecting to api.github.com" >&2; exit 1; }
  grep -qx "\$3" "$work/existing" 2> /dev/null && exit 0
  echo "release not found" >&2; exit 1
fi
exit 0
EOF
# git: ls-remote finds the tags listed in $work/tags.
cat > "$work/bin/git" << EOF
#!/bin/sh
for a; do case \$a in refs/tags/*) grep -qx "\${a#refs/tags/}" "$work/tags" 2> /dev/null; exit; esac; done
exit 2
EOF
chmod +x "$work/bin/gh" "$work/bin/git"
export PATH="$work/bin:$PATH"
printf 'v0.2.0\nv0.3.0-rc1\n' > "$work/tags"

dist() { # VERSION DIR: the three files, their .sha256 and a manifest naming them
  mkdir -p "$2"
  for f in NInferno-Setup.exe ninferno-windows-x86_64.zip ninferno-wsl-rootfs.tar.gz; do
    echo "$f $1" > "$2/$f"
    (cd "$2" && sha256sum "$f" > "$f.sha256")
  done
  python3 "$REPO/tools/release/manifest.py" --version "$1" --base "https://releases.invalid/v$1" "$2" > "$2/latest.json"
  sign "$2/latest.json"
}
dist 0.2.0 "$work/d2"
dist 0.3.0-rc1 "$work/drc"
G=("$root/tools/release/github_release.sh")

expect "a draft release" 0 "${G[@]}" 0.2.0 "$work/d2"
grep -q -- '^release create v0.2.0 --verify-tag --title NInferno 0.2.0 --notes-file .*/v0.2.0.md --draft ' "$work/gh.log" ||
  { fail "the draft is not made as expected"; cat "$work/gh.log"; }
for f in NInferno-Setup.exe ninferno-windows-x86_64.zip ninferno-wsl-rootfs.tar.gz; do
  grep -q "$work/d2/$f $work/d2/$f.sha256" "$work/gh.log" || fail "$f is not uploaded with its .sha256"
done
! grep -q -- --prerelease "$work/gh.log" || fail "0.2.0 was marked a pre-release"
expect "--publish" 0 "${G[@]}" 0.2.0 "$work/d2" --publish
! grep -q -- --draft "$work/gh.log" || fail "--publish made a draft"
expect "a pre-release" 0 "${G[@]}" 0.3.0-rc1 "$work/drc"
grep -q -- --prerelease "$work/gh.log" || fail "0.3.0-rc1 was not marked a pre-release"

# Refusals: nothing is created.
refused() { # what  expected-text  command...
  local what=$1 text=$2; shift 2
  expect "$what" 1 "$@"; said "$text"; ! created || fail "$what: a release was created"
}
refused "a version that is no release" "is not a release version" "${G[@]}" 0.2 "$work/d2"
refused "no notes" "no release notes" "${G[@]}" 0.4.0 "$work/d2"
refused "an unknown argument" "unknown argument" "${G[@]}" 0.2.0 "$work/d2" --yes
refused "another version's files" "is the manifest of 0.2.0, not 0.3.0-rc1" "${G[@]}" 0.3.0-rc1 "$work/d2"
cp -r "$work/d2" "$work/changed"
echo changed > "$work/changed/ninferno-windows-x86_64.zip"
refused "a file changed after its .sha256" "ninferno-windows-x86_64.zip does not match its .sha256" "${G[@]}" 0.2.0 "$work/changed"
(cd "$work/changed" && sha256sum ninferno-windows-x86_64.zip > ninferno-windows-x86_64.zip.sha256)
refused "a file the manifest does not name" "is not the file latest.json names" "${G[@]}" 0.2.0 "$work/changed"
cp -r "$work/d2" "$work/missing" && rm "$work/missing/ninferno-wsl-rootfs.tar.gz"
refused "a missing file" "has no ninferno-wsl-rootfs.tar.gz" "${G[@]}" 0.2.0 "$work/missing"
cp -r "$work/d2" "$work/unsigned" && rm "$work/unsigned/latest.json.sig"
refused "an unsigned manifest" "sign latest.json first" "${G[@]}" 0.2.0 "$work/unsigned"
cp -r "$work/d2" "$work/forged" && sed -i 's/"min_version": ""/"min_version": "0.1.0"/' "$work/forged/latest.json"
refused "a manifest changed after it was signed" "is not signed by the release key" "${G[@]}" 0.2.0 "$work/forged"
touch "$work/gh-down"
refused "GitHub that cannot be asked" "cannot ask GitHub whether v0.2.0 has a release" "${G[@]}" 0.2.0 "$work/d2"
rm "$work/gh-down"
printf 'v0.3.0-rc1\n' > "$work/tags"
refused "a tag not on origin" "the tag v0.2.0 is not on origin" "${G[@]}" 0.2.0 "$work/d2"
printf 'v0.2.0\n' > "$work/tags"
echo v0.2.0 > "$work/existing"
refused "a release that is there" "already exists" "${G[@]}" 0.2.0 "$work/d2"

[ "$failures" = 0 ] && echo "github release: ok" || { echo "$failures failure(s)"; exit 1; }
