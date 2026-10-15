#!/usr/bin/env bash
# Clone chunkmirage at the commit web/chunkmirage.lock names into web/chunkmirage/,
# so this app can import its browser engine (web/chunkmirage/web/src/*.ts) as
# source. Idempotent: a clone already at that commit is left alone; another
# commit is re-fetched. A clone script rather than a git submodule: it needs
# no .gitmodules, works the same in CI and locally, and the pin is one line.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
source <(grep -E '^(repo|sha)=' "$here/chunkmirage.lock")
dest="$here/chunkmirage"
if [ -d "$dest/.git" ] && git -C "$dest" cat-file -e "$sha^{commit}" 2>/dev/null \
   && [ "$(git -C "$dest" rev-parse HEAD)" = "$(git -C "$dest" rev-parse "$sha^{commit}")" ]; then
  echo "chunkmirage already at $sha in $dest"
  exit 0
fi
rm -rf "$dest"
mkdir -p "$dest"
git -C "$dest" init -q
git -C "$dest" remote add origin "$repo"
git -C "$dest" fetch -q --depth 1 origin "$sha"
git -C "$dest" checkout -q FETCH_HEAD
echo "chunkmirage $(git -C "$dest" rev-parse --short HEAD) cloned into $dest"
