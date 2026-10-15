#!/usr/bin/env bash
# Build the Neuroglancer fork that paints (briossant/neuroglancer, branch
# feature/voxel-annotation, upstream PR google/neuroglancer#858) at the commit
# web/neuroglancer.lock names, and put its browser client in web/ng/.
# vite.config.ts serves that directory at /ng/ in development (instead of
# relaying upstream Neuroglancer from appspot, which has no painting tools)
# and copies it into the build at ./ng/, the deployed site's /app/ng/.
#
# Idempotent: web/ng/.rev records the commit a build came from, so a rerun at
# the same commit does nothing; another commit is rebuilt. The build is
# Neuroglancer's own (rspack, through its build_tools/cli.ts):
# `npm run build -- --output=<dir>` writes a self-contained client whose
# index.html names its scripts by relative path (rspack's default publicPath,
# "auto", also resolves chunks, workers and WebAssembly from the running
# script's URL), so the client works from any subdirectory as built and
# nothing in it is rewritten. Its source maps are dropped: they are as large
# as the client itself and nobody debugs the viewer on the deployed site.
#
# The build happens in a temporary directory and is moved into web/ng/ only
# when it succeeds: vite.config.ts takes a web/ng/ that exists as a complete
# build, so a half-built one would leave development without a viewer.
# Needs Node >= 22.18 (the fork's engines field), git, and a few minutes:
# its npm ci installs rspack and friends, about 700 MB.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
source <(grep -E '^(repo|sha)=' "$here/neuroglancer.lock")
dest="$here/ng"
if [ -f "$dest/.rev" ] && [ "$(cat "$dest/.rev")" = "$sha" ] && [ -f "$dest/index.html" ]; then
  echo "neuroglancer already built at $sha in $dest"
  exit 0
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/neuroglancer.XXXXXX")
trap 'rm -rf "$work"' EXIT
src="$work/src"
build="$work/client"

# Shallow clone of just that commit, by its full SHA (GitHub serves it), as
# fetch-chunkmirage.sh does.
mkdir -p "$src"
git -C "$src" init -q
git -C "$src" remote add origin "$repo"
git -C "$src" fetch -q --depth 1 origin "$sha"
git -C "$src" checkout -q FETCH_HEAD
echo "neuroglancer $(git -C "$src" rev-parse --short HEAD) cloned; building the client"

# The fork's own build. Playwright is a devDependency for its tests, and its
# browser download is not needed to build.
(
  cd "$src"
  PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm ci --no-audit --no-fund
  npm run build -- --output="$build"
)
find "$build" -name '*.map' -delete
[ -f "$build/index.html" ] || { echo "the build produced no $build/index.html" >&2; exit 1; }

rm -rf "$dest"
mv "$build" "$dest"
echo "$sha" > "$dest/.rev"
echo "neuroglancer $sha built into $dest ($(du -sh "$dest" | cut -f1))"
