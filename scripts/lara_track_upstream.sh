#!/usr/bin/env bash
# Move the fork's own commits (branch lara) onto the newest upstream release and run the fork's own tests.
#
#   scripts/lara_track_upstream.sh [--target <tag>] [--force] [--upstream <remote>] [--lara <ref>]
#
# Exit codes: 0 up to date or rebased and green (branch lara-<tag> created), 2 conflicts, 3 tests failed.
# The fork's own commits are base..lara, where base is the newest release tag lara contains. Commits upstream
# already carries are dropped by the rebase; the summary counts them so the fork shrinks.
set -euo pipefail

upstream=upstream
lara=lara
target=""
force=0
while [ $# -gt 0 ]; do
  case "$1" in
    --target) target=$2; shift ;;
    --force) force=1 ;;
    --upstream) upstream=$2; shift ;;
    --lara) lara=$2; shift ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
  shift
done

release_tags() { git tag --list 'v[0-9][0-9][0-9][0-9].[0-9]*.[0-9]*' --sort=-version:refname; }

git fetch --quiet --tags "$upstream"
base=""
for tag in $(release_tags); do
  if git merge-base --is-ancestor "$tag" "$lara"; then base=$tag; break; fi
done
[ -n "$base" ] || { echo "no upstream release tag is an ancestor of $lara" >&2; exit 64; }
[ -n "$target" ] || target=$(release_tags | head -1)

echo "base: $base  target: $target  fork commits: $(git rev-list --count "$base..$lara")"
if [ "$target" = "$base" ] && [ "$force" = 0 ]; then
  echo "up to date"
  exit 0
fi

branch="lara-$target"
git checkout --quiet -B "$branch" "$lara"
before=$(git rev-list --count "$base..$lara")
if ! git rebase --quiet --onto "$target" "$base" "$branch" >/dev/null; then
  echo "conflicts while moving $lara onto $target:"
  git diff --name-only --diff-filter=U
  git rebase --abort
  exit 2
fi
after=$(git rev-list --count "$target..$branch")
echo "moved $after of $before fork commits onto $target; $((before - after)) already upstream and dropped"

mapfile -t own_tests < <(git diff --name-only "$target..$branch" -- 'tests/**/test_*.py')
echo "fork tests: ${own_tests[*]:-none}"
[ ${#own_tests[@]} -eq 0 ] || python -m pytest -q -p no:cacheprovider "${own_tests[@]}" || exit 3
echo "green: $branch"
