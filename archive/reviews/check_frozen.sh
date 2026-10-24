#!/usr/bin/env bash
# Usage: bash reviews/check_frozen.sh <builder-commit> [base-ref]
#
# Compares base (default: main) with the builder's commit and fails (exit 1) if
#   1. a FROZEN file from plan/RULES.md changed (modified, added, deleted or renamed), or
#   2. any file outside the builder's ALLOWED write list changed, or
#   3. a file the builder may only extend (hands_on.py, progress.md, debug_cheatsheet.md) lost lines.
# Exit 2 = usage / git error.
#
# The comparison is base...commit (changes since the common ancestor), so commits that landed
# on main after the builder branched are not blamed on the builder. If the builder is not based on
# the current tip of main, the script says so (informational, not a failure).

set -u

COMMIT="${1:-}"
BASE="${2:-main}"
if [ -z "$COMMIT" ]; then
  echo "usage: bash reviews/check_frozen.sh <builder-commit> [base-ref]" >&2
  exit 2
fi
if ! git rev-parse --verify --quiet "${COMMIT}^{commit}" >/dev/null; then
  echo "error: '$COMMIT' is not a commit in this repository" >&2
  exit 2
fi
if ! git rev-parse --verify --quiet "${BASE}^{commit}" >/dev/null; then
  echo "error: base ref '$BASE' not found" >&2
  exit 2
fi
COMMIT_FULL=$(git rev-parse "${COMMIT}^{commit}")
BASE_FULL=$(git rev-parse "${BASE}^{commit}")
MERGE_BASE=$(git merge-base "$BASE_FULL" "$COMMIT_FULL" 2>/dev/null || true)

echo "base   : $BASE ($BASE_FULL)"
echo "commit : $COMMIT_FULL"
if [ -z "$MERGE_BASE" ]; then
  echo "error: no common ancestor between $BASE and $COMMIT" >&2
  exit 2
fi
if [ "$MERGE_BASE" != "$BASE_FULL" ]; then
  echo "note   : builder commit is not based on the current tip of $BASE (merge-base $MERGE_BASE)."
  echo "         Only changes since the merge-base are checked."
fi
echo

# --- policy: keep in sync with plan/RULES.md --------------------------------------------------
# Frozen: exact paths, plus whole directories (trailing slash).
FROZEN_FILES="
prototype/trained_lstm.npz
prototype/training_data.py
prototype/scorecard_data.py
prototype/holdout_data.py
prototype/minutes_tests.py
prototype/normalise_tests.py
prototype/lstm_model.py
prototype/train_lstm.py
prototype/meeting_minutes.py
prototype/transcriber.py
prototype/hands_on_reference.py
"
FROZEN_DIRS="
testing/
"
# Allowed write list. Patterns are matched with bash `case` globs (so * also crosses '/').
# HANDOFF-WP*.md is the handoff the work packages require; plan/ and reviews/ are not allowed.
ALLOWED_PATTERNS="
notebook/*
demo/*
prototype/demo.py
progress.md
debug_cheatsheet.md
prototype/hands_on.py
HANDOFF-WP*.md
"
# Additive only: any deleted line is a failure (the reviewer then judges it by hand).
ADDITIVE_ONLY="
prototype/hands_on.py
progress.md
debug_cheatsheet.md
"
# ----------------------------------------------------------------------------------------------

in_list() { # in_list <item> <newline-separated list>
  local item="$1" line
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    [ "$line" = "$item" ] && return 0
  done <<EOF
$2
EOF
  return 1
}

is_frozen() {
  local p="$1" d
  in_list "$p" "$FROZEN_FILES" && return 0
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    case "$p" in "$d"*) return 0 ;; esac
  done <<EOF
$FROZEN_DIRS
EOF
  return 1
}

is_allowed() {
  local p="$1" pat
  while IFS= read -r pat; do
    [ -z "$pat" ] && continue
    # shellcheck disable=SC2254
    case "$p" in $pat) return 0 ;; esac
  done <<EOF
$ALLOWED_PATTERNS
EOF
  return 1
}

# --no-renames: a rename shows as delete + add, so renaming a frozen file out is caught as well.
CHANGED=$(git diff --name-only --no-renames -z "$BASE_FULL...$COMMIT_FULL" | tr '\0' '\n')

if [ -z "$CHANGED" ]; then
  echo "no files differ between $BASE and the commit"
  echo
  echo "RESULT: OK (nothing changed)"
  exit 0
fi

echo "files changed since merge-base:"
printf '%s\n' "$CHANGED" | sed 's/^/  /'
echo

FROZEN_HITS=""
OUTSIDE_HITS=""
while IFS= read -r f; do
  [ -z "$f" ] && continue
  if is_frozen "$f"; then
    FROZEN_HITS="${FROZEN_HITS}${f}"$'\n'
  elif ! is_allowed "$f"; then
    OUTSIDE_HITS="${OUTSIDE_HITS}${f}"$'\n'
  fi
done <<EOF
$CHANGED
EOF

FAIL=0

if [ -n "$FROZEN_HITS" ]; then
  FAIL=1
  echo "FAIL frozen files that differ from $BASE:"
  printf '%s' "$FROZEN_HITS" | sed 's/^/  /'
  echo
else
  echo "frozen: none changed"
fi

if [ -n "$OUTSIDE_HITS" ]; then
  FAIL=1
  echo "FAIL files outside the builder's allowed write list:"
  printf '%s' "$OUTSIDE_HITS" | sed 's/^/  /'
  echo
else
  echo "outside allowed list: none"
fi

# additive-only files: report deleted-line counts
while IFS= read -r f; do
  [ -z "$f" ] && continue
  if printf '%s\n' "$CHANGED" | grep -qxF "$f"; then
    # numstat: added<TAB>deleted<TAB>path ("-" for binary)
    read -r added deleted _ < <(git diff --numstat --no-renames "$BASE_FULL...$COMMIT_FULL" -- "$f")
    if [ "$deleted" != "0" ]; then
      FAIL=1
      echo "FAIL $f is additive-only but $deleted line(s) were removed or changed (added: $added)"
    else
      echo "$f: additive only (added $added, deleted 0)"
    fi
  fi
done <<EOF
$ADDITIVE_ONLY
EOF

echo
if [ "$FAIL" -ne 0 ]; then
  echo "RESULT: FAIL"
  exit 1
fi
echo "RESULT: OK"
exit 0
