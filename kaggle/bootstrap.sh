#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: bootstrap.sh <source-directory> <working-directory> <synthetic|real> [mounted-data-root]" >&2
  exit 2
fi

source_dir=$(realpath "$1")
work_dir=$(realpath -m "$2")
suite="$3"
input_root="${4:-}"

case "$suite" in
  synthetic|real) ;;
  *) echo "suite must be synthetic or real" >&2; exit 2 ;;
esac

if [[ ! -f "$source_dir/experiments/kaggle_runner.py" ]]; then
  echo "source directory must contain experiments/kaggle_runner.py" >&2
  exit 2
fi
is_same_or_child() {
  local candidate="$1"
  local parent="${2%/}"
  if [[ -z "$parent" ]]; then
    parent="/"
  fi
  if [[ "$parent" == "/" ]]; then
    [[ "$candidate" == /* ]]
  else
    [[ "$candidate" == "$parent" || "$candidate" == "$parent/"* ]]
  fi
}

if is_same_or_child "$work_dir" "$source_dir" || is_same_or_child "$source_dir" "$work_dir"; then
  echo "source and working roots overlap" >&2
  exit 2
fi
if [[ "$suite" == real && ( -z "$input_root" || ! -d "$input_root" ) ]]; then
  echo "real suite requires an existing mounted-data-root" >&2
  exit 2
fi

mkdir -p "$work_dir/repo" "$work_dir/results"
# Keep immutable/private source data mounted from /kaggle/input; only stage code.
tar --exclude='./data' \
  --exclude='./.git' \
  --exclude='./.pytest_cache' \
  --exclude='*/__pycache__' \
  -cf - -C "$source_dir" . | tar -xf - -C "$work_dir/repo"

if [[ ! -f "$work_dir/repo/requirements.txt" ]]; then
  echo "source bundle must contain requirements.txt" >&2
  exit 2
fi
# Kaggle images ship a CUDA-compatible Torch build. Keep shared dependencies
# while leaving the preinstalled Torch version untouched.
kaggle_requirements="$work_dir/requirements-kaggle.txt"
grep -Ev '^(--extra-index-url|torch==)' "$work_dir/repo/requirements.txt" \
  > "$kaggle_requirements"
if [[ -n "${RECSYS_KAGGLE_WHEELHOUSE:-}" ]]; then
  python -m pip install --no-index --find-links "$RECSYS_KAGGLE_WHEELHOUSE" \
    -r "$kaggle_requirements"
else
  python -m pip install -r "$kaggle_requirements"
fi

cd "$work_dir/repo"
args=(--repo-root "$work_dir/repo" --working-root "$work_dir/results" --suite "$suite")
if [[ "$suite" == real ]]; then
  args+=(--input-data-root "$input_root")
fi
python -m experiments.kaggle_runner "${args[@]}"
