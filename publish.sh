#!/usr/bin/env bash
set -euo pipefail

# Publish mlxops-xai to PyPI
# Usage:
#   ./publish.sh          — publish to PyPI (production)
#   ./publish.sh --test   — publish to TestPyPI first

TARGET="pypi"
if [[ "${1:-}" == "--test" ]]; then
  TARGET="testpypi"
fi

# Check required tools
for cmd in python hatch twine; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "Error: '$cmd' is not installed." >&2
    exit 1
  fi
done

VERSION=$(python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['version'])")
echo "Publishing mlxops-xai v${VERSION} to ${TARGET}..."

# Clean previous builds
rm -rf dist/

# Build sdist and wheel
hatch build

# Confirm before publishing
echo ""
ls -lh dist/
echo ""
read -rp "Publish the above artifacts to ${TARGET}? [y/N] " confirm
if [[ "$(echo "$confirm" | tr '[:upper:]' '[:lower:]')" != "y" ]]; then
  echo "Aborted."
  exit 0
fi

if [[ "$TARGET" == "testpypi" ]]; then
  twine upload --repository testpypi dist/*
else
  twine upload dist/*
fi

echo "Done. https://pypi.org/project/mlxops-xai/${VERSION}/"
