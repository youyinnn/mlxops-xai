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

CURRENT=$(python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['version'])")

# Compute suggested next versions
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"
NEXT_PATCH="${MAJOR}.${MINOR}.$((PATCH + 1))"
NEXT_MINOR="${MAJOR}.$((MINOR + 1)).0"
NEXT_MAJOR="$((MAJOR + 1)).0.0"

echo "Current version: ${CURRENT}"
echo ""
echo "  1) patch  →  ${NEXT_PATCH}"
echo "  2) minor  →  ${NEXT_MINOR}"
echo "  3) major  →  ${NEXT_MAJOR}"
echo "  4) custom"
echo ""
read -rp "Choose [1-4]: " choice

case "$choice" in
  1) VERSION="$NEXT_PATCH" ;;
  2) VERSION="$NEXT_MINOR" ;;
  3) VERSION="$NEXT_MAJOR" ;;
  4)
    read -rp "Enter version: " VERSION
    if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "Error: version must be in X.Y.Z format." >&2
      exit 1
    fi
    ;;
  *)
    echo "Invalid choice." >&2
    exit 1
    ;;
esac

echo ""
echo "Bumping ${CURRENT} → ${VERSION} in pyproject.toml..."
python -c "
import re, pathlib
p = pathlib.Path('pyproject.toml')
p.write_text(re.sub(r'^(version\s*=\s*)\"${CURRENT}\"', r'\1\"${VERSION}\"', p.read_text(), count=1, flags=re.MULTILINE))
"

# Clean previous builds
rm -rf dist/

# Build sdist and wheel
echo "Building v${VERSION}..."
hatch build

# Confirm before publishing
echo ""
ls -lh dist/
echo ""
read -rp "Publish the above artifacts to ${TARGET}? [y/N] " confirm
if [[ "$(echo "$confirm" | tr '[:upper:]' '[:lower:]')" != "y" ]]; then
  echo "Aborted. pyproject.toml has been updated to ${VERSION} but nothing was uploaded."
  exit 0
fi

if [[ "$TARGET" == "testpypi" ]]; then
  twine upload --repository testpypi dist/*
else
  twine upload dist/*
fi

echo "Done. https://pypi.org/project/mlxops-xai/${VERSION}/"
