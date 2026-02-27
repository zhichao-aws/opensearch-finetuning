#!/usr/bin/env bash
set -euo pipefail

REPO="zhichao-aws/opensearch-finetuning"
TAG="${1:-v1.0.0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "==> Building lambda zips and training archive..."

# Lambda zips — each directory under lambdas/ becomes <dir-name>-lambda.zip
for dir in "$SCRIPT_DIR"/lambdas/*/; do
    name="$(basename "$dir")"
    zip_name="${name}-lambda.zip"
    echo "  Packaging $zip_name"
    (cd "$dir" && zip -r "$BUILD_DIR/$zip_name" . -x '*.pyc' '__pycache__/*')
done

# Training script tarball — flat files (no top-level directory)
echo "  Packaging training-script.tar.gz"
tar czf "$BUILD_DIR/training-script.tar.gz" -C "$SCRIPT_DIR/training-script" .

echo ""
echo "==> Build artifacts in $BUILD_DIR:"
ls -lh "$BUILD_DIR"

echo ""
echo "==> Uploading to GitHub release $TAG ..."

# Create the release if it doesn't exist yet; ignore error if it already exists
gh release create "$TAG" --repo "$REPO" --title "$TAG" --notes "Release $TAG" 2>/dev/null || true

# Upload (overwrite existing assets with --clobber)
gh release upload "$TAG" \
    "$BUILD_DIR/data-extractor-lambda.zip" \
    "$BUILD_DIR/s3-validator-lambda.zip" \
    "$BUILD_DIR/bedrock-orchestrator-lambda.zip" \
    "$BUILD_DIR/register-model-lambda.zip" \
    "$BUILD_DIR/training-script.tar.gz" \
    --repo "$REPO" \
    --clobber

echo ""
echo "==> Done! Release: https://github.com/$REPO/releases/tag/$TAG"
