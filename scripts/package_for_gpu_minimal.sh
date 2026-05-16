#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="../gpu_transfer"
PACKAGE_NAME="mt_nllb_multilingual_gpu_minimal.tar.gz"
STAGE_DIR="../gpu_transfer/mt_nllb_multilingual_gpu_minimal"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

echo "Staging minimal GPU package..."

mkdir -p "$STAGE_DIR/src"
mkdir -p "$STAGE_DIR/configs"
mkdir -p "$STAGE_DIR/data/processed/jav_eng"
mkdir -p "$STAGE_DIR/data/processed/joint"

cp -R src/*.py "$STAGE_DIR/src/"
cp requirements.txt "$STAGE_DIR/requirements.txt"

cp configs/java_only.yaml "$STAGE_DIR/configs/"
cp configs/joint_balanced.yaml "$STAGE_DIR/configs/"
cp configs/zero_shot.yaml "$STAGE_DIR/configs/"

cp data/processed/jav_eng/train.jsonl "$STAGE_DIR/data/processed/jav_eng/"
cp data/processed/jav_eng/dev.jsonl "$STAGE_DIR/data/processed/jav_eng/"
cp data/processed/jav_eng/test.jsonl "$STAGE_DIR/data/processed/jav_eng/"
cp data/processed/joint/train_balanced.jsonl "$STAGE_DIR/data/processed/joint/"

mkdir -p "$PACKAGE_DIR"

tar -czf "$PACKAGE_DIR/$PACKAGE_NAME" -C "$STAGE_DIR" .

echo "Done."
echo "Package path: $PACKAGE_DIR/$PACKAGE_NAME"
ls -lh "$PACKAGE_DIR/$PACKAGE_NAME"
