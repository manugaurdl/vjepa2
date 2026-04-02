#!/bin/bash
# Download and extract UCF101 dataset + train/test splits
# Usage: bash scripts/prepare_ucf101.sh [TARGET_DIR]

set -euo pipefail

TARGET_DIR="${1:-/nas/manu/ucf101}"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Download videos (~7GB)
if [ ! -d "UCF-101" ]; then
    echo "Downloading UCF101 videos..."
    wget --no-check-certificate -q --show-progress https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O UCF101.rar
    echo "Extracting..."
    unrar x -o+ UCF101.rar
    rm UCF101.rar
else
    echo "UCF-101/ already exists, skipping download."
fi

# Download train/test split files
if [ ! -d "ucfTrainTestlist" ]; then
    echo "Downloading train/test splits..."
    wget --no-check-certificate -q --show-progress https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip -O splits.zip
    unzip -o splits.zip
    rm splits.zip
else
    echo "ucfTrainTestlist/ already exists, skipping download."
fi

# Generate CSVs
mkdir -p data
echo "Generating train/test CSVs..."
python "$(dirname "$0")/create_ucf101_csv.py" \
    --video_dir "$TARGET_DIR/UCF-101" \
    --split_dir "$TARGET_DIR/ucfTrainTestlist" \
    --output_dir "$TARGET_DIR/data"

echo "Done. CSVs at $TARGET_DIR/data/{train,test}.csv"
