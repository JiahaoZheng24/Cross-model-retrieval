#!/bin/bash

#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu
#$ -l gpu_card=1
#$ -N MSR_VTT_Download_Setup

set -e

# Activate environment
conda activate imagebind
export PYTHONPATH="/scratch365/jzheng7/ImageBind:${PYTHONPATH}"
cd /scratch365/jzheng7/ImageBind
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export TORCH_BACKEND=CUDA

echo "╔════════════════════════════════════════════════════════════╗"
echo "║    MSR-VTT Complete Download & Setup (One-Click)          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Start: $(date)"
echo ""

# ============================================
# Configuration
# ============================================
BASE_DIR="/scratch365/jzheng7/ImageBind"
DOWNLOAD_DIR="$BASE_DIR/msrvtt_downloads"
VIDEO_DIR="$BASE_DIR/msrvtt_videos"
ANNOTATION_DIR="$BASE_DIR/msrvtt_annotation"
TRAIN_EMB_DIR="$BASE_DIR/msrvtt_train_embeddings"
TEST_EMB_DIR="$BASE_DIR/msrvtt_results"

mkdir -p "$DOWNLOAD_DIR" "$VIDEO_DIR" "$ANNOTATION_DIR"

echo "Directories:"
echo "  Videos:      $VIDEO_DIR"
echo "  Annotations: $ANNOTATION_DIR"
echo "  Downloads:   $DOWNLOAD_DIR"
echo ""

# ============================================
# Step 1: Download Annotations
# ============================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/6] Downloading MSR-VTT Annotations (~3MB)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ -f "$ANNOTATION_DIR/MSRVTT_data.json" ]] && \
   [[ -f "$ANNOTATION_DIR/MSRVTT_JSFUSION_test.csv" ]]; then
    echo "✓ Annotations already downloaded"
else
    echo "→ Downloading from CLIP4Clip releases..."
    cd "$DOWNLOAD_DIR"

    wget -c "https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip" \
        -O msrvtt_data.zip

    echo ""
    echo "→ Extracting..."
    unzip -q -o msrvtt_data.zip

    echo "→ Moving files to annotation directory..."
    if [[ -d "msrvtt_data" ]]; then
        cp -v msrvtt_data/* "$ANNOTATION_DIR/"
    fi

    echo "✓ Annotations downloaded and extracted"
fi

# ============================================
# Step 2: Download Videos
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/6] Downloading MSR-VTT Videos (~40GB)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  This is a large download and will take 1-3 hours"
echo ""

VIDEO_COUNT=$(find "$VIDEO_DIR" -name "*.mp4" 2>/dev/null | wc -l)

if [[ $VIDEO_COUNT -ge 9000 ]]; then
    echo "✓ Videos already downloaded ($VIDEO_COUNT files found)"
else
    echo "→ Current video count: $VIDEO_COUNT"
    echo "→ Downloading from Frozen-in-Time..."
    cd "$DOWNLOAD_DIR"

    # Download with resume support
    wget -c "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip" \
        -O MSRVTT.zip

    echo ""
    echo "→ Extracting videos (this may take 30-60 minutes)..."
    unzip -q -o MSRVTT.zip

    echo "→ Moving videos to video directory..."

    # Try different possible locations
    for VIDEO_SOURCE in \
        "MSRVTT/videos/all" \
        "MSRVTT/videos" \
        "videos/all" \
        "videos"; do

        if [[ -d "$DOWNLOAD_DIR/$VIDEO_SOURCE" ]]; then
            echo "  Found videos in: $VIDEO_SOURCE"

            VIDEO_FILES=$(find "$DOWNLOAD_DIR/$VIDEO_SOURCE" -name "*.mp4" | wc -l)
            echo "  Moving $VIDEO_FILES video files..."

            find "$DOWNLOAD_DIR/$VIDEO_SOURCE" -name "*.mp4" -exec mv {} "$VIDEO_DIR/" \;

            echo "  ✓ Videos moved"
            break
        fi
    done

    VIDEO_COUNT=$(find "$VIDEO_DIR" -name "*.mp4" | wc -l)
    echo "✓ Total videos: $VIDEO_COUNT"
fi

# ============================================
# Step 3: Verify Dataset
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/6] Verifying Dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check videos
VIDEO_COUNT=$(find "$VIDEO_DIR" -name "*.mp4" | wc -l)
echo "Videos: $VIDEO_COUNT / 10000"

if [[ $VIDEO_COUNT -ge 9000 ]]; then
    echo "  ✓ Sufficient videos for training"
elif [[ $VIDEO_COUNT -ge 1000 ]]; then
    echo "  ⚠️  Partial dataset, but can proceed with reduced training"
else
    echo "  ❌ Insufficient videos. Please check download."
    exit 1
fi

# Check annotations
echo ""
echo "Annotations:"
for FILE in "MSRVTT_data.json" "MSRVTT_JSFUSION_test.csv" "MSRVTT_train.9k.csv"; do
    if [[ -f "$ANNOTATION_DIR/$FILE" ]]; then
        SIZE=$(du -h "$ANNOTATION_DIR/$FILE" | cut -f1)
        echo "  ✓ $FILE ($SIZE)"
    else
        echo "  ❌ $FILE (missing)"
    fi
done

# ============================================
# Step 4: Generate Training Embeddings
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/6] Generating Training Set Embeddings"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ -f "$TRAIN_EMB_DIR/emb_text.pt" ]] && \
   [[ -f "$TRAIN_EMB_DIR/emb_video.pt" ]]; then
    echo "✓ Training embeddings already exist"
else
    echo "→ Generating embeddings for training set (6513 videos)..."
    echo "  This will take ~1-2 hours"
    echo ""

    cd /scratch365/jzheng7/ImageBind
    python generate_train_embeddings.py

    echo "✓ Training embeddings generated"
fi

# ============================================
# Step 5: Generate Test Embeddings
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[5/6] Generating Test Set Embeddings (1kA)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ -f "$TEST_EMB_DIR/emb_text.pt" ]] && \
   [[ -f "$TEST_EMB_DIR/emb_video.pt" ]]; then
    echo "✓ Test embeddings already exist"
else
    echo "→ Generating embeddings for test set (1000 videos)..."
    echo ""

    cd /scratch365/jzheng7/ImageBind
    python eval_msrvtt_1kA.py

    echo "✓ Test embeddings generated"
fi

# ============================================
# Step 6: Run Data Leakage Diagnostic
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[6/6] Running Data Leakage Diagnostic"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd /scratch365/jzheng7/ImageBind
python diagnose_data_leakage.py

# ============================================
# Cleanup
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Cleanup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

DOWNLOAD_SIZE=$(du -sh "$DOWNLOAD_DIR" 2>/dev/null | cut -f1 || echo "0")
echo "Download directory size: $DOWNLOAD_SIZE"
echo ""
echo "You can safely delete zip files in $DOWNLOAD_DIR to save space:"
echo "  rm $DOWNLOAD_DIR/*.zip"
echo ""

# ============================================
# Summary
# ============================================
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE! ✓                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "End: $(date)"
echo ""

echo "Dataset Summary:"
echo "  Videos:             $VIDEO_COUNT files"
echo "  Training embeddings: $(ls -lh $TRAIN_EMB_DIR/emb_text.pt 2>/dev/null | awk '{print $5}' || echo 'N/A')"
echo "  Test embeddings:     $(ls -lh $TEST_EMB_DIR/emb_text.pt 2>/dev/null | awk '{print $5}' || echo 'N/A')"
echo ""

echo "Next Steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Option 1: Run the corrected benchmark pipeline"
echo "  qsub run_pcme_benchmark_fixed.sh"
echo ""
echo "Option 2: Train PCME manually"
echo "  python train_pcme_projector.py \\"
echo "    --emb_dir /scratch365/jzheng7/ImageBind/msrvtt_train_embeddings \\"
echo "    --save_dir /scratch365/jzheng7/ImageBind/pcme_checkpoints_correct \\"
echo "    --epochs 50"
echo ""
echo "Then evaluate:"
echo "  python measure_latency_memory_variance.py \\"
echo "    --emb_dir /scratch365/jzheng7/ImageBind/msrvtt_results \\"
echo "    --checkpoint /scratch365/jzheng7/ImageBind/pcme_checkpoints_correct/best_projectors.pth"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "For more details, see: PCME_Data_Leakage_Analysis.md"
echo ""
