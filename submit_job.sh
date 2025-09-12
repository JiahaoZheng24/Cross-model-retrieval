#!/bin/bash

#$ -M @nd.edu
#$ -m abe
#$ -pe smp 8  # Use 8 cores for parallel processing
#$ -q gpu  # Run on the GPU cluster
#$ -l gpu_card=1    # Only need 1 GPU card (A10 has 24GB, sufficient)
#$ -N crossmodal_retrieval_exp    # Job name

# Set GPU visibility (only use first GPU)
export CUDA_VISIBLE_DEVICES=0

# Change to the actual project directory where your files are located
PROJECT_DIR="/users/jzheng7/crossmodal_retrieval"
cd $PROJECT_DIR

# Set output directory
OUTPUT_DIR="/users/jzheng7/crossmodal_retrieval/result"
mkdir -p $OUTPUT_DIR

# Create backup directory
mkdir -p $HOME/backup_results

echo "=== Cross-Modal Retrieval Experiment Started ==="
echo "Job ID: $JOB_ID"
echo "Host: $HOSTNAME"
echo "Date: $(date)"
echo "Project Directory: $PROJECT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"

# Check if we're in the right directory and files exist
echo "Current directory: $(pwd)"
echo "Available files:"
ls -la

# Check if required files exist
if [ ! -f "coco_dataset_solution.py" ]; then
    echo "Error: coco_dataset_solution.py not found!"
    exit 1
fi

if [ ! -f "run_experiments.py" ]; then
    echo "Error: run_experiments.py not found!"
    exit 1
fi

# Try to activate your cmr environment
echo ""
# 1) 正确初始化 conda（任选其一会命中），保证后面的 `conda activate cmr` 生效
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# 2) 设定 HF 缓存路径（先在登录节点预下载一次模型）
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$TRANSFORMERS_CACHE"

echo "=== Environment Setup ==="
echo "Attempting to activate cmr environment..."
source activate cmr 2>/dev/null || conda activate cmr 2>/dev/null || echo "Warning: Failed to activate cmr environment, using base"

# Show current environment
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

# Install required packages
echo "Installing required packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
pip install faiss-gpu numpy scipy scikit-learn matplotlib seaborn pandas tqdm h5py --quiet
pip install transformers pillow requests --quiet

# Test imports
echo "Testing package imports..."
python -c "
import torch
import numpy as np
import transformers
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'✓ Current device: {torch.cuda.current_device()}')
print(f'✓ Transformers {transformers.__version__}')
"

echo ""
echo "=== Step 1: Create COCO Dataset ==="
python coco_dataset_solution.py

echo ""
echo "=== Step 2: Run Cross-Modal Retrieval Experiments ==="
python run_experiments.py \
    --data_path ./coco_data/coco_embeddings.npz \
    --output_dir $OUTPUT_DIR \
    --embedding_dims 256 512 1024 \
    --precisions fp16 int8 int4 \
    --candidate_sizes 32 100 1000 \
    --recall_k 1 5 10 \
    --batch_size 128 \
    --num_workers 4

echo ""
echo "=== Step 3: Generate Results Summary ==="
python analyze_results.py \
    --results_dir $OUTPUT_DIR \
    --output_path $OUTPUT_DIR/experiment_summary.json

echo ""
echo "=== Experiment Completed ==="
echo "Results saved to: $OUTPUT_DIR"
echo "Job finished at: $(date)"

# Copy important results to backup location
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
    backup_dir="$HOME/backup_results/crossmodal_$(date +%Y%m%d_%H%M%S)"
    cp -r $OUTPUT_DIR $backup_dir
    echo "Backup created in: $backup_dir"
else
    echo "No results to backup"
fi