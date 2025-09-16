#!/bin/bash

#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -q gpu
#$ -l gpu_card=1
#$ -N crossmodal_retrieval_exp

export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR="/users/jzheng7/crossmodal_retrieval"
cd $PROJECT_DIR

OUTPUT_DIR="/users/jzheng7/crossmodal_retrieval/result"
mkdir -p $OUTPUT_DIR
mkdir -p $HOME/backup_results

echo "=== Cross-Modal Retrieval Experiment Started ==="
echo "Job ID: $JOB_ID"
echo "Host: $HOSTNAME"
echo "Date: $(date)"
echo "Project Directory: $PROJECT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"

# Check files
if [ ! -f "coco_dataset_solution.py" ]; then
    echo "Error: coco_dataset_solution.py not found!"
    exit 1
fi
if [ ! -f "run_experiments.py" ]; then
    echo "Error: run_experiments.py not found!"
    exit 1
fi

# Conda init
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$TRANSFORMERS_CACHE"

echo "=== Environment Setup ==="
source activate cmr 2>/dev/null || conda activate cmr 2>/dev/null || echo "Warning: using base env"
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

# Packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
pip install faiss-gpu numpy scipy scikit-learn matplotlib seaborn pandas tqdm h5py --quiet
pip install transformers pillow requests --quiet

# Test imports
python -c "
import torch, transformers, numpy
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')
print(f'✓ Transformers {transformers.__version__}')
"

echo ""
echo "=== Step 1: Create COCO Dataset (with shuffle & seed=42) ==="
python coco_dataset_solution.py

echo ""
echo "=== Step 2: Run Cross-Modal Retrieval Experiments ==="
python run_experiments.py \
    --data_path ./coco_data/coco_embeddings.npz \
    --output_dir $OUTPUT_DIR \
    --embedding_dims 256 512 1024 \
    --precisions fp16 int8 int4 \
    --candidate_sizes 100 200 300 400 500 \
    --recall_k 1 2 3 4 5 6 7 8 9 10 \
    --batch_size 128 \
    --num_workers 4 \
    --seed 42

echo ""
echo "=== Step 3: Generate Results Summary + PNG Plots ==="
python analyze_results.py \
    --results_dir $OUTPUT_DIR \
    --output_path $OUTPUT_DIR/experiment_summary.json

echo ""
echo "=== Experiment Completed ==="
echo "Results saved to: $OUTPUT_DIR"
echo "Job finished at: $(date)"

# Backup
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
    backup_dir="$HOME/backup_results/crossmodal_$(date +%Y%m%d_%H%M%S)"
    cp -r $OUTPUT_DIR $backup_dir
    echo "Backup created in: $backup_dir"
else
    echo "No results to backup"
fi
