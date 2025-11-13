#!/bin/bash

#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu
#$ -l gpu_card=1
#$ -N PCME_Quantization_All

set -e

conda activate imagebind
export PYTHONPATH="/scratch365/jzheng7/ImageBind:${PYTHONPATH}"
cd /scratch365/jzheng7/ImageBind

ROOT="/scratch365/jzheng7/ImageBind"
TRAIN_EMB_DIR="$ROOT/msrvtt_train_embeddings"
TEST_EMB_DIR="$ROOT/msrvtt_results"
CKPT_DIR="$ROOT/pcme_checkpoints_correct"
FLOAT32_CKPT="$CKPT_DIR/best_projectors.pth"
INT8_CKPT="$CKPT_DIR/best_projectors_int8.pth"
INT4_CKPT="$CKPT_DIR/best_projectors_int4.pth"

echo "========================================"
echo "  PCME Quantization Benchmark Pipeline"
echo "  Float32 / INT8 / INT4 Comparison"
echo "========================================"
echo "Start: $(date)"
echo ""

# ============================================
# Step 1: Quantize to INT8 if needed
# ============================================
echo "[1/5] Checking INT8 checkpoint..."
echo ""

if [[ -f "$INT8_CKPT" ]]; then
    echo "  ✓ Found existing INT8 checkpoint"
    if [[ "$FLOAT32_CKPT" -nt "$INT8_CKPT" ]]; then
        echo "  ⚠️  Float32 checkpoint is newer, re-quantizing..."
        rm "$INT8_CKPT"
    fi
fi

if [[ ! -f "$INT8_CKPT" ]]; then
    echo "  → Quantizing to INT8..."
    python quantize_projectors.py \
        --ckpt "$FLOAT32_CKPT" \
        --calib_emb_dir "$TRAIN_EMB_DIR" \
        --output "$INT8_CKPT" \
        --n_calib_samples 500
    echo "  ✓ INT8 quantization complete"
fi

# ============================================
# Step 2: Quantize to INT4 if needed
# ============================================
echo ""
echo "[2/5] Checking INT4 checkpoint..."
echo ""

if [[ -f "$INT4_CKPT" ]]; then
    echo "  ✓ Found existing INT4 checkpoint"
    if [[ "$FLOAT32_CKPT" -nt "$INT4_CKPT" ]]; then
        echo "  ⚠️  Float32 checkpoint is newer, re-quantizing..."
        rm "$INT4_CKPT"
    fi
fi

if [[ ! -f "$INT4_CKPT" ]]; then
    echo "  → Quantizing to INT4 (mu only)..."
    python quantize_projectors.py \
        --ckpt "$FLOAT32_CKPT" \
        --calib_emb_dir "$TRAIN_EMB_DIR" \
        --output "$INT4_CKPT" \
        --n_calib_samples 500 \
        --int4_mu
    echo "  ✓ INT4 quantization complete"
fi

# ============================================
# Step 3: Benchmark Float32
# ============================================
echo ""
echo "[3/5] Benchmarking Float32..."
echo ""

python measure_latency_memory_variance.py \
    --emb_dir "$TEST_EMB_DIR" \
    --ckpt "$FLOAT32_CKPT" \
    --runs 10 \
    --warmup 3 \
    --num_samples 15 \
    --eval_sigma_scale 0.0 \
    --save "$TEST_EMB_DIR/variance_analysis_float32.json"

echo "  ✓ Float32 benchmark complete"

# ============================================
# Step 4: Benchmark INT8
# ============================================
echo ""
echo "[4/5] Benchmarking INT8..."
echo ""

python measure_latency_memory_variance.py \
    --emb_dir "$TEST_EMB_DIR" \
    --ckpt "$INT8_CKPT" \
    --quantized \
    --runs 10 \
    --warmup 3 \
    --num_samples 15 \
    --eval_sigma_scale 0.0 \
    --save "$TEST_EMB_DIR/variance_analysis_int8.json"

echo "  ✓ INT8 benchmark complete"

# ============================================
# Step 5: Benchmark INT4
# ============================================
echo ""
echo "[5/5] Benchmarking INT4..."
echo ""

python measure_latency_memory_variance.py \
    --emb_dir "$TEST_EMB_DIR" \
    --ckpt "$INT4_CKPT" \
    --quantized \
    --runs 10 \
    --warmup 3 \
    --num_samples 15 \
    --eval_sigma_scale 0.0 \
    --save "$TEST_EMB_DIR/variance_analysis_int4.json"

echo "  ✓ INT4 benchmark complete"

# ============================================
# Step 6: Compare all results
# ============================================
echo ""
echo "========================================"
echo "  COMPARISON: Float32 vs INT8 vs INT4"
echo "========================================"
echo ""

python -c "
import json
from pathlib import Path

# Load results
with open('$TEST_EMB_DIR/variance_analysis_float32.json') as f:
    float32 = json.load(f)

with open('$TEST_EMB_DIR/variance_analysis_int8.json') as f:
    int8 = json.load(f)

with open('$TEST_EMB_DIR/variance_analysis_int4.json') as f:
    int4 = json.load(f)

# Extract metrics
f32_lat = float32['summary']['pcme']['latency_ms']['mean']
int8_lat = int8['summary']['pcme']['latency_ms']['mean']
int4_lat = int4['summary']['pcme']['latency_ms']['mean']

f32_mem = float32['summary']['pcme']['gpu_mem_mb']['mean']
int8_mem = int8['summary']['pcme']['gpu_mem_mb']['mean']
int4_mem = int4['summary']['pcme']['gpu_mem_mb']['mean']

f32_v2t = float32['summary']['retrieval']['pcme']['v2t']['R@k']['1']
int8_v2t = int8['summary']['retrieval']['pcme']['v2t']['R@k']['1']
int4_v2t = int4['summary']['retrieval']['pcme']['v2t']['R@k']['1']

f32_t2v = float32['summary']['retrieval']['pcme']['t2v']['R@k']['1']
int8_t2v = int8['summary']['retrieval']['pcme']['t2v']['R@k']['1']
int4_t2v = int4['summary']['retrieval']['pcme']['t2v']['R@k']['1']

# Print comparison table
print('Metric                Float32         INT8            INT4')
print('─' * 70)
print(f'Latency (ms)          {f32_lat:>6.2f}          {int8_lat:>6.2f}          {int4_lat:>6.2f}')
print(f'GPU Memory (MB)       {f32_mem:>6.1f}          {int8_mem:>6.1f}          {int4_mem:>6.1f}')
print(f'V2T R@1 (%)           {f32_v2t:>6.2f}          {int8_v2t:>6.2f}          {int4_v2t:>6.2f}')
print(f'T2V R@1 (%)           {f32_t2v:>6.2f}          {int8_t2v:>6.2f}          {int4_t2v:>6.2f}')
print()

# Accuracy drops
int8_drop = f32_v2t - int8_v2t
int4_drop = f32_v2t - int4_v2t

print('Accuracy Drop:')
print(f'  INT8 vs Float32:     {int8_drop:+.2f}%')
print(f'  INT4 vs Float32:     {int4_drop:+.2f}%')
print()

# Model sizes (estimated from checkpoint files)
import os
f32_size = os.path.getsize('$FLOAT32_CKPT') / (1024**2)
int8_size = os.path.getsize('$INT8_CKPT') / (1024**2)
int4_size = os.path.getsize('$INT4_CKPT') / (1024**2)

print('Model Size:')
print(f'  Float32:             {f32_size:>6.2f} MB')
print(f'  INT8:                {int8_size:>6.2f} MB ({f32_size/int8_size:.2f}x smaller)')
print(f'  INT4:                {int4_size:>6.2f} MB ({f32_size/int4_size:.2f}x smaller)')
print()

# Verdict
if int8_drop < 1.5:
    print('✅ INT8: Excellent! Accuracy loss < 1.5%')
elif int8_drop < 3.0:
    print('⚠️  INT8: Acceptable, but could tune further')
else:
    print('❌ INT8: Accuracy drop too large')

if int4_drop < 2.0:
    print('✅ INT4: Excellent! Accuracy loss < 2%')
elif int4_drop < 4.0:
    print('⚠️  INT4: Acceptable for aggressive compression')
else:
    print('❌ INT4: Accuracy drop too large')

print()
print('Results saved:')
print(f'  Float32: $TEST_EMB_DIR/variance_analysis_float32.json')
print(f'  INT8:    $TEST_EMB_DIR/variance_analysis_int8.json')
print(f'  INT4:    $TEST_EMB_DIR/variance_analysis_int4.json')
"

echo ""
echo "========================================"
echo "  ✓ All Done!"
echo "========================================"
echo "End: $(date)"
echo ""