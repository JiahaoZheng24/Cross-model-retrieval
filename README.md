# PCME: Probabilistic Cross-Modal Embeddings for Video-Text Retrieval

Implementation of Probabilistic Cross-Modal Embeddings (PCME) on MSR-VTT dataset, comparing against ImageBind baseline.

## 🔥 Key Results

- **Video→Text R@1**: 30.9% → 44.5% (+13.6% improvement)
- **Text→Video R@1**: 38.8% → 38.2% (-0.6%, expected asymmetry)
- **Latency**: 22x slower (0.46ms → 10.28ms) due to Monte Carlo sampling
- **GPU Memory**: 1.84x increase (199MB → 368MB)

## 📁 Project Files

```
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
# Setup & Download
├── download_msrvtt.py              # Download MSR-VTT dataset
├── setup_msrvtt_complete.sh        # Complete setup (cluster job)
├── diagnose_data_leakage.py        # Verify train/test split
│
# Generate Embeddings
├── eval_msrvtt_1kA.py              # Generate test embeddings (1000 samples)
├── generate_train_embeddings.py    # Generate train embeddings (6513 samples)
│
# Training & Evaluation
├── train_pcme_projector.py         # Train PCME probabilistic projectors
├── measure_latency_memory_variance.py  # Benchmark performance
├── run_pcme_benchmark.sh           # End-to-end pipeline (cluster job)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/JiahaoZheng24/Cross-model-retrieval.git
cd Cross-model-retrieval

# Create conda environment
conda create -n imagebind python=3.10
conda activate imagebind

# Install dependencies
pip install -r requirements.txt
conda install -c conda-forge ffmpeg

# Clone ImageBind (required for feature extraction)
cd ..
git clone https://github.com/facebookresearch/ImageBind.git
export PYTHONPATH="${PWD}/ImageBind:${PYTHONPATH}"
cd Cross-model-retrieval
```

### 2. Download Dataset

```bash
# Interactive download (recommended)
python download_msrvtt.py
```

Or for cluster environments:
```bash
qsub setup_msrvtt_complete.sh
```

### 3. Generate Embeddings

```bash
# Training set (6513 samples)
python generate_train_embeddings.py

# Test set (1000 samples - MSR-VTT 1kA)
python eval_msrvtt_1kA.py

# Verify no data leakage
python diagnose_data_leakage.py
```

### 4. Train PCME

```bash
python train_pcme_projector.py \
  --emb_dir ./msrvtt_train_embeddings \
  --save_dir ./pcme_checkpoints \
  --epochs 40 \
  --batch_size 64 \
  --lr 1e-5 \
  --temperature 0.07 \
  --loss_type pcme_mc \
  --n_samples 5 \
  --var_reg_type upper_bound \
  --max_var 0.09 \
  --var_reg_weight 0.05
```

### 5. Evaluate

```bash
python measure_latency_memory_variance.py \
  --emb_dir ./msrvtt_results \
  --ckpt ./pcme_checkpoints/best_projectors.pth \
  --runs 10 \
  --warmup 5 \
  --num_samples 15
```

### 6. Complete Pipeline (Recommended)

For cluster environments:
```bash
qsub run_pcme_benchmark.sh
```

This runs steps 3-5 automatically with proper train/test separation.

## 📊 Detailed Results

### Retrieval Performance (MSR-VTT 1kA)

| Direction | Metric | ImageBind | PCME | Δ |
|-----------|--------|-----------|------|---|
| Text→Video | R@1 | 38.8% | 38.2% | -0.6% |
| Text→Video | R@5 | 63.7% | 65.0% | +1.3% |
| Text→Video | R@10 | 72.7% | 74.8% | +2.1% |
| **Video→Text** | **R@1** | **30.9%** | **44.5%** | **+13.6%** |
| Video→Text | R@5 | 54.3% | 71.5% | +17.2% |
| Video→Text | R@10 | 64.7% | 80.1% | +15.4% |

### Why Asymmetric Performance?

**PCME excels at Video→Text but not Text→Video:**
- Videos naturally admit multiple textual descriptions (one-to-many)
- Uncertainty modeling helps when there are multiple valid matches
- Text queries are typically more specific (deterministic matching preferred)

### Computational Overhead

| Metric | ImageBind | PCME | Overhead |
|--------|-----------|------|----------|
| Latency (mean) | 0.46 ms | 10.28 ms | 22.2x |
| Latency (std) | 0.024 ms | 0.335 ms | 14.0x |
| GPU Memory | 199.5 MB | 367.6 MB | 1.84x |

**Monte Carlo sampling** (15 samples during inference) is the main bottleneck.

## 🏗️ Architecture

### Probabilistic Projector

Maps deterministic ImageBind embeddings to Gaussian distributions:

```python
class ProbabilisticProjector(nn.Module):
    def __init__(self, dim=1024, hidden=2048):
        super().__init__()
        # Mean head (with residual)
        self.mu_proj = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hidden, dim)
        )
        # Variance head (clamped for stability)
        self.logvar_proj = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hidden, dim)
        )
    
    def forward(self, x):
        mu = x + self.mu_proj(x)  # Residual
        mu = F.normalize(mu, dim=-1)  # Keep on unit sphere
        logvar = torch.clamp(self.logvar_proj(x), -5, 2)
        return mu, logvar
```

### Monte Carlo Similarity

Computes expected cosine similarity between distributions:

```python
E[cos(z_t, z_v)] where z_t ~ N(μ_t, Σ_t), z_v ~ N(μ_v, Σ_v)
```

Approximated by sampling and averaging.

## 🔧 Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--loss_type` | `pcme_mc` | Loss: `pcme_mc` (Monte Carlo) or `deterministic` |
| `--n_samples` | 5 | Monte Carlo samples during training |
| `--var_reg_type` | `upper_bound` | Variance regularization strategy |
| `--var_reg_weight` | 0.05 | Weight for variance regularization |
| `--max_var` | 0.09 | Maximum variance threshold |
| `--temperature` | 0.07 | Temperature for contrastive loss |

### Variance Regularization Options

- `kl`: KL divergence to N(0, I) prior
- `lower_bound`: Prevent collapse below threshold
- `upper_bound`: Prevent excessive variance (recommended)
- `target`: Pull variance toward target value

## ⚠️ Common Issues

### Issue 1: Data Leakage

**Symptom**: Improvement > 20%

**Solution**:
```bash
python diagnose_data_leakage.py
```

Ensure training set = 6513 samples, test set = 1000 samples.

### Issue 2: Variance Collapse

**Symptom**: `Variance: text=0.000001`

**Solution**: Use upper-bound regularization:
```bash
--var_reg_type upper_bound --max_var 0.09 --var_reg_weight 0.05
```

### Issue 3: Out of Memory

**Solution**: Reduce batch size or MC samples:
```bash
--batch_size 32 --n_samples 3
```

## 📖 Citation

```bibtex
@inproceedings{pcme2025,
  title={Probabilistic Cross-Modal Embeddings for Video-Text Retrieval},
  author={Jiahao Zheng},
  booktitle={Design Automation Conference (DAC)},
  year={2025}
}
```

## 📚 Related Work

- **ImageBind**: [Girdhar et al., CVPR 2023](https://arxiv.org/abs/2305.05665)
- **PCME**: [Chun et al., CVPR 2021](https://arxiv.org/abs/2101.05068)
- **MSR-VTT**: [Xu et al., CVPR 2016](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Contact

- **Author**: Jiahao Zheng
- **Email**: jzheng7@nd.edu
- **Institution**: University of Notre Dame

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

**Note**: This is research code. For production deployment, consider optimizing Monte Carlo sampling with GPU kernels or implementing quantization.