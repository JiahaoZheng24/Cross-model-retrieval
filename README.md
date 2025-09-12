# Cross-Modal Retrieval

This repository implements experiments for cross‑modal retrieval (e.g., matching text and images/videos), comparing different embedding dimensions, quantization precisions, and candidate set sizes. The goal is to study trade‑offs in **accuracy (Recall, MRR)**, **speed**, and **memory usage**.

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Environment Setup](#environment-setup)  
- [Usage / Running Experiments](#usage--running-experiments)  
- [Analysis of Results](#analysis-of-results)  
- [Using `submit_job.sh` on Cluster](#using-submit_jobsh-on-cluster)  
- [Dependencies](#dependencies)  
- [License](#license)

---

## Project Structure

```
Cross-model-retrieval/
├── coco_data/                 # Generated dataset embeddings, annotations etc.
├── result/                    # Experiment outputs (metrics, intermediate files)
├── analyze_results.py         # Script to aggregate and summarize results
├── coco_dataset_solution.py   # Data preparation: build embeddings etc.
├── run_experiments.py         # Main script to run retrieval experiments
├── submit_job.sh              # Script for HPC / batch submission
├── requirements.txt           # Python dependencies
├── README.md                  # (This file)
└── LICENSE                    # MIT license
```

---

## Environment Setup

We use **Conda** for environment management and **pip** for dependency installation.  

**1. Create Conda Environment**

```bash
conda create -n crossmodal_exp python=3.9 -y
```

**2. Activate Environment**

```bash
conda activate crossmodal_exp
```

**3. Install Dependencies**

- GPU version (CUDA 11.8):

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

- CPU version (if no GPU available):

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

- Other required packages:

  ```bash
  pip install faiss-gpu  # or faiss-cpu if no GPU
  pip install numpy scipy scikit-learn matplotlib seaborn pandas tqdm h5py pillow
  ```

---

## Dependencies

You can also install all dependencies from `requirements.txt`:

```
torch>=1.13.0
torchvision>=0.14.0
torchaudio>=0.13.0
faiss-gpu>=1.7.2
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
tqdm>=4.62.0
h5py>=3.6.0
pillow>=8.3.0
```

Install via:

```bash
pip install -r requirements.txt
```

---

## Usage / Running Experiments

The workflow comprises three main phases:

1. **Data Preparation**  
   Build the embedding dataset from COCO (or whichever data you use) using `coco_dataset_solution.py`:

   ```bash
   python coco_dataset_solution.py
   ```

   This script should generate embeddings (e.g., via CLIP), and save required files under `coco_data/`.

2. **Running Retrieval Experiments**  
   Use `run_experiments.py` to evaluate with multiple configurations:

   ```bash
   python run_experiments.py \
     --data_path ./coco_data/coco_embeddings.npz \
     --output_dir ./result \
     --embedding_dims 256 512 1024 \
     --precisions fp16 int8 int4 \
     --candidate_sizes 32 100 1000 \
     --recall_k 1 5 10 \
     --batch_size 128 \
     --num_workers 4
   ```

   Adjust parameters as needed for your machine. This generates metrics (Recall@K, MRR, speed, memory) under different settings and saves into `result/`.

3. **Analysis of Results**  
   Aggregate all experiment outputs into a summary JSON or other formats with `analyze_results.py`:

   ```bash
   python analyze_results.py \
     --results_dir ./result \
     --output_path ./result/experiment_summary.json
   ```

   The summary will include things like average Recall@K, speed, memory usage across all configurations.

---

## Using `submit_job.sh` on Cluster

If you are on an HPC or batch system that uses **job scheduler** (Slurm / SGE / PBS etc.), you can use the provided `submit_job.sh` script. It automates:

- Initializing the conda environment  
- Installing dependencies if needed  
- Running the three phases (data prep → experiments → analysis)  
- Saving or backing up results

Example:

```bash
qsub submit_job.sh
# or
sbatch submit_job.sh
```

You may need to adjust environment paths or scheduler directives inside `submit_job.sh` depending on your cluster setup.

---

## Analysis of Results

- After experiments, you will find many configurations compared. Key metrics to look at:

  - **Recall@K** (e.g., @1, @5, @10) — how often the correct item is ranked in top‑K  
  - **MRR** — mean reciprocal rank  
  - **Speed** — how many queries/second, or latency  
  - **Memory usage** — embedding size, quantization overhead

- Typical trade‑offs you will observe:

  - Higher embedding dimension → usually more accurate but slower & more memory  
  - Lower precision (e.g. int4) → saves memory, may be slightly slower / slightly less accurate  
  - Larger candidate set size → higher recall, but with computational cost

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## Contact / Contributing

- If you find issues or have improvements (e.g. more quantization methods, more datasets), feel free to open an issue or PR.  
- Maintain readability and structure (scripts, result folders) when contributing.

---
