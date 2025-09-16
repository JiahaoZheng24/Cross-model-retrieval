# Cross-Modal Retrieval (Text ↔ Video/Image)

Minimal, reproducible pipeline for cross-modal retrieval. Compare **embedding dims**, **precisions**, and **candidate sizes** with **Recall@10**, **MRR**, **NDCG@10**, speed, and memory.

---

## Environment (Conda: `cmr`)
```bash
conda create -n cmr python=3.9 -y
conda activate cmr
# PyTorch (pick a build)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Core deps
pip install numpy scipy scikit-learn matplotlib seaborn pandas tqdm h5py pillow requests transformers
# FAISS (optional)
pip install faiss-gpu  # or faiss-cpu
```

---

## Dataset & Ground Truth
- Default: **COCO** (images + captions). We treat images as the “video” gallery for speed.
- Run `coco_dataset_solution.py` to extract CLIP (ViT-B/32) features and save:
  ```
  ./coco_data/coco_embeddings.npz
    text_256, text_512, text_1024, ...
    video_256, video_512, video_1024, ...
    ground_truth  # length N; ground_truth[i] = true video index for text i
  ```
- **Shuffle**: we shuffle **queries only** (text_*). The same permutation is applied to the **order** of `ground_truth` so it still points to the correct (unshuffled) gallery.

---

## Similarity & Quantization
- All embeddings are **L2-normalized**; similarity = **dot product** (≡ cosine).
- Precisions (simulated): `fp16`, `int8`, `int4` to study accuracy–memory–speed trade-offs.

---

## Evaluation
- Directions: **Text→Video (T2V)** uses `ground_truth`; **Video→Text (V2T)** uses the **inverse map** `gt_inv[video_idx] = text_idx`.
- For each query, sample a candidate set of size **C** and **always include GT**.
- Metrics: **Recall@10** (default), **MRR**, **NDCG@10**.
- Random baseline: E[Recall@K] = K/C (e.g., K=10, C∈{100,200,300,400,500} ⇒ {0.10, 0.05, 0.033, 0.025, 0.02}).

---

## Quick Start
```bash
# 1) Build data (creates ./coco_data/coco_embeddings.npz)
conda activate cmr
python coco_dataset_solution.py

# 2) Run experiments
python run_experiments.py   --data_path ./coco_data/coco_embeddings.npz   --output_dir ./result   --embedding_dims 256 512 1024   --precisions fp16 int8 int4   --candidate_sizes 100 200 300 400 500   --recall_k 10   --seed 42

# 3) Analyze & plot
python analyze_results.py   --results_dir ./result   --output_path ./result/experiment_summary.json
# (Add --lang zh for a Chinese summary)
```

---

## Project Files
- `coco_dataset_solution.py` — build COCO-based embeddings (`.npz`) with shuffle.
- `run_experiments.py` — evaluate T↔V with GT-included candidate pools; saves JSON.
- `analyze_results.py` — aggregate, export CSV/PNGs, and a brief summary.
- `submit_job.sh` — example cluster script (assumes **conda env: `cmr`**).

---

## Tips
- If Recall@10 ≈ **K/C**, check:
  - GT is included in every candidate pool.
  - T2V uses `ground_truth`; V2T uses **inverse GT**.
  - COCO images/captions actually loaded (no placeholder fallbacks).
- Larger **C** lowers Recall@10 (harder baseline) and slows ranking; watch **lift over K/C**.
