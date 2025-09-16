# Cross-Modal Retrieval (Text ↔ Video/Image)

A minimal, reproducible pipeline for **cross-modal retrieval**. It compares **embedding dimensions**, **numeric precisions**, and **candidate set sizes** using **Recall@10**, **MRR**, **NDCG@10**, plus speed and memory. Code paths use `text_*` (queries) and `video_*` (gallery).

---

## 1) Environment (Conda env: `cmr`)

```bash
# Create & activate
conda create -n cmr python=3.9 -y
conda activate cmr

# PyTorch (choose a build)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Or CPU-only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Core deps
pip install numpy scipy scikit-learn matplotlib seaborn pandas tqdm h5py pillow requests transformers
# Optional ANN
pip install faiss-gpu  # or: pip install faiss-cpu
```

> The cluster script `submit_job.sh` also assumes the conda env name **`cmr`**.

---

## 2) Dataset (COCO 2017) & Ground Truth

We use **COCO** (images + captions) to simulate the “video” gallery with images for fast iterations.

### Download (official URLs)

```bash
# Images
wget http://images.cocodataset.org/zips/val2017.zip

# Annotations (contains captions_train2017.json / captions_val2017.json, etc.)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

### Unzip to expected layout

```bash
mkdir -p coco_data
unzip val2017.zip -d coco_data/              # -> coco_data/val2017
unzip annotations_trainval2017.zip -d coco_data/  # -> coco_data/annotations/*.json
```

### Build embeddings

```bash
# Extract CLIP (ViT-B/32) features, make multi-dim projections, L2-normalize,
# shuffle queries (text side) only, and write a single .npz file:
python coco_dataset_solution.py
```

This creates:
```
./coco_data/coco_embeddings.npz
  text_256, text_512, text_1024, ...
  video_256, video_512, video_1024, ...
  ground_truth   # list/array, ground_truth[i] = true video index for text i
```
Notes:
- We **shuffle the query order (text_*) only** and apply the same permutation to the **order** of `ground_truth` so it still points to the (unshuffled) gallery.
- If you shuffle the gallery yourself, you must remap `ground_truth` via the **inverse permutation**.

---

## 3) Similarity & Quantization

- All embeddings are **L2-normalized**, and scoring uses **dot product** (equivalent to **cosine similarity** after normalization).
- Precisions for ablations: `fp16`, `int8`, `int4` (simple uniform quantization simulation to study accuracy–memory–speed trade-offs).

---

## 4) Evaluation Protocol

Both directions are evaluated:

- **Text → Video (T2V)** uses `ground_truth`: `ground_truth[i] = video_idx`.
- **Video → Text (V2T)** uses the automatically built **inverse map** `gt_inv[video_idx] = text_idx`.

For each query, we sample a candidate set of size **C** from the gallery and **always include the GT** item; ranking is done within that pool.

**Metrics**
- **Recall@10** (default), **MRR**, **NDCG@10**

**Random baseline**
- With GT guaranteed in the pool, random ranking yields \( \mathbb{E}[\mathrm{Recall@K}] = K/C \).
- For K=10 and C ∈ {100, 200, 300, 400, 500}, baselines are {0.10, 0.05, 0.033, 0.025, 0.02}.

**Reproducibility**
- We fix `--seed` (default 42) and use an independent `np.random.default_rng(seed)` for candidate sampling.

---

## 5) Run Experiments

```bash
python run_experiments.py   --data_path ./coco_data/coco_embeddings.npz   --output_dir ./result   --embedding_dims 256 512 1024   --precisions fp16 int8 int4   --candidate_sizes 100 200 300 400 500   --recall_k 10   --batch_size 128   --num_workers 4   --seed 42
```

What happens:
- For each (dim, precision, C), we rank both directions.
- **GT is always in the candidate pool**.
- Results are saved to `./result/experiment_results.json`.

---

## 6) Analyze & Plot

```bash
# English summary (default)
python analyze_results.py   --results_dir ./result   --output_path ./result/experiment_summary.json

# Chinese summary (optional)
# python analyze_results.py --results_dir ./result --output_path ./result/experiment_summary.json --lang zh
```

Artifacts written to `./result/`:
- `experiment_summary.json` — totals, best configs, grouped averages, short summary
- `detailed_results.csv` — one row per config
- `summary_by_candidate.csv` — aggregated Recall@10 / MRR by candidate size
- `experiment_analysis.png`, `summary_recall10.png`, `summary_mrr.png`

---

## 7) Tips / Troubleshooting

- If Recall@10 ≈ **K/C**, check:
  - GT is included in every candidate pool (code enforces this).
  - T2V uses `ground_truth`; V2T uses the **inverse map**.
  - COCO files are present and images actually load (avoid placeholder fallbacks).
- Larger **C** lowers Recall@10 (harder baseline) and slows ranking; compare **lift over K/C**.
- The cluster script `submit_job.sh` runs: dataset → experiments → analysis (assumes `cmr`).

---

## 8) Repository Layout

```
.
├── coco_data/                 # Generated dataset & metadata
├── result/                    # Experiment outputs (JSON/CSV/PNGs)
├── analyze_results.py         # Aggregation & plots
├── coco_dataset_solution.py   # COCO feature builder (CLIP, shuffle queries)
├── run_experiments.py         # Main evaluation loop (GT-included sampling; T↔V with inverse GT)
├── submit_job.sh              # Example cluster script (env: cmr)
└── README.md
```
