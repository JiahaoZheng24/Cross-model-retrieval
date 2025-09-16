# Cross-Modal Retrieval (Text ↔ Video/Image)

A minimal, reproducible pipeline for **cross-modal retrieval** (text ↔ image/video). It compares **embedding dimensions**, **numeric precisions**, and **candidate set sizes** with **Recall@10**, **MRR**, **NDCG@10**, plus speed and memory. Code uses `text_*` (queries) and `video_*` (gallery).

---

## 1) Environment

```bash
# Conda env is named *cmr*
conda create -n cmr python=3.9 -y
conda activate cmr

# Install all dependencies
pip install -r requirements.txt
```

> The cluster script `submit_job.sh` also assumes the env is **cmr**.

---

## 2) Dataset used by this repo

We use **COCO 2017 validation** (images + captions) as the default dataset.

- **What the script does (`coco_dataset_solution.py`)**
  - Loads **captions** from `coco_data/annotations/captions_val2017.json` if present; otherwise logs how to fetch the official annotations ZIP and runs a **synthetic fallback** so the pipeline still works.
  - Fetches images **online** via the official COCO server (no manual download required for images).
  - Extracts CLIP (ViT-B/32) features for text & image, creates multi-scale embeddings (256/512/1024), **L2-normalizes**, and writes one NPZ dataset.

- **Exact URLs hard-coded in the code**
  - Annotations ZIP: `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
  - Images base path: `http://images.cocodataset.org/val2017/` (files fetched by filename)
  - Synthetic fallback (placeholder): `https://via.placeholder.com/640x480/color/text=Image{i}`

- **Output produced**
  ```
  ./coco_data/coco_embeddings.npz
    text_256, text_512, text_1024, ...
    video_256, video_512, video_1024, ...
    ground_truth  # ground_truth[i] = true gallery index for text i
  ```
  *(Default sample count is set in the script; queries are shuffled, and `ground_truth` order is permuted accordingly.)*

---

## 3) Default Experiment Settings (this repo)

These are the defaults wired into `run_experiments.py` and used in our analyses:

- **Directions**: Text→Video (T2V) and Video→Text (V2T, uses inverse GT)
- **Metrics**: **Recall@10** (primary), **MRR**, **NDCG@10**
- **Candidate sizes (C)**: **100, 200, 300, 400, 500**
- **Recall@K**: **K = 10**
- **Embedding dimensions**: **256, 512, 1024**
- **Precisions**: **fp16, int8, int4** (simple uniform quantization simulation)
- **RNG / Seed**: **42** (NumPy `default_rng`; Python/Torch also seeded)
- **Shuffling**: **queries only** (text side); `ground_truth` order permuted to match
- **Candidate sampling**: For each query, sample C−1 negatives and **always include the GT** item; rank **within** the pool
- **Similarity**: **dot product** on **L2-normalized** embeddings (i.e., **cosine similarity**)
- **Batch / Workers**: `--batch_size 128`, `--num_workers 4` (placeholders for scaling encoders)

**Random baseline**: with GT in the pool, random ranking yields \( \mathbb{E}[\mathrm{Recall@K}] = K/C \).  
For K=10 and C∈{100,200,300,400,500}: {0.10, 0.05, 0.033, 0.025, 0.02}.

---

## 4) Run the pipeline

```bash
# 1) Build the dataset file (uses online COCO images; captions if available)
conda activate cmr
python coco_dataset_solution.py

# 2) Run experiments
python run_experiments.py   --data_path ./coco_data/coco_embeddings.npz   --output_dir ./result   --embedding_dims 256 512 1024   --precisions fp16 int8 int4   --candidate_sizes 100 200 300 400 500   --recall_k 10   --batch_size 128   --num_workers 4   --seed 42

# 3) Analyze & plot
python analyze_results.py   --results_dir ./result   --output_path ./result/experiment_summary.json
```

Artifacts are written to `./result/`:
- `experiment_results.json`, `experiment_summary.json`
- `detailed_results.csv`, `summary_by_candidate.csv`
- `experiment_analysis.png`, `summary_recall10.png`, `summary_mrr.png`

---

## 5) Repo layout

```
.
├── coco_data/                 # Generated dataset & metadata
├── result/                    # Experiment outputs (JSON/CSV/PNGs)
├── coco_dataset_solution.py   # Builds COCO-based embeddings (uses online images)
├── run_experiments.py         # Main evaluation (GT-included sampling; T↔V w/ inverse GT)
├── analyze_results.py         # Aggregation & plots (Recall@10 focus)
├── submit_job.sh              # Example cluster script (env: cmr)
├── requirements.txt
└── README.md
```
