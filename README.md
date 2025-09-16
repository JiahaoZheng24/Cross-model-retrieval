# Cross-Modal Retrieval (Text ↔ Video/Image)

A minimal, reproducible pipeline for **cross‑modal retrieval** (text ↔ image/video). It compares **embedding dimensions**, **numeric precisions**, and **candidate sizes** with **Recall@10**, **MRR**, **NDCG@10**, plus speed and memory. Code uses `text_*` (queries) and `video_*` (gallery).

---

## 1) Environment

```bash
# Conda env name is *cmr*
conda create -n cmr python=3.9 -y
conda activate cmr

# Install all deps from the repo
pip install -r requirements.txt
```

> The cluster script `submit_job.sh` also assumes the env is named **cmr**.

---

## 2) Dataset used by this repo

We use **COCO 2017 validation** (images + captions) as the default dataset.

- **What the script does (`coco_dataset_solution.py`)**
  - Loads **captions** from `coco_data/annotations/captions_val2017.json` **if present**.
  - Constructs **image URLs** directly from the official COCO server; **no need to download images** locally.
  - If captions are missing, the script logs the official download command and **falls back to a synthetic sample set** (placeholder images + template captions) so you can still run end‑to‑end.
  - Extracts CLIP (ViT‑B/32) features for both text and image, creates multi‑scale embeddings (256/512/1024), **L2‑normalizes**, and writes one dataset file.

- **Exact URLs hard‑coded in the code**
  - Annotations ZIP: `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
  - Images base path: `http://images.cocodataset.org/val2017/` (images are fetched by filename via HTTP)
  - Placeholder (for synthetic mode): `https://via.placeholder.com/640x480/color/text=Image{i}`

- **Output (created by the script)**
  ```
  ./coco_data/coco_embeddings.npz
    text_256, text_512, text_1024, ...
    video_256, video_512, video_1024, ...
    ground_truth  # ground_truth[i] = true gallery index for text i
  ```
  *Default #samples = 1000 (fixed in the script’s `main()`).*

> You don’t need to manually download images. If you want real captions, place `captions_val2017.json` under `./coco_data/annotations/` or follow the logged command the script prints (same official URL as above).

---

## 3) How embeddings are compared

- All embeddings are **L2‑normalized**.
- Similarity is the **dot product**, which equals **cosine similarity** after normalization.
- Quantization ablations simulate `fp16`, `int8`, and `int4` to study accuracy–speed–memory trade‑offs.

---

## 4) Evaluation protocol

We evaluate both directions:

- **Text → Video (T2V)** uses `ground_truth[i] = video_idx`.
- **Video → Text (V2T)** builds the inverse map `gt_inv[video_idx] = text_idx`.

For each query we sample a candidate set of size **C** and **always include the GT** item; ranking is done **within the pool**.

**Metrics:** **Recall@10** (default), **MRR**, **NDCG@10**.  
**Random baseline:** \(\mathbb{E}[\mathrm{Recall@K}] = K/C\). For K=10 and C∈{100,200,300,400,500}: {0.10, 0.05, 0.033, 0.025, 0.02}.

**Reproducibility:** fixed `--seed` (42) and independent `np.random.default_rng(seed)` for candidate sampling. The query side is shuffled; `ground_truth` order is permuted accordingly.

---

## 5) Run the pipeline

```bash
# 1) Build the dataset file (uses COCO online images; captions if available)
conda activate cmr
python coco_dataset_solution.py

# 2) Run experiments (Recall@10; candidate sizes 100..500)
python run_experiments.py   --data_path ./coco_data/coco_embeddings.npz   --output_dir ./result   --embedding_dims 256 512 1024   --precisions fp16 int8 int4   --candidate_sizes 100 200 300 400 500   --recall_k 10   --seed 42

# 3) Analyze & plot
python analyze_results.py   --results_dir ./result   --output_path ./result/experiment_summary.json
```

Artifacts are written to `./result/`:
- `experiment_results.json`, `experiment_summary.json`
- `detailed_results.csv`, `summary_by_candidate.csv`
- `experiment_analysis.png`, `summary_recall10.png`, `summary_mrr.png`

---

## 6) Repo layout

```
.
├── coco_data/                 # Generated dataset & metadata
├── result/                    # Experiment outputs (JSON/CSV/PNGs)
├── coco_dataset_solution.py   # Builds COCO-based embeddings (uses online images)
├── run_experiments.py         # Main evaluation loop (GT-included sampling; T↔V with inverse GT)
├── analyze_results.py         # Aggregation & plots (Recall@10 focus)
├── submit_job.sh              # Example cluster script (env: cmr)
├── requirements.txt
└── README.md
```
