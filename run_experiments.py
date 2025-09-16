#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for running cross-modal retrieval experiments.

Key features in this version:
- Recall@K defaults to [10] (you can override via --recall_k).
- Candidate sizes default to [100, 200, 300, 400, 500].
- Uses provided ground_truth mapping for T->V, and its inverse for V->T.
- Always includes the ground-truth item in each query's candidate pool.
- Deterministic sampling with a fixed RNG (np.random.default_rng(seed)).

Expected dataset (.npz):
  text_256, text_512, text_1024, ...
  video_256, video_512, video_1024, ...
  ground_truth : list/array of ints, where ground_truth[i] = true video idx for text i.

Output:
  <output_dir>/experiment_results.json
"""

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------- Logging ------------------------------------- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("crossmodal")


# ---------------------------- Config -------------------------------------- #
@dataclass
class ExperimentConfig:
    embedding_dims: List[int]
    precisions: List[str]
    candidate_sizes: List[int]
    recall_k_values: List[int]  # default [10]
    batch_size: int = 128
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# ----------------------- Quantization Utilities --------------------------- #
class QuantizationUtils:
    """(Simple) utilities for fake quantization used in similarity calc."""

    @staticmethod
    def quantize_to_int8(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
        mn, mx = x.min(), x.max()
        scale = (mx - mn) / 255.0 if mx > mn else 1.0
        q = np.round((x - mn) / scale).astype(np.int8)
        return q, scale, mn

    @staticmethod
    def quantize_to_int4(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
        mn, mx = x.min(), x.max()
        scale = (mx - mn) / 15.0 if mx > mn else 1.0
        q = np.round((x - mn) / scale).astype(np.int16)
        q = np.clip(q, 0, 15).astype(np.int8)
        return q, scale, mn


# ---------------------------- Metrics ------------------------------------- #
class RetrievalMetrics:
    @staticmethod
    def recall_at_k(retrieved_indices: np.ndarray, ground_truth: List[int], k: int) -> float:
        """retrieved_indices: (N, k_max) of global indices; ground_truth: length N."""
        hits = 0
        N = len(ground_truth)
        k = max(1, int(k))
        for i in range(N):
            gt = int(ground_truth[i])
            if gt in retrieved_indices[i, :k]:
                hits += 1
        return hits / N if N > 0 else 0.0

    @staticmethod
    def mrr(retrieved_indices: np.ndarray, ground_truth: List[int]) -> float:
        rr = []
        N = len(ground_truth)
        for i in range(N):
            gt = int(ground_truth[i])
            pos = np.where(retrieved_indices[i] == gt)[0]
            rr.append(1.0 / (pos[0] + 1) if len(pos) > 0 else 0.0)
        return float(np.mean(rr)) if N > 0 else 0.0

    @staticmethod
    def ndcg_at_k(retrieved_indices: np.ndarray, ground_truth: List[int], k: int) -> float:
        """Binary relevance: 1 for the GT item; 0 otherwise."""
        k = max(1, int(k))
        N = len(ground_truth)
        scores = []
        for i in range(N):
            gt = int(ground_truth[i])
            topk = retrieved_indices[i, :k]
            if gt in topk:
                rank = int(np.where(topk == gt)[0][0]) + 1  # 1-based
                dcg = 1.0 / np.log2(rank + 1)
                scores.append(dcg)  # IDCG == 1.0
            else:
                scores.append(0.0)
        return float(np.mean(scores)) if N > 0 else 0.0


# ----------------------- Retriever (core logic) --------------------------- #
class CrossModalRetriever:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.metrics = RetrievalMetrics()
        self.rng = np.random.default_rng(self.config.seed)
        random.seed(self.config.seed)

    def _compute_similarity(self, Q: np.ndarray, G: np.ndarray, precision: str) -> np.ndarray:
        """Return cosine-like similarity via dot-product (assumes L2-normalized inputs)."""
        if precision == "int8":
            Qq, _, _ = QuantizationUtils.quantize_to_int8(Q)
            Gq, _, _ = QuantizationUtils.quantize_to_int8(G)
            Q, G = Qq.astype(np.float32), Gq.astype(np.float32)
        elif precision == "int4":
            Qq, _, _ = QuantizationUtils.quantize_to_int4(Q)
            Gq, _, _ = QuantizationUtils.quantize_to_int4(G)
            Q, G = Qq.astype(np.float32), Gq.astype(np.float32)
        elif precision == "fp16":
            Q, G = Q.astype(np.float16), G.astype(np.float16)
        # dot product (if embeddings are normalized, equals cosine similarity)
        return (Q @ G.T).astype(np.float32)

    def retrieve_topk(
        self,
        query_embeddings: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_size: int,
        k: int,
        precision: str,
        gt_list: List[int],
    ) -> Tuple[np.ndarray, float]:
        """
        For each query i, build a candidate pool of size 'candidate_size' from the candidate set,
        making sure the ground-truth index gt_list[i] is included. Then rank by similarity and
        return top-k global indices.

        Returns:
            top_k_global: (N, k) array of global candidate indices.
            elapsed_sec: float
        """
        t0 = time.time()
        Nq = len(query_embeddings)
        Ng = len(candidate_embeddings)
        k = int(k)
        if k > candidate_size:
            logger.warning(f"k={k} > candidate_size={candidate_size}; clipping k to candidate_size.")
            k = candidate_size

        all_indices = np.arange(Ng, dtype=int)
        top_k_global = np.zeros((Nq, k), dtype=int)

        for i in range(Nq):
            gt_idx = int(gt_list[i]) if i < len(gt_list) else None

            if candidate_size >= Ng:
                cand = all_indices.copy()
                # if gt exists but somehow out of range, log it
                if gt_idx is not None and (gt_idx < 0 or gt_idx >= Ng):
                    logger.warning(f"[T{i}] GT {gt_idx} out of range 0..{Ng-1}")
            else:
                # sample negatives excluding GT; then add GT; then shuffle
                if gt_idx is not None and 0 <= gt_idx < Ng:
                    pool = np.delete(all_indices, gt_idx)
                    need = max(0, candidate_size - 1)
                    neg = self.rng.choice(pool, size=need, replace=False)
                    cand = np.concatenate([[gt_idx], neg])
                else:
                    # no valid GT, fallback to pure random pool
                    cand = self.rng.choice(all_indices, size=candidate_size, replace=False)
                self.rng.shuffle(cand)

            # Sanity: ensure GT is inside the pool when valid
            if gt_idx is not None and 0 <= gt_idx < Ng:
                if gt_idx not in cand:
                    # In very rare cases (shouldn't happen), enforce
                    cand[0] = gt_idx
                    self.rng.shuffle(cand)

            # Rank within the pool
            q = query_embeddings[i:i + 1]        # (1, D)
            G = candidate_embeddings[cand]       # (C, D)
            sim = self._compute_similarity(q, G, precision)  # (1, C)
            order = np.argsort(-sim[0])[:k]
            top_k_global[i] = cand[order]

        elapsed = time.time() - t0
        return top_k_global, elapsed

    @staticmethod
    def _memory_usage_mb(embeddings: np.ndarray, precision: str) -> float:
        if precision == "fp16":
            bpe = 2
        elif precision == "int8":
            bpe = 1
        elif precision == "int4":
            bpe = 0.5
        else:
            bpe = 4  # fp32
        return float(embeddings.size * bpe / (1024 * 1024))

    def run_single_experiment(
        self,
        text_emb: np.ndarray,
        video_emb: np.ndarray,
        ground_truth: List[int],
        embedding_dim: int,
        precision: str,
        candidate_size: int,
    ) -> Dict:
        """Run one (dim, precision, candidate_size) config."""
        logger.info(f"Running: dim={embedding_dim}, prec={precision}, cand={candidate_size}")

        max_k = max(self.config.recall_k_values)

        # Build inverse mapping for V->T: gt_inv[video_idx] = true text_idx
        gt_inv = np.empty(len(video_emb), dtype=int)
        gt_inv.fill(-1)
        for t_idx, v_idx in enumerate(ground_truth):
            if 0 <= int(v_idx) < len(video_emb):
                gt_inv[int(v_idx)] = t_idx

        # Text -> Video
        t2v_topk, t2v_time = self.retrieve_topk(
            text_emb, video_emb, candidate_size, max_k, precision, gt_list=ground_truth
        )
        t2v = {}
        for k in self.config.recall_k_values:
            t2v[f"recall@{k}"] = self.metrics.recall_at_k(t2v_topk, ground_truth, k)
            t2v[f"ndcg@{k}"] = self.metrics.ndcg_at_k(t2v_topk, ground_truth, k)
        t2v["mrr"] = self.metrics.mrr(t2v_topk, ground_truth)

        # Video -> Text
        v2t_topk, v2t_time = self.retrieve_topk(
            video_emb, text_emb, candidate_size, max_k, precision, gt_list=gt_inv.tolist()
        )
        v2t = {}
        for k in self.config.recall_k_values:
            v2t[f"recall@{k}"] = self.metrics.recall_at_k(v2t_topk, gt_inv.tolist(), k)
            v2t[f"ndcg@{k}"] = self.metrics.ndcg_at_k(v2t_topk, gt_inv.tolist(), k)
        v2t["mrr"] = self.metrics.mrr(v2t_topk, gt_inv.tolist())

        # Memory estimate (per embedding matrix, at given precision)
        mem_mb = self._memory_usage_mb(text_emb, precision)

        return {
            "text_to_video": t2v,
            "video_to_text": v2t,
            "timing": {
                "text_to_video_time": float(t2v_time),
                "video_to_text_time": float(v2t_time),
                "total_time": float(t2v_time + v2t_time),
            },
            "memory_usage_mb": float(mem_mb),
            "config": {
                "embedding_dim": int(embedding_dim),
                "precision": str(precision),
                "candidate_size": int(candidate_size),
                "num_queries": int(len(text_emb)),
            },
        }


# ---------------------------- IO helpers --------------------------------- #
def load_dataset(data_path: str) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Load embeddings and ground-truth mapping from .npz."""
    logger.info(f"Loading dataset: {data_path}")
    data = np.load(data_path, allow_pickle=True)

    embeddings: Dict[str, np.ndarray] = {}
    for key in data.files:
        if key != "ground_truth":
            embeddings[key] = data[key]

    gt = data["ground_truth"].tolist()
    logger.info(
        "Loaded %d pairs | text dims available: %s",
        len(gt),
        [k.split("_")[1] for k in embeddings.keys() if k.startswith("text_")],
    )
    return embeddings, gt


def save_results(results: Dict, output_dir: str) -> str:
    out_path = Path(output_dir) / "experiment_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {out_path}")
    return str(out_path)


# ----------------------------- Runner ------------------------------------ #
def run_full_experiments(config: ExperimentConfig, data_path: str, output_dir: str) -> Dict:
    # Global seeds
    np.random.seed(config.seed)
    random.seed(config.seed)
    try:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    embeddings, ground_truth = load_dataset(data_path)
    retriever = CrossModalRetriever(config)

    combos = [
        (dim, prec, cand)
        for dim in config.embedding_dims
        for prec in config.precisions
        for cand in config.candidate_sizes
    ]
    logger.info(f"Total experiment configs: {len(combos)}")

    results: Dict[str, Dict] = {}
    with tqdm(total=len(combos), desc="Experiments") as pbar:
        for dim, prec, cand in combos:
            key = f"dim_{dim}_prec_{prec}_cand_{cand}"
            try:
                text_key = f"text_{dim}"
                video_key = f"video_{dim}"
                text_emb = embeddings[text_key]
                video_emb = embeddings[video_key]

                res = retriever.run_single_experiment(
                    text_emb=text_emb,
                    video_emb=video_emb,
                    ground_truth=ground_truth,
                    embedding_dim=dim,
                    precision=prec,
                    candidate_size=cand,
                )
                results[key] = res

                # quick log (Recall@10 if present)
                k_show = config.recall_k_values[0]
                t2v_r = res["text_to_video"].get(f"recall@{k_show}", 0.0)
                v2t_r = res["video_to_text"].get(f"recall@{k_show}", 0.0)
                logger.info(f"[{key}] T2V R@{k_show}={t2v_r:.4f} | V2T R@{k_show}={v2t_r:.4f}")

            except Exception as e:
                logger.exception(f"Failed: {key}")
                results[key] = {"error": str(e)}

            pbar.update(1)

    save_results(results, output_dir)
    return results


# -------------------------------- Main ----------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Cross-Modal Retrieval Experiments")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset .npz")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write results JSON")
    parser.add_argument("--embedding_dims", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--precisions", nargs="+", type=str, default=["fp16", "int8", "int4"])
    parser.add_argument("--candidate_sizes", nargs="+", type=int, default=[100, 200, 300, 400, 500])
    parser.add_argument("--recall_k", nargs="+", type=int, default=[10], help="Recall@K values (default: 10)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = ExperimentConfig(
        embedding_dims=args.embedding_dims,
        precisions=args.precisions,
        candidate_sizes=args.candidate_sizes,
        recall_k_values=args.recall_k,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    logger.info("Starting Cross-Modal Retrieval Experiments")
    logger.info(f"Config: {config}")

    run_full_experiments(config, args.data_path, args.output_dir)

    logger.info("All experiments finished.")


if __name__ == "__main__":
    main()
