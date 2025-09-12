"""
Main script for running cross-modal retrieval experiments
Tests different embedding dimensions, precisions, and candidate sizes
"""

import argparse
import numpy as np
import torch
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for cross-modal retrieval experiments"""
    embedding_dims: List[int]
    precisions: List[str]
    candidate_sizes: List[int]
    recall_k_values: List[int]
    batch_size: int = 128
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class QuantizationUtils:
    """Utilities for embedding quantization"""

    @staticmethod
    def quantize_to_int8(embeddings: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Quantize embeddings to INT8"""
        min_val = embeddings.min()
        max_val = embeddings.max()
        scale = (max_val - min_val) / 255.0
        quantized = np.round((embeddings - min_val) / scale).astype(np.int8)
        return quantized, scale, min_val

    @staticmethod
    def dequantize_from_int8(quantized: np.ndarray, scale: float, min_val: float) -> np.ndarray:
        """Dequantize from INT8"""
        return quantized.astype(np.float32) * scale + min_val

    @staticmethod
    def quantize_to_int4(embeddings: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Quantize embeddings to INT4"""
        min_val = embeddings.min()
        max_val = embeddings.max()
        scale = (max_val - min_val) / 15.0
        quantized = np.round((embeddings - min_val) / scale).astype(np.int8)
        quantized = np.clip(quantized, 0, 15)
        return quantized, scale, min_val


class RetrievalMetrics:
    """Calculate retrieval metrics"""

    @staticmethod
    def calculate_recall_at_k(retrieved_indices: np.ndarray, ground_truth: List[int], k: int) -> float:
        """Calculate Recall@K"""
        hits = 0
        for i, gt in enumerate(ground_truth):
            if gt in retrieved_indices[i, :k]:
                hits += 1
        return hits / len(ground_truth)

    @staticmethod
    def calculate_mrr(retrieved_indices: np.ndarray, ground_truth: List[int]) -> float:
        """Calculate Mean Reciprocal Rank"""
        reciprocal_ranks = []
        for i, gt in enumerate(ground_truth):
            rank_pos = np.where(retrieved_indices[i] == gt)[0]
            if len(rank_pos) > 0:
                reciprocal_ranks.append(1.0 / (rank_pos[0] + 1))
            else:
                reciprocal_ranks.append(0.0)
        return np.mean(reciprocal_ranks)

    @staticmethod
    def calculate_ndcg_at_k(retrieved_indices: np.ndarray, ground_truth: List[int], k: int) -> float:
        """Calculate NDCG@K (simplified version)"""
        ndcg_scores = []
        for i, gt in enumerate(ground_truth):
            retrieved = retrieved_indices[i, :k]
            if gt in retrieved:
                rank = np.where(retrieved == gt)[0][0] + 1
                dcg = 1.0 / np.log2(rank + 1)
                idcg = 1.0  # Perfect ranking has score 1
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)
        return np.mean(ndcg_scores)


class CrossModalRetriever:
    """Cross-modal retrieval system"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.metrics_calc = RetrievalMetrics()

    def compute_similarity_matrix(self, query_emb: np.ndarray, candidate_emb: np.ndarray,
                                  precision: str) -> np.ndarray:
        """Compute similarity matrix with specified precision"""

        if precision == "int8":
            # Quantize both embeddings
            query_q, q_scale, q_min = QuantizationUtils.quantize_to_int8(query_emb)
            cand_q, c_scale, c_min = QuantizationUtils.quantize_to_int8(candidate_emb)

            # Compute similarity in quantized space (approximate)
            query_emb = query_q.astype(np.float32)
            candidate_emb = cand_q.astype(np.float32)

        elif precision == "int4":
            # Quantize to INT4
            query_q, q_scale, q_min = QuantizationUtils.quantize_to_int4(query_emb)
            cand_q, c_scale, c_min = QuantizationUtils.quantize_to_int4(candidate_emb)

            query_emb = query_q.astype(np.float32)
            candidate_emb = cand_q.astype(np.float32)

        elif precision == "fp16":
            # Use half precision
            query_emb = query_emb.astype(np.float16)
            candidate_emb = candidate_emb.astype(np.float16)

        # Compute cosine similarity
        similarity = np.dot(query_emb, candidate_emb.T)
        return similarity.astype(np.float32)

    def retrieve_topk(self, query_embeddings: np.ndarray, candidate_embeddings: np.ndarray,
                      candidate_size: int, k: int, precision: str) -> Tuple[np.ndarray, float]:
        """
        Retrieve top-k candidates (per-query candidate pool).
        Minimal-change fix: ensure each query's ground-truth candidate is in its
        own candidate pool. Assumes 1:1 alignment (i -> i) from dataset builder.
        """

        start_time = time.time()

        num_queries = len(query_embeddings)
        num_candidates_total = len(candidate_embeddings)
        k = int(k)

        # 存放最终的“全局索引”的 top-k 结果（每行对应一个 query）
        top_k_global = np.zeros((num_queries, k), dtype=int)

        # 逐条 query 构造候选：包含其真值 i，再随机补足
        for i in range(num_queries):
            # 真值索引（COCO 构造里是 i -> i；若超界则回退到全随机）
            gt_idx = i if i < num_candidates_total else None

            if num_candidates_total <= candidate_size:
                candidate_idx = np.arange(num_candidates_total)
            else:
                # 预留 1 个位置给真值，其余随机
                need_random = candidate_size - (1 if gt_idx is not None else 0)
                # 可选集合：除去 gt_idx（避免重复）
                if gt_idx is not None:
                    pool = np.delete(np.arange(num_candidates_total), gt_idx)
                    rnd = np.random.choice(pool, size=need_random, replace=False)
                    candidate_idx = np.concatenate([[gt_idx], rnd])
                else:
                    candidate_idx = np.random.choice(np.arange(num_candidates_total),
                                                     size=candidate_size, replace=False)

            # 取出该 query 和它的候选嵌入
            q = query_embeddings[i:i + 1]  # (1, d)
            cand = candidate_embeddings[candidate_idx]  # (C, d)

            # 计算相似度（保持你原有的精度路径）
            sim = self.compute_similarity_matrix(q, cand, precision)  # (1, C)

            # 该 query 的 top-k 在“候选内”的索引
            local_topk = np.argsort(-sim, axis=1)[:, :k]  # (1, k)

            # 映射回“全局候选”索引
            top_k_global[i, :] = candidate_idx[local_topk[0]]

        retrieval_time = time.time() - start_time
        return top_k_global, retrieval_time

    def run_single_experiment(self, text_emb: np.ndarray, video_emb: np.ndarray,
                              ground_truth: List[int], embedding_dim: int,
                              precision: str, candidate_size: int) -> Dict:
        """Run single experiment configuration"""

        logger.info(f"Running experiment: dim={embedding_dim}, precision={precision}, candidates={candidate_size}")

        results = {}
        max_k = max(self.config.recall_k_values)

        # Text-to-Video retrieval
        logger.info("Running Text-to-Video retrieval...")
        t2v_indices, t2v_time = self.retrieve_topk(
            text_emb, video_emb, candidate_size, max_k, precision
        )

        # Calculate T2V metrics
        t2v_metrics = {}
        for k in self.config.recall_k_values:
            t2v_metrics[f'recall@{k}'] = self.metrics_calc.calculate_recall_at_k(
                t2v_indices, ground_truth, k
            )
            t2v_metrics[f'ndcg@{k}'] = self.metrics_calc.calculate_ndcg_at_k(
                t2v_indices, ground_truth, k
            )

        t2v_metrics['mrr'] = self.metrics_calc.calculate_mrr(t2v_indices, ground_truth)

        # Video-to-Text retrieval
        logger.info("Running Video-to-Text retrieval...")
        v2t_indices, v2t_time = self.retrieve_topk(
            video_emb, text_emb, candidate_size, max_k, precision
        )

        # Calculate V2T metrics
        v2t_metrics = {}
        for k in self.config.recall_k_values:
            v2t_metrics[f'recall@{k}'] = self.metrics_calc.calculate_recall_at_k(
                v2t_indices, ground_truth, k
            )
            v2t_metrics[f'ndcg@{k}'] = self.metrics_calc.calculate_ndcg_at_k(
                v2t_indices, ground_truth, k
            )

        v2t_metrics['mrr'] = self.metrics_calc.calculate_mrr(v2t_indices, ground_truth)

        # Memory usage calculation
        memory_usage = self.calculate_memory_usage(text_emb, precision)

        results = {
            'text_to_video': t2v_metrics,
            'video_to_text': v2t_metrics,
            'timing': {
                'text_to_video_time': t2v_time,
                'video_to_text_time': v2t_time,
                'total_time': t2v_time + v2t_time
            },
            'memory_usage_mb': memory_usage,
            'config': {
                'embedding_dim': embedding_dim,
                'precision': precision,
                'candidate_size': candidate_size,
                'num_queries': len(text_emb)
            }
        }

        return results

    def calculate_memory_usage(self, embeddings: np.ndarray, precision: str) -> float:
        """Calculate memory usage in MB"""
        if precision == "fp16":
            bytes_per_element = 2
        elif precision == "int8":
            bytes_per_element = 1
        elif precision == "int4":
            bytes_per_element = 0.5
        else:  # fp32
            bytes_per_element = 4

        return embeddings.size * bytes_per_element / (1024 * 1024)


def load_dataset(data_path: str) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Load dataset from file"""
    logger.info(f"Loading dataset from: {data_path}")

    data = np.load(data_path)
    embeddings = {}

    for key in data.keys():
        if key != 'ground_truth':
            embeddings[key] = data[key]

    ground_truth = data['ground_truth'].tolist()

    logger.info(
        f"Loaded {len(ground_truth)} samples with embedding dims: {[key.split('_')[1] for key in embeddings.keys() if key.startswith('text_')]}")

    return embeddings, ground_truth


def run_full_experiments(config: ExperimentConfig, data_path: str, output_dir: str) -> Dict:
    """Run complete set of experiments"""

    # Load data
    embeddings, ground_truth = load_dataset(data_path)

    # Initialize retriever
    retriever = CrossModalRetriever(config)

    # Calculate total number of experiments
    total_experiments = len(config.embedding_dims) * len(config.precisions) * len(config.candidate_sizes)
    logger.info(f"Running {total_experiments} experiment configurations...")

    results = {}

    with tqdm(total=total_experiments, desc="Experiments") as pbar:
        for embedding_dim in config.embedding_dims:
            for precision in config.precisions:
                for candidate_size in config.candidate_sizes:

                    exp_key = f"dim_{embedding_dim}_prec_{precision}_cand_{candidate_size}"

                    try:
                        # Get embeddings for this dimension
                        text_emb = embeddings[f'text_{embedding_dim}']
                        video_emb = embeddings[f'video_{embedding_dim}']

                        # Run experiment
                        result = retriever.run_single_experiment(
                            text_emb, video_emb, ground_truth,
                            embedding_dim, precision, candidate_size
                        )

                        results[exp_key] = result

                        # Log progress
                        logger.info(
                            f"Completed {exp_key}: T2V R@1={result['text_to_video']['recall@1']:.3f}, V2T R@1={result['video_to_text']['recall@1']:.3f}")

                    except Exception as e:
                        logger.error(f"Failed experiment {exp_key}: {str(e)}")
                        results[exp_key] = {'error': str(e)}

                    pbar.update(1)

    # Save results
    output_path = Path(output_dir) / "experiment_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")

    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Cross-Modal Retrieval Experiments")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--embedding_dims", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--precisions", nargs="+", type=str, default=["fp16", "int8", "int4"])
    parser.add_argument("--candidate_sizes", nargs="+", type=int, default=[32, 100, 1000])
    parser.add_argument("--recall_k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # Create configuration
    config = ExperimentConfig(
        embedding_dims=args.embedding_dims,
        precisions=args.precisions,
        candidate_sizes=args.candidate_sizes,
        recall_k_values=args.recall_k,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    logger.info("Starting Cross-Modal Retrieval Experiments")
    logger.info(f"Configuration: {config}")

    # Run experiments
    results = run_full_experiments(config, args.data_path, args.output_dir)

    logger.info("Experiments completed successfully!")
    print(f"Results saved to: {args.output_dir}/experiment_results.json")


if __name__ == "__main__":
    main()