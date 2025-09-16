#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Results Analysis Script for Cross-Modal Retrieval Experiments
Only uses Recall@10; generates summary + PNG plots.

Inputs
------
- <results_dir>/experiment_results.json  (output of run_experiments.py)

Outputs
-------
- experiment_summary.json
- detailed_results.csv
- summary_by_candidate.csv
- experiment_analysis.png
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze cross-modal retrieval experiment results (Recall@10 only)."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / "experiment_results.json"

    def load_results(self) -> Dict:
        """Load experiment results from JSON file"""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        logger.info(f"Loading results from: {self.results_file}")
        with open(self.results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} experiment configurations")
        return results

    def create_results_dataframe(self, results: Dict) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis (Recall@10 only)."""
        data: List[Dict] = []

        def _get(d: Dict, key: str, default=0.0):
            return d.get(key, default)

        for exp_key, exp_result in results.items():
            if not isinstance(exp_result, dict):
                logger.warning(f"Unexpected result type for {exp_key}: {type(exp_result)}")
                continue
            if 'error' in exp_result:
                logger.warning(f"Skipping failed experiment: {exp_key} | {exp_result.get('error')}")
                continue

            # Parse experiment key: dim_<D>_prec_<p>_cand_<C>
            parts = exp_key.split('_')
            try:
                dim = int(parts[1])
                precision = parts[3]
                candidates = int(parts[5])
            except Exception:
                logger.warning(f"Unexpected experiment key format: {exp_key}")
                continue

            config = exp_result.get('config', {})
            t2v = exp_result.get('text_to_video', {})
            v2t = exp_result.get('video_to_text', {})
            timing = exp_result.get('timing', {})

            num_q = int(config.get('num_queries', 0))
            total_time = float(timing.get('total_time', 0.0))
            qps = (num_q / total_time) if total_time > 0 else 0.0

            row = {
                'experiment': exp_key,
                'embedding_dim': dim,
                'precision': precision,
                'candidate_size': candidates,
                'num_samples': num_q,

                # Recall@10 only
                't2v_recall@10': float(_get(t2v, 'recall@10', 0.0)),
                'v2t_recall@10': float(_get(v2t, 'recall@10', 0.0)),
                'avg_recall@10': 0.5 * (
                    float(_get(t2v, 'recall@10', 0.0)) + float(_get(v2t, 'recall@10', 0.0))
                ),

                # MRR & NDCG@10 (optional in results)
                't2v_mrr': float(_get(t2v, 'mrr', 0.0)),
                'v2t_mrr': float(_get(v2t, 'mrr', 0.0)),
                'avg_mrr': 0.5 * (float(_get(t2v, 'mrr', 0.0)) + float(_get(v2t, 'mrr', 0.0))),
                't2v_ndcg@10': float(_get(t2v, 'ndcg@10', 0.0)),
                'v2t_ndcg@10': float(_get(v2t, 'ndcg@10', 0.0)),
                'avg_ndcg@10': 0.5 * (float(_get(t2v, 'ndcg@10', 0.0)) + float(_get(v2t, 'ndcg@10', 0.0))),

                # Timing / memory
                't2v_time': float(timing.get('text_to_video_time', 0.0)),
                'v2t_time': float(timing.get('video_to_text_time', 0.0)),
                'total_time': total_time,
                'memory_usage_mb': float(exp_result.get('memory_usage_mb', 0.0)),

                # Efficiency
                'queries_per_second': qps,
                'mb_per_query': (float(exp_result.get('memory_usage_mb', 0.0)) / num_q) if num_q > 0 else 0.0,
            }

            data.append(row)

        df = pd.DataFrame(data)
        logger.info(f"Created DataFrame with {len(df)} experiments")
        return df

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics (Recall@10 only)."""
        if len(df) == 0:
            return {"error": "No experiments to summarize."}

        summary: Dict = {
            'total_experiments': int(len(df)),
            'embedding_dimensions': sorted(df['embedding_dim'].unique().tolist()),
            'precisions': sorted(df['precision'].unique().tolist()),
            'candidate_sizes': sorted(df['candidate_size'].unique().tolist()),
        }

        # Best configs (use Recall@10 only)
        best_r10_idx = df['avg_recall@10'].idxmax()
        best_qps_idx = df['queries_per_second'].idxmax()
        best_mem_idx = df['mb_per_query'].idxmin()

        summary['best_performance'] = {
            'highest_avg_recall@10': {
                'value': float(df.loc[best_r10_idx, 'avg_recall@10']),
                'experiment': df.loc[best_r10_idx, 'experiment'],
                'config': {
                    'dim': int(df.loc[best_r10_idx, 'embedding_dim']),
                    'precision': df.loc[best_r10_idx, 'precision'],
                    'candidates': int(df.loc[best_r10_idx, 'candidate_size'])
                }
            },
            'fastest_retrieval': {
                'value': float(df.loc[best_qps_idx, 'queries_per_second']),
                'experiment': df.loc[best_qps_idx, 'experiment'],
                'config': {
                    'dim': int(df.loc[best_qps_idx, 'embedding_dim']),
                    'precision': df.loc[best_qps_idx, 'precision'],
                    'candidates': int(df.loc[best_qps_idx, 'candidate_size'])
                }
            },
            'most_memory_efficient': {
                'value': float(df.loc[best_mem_idx, 'mb_per_query']),
                'experiment': df.loc[best_mem_idx, 'experiment'],
                'config': {
                    'dim': int(df.loc[best_mem_idx, 'embedding_dim']),
                    'precision': df.loc[best_mem_idx, 'precision'],
                    'candidates': int(df.loc[best_mem_idx, 'candidate_size'])
                }
            }
        }

        # Aggregations (Recall@10)
        summary['performance_by_dim'] = df.groupby('embedding_dim')['avg_recall@10'].mean().to_dict()
        summary['performance_by_precision'] = df.groupby('precision')['avg_recall@10'].mean().to_dict()
        summary['performance_by_candidates'] = df.groupby('candidate_size')['avg_recall@10'].mean().to_dict()

        # Speed & memory aggregations
        summary['speed_by_dim'] = df.groupby('embedding_dim')['queries_per_second'].mean().to_dict()
        summary['speed_by_precision'] = df.groupby('precision')['queries_per_second'].mean().to_dict()
        summary['speed_by_candidates'] = df.groupby('candidate_size')['queries_per_second'].mean().to_dict()
        summary['memory_by_dim'] = df.groupby('embedding_dim')['mb_per_query'].mean().to_dict()
        summary['memory_by_precision'] = df.groupby('precision')['mb_per_query'].mean().to_dict()

        return summary

    def create_visualizations(self, df: pd.DataFrame):
        """Create visualization plots (all based on Recall@10)."""
        plt.style.use('default')
        sns.set_palette("husl")

        # 2x3 dashboard (保持你原始布局，只把 y 改成 avg_recall@10)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cross-Modal Retrieval Experiment Results (Recall@10)', fontsize=16)

        # Plot 1: Recall@10 by embedding dimension
        sns.boxplot(data=df, x='embedding_dim', y='avg_recall@10', ax=axes[0, 0])
        axes[0, 0].set_title('Average Recall@10 by Embedding Dimension')
        axes[0, 0].set_xlabel('Embedding Dimension')
        axes[0, 0].set_ylabel('Average Recall@10')

        # Plot 2: Performance by precision
        sns.boxplot(data=df, x='precision', y='avg_recall@10', ax=axes[0, 1])
        axes[0, 1].set_title('Average Recall@10 by Precision')
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_ylabel('Average Recall@10')

        # Plot 3: Performance by candidate size
        sns.boxplot(data=df, x='candidate_size', y='avg_recall@10', ax=axes[0, 2])
        axes[0, 2].set_title('Average Recall@10 by Candidate Size')
        axes[0, 2].set_xlabel('Candidate Size')
        axes[0, 2].set_ylabel('Average Recall@10')

        # Plot 4: Speed vs Accuracy (Recall@10)
        sns.scatterplot(data=df, x='queries_per_second', y='avg_recall@10',
                        hue='precision', size='embedding_dim', ax=axes[1, 0])
        axes[1, 0].set_title('Speed vs Accuracy (Recall@10)')
        axes[1, 0].set_xlabel('Queries per Second')
        axes[1, 0].set_ylabel('Average Recall@10')

        # Plot 5: Memory usage by configuration
        pivot_memory = df.pivot_table(values='memory_usage_mb',
                                      index='embedding_dim',
                                      columns='precision',
                                      aggfunc='mean')
        sns.heatmap(pivot_memory, annot=True, fmt='.1f', ax=axes[1, 1])
        axes[1, 1].set_title('Memory Usage (MB) by Configuration')

        # Plot 6: Text-to-Video vs Video-to-Text performance (Recall@10)
        sns.scatterplot(data=df, x='t2v_recall@10', y='v2t_recall@10',
                        hue='embedding_dim', style='precision', ax=axes[1, 2])
        axes[1, 2].set_title('T2V vs V2T Performance (Recall@10)')
        axes[1, 2].set_xlabel('Text-to-Video Recall@10')
        axes[1, 2].set_ylabel('Video-to-Text Recall@10')
        axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line

        plt.tight_layout()
        plot_path = self.results_dir / "experiment_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualizations saved to: {plot_path}")
        plt.close()

    def print_summary_report(self, summary: Dict):
        """Print a formatted summary report (Recall@10 only)."""

        print("\n" + "=" * 80)
        print("CROSS-MODAL RETRIEVAL EXPERIMENT SUMMARY (Recall@10)")
        print("=" * 80)

        if "error" in summary:
            print("No experiments to summarize.")
            print("\n" + "=" * 80)
            return

        print(f"\nDataset Overview:")
        print(f"  Total experiments: {summary['total_experiments']}")
        print(f"  Embedding dimensions: {summary['embedding_dimensions']}")
        print(f"  Precisions tested: {summary['precisions']}")
        print(f"  Candidate sizes: {summary['candidate_sizes']}")

        print(f"\nBest Performance:")
        best = summary['best_performance']
        print(f"  Highest Average Recall@10: {best['highest_avg_recall@10']['value']:.4f}")
        print(f"    Configuration: {best['highest_avg_recall@10']['config']}")
        print(f"  Fastest Retrieval: {best['fastest_retrieval']['value']:.2f} queries/sec")
        print(f"    Configuration: {best['fastest_retrieval']['config']}")
        print(f"  Most Memory Efficient: {best['most_memory_efficient']['value']:.4f} MB/query")
        print(f"    Configuration: {best['most_memory_efficient']['config']}")

        print("\n" + "=" * 80)

    def analyze_results(self, output_path: str):
        """Main analysis function"""
        logger.info("Starting results analysis.")

        # Load results
        results = self.load_results()

        # Create DataFrame
        df = self.create_results_dataframe(results)

        # Generate summary statistics
        summary = self.generate_summary_statistics(df)

        # Create visualizations
        self.create_visualizations(df)

        # Save detailed results
        csv_path = self.results_dir / "detailed_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Detailed results saved to: {csv_path}")

        # Save summary
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to: {output_path}")

        # Also dump per-candidate CSV (Recall@10 & MRR)
        agg = (df.groupby('candidate_size')[['avg_recall@10', 'avg_mrr']]
               .mean().reset_index().sort_values('candidate_size'))
        csv2 = self.results_dir / "summary_by_candidate.csv"
        agg.to_csv(csv2, index=False)
        logger.info(f"Wrote {csv2}")

        # Print report
        self.print_summary_report(summary)

        logger.info("Analysis completed successfully!")


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description="Analyze cross-modal retrieval results (Recall@10 only)")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing experiment_results.json")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save analysis summary")
    args = parser.parse_args()

    analyzer = ResultsAnalyzer(args.results_dir)
    analyzer.analyze_results(args.output_path)


if __name__ == "__main__":
    main()
