"""
Results Analysis Script for Cross-Modal Retrieval Experiments
Analyzes and summarizes experimental results
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze cross-modal retrieval experiment results"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / "experiment_results.json"

    def load_results(self) -> Dict:
        """Load experiment results from JSON file"""

        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        logger.info(f"Loading results from: {self.results_file}")

        with open(self.results_file, 'r') as f:
            results = json.load(f)

        logger.info(f"Loaded {len(results)} experiment configurations")
        return results

    def create_results_dataframe(self, results: Dict) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""

        data = []

        for exp_key, exp_result in results.items():
            if 'error' in exp_result:
                logger.warning(f"Skipping failed experiment: {exp_key}")
                continue

            # Parse experiment key
            parts = exp_key.split('_')
            dim = int(parts[1])
            precision = parts[3]
            candidates = int(parts[5])

            # Extract metrics
            config = exp_result['config']
            t2v = exp_result['text_to_video']
            v2t = exp_result['video_to_text']
            timing = exp_result['timing']

            # Create row
            row = {
                'experiment': exp_key,
                'embedding_dim': dim,
                'precision': precision,
                'candidate_size': candidates,
                'num_samples': config['num_queries'],

                # Text-to-Video metrics
                't2v_recall@1': t2v['recall@1'],
                't2v_recall@5': t2v['recall@5'],
                't2v_recall@10': t2v['recall@10'],
                't2v_mrr': t2v['mrr'],
                't2v_ndcg@1': t2v.get('ndcg@1', 0),
                't2v_ndcg@5': t2v.get('ndcg@5', 0),
                't2v_ndcg@10': t2v.get('ndcg@10', 0),

                # Video-to-Text metrics
                'v2t_recall@1': v2t['recall@1'],
                'v2t_recall@5': v2t['recall@5'],
                'v2t_recall@10': v2t['recall@10'],
                'v2t_mrr': v2t['mrr'],
                'v2t_ndcg@1': v2t.get('ndcg@1', 0),
                'v2t_ndcg@5': v2t.get('ndcg@5', 0),
                'v2t_ndcg@10': v2t.get('ndcg@10', 0),

                # Average metrics
                'avg_recall@1': (t2v['recall@1'] + v2t['recall@1']) / 2,
                'avg_recall@5': (t2v['recall@5'] + v2t['recall@5']) / 2,
                'avg_recall@10': (t2v['recall@10'] + v2t['recall@10']) / 2,
                'avg_mrr': (t2v['mrr'] + v2t['mrr']) / 2,

                # Performance metrics
                't2v_time': timing['text_to_video_time'],
                'v2t_time': timing['video_to_text_time'],
                'total_time': timing['total_time'],
                'memory_usage_mb': exp_result['memory_usage_mb'],

                # Efficiency metrics
                'queries_per_second': config['num_queries'] / timing['total_time'],
                'mb_per_query': exp_result['memory_usage_mb'] / config['num_queries']
            }

            data.append(row)

        df = pd.DataFrame(data)
        logger.info(f"Created DataFrame with {len(df)} experiments")
        return df

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics"""

        summary = {
            'total_experiments': len(df),
            'embedding_dimensions': sorted(df['embedding_dim'].unique().tolist()),
            'precisions': sorted(df['precision'].unique().tolist()),
            'candidate_sizes': sorted(df['candidate_size'].unique().tolist()),

            'best_performance': {
                'highest_avg_recall@1': {
                    'value': df['avg_recall@1'].max(),
                    'experiment': df.loc[df['avg_recall@1'].idxmax(), 'experiment'],
                    'config': {
                        'dim': int(df.loc[df['avg_recall@1'].idxmax(), 'embedding_dim']),
                        'precision': df.loc[df['avg_recall@1'].idxmax(), 'precision'],
                        'candidates': int(df.loc[df['avg_recall@1'].idxmax(), 'candidate_size'])
                    }
                },
                'fastest_retrieval': {
                    'value': df['queries_per_second'].max(),
                    'experiment': df.loc[df['queries_per_second'].idxmax(), 'experiment'],
                    'config': {
                        'dim': int(df.loc[df['queries_per_second'].idxmax(), 'embedding_dim']),
                        'precision': df.loc[df['queries_per_second'].idxmax(), 'precision'],
                        'candidates': int(df.loc[df['queries_per_second'].idxmax(), 'candidate_size'])
                    }
                },
                'most_memory_efficient': {
                    'value': df['mb_per_query'].min(),
                    'experiment': df.loc[df['mb_per_query'].idxmin(), 'experiment'],
                    'config': {
                        'dim': int(df.loc[df['mb_per_query'].idxmin(), 'embedding_dim']),
                        'precision': df.loc[df['mb_per_query'].idxmin(), 'precision'],
                        'candidates': int(df.loc[df['mb_per_query'].idxmin(), 'candidate_size'])
                    }
                }
            }
        }

        # Performance by configuration
        summary['performance_by_dim'] = df.groupby('embedding_dim')['avg_recall@1'].mean().to_dict()
        summary['performance_by_precision'] = df.groupby('precision')['avg_recall@1'].mean().to_dict()
        summary['performance_by_candidates'] = df.groupby('candidate_size')['avg_recall@1'].mean().to_dict()

        # Speed by configuration
        summary['speed_by_dim'] = df.groupby('embedding_dim')['queries_per_second'].mean().to_dict()
        summary['speed_by_precision'] = df.groupby('precision')['queries_per_second'].mean().to_dict()
        summary['speed_by_candidates'] = df.groupby('candidate_size')['queries_per_second'].mean().to_dict()

        # Memory by configuration
        summary['memory_by_dim'] = df.groupby('embedding_dim')['mb_per_query'].mean().to_dict()
        summary['memory_by_precision'] = df.groupby('precision')['mb_per_query'].mean().to_dict()

        return summary

    def create_visualizations(self, df: pd.DataFrame):
        """Create visualization plots"""

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cross-Modal Retrieval Experiment Results', fontsize=16)

        # Plot 1: Recall@1 by embedding dimension
        sns.boxplot(data=df, x='embedding_dim', y='avg_recall@1', ax=axes[0, 0])
        axes[0, 0].set_title('Average Recall@1 by Embedding Dimension')
        axes[0, 0].set_xlabel('Embedding Dimension')
        axes[0, 0].set_ylabel('Average Recall@1')

        # Plot 2: Performance by precision
        sns.boxplot(data=df, x='precision', y='avg_recall@1', ax=axes[0, 1])
        axes[0, 1].set_title('Average Recall@1 by Precision')
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_ylabel('Average Recall@1')

        # Plot 3: Performance by candidate size
        sns.boxplot(data=df, x='candidate_size', y='avg_recall@1', ax=axes[0, 2])
        axes[0, 2].set_title('Average Recall@1 by Candidate Size')
        axes[0, 2].set_xlabel('Candidate Size')
        axes[0, 2].set_ylabel('Average Recall@1')

        # Plot 4: Speed vs Accuracy trade-off
        sns.scatterplot(data=df, x='queries_per_second', y='avg_recall@1',
                        hue='precision', size='embedding_dim', ax=axes[1, 0])
        axes[1, 0].set_title('Speed vs Accuracy Trade-off')
        axes[1, 0].set_xlabel('Queries per Second')
        axes[1, 0].set_ylabel('Average Recall@1')

        # Plot 5: Memory usage by configuration
        pivot_memory = df.pivot_table(values='memory_usage_mb',
                                      index='embedding_dim',
                                      columns='precision',
                                      aggfunc='mean')
        sns.heatmap(pivot_memory, annot=True, fmt='.1f', ax=axes[1, 1])
        axes[1, 1].set_title('Memory Usage (MB) by Configuration')

        # Plot 6: Text-to-Video vs Video-to-Text performance
        sns.scatterplot(data=df, x='t2v_recall@1', y='v2t_recall@1',
                        hue='embedding_dim', style='precision', ax=axes[1, 2])
        axes[1, 2].set_title('T2V vs V2T Performance')
        axes[1, 2].set_xlabel('Text-to-Video Recall@1')
        axes[1, 2].set_ylabel('Video-to-Text Recall@1')
        axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line

        plt.tight_layout()

        # Save plot
        plot_path = self.results_dir / "experiment_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualizations saved to: {plot_path}")

        plt.close()

    def print_summary_report(self, summary: Dict):
        """Print a formatted summary report"""

        print("\n" + "=" * 80)
        print("CROSS-MODAL RETRIEVAL EXPERIMENT SUMMARY")
        print("=" * 80)

        print(f"\nDataset Overview:")
        print(f"  Total experiments: {summary['total_experiments']}")
        print(f"  Embedding dimensions: {summary['embedding_dimensions']}")
        print(f"  Precisions tested: {summary['precisions']}")
        print(f"  Candidate sizes: {summary['candidate_sizes']}")

        print(f"\nBest Performance:")
        best = summary['best_performance']
        print(f"  Highest Average Recall@1: {best['highest_avg_recall@1']['value']:.4f}")
        print(f"    Configuration: {best['highest_avg_recall@1']['config']}")

        print(f"  Fastest Retrieval: {best['fastest_retrieval']['value']:.2f} queries/sec")
        print(f"    Configuration: {best['fastest_retrieval']['config']}")

        print(f"  Most Memory Efficient: {best['most_memory_efficient']['value']:.4f} MB/query")
        print(f"    Configuration: {best['most_memory_efficient']['config']}")

        print(f"\nPerformance by Embedding Dimension:")
        for dim, perf in summary['performance_by_dim'].items():
            print(f"  {dim}D: {perf:.4f} avg recall@1")

        print(f"\nPerformance by Precision:")
        for prec, perf in summary['performance_by_precision'].items():
            print(f"  {prec}: {perf:.4f} avg recall@1")

        print(f"\nPerformance by Candidate Size:")
        for size, perf in summary['performance_by_candidates'].items():
            print(f"  {size} candidates: {perf:.4f} avg recall@1")

        print("\n" + "=" * 80)

    def analyze_results(self, output_path: str):
        """Main analysis function"""

        logger.info("Starting results analysis...")

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
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to: {output_path}")

        # Print report
        self.print_summary_report(summary)

        logger.info("Analysis completed successfully!")


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description="Analyze cross-modal retrieval results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing experiment_results.json")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save analysis summary")

    args = parser.parse_args()

    # Create analyzer
    analyzer = ResultsAnalyzer(args.results_dir)

    # Run analysis
    analyzer.analyze_results(args.output_path)


if __name__ == "__main__":
    main()