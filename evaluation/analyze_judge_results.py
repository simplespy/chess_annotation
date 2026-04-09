#!/usr/bin/env python3
"""
Analyze judged commentary results from JSONL files.

Computes aggregate metrics and breakdowns across multiple dimensions.

Usage:
    python analyze_judge_results.py outputs_judge/judged_gpt4o_1.jsonl
    python analyze_judge_results.py outputs_judge/*.jsonl
    python analyze_judge_results.py --by-quality outputs_judge/judged_gpt4o_1.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import List, Dict
import numpy as np


def load_judged_results(file_path: str) -> List[dict]:
    """Load judged results from JSONL file."""
    with open(file_path) as f:
        return [json.loads(line) for line in f]


def compute_metrics(results: List[dict]) -> Dict:
    """Compute aggregate metrics from judged results."""
    if not results:
        return {}

    metrics = {
        'n_positions': len(results),
        'claim_accuracy': np.mean([r['judge_results']['claim_accuracy'] for r in results]),
        'recall': np.mean([r['judge_results']['recall'] for r in results]),
        'quality_correct': np.mean([r['judge_results']['quality_correct'] for r in results]),
        'fluency': np.mean([r['judge_results']['fluency'] for r in results]),
        'specificity': np.mean([r['judge_results']['specificity'] for r in results]),
        'composite': np.mean([r['judge_results']['composite'] for r in results]),
        'avg_claims_per_position': np.mean([r['judge_results']['n_claims'] for r in results]),
        'avg_atoms_per_position': np.mean([r['judge_results']['n_atoms'] for r in results]),
        'total_sanity_overrides': sum(r['judge_results']['n_sanity_overrides'] for r in results),
        'avg_alternatives_per_position': np.mean([r['judge_results']['n_alternatives'] for r in results]),
    }

    # Standard deviations
    metrics['claim_accuracy_std'] = np.std([r['judge_results']['claim_accuracy'] for r in results])
    metrics['recall_std'] = np.std([r['judge_results']['recall'] for r in results])
    metrics['composite_std'] = np.std([r['judge_results']['composite'] for r in results])

    return metrics


def breakdown_by_quality(results: List[dict]) -> Dict[str, Dict]:
    """Break down metrics by move quality."""
    by_quality = defaultdict(list)

    for r in results:
        quality = r.get('quality', 'unknown')
        by_quality[quality].append(r)

    breakdown = {}
    for quality, subset in by_quality.items():
        breakdown[quality] = compute_metrics(subset)

    return breakdown


def breakdown_by_wp_loss(results: List[dict]) -> Dict[str, Dict]:
    """Break down metrics by wp_loss bins."""
    bins = {
        'excellent (0-10)': [],
        'good (10-30)': [],
        'inaccuracy (30-100)': [],
        'mistake (100-300)': [],
        'blunder (300+)': [],
    }

    for r in results:
        wp_loss = abs(r.get('wp_loss', 0))
        if wp_loss < 10:
            bins['excellent (0-10)'].append(r)
        elif wp_loss < 30:
            bins['good (10-30)'].append(r)
        elif wp_loss < 100:
            bins['inaccuracy (30-100)'].append(r)
        elif wp_loss < 300:
            bins['mistake (100-300)'].append(r)
        else:
            bins['blunder (300+)'].append(r)

    breakdown = {}
    for bin_name, subset in bins.items():
        if subset:
            breakdown[bin_name] = compute_metrics(subset)

    return breakdown


def print_metrics(metrics: Dict, indent: int = 0):
    """Pretty-print metrics."""
    prefix = "  " * indent

    print(f"{prefix}Positions: {metrics['n_positions']}")
    print(f"{prefix}Claim Accuracy: {metrics['claim_accuracy']:.1%} (±{metrics['claim_accuracy_std']:.1%})")
    print(f"{prefix}Recall: {metrics['recall']:.1%} (±{metrics['recall_std']:.1%})")
    print(f"{prefix}Quality Correct: {metrics['quality_correct']:.1%}")
    print(f"{prefix}Fluency: {metrics['fluency']:.2f} / 5.0")
    print(f"{prefix}Specificity: {metrics['specificity']:.2f} / 5.0")
    print(f"{prefix}Composite Score: {metrics['composite']:.3f} (±{metrics['composite_std']:.3f})")
    print(f"{prefix}Avg Claims/Position: {metrics['avg_claims_per_position']:.1f}")
    print(f"{prefix}Avg Atoms/Position: {metrics['avg_atoms_per_position']:.1f}")
    print(f"{prefix}Total Sanity Overrides: {metrics['total_sanity_overrides']}")
    print(f"{prefix}Avg Alternatives/Position: {metrics['avg_alternatives_per_position']:.1f}")


def analyze_file(file_path: str, by_quality: bool = False, by_wp_loss: bool = False):
    """Analyze a single judged results file."""
    print(f"\n{'='*80}")
    print(f"FILE: {file_path}")
    print(f"{'='*80}")

    results = load_judged_results(file_path)

    # Overall metrics
    print("\nOVERALL METRICS")
    print("-" * 80)
    overall = compute_metrics(results)
    print_metrics(overall)

    # By quality
    if by_quality:
        print("\n\nBREAKDOWN BY QUALITY")
        print("-" * 80)
        quality_breakdown = breakdown_by_quality(results)
        for quality in sorted(quality_breakdown.keys()):
            print(f"\n{quality.upper()}:")
            print_metrics(quality_breakdown[quality], indent=1)

    # By wp_loss
    if by_wp_loss:
        print("\n\nBREAKDOWN BY WP_LOSS")
        print("-" * 80)
        wp_breakdown = breakdown_by_wp_loss(results)
        for bin_name in ['excellent (0-10)', 'good (10-30)', 'inaccuracy (30-100)',
                         'mistake (100-300)', 'blunder (300+)']:
            if bin_name in wp_breakdown:
                print(f"\n{bin_name.upper()}:")
                print_metrics(wp_breakdown[bin_name], indent=1)

    return overall


def compare_files(file_paths: List[str]):
    """Compare metrics across multiple files."""
    print(f"\n{'='*80}")
    print(f"COMPARISON ACROSS {len(file_paths)} FILES")
    print(f"{'='*80}\n")

    all_metrics = {}
    for file_path in file_paths:
        results = load_judged_results(file_path)
        all_metrics[file_path] = compute_metrics(results)

    # Print comparison table
    print(f"{'File':<40} {'N':<6} {'Claim%':<8} {'Recall%':<9} {'Qual%':<7} {'Comp':<7}")
    print("-" * 80)

    for file_path, metrics in all_metrics.items():
        file_name = file_path.split('/')[-1][:38]
        print(f"{file_name:<40} "
              f"{metrics['n_positions']:<6} "
              f"{metrics['claim_accuracy']:>6.1%}  "
              f"{metrics['recall']:>7.1%}  "
              f"{metrics['quality_correct']:>5.1%}  "
              f"{metrics['composite']:>6.3f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze judged commentary results')
    parser.add_argument('files', nargs='+', help='Judged JSONL file(s) to analyze')
    parser.add_argument('--by-quality', action='store_true',
                       help='Break down metrics by quality label')
    parser.add_argument('--by-wp-loss', action='store_true',
                       help='Break down metrics by wp_loss bins')
    parser.add_argument('--compare', action='store_true',
                       help='Compare metrics across multiple files (table view)')

    args = parser.parse_args()

    if args.compare and len(args.files) > 1:
        compare_files(args.files)
    else:
        for file_path in args.files:
            analyze_file(file_path, by_quality=args.by_quality, by_wp_loss=args.by_wp_loss)

    print()  # Final newline


if __name__ == '__main__':
    main()
