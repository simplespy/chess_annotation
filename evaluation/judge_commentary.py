#!/usr/bin/env python3
"""
Commentary judging CLI for evaluating chess explanations.

Evaluates generated commentary using the full improved pipeline with:
- Atomic claim verification
- Tool-based fact-checking
- Sanity checks
- Gold atom matching
- Multi-dimensional scoring
- Automatic filtering of <think> tags (used by Qwen models)

Usage:
    # Judge commentary in a file with generated text
    python judge_commentary.py --input generated.jsonl --output judged.jsonl

    # Use specific provider/model for judging
    python judge_commentary.py --input generated.jsonl --provider anthropic \
        --model claude-sonnet-4-5-20250929

    # Use Qwen for judging
    python judge_commentary.py --input generated.jsonl --provider qwen \
        --model Qwen/Qwen3-32B --base-url http://localhost:8000/v1
"""

import argparse
import json
import os
import re
import subprocess
import sys
import chess

from judge import judge_explanation_improved


def remove_think_tags(text):
    """
    Remove <think>...</think> tags from text.

    Used for Qwen models that output reasoning/thinking content wrapped in
    <think> tags that should be excluded from the actual commentary.

    Args:
        text: Input text potentially containing <think> tags

    Returns:
        Text with all <think>...</think> blocks removed
    """
    # Remove <think>...</think> blocks (case-insensitive, handles multiline)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Clean up any extra whitespace left behind
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned).strip()
    return cleaned


def judge_file(input_path, output_path, commentary_field='generated_commentary',
              provider="openai", model="gpt-4o", base_url=None, api_key=None,
              verbose=True):
    """
    Judge commentary for positions in a JSONL file.

    Args:
        input_path: Input JSONL file with positions and commentary
        output_path: Output JSONL file for results
        commentary_field: Field name containing the commentary to judge
        provider: LLM provider ('openai', 'anthropic', 'qwen')
        model: Model name
        base_url: Base URL for OpenAI-compatible APIs (for Qwen)
        api_key: API key (optional)
        verbose: Print detailed progress
    """
    with open(input_path) as f:
        entries = [json.loads(line) for line in f]

    print(f"Loaded {len(entries)} entries from {input_path}")
    print(f"Provider: {provider}, Model: {model}")
    print(f"Commentary field: {commentary_field}")

    with open(output_path, 'w') as out_f:
        for i, entry_data in enumerate(entries):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(entries)}] Judging position...")
            print(f"{'='*60}")

            # Extract commentary
            commentary = entry_data.get(commentary_field)
            if not commentary:
                print(f"  WARNING: No commentary found in field '{commentary_field}', skipping")
                continue

            # Filter out <think> tags (used by Qwen models)
            original_commentary = commentary
            commentary = remove_think_tags(commentary)
            if commentary != original_commentary:
                if verbose:
                    print(f"  Filtered <think> tags from commentary")

            # Build entry for judging
            board = chess.Board(entry_data['fen'])
            move_san = entry_data.get('move_san', board.san(chess.Move.from_uci(entry_data['move_uci'])))
            entry = {
                'fen': entry_data['fen'],
                'move_uci': entry_data['move_uci'],
                'move_san': move_san,
                'wp_loss': entry_data.get('wp_loss', 0),
                'extracted': entry_data.get('extracted', {}),
            }

            # Get gold atoms
            gold_atoms = entry_data.get('extracted', {}).get('reasoning', [])
            print(f"  Gold atoms: {len(gold_atoms)}")
            print(f"  Commentary: {commentary[:100]}...")

            # Judge
            results = judge_explanation_improved(
                entry, commentary, gold_atoms=gold_atoms,
                provider=provider, model=model, verbose=verbose,
                base_url=base_url, api_key=api_key)

            # Combine with input
            output_entry = {
                **entry_data,
                'judge_results': {
                    'claim_accuracy': results['claim_accuracy'],
                    'n_claims': results['n_claims'],
                    'n_correct_claims': results['n_correct_claims'],
                    'recall': results['recall'],
                    'quality_correct': results['quality_correct'],
                    'gen_assessment': results['gen_assessment'],
                    'actual_quality': results['actual_quality'],
                    'fluency': results['fluency'],
                    'specificity': results['specificity'],
                    'composite': results['composite'],
                    'n_atoms': len(results['atoms']),
                    'n_verified': sum(1 for r in results['verification'] if r.get('verified')),
                    'n_sanity_overrides': results['n_sanity_overrides'],
                    'n_alternatives': len(results.get('alternatives', [])),
                },
                'think_tags_filtered': commentary != original_commentary,
                'filtered_commentary': commentary if commentary != original_commentary else None,
                'atoms': results['atoms'],
                'claims': results['claims'],
                'verification': results['verification'],
                'alternatives': results.get('alternatives', []),
                'judge_model': model,
                'judge_provider': provider,
            }

            out_f.write(json.dumps(output_entry) + '\n')
            out_f.flush()

            print(f"\n  Results:")
            print(f"    Claim Accuracy: {results['claim_accuracy']:.0%} ({results['n_correct_claims']}/{results['n_claims']})")
            print(f"    Recall: {results['recall']:.0%}")
            print(f"    Quality: {'✓' if results['quality_correct'] else '✗'} ({results['gen_assessment']} vs {results['actual_quality']})")
            print(f"    Fluency: {results['fluency']:.2f}/5")
            print(f"    Specificity: {results['specificity']:.2f}/5")
            print(f"    Composite: {results['composite']:.3f}")
            print(f"    Sanity Overrides: {results['n_sanity_overrides']}")
            if results.get('alternatives'):
                print(f"    Alternatives: {len(results['alternatives'])}")

    print(f"\n✓ Wrote {len(entries)} results to {output_path}")

    # Run analysis and save to file
    output_dir = os.path.dirname(output_path) or '.'
    output_basename = os.path.basename(output_path)
    output_name = os.path.splitext(output_basename)[0]
    analysis_path = os.path.join(output_dir, f"analysis_{output_name}.txt")

    print(f"\nRunning analysis...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        analyze_script = os.path.join(script_dir, 'analyze_judge_results.py')

        with open(analysis_path, 'w') as f:
            subprocess.run(
                ['python3', analyze_script, output_path],
                stdout=f,
                stderr=subprocess.PIPE,
                check=True
            )
        print(f"✓ Analysis saved to {analysis_path}")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Analysis failed: {e.stderr.decode()}")
    except Exception as e:
        print(f"⚠ Analysis failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Judge chess move commentary')
    parser.add_argument('--input', required=True, help='Input JSONL file with positions and commentary')
    parser.add_argument('--output', required=True, help='Output JSONL file for results')
    parser.add_argument('--commentary-field', default='generated_commentary',
                       help='Field name containing commentary to judge')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'qwen'], default='openai',
                       help='LLM provider')
    parser.add_argument('--model', default='gpt-4o', help='Model name')
    parser.add_argument('--base-url', help='Base URL for OpenAI-compatible API (for Qwen)')
    parser.add_argument('--api-key', help='API key (optional)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')

    args = parser.parse_args()

    judge_file(
        args.input,
        args.output,
        commentary_field=args.commentary_field,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
