#!/usr/bin/env python3
"""
Batch evaluation CLI for chess commentary generation and judging.

Usage:
    python batch_evaluate.py --input data.jsonl --n 10 --provider openai --model gpt-4o
    python batch_evaluate.py --input data.jsonl --indices 0 1 2 3 --provider anthropic
    python batch_evaluate.py --input data.jsonl --n 20 --provider qwen --base-url http://localhost:8000/v1
"""

import argparse
import json
import re
import random
import sys
import numpy as np
import chess

from generation import generate_commentary, generate_commentary_raw
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


def batch_evaluate(accepted_positions, indices=None, n=10, seed=42,
                   provider="openai", model="gpt-4o", gen_model=None,
                   output_path=None, use_engine=False, use_tools=True,
                   base_url=None, api_key=None):
    """
    Run generation + improved judge on multiple positions.

    This function:
    1. Selects positions to evaluate (random sample or specified indices)
    2. For each position:
       - Generates commentary (with or without engine lines)
       - Judges with full improved pipeline
       - Saves results incrementally if output_path provided
    3. Computes aggregate metrics

    Args:
        accepted_positions: List of test positions
        indices: specific indices to evaluate (overrides n/seed)
        n: number of random positions
        seed: random seed
        provider: LLM provider ('openai', 'anthropic', 'qwen')
        model: judge model
        gen_model: generation model (defaults to model)
        output_path: optional JSONL path to save results
        use_engine: if False, build generation prompts without engine lines
        use_tools: Enable tool calling in generation
        base_url: Base URL for OpenAI-compatible APIs (for Qwen)
        api_key: API key (optional)

    Returns:
        List of result dicts with metrics for each position
    """
    gen_model = gen_model or model

    if indices is None:
        rng = random.Random(seed)
        indices = rng.sample(range(len(accepted_positions)), min(n, len(accepted_positions)))

    all_results = []

    for idx in indices:
        pos = accepted_positions[idx]
        board = chess.Board(pos['fen'])
        move_san = pos.get('move_san', board.san(chess.Move.from_uci(pos['move_uci'])))
        print(f"\n{'='*60}")
        print(f"[{len(all_results)+1}/{len(indices)}] idx={idx} — {pos.get('game','?')} — {move_san}")
        print(f"{'='*60}")

        el = pos.get('engine_lines') if use_engine else None
        entry = {'fen': pos['fen'], 'move_uci': pos['move_uci'],
                 'annotation': pos.get('annotation', ''), 'wp_loss': pos.get('wp_loss', 0),
                 'move_san': move_san, 'extracted': pos.get('extracted', {})}

        prompt_mode = "engine" if use_engine else "raw"
        print(f"  Generating ({provider}/{gen_model}, {prompt_mode})...")
        if use_engine:
            gen_text, gen_log = generate_commentary(entry, el, provider=provider,
                                                    model=gen_model, use_tools=use_tools,
                                                    base_url=base_url, api_key=api_key)
        else:
            gen_text, gen_log = generate_commentary_raw(entry, provider=provider,
                                                        model=gen_model, use_tools=use_tools,
                                                        base_url=base_url, api_key=api_key)
        print(f"  Generated: {gen_text[:100]}...")

        # Filter out <think> tags (used by Qwen models)
        original_gen_text = gen_text
        gen_text_for_judging = remove_think_tags(gen_text)
        if gen_text_for_judging != original_gen_text:
            print(f"  Filtered <think> tags from generated text")

        print(f"  Judging ({provider}/{model}, improved)...")
        gold_atoms = pos['extracted'].get('reasoning', [])
        judge_results = judge_explanation_improved(
            entry, gen_text_for_judging, gold_atoms=gold_atoms,
            provider=provider, model=model, base_url=base_url, api_key=api_key)

        result_row = {
            'idx': idx,
            'fen': pos['fen'],
            'move_san': move_san,
            'game': pos.get('game', '?'),
            'wp_loss': pos.get('wp_loss', 0),
            'quality': pos.get('quality', '?'),
            'generated_text': gen_text,  # Original with <think> tags if present
            'think_tags_filtered': gen_text_for_judging != original_gen_text,
            'filtered_text': gen_text_for_judging if gen_text_for_judging != original_gen_text else None,
            'gen_tool_calls': len(gen_log),
            'claim_accuracy': judge_results['claim_accuracy'],
            'n_claims': judge_results['n_claims'],
            'n_correct_claims': judge_results['n_correct_claims'],
            'recall': judge_results['recall'],
            'quality_correct': judge_results['quality_correct'],
            'gen_assessment': judge_results['gen_assessment'],
            'actual_quality': judge_results['actual_quality'],
            'fluency': judge_results['fluency'],
            'specificity': judge_results['specificity'],
            'composite': judge_results['composite'],
            'n_atoms': len(judge_results['atoms']),
            'n_verified': sum(1 for r in judge_results['verification'] if r.get('verified')),
            'n_gold': len(gold_atoms),
            'n_covered': sum(1 for r in judge_results['matching'] if r.get('covered')),
            'n_sanity_overrides': judge_results['n_sanity_overrides'],
            'n_alternatives': len(judge_results.get('alternatives', [])),
            'alternatives_summary': [
                {
                    'move': alt['move'],
                    'stance': alt['stance'],
                    'eval_diff': alt['eval_diff'],
                }
                for alt in judge_results.get('alternatives', [])
            ],
            'gen_model': gen_model,
            'judge_model': model,
            'provider': provider,
            'used_engine_prompt': use_engine,
        }
        all_results.append(result_row)

        if output_path:
            with open(output_path, 'a') as f:
                f.write(json.dumps(result_row) + '\n')

        n_alts = len(judge_results.get('alternatives', []))
        alt_str = f" | Alternatives: {n_alts}" if n_alts > 0 else ""
        print(f"  Claim Accuracy: {judge_results['claim_accuracy']:.0%} | "
              f"Recall: {judge_results['recall']:.0%} | "
              f"Quality: {'OK' if judge_results['quality_correct'] else 'WRONG'} | "
              f"Fluency: {judge_results['fluency']:.1f} | "
              f"Specificity: {judge_results['specificity']:.1f} | "
              f"Composite: {judge_results['composite']:.3f} | "
              f"Overrides: {judge_results['n_sanity_overrides']}{alt_str}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(all_results)} positions)")
    print(f"{'='*60}")
    avg_claim_acc = np.mean([r['claim_accuracy'] for r in all_results])
    avg_rec = np.mean([r['recall'] for r in all_results])
    quality_acc = np.mean([r['quality_correct'] for r in all_results])
    avg_fluency = np.mean([r['fluency'] for r in all_results])
    avg_spec = np.mean([r['specificity'] for r in all_results])
    avg_comp = np.mean([r['composite'] for r in all_results])
    total_overrides = sum(r['n_sanity_overrides'] for r in all_results)
    print(f"  Avg Claim Accuracy: {avg_claim_acc:.1%}")
    print(f"  Avg Recall: {avg_rec:.1%}")
    print(f"  Quality Accuracy: {quality_acc:.1%}")
    print(f"  Avg Fluency: {avg_fluency:.2f} / 5")
    print(f"  Avg Specificity: {avg_spec:.2f} / 5")
    print(f"  Avg Composite: {avg_comp:.3f}")
    print(f"  Total Sanity Overrides: {total_overrides}")
    avg_alts = np.mean([r.get('n_alternatives', 0) for r in all_results])
    print(f"  Avg Alternatives/Position: {avg_alts:.1f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Batch evaluate chess commentary generation')
    parser.add_argument('--input', required=True, help='Input JSONL file with positions')
    parser.add_argument('--output', help='Output JSONL file for results')
    parser.add_argument('--indices', type=int, nargs='+', help='Specific indices to evaluate')
    parser.add_argument('--n', type=int, default=10, help='Number of positions to evaluate (if indices not specified)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'qwen'], default='openai',
                       help='LLM provider')
    parser.add_argument('--model', default='gpt-4o', help='Model name for judging')
    parser.add_argument('--gen-model', help='Model name for generation (defaults to --model)')
    parser.add_argument('--use-engine', action='store_true',
                       help='Use engine analysis in generation prompts')
    parser.add_argument('--no-tools', action='store_true',
                       help='Disable tool calling in generation')
    parser.add_argument('--base-url', help='Base URL for OpenAI-compatible API (for Qwen)')
    parser.add_argument('--api-key', help='API key (optional)')

    args = parser.parse_args()

    # Load positions
    print(f"Loading positions from {args.input}...")
    with open(args.input) as f:
        positions = [json.loads(line) for line in f]
    print(f"Loaded {len(positions)} positions")

    # Run evaluation
    results = batch_evaluate(
        positions,
        indices=args.indices,
        n=args.n,
        seed=args.seed,
        provider=args.provider,
        model=args.model,
        gen_model=args.gen_model,
        output_path=args.output,
        use_engine=args.use_engine,
        use_tools=not args.no_tools,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    if args.output:
        print(f"\nResults written to {args.output}")


if __name__ == '__main__':
    main()
