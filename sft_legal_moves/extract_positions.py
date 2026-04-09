#!/usr/bin/env python3
"""Extract evaluation positions from PGN files.

Usage:
    python extract_positions.py \
        --pgn_path data/lichess_db_standard_rated_2013-02.pgn \
        --out_path data/eval_positions_train.jsonl \
        --max_games 1400 --num_distractors 5 --num_vanilla 2000 --seed 42
"""

import argparse
import json
import random

from legal_move_puzzles import extract_all


def main():
    parser = argparse.ArgumentParser(description="Extract positions from PGN into JSONL.")
    parser.add_argument("--pgn_path", type=str, required=True, help="Path to input PGN file")
    parser.add_argument("--out_path", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--max_games", type=int, default=1400, help="Number of games to scan")
    parser.add_argument("--num_distractors", type=int, default=5, help="General distractors per position")
    parser.add_argument("--num_vanilla", type=int, default=2000, help="Vanilla positions to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rows = extract_all(
        args.pgn_path,
        max_games=args.max_games,
        num_general_distractors=args.num_distractors,
        num_vanilla_positions=args.num_vanilla,
        rng=random.Random(args.seed),
    )

    with open(args.out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(rows)} positions to {args.out_path}")


if __name__ == "__main__":
    main()
