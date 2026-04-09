#!/usr/bin/env python3
"""Generate a binary classification eval dataset for legal-move identification.

Each output row is a single (fen, move) pair with a legal/illegal label,
category, and subcategory.

Reads intermediate JSONL produced by extract_positions.py.

Usage:
    python generate_eval_binary.py \
        --data_path data/eval_positions.jsonl \
        --out_path data/eval_binary.jsonl

    # With per-tag overrides:
    python generate_eval_binary.py \
        --data_path data/eval_positions.jsonl \
        --out_path data/eval_binary.jsonl \
        --category_config category_config.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import chess

from legal_moves import move_to_san
from legal_move_puzzles import SUBCATEGORY_TO_CATEGORY, classify_legal_move


def flatten_position(
    row: dict,
    rng: random.Random,
    ratio: float = 1.0,
) -> List[dict]:
    """Flatten one intermediate position into per-move binary rows.

    Emits all illegal moves, then samples legal moves at the given ratio
    (legal:illegal).
    """
    fen = row["fen"]
    board = chess.Board(fen)
    position_tags = row.get("tags", [])
    phase = row.get("phase", "")

    # -- Collect all illegal moves --
    illegal_rows = []
    seen_uci = set()

    for entry in row.get("illegal_category", []):
        uci = entry["uci"]
        if uci in seen_uci:
            continue
        seen_uci.add(uci)
        subcategory = entry["type"]
        category = SUBCATEGORY_TO_CATEGORY.get(subcategory, subcategory)
        illegal_rows.append({
            "fen": fen,
            "move_uci": uci,
            "move_san": move_to_san(board, uci),
            "label": "illegal",
            "category": category,
            "subcategory": subcategory,
            "position_tags": position_tags,
            "phase": phase,
        })

    for entry in row.get("illegal_general", []):
        uci = entry["uci"]
        if uci in seen_uci:
            continue
        seen_uci.add(uci)
        subcategory = entry["type"]
        category = SUBCATEGORY_TO_CATEGORY.get(subcategory, subcategory)
        illegal_rows.append({
            "fen": fen,
            "move_uci": uci,
            "move_san": move_to_san(board, uci),
            "label": "illegal",
            "category": category,
            "subcategory": subcategory,
            "position_tags": position_tags,
            "phase": phase,
        })

    num_illegal = len(illegal_rows)
    if num_illegal == 0:
        return []

    # -- Classify and sample legal moves --
    all_legal = list(board.legal_moves)
    classified = []
    for m in all_legal:
        subcategory = classify_legal_move(board, m)
        classified.append((m, subcategory))

    num_legal_target = max(1, round(num_illegal * ratio))
    if len(classified) > num_legal_target:
        classified = rng.sample(classified, num_legal_target)

    legal_rows = []
    for m, subcategory in classified:
        uci = m.uci()
        category = SUBCATEGORY_TO_CATEGORY.get(subcategory, "legal")
        legal_rows.append({
            "fen": fen,
            "move_uci": uci,
            "move_san": move_to_san(board, uci),
            "label": "legal",
            "category": category,
            "subcategory": subcategory,
            "position_tags": position_tags,
            "phase": phase,
        })

    return illegal_rows + legal_rows


def load_category_config(path: str) -> Dict[str, dict]:
    """Load optional per-tag configuration.

    Format:
        {"pin": {"num_positions": 500, "ratio": 2.0}, ...}
    """
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate binary classification eval dataset."
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Intermediate JSONL from extract_positions.py")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--default_ratio", type=float, default=1.0,
                        help="Default legal:illegal ratio per position (default: 1.0)")
    parser.add_argument("--category_config", type=str, default=None,
                        help="Optional JSON config for per-tag sampling overrides")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load config
    tag_config: Dict[str, dict] = {}
    if args.category_config:
        tag_config = load_category_config(args.category_config)
        print(f"Loaded category config: {list(tag_config.keys())}")

    # Load data
    data_path = Path(args.data_path)
    rows = [json.loads(line) for line in data_path.open()]
    print(f"Read {len(rows)} positions from {data_path}")

    # Group positions by their primary tag (first tag, or 'vanilla')
    # A position can have multiple tags; we process it once per tag for
    # sampling purposes, but deduplicate at the end.
    by_tag: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        tags = row.get("tags", ["vanilla"])
        for t in tags:
            by_tag[t].append(i)

    print(f"Tags found: {', '.join(f'{t}({len(idxs)})' for t, idxs in sorted(by_tag.items()))}")

    # Determine which position indices to include, applying per-tag caps
    selected_indices = set()
    for tag, idxs in by_tag.items():
        cfg = tag_config.get(tag, {})
        num_positions = cfg.get("num_positions")

        pool = list(idxs)
        rng.shuffle(pool)
        if num_positions is not None:
            pool = pool[:num_positions]

        selected_indices.update(pool)

    print(f"Selected {len(selected_indices)} positions after tag-level sampling")

    # Flatten selected positions
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    label_counts = Counter()
    cat_counts = Counter()
    subcat_counts = Counter()

    with out_path.open("w") as fout:
        for i in sorted(selected_indices):
            row = rows[i]
            tags = row.get("tags", ["vanilla"])

            # Determine ratio: use tag-specific override if any tag matches
            ratio = args.default_ratio
            for t in tags:
                cfg = tag_config.get(t, {})
                if "ratio" in cfg:
                    ratio = cfg["ratio"]
                    break

            flat = flatten_position(row, rng, ratio=ratio)
            for entry in flat:
                fout.write(json.dumps(entry) + "\n")
                total_rows += 1
                label_counts[entry["label"]] += 1
                cat_counts[(entry["label"], entry["category"])] += 1
                subcat_counts[(entry["label"], entry["subcategory"])] += 1

    # Print summary
    print(f"\nWrote {total_rows} rows to {out_path}")
    print(f"\n{'Label':<10} {'Count':>8}")
    print("-" * 20)
    for label, count in sorted(label_counts.items()):
        print(f"{label:<10} {count:>8}")

    print(f"\n{'Label':<10} {'Category':<20} {'Count':>8}")
    print("-" * 40)
    for (label, cat), count in sorted(cat_counts.items()):
        print(f"{label:<10} {cat:<20} {count:>8}")

    print(f"\n{'Label':<10} {'Subcategory':<30} {'Count':>8}")
    print("-" * 50)
    for (label, subcat), count in sorted(subcat_counts.items()):
        print(f"{label:<10} {subcat:<30} {count:>8}")


if __name__ == "__main__":
    main()
