#!/usr/bin/env python3
"""Generate per-subcategory JSONL eval datasets for FEN comprehension.

Step 1: Read PGN, replay all games, collect all positions into a list.
Step 2: For each subcategory, sample positions from that list with its own
        seed, generate questions, balance if needed, write JSONL.

Usage
-----
    python generate_eval_fen_questions.py \
        --pgn_path ../data/lichess_db_standard_rated_2013-01.pgn \
        --out_dir data/ \
        --n_per_subcategory 500 \
        --max_games 5000 \
        --max_per_game 3 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import chess

# Allow imports from sft_legal_moves/ (parent directory)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legal_moves import iter_games, get_phase
from fen_question_gen import (
    GENERATORS,
    MOVE_GENERATORS,
    ALL_CATEGORIES,
    generate_questions,
)

# ── Which subcategories need balanced answers ────────────────────────────────

BALANCED_SUBCATEGORIES = {
    "bishop_pair",
    "can_castle_kingside",
    "can_castle_queenside",
    "is_file_half_open",
    "is_file_open",
    "is_in_check",
    "is_material_equal",
    "is_piece_on_square",
    "is_square_attacked",
    "is_square_occupied",
    "isolated_pawn",
    "passed_pawn",
    "pieces_between",
    "bishop_square_colors",
    "color_on_square",
    "pawn_islands",
    "same_rank",
    "same_file",
    "side_to_move",
    "square_color",
    "which_side_more_material",
}

# ── Subcategory → parent category map ────────────────────────────────────────

_SUBCAT_TO_CAT: dict[str, str] = {}


def _build_subcat_map():
    board = chess.Board()
    rng = random.Random(0)
    for cat, fn in GENERATORS.items():
        for item in fn(board, rng, 100):
            _SUBCAT_TO_CAT[item["subcategory"]] = cat
    for cat, fn in MOVE_GENERATORS.items():
        for item in fn(board, rng, 100, move_uci="e2e4"):
            _SUBCAT_TO_CAT[item["subcategory"]] = cat


_build_subcat_map()


# ── Step 1: Extract positions from PGN ───────────────────────────────────────

def extract_positions(
    pgn_path: str,
    max_games: int,
    max_per_game: int,
    seed: int,
) -> list[dict]:
    """Read PGN, collect positions into a flat list."""
    rng = random.Random(seed)
    positions = []
    game_count = 0

    for game in iter_games(pgn_path, max_games):
        game_count += 1
        moves = list(game.mainline_moves())
        if len(moves) < 6:
            continue

        game_id = game.headers.get("Site", f"game_{game_count}")

        eligible = list(range(4, len(moves)))
        rng.shuffle(eligible)
        chosen = set(eligible[:max_per_game])

        board = game.board()
        for ply, move in enumerate(moves):
            if ply in chosen:
                positions.append({
                    "fen": board.fen(),
                    "move_uci": move.uci(),
                    "move_san": board.san(move),
                    "game_id": game_id,
                    "ply": ply,
                    "phase": get_phase(board),
                })
            board.push(move)

        if game_count % 500 == 0:
            print(f"  {game_count} games, {len(positions)} positions...",
                  file=sys.stderr)

    print(f"Extracted {len(positions)} positions from {game_count} games.\n",
          file=sys.stderr)
    return positions


# ── Step 2: Generate questions for one subcategory ───────────────────────────

# Balanced subcategories where we collapse into Yes/No before balancing
COLLAPSE_YES_NO: set[str] = set()

# Balanced subcategories where we merge small classes
MERGE_CLASSES = {
    "pawn_islands": lambda a: "3+" if a not in ("0", "1", "2") else a,
}


def generate_subcategory(
    subcategory: str,
    positions: list[dict],
    n_target: int,
    seed: int,
) -> list[dict]:
    """Walk positions, fill per-class buckets, stop when full."""
    cat = _SUBCAT_TO_CAT.get(subcategory)
    if cat is None:
        return []

    rng = random.Random(seed)
    needs_balance = subcategory in BALANCED_SUBCATEGORIES

    indices = list(range(len(positions)))
    rng.shuffle(indices)

    if not needs_balance:
        # Simple: collect n_target questions
        result = []
        for idx in indices:
            if len(result) >= n_target:
                break
            pos = positions[idx]
            qs = generate_questions(
                pos["fen"], categories=[cat], n_per_category=5,
                seed=seed + idx, move_uci=pos["move_uci"],
            )
            for q in qs:
                if q["subcategory"] != subcategory:
                    continue
                q["game_id"] = pos["game_id"]
                q["ply"] = pos["ply"]
                q["phase"] = pos["phase"]
                q["move_uci"] = pos["move_uci"]
                q["move_san"] = pos["move_san"]
                result.append(q)
                if len(result) >= n_target:
                    break
        return result

    # Balanced: fill per-class buckets directly
    # First pass: discover classes (scan a small sample)
    # We do it inline: accept questions into buckets, skip if bucket full.
    # Once all buckets hit per_class, we're done.
    collapse_fn = None
    if subcategory in COLLAPSE_YES_NO:
        collapse_fn = lambda a: "Yes" if a.startswith("Yes") else a
    elif subcategory in MERGE_CLASSES:
        collapse_fn = MERGE_CLASSES[subcategory]

    buckets: dict[str, list[dict]] = defaultdict(list)
    # We don't know n_classes upfront, so we use a two-pass approach:
    # Pass 1: scan first 200 positions to discover classes
    # Pass 2: fill buckets

    # Discover classes
    seen_classes: set[str] = set()
    for idx in indices[:min(500, len(indices))]:
        pos = positions[idx]
        qs = generate_questions(
            pos["fen"], categories=[cat], n_per_category=5,
            seed=seed + idx, move_uci=pos["move_uci"],
        )
        for q in qs:
            if q["subcategory"] != subcategory:
                continue
            ans = q["answer"]
            if collapse_fn:
                ans = collapse_fn(ans)
            seen_classes.add(ans)

    n_classes = max(len(seen_classes), 1)
    per_class = n_target // n_classes

    # Fill buckets
    for idx in indices:
        # Check if all buckets full
        if all(len(buckets[c]) >= per_class for c in seen_classes):
            break

        pos = positions[idx]
        qs = generate_questions(
            pos["fen"], categories=[cat], n_per_category=5,
            seed=seed + idx, move_uci=pos["move_uci"],
        )
        for q in qs:
            if q["subcategory"] != subcategory:
                continue
            ans = q["answer"]
            bucket_key = collapse_fn(ans) if collapse_fn else ans
            seen_classes.add(bucket_key)
            # Recompute per_class if new class discovered
            n_classes = len(seen_classes)
            per_class = n_target // n_classes

            if len(buckets[bucket_key]) >= per_class:
                continue
            q["game_id"] = pos["game_id"]
            q["ply"] = pos["ply"]
            q["phase"] = pos["phase"]
            q["move_uci"] = pos["move_uci"]
            q["move_san"] = pos["move_san"]
            buckets[bucket_key].append(q)

    result = []
    for items in buckets.values():
        result.extend(items[:per_class])
    rng.shuffle(result)
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-subcategory FEN comprehension eval datasets."
    )
    parser.add_argument("--pgn_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/")
    parser.add_argument("--n_per_subcategory", type=int, default=500)
    parser.add_argument("--max_games", type=int, default=5000)
    parser.add_argument("--max_per_game", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subcategories", type=str, default=None,
                        help=f"Comma-separated subset. "
                             f"Available: {', '.join(sorted(_SUBCAT_TO_CAT))}")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subcats = (
        [s.strip() for s in args.subcategories.split(",")]
        if args.subcategories
        else sorted(_SUBCAT_TO_CAT)
    )

    # Step 1: extract all positions once
    print("Step 1: Extracting positions from PGN...", file=sys.stderr)
    positions = extract_positions(
        args.pgn_path, args.max_games, args.max_per_game, args.seed,
    )

    # Step 2: per-subcategory generation
    print(f"Step 2: Generating {args.n_per_subcategory} examples "
          f"for {len(subcats)} subcategories\n", file=sys.stderr)

    print(f"  {'Subcategory':<28s} {'Bal':>3} {'Count':>5} "
          f"{'Classes':>7}  Answer distribution", file=sys.stderr)
    print("-" * 95, file=sys.stderr)

    grand_total = 0
    for i, subcat in enumerate(subcats):
        sub_seed = args.seed + (i + 1) * 1000
        result = generate_subcategory(
            subcat, positions, args.n_per_subcategory, sub_seed,
        )

        out_path = out_dir / f"{subcat}.jsonl"
        with out_path.open("w") as f:
            for q in result:
                f.write(json.dumps(q) + "\n")

        ans_counts = Counter(q["answer"] for q in result)
        bal = "Y" if subcat in BALANCED_SUBCATEGORIES else "N"
        dist = ", ".join(f"{a}={c}" for a, c in ans_counts.most_common(4))
        if len(ans_counts) > 4:
            dist += ", ..."
        print(f"  {subcat:<28s} {bal:>3} {len(result):>5} "
              f"{len(ans_counts):>7}  {dist}", file=sys.stderr)
        grand_total += len(result)

    print(f"\nTotal: {grand_total} questions across {len(subcats)} subcategories.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
