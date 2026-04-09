#!/usr/bin/env python3
"""Extract a balanced binary eval dataset with a target count per subcategory.

Each subcategory gets its own independent pass through the PGN file(s), reading
games in small batches (default 100) to keep memory low.  Per-subcategory RNG
(derived from master seed via SHA-256) controls within-batch shuffling so each
subcategory samples different positions even from the same games.

Features:
  - Per-subcategory seed derived from master seed (SHA-256).
  - Independent PGN pass per subcategory (games read in batches, not all at once).
  - Per-game cap (default 3) → within a game, randomly pick up to 3 positions.
  - One move per position per subcategory → maximises position diversity.
  - --subcategories flag → run a subset (for parallel jobs or rare types).

Usage:
    # All subcategories at once (writes one .jsonl per subcategory):
    python extract_balanced_eval.py \
        --pgn_paths data/lichess_2013-01.pgn data/lichess_2013-02.pgn \
        --out_dir eval_legal_binary/data \
        --target 200 --seed 42

    # Just a few rare ones (can scan more games):
    python extract_balanced_eval.py \
        --pgn_paths data/big.pgn \
        --out_dir eval_legal_binary/data \
        --subcategories ep_wrong_pawn non_king_double_check promo_push_blocked \
        --target 200 --max_games 100000 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import chess
import chess.pgn

from legal_moves import get_phase, move_to_san
from legal_move_puzzles import (
    SUBCATEGORY_TO_CATEGORY,
    classify_legal_move,
    detect_en_passant, build_en_passant_illegals,
    analyze_check, build_check_candidates,
    detect_double_check, build_double_check_illegals,
    detect_illegal_king_moves, build_illegal_king_illegals,
    detect_pin, build_pin_illegals,
    detect_promotion, build_promotion_illegals,
    _gen_backward_pawn, _gen_friendly_fire, _gen_blocked_sliding,
    _gen_pawn_double_push_wrong_rank, _gen_pawn_double_push_blocked,
    _gen_pawn_push_onto_piece, _gen_pawn_diagonal_to_empty,
    _gen_pawn_capture_friendly, _gen_castling_path_occupied,
    _gen_castling_no_rights, _gen_ep_pinned,
    _gen_wrong_geometry,
)

ALL_SUBCATEGORIES = sorted(SUBCATEGORY_TO_CATEGORY.keys())


# ── Position scanning (for a single target subcategory) ─────────────────────


def extract_subcategory_moves(
    board: chess.Board,
    target_sub: str,
) -> Optional[List[str]]:
    """Return list of UCI moves for target_sub in this position, or None."""
    legal_ucis = set(m.uci() for m in board.legal_moves)
    category = SUBCATEGORY_TO_CATEGORY[target_sub]

    # ── Legal subcategories ──
    if category == "legal":
        matches = []
        for m in board.legal_moves:
            if classify_legal_move(board, m) == target_sub:
                matches.append(m.uci())
        return matches if matches else None

    # ── Illegal subcategories ──
    seen: Set[str] = set()
    matches: List[str] = []

    def _collect(pairs):
        for uci, t in pairs:
            if t == target_sub and uci not in legal_ucis and uci not in seen:
                seen.add(uci)
                matches.append(uci)

    # Category-specific generators
    if target_sub in ("ep_fake_diagonal", "ep_wrong_pawn"):
        ep = detect_en_passant(board)
        if ep:
            _collect(build_en_passant_illegals(board, ep))

    elif target_sub in ("non_king_double_check",):
        dc = detect_double_check(board)
        if dc:
            _collect(build_double_check_illegals(board, dc))

    elif target_sub == "non_evasion_in_check":
        if board.is_check():
            dc = detect_double_check(board)
            if not dc:
                ci = analyze_check(board)
                if ci and ci.evasion_types >= 2:
                    _collect(build_check_candidates(board, ci))

    elif target_sub in ("king_to_attacked", "castling_in_check"):
        # Can come from check, double_check, or illegal_king
        dc = detect_double_check(board)
        if dc:
            _collect(build_double_check_illegals(board, dc))
        elif board.is_check():
            ci = analyze_check(board)
            if ci and ci.evasion_types >= 2:
                _collect(build_check_candidates(board, ci))
        if not board.is_check():
            ik = detect_illegal_king_moves(board)
            if ik:
                _collect(build_illegal_king_illegals(board, ik))

    elif target_sub == "castling_through_attacked":
        if not board.is_check():
            ik = detect_illegal_king_moves(board)
            if ik:
                _collect(build_illegal_king_illegals(board, ik))

    elif target_sub == "pin_breaking":
        pin = detect_pin(board)
        if pin:
            _collect(build_pin_illegals(board, pin))

    elif target_sub in ("promo_push_blocked", "promo_capture_empty"):
        promo = detect_promotion(board)
        if promo:
            _collect(build_promotion_illegals(board, promo))

    else:
        # General distractors
        turn = board.turn
        gen_funcs = {
            "backward_pawn": _gen_backward_pawn,
            "friendly_fire": _gen_friendly_fire,
            "blocked_sliding": _gen_blocked_sliding,
            "pawn_double_wrong_rank": _gen_pawn_double_push_wrong_rank,
            "pawn_double_push_blocked": _gen_pawn_double_push_blocked,
            "pawn_push_onto_piece": _gen_pawn_push_onto_piece,
            "pawn_diagonal_to_empty": _gen_pawn_diagonal_to_empty,
            "pawn_capture_friendly": _gen_pawn_capture_friendly,
            "wrong_ep": _gen_pawn_diagonal_to_empty,  # wrong_ep comes from this generator
            "castling_path_occupied": _gen_castling_path_occupied,
            "castling_no_rights": _gen_castling_no_rights,
            "ep_pinned": _gen_ep_pinned,
            "wrong_geometry_knight": _gen_wrong_geometry,
            "wrong_geometry_bishop": _gen_wrong_geometry,
            "wrong_geometry_rook": _gen_wrong_geometry,
            "wrong_geometry_queen": _gen_wrong_geometry,
            "wrong_geometry_king": _gen_wrong_geometry,
        }
        func = gen_funcs.get(target_sub)
        if func:
            for move, t in func(board, turn):
                uci = move.uci()
                if t == target_sub and uci not in legal_ucis and uci not in seen:
                    seen.add(uci)
                    matches.append(uci)

    return matches if matches else None


# ── Seed derivation ─────────────────────────────────────────────────────────


def derive_seed(master_seed: int, subcategory: str) -> int:
    """Deterministically derive a per-subcategory seed from the master seed."""
    h = hashlib.sha256(f"{master_seed}:{subcategory}".encode()).hexdigest()
    return int(h[:8], 16)


# ── Batched game iterator ──────────────────────────────────────────────────


def iter_game_batches(
    pgn_paths: List[str],
    batch_size: int,
    max_games: int,
) -> List[List[chess.pgn.Game]]:
    """Yield batches of games from PGN files. Returns list of batches."""
    batches: List[List[chess.pgn.Game]] = []
    batch: List[chess.pgn.Game] = []
    total = 0

    for pgn_path in pgn_paths:
        if total >= max_games:
            break
        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
            while total < max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                batch.append(game)
                total += 1
                if len(batch) >= batch_size:
                    batches.append(batch)
                    batch = []

    if batch:
        batches.append(batch)

    return batches, total


# ── Process a single game for a subcategory ────────────────────────────────


def scan_game(
    game: chess.pgn.Game,
    subcat: str,
) -> List[Tuple[str, str, List[str]]]:
    """Return list of (fen, phase, [ucis]) for positions matching subcat."""
    board = game.board()
    last_move = None
    candidates: List[Tuple[str, str, List[str]]] = []

    for game_move in game.mainline_moves():
        if last_move is not None:
            ucis = extract_subcategory_moves(board, subcat)
            if ucis:
                candidates.append((board.fen(en_passant="xfen"), get_phase(board), ucis))
        board.push(game_move)
        last_move = game_move

    return candidates


# ── Per-subcategory extraction (streaming batches) ─────────────────────────


def extract_for_subcategory(
    subcat: str,
    pgn_paths: List[str],
    master_seed: int,
    target: int,
    per_game_cap: int,
    batch_size: int,
    max_games: int,
) -> List[dict]:
    """Independent PGN pass: read in batches, shuffle within batch, collect."""
    seed = derive_seed(master_seed, subcat)
    rng = random.Random(seed)

    label = "legal" if SUBCATEGORY_TO_CATEGORY[subcat] == "legal" else "illegal"
    category = SUBCATEGORY_TO_CATEGORY[subcat]

    collected: List[dict] = []
    total_games = 0

    for pgn_path in pgn_paths:
        if len(collected) >= target or total_games >= max_games:
            break

        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
            batch: List[chess.pgn.Game] = []

            while len(collected) < target and total_games < max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                batch.append(game)
                total_games += 1

                # Process when batch is full
                if len(batch) >= batch_size:
                    _process_batch(batch, subcat, rng, label, category,
                                   per_game_cap, target, collected)
                    print(f"    {subcat}: {len(collected)}/{target} after {total_games} games", flush=True)
                    batch = []

            # Process remaining games in partial batch
            if batch and len(collected) < target:
                _process_batch(batch, subcat, rng, label, category,
                               per_game_cap, target, collected)
                print(f"    {subcat}: {len(collected)}/{target} after {total_games} games", flush=True)

    return collected


def _process_batch(
    batch: List[chess.pgn.Game],
    subcat: str,
    rng: random.Random,
    label: str,
    category: str,
    per_game_cap: int,
    target: int,
    collected: List[dict],
) -> None:
    """Shuffle batch with subcategory RNG, scan games, collect positions."""
    # Shuffle game order within this batch
    indices = list(range(len(batch)))
    rng.shuffle(indices)

    for idx in indices:
        if len(collected) >= target:
            return

        candidates = scan_game(batch[idx], subcat)
        if not candidates:
            continue

        # Randomly pick up to per_game_cap positions
        rng.shuffle(candidates)
        for fen, phase, ucis in candidates[:per_game_cap]:
            move_uci = rng.choice(ucis)
            board_tmp = chess.Board(fen)
            move_san = move_to_san(board_tmp, move_uci)

            collected.append({
                "fen": fen,
                "move_uci": move_uci,
                "move_san": move_san,
                "label": label,
                "category": category,
                "subcategory": subcat,
                "phase": phase,
            })
            if len(collected) >= target:
                return


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extract balanced binary eval, target N per subcategory."
    )
    parser.add_argument("--pgn_paths", type=str, nargs="+", required=True,
                        help="PGN file(s) to scan")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory (one JSONL per subcategory)")
    parser.add_argument("--target", type=int, default=200,
                        help="Target (fen, move) pairs per subcategory (default: 200)")
    parser.add_argument("--max_games", type=int, default=100000,
                        help="Max total games to scan per subcategory")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Games to read per batch (default: 100)")
    parser.add_argument("--per_game_cap", type=int, default=3,
                        help="Max positions per game per subcategory (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--subcategories", type=str, nargs="*", default=None,
                        help="Subset of subcategories to extract (default: all)")
    args = parser.parse_args()

    # Validate subcategories
    if args.subcategories:
        unknown = set(args.subcategories) - set(ALL_SUBCATEGORIES)
        if unknown:
            parser.error(f"Unknown subcategories: {unknown}\nValid: {ALL_SUBCATEGORIES}")
        active_subs = sorted(set(args.subcategories))
    else:
        active_subs = ALL_SUBCATEGORIES

    print(f"Extracting {len(active_subs)} subcategories, target {args.target} each")
    print(f"Batch size: {args.batch_size}, max games per subcategory: {args.max_games}\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for i, subcat in enumerate(active_subs):
        rows = extract_for_subcategory(
            subcat, args.pgn_paths, args.seed, args.target,
            args.per_game_cap, args.batch_size, args.max_games,
        )
        summary[subcat] = rows

        # Write per-subcategory file
        out_file = out_dir / f"{subcat}.jsonl"
        with out_file.open("w") as fout:
            for row in rows:
                fout.write(json.dumps(row) + "\n")

        phases = Counter(r["phase"] for r in rows)
        phase_str = "  ".join(f"{p}:{c}" for p, c in sorted(phases.items()))
        status = "OK" if len(rows) >= args.target else f"need {args.target - len(rows)} more"
        print(f"  [{i+1}/{len(active_subs)}] {subcat:<30} {len(rows):>4} / {args.target}  {phase_str}  {status}")

    # ── Summary ──
    total = sum(len(rows) for rows in summary.values())
    shortfall = [s for s, rows in summary.items() if len(rows) < args.target]

    print(f"\nTotal rows: {total}")
    print(f"Output dir: {out_dir}/  ({len(active_subs)} files)")

    if shortfall:
        print(f"\n{len(shortfall)} subcategories below target: {', '.join(shortfall)}")
        print("Increase --max_games or add more PGN files.")


if __name__ == "__main__":
    main()
