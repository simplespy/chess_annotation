#!/usr/bin/env python3
"""Generate SFT training data for legal-move identification in multiple-choice format.

Reads JSONL produced by extract_eval_positions.ipynb, samples a small candidate
set per position (mix of legal + illegal moves), and writes reasoning traces
using per-type templates.

Usage:
    python generate_sft_legal_moves.py \
        --data_path data/eval_positions_preview.jsonl \
        --template_dir reasoning_templates/ \
        --out_path data/sft_legal_moves.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess

from legal_moves import (
    PIECE_NAME, PIECE_SYMBOL_TO_NAME, GEOMETRY_REASONS, CASTLE_SAFE_SQUARES,
    piece_desc, get_attacker_desc, get_pinner_desc, find_blocker, move_to_san,
)


# ── Template loading ─────────────────────────────────────────────────────────

def load_templates(template_dir: str) -> Dict[str, str]:
    """Load all .txt templates from directory into a dict keyed by filename stem."""
    templates = {}
    for f in Path(template_dir).glob("*.txt"):
        templates[f.stem] = f.read_text().strip()
    return templates


# ── Move description helpers ─────────────────────────────────────────────────

def describe_legal_move(board: chess.Board, uci: str, templates: Dict[str, str]) -> str:
    """Generate brief description of a legal move.

    In check positions, uses check-specific templates (king escape, capture
    checker, block check) instead of the generic legal_move template.
    """
    move = chess.Move.from_uci(uci)
    piece = board.piece_at(move.from_square)
    san = move_to_san(board, uci)

    if piece is None:
        return templates["legal_move"].format(move=san, piece_name="piece", move_desc="")

    name = PIECE_NAME[piece.piece_type]
    dest_square = chess.square_name(move.to_square)

    # ── Check-specific descriptions ──
    if board.is_check():
        checkers = list(board.checkers())
        checker_desc = " and ".join(piece_desc(board, sq) for sq in checkers)

        # King move → escape
        if piece.piece_type == chess.KING:
            return templates["legal_king_escape"].format(
                move=san, dest_square=dest_square,
            )
        # Non-king capture of the checking piece → capture checker
        if move.to_square in checkers:
            return templates["legal_capture_checker"].format(
                move=san, checker_desc=checker_desc, piece_name=name,
            )
        # Otherwise → block (interposition)
        return templates["legal_block_check"].format(
            move=san, checker_desc=checker_desc,
            piece_name=name, dest_square=dest_square,
        )

    # ── Normal (non-check) descriptions ──
    descs = []

    target = board.piece_at(move.to_square)
    if target:
        descs.append(f" capturing {piece_desc(board, move.to_square)}")
    if board.is_castling(move):
        side = "kingside" if chess.square_file(move.to_square) > chess.square_file(move.from_square) else "queenside"
        descs.append(f" ({side} castling)")
    if board.is_en_passant(move):
        descs.append(" (en passant)")
    if move.promotion:
        descs.append(f" promoting to {PIECE_NAME[move.promotion]}")

    board.push(move)
    if board.is_check():
        descs.append(" with check")
    board.pop()

    move_desc = ",".join(descs) if descs else ""
    return templates["legal_move"].format(move=san, piece_name=name, move_desc=move_desc)


def describe_illegal_move(
    board: chess.Board, uci: str, move_type: str,
    templates: Dict[str, str], row: dict,
) -> str:
    """Generate type-specific explanation of why a move is illegal."""
    move = chess.Move.from_uci(uci)
    san = move_to_san(board, uci)
    piece = board.piece_at(move.from_square)
    opp = not board.turn
    dest_square = chess.square_name(move.to_square)
    from_square = chess.square_name(move.from_square)
    piece_name = PIECE_NAME.get(piece.piece_type, "piece") if piece else "piece"

    if move_type == "king_to_attacked":
        return templates["king_to_attacked"].format(
            move=san, dest_square=dest_square,
            attacker_desc=get_attacker_desc(board, move.to_square, opp),
        )

    elif move_type == "castling_through_attacked":
        safe_sqs = CASTLE_SAFE_SQUARES.get(uci, [])
        attacked_sq = dest_square
        for sq in safe_sqs:
            if board.is_attacked_by(opp, sq):
                attacked_sq = chess.square_name(sq)
                break
        return templates["castling_through_attacked"].format(
            move=san, attacked_square=attacked_sq,
            attacker_desc=get_attacker_desc(board, chess.parse_square(attacked_sq), opp),
        )

    elif move_type == "castling_in_check":
        return templates["castling_in_check"].format(move=san)

    elif move_type == "pin_breaking":
        return templates["pin_breaking"].format(
            move=san, piece_name=piece_name, from_square=from_square,
            pinner_desc=get_pinner_desc(board, move.from_square),
        )

    elif move_type == "non_king_double_check":
        checker_squares = row.get("checker_squares", [])
        checker_pieces = row.get("checker_pieces", [])
        if len(checker_squares) >= 2:
            c1 = f"the {PIECE_SYMBOL_TO_NAME.get(checker_pieces[0], 'piece')} on {checker_squares[0]}"
            c2 = f"the {PIECE_SYMBOL_TO_NAME.get(checker_pieces[1], 'piece')} on {checker_squares[1]}"
        else:
            c1, c2 = "one piece", "another piece"
        return templates["non_king_double_check"].format(move=san, checker1=c1, checker2=c2)

    elif move_type == "ep_fake_diagonal":
        return templates["ep_fake_diagonal"].format(move=san, dest_square=dest_square)

    elif move_type == "ep_wrong_pawn":
        adj_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        return templates["ep_wrong_pawn"].format(
            move=san, adjacent_pawn_square=chess.square_name(adj_sq),
        )

    elif move_type == "wrong_ep":
        adj_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        return templates["wrong_ep"].format(
            move=san, adjacent_pawn_square=chess.square_name(adj_sq),
        )

    elif move_type == "promo_push_blocked":
        blocker = board.piece_at(move.to_square)
        blocking_piece = piece_desc(board, move.to_square) if blocker else "a piece"
        return templates["promo_push_blocked"].format(
            move=san, dest_square=dest_square, blocking_piece=blocking_piece,
        )

    elif move_type == "promo_capture_empty":
        return templates["promo_capture_empty"].format(move=san, dest_square=dest_square)

    elif move_type == "backward_pawn":
        return templates["backward_pawn"].format(move=san)

    elif move_type == "friendly_fire":
        target = board.piece_at(move.to_square)
        target_piece = PIECE_NAME.get(target.piece_type, "piece") if target else "piece"
        return templates["friendly_fire"].format(
            move=san, piece_name=piece_name, target_piece=target_piece, dest_square=dest_square,
        )

    elif move_type == "blocked_sliding":
        return templates["blocked_sliding"].format(
            move=san, piece_name=piece_name, from_square=from_square,
            dest_square=dest_square,
            blocker_desc=find_blocker(board, move.from_square, move.to_square),
        )

    elif move_type == "pawn_double_wrong_rank":
        return templates["pawn_double_wrong_rank"].format(move=san, from_square=from_square)

    elif move_type == "pawn_push_onto_piece":
        blocker = board.piece_at(move.to_square)
        blocking_piece = piece_desc(board, move.to_square) if blocker else "a piece"
        return templates["pawn_push_onto_piece"].format(
            move=san, dest_square=dest_square, blocking_piece=blocking_piece,
        )

    elif move_type == "pawn_diagonal_to_empty":
        return templates["pawn_diagonal_to_empty"].format(move=san, dest_square=dest_square)

    elif move_type == "pawn_capture_friendly":
        target = board.piece_at(move.to_square)
        target_piece = PIECE_NAME.get(target.piece_type, "piece") if target else "piece"
        return templates["pawn_capture_friendly"].format(
            move=san, target_piece=target_piece, dest_square=dest_square,
        )

    elif move_type == "non_evasion_in_check":
        checkers = list(board.checkers())
        checker_desc = " and ".join(piece_desc(board, sq) for sq in checkers)
        return templates["non_evasion_in_check"].format(
            move=san, checker_desc=checker_desc, piece_name=piece_name,
        )

    elif move_type == "pawn_double_push_blocked":
        mid_sq = move.from_square + (8 if board.turn == chess.WHITE else -8)
        blocker = board.piece_at(mid_sq)
        blocker_desc = piece_desc(board, mid_sq) if blocker else "a piece"
        mid_square = chess.square_name(mid_sq)
        return templates["pawn_double_push_blocked"].format(
            move=san, blocker_desc=blocker_desc, mid_square=mid_square,
        )

    elif move_type == "castling_path_occupied":
        from legal_moves import CASTLE_INFO
        uci = move.uci()
        blocker_desc = "a piece"
        for info in CASTLE_INFO[board.turn]:
            if chess.Move(info["king_from"], info["king_to"]).uci() == uci:
                for sq in info["clear_sqs"]:
                    p = board.piece_at(sq)
                    if p is not None:
                        blocker_desc = piece_desc(board, sq)
                        break
                break
        return templates["castling_path_occupied"].format(
            move=san, blocker_desc=blocker_desc,
        )

    elif move_type == "castling_no_rights":
        from legal_moves import CASTLE_INFO
        side = "kingside" if move.to_square in (chess.G1, chess.G8) else "queenside"
        # Determine which piece lost the rights — we can't know for certain,
        # so use a generic description based on side.
        lost_piece = f"{side} rook or king"
        return templates["castling_no_rights"].format(move=san, lost_piece=lost_piece)

    elif move_type == "ep_pinned":
        # The captured pawn is on the same file as ep_square, one rank behind
        captured_sq = chess.square(
            chess.square_file(move.to_square),
            chess.square_rank(move.from_square),
        )
        captured_square = chess.square_name(captured_sq)
        # Find the lateral attacker revealed by removing both pawns
        king_sq = board.king(board.turn)
        king_rank = chess.square_rank(king_sq)
        attacker_desc = "an attack"
        if king_rank == chess.square_rank(move.from_square):
            for direction in [-1, 1]:
                sq = king_sq
                while True:
                    sq += direction
                    if sq < 0 or sq > 63 or chess.square_rank(sq) != king_rank:
                        break
                    if sq == move.from_square or sq == captured_sq:
                        continue
                    p = board.piece_at(sq)
                    if p is not None:
                        if p.color != board.turn and p.piece_type in (chess.ROOK, chess.QUEEN):
                            attacker_desc = piece_desc(board, sq)
                        break
        return templates["ep_pinned"].format(
            move=san, captured_square=captured_square, attacker_desc=attacker_desc,
        )

    elif move_type.startswith("wrong_geometry"):
        reason = GEOMETRY_REASONS.get(move_type, "this geometry is not valid for this piece")
        return templates["wrong_geometry"].format(
            move=san, piece_name=piece_name, geometry_reason=reason,
        )

    else:
        return f"Consider the move {san}. This is illegal ({move_type})."


# ── Candidate sampling ───────────────────────────────────────────────────────

def sample_candidates(
    row: dict, rng: random.Random,
    num_legal: int = 3,
    num_illegal_cat: Optional[int] = None,
    num_illegal_gen: int = 2,
) -> dict:
    """Sample a multiple-choice set from a row.

    Returns dict with keys: "legal", "illegal_cat", "illegal_gen".
    Each value is a list of {"uci": str, "type": str|None}.
    Includes ALL category illegals + sampled legal + sampled general illegals.
    """
    cat = list(row.get("illegal_category", []))
    if num_illegal_cat is not None and len(cat) > num_illegal_cat:
        cat = rng.sample(cat, num_illegal_cat)

    gen = list(row.get("illegal_general", []))
    if len(gen) > num_illegal_gen:
        gen = rng.sample(gen, num_illegal_gen)

    all_legal = list(row["correct_outputs_uci"])
    game_move = row["game_move_uci"]

    legal_sample = []
    if game_move in all_legal:
        legal_sample.append(game_move)
        remaining = [m for m in all_legal if m != game_move]
    else:
        remaining = list(all_legal)

    need = num_legal - len(legal_sample)
    if need > 0 and remaining:
        legal_sample += rng.sample(remaining, min(need, len(remaining)))

    return {
        "legal": [{"uci": u, "type": None} for u in legal_sample],
        "illegal_cat": cat,
        "illegal_gen": gen,
    }


# ── Trace generation ─────────────────────────────────────────────────────────

def generate_candidate_analysis(
    board: chess.Board, templates: Dict[str, str], row: dict,
    shuffled_order: List[Tuple[str, str, Optional[str]]],
) -> str:
    """Generate the {candidate_analysis} section."""
    lines = []
    for uci, category, move_type in shuffled_order:
        if category == "legal":
            lines.append(describe_legal_move(board, uci, templates))
        else:
            lines.append(describe_illegal_move(board, uci, move_type, templates, row))
    return "\n\n".join(lines)


def format_reasoning_trace(
    row: dict, sampled: dict,
    templates: Dict[str, str], rng: random.Random,
) -> Tuple[str, str]:
    """Fill the wrapper template with all components.

    Returns (input_text, output_text).
    """
    board = chess.Board(row["fen"])
    last_move_san = row["last_move_uci"]
    turn_color = "White" if board.turn == chess.WHITE else "Black"

    # Build shuffled candidate list: (uci, category, type)
    all_entries: List[Tuple[str, str, Optional[str]]] = []
    for d in sampled["legal"]:
        all_entries.append((d["uci"], "legal", None))
    for d in sampled["illegal_cat"]:
        all_entries.append((d["uci"], "illegal_cat", d["type"]))
    for d in sampled["illegal_gen"]:
        all_entries.append((d["uci"], "illegal_gen", d["type"]))
    rng.shuffle(all_entries)

    san_candidates = [move_to_san(board, uci) for uci, _, _ in all_entries]
    candidates_str = ", ".join(san_candidates)

    legal_sans = sorted(move_to_san(board, d["uci"]) for d in sampled["legal"])
    legal_moves_str = ", ".join(legal_sans)

    analysis = generate_candidate_analysis(board, templates, row, all_entries)

    output_text = templates["reasoning_template"].format(
        fen=row["fen"],
        last_move=last_move_san,
        turn_color=turn_color,
        candidates=candidates_str,
        candidate_analysis=analysis,
        legal_moves=legal_moves_str,
    )

    input_text = (
        f"FEN: {row['fen']}\n"
        f"Previous move: {last_move_san}\n"
        f"Which of these moves are legal? {candidates_str}"
    )

    return input_text, output_text


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT training data for legal-move identification."
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to JSONL from extract_eval_positions.ipynb")
    parser.add_argument("--template_dir", type=str, required=True,
                        help="Directory containing reasoning template .txt files")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Output JSONL path for SFT data")
    parser.add_argument("--num_legal", type=int, default=3, help="Legal moves to sample per position")
    parser.add_argument("--num_illegal_gen", type=int, default=2, help="General illegals to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    templates = load_templates(args.template_dir)

    required = [
        "reasoning_template", "legal_move",
        "legal_king_escape", "legal_capture_checker", "legal_block_check",
        "king_to_attacked", "castling_through_attacked", "castling_in_check",
        "pin_breaking", "non_king_double_check",
        "ep_fake_diagonal", "ep_wrong_pawn",
        "promo_push_blocked", "promo_capture_empty",
        "backward_pawn", "friendly_fire", "blocked_sliding",
        "pawn_double_wrong_rank", "pawn_push_onto_piece",
        "pawn_diagonal_to_empty", "pawn_capture_friendly",
        "wrong_geometry",
        "non_evasion_in_check", "pawn_double_push_blocked",
        "castling_path_occupied", "castling_no_rights",
        "wrong_ep", "ep_pinned",
    ]
    missing = [t for t in required if t not in templates]
    if missing:
        raise FileNotFoundError(f"Missing templates: {missing}")
    print(f"Loaded {len(templates)} templates from {args.template_dir}")

    data_path = Path(args.data_path)
    rows = [json.loads(line) for line in data_path.open()]
    print(f"Read {len(rows)} positions from {data_path}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_written = 0
    num_errors = 0

    with out_path.open("w") as fout:
        for i, row in enumerate(rows):
            try:
                sampled = sample_candidates(
                    row, rng,
                    num_legal=args.num_legal,
                    num_illegal_gen=args.num_illegal_gen,
                )
                input_text, output_text = format_reasoning_trace(row, sampled, templates, rng)
                record = {
                    "input": input_text,
                    "output": output_text,
                    "fen": row["fen"],
                    "tags": row["tags"],
                }
                fout.write(json.dumps(record) + "\n")
                num_written += 1
            except Exception as e:
                num_errors += 1
                if num_errors <= 5:
                    print(f"  Error on row {i}: {e}")

    print(f"\nWrote {num_written} traces to {out_path}")
    if num_errors:
        print(f"  ({num_errors} errors)")


if __name__ == "__main__":
    main()
