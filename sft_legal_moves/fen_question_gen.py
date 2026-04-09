#!/usr/bin/env python3
"""Generate ground-truth Q&A pairs from a FEN to test LLM position reading.

Each question probes a specific fact about the position (piece locations,
material, pawn structure, metadata, etc.).  Answers are computed with
python-chess so they are guaranteed correct.

Usage
-----
Single FEN (prints to stdout):

    python fen_question_gen.py \
        --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1" \
        --n_per_category 5 --seed 42

Batch mode (JSONL or one-FEN-per-line):

    python fen_question_gen.py \
        --fen_file data/positions.jsonl \
        --out_path data/fen_questions.jsonl \
        --n_per_category 3 --seed 42

Subset of categories:

    python fen_question_gen.py --fen "..." \
        --categories square_contents,piece_counting,material
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import chess

# ── Helpers ──────────────────────────────────────────────────────────────────

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

COLOR_NAMES = {chess.WHITE: "White", chess.BLACK: "Black"}

FILE_NAMES = list("abcdefgh")


def _sq_name(sq: int) -> str:
    return chess.square_name(sq)


def _piece_desc(piece: chess.Piece) -> str:
    """e.g. 'White knight'."""
    return f"{COLOR_NAMES[piece.color]} {PIECE_NAMES[piece.piece_type]}"


def _material(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for pt, val in PIECE_VALUES.items():
        total += len(board.pieces(pt, color)) * val
    return total


def _make(question: str, answer, category: str, subcategory: str) -> dict:
    return {
        "question": question,
        "answer": str(answer),
        "category": category,
        "subcategory": subcategory,
    }


def _is_light_square(sq: int) -> bool:
    """Light square = (file + rank) is odd in 0-indexed coords."""
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1


# ── Category generators ─────────────────────────────────────────────────────
# Each returns a list of Q&A dicts.  `n` is a *target* count — generators
# produce up to n questions, may return fewer if the position doesn't allow it.


def gen_square_contents(board: chess.Board, rng: random.Random, n: int) -> List[dict]:
    results = []

    # Build occupied and empty lists so we can balance piece_on_square
    # (random squares skew heavily toward "Empty")
    occupied = [sq for sq in range(64) if board.piece_at(sq) is not None]
    empty = [sq for sq in range(64) if board.piece_at(sq) is None]
    rng.shuffle(occupied)
    rng.shuffle(empty)

    subcategories = ["piece_on_square", "is_square_occupied", "color_on_square"]

    # For piece_on_square, alternate between occupied and empty squares
    occ_idx, emp_idx = 0, 0

    squares = list(range(64))
    rng.shuffle(squares)

    for sq in squares:
        if len(results) >= n:
            break
        sub = rng.choice(subcategories)

        # For piece_on_square, bias toward occupied squares (~70%)
        if sub == "piece_on_square":
            if rng.random() < 0.7 and occ_idx < len(occupied):
                sq = occupied[occ_idx]
                occ_idx += 1
            elif emp_idx < len(empty):
                sq = empty[emp_idx]
                emp_idx += 1

        piece = board.piece_at(sq)
        name = _sq_name(sq)

        if sub == "piece_on_square":
            answer = _piece_desc(piece) if piece else "Empty"
            results.append(_make(
                f"What piece is on {name}?", answer,
                "square_contents", "piece_on_square",
            ))
        elif sub == "is_square_occupied":
            answer = "Occupied" if piece else "Empty"
            results.append(_make(
                f"Is {name} occupied or empty?", answer,
                "square_contents", "is_square_occupied",
            ))
        else:
            answer = COLOR_NAMES[piece.color] if piece else "Empty"
            results.append(_make(
                f"What color is the piece on {name}?", answer,
                "square_contents", "color_on_square",
            ))

    return results


def gen_piece_counting(board: chess.Board, rng: random.Random, n: int) -> List[dict]:
    results = []
    options = []

    # count specific piece type per color
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            cnt = len(board.pieces(pt, color))
            cname = COLOR_NAMES[color]
            pname = PIECE_NAMES[pt]
            options.append(_make(
                f"How many {cname.lower()} {pname}s are on the board?", cnt,
                "piece_counting", "count_piece_type",
            ))

    # total per color
    for color in (chess.WHITE, chess.BLACK):
        cnt = sum(1 for sq in range(64) if board.piece_at(sq) and board.piece_at(sq).color == color)
        cname = COLOR_NAMES[color]
        options.append(_make(
            f"How many total pieces (including pawns) does {cname} have?", cnt,
            "piece_counting", "count_color_pieces",
        ))

    # total on board
    cnt = sum(1 for sq in range(64) if board.piece_at(sq))
    options.append(_make(
        "How many pieces (including pawns) are on the board in total?", cnt,
        "piece_counting", "count_all_pieces",
    ))

    rng.shuffle(options)
    return options[:n]


def gen_piece_location(board: chess.Board, rng: random.Random, n: int) -> List[dict]:
    results = []
    options = []

    # king square
    for color in (chess.WHITE, chess.BLACK):
        king_sq = board.king(color)
        if king_sq is not None:
            cname = COLOR_NAMES[color]
            options.append(_make(
                f"Which square is the {cname.lower()} king on?",
                _sq_name(king_sq),
                "piece_location", "king_square",
            ))

    # list squares for a piece type
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            sqs = board.pieces(pt, color)
            if not sqs:
                continue
            cname = COLOR_NAMES[color]
            pname = PIECE_NAMES[pt]
            sq_list = ", ".join(sorted(_sq_name(s) for s in sqs))
            options.append(_make(
                f"List all squares occupied by {cname.lower()} {pname}s.",
                sq_list,
                "piece_location", "list_piece_squares",
            ))

    # is there a piece on square?
    squares = list(range(64))
    rng.shuffle(squares)
    for sq in squares[:8]:
        color = rng.choice([chess.WHITE, chess.BLACK])
        pt = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        piece = board.piece_at(sq)
        answer = "Yes" if (piece and piece.color == color and piece.piece_type == pt) else "No"
        cname = COLOR_NAMES[color]
        pname = PIECE_NAMES[pt]
        options.append(_make(
            f"Is there a {cname.lower()} {pname} on {_sq_name(sq)}?",
            answer,
            "piece_location", "is_piece_on_square",
        ))

    rng.shuffle(options)
    return options[:n]


def gen_material(board: chess.Board, rng: random.Random, n: int) -> List[dict]:
    results = []
    options = []

    w_mat = _material(board, chess.WHITE)
    b_mat = _material(board, chess.BLACK)

    if w_mat > b_mat:
        more = "White"
    elif b_mat > w_mat:
        more = "Black"
    else:
        more = "Equal"

    options.append(_make(
        "Which side has more material? (P=1, N=3, B=3, R=5, Q=9)",
        more, "material", "which_side_more_material",
    ))

    for color in (chess.WHITE, chess.BLACK):
        val = _material(board, color)
        cname = COLOR_NAMES[color]
        options.append(_make(
            f"What is {cname}'s total material value? (P=1, N=3, B=3, R=5, Q=9)",
            val, "material", "material_value",
        ))

    options.append(_make(
        "Is the material equal? (P=1, N=3, B=3, R=5, Q=9)",
        "Yes" if w_mat == b_mat else "No",
        "material", "is_material_equal",
    ))

    rng.shuffle(options)
    return options[:n]


def gen_rank_file(board: chess.Board, rng: random.Random, n: int) -> List[dict]:
    options = []

    # count color pieces on a rank
    for color in (chess.WHITE, chess.BLACK):
        rank = rng.randint(0, 7)
        cnt = 0
        for file in range(8):
            p = board.piece_at(chess.square(file, rank))
            if p and p.color == color:
                cnt += 1
        cname = COLOR_NAMES[color]
        options.append(_make(
            f"How many {cname.lower()} pieces are on rank {rank + 1}?",
            cnt, "rank_file", "count_on_rank",
        ))

    # count all pieces on a file
    for _ in range(2):
        file = rng.randint(0, 7)
        cnt = sum(1 for rank in range(8) if board.piece_at(chess.square(file, rank)))
        fname = FILE_NAMES[file]
        options.append(_make(
            f"How many pieces are on the {fname}-file?",
            cnt, "rank_file", "count_on_file",
        ))

    # open file
    for file in range(8):
        fname = FILE_NAMES[file]
        has_pawn = False
        for rank in range(8):
            p = board.piece_at(chess.square(file, rank))
            if p and p.piece_type == chess.PAWN:
                has_pawn = True
                break
        options.append(_make(
            f"Is the {fname}-file open (no pawns at all)?",
            "No" if has_pawn else "Yes",
            "rank_file", "is_file_open",
        ))

    # half-open file
    for color in (chess.WHITE, chess.BLACK):
        file = rng.randint(0, 7)
        fname = FILE_NAMES[file]
        cname = COLOR_NAMES[color]
        opp = not color
        own_pawn = any(
            board.piece_at(chess.square(file, r))
            and board.piece_at(chess.square(file, r)).piece_type == chess.PAWN
            and board.piece_at(chess.square(file, r)).color == color
            for r in range(8)
        )
        opp_pawn = any(
            board.piece_at(chess.square(file, r))
            and board.piece_at(chess.square(file, r)).piece_type == chess.PAWN
            and board.piece_at(chess.square(file, r)).color == opp
            for r in range(8)
        )
        is_half_open = (not own_pawn) and opp_pawn
        options.append(_make(
            f"Is the {fname}-file half-open for {cname}? (no {cname.lower()} pawns, but opponent has a pawn)",
            "Yes" if is_half_open else "No",
            "rank_file", "is_file_half_open",
        ))

    rng.shuffle(options)
    return options[:n]


def gen_pawn_structure(board: chess.Board, rng: random.Random, n: int) -> List[dict]:
    options = []

    for color in (chess.WHITE, chess.BLACK):
        cname = COLOR_NAMES[color]
        pawns = board.pieces(chess.PAWN, color)
        pawn_files = [chess.square_file(s) for s in pawns]

        # doubled pawns
        from collections import Counter
        file_counts = Counter(pawn_files)
        doubled_files = [f for f, c in file_counts.items() if c >= 2]
        if doubled_files:
            file_str = ", ".join(f"{FILE_NAMES[f]}-file" for f in sorted(doubled_files))
            answer = f"Yes, on the {file_str}"
        else:
            answer = "No"
        options.append(_make(
            f"Does {cname} have doubled pawns? If so, on which file(s)?",
            answer, "pawn_structure", "doubled_pawns",
        ))

        # isolated pawns
        has_isolated = False
        for f in set(pawn_files):
            neighbors = {f - 1, f + 1}
            if not neighbors.intersection(set(pawn_files)):
                has_isolated = True
                break
        options.append(_make(
            f"Does {cname} have any isolated pawns?",
            "Yes" if has_isolated else "No",
            "pawn_structure", "isolated_pawn",
        ))

        # pawn islands
        if pawn_files:
            sorted_files = sorted(set(pawn_files))
            islands = 1
            for i in range(1, len(sorted_files)):
                if sorted_files[i] - sorted_files[i - 1] > 1:
                    islands += 1
        else:
            islands = 0
        options.append(_make(
            f"How many pawn islands does {cname} have?",
            islands, "pawn_structure", "pawn_islands",
        ))

        # passed pawns
        has_passed = False
        opp = not color
        opp_pawn_files_by_rank = {}
        for s in board.pieces(chess.PAWN, opp):
            f = chess.square_file(s)
            r = chess.square_rank(s)
            opp_pawn_files_by_rank.setdefault(f, []).append(r)

        for s in pawns:
            f = chess.square_file(s)
            r = chess.square_rank(s)
            is_passed = True
            for adj_f in (f - 1, f, f + 1):
                if adj_f < 0 or adj_f > 7:
                    continue
                for opp_r in opp_pawn_files_by_rank.get(adj_f, []):
                    if color == chess.WHITE and opp_r > r:
                        is_passed = False
                        break
                    if color == chess.BLACK and opp_r < r:
                        is_passed = False
                        break
                if not is_passed:
                    break
            if is_passed:
                has_passed = True
                break

        options.append(_make(
            f"Does {cname} have a passed pawn?",
            "Yes" if has_passed else "No",
            "pawn_structure", "passed_pawn",
        ))

    rng.shuffle(options)
    return options[:n]


def gen_metadata(board: chess.Board, rng: random.Random, n: int) -> List[dict]:
    options = []

    # side to move
    options.append(_make(
        "Whose turn is it to move?",
        COLOR_NAMES[board.turn],
        "metadata", "side_to_move",
    ))

    # castling rights
    for color in (chess.WHITE, chess.BLACK):
        cname = COLOR_NAMES[color]
        options.append(_make(
            f"Can {cname} castle kingside?",
            "Yes" if board.has_kingside_castling_rights(color) else "No",
            "metadata", "can_castle_kingside",
        ))
        options.append(_make(
            f"Can {cname} castle queenside?",
            "Yes" if board.has_queenside_castling_rights(color) else "No",
            "metadata", "can_castle_queenside",
        ))

    # en passant
    ep = board.ep_square
    if ep is not None:
        answer = f"Yes, {_sq_name(ep)}"
    else:
        answer = "No"
    options.append(_make(
        "Is there an en passant square available? If so, which?",
        answer, "metadata", "en_passant_square",
    ))

    # fullmove number
    options.append(_make(
        "What is the fullmove number?",
        board.fullmove_number, "metadata", "fullmove_number",
    ))

    # halfmove clock
    options.append(_make(
        "What is the halfmove clock?",
        board.halfmove_clock, "metadata", "halfmove_clock",
    ))

    rng.shuffle(options)
    return options[:n]


def gen_spatial(board: chess.Board, rng: random.Random, n: int) -> List[dict]:
    options = []

    # pieces between two pieces on same rank/file/diagonal
    occupied = [sq for sq in range(64) if board.piece_at(sq)]
    rng.shuffle(occupied)

    generated_between = 0
    for i, sq_a in enumerate(occupied):
        if generated_between >= max(n, 4):
            break
        for sq_b in occupied[i + 1:]:
            if generated_between >= max(n, 4):
                break
            fa, ra = chess.square_file(sq_a), chess.square_rank(sq_a)
            fb, rb = chess.square_file(sq_b), chess.square_rank(sq_b)

            # same rank
            if ra == rb:
                between = [chess.square(f, ra) for f in range(min(fa, fb) + 1, max(fa, fb))]
            # same file
            elif fa == fb:
                between = [chess.square(fa, r) for r in range(min(ra, rb) + 1, max(ra, rb))]
            # same diagonal
            elif abs(fa - fb) == abs(ra - rb):
                df = 1 if fb > fa else -1
                dr = 1 if rb > ra else -1
                between = []
                cf, cr = fa + df, ra + dr
                while cf != fb or cr != rb:
                    between.append(chess.square(cf, cr))
                    cf += df
                    cr += dr
            else:
                continue

            if not between:
                continue

            cnt = sum(1 for s in between if board.piece_at(s))
            answer = "Yes" if cnt > 0 else "No"
            options.append(_make(
                f"Are there any pieces between {_sq_name(sq_a)} and {_sq_name(sq_b)}?",
                answer, "spatial", "pieces_between",
            ))
            generated_between += 1

    # king and rook same rank
    for color in (chess.WHITE, chess.BLACK):
        cname = COLOR_NAMES[color]
        king_sq = board.king(color)
        if king_sq is None:
            continue
        rooks = board.pieces(chess.ROOK, color)
        if not rooks:
            continue
        kr = chess.square_rank(king_sq)
        kf = chess.square_file(king_sq)
        for rsq in rooks:
            rr = chess.square_rank(rsq)
            rf = chess.square_file(rsq)
            options.append(_make(
                f"Are the {cname.lower()} king and the {cname.lower()} rook on {_sq_name(rsq)} on the same rank?",
                "Yes" if kr == rr else "No",
                "spatial", "same_rank",
            ))
            options.append(_make(
                f"Are the {cname.lower()} king and the {cname.lower()} rook on {_sq_name(rsq)} on the same file?",
                "Yes" if kf == rf else "No",
                "spatial", "same_file",
            ))

    rng.shuffle(options)
    return options[:n]


def gen_checks_attacks(board: chess.Board, rng: random.Random, n: int) -> List[dict]:
    options = []

    # is king in check
    for color in (chess.WHITE, chess.BLACK):
        cname = COLOR_NAMES[color]
        # Check if this side's king is in check: it's in check when it's
        # that side's turn and board.is_check(), OR we can use attackers.
        king_sq = board.king(color)
        if king_sq is None:
            continue
        attackers = board.attackers(not color, king_sq)
        in_check = len(attackers) > 0
        options.append(_make(
            f"Is the {cname.lower()} king in check?",
            "Yes" if in_check else "No",
            "checks_attacks", "is_in_check",
        ))

    # how many pieces attack a square
    squares = list(range(64))
    rng.shuffle(squares)
    for sq in squares[:6]:
        for color in (chess.WHITE, chess.BLACK):
            cnt = len(board.attackers(color, sq))
            cname = COLOR_NAMES[color]
            options.append(_make(
                f"How many {cname.lower()} pieces attack {_sq_name(sq)}?",
                cnt, "checks_attacks", "attackers_of_square",
            ))

    # is square attacked by color
    rng.shuffle(squares)
    for sq in squares[:4]:
        color = rng.choice([chess.WHITE, chess.BLACK])
        cname = COLOR_NAMES[color]
        attacked = board.is_attacked_by(color, sq)
        options.append(_make(
            f"Is {_sq_name(sq)} attacked by {cname.lower()}?",
            "Yes" if attacked else "No",
            "checks_attacks", "is_square_attacked",
        ))

    rng.shuffle(options)
    return options[:n]


def gen_board_geometry(board: chess.Board, rng: random.Random, n: int) -> List[dict]:
    options = []

    # square color
    squares = list(range(64))
    rng.shuffle(squares)
    for sq in squares[:4]:
        light = _is_light_square(sq)
        options.append(_make(
            f"Is {_sq_name(sq)} a light or dark square?",
            "Light" if light else "Dark",
            "board_geometry", "square_color",
        ))

    # bishop pair
    for color in (chess.WHITE, chess.BLACK):
        cname = COLOR_NAMES[color]
        bishops = board.pieces(chess.BISHOP, color)
        has_pair = len(bishops) >= 2
        options.append(_make(
            f"Does {cname} have the bishop pair (two or more bishops)?",
            "Yes" if has_pair else "No",
            "board_geometry", "bishop_pair",
        ))

    # bishop square colors
    for color in (chess.WHITE, chess.BLACK):
        cname = COLOR_NAMES[color]
        bishops = board.pieces(chess.BISHOP, color)
        if not bishops:
            answer = "None"
        else:
            colors = {_is_light_square(s) for s in bishops}
            if colors == {True, False}:
                answer = "Both"
            elif colors == {True}:
                answer = "Light"
            else:
                answer = "Dark"
        options.append(_make(
            f"What color square(s) are {cname.lower()}'s bishop(s) on?",
            answer, "board_geometry", "bishop_square_colors",
        ))

    rng.shuffle(options)
    return options[:n]


def gen_after_move(board: chess.Board, rng: random.Random, n: int,
                   move_uci: Optional[str] = None) -> List[dict]:
    """Questions about the position *after* a move is played."""
    if move_uci is None:
        return []

    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return []

    # Compute SAN before pushing
    move_san = board.san(move)
    result = board.copy()
    result.push(move)

    options = []

    # en passant available after move?
    ep = result.ep_square
    if ep is not None:
        answer = f"Yes, {_sq_name(ep)}"
    else:
        answer = "No"
    options.append(_make(
        f"After playing {move_san}, is an en passant square available? If so, which?",
        answer, "after_move", "en_passant_after_move",
    ))

    rng.shuffle(options)
    return options[:n]


# ── Registry ─────────────────────────────────────────────────────────────────

# Generators that need only (board, rng, n)
GENERATORS: Dict[str, callable] = {
    "square_contents": gen_square_contents,
    "piece_counting": gen_piece_counting,
    "piece_location": gen_piece_location,
    "material": gen_material,
    "rank_file": gen_rank_file,
    "pawn_structure": gen_pawn_structure,
    "metadata": gen_metadata,
    "spatial": gen_spatial,
    "checks_attacks": gen_checks_attacks,
    "board_geometry": gen_board_geometry,
}

# Generators that need (board, rng, n, move_uci)
MOVE_GENERATORS: Dict[str, callable] = {
    "after_move": gen_after_move,
}

ALL_CATEGORIES = list(GENERATORS.keys()) + list(MOVE_GENERATORS.keys())


# ── Public API ───────────────────────────────────────────────────────────────

def generate_questions(
    fen: str,
    categories: Optional[List[str]] = None,
    n_per_category: int = 3,
    seed: Optional[int] = None,
    move_uci: Optional[str] = None,
) -> List[dict]:
    """Generate ground-truth Q&A pairs for a FEN.

    Parameters
    ----------
    fen : str
        The position to generate questions about.
    categories : list[str] or None
        Subset of category names to use (default: all).
    n_per_category : int
        Target number of questions per category.
    seed : int or None
        Random seed for reproducibility.
    move_uci : str or None
        UCI move string, required for ``after_move`` category.

    Returns
    -------
    list[dict]
        Each dict has keys: fen, question, answer, category, subcategory.
    """
    board = chess.Board(fen)
    rng = random.Random(seed)

    cats = categories if categories else list(GENERATORS.keys())
    # Include move-dependent categories only when move_uci is provided
    if categories is None and move_uci is not None:
        cats = cats + list(MOVE_GENERATORS.keys())

    results = []

    for cat in cats:
        if cat in GENERATORS:
            items = GENERATORS[cat](board, rng, n_per_category)
        elif cat in MOVE_GENERATORS:
            items = MOVE_GENERATORS[cat](board, rng, n_per_category, move_uci=move_uci)
        else:
            print(f"Warning: unknown category '{cat}', skipping", file=sys.stderr)
            continue
        for item in items:
            item["fen"] = fen
        results.extend(items)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate FEN comprehension Q&A pairs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fen", type=str, help="Single FEN string")
    group.add_argument("--fen_file", type=str,
                       help="File of FENs: one per line, or JSONL with 'fen' field")

    parser.add_argument("--out_path", type=str, default=None,
                        help="Output JSONL path (default: stdout)")
    parser.add_argument("--n_per_category", type=int, default=3,
                        help="Questions per category per position (default: 3)")
    parser.add_argument("--move_uci", type=str, default=None,
                        help="UCI move for after_move questions (single FEN mode)")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated category subset (default: all). "
                             f"Available: {', '.join(ALL_CATEGORIES)}")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    cats = args.categories.split(",") if args.categories else None

    # Collect FENs
    fens: List[str] = []
    if args.fen:
        fens = [args.fen]
    else:
        path = Path(args.fen_file)
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            # Try JSONL
            if line.startswith("{"):
                row = json.loads(line)
                fens.append(row["fen"])
            else:
                fens.append(line)

    print(f"Generating questions for {len(fens)} position(s), "
          f"n_per_category={args.n_per_category}, "
          f"categories={cats or 'all'}", file=sys.stderr)

    # Generate
    out_file = open(args.out_path, "w") if args.out_path else sys.stdout
    total = 0
    for fen in fens:
        questions = generate_questions(
            fen, categories=cats, n_per_category=args.n_per_category,
            seed=args.seed, move_uci=args.move_uci,
        )
        for q in questions:
            out_file.write(json.dumps(q) + "\n")
            total += 1

    if args.out_path:
        out_file.close()

    print(f"Generated {total} questions total.", file=sys.stderr)


if __name__ == "__main__":
    main()
