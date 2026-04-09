"""Low-level chess helpers shared across extraction and generation pipelines.

Provides:
- Piece name / description utilities
- Castling geometry table
- Attacker / pinner / blocker descriptions (for reasoning templates)
- UCI → SAN conversion (handles illegal moves gracefully)
- Position phase classification
- PGN game iterator
"""

from __future__ import annotations

from typing import Optional

import chess
import chess.pgn

# ── Constants ────────────────────────────────────────────────────────────────

PIECE_NAME = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

PIECE_SYMBOL_TO_NAME = {
    "p": "pawn", "n": "knight", "b": "bishop",
    "r": "rook", "q": "queen", "k": "king",
    "P": "pawn", "N": "knight", "B": "bishop",
    "R": "rook", "Q": "queen", "K": "king",
}

# Castling geometry used by check, double-check, and illegal-king detectors.
CASTLE_INFO = {
    chess.WHITE: [
        # kingside O-O
        {"rights_fn": "has_kingside_castling_rights",
         "king_from": chess.E1, "king_to": chess.G1,
         "rook_sq": chess.H1,
         "clear_sqs": [chess.F1, chess.G1],
         "safe_sqs":  [chess.E1, chess.F1, chess.G1]},
        # queenside O-O-O
        {"rights_fn": "has_queenside_castling_rights",
         "king_from": chess.E1, "king_to": chess.C1,
         "rook_sq": chess.A1,
         "clear_sqs": [chess.B1, chess.C1, chess.D1],
         "safe_sqs":  [chess.E1, chess.D1, chess.C1]},
    ],
    chess.BLACK: [
        {"rights_fn": "has_kingside_castling_rights",
         "king_from": chess.E8, "king_to": chess.G8,
         "rook_sq": chess.H8,
         "clear_sqs": [chess.F8, chess.G8],
         "safe_sqs":  [chess.E8, chess.F8, chess.G8]},
        {"rights_fn": "has_queenside_castling_rights",
         "king_from": chess.E8, "king_to": chess.C8,
         "rook_sq": chess.A8,
         "clear_sqs": [chess.B8, chess.C8, chess.D8],
         "safe_sqs":  [chess.E8, chess.D8, chess.C8]},
    ],
}

# Lookup for castling-through-attacked reasoning: uci → squares that must be safe
CASTLE_SAFE_SQUARES = {
    "e1g1": [chess.E1, chess.F1, chess.G1],
    "e1c1": [chess.E1, chess.D1, chess.C1],
    "e8g8": [chess.E8, chess.F8, chess.G8],
    "e8c8": [chess.E8, chess.D8, chess.C8],
}

GEOMETRY_REASONS = {
    "wrong_geometry_knight": "knights move in an L-shape (2+1), not diagonally",
    "wrong_geometry_bishop": "bishops move diagonally, not in straight lines",
    "wrong_geometry_rook": "rooks move in straight lines, not diagonally",
    "wrong_geometry_queen": "queens move in straight lines or diagonally, not in an L-shape",
    "wrong_geometry_king": "kings can only move one square at a time (except castling)",
}


# ── Piece description helpers ────────────────────────────────────────────────

def piece_desc(board: chess.Board, sq: int) -> str:
    """Describe a piece, e.g. 'the white bishop on c4'."""
    piece = board.piece_at(sq)
    if piece is None:
        return f"an empty square {chess.square_name(sq)}"
    color = "white" if piece.color == chess.WHITE else "black"
    name = PIECE_NAME[piece.piece_type]
    return f"the {color} {name} on {chess.square_name(sq)}"


def get_attacker_desc(board: chess.Board, square: int, by_color: chess.Color) -> str:
    """Describe what attacks a given square (e.g. 'the white bishop on c4')."""
    attackers = board.attackers(by_color, square)
    if not attackers:
        return "opponent pieces"
    descs = [piece_desc(board, sq) for sq in attackers]
    return " and ".join(descs)


def get_pinner_desc(board: chess.Board, pinned_sq: int) -> str:
    """Describe the piece pinning a given square to the king."""
    piece = board.piece_at(pinned_sq)
    if piece is None:
        return "an opponent piece"
    turn = piece.color
    pin_mask = board.pin(turn, pinned_sq)
    opp = not turn
    for sq in pin_mask:
        p = board.piece_at(sq)
        if p and p.color == opp and p.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            return piece_desc(board, sq)
    return "an opponent piece"


def find_blocker(board: chess.Board, from_sq: int, to_sq: int) -> str:
    """Find the first piece blocking the path from from_sq to to_sq."""
    from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
    to_rank, to_file = chess.square_rank(to_sq), chess.square_file(to_sq)

    dr = 0 if to_rank == from_rank else (1 if to_rank > from_rank else -1)
    df = 0 if to_file == from_file else (1 if to_file > from_file else -1)

    r, f = from_rank + dr, from_file + df
    while 0 <= r <= 7 and 0 <= f <= 7:
        sq = chess.square(f, r)
        if sq == to_sq:
            break
        if board.piece_at(sq) is not None:
            return piece_desc(board, sq)
        r += dr
        f += df
    return "a piece"


# ── SAN conversion ───────────────────────────────────────────────────────────

def move_to_san(board: chess.Board, uci: str) -> str:
    """Convert UCI to SAN, handling illegal moves gracefully."""
    move = chess.Move.from_uci(uci)
    try:
        return board.san(move)
    except (ValueError, AssertionError):
        piece = board.piece_at(move.from_square)
        if piece is None:
            return uci
        symbol = piece.symbol().upper()
        if symbol == "P":
            symbol = ""
        dest = chess.square_name(move.to_square)
        target = board.piece_at(move.to_square)
        cap = "x" if target else ""
        promo = ""
        if move.promotion:
            promo = "=" + chess.piece_symbol(move.promotion).upper()
        if piece.piece_type == chess.PAWN and cap:
            return f"{chess.FILE_NAMES[chess.square_file(move.from_square)]}x{dest}{promo}"
        return f"{symbol}{cap}{dest}{promo}"


# ── Position classification ──────────────────────────────────────────────────

def get_phase(board: chess.Board) -> str:
    """Classify position into opening / middlegame / endgame."""
    fullmove = board.fullmove_number
    piece_map = board.piece_map()
    non_pawn_non_king = sum(
        1 for p in piece_map.values()
        if p.piece_type not in (chess.PAWN, chess.KING)
    )
    total = len(piece_map)

    if fullmove <= 12:
        return "opening"
    if non_pawn_non_king >= 6 and total >= 16:
        return "middlegame"
    return "endgame"


# ── PGN iteration ────────────────────────────────────────────────────────────

def iter_games(pgn_path: str, max_games: Optional[int] = None):
    """Yield chess.pgn.Game objects from a PGN file."""
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        count = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game
            count += 1
            if max_games is not None and count >= max_games:
                break
