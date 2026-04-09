"""Generate plausible-but-wrong FEN strings as distractors for MCQ tasks.

Perturbation categories:
  1. Piece placement — visual board differences (swap, shift, mirror, etc.)
  2. Metadata — active color, castling, en passant, move counters
  3. Next-state mental errors — common mistakes when computing board after a move

Each perturbation is a standalone function returning a modified FEN or None.
The main entry point is ``generate_wrong_fens()``.

Depends on: python-chess
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import chess

# ── Board ↔ array helpers ────────────────────────────────────────────────────

Piece = Optional[chess.Piece]
Board8x8 = List[List[Piece]]  # board[rank][file], rank 0 = rank 1 (white side)


def board_to_array(board: chess.Board) -> Board8x8:
    """Convert a chess.Board to an 8×8 array (rank 0 = rank 1)."""
    arr: Board8x8 = []
    for rank in range(8):
        row: List[Piece] = []
        for file in range(8):
            row.append(board.piece_at(chess.square(file, rank)))
        arr.append(row)
    return arr


def array_to_board(arr: Board8x8) -> chess.Board:
    """Convert an 8×8 array back to a chess.Board (pieces only, no metadata)."""
    board = chess.Board.empty()
    for rank in range(8):
        for file in range(8):
            piece = arr[rank][file]
            if piece is not None:
                board.set_piece_at(chess.square(file, rank), piece)
    return board


def array_to_placement(arr: Board8x8) -> str:
    """Convert 8×8 array to the piece-placement part of a FEN (ranks 8→1)."""
    ranks = []
    for rank in range(7, -1, -1):  # rank 8 down to rank 1
        fen_rank = ""
        empty = 0
        for file in range(8):
            piece = arr[rank][file]
            if piece is None:
                empty += 1
            else:
                if empty:
                    fen_rank += str(empty)
                    empty = 0
                fen_rank += piece.symbol()
        if empty:
            fen_rank += str(empty)
        ranks.append(fen_rank)
    return "/".join(ranks)


def rebuild_fen(placement: str, active: str, castling: str,
                ep: str, halfmove: str, fullmove: str) -> str:
    """Reassemble a full FEN from its six fields."""
    return f"{placement} {active} {castling} {ep} {halfmove} {fullmove}"


def parse_fen_fields(fen: str) -> Tuple[str, str, str, str, str, str]:
    """Split a FEN into its six fields."""
    parts = fen.split()
    return (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])


# ── Validation ───────────────────────────────────────────────────────────────

def is_valid_distractor(fen: str, correct_fen: str) -> bool:
    """Check that a FEN is a valid distractor.

    Criteria:
      1. Parseable by python-chess (syntactically valid)
      2. Exactly one white king and one black king
      3. No pawns on ranks 1 or 8
      4. Different from the correct FEN
    """
    if fen == correct_fen:
        return False
    try:
        board = chess.Board(fen)
    except ValueError:
        return False

    # Exactly one king per side
    white_kings = len(board.pieces(chess.KING, chess.WHITE))
    black_kings = len(board.pieces(chess.KING, chess.BLACK))
    if white_kings != 1 or black_kings != 1:
        return False

    # No pawns on ranks 1 or 8
    for file in range(8):
        for rank in (0, 7):  # rank index 0 = rank 1, rank index 7 = rank 8
            piece = board.piece_at(chess.square(file, rank))
            if piece is not None and piece.piece_type == chess.PAWN:
                return False

    return True


# ── Category 1: Piece Placement Perturbations ───────────────────────────────

def perturb_swap_pieces(fen: str, rng: random.Random) -> Optional[str]:
    """Swap positions of two same-color non-king pieces."""
    board = chess.Board(fen)
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)
    arr = board_to_array(board)

    for color in (chess.WHITE, chess.BLACK):
        pieces = [(rank, file)
                  for rank in range(8) for file in range(8)
                  if arr[rank][file] is not None
                  and arr[rank][file].color == color
                  and arr[rank][file].piece_type != chess.KING]
        if len(pieces) < 2:
            continue

    # Collect all swappable pieces (both colors)
    candidates = []
    for color in (chess.WHITE, chess.BLACK):
        pieces = [(rank, file)
                  for rank in range(8) for file in range(8)
                  if arr[rank][file] is not None
                  and arr[rank][file].color == color
                  and arr[rank][file].piece_type != chess.KING]
        if len(pieces) >= 2:
            candidates.append(pieces)

    if not candidates:
        return None

    pieces = rng.choice(candidates)
    rng.shuffle(pieces)
    r1, f1 = pieces[0]
    r2, f2 = pieces[1]

    # Only swap if they are different piece types (otherwise no visible change)
    if arr[r1][f1].piece_type == arr[r2][f2].piece_type:
        # Try to find a pair with different types
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                ri, fi = pieces[i]
                rj, fj = pieces[j]
                if arr[ri][fi].piece_type != arr[rj][fj].piece_type:
                    r1, f1, r2, f2 = ri, fi, rj, fj
                    break
            else:
                continue
            break
        else:
            # All same type — swap anyway (positions still differ)
            pass

    arr[r1][f1], arr[r2][f2] = arr[r2][f2], arr[r1][f1]
    new_placement = array_to_placement(arr)
    return rebuild_fen(new_placement, active, castling, ep, hm, fm)


def perturb_shift_piece(fen: str, rng: random.Random) -> Optional[str]:
    """Move one non-king piece to an adjacent empty square."""
    board = chess.Board(fen)
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)
    arr = board_to_array(board)

    # Collect non-king pieces
    pieces = [(rank, file)
              for rank in range(8) for file in range(8)
              if arr[rank][file] is not None
              and arr[rank][file].piece_type != chess.KING]
    rng.shuffle(pieces)

    for rank, file in pieces:
        # Find adjacent empty squares
        neighbors = []
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                nr, nf = rank + dr, file + df
                if 0 <= nr <= 7 and 0 <= nf <= 7 and arr[nr][nf] is None:
                    neighbors.append((nr, nf))
        if not neighbors:
            continue
        nr, nf = rng.choice(neighbors)
        piece = arr[rank][file]
        # Don't put pawns on ranks 1 or 8
        if piece.piece_type == chess.PAWN and nr in (0, 7):
            neighbors = [(r, f) for r, f in neighbors if r not in (0, 7)]
            if not neighbors:
                continue
            nr, nf = rng.choice(neighbors)
        arr[nr][nf] = piece
        arr[rank][file] = None
        new_placement = array_to_placement(arr)
        return rebuild_fen(new_placement, active, castling, ep, hm, fm)

    return None


def perturb_mirror_horizontal(fen: str, rng: random.Random) -> Optional[str]:
    """Mirror the board left-right (a↔h, b↔g, etc.)."""
    board = chess.Board(fen)
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)
    arr = board_to_array(board)

    new_arr: Board8x8 = []
    for rank in range(8):
        new_arr.append(list(reversed(arr[rank])))

    # Update castling — mirroring swaps kingside ↔ queenside
    new_castling = ""
    if "K" in castling:
        new_castling += "Q"
    if "Q" in castling:
        new_castling += "K"
    if "k" in castling:
        new_castling += "q"
    if "q" in castling:
        new_castling += "k"
    # Re-sort to standard order
    order = "KQkq"
    new_castling = "".join(c for c in order if c in new_castling) or "-"

    # Update en passant square
    new_ep = ep
    if ep != "-":
        ep_file = ord(ep[0]) - ord("a")
        mirrored_file = 7 - ep_file
        new_ep = chr(ord("a") + mirrored_file) + ep[1]

    new_placement = array_to_placement(new_arr)
    return rebuild_fen(new_placement, active, new_castling, new_ep, hm, fm)


def perturb_mirror_vertical(fen: str, rng: random.Random) -> Optional[str]:
    """Mirror the board top-bottom (rank 1↔8, 2↔7, etc.).

    Also flips piece colors so the position makes visual sense.
    """
    board = chess.Board(fen)
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)
    arr = board_to_array(board)

    new_arr: Board8x8 = []
    for rank in range(7, -1, -1):
        row: List[Piece] = []
        for file in range(8):
            piece = arr[rank][file]
            if piece is not None:
                # Flip color
                new_color = not piece.color
                row.append(chess.Piece(piece.piece_type, new_color))
            else:
                row.append(None)
        new_arr.append(row)

    # Flip active color
    new_active = "b" if active == "w" else "w"

    # Flip castling
    new_castling = ""
    for ch in castling:
        if ch == "-":
            new_castling = "-"
            break
        new_castling += ch.swapcase()
    order = "KQkq"
    new_castling = "".join(c for c in order if c in new_castling) or "-"

    # Flip en passant rank
    new_ep = ep
    if ep != "-":
        ep_rank = int(ep[1])
        new_rank = 9 - ep_rank
        new_ep = ep[0] + str(new_rank)

    new_placement = array_to_placement(new_arr)
    return rebuild_fen(new_placement, new_active, new_castling, new_ep, hm, fm)


def perturb_change_piece_color(fen: str, rng: random.Random) -> Optional[str]:
    """Flip one non-king piece's color."""
    board = chess.Board(fen)
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)
    arr = board_to_array(board)

    pieces = [(rank, file)
              for rank in range(8) for file in range(8)
              if arr[rank][file] is not None
              and arr[rank][file].piece_type != chess.KING]
    if not pieces:
        return None
    rng.shuffle(pieces)

    for rank, file in pieces:
        piece = arr[rank][file]
        new_piece = chess.Piece(piece.piece_type, not piece.color)
        # Don't put pawns on ranks 1 or 8
        if new_piece.piece_type == chess.PAWN and rank in (0, 7):
            continue
        arr[rank][file] = new_piece
        new_placement = array_to_placement(arr)
        return rebuild_fen(new_placement, active, castling, ep, hm, fm)

    return None


def perturb_change_piece_type(fen: str, rng: random.Random) -> Optional[str]:
    """Change one non-king piece to a different type."""
    board = chess.Board(fen)
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)
    arr = board_to_array(board)

    types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

    pieces = [(rank, file)
              for rank in range(8) for file in range(8)
              if arr[rank][file] is not None
              and arr[rank][file].piece_type != chess.KING]
    if not pieces:
        return None

    rank, file = rng.choice(pieces)
    piece = arr[rank][file]
    candidates = [t for t in types if t != piece.piece_type]
    # Don't put pawns on ranks 1 or 8
    if rank in (0, 7):
        candidates = [t for t in candidates if t != chess.PAWN]
    if not candidates:
        return None
    new_type = rng.choice(candidates)
    arr[rank][file] = chess.Piece(new_type, piece.color)
    new_placement = array_to_placement(arr)
    return rebuild_fen(new_placement, active, castling, ep, hm, fm)


def perturb_add_piece(fen: str, rng: random.Random) -> Optional[str]:
    """Add an extra pawn or minor piece on an empty square."""
    board = chess.Board(fen)
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)
    arr = board_to_array(board)

    empties = [(rank, file)
               for rank in range(8) for file in range(8)
               if arr[rank][file] is None]
    if not empties:
        return None

    color = rng.choice([chess.WHITE, chess.BLACK])
    # Weight toward pawns and minor pieces (more common)
    piece_type = rng.choice([chess.PAWN, chess.PAWN, chess.KNIGHT, chess.BISHOP,
                             chess.ROOK, chess.QUEEN])

    rng.shuffle(empties)
    for rank, file in empties:
        if piece_type == chess.PAWN and rank in (0, 7):
            continue
        arr[rank][file] = chess.Piece(piece_type, color)
        new_placement = array_to_placement(arr)
        return rebuild_fen(new_placement, active, castling, ep, hm, fm)

    return None


def perturb_remove_piece(fen: str, rng: random.Random) -> Optional[str]:
    """Remove one non-king piece from the board."""
    board = chess.Board(fen)
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)
    arr = board_to_array(board)

    pieces = [(rank, file)
              for rank in range(8) for file in range(8)
              if arr[rank][file] is not None
              and arr[rank][file].piece_type != chess.KING]
    if not pieces:
        return None

    rank, file = rng.choice(pieces)
    arr[rank][file] = None
    new_placement = array_to_placement(arr)
    return rebuild_fen(new_placement, active, castling, ep, hm, fm)


def perturb_swap_adjacent_ranks(fen: str, rng: random.Random) -> Optional[str]:
    """Swap two neighboring ranks entirely."""
    board = chess.Board(fen)
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)
    arr = board_to_array(board)

    # Pick a random rank pair (0-1, 1-2, ..., 6-7)
    rank = rng.randint(0, 6)
    arr[rank], arr[rank + 1] = arr[rank + 1], arr[rank]

    # Check pawns not on ranks 1/8 after swap
    for file in range(8):
        for r in (0, 7):
            piece = arr[r][file]
            if piece is not None and piece.piece_type == chess.PAWN:
                # Undo and try a different pair
                arr[rank], arr[rank + 1] = arr[rank + 1], arr[rank]
                # Try all pairs
                for r2 in range(7):
                    if r2 == rank:
                        continue
                    arr[r2], arr[r2 + 1] = arr[r2 + 1], arr[r2]
                    valid = True
                    for f2 in range(8):
                        for rcheck in (0, 7):
                            p = arr[rcheck][f2]
                            if p is not None and p.piece_type == chess.PAWN:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        new_placement = array_to_placement(arr)
                        return rebuild_fen(new_placement, active, castling, ep, hm, fm)
                    arr[r2], arr[r2 + 1] = arr[r2 + 1], arr[r2]
                return None

    new_placement = array_to_placement(arr)
    return rebuild_fen(new_placement, active, castling, ep, hm, fm)


def perturb_empty_square_error(fen: str, rng: random.Random) -> Optional[str]:
    """Change a digit in the FEN placement (shifting a piece horizontally).

    E.g. ``3p4`` → ``2p5`` — the pawn slides one square left.
    """
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)
    ranks = placement.split("/")

    # Find ranks that contain at least one digit > 1 adjacent to a piece
    candidates = []
    for i, rank_str in enumerate(ranks):
        for j, ch in enumerate(rank_str):
            if ch.isdigit() and int(ch) >= 1:
                # Check if adjacent to a piece character
                has_piece_neighbor = False
                if j > 0 and rank_str[j - 1].isalpha():
                    has_piece_neighbor = True
                if j < len(rank_str) - 1 and rank_str[j + 1].isalpha():
                    has_piece_neighbor = True
                if has_piece_neighbor:
                    candidates.append((i, j))

    if not candidates:
        return None

    rank_idx, char_idx = rng.choice(candidates)
    rank_str = ranks[rank_idx]
    digit = int(rank_str[char_idx])

    # Decide: increment or decrement the digit
    choices = []
    if digit > 1:
        choices.append(-1)
    if digit < 7:
        choices.append(1)
    if not choices:
        return None

    delta = rng.choice(choices)
    new_digit = digit + delta

    # Find the adjacent piece and adjust its neighbor digit
    # To keep total width at 8, we need to adjust a neighboring digit
    new_rank = list(rank_str)
    new_rank[char_idx] = str(new_digit)

    # Find a neighboring digit to compensate
    compensated = False
    for neighbor_idx in ([char_idx + 1, char_idx - 1]
                         if rng.random() < 0.5
                         else [char_idx - 1, char_idx + 1]):
        if 0 <= neighbor_idx < len(new_rank) and new_rank[neighbor_idx].isdigit():
            n_digit = int(new_rank[neighbor_idx])
            n_new = n_digit - delta
            if 1 <= n_new <= 7:
                new_rank[neighbor_idx] = str(n_new)
                compensated = True
                break

    if not compensated:
        # No neighboring digit to compensate — just do a simple ±1 shift
        # This will make the rank invalid width, so skip
        return None

    ranks[rank_idx] = "".join(new_rank)
    new_placement = "/".join(ranks)

    # Validate the rank width = 8
    test_board = chess.Board.empty()
    try:
        test_board.set_board_fen(new_placement)
    except ValueError:
        return None

    return rebuild_fen(new_placement, active, castling, ep, hm, fm)


# ── Category 2: Metadata Perturbations ──────────────────────────────────────

def perturb_flip_active_color(fen: str, rng: random.Random) -> Optional[str]:
    """Flip active color: w → b or b → w."""
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)
    new_active = "b" if active == "w" else "w"
    return rebuild_fen(placement, new_active, castling, ep, hm, fm)


def perturb_wrong_castling(fen: str, rng: random.Random) -> Optional[str]:
    """Add or remove a castling flag."""
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)

    all_flags = list("KQkq")
    current = set(castling) if castling != "-" else set()

    if current:
        # Either remove one or add one
        if rng.random() < 0.5 and len(current) > 0:
            # Remove a flag
            flag = rng.choice(list(current))
            current.discard(flag)
        else:
            # Add a flag
            missing = [f for f in all_flags if f not in current]
            if missing:
                current.add(rng.choice(missing))
            else:
                # All flags present — remove one
                flag = rng.choice(list(current))
                current.discard(flag)
    else:
        # No castling — add one
        current.add(rng.choice(all_flags))

    new_castling = "".join(f for f in all_flags if f in current) or "-"
    if new_castling == castling:
        return None
    return rebuild_fen(placement, active, new_castling, ep, hm, fm)


def perturb_wrong_en_passant(fen: str, rng: random.Random) -> Optional[str]:
    """Add, remove, or change the en passant square."""
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)

    if ep != "-":
        # Remove it or change it
        if rng.random() < 0.5:
            return rebuild_fen(placement, active, castling, "-", hm, fm)
        else:
            ep_rank = "6" if active == "w" else "3"
            new_file = rng.choice([f for f in "abcdefgh" if f != ep[0]])
            new_ep = new_file + ep_rank
            return rebuild_fen(placement, active, castling, new_ep, hm, fm)
    else:
        # Add a fake en passant
        ep_rank = "6" if active == "w" else "3"
        ep_file = rng.choice(list("abcdefgh"))
        new_ep = ep_file + ep_rank
        return rebuild_fen(placement, active, castling, new_ep, hm, fm)


def perturb_wrong_move_counters(fen: str, rng: random.Random) -> Optional[str]:
    """Off-by-one in halfmove or fullmove clock."""
    placement, active, castling, ep, hm, fm = parse_fen_fields(fen)

    if rng.random() < 0.5:
        # Modify halfmove
        h = int(hm)
        delta = rng.choice([-1, 1])
        new_h = max(0, h + delta)
        if str(new_h) == hm:
            new_h = h + 1
        return rebuild_fen(placement, active, castling, ep, str(new_h), fm)
    else:
        # Modify fullmove
        f = int(fm)
        delta = rng.choice([-1, 1])
        new_f = max(1, f + delta)
        if str(new_f) == fm:
            new_f = f + 1
        return rebuild_fen(placement, active, castling, ep, hm, str(new_f))


# ── Category 3: Next-State Mental Errors ────────────────────────────────────

def _apply_move(fen: str, move_uci: str) -> Optional[chess.Board]:
    """Apply a UCI move to a board, returning the resulting board or None."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return None
        board.push(move)
        return board
    except (ValueError, IndexError):
        return None


def perturb_ghost_piece(fen: str, move_uci: str, rng: random.Random) -> Optional[str]:
    """Apply move but forget to clear the origin square (piece on both squares)."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return None

    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        return None

    # Apply the move normally
    result = board.copy()
    result.push(move)

    # Put the moving piece back on the origin square (the "ghost")
    result.set_piece_at(move.from_square, moving_piece)

    return result.fen()


def perturb_missing_capture(fen: str, move_uci: str, rng: random.Random) -> Optional[str]:
    """Move piece to destination but don't remove the captured piece."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return None

    captured = board.piece_at(move.to_square)
    if captured is None:
        return None  # Not a capture

    moving_piece = board.piece_at(move.from_square)

    # Apply the move normally
    result = board.copy()
    result.push(move)

    # Put the captured piece back (but the moving piece is also there now)
    # Actually: the mover landed on to_square, overwriting captured.
    # "Missing capture" = mover goes to dest, but captured piece stays.
    # We need to put captured on to_square and mover also on to_square — that's impossible.
    # Instead: mover goes to dest but captured piece gets displaced to origin.
    # Simpler interpretation: mover moves, but captured piece remains on to_square.
    # Since mover is already on to_square, we keep captured instead.
    # Better interpretation: don't apply capture — piece lands, captured piece stays.
    # We'll build it manually.
    result_board = board.copy()
    # Manually: remove piece from origin, place on dest (don't remove captured)
    # But python-chess won't allow two pieces on same square.
    # So: origin cleared, dest has moving piece, but we put captured on origin.
    result.set_piece_at(move.from_square, captured)

    return result.fen()


def perturb_no_ep_capture(fen: str, move_uci: str, rng: random.Random) -> Optional[str]:
    """En passant pawn moves but captured pawn remains."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return None
    if not board.is_en_passant(move):
        return None

    # The captured pawn's square
    captured_sq = chess.square(chess.square_file(move.to_square),
                               chess.square_rank(move.from_square))
    captured_pawn = board.piece_at(captured_sq)

    # Apply the move normally (which removes the captured pawn)
    result = board.copy()
    result.push(move)

    # Put the captured pawn back
    if captured_pawn is not None:
        result.set_piece_at(captured_sq, captured_pawn)

    return result.fen()


def perturb_no_castling_rook(fen: str, move_uci: str, rng: random.Random) -> Optional[str]:
    """King moves for castling but rook stays in place."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return None
    if not board.is_castling(move):
        return None

    turn = board.turn

    # Determine rook origin and destination
    if chess.square_file(move.to_square) == 6:  # Kingside
        rook_from = chess.square(7, chess.square_rank(move.from_square))
        rook_to = chess.square(5, chess.square_rank(move.from_square))
    else:  # Queenside
        rook_from = chess.square(0, chess.square_rank(move.from_square))
        rook_to = chess.square(3, chess.square_rank(move.from_square))

    # Apply the move normally
    result = board.copy()
    result.push(move)

    # Undo the rook move: remove rook from destination, put back on origin
    rook_piece = result.piece_at(rook_to)
    if rook_piece is not None:
        result.remove_piece_at(rook_to)
        result.set_piece_at(rook_from, rook_piece)

    return result.fen()


def perturb_wrong_promotion(fen: str, move_uci: str, rng: random.Random) -> Optional[str]:
    """Promote to a different piece than specified."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return None
    if move.promotion is None:
        return None

    # Pick a different promotion piece
    promo_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    alternatives = [p for p in promo_types if p != move.promotion]
    wrong_promo = rng.choice(alternatives)

    # Apply the wrong promotion
    wrong_move = chess.Move(move.from_square, move.to_square, promotion=wrong_promo)
    result = board.copy()
    # Manually apply: clear origin, place promoted piece on dest
    result.remove_piece_at(move.from_square)
    captured = result.piece_at(move.to_square)
    result.set_piece_at(move.to_square, chess.Piece(wrong_promo, board.turn))

    # Update metadata
    new_active = "b" if board.turn == chess.WHITE else "w"
    hm = "0"  # promotion resets halfmove
    fm = str(board.fullmove_number + (1 if board.turn == chess.BLACK else 0))

    arr = board_to_array(result)
    new_placement = array_to_placement(arr)
    # Strip castling rights if rook was captured
    castling = _update_castling_after_move(board, move)
    return rebuild_fen(new_placement, new_active, castling, "-", hm, fm)


def perturb_off_by_one_dest(fen: str, move_uci: str, rng: random.Random) -> Optional[str]:
    """Piece lands on an adjacent square to the correct destination."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return None

    dest_rank = chess.square_rank(move.to_square)
    dest_file = chess.square_file(move.to_square)

    # Find adjacent empty squares to the destination
    neighbors = []
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            nr, nf = dest_rank + dr, dest_file + df
            if 0 <= nr <= 7 and 0 <= nf <= 7:
                sq = chess.square(nf, nr)
                if sq != move.from_square and board.piece_at(sq) is None:
                    neighbors.append((nr, nf))
    if not neighbors:
        return None

    # Apply the move normally first
    result = board.copy()
    result.push(move)

    # Now move the piece from correct dest to wrong dest
    moved_piece = result.piece_at(move.to_square)
    if moved_piece is None:
        return None

    nr, nf = rng.choice(neighbors)
    # Don't put pawns on ranks 1/8
    if moved_piece.piece_type == chess.PAWN and nr in (0, 7):
        neighbors = [(r, f) for r, f in neighbors if r not in (0, 7)]
        if not neighbors:
            return None
        nr, nf = rng.choice(neighbors)

    wrong_sq = chess.square(nf, nr)
    result.remove_piece_at(move.to_square)
    result.set_piece_at(wrong_sq, moved_piece)

    return result.fen()


def _update_castling_after_move(board: chess.Board, move: chess.Move) -> str:
    """Compute castling rights string after a move."""
    result = board.copy()
    result.push(move)
    castling = ""
    if result.has_kingside_castling_rights(chess.WHITE):
        castling += "K"
    if result.has_queenside_castling_rights(chess.WHITE):
        castling += "Q"
    if result.has_kingside_castling_rights(chess.BLACK):
        castling += "k"
    if result.has_queenside_castling_rights(chess.BLACK):
        castling += "q"
    return castling or "-"


def perturb_stale_metadata(fen: str, move_uci: str, rng: random.Random) -> Optional[str]:
    """Correct piece placement after move, but metadata not updated.

    Keeps active color, castling, en passant, or move counters from before the move.
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return None

    _, old_active, old_castling, old_ep, old_hm, old_fm = parse_fen_fields(fen)

    result = board.copy()
    result.push(move)
    correct_fen = result.fen()
    new_placement = parse_fen_fields(correct_fen)[0]
    _, new_active, new_castling, new_ep, new_hm, new_fm = parse_fen_fields(correct_fen)

    # Pick a metadata field to keep stale
    stale_options = []
    if old_active != new_active:
        stale_options.append("active")
    if old_castling != new_castling:
        stale_options.append("castling")
    if old_ep != new_ep:
        stale_options.append("ep")
    if old_hm != new_hm or old_fm != new_fm:
        stale_options.append("counters")

    if not stale_options:
        return None

    choice = rng.choice(stale_options)
    if choice == "active":
        return rebuild_fen(new_placement, old_active, new_castling, new_ep, new_hm, new_fm)
    elif choice == "castling":
        return rebuild_fen(new_placement, new_active, old_castling, new_ep, new_hm, new_fm)
    elif choice == "ep":
        return rebuild_fen(new_placement, new_active, new_castling, old_ep, new_hm, new_fm)
    else:  # counters
        return rebuild_fen(new_placement, new_active, new_castling, new_ep, old_hm, old_fm)


# ── Main entry point ────────────────────────────────────────────────────────

# Registry mapping flag name → perturbation function (and whether it needs a move)
_PLACEMENT_PERTURBATIONS: Dict[str, callable] = {
    "swap_pieces": perturb_swap_pieces,
    "shift_piece": perturb_shift_piece,
    "mirror_horizontal": perturb_mirror_horizontal,
    "mirror_vertical": perturb_mirror_vertical,
    "change_piece_color": perturb_change_piece_color,
    "change_piece_type": perturb_change_piece_type,
    "add_piece": perturb_add_piece,
    "remove_piece": perturb_remove_piece,
    "swap_adjacent_ranks": perturb_swap_adjacent_ranks,
    "empty_square_error": perturb_empty_square_error,
}

_METADATA_PERTURBATIONS: Dict[str, callable] = {
    "flip_active_color": perturb_flip_active_color,
    "wrong_castling": perturb_wrong_castling,
    "wrong_en_passant": perturb_wrong_en_passant,
    "wrong_move_counters": perturb_wrong_move_counters,
}

_NEXT_STATE_PERTURBATIONS: Dict[str, callable] = {
    "ghost_piece": perturb_ghost_piece,
    "missing_capture": perturb_missing_capture,
    "no_ep_capture": perturb_no_ep_capture,
    "no_castling_rook": perturb_no_castling_rook,
    "wrong_promotion": perturb_wrong_promotion,
    "off_by_one_dest": perturb_off_by_one_dest,
    "stale_metadata": perturb_stale_metadata,
}

_MAX_RETRIES = 50


def generate_wrong_fens(
    fen: str,
    n: int = 3,
    # --- Piece placement perturbations (bool flags) ---
    swap_pieces: bool = True,
    shift_piece: bool = True,
    mirror_horizontal: bool = True,
    mirror_vertical: bool = True,
    change_piece_color: bool = True,
    change_piece_type: bool = True,
    add_piece: bool = True,
    remove_piece: bool = True,
    swap_adjacent_ranks: bool = True,
    empty_square_error: bool = True,
    # --- Metadata perturbations ---
    flip_active_color: bool = False,
    wrong_castling: bool = False,
    wrong_en_passant: bool = False,
    wrong_move_counters: bool = False,
    # --- Next-state mental errors (require `move`) ---
    ghost_piece: bool = False,
    missing_capture: bool = False,
    no_ep_capture: bool = False,
    no_castling_rook: bool = False,
    wrong_promotion: bool = False,
    off_by_one_dest: bool = False,
    stale_metadata: bool = False,
    # ---
    move: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """Return n plausible-but-wrong FEN strings as distractors.

    Enabled perturbations are sampled uniformly. Each result is
    validated (parseable, one king/side, no pawns on 1/8) and
    deduplicated against the correct FEN and all other distractors.

    Parameters
    ----------
    fen : str
        The correct FEN to generate distractors for.
    n : int
        Number of distractor FENs to generate.
    move : str or None
        UCI move string, required when next-state flags are enabled.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list[str]
        Up to *n* wrong FEN strings (may return fewer if perturbations
        are exhausted for the given position).
    """
    rng = random.Random(seed)

    # Build list of enabled perturbation functions
    local_flags = {
        "swap_pieces": swap_pieces,
        "shift_piece": shift_piece,
        "mirror_horizontal": mirror_horizontal,
        "mirror_vertical": mirror_vertical,
        "change_piece_color": change_piece_color,
        "change_piece_type": change_piece_type,
        "add_piece": add_piece,
        "remove_piece": remove_piece,
        "swap_adjacent_ranks": swap_adjacent_ranks,
        "empty_square_error": empty_square_error,
        "flip_active_color": flip_active_color,
        "wrong_castling": wrong_castling,
        "wrong_en_passant": wrong_en_passant,
        "wrong_move_counters": wrong_move_counters,
        "ghost_piece": ghost_piece,
        "missing_capture": missing_capture,
        "no_ep_capture": no_ep_capture,
        "no_castling_rook": no_castling_rook,
        "wrong_promotion": wrong_promotion,
        "off_by_one_dest": off_by_one_dest,
        "stale_metadata": stale_metadata,
    }

    enabled = []
    for name, flag in local_flags.items():
        if not flag:
            continue
        if name in _PLACEMENT_PERTURBATIONS:
            enabled.append((name, _PLACEMENT_PERTURBATIONS[name], False))
        elif name in _METADATA_PERTURBATIONS:
            enabled.append((name, _METADATA_PERTURBATIONS[name], False))
        elif name in _NEXT_STATE_PERTURBATIONS:
            if move is None:
                raise ValueError(
                    f"Next-state perturbation '{name}' requires `move` parameter"
                )
            enabled.append((name, _NEXT_STATE_PERTURBATIONS[name], True))

    if not enabled:
        return []

    results: List[str] = []
    seen: set = {fen}

    for _ in range(_MAX_RETRIES):
        if len(results) >= n:
            break

        # Sample a random perturbation
        name, func, needs_move = rng.choice(enabled)

        if needs_move:
            candidate = func(fen, move, rng)
        else:
            candidate = func(fen, rng)

        if candidate is None:
            continue

        # For next-state perturbations, validate against the *correct result*
        # (the FEN after applying the move), not the input FEN
        if needs_move:
            correct_result = _apply_move(fen, move)
            validation_fen = correct_result.fen() if correct_result else fen
        else:
            validation_fen = fen

        if not is_valid_distractor(candidate, validation_fen):
            # Also check it's not equal to the input FEN itself
            if candidate == fen:
                continue
            # For next-state, also accept if different from input
            if not needs_move:
                continue
            if candidate == validation_fen:
                continue
            # Re-validate without the "different from correct" check
            # — just check parseability and king/pawn constraints
            try:
                test_board = chess.Board(candidate)
            except ValueError:
                continue
            wk = len(test_board.pieces(chess.KING, chess.WHITE))
            bk = len(test_board.pieces(chess.KING, chess.BLACK))
            if wk != 1 or bk != 1:
                continue
            pawn_ok = True
            for file in range(8):
                for rank in (0, 7):
                    p = test_board.piece_at(chess.square(file, rank))
                    if p is not None and p.piece_type == chess.PAWN:
                        pawn_ok = False
                        break
                if not pawn_ok:
                    break
            if not pawn_ok:
                continue

        if candidate in seen:
            continue

        seen.add(candidate)
        results.append(candidate)

    return results
