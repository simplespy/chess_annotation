#!/usr/bin/env python3
"""Tests for fen_question_gen.py — verify generated answers against python-chess.

Run:
    python test_fen_questions.py
    python test_fen_questions.py -v          # verbose
    python test_fen_questions.py --from-data  # also validate generated JSONL files
"""

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

from fen_question_gen import (
    GENERATORS,
    MOVE_GENERATORS,
    generate_questions,
    _material,
    _is_light_square,
)


# ── Test positions ───────────────────────────────────────────────────────────
# Each covers a different interesting scenario.

# Starting position after 1.e4: EP available, all castling, known material
FEN_AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# Mid-game: imbalanced material, some castling lost, pins
FEN_MIDGAME = "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 4 9"

# Endgame: few pieces, passed pawns likely
FEN_ENDGAME = "8/5k2/4p3/3pP1K1/2pP4/2P5/8/8 w - - 0 50"

# Position with check
FEN_CHECK = "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3"

# Position with doubled pawns, isolated pawns
FEN_PAWN_STRUCTURE = "r1bqkb1r/pp3ppp/2n1pn2/2pp4/3P1B2/4PN2/PPP2PPP/RN1QKB1R w KQkq - 0 5"


class TestSquareContents(unittest.TestCase):
    """Verify piece_on_square, is_square_occupied, color_on_square."""

    def _extract_square(self, question_text):
        """Pull the algebraic square name out of any square_contents question."""
        # Formats: "What piece is on e4?", "Is e4 occupied or empty?",
        #          "What color is the piece on e4?"
        for word in question_text.replace("?", "").split():
            if len(word) == 2 and word[0] in "abcdefgh" and word[1] in "12345678":
                return chess.parse_square(word)
        raise ValueError(f"No square found in: {question_text}")

    def test_starting_position_pieces(self):
        """Known pieces on known squares in near-starting position."""
        board = chess.Board(FEN_AFTER_E4)
        qs = generate_questions(FEN_AFTER_E4, categories=["square_contents"],
                                n_per_category=64, seed=0)

        for q in qs:
            sq = self._extract_square(q["question"])
            piece = board.piece_at(sq)

            if q["subcategory"] == "piece_on_square":
                if piece is None:
                    self.assertEqual(q["answer"], "Empty")
                else:
                    self.assertIn(chess.piece_name(piece.piece_type), q["answer"].lower())

            elif q["subcategory"] == "is_square_occupied":
                expected = "Occupied" if piece else "Empty"
                self.assertEqual(q["answer"], expected)

            elif q["subcategory"] == "color_on_square":
                if piece is None:
                    self.assertEqual(q["answer"], "Empty")
                else:
                    expected = "White" if piece.color == chess.WHITE else "Black"
                    self.assertEqual(q["answer"], expected)


class TestPieceCounting(unittest.TestCase):
    """Verify count_piece_type, count_color_pieces, count_all_pieces."""

    def test_starting_position_counts(self):
        board = chess.Board(FEN_AFTER_E4)
        qs = generate_questions(FEN_AFTER_E4, categories=["piece_counting"],
                                n_per_category=20, seed=0)
        for q in qs:
            if q["subcategory"] == "count_all_pieces":
                # 32 pieces after 1.e4
                self.assertEqual(q["answer"], "32")

            elif q["subcategory"] == "count_color_pieces":
                # Each side has 16
                self.assertEqual(q["answer"], "16")

    def test_endgame_counts(self):
        board = chess.Board(FEN_ENDGAME)
        total = sum(1 for sq in range(64) if board.piece_at(sq))
        qs = generate_questions(FEN_ENDGAME, categories=["piece_counting"],
                                n_per_category=20, seed=0)
        for q in qs:
            if q["subcategory"] == "count_all_pieces":
                self.assertEqual(q["answer"], str(total))


class TestMaterial(unittest.TestCase):
    """Verify material_value, which_side_more_material, is_material_equal."""

    def test_starting_position_equal(self):
        qs = generate_questions(FEN_AFTER_E4, categories=["material"],
                                n_per_category=10, seed=0)
        for q in qs:
            if q["subcategory"] == "is_material_equal":
                self.assertEqual(q["answer"], "Yes")
            elif q["subcategory"] == "which_side_more_material":
                self.assertEqual(q["answer"], "Equal")
            elif q["subcategory"] == "material_value":
                # Both sides: 8P+2N+2B+2R+Q = 8+6+6+10+9 = 39
                self.assertEqual(q["answer"], "39")

    def test_imbalanced_material(self):
        # White has extra knight compared to a standard position
        fen = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen)
        w = _material(board, chess.WHITE)
        b = _material(board, chess.BLACK)
        qs = generate_questions(fen, categories=["material"],
                                n_per_category=10, seed=0)
        for q in qs:
            if q["subcategory"] == "which_side_more_material":
                if w > b:
                    self.assertEqual(q["answer"], "White")
                elif b > w:
                    self.assertEqual(q["answer"], "Black")
                else:
                    self.assertEqual(q["answer"], "Equal")


class TestMetadata(unittest.TestCase):
    """Verify side_to_move, castling, en passant, counters."""

    def test_after_e4_metadata(self):
        qs = generate_questions(FEN_AFTER_E4, categories=["metadata"],
                                n_per_category=20, seed=0)
        found = {}
        for q in qs:
            found[q["subcategory"]] = q["answer"]

        self.assertEqual(found["side_to_move"], "Black")
        self.assertEqual(found.get("en_passant_square"), "Yes, e3")
        self.assertEqual(found.get("fullmove_number"), "1")
        self.assertEqual(found.get("halfmove_clock"), "0")

    def test_no_castling(self):
        """Position with no castling rights."""
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
        qs = generate_questions(fen, categories=["metadata"],
                                n_per_category=20, seed=0)
        for q in qs:
            if "castle" in q["question"].lower():
                self.assertEqual(q["answer"], "No")


class TestPieceLocation(unittest.TestCase):
    """Verify king_square, list_piece_squares, is_piece_on_square."""

    def test_king_squares(self):
        board = chess.Board(FEN_AFTER_E4)
        qs = generate_questions(FEN_AFTER_E4, categories=["piece_location"],
                                n_per_category=20, seed=0)
        for q in qs:
            if q["subcategory"] == "king_square":
                if "white" in q["question"].lower():
                    self.assertEqual(q["answer"], "e1")
                elif "black" in q["question"].lower():
                    self.assertEqual(q["answer"], "e8")


class TestRankFile(unittest.TestCase):
    """Verify count_on_rank, count_on_file, is_file_open, is_file_half_open."""

    def test_half_open_not_open(self):
        """An open file (no pawns at all) should NOT count as half-open."""
        # e-file: white pawn on e4, no black pawn → half-open for Black
        # but if a file has NO pawns at all, it's open, not half-open
        board = chess.Board(FEN_AFTER_E4)
        qs = generate_questions(FEN_AFTER_E4, categories=["rank_file"],
                                n_per_category=50, seed=0)
        for q in qs:
            if q["subcategory"] == "is_file_half_open":
                # Parse file from question: "Is the X-file half-open for Y?"
                file_name = q["question"].split("the ")[1].split("-file")[0]
                file_idx = "abcdefgh".index(file_name)
                # Parse color
                color_name = q["question"].split("for ")[1].split("?")[0]
                color = chess.WHITE if "White" in color_name else chess.BLACK
                opp = not color

                # Verify: half-open = no own pawns AND opponent has pawn
                own_pawn = any(
                    board.piece_at(chess.square(file_idx, r))
                    and board.piece_at(chess.square(file_idx, r)).piece_type == chess.PAWN
                    and board.piece_at(chess.square(file_idx, r)).color == color
                    for r in range(8)
                )
                opp_pawn = any(
                    board.piece_at(chess.square(file_idx, r))
                    and board.piece_at(chess.square(file_idx, r)).piece_type == chess.PAWN
                    and board.piece_at(chess.square(file_idx, r)).color == opp
                    for r in range(8)
                )
                expected = "Yes" if (not own_pawn and opp_pawn) else "No"
                self.assertEqual(q["answer"], expected,
                                 f"File {file_name}, color {color_name}: "
                                 f"own_pawn={own_pawn}, opp_pawn={opp_pawn}")

    def test_open_file(self):
        """Verify is_file_open against python-chess."""
        board = chess.Board(FEN_MIDGAME)
        qs = generate_questions(FEN_MIDGAME, categories=["rank_file"],
                                n_per_category=50, seed=0)
        for q in qs:
            if q["subcategory"] == "is_file_open":
                file_name = q["question"].split("the ")[1].split("-file")[0]
                file_idx = "abcdefgh".index(file_name)
                has_pawn = any(
                    board.piece_at(chess.square(file_idx, r))
                    and board.piece_at(chess.square(file_idx, r)).piece_type == chess.PAWN
                    for r in range(8)
                )
                expected = "No" if has_pawn else "Yes"
                self.assertEqual(q["answer"], expected)


class TestPawnStructure(unittest.TestCase):
    """Verify doubled_pawns, isolated_pawn, pawn_islands, passed_pawn."""

    def test_known_doubled_pawns(self):
        """Position with white doubled e-pawns."""
        fen = "rnbqkbnr/pppp1ppp/8/8/4P3/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen)
        qs = generate_questions(fen, categories=["pawn_structure"],
                                n_per_category=20, seed=0)
        for q in qs:
            if q["subcategory"] == "doubled_pawns" and "White" in q["question"]:
                self.assertIn("Yes", q["answer"])
                self.assertIn("e-file", q["answer"])

    def test_pawn_islands_count(self):
        """Verify pawn island count."""
        # White pawns on a,b (1 island) and e (1 island) = 2 islands
        fen = "8/8/8/8/4P3/PP6/8/4K2k w - - 0 1"
        board = chess.Board(fen)
        qs = generate_questions(fen, categories=["pawn_structure"],
                                n_per_category=20, seed=0)
        for q in qs:
            if q["subcategory"] == "pawn_islands" and "White" in q["question"]:
                self.assertEqual(q["answer"], "2")


class TestChecksAttacks(unittest.TestCase):
    """Verify is_in_check, attackers_of_square, is_square_attacked."""

    def test_check_detected(self):
        """Qh4 gives check to white king."""
        qs = generate_questions(FEN_CHECK, categories=["checks_attacks"],
                                n_per_category=20, seed=0)
        for q in qs:
            if q["subcategory"] == "is_in_check":
                board = chess.Board(FEN_CHECK)
                if "white" in q["question"].lower():
                    king = board.king(chess.WHITE)
                    in_check = len(board.attackers(chess.BLACK, king)) > 0
                    expected = "Yes" if in_check else "No"
                    self.assertEqual(q["answer"], expected)

    def test_attackers_count(self):
        """Verify attacker counts against python-chess."""
        board = chess.Board(FEN_MIDGAME)
        qs = generate_questions(FEN_MIDGAME, categories=["checks_attacks"],
                                n_per_category=20, seed=0)
        for q in qs:
            if q["subcategory"] == "attackers_of_square":
                # Parse: "How many {color} pieces attack {square}?"
                parts = q["question"].split()
                sq_name = parts[-1].rstrip("?")
                sq = chess.parse_square(sq_name)
                color = chess.WHITE if "white" in q["question"].lower() else chess.BLACK
                expected = len(board.attackers(color, sq))
                self.assertEqual(q["answer"], str(expected))


class TestBoardGeometry(unittest.TestCase):
    """Verify square_color, bishop_pair, bishop_square_colors."""

    def test_square_colors(self):
        """a1 is dark, a2 is light, etc."""
        qs = generate_questions(FEN_AFTER_E4, categories=["board_geometry"],
                                n_per_category=20, seed=0)
        for q in qs:
            if q["subcategory"] == "square_color":
                sq_name = q["question"].split("Is ")[1].split(" a ")[0]
                sq = chess.parse_square(sq_name)
                expected = "Light" if _is_light_square(sq) else "Dark"
                self.assertEqual(q["answer"], expected)

    def test_bishop_pair(self):
        """Starting position has bishop pair for both sides."""
        qs = generate_questions(FEN_AFTER_E4, categories=["board_geometry"],
                                n_per_category=20, seed=0)
        for q in qs:
            if q["subcategory"] == "bishop_pair":
                self.assertEqual(q["answer"], "Yes")


class TestAfterMove(unittest.TestCase):
    """Verify en_passant_after_move."""

    def test_double_push_creates_ep(self):
        """1.e4 from starting position creates EP on e3."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        qs = generate_questions(fen, categories=["after_move"],
                                n_per_category=5, seed=0, move_uci="e2e4")
        self.assertTrue(len(qs) > 0)
        q = qs[0]
        self.assertEqual(q["answer"], "Yes, e3")

    def test_non_pawn_move_no_ep(self):
        """Nf3 doesn't create EP."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        qs = generate_questions(fen, categories=["after_move"],
                                n_per_category=5, seed=0, move_uci="g1f3")
        self.assertTrue(len(qs) > 0)
        self.assertEqual(qs[0]["answer"], "No")


class TestAllCategoriesCovered(unittest.TestCase):
    """Ensure every category produces questions and has required fields."""

    def test_all_categories_produce_output(self):
        qs = generate_questions(FEN_MIDGAME, n_per_category=5, seed=42,
                                move_uci="c4b5")
        cats = set(q["category"] for q in qs)
        for expected in GENERATORS:
            self.assertIn(expected, cats, f"Category {expected} missing")

    def test_required_fields(self):
        qs = generate_questions(FEN_MIDGAME, n_per_category=3, seed=42)
        for q in qs:
            self.assertIn("fen", q)
            self.assertIn("question", q)
            self.assertIn("answer", q)
            self.assertIn("category", q)
            self.assertIn("subcategory", q)
            self.assertTrue(len(q["question"]) > 0)
            self.assertTrue(len(q["answer"]) > 0)


class TestGeneratedData(unittest.TestCase):
    """Validate generated JSONL files in data/ against python-chess.

    Only runs with --from-data flag.
    """

    DATA_DIR = Path(__file__).parent / "data"

    def _load_subcategory(self, name: str) -> list[dict]:
        path = self.DATA_DIR / f"{name}.jsonl"
        if not path.exists():
            self.skipTest(f"{path} not found")
        with path.open() as f:
            return [json.loads(line) for line in f]

    def test_is_in_check_answers(self):
        """Every is_in_check answer matches python-chess attackers."""
        rows = self._load_subcategory("is_in_check")
        for row in rows:
            board = chess.Board(row["fen"])
            # Parse which color king from question
            if "white" in row["question"].lower():
                king_sq = board.king(chess.WHITE)
                attackers = board.attackers(chess.BLACK, king_sq) if king_sq else chess.SquareSet()
            else:
                king_sq = board.king(chess.BLACK)
                attackers = board.attackers(chess.WHITE, king_sq) if king_sq else chess.SquareSet()
            expected = "Yes" if len(attackers) > 0 else "No"
            self.assertEqual(row["answer"], expected,
                             f"FEN: {row['fen']}, Q: {row['question']}")

    def test_side_to_move_answers(self):
        """side_to_move matches board.turn."""
        rows = self._load_subcategory("side_to_move")
        for row in rows:
            board = chess.Board(row["fen"])
            expected = "White" if board.turn == chess.WHITE else "Black"
            self.assertEqual(row["answer"], expected)

    def test_material_value_answers(self):
        """material_value matches manual counting."""
        rows = self._load_subcategory("material_value")
        for row in rows:
            board = chess.Board(row["fen"])
            if "White" in row["question"]:
                expected = _material(board, chess.WHITE)
            else:
                expected = _material(board, chess.BLACK)
            self.assertEqual(row["answer"], str(expected),
                             f"FEN: {row['fen']}, Q: {row['question']}")

    def test_can_castle_answers(self):
        """Castling rights match python-chess."""
        for subcat in ("can_castle_kingside", "can_castle_queenside"):
            rows = self._load_subcategory(subcat)
            for row in rows:
                board = chess.Board(row["fen"])
                if "White" in row["question"]:
                    color = chess.WHITE
                else:
                    color = chess.BLACK
                if "kingside" in row["question"]:
                    has = board.has_kingside_castling_rights(color)
                else:
                    has = board.has_queenside_castling_rights(color)
                expected = "Yes" if has else "No"
                self.assertEqual(row["answer"], expected,
                                 f"FEN: {row['fen']}, Q: {row['question']}")

    def test_en_passant_square_answers(self):
        """EP square matches board.ep_square."""
        rows = self._load_subcategory("en_passant_square")
        for row in rows:
            board = chess.Board(row["fen"])
            if board.ep_square is not None:
                expected = f"Yes, {chess.square_name(board.ep_square)}"
            else:
                expected = "No"
            self.assertEqual(row["answer"], expected)

    def test_square_color_answers(self):
        """Light/dark matches coordinate parity."""
        rows = self._load_subcategory("square_color")
        for row in rows:
            sq_name = row["question"].split("Is ")[1].split(" a ")[0]
            sq = chess.parse_square(sq_name)
            expected = "Light" if _is_light_square(sq) else "Dark"
            self.assertEqual(row["answer"], expected)

    def test_is_square_attacked_answers(self):
        """is_square_attacked matches board.is_attacked_by."""
        rows = self._load_subcategory("is_square_attacked")
        for row in rows:
            board = chess.Board(row["fen"])
            sq_name = row["question"].split("Is ")[1].split(" attacked")[0]
            sq = chess.parse_square(sq_name)
            color = chess.WHITE if "white" in row["question"].lower().split("by")[1] else chess.BLACK
            expected = "Yes" if board.is_attacked_by(color, sq) else "No"
            self.assertEqual(row["answer"], expected,
                             f"FEN: {row['fen']}, Q: {row['question']}")

    def test_all_files_valid_json(self):
        """Every JSONL file has valid JSON with required fields."""
        if not self.DATA_DIR.exists():
            self.skipTest("data/ not found")
        for path in self.DATA_DIR.glob("*.jsonl"):
            with path.open() as f:
                for i, line in enumerate(f):
                    row = json.loads(line)
                    self.assertIn("fen", row, f"{path.name}:{i}")
                    self.assertIn("question", row, f"{path.name}:{i}")
                    self.assertIn("answer", row, f"{path.name}:{i}")
                    self.assertIn("category", row, f"{path.name}:{i}")
                    self.assertIn("subcategory", row, f"{path.name}:{i}")
                    # FEN should be parseable
                    chess.Board(row["fen"])


if __name__ == "__main__":
    # Support --from-data flag to run data validation tests too
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--from-data", action="store_true",
                        help="Also validate generated JSONL files")
    known, remaining = parser.parse_known_args()

    if not known.from_data:
        # Remove data tests from the suite
        for name in list(vars(TestGeneratedData)):
            if name.startswith("test_"):
                delattr(TestGeneratedData, name)

    unittest.main(argv=[sys.argv[0]] + remaining)
