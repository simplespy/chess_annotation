import os
import json
import numpy as np
import pandas as pd
import chess
import chess.svg
import chess.engine
import openai
from stockfish import Stockfish
import math
from tqdm import tqdm

K = 0.00368208
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JSONL_PATH = "data/1k.jsonl"

def cp_to_winpct(cp):
    """Lichess win% formula: Win% = 50 + 50*tanh(k*cp/2)."""
    return 50 + 50 * math.tanh(K * cp / 2)

def filter_dataset(jsonl_path, depth=20, wp_threshold=10):
    """Split dataset into optimal-move and other-move JSONL files.

    A move is 'optimal' if it is the top engine move OR the win%
    loss (from the moving side's perspective) is <= wp_threshold.
    """
    dataset = [json.loads(line) for line in open(jsonl_path)]
    base = os.path.splitext(jsonl_path)[0]
    optimal_path = f"{base}_optimal_move.jsonl"
    other_path = f"{base}_other_move.jsonl"

    optimal, other = [], []

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        for entry in tqdm(dataset, desc="Classifying moves"):
            fen = entry["fen"]
            move_uci = entry["move_uci"]
            board = chess.Board(fen)
            played_move = chess.Move.from_uci(move_uci)
            turn = board.turn

            # Best-move analysis
            info_best = engine.analyse(board, chess.engine.Limit(depth=depth))
            best_move = info_best["pv"][0]
            best_cp = info_best["score"].white().score(mate_score=10000)

            best_san = board.san(best_move)
            entry["best_move_uci"] = best_move.uci()
            entry["best_move_san"] = best_san

            # Played move IS the best move
            if played_move == best_move:
                entry["wp_loss"] = 0.0
                entry["is_top_engine_move"] = True
                optimal.append(entry)
                continue

            # Evaluate position after the played move
            board.push(played_move)
            info_played = engine.analyse(board, chess.engine.Limit(depth=depth))
            played_cp = info_played["score"].white().score(mate_score=10000)

            best_wp = cp_to_winpct(best_cp)
            played_wp = cp_to_winpct(played_cp)

            # Win% loss from the moving side's perspective
            if turn == chess.WHITE:
                wp_loss = best_wp - played_wp
            else:
                wp_loss = played_wp - best_wp

            entry["wp_loss"] = round(wp_loss, 2)
            entry["is_top_engine_move"] = False

            if wp_loss <= wp_threshold:
                optimal.append(entry)
            else:
                other.append(entry)

    # Write output files
    with open(optimal_path, "w") as f:
        for e in optimal:
            f.write(json.dumps(e) + "\n")

    with open(other_path, "w") as f:
        for e in other:
            f.write(json.dumps(e) + "\n")

    print(f"Optimal: {len(optimal)} | Other: {len(other)} | Total: {len(dataset)}")
    print(f"  -> {optimal_path}")
    print(f"  -> {other_path}")
    return optimal, other

optimal, other = filter_dataset(JSONL_PATH, depth=20)