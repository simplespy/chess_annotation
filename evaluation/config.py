"""Configuration constants for chess evaluation tools."""
import os

# Stockfish path - can be overridden by STOCKFISH_PATH environment variable
STOCKFISH_PATH = os.getenv('STOCKFISH_PATH', '/opt/homebrew/bin/stockfish')

# Lichess win% formula constant
K = 0.00368208
