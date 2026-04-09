# SFT Legal Moves

Train and evaluate models on **legal-move identification** in chess. Given a position and a small set of candidate moves (mix of legal and illegal), the model must identify which are legal, with type-specific reasoning.

## Quick Start

```bash
source activate dev  # or: pip install python-chess

# Step 1: Extract positions from PGN → intermediate JSONL
python extract_positions.py \
    --pgn_path data/lichess_db_standard_rated_2013-02.pgn \
    --out_path data/train_positions.jsonl \
    --max_games 1400 --seed 42

# Step 2: Generate SFT reasoning traces
python generate_sft_legal_moves.py \
    --data_path data/train_positions.jsonl \
    --template_dir reasoning_templates/ \
    --out_path data/sft_legal_moves_train.jsonl \
    --seed 42
```

## Data Splits

| Split | PGN source | Positions | SFT traces | Seed |
|---|---|---|---|---|
| **Train** | `lichess_db_standard_rated_2013-02.pgn` | 31,143 | 31,143 | 42 |
| **Eval** | `lichess_db_standard_rated_2013-01.pgn` | 29,016 | 29,016 | 99 |

Both extracted with `--max_games 1400 --num_vanilla 2000 --num_distractors 5`.

### Tag distribution (train / eval)

| Tag | Train | Eval |
|---|---|---|
| `illegal_king` | 22,317 | 20,380 |
| `pin` | 4,361 | 4,199 |
| `check` | 3,254 | 3,134 |
| `vanilla` | 2,000 | 2,000 |
| `promotion` | 528 | 489 |
| `en_passant` | 168 | 120 |
| `double_check` | 20 | 21 |

## Pipeline

```
PGN file
  │
  ▼
extract_positions.py          →  train_positions.jsonl / eval_positions.jsonl
  (detect categories,             (intermediate: FEN + typed illegal moves)
   build illegal distractors)
  │
  ▼
generate_sft_legal_moves.py   →  sft_legal_moves_train.jsonl / sft_legal_moves_eval.jsonl
  (sample candidates,              (final: input prompt + reasoning trace)
   fill reasoning templates)
  │
  ▼
generate_eval_binary.py       →  eval_binary.jsonl
  (flatten to per-move rows,       (binary classification: fen + move + label)
   balanced sampling)
```

## File Structure

```
sft_legal_moves/
├── legal_moves.py                # Shared helpers: piece descriptions, SAN conversion,
│                                 #   attacker/pinner/blocker detection, castling geometry,
│                                 #   phase classification, PGN iterator
├── legal_move_puzzles.py         # Position detectors (detect_*), illegal-move builders
│                                 #   (build_*), general distractors, extract_all(),
│                                 #   classify_legal_move(), SUBCATEGORY_TO_CATEGORY
├── extract_positions.py          # CLI: PGN → intermediate JSONL
├── generate_sft_legal_moves.py   # CLI: intermediate JSONL → SFT traces
├── generate_eval_binary.py       # CLI: intermediate JSONL → binary eval JSONL
├── extract_eval_positions.ipynb  # Interactive notebook for exploration/debugging
├── visualize_reasoning.ipynb    # Visualize boards + template-filled reasoning traces
├── reasoning_templates/          # One .txt template per move type
│   ├── reasoning_template.txt    #   Wrapper: <think>...<answer> structure
│   ├── legal_move.txt            #   Legal move (non-check)
│   ├── legal_king_escape.txt     #   King escapes check
│   ├── legal_capture_checker.txt #   Piece captures the checking piece
│   ├── legal_block_check.txt     #   Piece blocks the check ray
│   ├── king_to_attacked.txt      #   King moves to attacked square
│   ├── castling_through_attacked.txt
│   ├── castling_in_check.txt
│   ├── pin_breaking.txt
│   ├── non_king_double_check.txt
│   ├── ep_fake_diagonal.txt      #   Pawn diagonal to empty (no adjacent enemy pawn)
│   ├── ep_wrong_pawn.txt         #   Adjacent enemy pawn didn't just double-push
│   ├── promo_push_blocked.txt
│   ├── promo_capture_empty.txt
│   ├── backward_pawn.txt
│   ├── friendly_fire.txt
│   ├── blocked_sliding.txt
│   ├── pawn_double_wrong_rank.txt
│   ├── pawn_push_onto_piece.txt
│   ├── pawn_diagonal_to_empty.txt  #   Pawn diagonal to empty (no capture, non-EP)
│   ├── pawn_capture_friendly.txt   #   Pawn captures own piece diagonally
│   └── wrong_geometry.txt
├── data/
│   ├── lichess_db_standard_rated_2013-01.pgn  # Eval source
│   ├── lichess_db_standard_rated_2013-02.pgn  # Train source
│   ├── train_positions.jsonl       # Intermediate: extracted positions (train)
│   ├── eval_positions.jsonl        # Intermediate: extracted positions (eval)
│   ├── sft_legal_moves_train.jsonl # Final SFT data (train)
│   └── sft_legal_moves_eval.jsonl  # Final SFT data (eval)
└── README.md
```

## Output Formats

### Intermediate JSONL (extract_positions.py)

| Field | Type | Description |
|---|---|---|
| `fen` | str | Board position (FEN) |
| `last_move_uci` | str | Move that led to this position |
| `game_move_uci` | str | Move actually played in the game |
| `next_move_candidates_uci` | list[str] | All candidate moves (legal + illegal) |
| `correct_outputs_uci` | list[str] | All legal moves |
| `illegal_category` | list[dict] | Category-specific illegals: `{"uci", "type"}` |
| `illegal_general` | list[dict] | General distractors: `{"uci", "type"}` |
| `tags` | list[str] | Position categories |
| `phase` | str | `opening` / `middlegame` / `endgame` |

### SFT JSONL (generate_sft_legal_moves.py)

| Field | Type | Description |
|---|---|---|
| `input` | str | Prompt: FEN + previous move + candidate list (SAN) |
| `output` | str | `<think>` per-move analysis `</think><answer>\boxed{legal moves}</answer>` |
| `fen` | str | Board position for reference |
| `tags` | list[str] | Position categories for filtering |

### Binary Eval JSONL (generate_eval_binary.py)

| Field | Type | Description |
|---|---|---|
| `fen` | str | Board position (FEN) |
| `move_uci` | str | Move in UCI notation |
| `move_san` | str | Move in SAN notation |
| `label` | str | `legal` or `illegal` |
| `category` | str | Parent category (e.g. `king`, `pawn`, `legal`) |
| `subcategory` | str | Specific type (e.g. `king_to_attacked`, `legal_capture`) |
| `position_tags` | list[str] | Position-level tags |
| `phase` | str | `opening` / `middlegame` / `endgame` |

**Example output** (check position -- legal moves explain how they resolve the check):
```
<think>
The current position is: r3k2r/pp2b1pp/2p1N1b1/q7/8/3P1Q2/PPP2PPP/R3K2R w KQkq - 1 15.
The previous move was d8a5. It is White's turn.

I need to determine which of these candidate moves are legal: Kd1, Kf1, c3, O-O-O, Kd2, O-O

Consider the move Kd1. The king is in check. This king move to d1 escapes the check. This is legal.
Consider the move Kf1. The king is in check. This king move to f1 escapes the check. This is legal.
Consider the move c3. The king is in check from the black queen on a5. The pawn moves to c3,
  blocking the line of attack. This is legal.
Consider the move O-O-O. This is castling, but the king is currently in check.
  I cannot castle while in check, so this is illegal.
Consider the move Kd2. This moves the king to d2, which is controlled by
  the black queen on a5. Moving the king to an attacked square is illegal.
Consider the move O-O. This is castling, but the king is currently in check.
  I cannot castle while in check, so this is illegal.

From the candidates, the legal moves are: Kd1, Kf1, c3
</think>
<answer>
\boxed{Kd1, Kf1, c3}
</answer>
```

### Sampling strategy (generate_sft_legal_moves.py)

Per position, the candidate set contains:
1. **All category illegals** (the point of the exercise)
2. **`--num_illegal_gen` general illegals** (default 2, randomly sampled)
3. **`--num_legal` legal moves** (default 3, always includes the game move)

Candidates are shuffled before presentation.

## Position Categories

### Legal move reasoning

In check positions, legal moves get check-specific explanations:

| Template | When used |
|---|---|
| `legal_king_escape` | King move that escapes check |
| `legal_capture_checker` | Non-king piece captures the checking piece |
| `legal_block_check` | Non-king piece interposes on the check ray |
| `legal_move` | All legal moves in non-check positions |

In non-check positions, `legal_move` annotates captures, castling, en passant, promotions, and delivered checks.

### Category-specific illegal move types

| Category | Illegal types | Description |
|---|---|---|
| `en_passant` | `ep_fake_diagonal`, `ep_wrong_pawn` | Diagonal to empty (no ep target) or adjacent pawn didn't just push |
| `check` | `king_to_attacked`, `castling_in_check`, `non_evasion_in_check` | King to attacked square, castling while in check, non-king move that doesn't address check |
| `double_check` | `king_to_attacked`, `non_king_double_check`, `castling_in_check` | + non-king moves that only address one checker |
| `illegal_king` | `king_to_attacked`, `castling_through_attacked` | King to attacked square, castling through attacked |
| `pin` | `pin_breaking` | Pinned piece moves off the pin ray |
| `promotion` | `promo_push_blocked`, `promo_capture_empty` | Push onto occupied square, diagonal capture to empty |

### General distractor types (added to all positions)

| Type | Description |
|---|---|
| `backward_pawn` | Pawn moves in wrong direction |
| `friendly_fire` | Piece "captures" own piece |
| `blocked_sliding` | Sliding piece moves through a blocker |
| `pawn_double_wrong_rank` | Double-push from non-starting rank |
| `pawn_double_push_blocked` | Double-push from starting rank with blocked intermediate square |
| `pawn_push_onto_piece` | Pawn pushes forward into occupied square |
| `pawn_diagonal_to_empty` | Pawn moves diagonally to empty square (no capture) |
| `pawn_capture_friendly` | Pawn captures own piece diagonally |
| `wrong_ep` | Pawn on EP rank tries diagonal behind adjacent enemy pawn, but EP not available |
| `castling_path_occupied` | Castling when pieces sit between king and rook |
| `wrong_geometry_knight` | Knight moves diagonally (like a bishop) |
| `wrong_geometry_bishop` | Bishop moves straight (like a rook) |
| `wrong_geometry_rook` | Rook moves diagonally (like a bishop) |
| `wrong_geometry_queen` | Queen moves in L-shape (like a knight) |
| `wrong_geometry_king` | King moves more than one square (non-castling) |

### Legal move subcategories (binary eval only)

| Subcategory | When used |
|---|---|
| `legal_move` | Normal piece move (non-check, non-special) |
| `legal_capture` | Captures an enemy piece |
| `legal_castling` | Legal castling |
| `legal_en_passant` | Legal en passant |
| `legal_promotion` | Pawn promotes |
| `legal_check` | Delivers check |
| `legal_king_escape` | King moves out of check |
| `legal_capture_checker` | Non-king captures the checking piece |
| `legal_block_check` | Non-king interposes on the check ray |

## CLI Reference

### extract_positions.py

```
python extract_positions.py \
    --pgn_path PATH       # Input PGN file
    --out_path PATH       # Output JSONL
    --max_games N         # Games to scan (default: 1400)
    --num_distractors N   # General distractors per position (default: 5)
    --num_vanilla N       # Vanilla positions to sample (default: 2000)
    --seed N              # Random seed (default: 42)
```

### generate_sft_legal_moves.py

```
python generate_sft_legal_moves.py \
    --data_path PATH      # Input JSONL from extract_positions.py
    --template_dir PATH   # Directory with reasoning template .txt files
    --out_path PATH       # Output SFT JSONL
    --num_legal N         # Legal moves per candidate set (default: 3)
    --num_illegal_gen N   # General illegals per candidate set (default: 2)
    --seed N              # Random seed (default: 42)
```

### generate_eval_binary.py

```
python generate_eval_binary.py \
    --data_path PATH      # Input JSONL from extract_positions.py
    --out_path PATH       # Output binary eval JSONL
    --default_ratio F     # Legal:illegal ratio per position (default: 1.0)
    --category_config P   # Optional JSON for per-tag overrides
    --seed N              # Random seed (default: 42)
```

Category config example:
```json
{
    "pin": {"num_positions": 500, "ratio": 2.0},
    "check": {"num_positions": 300},
    "en_passant": {"num_positions": null}
}
```

### Notebooks

```bash
jupyter notebook extract_eval_positions.ipynb   # Extraction + sanity checks
jupyter notebook visualize_reasoning.ipynb      # Visualize reasoning templates
```

`extract_eval_positions.ipynb` imports from `legal_moves.py` and `legal_move_puzzles.py`, runs extraction, and provides interactive visualization and browsing (`browse(rows, tag_filter='pin')`).

`visualize_reasoning.ipynb` loads extracted positions, samples candidates, fills reasoning templates, and displays board SVGs with color-coded arrows alongside the per-move template text and full reasoning trace. Use `browse(tag_filter='check')` to step through positions by category.

## Data Source

Positions are extracted from the [Lichess Open Database](https://database.lichess.org/) (standard rated games).

1. Download from https://database.lichess.org/standard/
2. Decompress `.zst` → `.pgn` (`zstd -d <file>.pgn.zst`)
3. Place `.pgn` files in `data/`

## Dependencies

- `python-chess` (tested with v1.9.4)
- `jupyter` / `ipython` (for the notebook only)
