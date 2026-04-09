# Evaluation

Extract structured atoms from chess commentary and evaluate LLM-generated explanations.

## Pipeline

```
logical_chess.jsonl (1832 positions)
        │
        ▼
  extract_all.py prepare        ← Stockfish analysis + batch request
        │
        ▼
  extract_all.py submit         ← OpenAI Batch API (50% off)
        │
        ▼
  extract_all.py collect        ← download results
        │
        ▼
  extract_all.py process        ← postprocess → included.jsonl / excluded.jsonl
        │
        ▼
  extract_all.py filter-prepare ← build filter batch request
        │
        ▼
  extract_all.py submit         ← OpenAI Batch API (reuse)
        │
        ▼
  extract_all.py collect        ← download results (reuse)
        │
        ▼
  extract_all.py filter-process ← apply filter → included_filtered.jsonl / excluded_filtered.jsonl
        │
        ▼
  filter_atoms.ipynb            ← interactive review of filter results
        │
        ▼
  train.jsonl + test_unfiltered.jsonl  ← game-wise split (no leakage)
        │
        ▼
  review_gold.ipynb             ← human review → test_accepted / test_rejected / test_again
```

## Scripts

### `extract_all.py`

All stages of the pipeline in one script. Subcommands:

| Subcommand | Description |
|------------|-------------|
| `prepare` | Run Stockfish analysis, build batch request + metadata |
| `submit` | Upload batch file to OpenAI Batch API |
| `collect` | Poll + download batch results |
| `process` | Join batch output with metadata, postprocess filter |
| `sync` | Run extraction synchronously (no batch, direct API) |
| `filter-prepare` | Build batch request for filter pass |
| `filter-process` | Join filter batch output, apply filter logic |
| `filter` | Run filter synchronously (no batch) |

#### Step 1–4: Extract atoms

```bash
# 1. Run Stockfish, build batch request
python evaluation/extract_all.py prepare \
    --input evaluation/data/logical_chess.jsonl \
    --batch-file evaluation/data/logical_chess_atomize/batch_input.jsonl \
    --meta-file evaluation/data/logical_chess_atomize/batch_meta.jsonl

# 2. Submit to OpenAI Batch API (50% off)
python evaluation/extract_all.py submit \
    --batch-file evaluation/data/logical_chess_atomize/batch_input.jsonl

# 3. Poll + download results
python evaluation/extract_all.py collect \
    --batch-id <BATCH_ID> \
    --output evaluation/data/logical_chess_atomize/batch_output.jsonl --poll

# 4. Postprocess → included/excluded
python evaluation/extract_all.py process \
    --meta-file evaluation/data/logical_chess_atomize/batch_meta.jsonl \
    --batch-output evaluation/data/logical_chess_atomize/batch_output.jsonl \
    --out-included evaluation/data/logical_chess_atomize/included.jsonl \
    --out-excluded evaluation/data/logical_chess_atomize/excluded.jsonl
```

`sync` subcommand available for small runs without the batch API.

#### Step 5: Filter atoms

LLM second pass to clean up extracted atoms. Actions:
- **keep** — concrete verifiable fact
- **contextualize** — add referential context (which move/piece/square) so atom is self-contained
- **move_to_alternative** — detailed analysis of an alternative move (with optional `kept_brief` for split)
- **remove** — generic labels only

Also reviews existing alternative atoms (keep/remove/paraphrase) and deduplicates.

```bash
# 5a. Via Batch API (recommended — 50% off)
python evaluation/extract_all.py filter-prepare \
    --input evaluation/data/logical_chess_atomize/included.jsonl \
    --batch-file evaluation/data/logical_chess_atomize/filter_batch_input.jsonl

python evaluation/extract_all.py submit \
    --batch-file evaluation/data/logical_chess_atomize/filter_batch_input.jsonl

python evaluation/extract_all.py collect \
    --batch-id <BATCH_ID> \
    --output evaluation/data/logical_chess_atomize/filter_batch_output.jsonl --poll

python evaluation/extract_all.py filter-process \
    --input evaluation/data/logical_chess_atomize/included.jsonl \
    --batch-output evaluation/data/logical_chess_atomize/filter_batch_output.jsonl \
    --out-included evaluation/data/logical_chess_atomize/included_filtered.jsonl \
    --out-excluded evaluation/data/logical_chess_atomize/excluded_filtered.jsonl

# 5b. Or synchronously (no batch, direct API — has resume support)
python evaluation/extract_all.py filter \
    --input evaluation/data/logical_chess_atomize/included.jsonl \
    --out-included evaluation/data/logical_chess_atomize/included_filtered.jsonl \
    --out-excluded evaluation/data/logical_chess_atomize/excluded_filtered.jsonl
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `extract_atoms.ipynb` | Interactive extraction pipeline with tool-augmented LLM. Contextual atom system prompt, hardcoded demos, generation loop. |
| `eval_tools.ipynb` | LLM-as-a-Judge: generate NL commentary, then evaluate via atom-level decomposition → verification → matching. |
| `compare_contextual.ipynb` | Comparison notebook — contextual atom style (atoms carry full move-sequence prefix). |
| `compare_flat.ipynb` | Comparison notebook — flat atom style (conclusions only, move sequences in `variation` field). |
| `filter_atoms.ipynb` | Interactive filter notebook — same logic as `extract_all.py filter`, with visual review. |

## Data

### `data/logical_chess.jsonl`

Source dataset. 1832 annotated positions from *Logical Chess: Move by Move*.

```json
{"game_id": "...", "fen": "...", "move_uci": "e2e4", "move_san": "e4",
 "annotation": "This is an excellent opening move...",
 "metadata": {"White": "Scheve", "Black": "Teichmann", ...}}
```

### `data/logical_chess_atomize/`

| File | Description |
|------|-------------|
| `batch_input.jsonl` | OpenAI Batch API requests (extraction) |
| `batch_meta.jsonl` | Metadata (engine lines, wp_loss) keyed by `custom_id` |
| `batch_output.jsonl` | Raw extraction LLM responses |
| `included.jsonl` | 1302 positions with extracted atoms (32 games) |
| `excluded.jsonl` | 530 positions excluded (too minimal, conflicts, etc.) |
| `filter_batch_input.jsonl` | OpenAI Batch API requests (filter) |
| `filter_batch_output.jsonl` | Raw filter LLM responses |
| `included_filtered.jsonl` | Positions after filter pass (atoms cleaned up) |
| `excluded_filtered.jsonl` | Positions excluded by filter (0 reasoning atoms) |
| `train.jsonl` | Training split (seed=99, game-wise) |
| `test_unfiltered.jsonl` | Test split (seed=99, game-wise) |
| `review_gold.ipynb` | Human review UI for test set (randomized order, seed=42) |
| `filter_atoms.ipynb` | Interactive filter review notebook |
| `test_accepted.jsonl` | Accepted after review |
| `test_rejected.jsonl` | Rejected after review |
| `test_again.jsonl` | Flagged for re-review |

### Extraction output schema

```json
{
  "position_number": 5,
  "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
  "move_uci": "f1c4", "move_san": "Bc4",
  "annotation": "The bishop seizes a valuable diagonal...",
  "wp_loss": 0.64, "quality": "good",
  "game": "Scheve – Teichmann",
  "engine_lines": [{"move_san": "Bb5", "eval": "+0.32", "cp": 32, ...}, ...],
  "extracted": {
    "include": true,
    "quality": "good",
    "reasoning": [
      "Bc4 develops White's king bishop and clears the way for early castling.",
      "Bc4 places the bishop on the a2-g8 diagonal through the center.",
      "Bc4 attacks the f7-pawn.",
      "The f7-pawn is defended only by the king, making it a vulnerable target."
    ],
    "book_commentary": "..."
  },
  "model": "gpt-5.4"
}
```

## Key concepts

- **Contextual atoms**: each reasoning atom is self-contained with full move-sequence prefix ("After Nxd7 Nh5, White can capture on g6") so it can be verified independently.
- **wp_loss**: win-percentage loss from Stockfish eval. `Win% = 50 + 50 * tanh(0.00368208 * cp / 2)`.
- **Quality**: good (≤10%), inaccuracy (>10%), mistake (>20%), blunder (>30%).
- **Postprocess filter**: catches quality/engine conflicts, missing reasoning, empty alternatives.
- **Filter pass**: LLM second pass — contextualize (add referential context), move to alternative (with dedup), remove generic labels. Reviews existing alternatives too.
