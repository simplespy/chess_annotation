# Chess Commentary Generation & Evaluation Pipeline

A comprehensive evaluation framework for chess move commentary generation using LLMs with tool-augmented verification.

## Overview

This pipeline generates and evaluates natural language explanations for chess moves using:

- **Multi-provider LLM support**: OpenAI (GPT-4o, etc.), Anthropic (Claude), Qwen (via OpenAI-compatible API)
- **Tool-augmented generation**: LLMs can call chess analysis tools (Stockfish, board state queries)
- **Comprehensive judging**: Atomic claim verification, tool-based fact-checking, sanity checks
- **Multi-dimensional scoring**: Claim accuracy, recall, quality assessment, fluency, specificity

## Quick Start

```bash
# 1. Generate commentary for positions
python generate_commentary.py --input positions.jsonl --output generated.jsonl

# 2. Judge the generated commentary
python judge_commentary.py --input generated.jsonl --output judged.jsonl

# 3. Or run both in one command
python batch_evaluate.py --input positions.jsonl --output results.jsonl --n 10
```

## Architecture

```
┌─────────────────┐
│  Input Position │
│  (FEN + move)   │
└────────┬────────┘
         │
         ├──────────────────┐
         │                  │
         v                  v
┌────────────────┐  ┌──────────────┐
│   GENERATION   │  │   Optional:  │
│                │  │   Engine     │
│ - GPT/Claude/  │  │   Analysis   │
│   Qwen         │  │   (Stockfish)│
│ - Tool-calling │  └──────────────┘
└────────┬───────┘
         │
         v
┌─────────────────────────────────┐
│   Generated Commentary          │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│         JUDGING PIPELINE        │
│                                 │
│ 1. Decomposition into atoms     │
│ 2. Alternative move extraction  │
│ 3. Per-atom verification        │
│    ├─ Tool-calling (Stockfish)  │
│    └─ Sanity checks             │
│ 4. Gold atom matching (recall)  │
│ 5. Quality assessment           │
│ 6. Fluency & specificity        │
│ 7. Composite score              │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│   Evaluation Results (JSONL)    │
│                                 │
│ - Claim accuracy: X%            │
│ - Recall: Y%                    │
│ - Composite: Z                  │
│ - Detailed atom verification    │
└─────────────────────────────────┘
```

## Modules

### Core Modules

- **`llm_client.py`**: Unified LLM API client (OpenAI, Anthropic, Qwen)
- **`generation.py`**: Commentary generation functions
- **`judge.py`**: Comprehensive judging pipeline (verification, matching, scoring)
- **`scoring.py`**: Fluency, specificity, and composite scoring
- **`chess_tools.py`**: Chess analysis tools for LLMs (Stockfish, board queries)
- **`eval_utils.py`**: Utility functions (engine analysis, display)

### CLI Scripts

- **`generate_commentary.py`**: Generate commentary for positions
- **`judge_commentary.py`**: Judge existing commentary
- **`batch_evaluate.py`**: Full generate + judge pipeline
- **`analyze_judge_results.py`**: Analyze judged results (aggregate metrics, breakdowns)

## Installation

```bash
# Python dependencies
pip install chess numpy anthropic openai python-dotenv

# Stockfish
brew install stockfish  # macOS
# or
apt-get install stockfish  # Linux
```

Set environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export STOCKFISH_PATH="/path/to/stockfish"  # if not in default location
```

## Usage Examples

### 1. Generate Commentary

```bash
# Basic generation (tool-augmented, no engine)
python generate_commentary.py --input positions.jsonl --output generated.jsonl

# With engine analysis
python generate_commentary.py --input positions.jsonl --output generated.jsonl --use-engine

# Use Claude
python generate_commentary.py --input positions.jsonl --output generated.jsonl \
    --provider anthropic --model claude-sonnet-4-5-20250929

# Use Qwen (local vLLM server)
python generate_commentary.py --input positions.jsonl --output generated.jsonl \
    --provider qwen --model Qwen/Qwen3-32B --base-url http://localhost:8000/v1
```

### 2. Judge Commentary

```bash
# Judge generated commentary
python judge_commentary.py --input generated.jsonl --output judged.jsonl

# Use different model for judging
python judge_commentary.py --input generated.jsonl --output judged.jsonl \
    --provider anthropic --model claude-sonnet-4-5-20250929
```

### 3. Batch Evaluation

```bash
# Evaluate 10 random positions
python batch_evaluate.py --input positions.jsonl --output results.jsonl --n 10

# Evaluate specific indices
python batch_evaluate.py --input positions.jsonl --output results.jsonl --indices 0 5 10 15

# Use different models for generation vs judging
python batch_evaluate.py --input positions.jsonl --output results.jsonl \
    --gen-model gpt-4o --model claude-sonnet-4-5-20250929 --n 20
```

### 4. Analyze Results

```bash
# Analyze single file
python analyze_judge_results.py outputs_judge/judged_gpt4o_1.jsonl

# Save report to file
python analyze_judge_results.py outputs_judge/judged_gpt4o_1.jsonl > report.txt

# Compare multiple files
python analyze_judge_results.py --compare outputs_judge/*.jsonl

# Break down by quality level
python analyze_judge_results.py --by-quality outputs_judge/judged_gpt4o_1.jsonl

# Break down by wp_loss bins
python analyze_judge_results.py --by-wp-loss outputs_judge/judged_gpt4o_1.jsonl
```

## Data Format

### Input (positions.jsonl)

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "move_uci": "e7e5",
  "move_san": "e5",
  "wp_loss": 0.0,
  "annotation": "Black responds symmetrically...",
  "extracted": {
    "reasoning": [
      "e5 controls the center",
      "e5 opens lines for the bishop and queen"
    ]
  }
}
```

### Output (results.jsonl)

```json
{
  "idx": 0,
  "fen": "...",
  "generated_text": "e5 is a natural response...",
  "claim_accuracy": 0.85,
  "recall": 0.75,
  "quality_correct": true,
  "fluency": 4.2,
  "specificity": 3.8,
  "composite": 0.782,
  "n_atoms": 8,
  "n_verified": 6,
  ...
}
```

## Evaluation Metrics

### Claim Accuracy (0-1)
Percentage of claims where all verifiable atoms are verified.

### Recall (0-1)
Percentage of gold standard atoms covered by generated atoms (semantic matching).

### Quality Assessment
Does the explanation correctly assess move quality?
- Ground truth from wp_loss: good (<5%), inaccuracy (5-20%), mistake (20-30%), blunder (>30%)
- Generated assessment from quality atoms

### Fluency & Specificity (1-5)
- **Fluency**: Grammar, clarity, coherence
- **Specificity**: Concrete squares/pieces vs. generic claims

### Composite Score (0-1)
Weighted combination:
- 30% Claim Accuracy
- 25% Recall
- 20% Quality Correctness
- 15% Specificity
- 10% Fluency

## Chess Tools

The pipeline provides 25+ tools for LLM fact-checking:

### Board State
- `get_legal_moves`, `get_piece_at`, `get_squares`, `get_material`

### Attacks & Defense  
- `get_attacks`, `get_attackers`, `count_attackers_defenders`, `is_check`

### Tactics
- `is_pinned`, `check_ray_alignment`, `check_threat`

### Evaluation
- `get_engine_eval`, `get_top_moves`, `eval_move`, `compare_moves`

### Variations
- `try_variation`, `make_move`

## Provider Support

### OpenAI
- Models: gpt-4o, gpt-4.1, gpt-5 (when available)
- Default provider, fast and reliable

### Anthropic
- Models: claude-sonnet-4-5, claude-opus-4-6, claude-haiku-4-5
- Excellent reasoning quality

### Qwen (via vLLM)
Deploy locally:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --port 8000
```
Then use with `--provider qwen --base-url http://localhost:8000/v1`

## Advanced Usage

### Programmatic API

```python
from generation import generate_commentary_raw
from judge import judge_explanation_improved

# Generate
entry = {'fen': '...', 'move_uci': 'e2e4', 'wp_loss': 0.0}
commentary, tool_log = generate_commentary_raw(
    entry, provider="openai", model="gpt-4o"
)

# Judge
gold_atoms = ["e4 controls the center", "e4 opens lines"]
results = judge_explanation_improved(
    entry, commentary, gold_atoms=gold_atoms,
    provider="openai", model="gpt-4o"
)

print(f"Composite: {results['composite']:.3f}")
print(f"Claim accuracy: {results['claim_accuracy']:.0%}")
```

### Custom Provider/Model

```python
# Use custom OpenAI-compatible endpoint
commentary, log = generate_commentary_raw(
    entry,
    provider="qwen",
    model="my-model",
    base_url="https://my-api.com/v1",
    api_key="my-key"
)
```

## Performance Tips

1. **Use Haiku for large-scale evaluation** (fast, cheap)
2. **Cache engine analysis** (pre-compute and store)
3. **Parallel processing** (split with `--indices`)
4. **Local Qwen with vLLM** (no API costs, high throughput)

## Troubleshooting

**"Stockfish not found"**
- Install: `brew install stockfish` or `apt-get install stockfish`
- Or set `STOCKFISH_PATH` environment variable

**"API key not found"**
- Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
- Or pass `--api-key` to CLI scripts

**"Tool call failed"**
- Check Stockfish is accessible
- Verify FEN strings are valid
- Check `chess_tools.py` implementations

## Related Documentation

- Main extraction pipeline: see `README.md`
- Notebook documentation: see individual notebooks

## License

[Your license]
