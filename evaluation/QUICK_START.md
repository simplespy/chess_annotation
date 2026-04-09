# Quick Start Guide

## File Structure

```
evaluation/
├── README.md                    # Main extraction pipeline docs
├── README_EVALUATION.md         # This evaluation pipeline docs ⭐
├── QUICK_START.md              # This file
│
├── Core Modules
│   ├── llm_client.py           # Unified LLM API (OpenAI/Anthropic/Qwen)
│   ├── generation.py           # Commentary generation functions
│   ├── judge.py                # Judging pipeline (verification, scoring)
│   ├── scoring.py              # Fluency, specificity, composite scoring
│   ├── chess_tools.py          # Chess analysis tools for LLMs
│   ├── eval_utils.py           # Utility functions (engine, display)
│   └── config.py               # Configuration constants
│
├── CLI Scripts
│   ├── generate_commentary.py  # Generate commentary 🚀
│   ├── judge_commentary.py     # Judge commentary 🔍
│   └── batch_evaluate.py       # Full pipeline (gen + judge) ⚡
│
└── Other Files
    ├── extract_all.py          # Original extraction pipeline
    ├── extracted_functions.py  # Backup of extracted code
    └── *.ipynb                 # Notebooks for interactive work
```

## Essential Commands

### 1. Generate Commentary

```bash
# Basic (no engine, with tools)
./generate_commentary.py --input positions.jsonl --output generated.jsonl

# With engine analysis
./generate_commentary.py --input positions.jsonl --output generated.jsonl --use-engine

# With Claude
./generate_commentary.py --input positions.jsonl --output generated.jsonl \
    --provider anthropic --model claude-sonnet-4-5-20250929
```

### 2. Judge Commentary

```bash
# Basic judging
./judge_commentary.py --input generated.jsonl --output judged.jsonl

# With Claude for judging
./judge_commentary.py --input generated.jsonl --output judged.jsonl \
    --provider anthropic --model claude-sonnet-4-5-20250929
```

### 3. Full Pipeline (Generate + Judge)

```bash
# Evaluate 10 random positions
./batch_evaluate.py --input positions.jsonl --output results.jsonl --n 10

# Evaluate specific positions
./batch_evaluate.py --input positions.jsonl --output results.jsonl --indices 0 5 10

# Use different models
./batch_evaluate.py --input positions.jsonl --output results.jsonl \
    --gen-model gpt-4o --model claude-sonnet-4-5-20250929 --n 20
```

## Required Input Format

Your `positions.jsonl` should have:

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "move_uci": "e7e5",
  "move_san": "e5",
  "wp_loss": 0.0,
  "extracted": {
    "reasoning": [
      "e5 controls the center",
      "e5 opens lines for the bishop and queen"
    ]
  }
}
```

**Required fields**: `fen`, `move_uci`
**Recommended**: `move_san`, `wp_loss`, `extracted.reasoning` (gold atoms)

## Output Metrics

Each evaluation produces:

- **claim_accuracy** (0-1): Percentage of correct claims
- **recall** (0-1): Coverage of gold atoms
- **quality_correct** (bool): Correct quality assessment
- **fluency** (1-5): Writing quality
- **specificity** (1-5): Position-specificity
- **composite** (0-1): Overall weighted score

## Common Options

### Provider Options
- `--provider openai` (default): GPT-4o, GPT-4.1, etc.
- `--provider anthropic`: Claude Sonnet 4.5, Opus 4.6, Haiku 4.5
- `--provider qwen`: Qwen models via OpenAI-compatible API

### Model Examples
- OpenAI: `gpt-4o`, `gpt-4.1-mini`
- Anthropic: `claude-sonnet-4-5-20250929`, `claude-opus-4-6`, `claude-haiku-4-5-20251001`
- Qwen: `Qwen/Qwen3-32B`, `Qwen/Qwen3-8B`

### Other Flags
- `--use-engine`: Include Stockfish analysis in generation prompts
- `--no-tools`: Disable tool calling during generation
- `--ascii`: Include ASCII board in prompts
- `--base-url URL`: Custom API endpoint (for Qwen/vLLM)
- `--api-key KEY`: Custom API key
- `--quiet`: Reduce output verbosity

## Environment Variables

```bash
# Required (choose your provider)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional
export STOCKFISH_PATH="/path/to/stockfish"  # default: /opt/homebrew/bin/stockfish
```

## Quick Test

```bash
# 1. Generate commentary for 3 positions
./generate_commentary.py \
    --input evaluation/data/logical_chess_atomize/test_accepted.jsonl \
    --output test_generated.jsonl \
    --n 3

# 2. Judge the results
./judge_commentary.py \
    --input test_generated.jsonl \
    --output test_judged.jsonl

# 3. Or do both at once
./batch_evaluate.py \
    --input evaluation/data/logical_chess_atomize/test_accepted.jsonl \
    --output test_results.jsonl \
    --n 3
```

## Programmatic Usage

```python
from generation import generate_commentary_raw
from judge import judge_explanation_improved

# Generate
entry = {'fen': 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
         'move_uci': 'f1c4', 'wp_loss': 0.64}
text, tool_log = generate_commentary_raw(entry, provider="openai", model="gpt-4o")

# Judge
gold_atoms = ["Bc4 develops the bishop", "Bc4 attacks f7"]
results = judge_explanation_improved(entry, text, gold_atoms=gold_atoms,
                                      provider="openai", model="gpt-4o")

print(f"Composite score: {results['composite']:.3f}")
```

## Troubleshooting

### Import Errors
```bash
pip install chess numpy anthropic openai python-dotenv
```

### "Stockfish not found"
```bash
# macOS
brew install stockfish

# Linux
apt-get install stockfish

# Or set path manually
export STOCKFISH_PATH="/path/to/stockfish"
```

### "No module named 'chess_tools'"
Make sure you're running from the evaluation/ directory:
```bash
cd evaluation/
./generate_commentary.py --help
```

## Next Steps

1. **Read** `README_EVALUATION.md` for comprehensive documentation
2. **Try** the Quick Test above
3. **Explore** notebooks for interactive experimentation
4. **Customize** prompts in `generation.py` and `judge.py`

## Key Concepts

- **Atoms**: Single verifiable facts (e.g., "Bc4 attacks f7")
- **Claims**: Logical assertions containing multiple atoms
- **Tool-calling**: LLMs use chess analysis tools to verify facts
- **Sanity checks**: Programmatic validation to catch LLM errors
- **Alternative moves**: Extracting and verifying claims about alternatives

For detailed explanation of metrics, pipeline steps, and tool catalog, see `README_EVALUATION.md`.
