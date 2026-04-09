#!/usr/bin/env python3
"""
Atom extraction pipeline for chesspublishing commentary.

Modes:
  estimate    Estimate token usage and cost for N positions.
  prepare     Sample N positions, build OpenAI batch JSONL.
  submit      Submit batch file to OpenAI Batch API.
  collect     Poll for batch completion, download results.
  process     Parse batch results, report include/exclude stats.
  sync        Run extraction synchronously (works with any OpenAI-compatible endpoint).

Examples:
  # Estimate cost for 1000 positions
  python run_extraction.py estimate -n 1000

  # OpenAI batch workflow
  python run_extraction.py prepare -n 1000 --seed 42
  python run_extraction.py submit
  python run_extraction.py collect --batch-id batch_xxx
  python run_extraction.py process

  # Local Qwen3 (sync)
  python run_extraction.py sync -n 1000 --model Qwen/Qwen3-32B \
      --base-url http://127.0.0.1:8000/v1 --api-key not-needed

  # Claude via Anthropic-compatible endpoint
  python run_extraction.py sync -n 1000 --model claude-sonnet-4-5-20250929 \
      --base-url https://api.anthropic.com/v1
"""

import argparse
import json
import os
import random
import sys
import time

import chess
import openai
import tiktoken

# ── Paths ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'data', 'chesspublishinga.jsonl')
BATCH_INPUT = os.path.join(SCRIPT_DIR, 'data', 'batch_input.jsonl')
BATCH_META = os.path.join(SCRIPT_DIR, 'data', 'batch_meta.jsonl')
BATCH_OUTPUT = os.path.join(SCRIPT_DIR, 'data', 'batch_output.jsonl')
RESULTS_INCLUDED = os.path.join(SCRIPT_DIR, 'data', 'extraction_included.jsonl')
RESULTS_EXCLUDED = os.path.join(SCRIPT_DIR, 'data', 'extraction_excluded.jsonl')

DEFAULT_MODEL = 'gpt-5.4'

# ── System prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are extracting structured move explanations from chess commentary.

You will receive: a FEN, the move played, previous moves for context,
a context_fen, and the annotator's commentary.

For VARIATION entries (sidelines), you will also receive:
- mainline_move: the move the game actually played (this entry discusses
  an alternative)
- parent_comment: framing context from the annotator on the parent move

For MAINLINE entries with sidelines, you will also receive:
- alternatives: structured list of alternative moves from the PGN, each
  with move_san, annotation (the annotator's comment), and line
  (continuation moves). Use these to populate the "alternative" field
  in your output — extract reasoning atoms from the annotator's comments
  on each alternative.

Your job is to EXTRACT and STRUCTURE the commentary into a JSON object.

Output ONLY valid JSON. No preamble, no markdown fences, no explanation.

When include=true, output:
{
  "include": true,
  "fen": "<FEN of the position>",
  "move_san": "<the move>",
  "move_uci": "<UCI of that move>",
  "book_commentary": "<the full original commentary>",
  "reasoning": ["atomic fact 1", "atomic fact 2"],
  "variation": "19...Bxf6 gxf6 20.Rg4+ ...",
  "alternative": {
    "move": "Qxh7+",
    "reasoning": ["why it's worse/better"],
    "variation": "18.Qxh7+ Kf8 ..."
  }
}

When include=false, output ONLY:
{
  "include": false,
  "exclude_reason": "short explanation",
  "book_commentary": "<the full original commentary>"
}

MOVE ATTRIBUTION:
The commentary may explain a PREVIOUS move, not the tagged move.
Use the prev_moves + context_fen to determine which move the commentary
is actually about. The output fen/move_san/move_uci MUST correspond to
the position and move the commentary explains.

ATOM RULES:

1. Each reasoning atom explains something about the CURRENT MOVE from
   the CURRENT FEN. A model will be given only the FEN + move and must
   generate these atoms — so they must make sense in that context alone.

2. STRICT EXTRACTION. Extract ONLY what the commentary explicitly says.
   Do NOT invent, infer, or enrich with your own chess analysis.
   If the commentary says "well timed" but doesn't explain why, that
   is NOT enough — exclude it. The atoms must come from the text.

3. Atoms must be DIRECT CLAIMS about the position, not meta-descriptions
   of what the commentary says.
   BAD:  "Be5 is presented as a standard idea in this structure."
   BAD:  "The move helps White exert slight pressure."
   GOOD: "Be5 gives White the bishop pair, creating slight pressure."
   GOOD: "d5 prevents White from playing e4."

4. Short illustrative lines in atoms are fine to show a threat or idea
   (e.g. "Bxd4 threatens Bxf6 gxf6, Rg4+ with a mating attack").
   Long multi-move variations (5+ moves) belong in the "variation" field,
   NOT inside atoms.

5. Do NOT prefix atoms with preceding moves from earlier in the game.
   The atom must be about the position in the FEN, not how we got there.

6. Group logically connected setup+consequence into a SINGLE atom.

7. Name squares, pieces, and diagonals concretely.

8. DROP VAGUE LABELS. Every atom must state a SPECIFIC chess fact: a
   concrete square, piece, line, file, diagonal, threat, or plan. Drop
   any phrase that is merely an evaluative label or vague positional
   claim — even when the entry has other good atoms. Examples to drop:
   "The move is committal", "A slightly passive choice",
   "This keeps the tension", "A practical decision",
   "White's bishops aren't doing much", "Black has a sound game".
   These are assessments, not explanations. An atom must say WHAT
   square/piece/threat makes the move work, not just that the resulting
   position is good or bad.

9. INCLUDE/EXCLUDE. Set "include": false when:
   - The commentary contains NO atom-worthy content after applying
     rule 8 — i.e. everything is labels/assessments with no concrete
     facts (square, piece, threat, plan)
   - Historical anecdotes, biographies, no move-specific analysis
   - Generic chess philosophy not applied to this position
   - Game references with no analysis (e.g. "Kasparov-Karpov, 1985")
   Include when there is at least one CONCRETE fact — a specific square,
   piece, threat, or plan. "(stopping e4)" is enough.
   "a sound game" / "bishops aren't doing much" is NOT enough.

10. NEVER include generic philosophical statements as atoms.

11. For sidelines: the alternative field captures the MAINLINE move that
    the annotator is comparing against, if they discuss it.

12. For mainline entries with "alternatives" provided: use the annotator's
    comments and lines from the alternatives to populate the "alternative"
    field. Each alternative may have annotation text and a continuation
    line — extract reasoning atoms from these just as you would from the
    main commentary. If there are multiple alternatives, output
    "alternative" as a list of objects.
"""

# ── Few-shot demos ───────────────────────────────────────────────────────

DEMO_DATA = [
    {
        'fen': 'rn2k1nr/ppp2ppp/8/q7/1b1N2b1/2N5/PPPBBPPP/R2QK2R b KQkq - 0 8',
        'move_uci': 'a5e5', 'move_san': 'Qe5',
        'annotation': (
            "Black's response pins the e2-bishop and attacks the unprotected "
            "d4-knight. Black rejects 8... Bxe2 as the recapture by 9 Qxe2+ "
            "gains another tempo for White."),
        'prev_moves': '', 'context_fen': '', 'is_mainline': True,
        'mainline_move': None, 'parent_comment': None,
    },
    {
        'fen': 'r4rk1/pp1q1ppp/4pb2/8/2PpR3/1P2Q3/PB3PPP/5RK1 w - - 0 19',
        'move_uci': 'b2d4', 'move_san': 'Bxd4',
        'annotation': (
            "White regains the pawn, and his bishop now attacks in two directions. "
            "On the one hand, it threatens to take the a-pawn, on the other it aims "
            "at checkmate by... Bxf6 gxf6 21 Rg4+ Kh8 22 Qh6 Rg8 23 Qxf6+ and "
            "mate next move."),
        'prev_moves': '', 'context_fen': '', 'is_mainline': True,
        'mainline_move': None, 'parent_comment': None,
    },
    {
        'fen': 'r2qk2r/p1p1npp1/1pn1b2p/3pP3/3P1B2/2PB1N2/P1PQ2PP/1R3RK1 b kq - 1 12',
        'move_uci': 'e8g8', 'move_san': 'O-O',
        'annotation': (
            'Walking right into the teeth of the storm!\n'
            'Before making a move that suggests itself so readily, Black might have '
            'asked himself, "How can I exploit White\'s one weakness, the doubled '
            'pawns on the c-file?"\n'
            'He might then have hit upon 12... Na5, with the object of swinging the '
            'knight to c4. There it blockades the doubled pawn, interferes with the '
            'free movement of White\'s pieces, and in general sticks like a bone in '
            'the throat. White could capture the knight, but then he parts with one '
            'of his valuable bishops, and as a result of the exchange his pawn '
            'position would be inferior to Black\'s. Finally, Black could then anchor '
            'one of his pieces to great effect on d5, a square from which it could '
            'never be evicted by pawns.'),
        'prev_moves': '', 'context_fen': '', 'is_mainline': True,
        'mainline_move': None, 'parent_comment': None,
    },
]

DEMO_RESPONSES = [
    {
        'include': True,
        'fen': 'rn2k1nr/ppp2ppp/8/q7/1b1N2b1/2N5/PPPBBPPP/R2QK2R b KQkq - 0 8',
        'move_san': 'Qe5', 'move_uci': 'a5e5',
        'reasoning': [
            'Qe5 pins the bishop on e2 to the king on e1, since the queen on e5 attacks along the e-file.',
            'Qe5 simultaneously attacks the unprotected knight on d4.',
            'Black rejects Bxe2 because after Qxe2+, White recaptures with check, gaining a tempo.',
        ],
    },
    {
        'include': True,
        'fen': 'r4rk1/pp1q1ppp/4pb2/8/2PpR3/1P2Q3/PB3PPP/5RK1 w - - 0 19',
        'move_san': 'Bxd4', 'move_uci': 'b2d4',
        'reasoning': [
            'Bxd4 recaptures the pawn on d4.',
            'The bishop on d4 threatens to capture the undefended a7-pawn.',
            'The bishop on d4 also enables a mating threat: Bxf6 gxf6, Rg4+ Kh8, Qh6 with Qxf6+ and mate to follow.',
        ],
        'variation': 'Bxf6 gxf6 Rg4+ Kh8 Qh6 Rg8 Qxf6+',
    },
    {
        'include': True,
        'fen': 'r2qk2r/p1p1npp1/1pn1b2p/3pP3/3P1B2/2PB1N2/P1PQ2PP/1R3RK1 b kq - 1 12',
        'move_san': 'O-O', 'move_uci': 'e8g8',
        'reasoning': [
            "O-O castles into White's prepared kingside attack.",
            "Black misses the opportunity to exploit White's doubled c-pawns.",
        ],
        'alternative': {
            'move': 'Na5',
            'reasoning': [
                'Na5 reroutes the knight toward c4 to blockade the doubled c-pawn.',
                "A knight on c4 would interfere with the coordination of White's pieces.",
                'If White captures Bxc4 dxc4, the resulting pawn structure favors Black.',
                'After Na5, Black can later anchor a piece on d5, a square no pawn can attack.',
            ],
        },
    },
]

# ── Helpers ──────────────────────────────────────────────────────────────

def load_entries(path=DATA_FILE):
    with open(path) as f:
        entries = [json.loads(line) for line in f]
    return [e for e in entries if e.get('move_uci')]


def build_user_prompt(entry):
    board = chess.Board(entry['fen'])
    move_san = entry.get('move_san') or board.san(chess.Move.from_uci(entry['move_uci']))
    turn = 'White' if board.turn == chess.WHITE else 'Black'
    prompt = f"FEN: {entry['fen']}\nMove played: {move_san} ({turn})\n"
    if entry.get('prev_moves'):
        prompt += f"Previous moves: {entry['prev_moves']}\n"
    if entry.get('context_fen'):
        prompt += f"Context FEN (start of prev_moves): {entry['context_fen']}\n"
    if entry.get('mainline_move'):
        prompt += f"Mainline move: {entry['mainline_move']} (this entry discusses {move_san} as an alternative)\n"
    if entry.get('parent_comment'):
        prompt += f"Parent context: \"{entry['parent_comment']}\"\n"
    if entry.get('alternatives'):
        prompt += "\nAlternative variations from the PGN:\n"
        for alt in entry['alternatives']:
            prompt += f"  - {alt['move_san']}"
            if alt.get('annotation'):
                prompt += f": {alt['annotation']}"
            if alt.get('line'):
                prompt += f" (line: {alt['line']})"
            prompt += "\n"
    prompt += f'\nCommentary: \"{entry["annotation"]}\"\n'
    if entry.get('line'):
        prompt += f"Continuation line: {entry['line']}\n"
    return prompt


def build_demo_messages():
    demos = []
    for data, resp in zip(DEMO_DATA, DEMO_RESPONSES):
        r = {**resp, 'book_commentary': data['annotation']}
        demos.append({'role': 'user', 'content': build_user_prompt(data)})
        demos.append({'role': 'assistant', 'content': json.dumps(r, indent=2)})
    return demos


DEMO_MESSAGES = build_demo_messages()


def build_messages(entry):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    messages.extend(DEMO_MESSAGES)
    messages.append({'role': 'user', 'content': build_user_prompt(entry)})
    return messages


def parse_json_output(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"include": False, "exclude_reason": "JSON parse error", "raw": text}


def postprocess_filter(parsed):
    if parsed.get('include', True) and not parsed.get('reasoning'):
        return False, "no reasoning atoms extracted"
    alt = parsed.get('alternative')
    if alt:
        alts = alt if isinstance(alt, list) else [alt]
        if any(not a.get('reasoning') for a in alts):
            return False, "alternative has no reasoning"
    return parsed.get('include', True), parsed.get('exclude_reason')


def sample_entries(entries, n, seed):
    rng = random.Random(seed)
    indices = rng.sample(range(len(entries)), min(n, len(entries)))
    return [entries[i] for i in sorted(indices)]


# ── Token estimation ─────────────────────────────────────────────────────

def count_tokens(messages, model='gpt-5.4'):
    """Count tokens for a list of messages using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding('o200k_base')

    # OpenAI chat format overhead: 3 tokens per message + role tokens
    total = 0
    for m in messages:
        total += 4  # <|im_start|>{role}\n ... <|im_end|>\n
        total += len(enc.encode(m['content']))
    total += 2  # assistant priming
    return total


# ── Commands ─────────────────────────────────────────────────────────────

def cmd_estimate(args):
    """Estimate token usage and cost."""
    entries = load_entries()
    samples = sample_entries(entries, args.n, args.seed)

    # Count input tokens for each request
    input_tokens = []
    for entry in samples:
        msgs = build_messages(entry)
        input_tokens.append(count_tokens(msgs))

    # The prefix (system + demos) is the same for every request → cacheable
    prefix_msgs = [{'role': 'system', 'content': SYSTEM_PROMPT}] + DEMO_MESSAGES
    prefix_tokens = count_tokens(prefix_msgs)

    avg_input = sum(input_tokens) / len(input_tokens)
    total_input = sum(input_tokens)

    # Estimate output: included entries ~250 tokens, excluded ~40 tokens.
    # Assume 60% included based on prior runs.
    est_included_frac = 0.60
    avg_output = est_included_frac * 250 + (1 - est_included_frac) * 40
    total_output = int(avg_output * len(samples))

    # Cacheable portion: prefix is identical for every request
    cacheable_input = prefix_tokens * len(samples)
    unique_input = total_input - cacheable_input

    print(f"{'='*60}")
    print(f"TOKEN ESTIMATE — Step 1: Atom Extraction ({len(samples)} positions)")
    print(f"{'='*60}")
    print(f"  Prefix (system + demos):  {prefix_tokens:,} tokens (same every request → cacheable)")
    print(f"  Avg user prompt:          {avg_input - prefix_tokens:.0f} tokens")
    print(f"  Avg total input:          {avg_input:.0f} tokens")
    print(f"")
    print(f"  Total input tokens:       {total_input:,}")
    print(f"    Cacheable portion:      {cacheable_input:,} ({cacheable_input/total_input*100:.0f}%)")
    print(f"    Unique portion:         {unique_input:,} ({unique_input/total_input*100:.0f}%)")
    print(f"  Est. output tokens:       {total_output:,} (assuming {est_included_frac*100:.0f}% included)")
    print()

    # Step 2: Filter (only on included entries)
    # Filter prompt is ~1100 tokens, user context ~400 tokens per entry
    filter_prompt_tokens = 1100
    filter_user_tokens = 400
    filter_output_tokens = 200
    n_included = int(len(samples) * est_included_frac)
    filter_input_total = n_included * (filter_prompt_tokens + filter_user_tokens)
    filter_cacheable = n_included * filter_prompt_tokens
    filter_output_total = n_included * filter_output_tokens

    print(f"{'='*60}")
    print(f"TOKEN ESTIMATE — Step 2: Filter ({n_included} included positions)")
    print(f"{'='*60}")
    print(f"  Filter prompt:            {filter_prompt_tokens:,} tokens (cacheable)")
    print(f"  Avg user context:         {filter_user_tokens} tokens")
    print(f"  Avg output:               {filter_output_tokens} tokens")
    print(f"")
    print(f"  Total input tokens:       {filter_input_total:,}")
    print(f"    Cacheable portion:      {filter_cacheable:,}")
    print(f"  Total output tokens:      {filter_output_total:,}")
    print()

    # Combined
    combined_input = total_input + filter_input_total
    combined_cacheable = cacheable_input + filter_cacheable
    combined_unique = combined_input - combined_cacheable
    combined_output = total_output + filter_output_total

    print(f"{'='*60}")
    print(f"COMBINED TOTALS (both steps)")
    print(f"{'='*60}")
    print(f"  Total input tokens:       {combined_input:,}")
    print(f"    Cacheable:              {combined_cacheable:,} ({combined_cacheable/combined_input*100:.0f}%)")
    print(f"    Unique:                 {combined_unique:,} ({combined_unique/combined_input*100:.0f}%)")
    print(f"  Total output tokens:      {combined_output:,}")
    print()
    print(f"For cost: multiply by $/1M tokens from your pricing table.")
    print(f"  With caching (gpt-5.4 standard):")
    print(f"    Input:  {combined_unique:,} unique × $5.00/1M + {combined_cacheable:,} cached × $0.50/1M")
    print(f"    Output: {combined_output:,} × $22.50/1M")
    input_cost = combined_unique / 1e6 * 5.00 + combined_cacheable / 1e6 * 0.50
    output_cost = combined_output / 1e6 * 22.50
    print(f"    = ${input_cost:.2f} input + ${output_cost:.2f} output = ${input_cost + output_cost:.2f} total")
    print()
    print(f"  Without caching (gpt-5.4 standard):")
    input_cost_nc = combined_input / 1e6 * 5.00
    output_cost_nc = combined_output / 1e6 * 22.50
    print(f"    Input:  {combined_input:,} × $5.00/1M = ${input_cost_nc:.2f}")
    print(f"    Output: {combined_output:,} × $22.50/1M = ${output_cost_nc:.2f}")
    print(f"    = ${input_cost_nc + output_cost_nc:.2f} total")
    print()
    print(f"  Batch API (gpt-5.4, no caching, 50% off input):")
    input_cost_batch = combined_input / 1e6 * 2.50
    output_cost_batch = combined_output / 1e6 * 15.00
    print(f"    Input:  {combined_input:,} × $2.50/1M = ${input_cost_batch:.2f}")
    print(f"    Output: {combined_output:,} × $15.00/1M = ${output_cost_batch:.2f}")
    print(f"    = ${input_cost_batch + output_cost_batch:.2f} total")


def cmd_prepare(args):
    """Sample N positions and build batch_input.jsonl + batch_meta.jsonl."""
    entries = load_entries()
    samples = sample_entries(entries, args.n, args.seed)
    print(f"Sampled {len(samples)} positions (seed={args.seed})")

    with open(BATCH_INPUT, 'w') as f_batch, open(BATCH_META, 'w') as f_meta:
        for i, entry in enumerate(samples):
            custom_id = f"pos_{i:06d}"
            messages = build_messages(entry)

            batch_row = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_completion_tokens": 2048,
                },
            }
            f_batch.write(json.dumps(batch_row) + '\n')

            meta_row = {
                "custom_id": custom_id,
                "fen": entry['fen'],
                "move_uci": entry['move_uci'],
                "move_san": entry.get('move_san', ''),
                "annotation": entry['annotation'],
                "is_mainline": entry['is_mainline'],
                "mainline_move": entry.get('mainline_move'),
                "parent_comment": entry.get('parent_comment'),
                "game": entry.get('game', ''),
            }
            f_meta.write(json.dumps(meta_row) + '\n')

    print(f"Written: {BATCH_INPUT} ({len(samples)} requests)")
    print(f"Written: {BATCH_META}")


def cmd_submit(args):
    """Submit batch_input.jsonl to OpenAI Batch API."""
    client = openai.OpenAI()
    with open(BATCH_INPUT, 'rb') as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"Uploaded file: {file_obj.id}")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch submitted: {batch.id}")
    print(f"Status: {batch.status}")

    # Save batch ID for collect
    id_file = os.path.join(SCRIPT_DIR, 'data', 'batch_id.txt')
    with open(id_file, 'w') as f:
        f.write(batch.id)
    print(f"Batch ID saved to {id_file}")


def cmd_collect(args):
    """Poll for batch completion and download results."""
    client = openai.OpenAI()

    batch_id = args.batch_id
    if not batch_id:
        id_file = os.path.join(SCRIPT_DIR, 'data', 'batch_id.txt')
        if os.path.exists(id_file):
            batch_id = open(id_file).read().strip()
        else:
            print("No --batch-id and no data/batch_id.txt found.")
            sys.exit(1)

    print(f"Polling batch {batch_id}...")
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else '?'
        total = batch.request_counts.total if batch.request_counts else '?'
        print(f"  Status: {status}  ({completed}/{total} completed)")

        if status in ('completed', 'failed', 'expired', 'cancelled'):
            break
        time.sleep(args.poll_interval)

    if status != 'completed':
        print(f"Batch ended with status: {status}")
        if batch.errors:
            for e in batch.errors.data[:5]:
                print(f"  Error: {e.message}")
        sys.exit(1)

    # Download output
    content = client.files.content(batch.output_file_id)
    with open(BATCH_OUTPUT, 'wb') as f:
        f.write(content.read())
    print(f"Downloaded results to {BATCH_OUTPUT}")

    # Download errors if any
    if batch.error_file_id:
        err_path = os.path.join(SCRIPT_DIR, 'data', 'batch_errors.jsonl')
        err_content = client.files.content(batch.error_file_id)
        with open(err_path, 'wb') as f:
            f.write(err_content.read())
        print(f"Downloaded errors to {err_path}")


def cmd_process(args):
    """Join batch output with metadata, apply filters, report stats."""
    # Load metadata
    meta = {}
    with open(BATCH_META) as f:
        for line in f:
            row = json.loads(line)
            meta[row['custom_id']] = row

    # Process results
    included = 0
    excluded = 0
    parse_errors = 0
    total_atoms = 0
    results = []

    with open(BATCH_OUTPUT) as f:
        for line in f:
            row = json.loads(line)
            cid = row['custom_id']
            m = meta.get(cid, {})

            resp = row.get('response', {})
            body = resp.get('body', {})
            choices = body.get('choices', [])

            if not choices:
                parse_errors += 1
                continue

            text = choices[0].get('message', {}).get('content', '')
            parsed = parse_json_output(text)

            # Postprocess
            inc, reason = postprocess_filter(parsed)
            if not inc and parsed.get('include', True):
                parsed['include'] = False
                parsed['exclude_reason'] = reason

            result = {**m, 'extracted': parsed}

            # Token usage
            usage = body.get('usage', {})
            if usage:
                result['usage'] = usage

            results.append(result)

            if parsed.get('include'):
                included += 1
                total_atoms += len(parsed.get('reasoning', []))
            else:
                excluded += 1

    # Write separate included / excluded files
    with open(RESULTS_INCLUDED, 'w') as f_inc, open(RESULTS_EXCLUDED, 'w') as f_exc:
        for r in results:
            if r['extracted'].get('include'):
                f_inc.write(json.dumps(r) + '\n')
            else:
                f_exc.write(json.dumps(r) + '\n')

    total = included + excluded + parse_errors
    print(f"\n{'='*60}")
    print(f"EXTRACTION RESULTS ({total} positions)")
    print(f"{'='*60}")
    print(f"  Included:     {included:5d} ({included/total*100:.1f}%)")
    print(f"  Excluded:     {excluded:5d} ({excluded/total*100:.1f}%)")
    print(f"  Parse errors: {parse_errors:5d}")
    print(f"  Total atoms:  {total_atoms:5d} (avg {total_atoms/max(included,1):.1f} per included)")
    print(f"\nWritten: {RESULTS_INCLUDED} ({included} rows)")
    print(f"Written: {RESULTS_EXCLUDED} ({excluded + parse_errors} rows)")

    # Token usage summary
    total_prompt = sum(r.get('usage', {}).get('prompt_tokens', 0) for r in results)
    total_completion = sum(r.get('usage', {}).get('completion_tokens', 0) for r in results)
    total_cached = sum(r.get('usage', {}).get('prompt_tokens_details', {}).get('cached_tokens', 0)
                       for r in results)
    if total_prompt:
        print(f"\n  Actual token usage:")
        print(f"    Prompt tokens:     {total_prompt:,}")
        if total_cached:
            print(f"      Cached:          {total_cached:,} ({total_cached/total_prompt*100:.0f}%)")
        print(f"    Completion tokens: {total_completion:,}")


def cmd_sync(args):
    """Run extraction synchronously (any OpenAI-compatible endpoint)."""
    # Build client
    client_kwargs = {}
    if args.base_url:
        client_kwargs['base_url'] = args.base_url
    if args.api_key:
        client_kwargs['api_key'] = args.api_key
    client = openai.OpenAI(**client_kwargs)

    # Extra body for local models (e.g. Qwen3 thinking mode)
    extra_body = {}
    if args.extra_body:
        extra_body = json.loads(args.extra_body)

    entries = load_entries()
    samples = sample_entries(entries, args.n, args.seed)
    print(f"Running sync extraction: {len(samples)} positions, model={args.model}")

    # Resume support
    done_keys = set()
    if args.resume:
        for path in (RESULTS_INCLUDED, RESULTS_EXCLUDED):
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        r = json.loads(line)
                        done_keys.add((r['fen'], r['move_uci']))
        if done_keys:
            print(f"  Resuming: {len(done_keys)} already done")

    included = 0
    excluded = 0
    parse_errors = 0
    total_atoms = 0
    total_prompt = 0
    total_completion = 0

    mode = 'a' if args.resume else 'w'
    with open(RESULTS_INCLUDED, mode) as f_inc, open(RESULTS_EXCLUDED, mode) as f_exc:
        for i, entry in enumerate(samples):
            key = (entry['fen'], entry['move_uci'])
            if key in done_keys:
                continue

            messages = build_messages(entry)

            try:
                kwargs = dict(
                    model=args.model,
                    messages=messages,
                    temperature=0.3,
                    max_completion_tokens=2048,
                )
                if extra_body:
                    kwargs['extra_body'] = extra_body

                resp = client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content
                parsed = parse_json_output(text)

                # Usage
                if resp.usage:
                    total_prompt += resp.usage.prompt_tokens
                    total_completion += resp.usage.completion_tokens

            except Exception as e:
                print(f"  [{i+1}/{len(samples)}] ERROR: {e}")
                parsed = {"include": False, "exclude_reason": f"API error: {e}"}
                parse_errors += 1

            # Postprocess
            inc, reason = postprocess_filter(parsed)
            if not inc and parsed.get('include', True):
                parsed['include'] = False
                parsed['exclude_reason'] = reason

            result = {
                'fen': entry['fen'],
                'move_uci': entry['move_uci'],
                'move_san': entry.get('move_san', ''),
                'annotation': entry['annotation'],
                'is_mainline': entry['is_mainline'],
                'mainline_move': entry.get('mainline_move'),
                'game': entry.get('game', ''),
                'extracted': parsed,
            }

            if parsed.get('include'):
                f_inc.write(json.dumps(result) + '\n')
                included += 1
                total_atoms += len(parsed.get('reasoning', []))
            else:
                f_exc.write(json.dumps(result) + '\n')
                excluded += 1

            status = 'INC' if parsed.get('include') else 'EXC'
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(samples)}] {status}  (running: {included} inc / {excluded} exc)")

    total = included + excluded + parse_errors
    print(f"\n{'='*60}")
    print(f"EXTRACTION RESULTS ({total} positions)")
    print(f"{'='*60}")
    print(f"  Included:     {included:5d} ({included/total*100:.1f}%)")
    print(f"  Excluded:     {excluded:5d} ({excluded/total*100:.1f}%)")
    print(f"  Parse errors: {parse_errors:5d}")
    print(f"  Total atoms:  {total_atoms:5d} (avg {total_atoms/max(included,1):.1f} per included)")
    if total_prompt:
        print(f"\n  Token usage:")
        print(f"    Prompt:     {total_prompt:,}")
        print(f"    Completion: {total_completion:,}")
    print(f"\nWritten: {RESULTS_INCLUDED} ({included} rows)")
    print(f"Written: {RESULTS_EXCLUDED} ({excluded + parse_errors} rows)")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Atom extraction pipeline')
    sub = p.add_subparsers(dest='cmd')

    # estimate
    s = sub.add_parser('estimate', help='Estimate token usage and cost')
    s.add_argument('-n', type=int, default=1000)
    s.add_argument('--seed', type=int, default=42)

    # prepare
    s = sub.add_parser('prepare', help='Build batch JSONL for OpenAI Batch API')
    s.add_argument('-n', type=int, default=1000)
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--model', default=DEFAULT_MODEL)

    # submit
    sub.add_parser('submit', help='Submit batch to OpenAI')

    # collect
    s = sub.add_parser('collect', help='Poll and download batch results')
    s.add_argument('--batch-id', default=None)
    s.add_argument('--poll-interval', type=int, default=30)

    # process
    sub.add_parser('process', help='Parse batch results and report stats')

    # sync
    s = sub.add_parser('sync', help='Run synchronously (any endpoint)')
    s.add_argument('-n', type=int, default=1000)
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--model', default=DEFAULT_MODEL)
    s.add_argument('--base-url', default=None, help='OpenAI-compatible base URL')
    s.add_argument('--api-key', default=None)
    s.add_argument('--extra-body', default=None,
                   help='JSON string for extra_body (e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\')')
    s.add_argument('--resume', action='store_true', help='Resume from existing results file')

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        sys.exit(1)

    {
        'estimate': cmd_estimate,
        'prepare': cmd_prepare,
        'submit': cmd_submit,
        'collect': cmd_collect,
        'process': cmd_process,
        'sync': cmd_sync,
    }[args.cmd](args)


if __name__ == '__main__':
    main()
