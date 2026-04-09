#!/usr/bin/env python3
"""
Atom extraction for chesspublishing commentary using a local vLLM server.

Uses async concurrent requests to exploit vLLM's continuous batching
and automatic prefix caching (the shared system prompt + demos are
cached after the first request).

Output paths are derived from input:
  data/chesspublishinga.jsonl -> data/chesspublishinga_included.jsonl
                                  data/chesspublishinga_excluded.jsonl

Examples:
  # Process all positions
  python run_extraction_qwen.py run --input data/chesspublishinga.jsonl

  # Resume after interruption (skips already-done entries)
  python run_extraction_qwen.py run --input data/chesspublishinga.jsonl --resume

  # More concurrent requests for faster throughput
  python run_extraction_qwen.py run --input data/chesspublishinga.jsonl --workers 64

  # Different model / server
  python run_extraction_qwen.py run --input data/chesspublishingb.jsonl \
      --model Qwen/Qwen3-8B --base-url http://127.0.0.1:8001/v1

  # Stats on existing results (no API calls)
  python run_extraction_qwen.py stats --input data/chesspublishinga.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
import time

import chess
import openai

# ── Paths ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def derive_output_paths(input_path):
    """Derive _included.jsonl and _excluded.jsonl from input path.

    data/chesspublishinga.jsonl -> data/chesspublishinga_included.jsonl
                                   data/chesspublishinga_excluded.jsonl
    """
    base, ext = os.path.splitext(input_path)
    return base + '_included' + ext, base + '_excluded' + ext

DEFAULT_MODEL = 'Qwen/Qwen3-32B'
DEFAULT_BASE_URL = 'http://127.0.0.1:8000/v1'
DEFAULT_WORKERS = 32

# ── System prompt (shorter, stricter for smaller models) ─────────────────

SYSTEM_PROMPT = """\
You extract chess move explanations from commentary into JSON.

CRITICAL: You are a COPYIST, not an analyst. Every atom must come
DIRECTLY from the commentary text. Do NOT add your own chess knowledge.
If the commentary says "attacks e3", write that. Do NOT add "which is
a key central square" or any other analysis of your own.

Output ONLY valid JSON — no text before or after.

When include=true:
{
  "include": true,
  "fen": "<the FEN>",
  "move_san": "<the move>",
  "move_uci": "<UCI>",
  "move_assessment": "<good|inaccuracy|mistake|blunder>",
  "book_commentary": "<full original commentary>",
  "reasoning": ["fact from commentary", "another fact from commentary"],
  "variation": "optional continuation line",
  "alternative": {"move": "X", "reasoning": ["..."]}
}

When include=false:
{"include": false, "exclude_reason": "why", "book_commentary": "<text>"}

move_assessment values (based on the ANNOTATOR's words, NOT your chess
judgment):
  good — annotator views the move positively or neutrally
  inaccuracy — annotator is skeptical, hints at a better option
  mistake — annotator says the move is wrong or a clear error
  blunder — annotator says the move loses immediately or is a disaster

RULES:
1. ONLY extract what the commentary SAYS. Never add facts it doesn't state.
2. Each atom must explain WHAT the move DOES — a threat it creates, a
   square it controls, a piece it activates, a plan it enables.
   "Qd4 prevents Black's rooks from reaching the central files" = good atom.
   "Nd4 is a strong resource for White" = bad (vague label, drop it).
   "Rg8 allows Black to hold on" = bad (says nothing about what Rg8 does).
3. Atoms must be about the CURRENT move, not about later moves in a
   variation. If the commentary describes what happens several moves later,
   put that in "variation", not in atoms.
4. "Parent context" describes a PREVIOUS move's annotation. Do NOT extract
   reasoning atoms from it — it is provided only so you understand the
   framing. Extract atoms ONLY from the "Commentary" field.
5. Position evaluations are NOT atoms. "Black has an extra pawn",
   "good position", "equal", "winning" describe the resulting position,
   not what the move does. EXCLUDE if the commentary ONLY evaluates
   the position without explaining what the move achieves.
6. Drop vague labels/assessments even if mixed with good atoms.
7. EXCLUDE when commentary only shows a variation line with no explanation
   of WHY the move works. A bare line + "holds" / "nothing clear" / "wins"
   is not an explanation.
8. INCLUDE when there is at least one concrete fact about the move:
   a piece, square, threat, file, diagonal, or plan.
9. Long variations (5+ moves) go in "variation", not in atoms.
10. When "alternatives" are provided (PGN sidelines with annotations and
    continuation lines), use them to populate the "alternative" field.
    Extract reasoning atoms from the annotator's comments on each
    alternative. If there are multiple alternatives, output "alternative"
    as a list of objects.
11. "move_assessment" captures the annotator's evaluation of the move.
    Use ONLY the annotator's own sentiment from the commentary text.
"""

# ── Few-shot demos ───────────────────────────────────────────────────────

DEMO_DATA = [
    # Demo 1: Rab8 — concrete, good move
    {
        'fen': 'r4rk1/pppq1ppp/2nb1n2/4p3/2Q3b1/2NP1NP1/PP2PPBP/R1BR2K1 b - - 6 10',
        'move_uci': 'a8b8', 'move_san': 'Rab8',
        'annotation': 'By defending b7 Black frees his c6-knight to play ...Nd4, and prepares a tactical trick.',
        'prev_moves': '', 'context_fen': '', 'is_mainline': True,
        'mainline_move': None, 'parent_comment': 'Better than',
    },
    # Demo 2: Bg2 — concrete drawback, inaccuracy
    {
        'fen': 'rnbqkb1r/1p3pp1/4pn1p/1Pp5/p3p1P1/P1N5/1BPP1P1P/R2QKBNR w KQkq - 0 9',
        'move_uci': 'f1g2', 'move_san': 'Bg2',
        'annotation': ("This automatic-looking move leaves the b-pawn short of defenders"
                        " - White cannot play a4 because a black pawn stands in the way!"),
        'prev_moves': '', 'context_fen': '', 'is_mainline': True,
        'mainline_move': None, 'parent_comment': 'Now things start going crazy.',
    },
    # Demo 3: Nxd3 — loses, mistake
    {
        'fen': 'r4rk1/1p2bpp1/p1np3p/4p3/4Pn1q/1NPP1N1P/PP1B1PP1/R2QR1K1 b - - 5 14',
        'move_uci': 'f4d3', 'move_san': 'Nxd3',
        'annotation': (
            'Nxd3 again loses since after: 27. Red1 '
            '(Black can\'t prevent Bg2-e4 winning the pinned knight.)'),
        'prev_moves': '', 'context_fen': '', 'is_mainline': False,
        'mainline_move': 'a5',
        'parent_comment': 'A good preparatory move, making room on a7 for the bishop.',
    },
    # Demo 4: Nc6 — vague label only, exclude
    {
        'fen': 'rn1qkbnr/pb1p1ppp/1p2p3/2p5/4PP2/1PN5/PBPP2PP/R2QKBNR b KQkq - 0 5',
        'move_uci': 'b8c6', 'move_san': 'Nc6',
        'annotation': 'A perfectly natural move.',
        'prev_moves': '', 'context_fen': '', 'is_mainline': True,
        'mainline_move': None, 'parent_comment': None,
    },
    # Demo 5: Rae8 — position evaluation only, exclude
    {
        'fen': 'r4rk1/1p4bp/p2p4/2pP3n/4B1pP/2B2nP1/1P3PN1/3R1R1K b - - 0 28',
        'move_uci': 'a8e8', 'move_san': 'Rae8',
        'annotation': '28...Rae8 and Black has an extra pawn and a good position.',
        'prev_moves': '', 'context_fen': '', 'is_mainline': False,
        'mainline_move': None, 'parent_comment': 'also looks good, for example',
    },
]

DEMO_RESPONSES = [
    # Demo 1: Rab8 — good, three concrete atoms
    {
        'include': True,
        'fen': 'r4rk1/pppq1ppp/2nb1n2/4p3/2Q3b1/2NP1NP1/PP2PPBP/R1BR2K1 b - - 6 10',
        'move_san': 'Rab8', 'move_uci': 'a8b8',
        'move_assessment': 'good',
        'reasoning': [
            'Rab8 defends the b7-pawn.',
            'By defending b7, Rab8 frees the knight on c6 to play Nd4.',
            'Rab8 prepares a tactical trick.',
        ],
    },
    # Demo 2: Bg2 — inaccuracy, concrete atoms about the drawback
    {
        'include': True,
        'fen': 'rnbqkb1r/1p3pp1/4pn1p/1Pp5/p3p1P1/P1N5/1BPP1P1P/R2QKBNR w KQkq - 0 9',
        'move_san': 'Bg2', 'move_uci': 'f1g2',
        'move_assessment': 'inaccuracy',
        'reasoning': [
            'Bg2 leaves the pawn on b5 with too few defenders.',
            'White cannot support the b5-pawn with a4 because Black\'s pawn on a4 blocks that advance.',
        ],
    },
    # Demo 3: Nxd3 — mistake, loses material
    {
        'include': True,
        'fen': 'r4rk1/1p2bpp1/p1np3p/4p3/4Pn1q/1NPP1N1P/PP1B1PP1/R2QR1K1 b - - 5 14',
        'move_san': 'Nxd3', 'move_uci': 'f4d3',
        'move_assessment': 'mistake',
        'reasoning': [
            'Nxd3 loses because after Red1, Black cannot prevent Bg2-e4 winning the pinned knight.',
        ],
        'variation': '27. Red1 Bg2-e4',
    },
    # Demo 4: Nc6 — excluded, vague label only
    {
        'include': False,
        'exclude_reason': 'Commentary only labels the move as "natural" — no concrete fact about what Nc6 does.',
    },
    # Demo 5: Rae8 — excluded, position evaluation only
    {
        'include': False,
        'exclude_reason': 'Commentary only evaluates the resulting position ("extra pawn", "good position") without explaining what Rae8 does.',
    },
]

# ── Helpers ──────────────────────────────────────────────────────────────

def load_entries(path):
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


def make_result_row(entry, parsed):
    """Build output row from input entry + parsed extraction."""
    return {
        'custom_id': f"fen_{hash((entry['fen'], entry['move_uci'])) % 10**8:08d}",
        'fen': entry['fen'],
        'move_uci': entry['move_uci'],
        'move_san': entry.get('move_san', ''),
        'annotation': entry['annotation'],
        'is_mainline': entry['is_mainline'],
        'mainline_move': entry.get('mainline_move'),
        'parent_comment': entry.get('parent_comment'),
        'game': entry.get('game', ''),
        'extracted': parsed,
    }


# ── Async extraction ────────────────────────────────────────────────────

async def extract_one(client, entry, model, semaphore):
    """Extract atoms for a single entry, respecting concurrency limit."""
    async with semaphore:
        messages = build_messages(entry)
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=2048,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            text = resp.choices[0].message.content
            parsed = parse_json_output(text)
            usage = None
            if resp.usage:
                usage = {
                    'prompt_tokens': resp.usage.prompt_tokens,
                    'completion_tokens': resp.usage.completion_tokens,
                    'total_tokens': resp.usage.total_tokens,
                }
        except Exception as e:
            parsed = {"include": False, "exclude_reason": f"API error: {e}"}
            usage = None

        # Postprocess
        inc, reason = postprocess_filter(parsed)
        if not inc and parsed.get('include', True):
            parsed['include'] = False
            parsed['exclude_reason'] = reason

        row = make_result_row(entry, parsed)
        if usage:
            row['usage'] = usage
        return row


async def run_extraction(args):
    """Run extraction with concurrent async requests."""
    input_path = os.path.join(SCRIPT_DIR, args.input)
    inc_path, exc_path = derive_output_paths(input_path)

    entries = load_entries(input_path)
    print(f"Loaded {len(entries)} entries")
    print(f"Input:   {input_path}")
    print(f"Output:  {inc_path}")
    print(f"         {exc_path}")
    print(f"Model:   {args.model} @ {args.base_url}")
    print(f"Workers: {args.workers}")

    # Always skip already-done entries (append mode)
    done_keys = set()
    for path in (inc_path, exc_path):
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    done_keys.add((r['fen'], r['move_uci']))
    if done_keys:
        print(f"Already done: {len(done_keys)}")

    todo = [e for e in entries if (e['fen'], e['move_uci']) not in done_keys]
    if args.n is not None:
        todo = todo[:args.n]
    print(f"To process: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    client = openai.AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key or 'not-needed',
    )
    semaphore = asyncio.Semaphore(args.workers)

    # Process in chunks to write results incrementally
    CHUNK_SIZE = args.workers * 4
    included = 0
    excluded = 0
    total_atoms = 0
    total_prompt = 0
    total_completion = 0
    t0 = time.time()

    f_inc = open(inc_path, 'a')
    f_exc = open(exc_path, 'a')

    try:
        for chunk_start in range(0, len(todo), CHUNK_SIZE):
            chunk = todo[chunk_start:chunk_start + CHUNK_SIZE]
            tasks = [extract_one(client, e, args.model, semaphore) for e in chunk]
            results = await asyncio.gather(*tasks)

            for row in results:
                if row['extracted'].get('include'):
                    f_inc.write(json.dumps(row) + '\n')
                    included += 1
                    total_atoms += len(row['extracted'].get('reasoning', []))
                else:
                    f_exc.write(json.dumps(row) + '\n')
                    excluded += 1

                usage = row.get('usage', {})
                total_prompt += usage.get('prompt_tokens', 0)
                total_completion += usage.get('completion_tokens', 0)

            # Flush after each chunk
            f_inc.flush()
            f_exc.flush()

            done_so_far = chunk_start + len(chunk)
            elapsed = time.time() - t0
            rate = done_so_far / elapsed
            print(f"  {done_so_far}/{len(todo)} done  "
                  f"({included} inc / {excluded} exc)  "
                  f"[{rate:.1f} req/s, {elapsed:.0f}s elapsed]")
    finally:
        f_inc.close()
        f_exc.close()

    elapsed = time.time() - t0
    total = included + excluded
    print(f"\n{'='*60}")
    print(f"EXTRACTION RESULTS ({total} positions in {elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Included:     {included:5d} ({included/max(total,1)*100:.1f}%)")
    print(f"  Excluded:     {excluded:5d} ({excluded/max(total,1)*100:.1f}%)")
    print(f"  Total atoms:  {total_atoms:5d} (avg {total_atoms/max(included,1):.1f} per included)")
    print(f"  Throughput:   {total/elapsed:.1f} req/s")
    if total_prompt:
        print(f"\n  Token usage:")
        print(f"    Prompt:     {total_prompt:,}")
        print(f"    Completion: {total_completion:,}")
    print(f"\nWritten: {inc_path} ({included} rows)")
    print(f"Written: {exc_path} ({excluded} rows)")


def cmd_stats(args):
    """Report stats on existing results without making API calls."""
    input_path = os.path.join(SCRIPT_DIR, args.input)
    inc_path, exc_path = derive_output_paths(input_path)
    for label, path in [('Included', inc_path), ('Excluded', exc_path)]:
        if not os.path.exists(path):
            print(f"{label}: {path} not found")
            continue
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        n_atoms = sum(len(r['extracted'].get('reasoning', []))
                      for r in rows if r['extracted'].get('include'))
        n_alt = sum(1 for r in rows
                    if r['extracted'].get('include') and r['extracted'].get('alternative'))
        total_prompt = sum(r.get('usage', {}).get('prompt_tokens', 0) for r in rows)
        total_comp = sum(r.get('usage', {}).get('completion_tokens', 0) for r in rows)
        print(f"\n{label}: {len(rows)} rows ({path})")
        if label == 'Included':
            print(f"  Atoms: {n_atoms} (avg {n_atoms/max(len(rows),1):.1f})")
            print(f"  With alternatives: {n_alt}")
        if total_prompt:
            print(f"  Tokens: {total_prompt:,} prompt + {total_comp:,} completion")

        # Exclude reasons
        if label == 'Excluded':
            from collections import Counter
            reasons = Counter(r['extracted'].get('exclude_reason', '?') for r in rows)
            print("  Exclude reasons:")
            for reason, cnt in reasons.most_common(10):
                print(f"    {cnt:4d}  {reason}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Atom extraction via local vLLM (Qwen3)')
    sub = p.add_subparsers(dest='cmd')

    s = sub.add_parser('run', help='Run extraction')
    s.add_argument('--input', required=True,
                   help='Input JSONL file (e.g. data/chesspublishinga.jsonl)')
    s.add_argument('--model', default=DEFAULT_MODEL)
    s.add_argument('--base-url', default=DEFAULT_BASE_URL,
                   help='vLLM server URL')
    s.add_argument('--api-key', default=None)
    s.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                   help='Max concurrent requests')
    s.add_argument('-n', type=int, default=None,
                   help='Process at most N entries (appends to output)')

    s = sub.add_parser('stats', help='Report stats on existing results')
    s.add_argument('--input', required=True,
                   help='Input JSONL file (to derive output paths)')

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        sys.exit(1)

    if args.cmd == 'run':
        asyncio.run(run_extraction(args))
    elif args.cmd == 'stats':
        cmd_stats(args)


if __name__ == '__main__':
    main()
