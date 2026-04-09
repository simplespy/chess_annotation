#!/usr/bin/env python3
"""Extract structured atoms from all commentary positions via OpenAI Batch API.

Workflow:
    # 1. Run Stockfish, build batch request file + metadata
    python evaluation/extract_all.py prepare \
        --input evaluation/data/logical_chess.jsonl \
        --batch-file evaluation/data/batch_input.jsonl \
        --meta-file evaluation/data/batch_meta.jsonl

    # 2. Upload and submit batch job
    python evaluation/extract_all.py submit \
        --batch-file evaluation/data/batch_input.jsonl

    # 3. Check status / download results when done
    python evaluation/extract_all.py collect \
        --batch-id batch_abc123 \
        --output evaluation/data/batch_output.jsonl

    # 4. Process results into included/excluded
    python evaluation/extract_all.py process \
        --meta-file evaluation/data/batch_meta.jsonl \
        --batch-output evaluation/data/batch_output.jsonl \
        --out-included evaluation/data/logical_chess_included.jsonl \
        --out-excluded evaluation/data/logical_chess_excluded.jsonl

    # Optional: run synchronously (no batch, direct API calls)
    python evaluation/extract_all.py sync \
        --input evaluation/data/logical_chess.jsonl \
        --out-included evaluation/data/logical_chess_included.jsonl \
        --out-excluded evaluation/data/logical_chess_excluded.jsonl
"""

import argparse
import json
import math
import os
import sys
import time

import chess
import chess.engine
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────

STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'
K = 0.00368208  # Lichess win% formula constant
DEFAULT_MODEL = 'gpt-5.4'


# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are extracting structured move explanations from chess commentary.

You will receive: a FEN, the move played, engine analysis, and commentary
for that move. Your job is to EXTRACT and STRUCTURE the commentary into a
JSON object.

Output ONLY valid JSON. No preamble, no markdown fences, no explanation.

When include=true, output:
{
  "include": true,
  "book_commentary": "<the full original commentary>",
  "quality": "good | inaccuracy | mistake | blunder",
  "reasoning": ["atomic fact 1", "atomic fact 2"],
  "variation": "19.Bxh7+ Kh8 20.Bg6+ ...",
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

Quality classification (use the engine data provided):
  good       -> wp_loss <= 10%
  inaccuracy -> wp_loss > 10%
  mistake    -> wp_loss > 20%
  blunder    -> wp_loss > 30%

CRITICAL RULES:

1. EXTRACTION ONLY. Extract information present in the commentary.
   Do NOT invent strategic claims the author did not make.
   Do NOT add alternative moves not discussed in the commentary.

2. Board-verifiable enrichment is OK: piece locations, whether a move
   gives check, whether a square is controlled.

3. CONTEXTUAL ATOMS. Each reasoning atom must be self-contained with full
   positional context. If an atom references a position after a sequence of
   moves, include the FULL preceding move sequence so it can be verified
   independently.

   EXAMPLE — given this text:
   "Nxd7 removes one of the defenders of the knight on f6, aiming to
   undermine Black's kingside defenses. After the expected Nh5, White can
   capture the bishop on g6, opening up Black's king to further attacks."

   BAD atoms (later atoms lose context):
   - "Nxd7 removes one of the defenders of the knight on f6."
   - "After Nh5, White can capture the bishop on g6."  <- Nh5 from where?
   - "Capturing the bishop on g6 opens up Black's king."  <- after what?

   GOOD atoms (each carries full move sequence from root):
   - "Nxd7 removes one of the defenders of the knight on f6."
   - "After Nxd7 Nh5, White can capture the bishop on g6."
   - "After Nxd7 Nh5, capturing on g6 opens up Black's king to attack."

4. Group logically connected setup+consequence into a SINGLE atom.

5. Name squares, pieces, and diagonals concretely.

6. INCLUDE/EXCLUDE. Set "include": false when:
   - Commentary praises a move but engine shows wp_loss > 10%
   - Commentary doesn't explain WHY a bad move is bad
   - Historical anecdotes, biographies, no move-specific analysis
   - Generic chess philosophy not applied to this position
   - Raw notation only, or too minimal to extract atoms

7. NEVER include generic philosophical statements as atoms.

8. Depth matches the commentary's richness, NOT an imposed judgment.

9. For inaccuracy/mistake/blunder: reasoning = why this move fails;
   alternative = the better move (from commentary only).\
"""


# ── Hardcoded demo data ──────────────────────────────────────────────────

DEMO_DATA = [
    {
        'fen': 'rn2k1nr/ppp2ppp/8/q7/1b1N2b1/2N5/PPPBBPPP/R2QK2R b KQkq - 0 8',
        'move_uci': 'a5e5',
        'annotation': (
            'Black\u2019s response pins the e2-bishop and attacks the unprotected '
            'd4-knight. Black rejects 8... Bxe2 as the recapture by 9 Qxe2+ '
            'gains another tempo for White.'
        ),
        'wp_loss': 0.0,
        'engine_lines': [
            {'move_uci': 'a5e5', 'move_san': 'Qe5', 'eval': '+1.18',
             'pv_san': 'Qe5 a3 Bxe2 Qxe2 Qxe2+ Kxe2 Bxc3 Bxc3',
             'cp': 118, 'mate': None, 'is_top': True},
            {'move_uci': 'a5c5', 'move_san': 'Qc5', 'eval': '+2.81',
             'pv_san': 'Qc5 Bxg4 Qxd4 Qe2+ Ne7 O-O-O O-O Bg5',
             'cp': 281, 'mate': None, 'is_top': True},
            {'move_uci': 'g4d7', 'move_san': 'Bd7', 'eval': '+3.16',
             'pv_san': 'Bd7 a3 Bxc3 Bxc3 Qb6 Bc4 Ne7 Qh5',
             'cp': 316, 'mate': None, 'is_top': True},
        ],
    },
    {
        'fen': 'r4rk1/pp1q1ppp/4pb2/8/2PpR3/1P2Q3/PB3PPP/5RK1 w - - 0 19',
        'move_uci': 'b2d4',
        'annotation': (
            'White regains the pawn, and his bishop now attacks in two directions. '
            'On the one hand, it threatens to take the a-pawn, on the other it aims '
            'at checkmate by... Bxf6 gxf6 21 Rg4+ Kh8 22 Qh6 Rg8 23 Qxf6+ and '
            'mate next move.'
        ),
        'wp_loss': 0.0,
        'engine_lines': [
            {'move_uci': 'b2d4', 'move_san': 'Bxd4', 'eval': '+0.52',
             'pv_san': 'Bxd4 Bxd4 Rxd4 Qe7 Rfd1 Rfd8 g3 b6',
             'cp': 52, 'mate': None, 'is_top': True},
            {'move_uci': 'e3d3', 'move_san': 'Qd3', 'eval': '-1.66',
             'pv_san': 'Qd3 Rfd8 Bc1 Rac8 h4 Rc5 a4 h6',
             'cp': -166, 'mate': None, 'is_top': True},
            {'move_uci': 'e3f3', 'move_san': 'Qf3', 'eval': '-1.76',
             'pv_san': 'Qf3 Rfe8 Rfe1 e5 g4 Qc6 Qd3 h6',
             'cp': -176, 'mate': None, 'is_top': True},
        ],
    },
    {
        'fen': 'r2qk2r/p1p1npp1/1pn1b2p/3pP3/3P1B2/2PB1N2/P1PQ2PP/1R3RK1 b kq - 1 12',
        'move_uci': 'e8g8',
        'annotation': (
            'Walking right into the teeth of the storm!\n'
            'Before making a move that suggests itself so readily, Black might have '
            'asked himself, \u201cHow can I exploit White\u2019s one weakness, the doubled '
            'pawns on the c-file?\u201d\n'
            'He might then have hit upon 12... Na5, with the object of swinging the '
            'knight to c4. There it blockades the doubled pawn, interferes with the '
            'free movement of White\u2019s pieces, and in general sticks like a bone in '
            'the throat. White could capture the knight, but then he parts with one '
            'of his valuable bishops, and as a result of the exchange his pawn '
            'position would be inferior to Black\u2019s. Finally, Black could then anchor '
            'one of his pieces to great effect on d5, a square from which it could '
            'never be evicted by pawns.'
        ),
        'wp_loss': 24.13,
        'engine_lines': [
            {'move_uci': 'd8d7', 'move_san': 'Qd7', 'eval': '+1.05',
             'pv_san': 'Qd7 Qf2 Bf5 Ne1 Nd8 Bc1 Be4 Qg3',
             'cp': 105, 'mate': None, 'is_top': True},
            {'move_uci': 'c6a5', 'move_san': 'Na5', 'eval': '+1.20',
             'pv_san': 'Na5 Bb5+ c6 Bd3 Ng6 Bg3 Nc4 Qf2',
             'cp': 120, 'mate': None, 'is_top': True},
            {'move_uci': 'e6f5', 'move_san': 'Bf5', 'eval': '+1.37',
             'pv_san': 'Bf5 Bxf5 Nxf5 Rbe1 Qd7 Qd3 O-O-O Nd2',
             'cp': 137, 'mate': None, 'is_top': True},
            {'move_uci': 'e8g8', 'move_san': 'O-O', 'eval': '+4.70',
             'pv_san': 'O-O Bxh6 Ng6 h4 Nxh4 Bg5 Nxf3+ Rxf3',
             'cp': 470, 'mate': None, 'is_top': False},
        ],
    },
]

_DEMO_RESPONSES = [
    {
        'include': True,
        'quality': 'good',
        'reasoning': [
            'Qe5 pins the bishop on e2 to the king on e1.',
            'Qe5 attacks the unprotected knight on d4.',
            'After 8...Bxe2 9.Qxe2+, White recaptures with check, gaining a '
            'tempo \u2014 so Black rejects this line.',
        ],
    },
    {
        'include': True,
        'quality': 'good',
        'reasoning': [
            'Bxd4 regains the pawn on d4.',
            'After Bxd4, the bishop threatens to capture the a7-pawn.',
            'After Bxd4, the bishop sets up a mating sequence: Bxf6 gxf6, '
            'Rg4+ Kh8, Qh6 Rg8, Qxf6+ and mate next move.',
        ],
        'variation': '20.Bxf6 gxf6 21.Rg4+ Kh8 22.Qh6 Rg8 23.Qxf6+',
    },
    {
        'include': True,
        'quality': 'mistake',
        'reasoning': [
            "O-O castles into White's prepared kingside attack.",
            "White's one weakness is the doubled c-pawns \u2014 Black should "
            "exploit this instead of castling.",
        ],
        'alternative': {
            'move': 'Na5',
            'reasoning': [
                'After 12...Na5, the knight reroutes toward c4 to blockade '
                'the doubled c-pawn.',
                "After 12...Na5 followed by ...Nc4, the knight interferes "
                "with the free movement of White's pieces.",
                "After 12...Na5 ...Nc4, if White captures with Bxc4 dxc4, "
                "White's pawn structure becomes inferior to Black's.",
                'After 12...Na5, Black can anchor a piece on d5, a square '
                'that cannot be evicted by pawns.',
            ],
        },
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────

def cp_to_winpct(cp):
    return 50 + 50 * math.tanh(K * cp / 2)


def compute_wp_loss(engine_lines, move_uci, turn):
    top_lines = [l for l in engine_lines if l.get('is_top', True)]
    if not top_lines:
        return 0.0, '?', False
    best = top_lines[0]
    if best['move_uci'] == move_uci:
        return 0.0, best['move_san'], True
    played = next((l for l in engine_lines if l['move_uci'] == move_uci), None)
    if played is None:
        return 0.0, best['move_san'], False
    best_wp = cp_to_winpct(best['cp'])
    played_wp = cp_to_winpct(played['cp'])
    loss = (best_wp - played_wp) if turn == chess.WHITE else (played_wp - best_wp)
    return round(max(0, loss), 2), best['move_san'], False


def get_engine_analysis(fen, move_uci=None, depth=22, multipv=3):
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        result = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        lines = []
        for info in result:
            score = info['score'].white()
            pv = info.get('pv', [])
            if pv:
                b = board.copy()
                san_moves = []
                for m in pv[:8]:
                    san_moves.append(b.san(m))
                    b.push(m)
                cp = score.score(mate_score=10000)
                mate = score.mate()
                eval_str = f'M{mate}' if mate is not None else f'{cp/100:+.2f}'
                lines.append({
                    'move_uci': pv[0].uci(), 'move_san': board.san(pv[0]),
                    'eval': eval_str, 'pv_san': ' '.join(san_moves),
                    'cp': cp, 'mate': mate, 'is_top': True,
                })
        if move_uci and not any(l['move_uci'] == move_uci for l in lines):
            move_obj = chess.Move.from_uci(move_uci)
            move_san = board.san(move_obj)
            b2 = board.copy()
            b2.push(move_obj)
            info2 = engine.analyse(b2, chess.engine.Limit(depth=depth))
            sc2 = info2['score'].white()
            cp2 = sc2.score(mate_score=10000)
            mate2 = sc2.mate()
            eval2 = f'M{mate2}' if mate2 is not None else f'{cp2/100:+.2f}'
            pv2 = info2.get('pv', [])
            pv_sans = [move_san]
            b3 = b2.copy()
            for m in pv2[:7]:
                pv_sans.append(b3.san(m))
                b3.push(m)
            lines.append({
                'move_uci': move_uci, 'move_san': move_san,
                'eval': eval2, 'pv_san': ' '.join(pv_sans),
                'cp': cp2, 'mate': mate2, 'is_top': False,
            })
    return lines


def build_user_prompt(entry, engine_lines):
    board = chess.Board(entry['fen'])
    move_san = board.san(chess.Move.from_uci(entry['move_uci']))
    top_lines = [l for l in engine_lines if l.get('is_top', True)]
    played_line = next((l for l in engine_lines if not l.get('is_top', True)), None)
    pv_text = []
    for i, line in enumerate(top_lines, 1):
        pv_text.append(f"  {i}. {line['move_san']} ({line['eval']}): {line['pv_san']}")
    wp = entry.get('wp_loss', 0)
    if wp > 30:   hint = f' [BLUNDER \u2014 wp_loss: {wp:.1f}%]'
    elif wp > 20: hint = f' [MISTAKE \u2014 wp_loss: {wp:.1f}%]'
    elif wp > 10: hint = f' [INACCURACY \u2014 wp_loss: {wp:.1f}%]'
    else:         hint = f' [wp_loss: {wp:.1f}%]'
    prompt = (f'FEN: {entry["fen"]}\nMove played: {move_san}{hint}\n'
              f'Engine top-3:\n' + '\n'.join(pv_text) + '\n')
    if played_line:
        prompt += (f'Played move eval:\n  {played_line["move_san"]} '
                   f'({played_line["eval"]}): {played_line["pv_san"]}\n')
    prompt += f'\nCommentary: "{entry["annotation"]}"\n'
    return prompt


def build_demos():
    demos = []
    for data, resp in zip(DEMO_DATA, _DEMO_RESPONSES):
        r = {**resp, 'book_commentary': data['annotation']}
        demos.append({'role': 'user', 'content': build_user_prompt(data, data['engine_lines'])})
        demos.append({'role': 'assistant', 'content': json.dumps(r, indent=2)})
    return demos


DEMOS = build_demos()


def build_messages(entry, engine_lines):
    """Build the full message list for one extraction request."""
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    messages.extend(DEMOS)
    messages.append({'role': 'user', 'content': build_user_prompt(entry, engine_lines)})
    return messages


def postprocess_filter(entry, parsed):
    wp_loss = entry.get('wp_loss', 0)
    quality = parsed.get('quality', 'good')
    if quality == 'good' and wp_loss > 10:
        return False, f"quality/engine conflict: commentary says good but wp_loss={wp_loss:.1f}%"
    if quality in ('inaccuracy', 'mistake', 'blunder') and wp_loss < 2:
        return False, f"quality/engine conflict: commentary says {quality} but wp_loss={wp_loss:.1f}%"
    alt = parsed.get('alternative')
    if alt and not alt.get('reasoning'):
        return False, "alternative has no reasoning"
    if parsed.get('include', True) and not parsed.get('reasoning'):
        return False, "no reasoning atoms extracted"
    return parsed.get('include', True), parsed.get('exclude_reason')


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
        return {'include': False, 'exclude_reason': 'JSON parse error', 'raw': text}


# ── Verification ──────────────────────────────────────────────────────────

VERIFY_PROMPT = """\
Check whether each reasoning atom below is self-contained and independently
verifiable from the given FEN position. An atom that references a position
after moves MUST include the full preceding move sequence.

FEN: {fen}
Move played: {move_san}

Atoms:
{atoms_text}

For each atom, check:
- Does it reference moves without full context? (e.g. "After Nh5" without
  specifying what came before Nh5)
- Would a verifier need to read other atoms to understand this one?
- Are piece references grounded in a verifiable position?

Output ONLY valid JSON:
{{"results": [{{"atom_index": 1, "ok": true/false, "issue": "..."}}]}}
"""


def verify_atoms(entry, parsed, model="gpt-4o-mini"):
    """LLM-based self-containment check for extracted atoms."""
    import openai
    reasoning = parsed.get('reasoning', [])
    alt = parsed.get('alternative', {})
    alt_reasoning = alt.get('reasoning', []) if alt else []
    all_atoms = reasoning + alt_reasoning
    if not all_atoms:
        return {"results": [], "all_ok": True}

    board = chess.Board(entry['fen'])
    move_san = board.san(chess.Move.from_uci(entry['move_uci']))
    atoms_text = '\n'.join(f'{i+1}. "{a}"' for i, a in enumerate(all_atoms))

    prompt = VERIFY_PROMPT.format(
        fen=entry['fen'], move_san=move_san, atoms_text=atoms_text)

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=1024,
    )
    text = resp.choices[0].message.content.strip()
    result = parse_json_output(text)
    results = result.get('results', [])
    all_ok = all(r.get('ok', True) for r in results)
    return {"results": results, "all_ok": all_ok}


# ── Filter prompt ────────────────────────────────────────────────────────

FILTER_PROMPT = """\
You are filtering extracted reasoning atoms from chess commentary.

You will receive the FULL extracted JSON for a position (FEN, move played,
engine lines, original commentary, reasoning atoms, alternative section, etc.)

You have TWO jobs:
1. Classify each REASONING atom (keep / contextualize / move_to_alternative / remove)
2. Review each EXISTING ALTERNATIVE atom (keep / remove / paraphrase)

─── REASONING ATOMS ───

KEEP an atom if it states a concrete, verifiable positional fact about what
the move does: attacks, defends, controls, pins, blocks, develops, opens/
closes lines, creates threats, prevents opponent plans, explains why a move
is bad, etc.

CONTEXTUALIZE an atom that has useful content but is not self-contained on
its own. An atom is not self-contained if it references a consequence,
position, or piece without enough context to verify it independently.

Each atom stays SEPARATE — do NOT combine multiple atoms into one. Instead,
add the missing context so each atom is independently verifiable.
  Original: "On e4 the knight controls d6 and f6"
  Contextualized: "After Nd2, the knight on e4 would control d6 and f6."
Output: {"action": "contextualize", "new_text": "After Nd2, ..."}

IMPORTANT: Only add REFERENTIAL context — which move, which piece, which
square, which position. Do NOT add new chess claims, analysis, or
consequences that were not in the original atom. The goal is to make the
existing claim self-contained, not to enrich it.

MOVE_TO_ALTERNATIVE if the atom describes an ALTERNATIVE move (a move NOT
played) and its consequences.
  Output: {"action": "move_to_alternative", "move": "Bf4", "text": "..."}
  You may paraphrase the text so it reads naturally in the alternative section.

Conclusion vs. detailed analysis of alternatives:
- A CONCLUSION that the played move avoids or is better than an alternative
  is KEEP: "Qh4 avoids the inferior Qg5" -> KEEP.
- A DETAILED ANALYSIS of what happens after the alternative move is
  MOVE_TO_ALTERNATIVE: "After 32.Qc1 Nxd4 33.Bxd4 Rxd4, Black threatens
  Rd3" -> MOVE_TO_ALTERNATIVE for Qc1.
- When an atom MIXES both (conclusion + detailed line), SPLIT it using
  kept_brief:
  {"action": "move_to_alternative", "move": "Qc1",
   "text": "After 32.Qc1 Nxd4 33.Bxd4 Rxd4, Black threatens Rd3.",
   "kept_brief": "Qb2 avoids Qc1, which leads to dangerous threats on the d-file."}
  kept_brief stays in reasoning, text goes to the alternative.

DEDUPLICATION: Check the existing alternative section before adding. If an
alternative for that move already exists with equivalent content, do NOT add
duplicate text. You may instead paraphrase or remove the existing atom via
review_alternatives (see below).

REMOVE an atom ONLY if it is a generic label with no concrete positional
content. "Nbd2 is a typical Colle Attack manoeuvre." — naming an opening or
saying "typical/standard" without any positional claim is not a useful atom.

─── EXISTING ALTERNATIVE ATOMS ───

Review each atom in the existing alternative section(s). For each, choose:
- keep: leave as-is
- remove: delete (duplicate, or superseded by a moved reasoning atom)
- paraphrase: rewrite for clarity (provide new_text)

Output these in the "review_alternatives" array.

─── OUTPUT FORMAT ───

Output ONLY valid JSON:
{"atoms": [
  {"index": 0, "text": "...", "action": "keep"},
  {"index": 1, "text": "...", "action": "contextualize", "new_text": "..."},
  {"index": 2, "text": "...", "action": "move_to_alternative", "move": "Qc1",
   "text": "detailed line...", "kept_brief": "brief conclusion..."},
  {"index": 3, "text": "...", "action": "remove", "reason": "generic label"}
 ],
 "review_alternatives": [
  {"move": "Qc1", "index": 0, "action": "keep"},
  {"move": "Qc1", "index": 1, "action": "remove", "reason": "superseded"},
  {"move": "Bf4", "index": 0, "action": "paraphrase", "new_text": "..."}
 ]
}

review_alternatives uses the move name and 0-based index within that
alternative's reasoning array. Omit review_alternatives if there are no
existing alternatives.

Be conservative: when in doubt, KEEP. Prefer CONTEXTUALIZE over REMOVE when
an atom has useful content but just lacks context. Prefer MOVE_TO_ALTERNATIVE
over REMOVE when the atom is about a different move.
"""


# ── Filter helpers ───────────────────────────────────────────────────────

def _normalize_alts(alt):
    """Normalize alternative field to a list of {move, reasoning, ...} dicts."""
    if not alt:
        return []
    if isinstance(alt, list):
        return [dict(a) for a in alt]
    return [dict(alt)]


def _alt_quality(alt_move_san, engine_lines):
    """Compute quality label for an alternative move from engine lines."""
    top_lines = [l for l in engine_lines if l.get('is_top', True)]
    if not top_lines:
        return None
    best = top_lines[0]
    alt_line = next((l for l in engine_lines if l['move_san'] == alt_move_san), None)
    if alt_line is None:
        return None
    best_wp = cp_to_winpct(best['cp'])
    alt_wp = cp_to_winpct(alt_line['cp'])
    loss = abs(best_wp - alt_wp)
    if loss > 30:
        return 'blunder'
    elif loss > 20:
        return 'mistake'
    elif loss > 10:
        return 'inaccuracy'
    return 'good'


def build_filter_messages(row):
    """Build the system + user messages for one filter request."""
    extracted = row['extracted']
    reasoning = extracted.get('reasoning', [])
    alt = extracted.get('alternative')

    all_atoms = [{'index': i, 'text': a} for i, a in enumerate(reasoning)]
    context = {
        'fen': row['fen'],
        'move_uci': row['move_uci'],
        'move_san': row.get('move_san', ''),
        'annotation': row['annotation'],
        'quality': row.get('quality', ''),
        'wp_loss': row.get('wp_loss', 0),
        'engine_lines': [
            {'move_san': l['move_san'], 'eval': l['eval'],
             'pv_san': l['pv_san'], 'is_top': l.get('is_top', True)}
            for l in row.get('engine_lines', [])
        ],
        'reasoning': reasoning,
        'alternative': alt,
    }
    atoms_text = '\n'.join(f'{a["index"]}. "{a["text"]}"' for a in all_atoms)
    user_msg = (
        f'Position data:\n```json\n{json.dumps(context, indent=2)}\n```\n\n'
        f'Reasoning atoms to classify:\n{atoms_text}'
    )
    return [
        {'role': 'system', 'content': FILTER_PROMPT},
        {'role': 'user', 'content': user_msg},
    ]


def apply_filter_result(row, result):
    """Apply parsed LLM filter result to a row. Returns (filter_info, out_row)."""
    extracted = row['extracted']
    reasoning = extracted.get('reasoning', [])
    alt = extracted.get('alternative')
    all_atoms = [{'index': i, 'text': a} for i, a in enumerate(reasoning)]

    atoms_by_idx = {a['index']: a for a in result.get('atoms', [])}
    moved_to_alt = []

    for a in result['atoms']:
        if a.get('action') == 'move_to_alternative':
            moved_to_alt.append({
                'move': a.get('move', '?'),
                'text': a.get('text', ''),
            })

    filtered_reasoning = []
    for atom_info in all_atoms:
        idx = atom_info['index']
        a = atoms_by_idx.get(idx, {})
        action = a.get('action', 'keep')

        if action == 'remove':
            continue
        if action == 'move_to_alternative':
            if a.get('kept_brief'):
                filtered_reasoning.append(a['kept_brief'])
            continue
        if action == 'contextualize' and a.get('new_text'):
            filtered_reasoning.append(a['new_text'])
        else:
            filtered_reasoning.append(atom_info['text'])

    # Apply review_alternatives to existing alternatives
    alternatives = _normalize_alts(alt)
    review_alts = result.get('review_alternatives', [])

    alt_reviews = {}
    for ra in review_alts:
        key = (ra.get('move', ''), ra.get('index', 0))
        alt_reviews[key] = ra

    for a in alternatives:
        move_name = a['move']
        old_reasoning = a.get('reasoning', [])
        new_reasoning = []
        for i, atom_text in enumerate(old_reasoning):
            ra = alt_reviews.get((move_name, i))
            if ra:
                action = ra.get('action', 'keep')
                if action == 'remove':
                    continue
                elif action == 'paraphrase' and ra.get('new_text'):
                    new_reasoning.append(ra['new_text'])
                else:
                    new_reasoning.append(atom_text)
            else:
                new_reasoning.append(atom_text)
        a['reasoning'] = new_reasoning

    for ma in moved_to_alt:
        move_name = ma['move']
        existing = next((a for a in alternatives if a['move'] == move_name), None)
        if existing:
            existing['reasoning'].append(ma['text'])
        else:
            alternatives.append({'move': move_name, 'reasoning': [ma['text']]})

    alternatives = [a for a in alternatives if a.get('reasoning')]

    filter_info = {
        'atoms': result.get('atoms', []),
        'filtered_reasoning': filtered_reasoning,
        'alternatives': alternatives,
        'new_alternatives': moved_to_alt,
        'review_alternatives': review_alts,
    }

    # Build output row
    out_row = dict(row)
    filtered_extracted = dict(row['extracted'])
    filtered_extracted['reasoning'] = filtered_reasoning
    filtered_extracted['alternative'] = (
        alternatives[0] if len(alternatives) == 1 else (alternatives or None))
    out_row['extracted'] = filtered_extracted

    # Add quality labels to alternatives
    engine_lines = row.get('engine_lines', [])
    alt_field = filtered_extracted.get('alternative')
    if alt_field and engine_lines:
        alt_list = alt_field if isinstance(alt_field, list) else [alt_field]
        for a in alt_list:
            if 'quality' not in a and a.get('move'):
                q = _alt_quality(a['move'], engine_lines)
                if q:
                    a['quality'] = q

    return filter_info, out_row


def filter_atoms(row, client, model=DEFAULT_MODEL):
    """Run filter synchronously: call LLM + apply result. Returns (filter_info, out_row)."""
    extracted = row['extracted']
    reasoning = extracted.get('reasoning', [])
    alt = extracted.get('alternative')

    if not reasoning:
        empty_info = {'atoms': [], 'filtered_reasoning': [],
                      'alternatives': _normalize_alts(alt), 'new_alternatives': [],
                      'review_alternatives': []}
        out_row = dict(row)
        return empty_info, out_row

    messages = build_filter_messages(row)
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=0, max_completion_tokens=2048,
    )
    text = resp.choices[0].message.content.strip()
    result = parse_json_output(text)

    if 'atoms' not in result:
        # Fallback: keep everything as-is
        fallback = {'atoms': [], 'filtered_reasoning': reasoning,
                    'alternatives': _normalize_alts(alt), 'new_alternatives': [],
                    'review_alternatives': []}
        out_row = dict(row)
        return fallback, out_row

    return apply_filter_result(row, result)


# ── Commands ──────────────────────────────────────────────────────────────

def cmd_prepare(args):
    """Run Stockfish analysis for each position, produce batch_input.jsonl + batch_meta.jsonl."""
    with open(args.input) as f:
        dataset = [json.loads(line) for line in f]
    print(f"Loaded {len(dataset)} entries from {args.input}")

    start = args.start - 1
    end = args.end or len(dataset)
    end = min(end, len(dataset))
    subset = dataset[start:end]
    print(f"Preparing positions {args.start}..{end} ({len(subset)} total)")

    # Resume: skip positions already in meta file
    already_done = set()
    if os.path.exists(args.meta_file):
        with open(args.meta_file) as f:
            for line in f:
                row = json.loads(line)
                already_done.add(row['custom_id'])
        print(f"Resuming: {len(already_done)} already prepared")

    batch_f = open(args.batch_file, 'a')
    meta_f = open(args.meta_file, 'a')
    n_new = 0

    try:
        for i, entry in enumerate(subset):
            pos_num = start + i + 1
            custom_id = f"pos-{pos_num}"

            if custom_id in already_done:
                continue

            board = chess.Board(entry['fen'])
            move_san = board.san(chess.Move.from_uci(entry['move_uci']))
            game = (entry.get('metadata', {}).get('White', '?') + ' \u2013 ' +
                    entry.get('metadata', {}).get('Black', '?'))

            engine_lines = get_engine_analysis(entry['fen'], entry['move_uci'],
                                               depth=args.depth)
            wp, best_san, is_top = compute_wp_loss(
                engine_lines, entry['move_uci'], board.turn)
            entry['wp_loss'] = wp
            quality = ('blunder' if wp > 30 else 'mistake' if wp > 20
                       else 'inaccuracy' if wp > 10 else 'good')

            messages = build_messages(entry, engine_lines)

            # Batch API request format
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
            batch_f.write(json.dumps(batch_row) + '\n')

            # Metadata (everything we need to reconstruct the output row)
            meta_row = {
                "custom_id": custom_id,
                "position_number": pos_num,
                "fen": entry['fen'],
                "move_uci": entry['move_uci'],
                "move_san": move_san,
                "annotation": entry['annotation'],
                "wp_loss": wp,
                "quality": quality,
                "game": game,
                "engine_lines": engine_lines,
            }
            meta_f.write(json.dumps(meta_row) + '\n')

            n_new += 1
            if n_new % 50 == 0:
                print(f"  prepared {n_new} positions...")
                batch_f.flush()
                meta_f.flush()

            print(f"  [{pos_num}/{end}] {game} \u2014 {move_san} "
                  f"({quality}, wp={wp:.1f}%)")

    finally:
        batch_f.close()
        meta_f.close()

    print(f"\nDone. {n_new} new positions prepared.")
    print(f"  Batch file: {args.batch_file}")
    print(f"  Meta file:  {args.meta_file}")


def cmd_submit(args):
    """Upload batch_input.jsonl and submit to OpenAI Batch API."""
    import openai
    client = openai.OpenAI()

    print(f"Uploading {args.batch_file}...")
    with open(args.batch_file, 'rb') as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"  File ID: {file_obj.id}")

    print("Creating batch...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch ID: {batch.id}")
    print(f"  Status:   {batch.status}")
    print(f"\nRun this to check status / download results:")
    print(f"  python evaluation/extract_all.py collect "
          f"--batch-id {batch.id} --output <output.jsonl>")


def cmd_collect(args):
    """Check batch status and download results when complete."""
    import openai
    client = openai.OpenAI()

    batch = client.batches.retrieve(args.batch_id)
    print(f"Batch {args.batch_id}")
    print(f"  Status: {batch.status}")
    print(f"  Total:  {batch.request_counts.total}")
    print(f"  Done:   {batch.request_counts.completed}")
    print(f"  Failed: {batch.request_counts.failed}")

    if args.poll:
        while batch.status not in ('completed', 'failed', 'expired', 'cancelled'):
            print(f"  Status: {batch.status} "
                  f"({batch.request_counts.completed}/{batch.request_counts.total})... "
                  f"waiting {args.poll_interval}s")
            time.sleep(args.poll_interval)
            batch = client.batches.retrieve(args.batch_id)

    if batch.status != 'completed':
        print(f"\nBatch not yet complete (status: {batch.status}). "
              f"Re-run with --poll to wait.")
        return

    print(f"\nDownloading results...")
    content = client.files.content(batch.output_file_id)
    with open(args.output, 'wb') as f:
        f.write(content.read())
    print(f"  Saved to {args.output}")

    if batch.error_file_id:
        err_path = args.output.replace('.jsonl', '_errors.jsonl')
        err_content = client.files.content(batch.error_file_id)
        with open(err_path, 'wb') as f:
            f.write(err_content.read())
        print(f"  Errors saved to {err_path}")


def cmd_process(args):
    """Join batch output with metadata, apply postprocess filter, write included/excluded."""
    # Load metadata keyed by custom_id
    meta = {}
    with open(args.meta_file) as f:
        for line in f:
            row = json.loads(line)
            meta[row['custom_id']] = row
    print(f"Loaded {len(meta)} metadata entries from {args.meta_file}")

    # Load batch output
    with open(args.batch_output) as f:
        batch_results = [json.loads(line) for line in f]
    print(f"Loaded {len(batch_results)} batch results from {args.batch_output}")

    n_included = 0
    n_excluded = 0
    n_errors = 0

    inc_f = open(args.out_included, 'w')
    exc_f = open(args.out_excluded, 'w')

    try:
        for br in batch_results:
            cid = br['custom_id']
            m = meta.get(cid)
            if m is None:
                print(f"  WARNING: no metadata for {cid}, skipping")
                continue

            resp = br.get('response', {})
            if resp.get('status_code') != 200:
                print(f"  ERROR: {cid} status={resp.get('status_code')}")
                n_errors += 1
                continue

            text = resp['body']['choices'][0]['message']['content']
            parsed = parse_json_output(text)

            entry = {
                'fen': m['fen'], 'move_uci': m['move_uci'],
                'wp_loss': m['wp_loss'], 'annotation': m['annotation'],
            }
            pp_include, pp_reason = postprocess_filter(entry, parsed)
            if not pp_include and parsed.get('include', True):
                parsed['include'] = False
                parsed['exclude_reason'] = pp_reason

            # Optional verification
            verification = None
            if args.verify and parsed.get('include', True):
                verification = verify_atoms(entry, parsed,
                                            model=args.verify_model)
                if not verification['all_ok']:
                    issues = [r for r in verification['results']
                              if not r.get('ok')]
                    print(f"  VERIFY {cid}: {len(issues)} atom(s) flagged")

            row = {
                'position_number': m['position_number'],
                'fen': m['fen'],
                'move_uci': m['move_uci'],
                'move_san': m['move_san'],
                'annotation': m['annotation'],
                'wp_loss': m['wp_loss'],
                'quality': m['quality'],
                'game': m['game'],
                'engine_lines': m['engine_lines'],
                'extracted': parsed,
                'model': args.model,
            }
            if verification is not None:
                row['verification'] = verification

            if parsed.get('include', True):
                inc_f.write(json.dumps(row) + '\n')
                n_included += 1
            else:
                exc_f.write(json.dumps(row) + '\n')
                n_excluded += 1

    finally:
        inc_f.close()
        exc_f.close()

    print(f"\nDone. {n_included} included, {n_excluded} excluded, {n_errors} errors.")
    print(f"  Included -> {args.out_included}")
    print(f"  Excluded -> {args.out_excluded}")


def cmd_sync(args):
    """Run extraction synchronously (no batch API). Useful for small runs."""
    import openai
    client = openai.OpenAI()

    with open(args.input) as f:
        dataset = [json.loads(line) for line in f]
    print(f"Loaded {len(dataset)} entries from {args.input}")

    start = args.start - 1
    end = args.end or len(dataset)
    end = min(end, len(dataset))

    # Resume
    already_done = set()
    for path in [args.out_included, args.out_excluded]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    already_done.add((row['fen'], row['move_uci']))
    if already_done:
        print(f"Resuming: {len(already_done)} already processed")

    n_included = 0
    n_excluded = 0

    for idx in range(start, end):
        entry = dataset[idx]
        pos_num = idx + 1
        key = (entry['fen'], entry['move_uci'])
        if key in already_done:
            continue

        board = chess.Board(entry['fen'])
        move_san = board.san(chess.Move.from_uci(entry['move_uci']))
        game = (entry.get('metadata', {}).get('White', '?') + ' \u2013 ' +
                entry.get('metadata', {}).get('Black', '?'))

        engine_lines = get_engine_analysis(entry['fen'], entry['move_uci'],
                                           depth=args.depth)
        wp, best_san, is_top = compute_wp_loss(
            engine_lines, entry['move_uci'], board.turn)
        entry['wp_loss'] = wp
        quality = ('blunder' if wp > 30 else 'mistake' if wp > 20
                   else 'inaccuracy' if wp > 10 else 'good')

        print(f"[{pos_num}/{end}] {game} \u2014 {move_san} "
              f"({quality}, wp={wp:.1f}%)")

        messages = build_messages(entry, engine_lines)
        token_key = ("max_completion_tokens"
                     if any(m in args.model for m in ("gpt-4.1", "gpt-5", "o3", "o4"))
                     else "max_tokens")
        resp = client.chat.completions.create(
            model=args.model, messages=messages,
            temperature=0.3, **{token_key: 2048},
        )
        text = resp.choices[0].message.content.strip()
        parsed = parse_json_output(text)

        pp_include, pp_reason = postprocess_filter(entry, parsed)
        if not pp_include and parsed.get('include', True):
            parsed['include'] = False
            parsed['exclude_reason'] = pp_reason
            print(f"  ** POST-FILTER: excluded \u2014 {pp_reason}")

        verification = None
        if args.verify and parsed.get('include', True):
            verification = verify_atoms(entry, parsed)
            if not verification['all_ok']:
                issues = [r for r in verification['results']
                          if not r.get('ok')]
                print(f"  ** VERIFY: {len(issues)} atom(s) flagged")

        row = {
            'position_number': pos_num,
            'fen': entry['fen'], 'move_uci': entry['move_uci'],
            'move_san': move_san, 'annotation': entry['annotation'],
            'wp_loss': wp, 'quality': quality, 'game': game,
            'engine_lines': engine_lines, 'extracted': parsed,
            'model': args.model,
        }
        if verification is not None:
            row['verification'] = verification

        is_included = parsed.get('include', True)
        out_path = args.out_included if is_included else args.out_excluded
        with open(out_path, 'a') as f:
            f.write(json.dumps(row) + '\n')

        if is_included:
            n_included += 1
            print(f"  [INCL] {len(parsed.get('reasoning', []))} atoms")
        else:
            n_excluded += 1
            print(f"  [EXCL] {parsed.get('exclude_reason', '?')}")

    print(f"\nDone. {n_included} included, {n_excluded} excluded.")


def cmd_filter_prepare(args):
    """Build batch request JSONL for filter pass (no Stockfish needed)."""
    with open(args.input) as f:
        all_rows = [json.loads(line) for line in f]
    print(f"Loaded {len(all_rows)} positions from {args.input}")

    n_written = 0
    with open(args.batch_file, 'w') as bf:
        for row in all_rows:
            reasoning = row['extracted'].get('reasoning', [])
            if not reasoning:
                continue
            custom_id = f"filter-{row['position_number']}"
            messages = build_filter_messages(row)
            batch_row = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": messages,
                    "temperature": 0,
                    "max_completion_tokens": 2048,
                },
            }
            bf.write(json.dumps(batch_row) + '\n')
            n_written += 1

    print(f"Wrote {n_written} requests to {args.batch_file}")
    print(f"  ({len(all_rows) - n_written} skipped — no reasoning atoms)")
    print(f"\nNext: python evaluation/extract_all.py submit "
          f"--batch-file {args.batch_file}")


def cmd_filter_process(args):
    """Join filter batch output with original rows, apply filter logic."""
    # Load original rows keyed by position_number
    with open(args.input) as f:
        all_rows = {r['position_number']: r for r in (json.loads(l) for l in f)}
    print(f"Loaded {len(all_rows)} original rows from {args.input}")

    with open(args.batch_output) as f:
        batch_results = [json.loads(line) for line in f]
    print(f"Loaded {len(batch_results)} batch results from {args.batch_output}")

    n_included = 0
    n_excluded = 0
    n_errors = 0
    n_atoms_before = 0
    n_atoms_after = 0

    inc_f = open(args.out_included, 'w')
    exc_f = open(args.out_excluded, 'w')

    try:
        for br in batch_results:
            cid = br['custom_id']
            # custom_id is "filter-{position_number}"
            pos_num = int(cid.split('-', 1)[1])
            row = all_rows.get(pos_num)
            if row is None:
                print(f"  WARNING: no row for {cid}, skipping")
                continue

            resp = br.get('response', {})
            if resp.get('status_code') != 200:
                print(f"  ERROR: {cid} status={resp.get('status_code')}")
                n_errors += 1
                continue

            text = resp['body']['choices'][0]['message']['content']
            result = parse_json_output(text)

            if 'atoms' not in result:
                # Keep as-is on parse failure
                out_row = dict(row)
                out_row['filter_error'] = 'no atoms key in response'
                inc_f.write(json.dumps(out_row) + '\n')
                n_included += 1
                continue

            filter_info, out_row = apply_filter_result(row, result)
            out_row['filter_result'] = filter_info.get('atoms', [])

            old_n = len(row['extracted'].get('reasoning', []))
            new_n = len(filter_info['filtered_reasoning'])
            n_atoms_before += old_n
            n_atoms_after += new_n

            if not filter_info['filtered_reasoning']:
                out_row['exclude_reason'] = 'no reasoning atoms after filter'
                exc_f.write(json.dumps(out_row) + '\n')
                n_excluded += 1
            else:
                inc_f.write(json.dumps(out_row) + '\n')
                n_included += 1

        # Also write rows that had no reasoning (skipped in batch)
        processed_nums = set()
        for br in batch_results:
            pos_num = int(br['custom_id'].split('-', 1)[1])
            processed_nums.add(pos_num)
        for pos_num, row in all_rows.items():
            if pos_num not in processed_nums:
                reasoning = row['extracted'].get('reasoning', [])
                if not reasoning:
                    row_copy = dict(row)
                    row_copy['exclude_reason'] = 'no reasoning atoms'
                    exc_f.write(json.dumps(row_copy) + '\n')
                    n_excluded += 1
                else:
                    inc_f.write(json.dumps(row) + '\n')
                    n_included += 1

    finally:
        inc_f.close()
        exc_f.close()

    print(f"\nDone. {n_included} included, {n_excluded} excluded, {n_errors} errors.")
    print(f"Atoms: {n_atoms_before} -> {n_atoms_after} "
          f"({n_atoms_before - n_atoms_after} net removed)")
    print(f"  Included -> {args.out_included}")
    print(f"  Excluded -> {args.out_excluded}")


def cmd_filter(args):
    """Filter extracted atoms synchronously (no batch). Useful for small runs."""
    import openai
    client = openai.OpenAI()

    with open(args.input) as f:
        all_rows = [json.loads(line) for line in f]
    print(f"Loaded {len(all_rows)} positions from {args.input}")

    # Resume
    already_done = set()
    for path in [args.out_included, args.out_excluded]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    already_done.add(json.loads(line)['position_number'])
    if already_done:
        print(f"Resuming: {len(already_done)} already done")

    n_processed = 0
    n_atoms_before = 0
    n_atoms_after = 0
    n_ctx = 0
    n_moved = 0
    n_excluded = 0

    for row in all_rows:
        if row['position_number'] in already_done:
            continue

        board = chess.Board(row['fen'])
        san = board.san(chess.Move.from_uci(row['move_uci']))

        result, out_row = filter_atoms(row, client, model=args.model)

        old_reasoning = row['extracted'].get('reasoning', [])
        new_reasoning = result['filtered_reasoning']
        n_before = len(old_reasoning)
        n_after = len(new_reasoning)
        n_atoms_before += n_before
        n_atoms_after += n_after
        n_ctx += sum(1 for a in result.get('atoms', [])
                     if a.get('action') == 'contextualize')
        n_moved += len(result.get('new_alternatives', []))

        out_row['filter_result'] = result.get('atoms', [])

        if not new_reasoning:
            out_row['exclude_reason'] = 'no reasoning atoms after filter'
            with open(args.out_excluded, 'a') as f:
                f.write(json.dumps(out_row) + '\n')
            n_excluded += 1
            print(f"  [{row['position_number']}] {row['game']} — {san}: "
                  f"EXCLUDED (0 reasoning atoms)")
        else:
            with open(args.out_included, 'a') as f:
                f.write(json.dumps(out_row) + '\n')

        n_processed += 1
        if n_processed % 50 == 0:
            print(f"  {n_processed}/{len(all_rows) - len(already_done)} done "
                  f"({n_atoms_before} -> {n_atoms_after} atoms)")
        elif n_before != n_after and new_reasoning:
            print(f"  [{row['position_number']}] {row['game']} — {san}: "
                  f"{n_before} -> {n_after}")

    print(f"\nDone. {n_processed} positions filtered.")
    print(f"Included: {n_processed - n_excluded} | Excluded: {n_excluded}")
    print(f"Atoms: {n_atoms_before} -> {n_atoms_after} "
          f"({n_atoms_before - n_atoms_after} net removed, "
          f"{(n_atoms_before - n_atoms_after) / max(n_atoms_before, 1) * 100:.1f}%)")
    print(f"  Contextualized: {n_ctx}, Moved to alternative: {n_moved}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract structured atoms from chess commentary.")
    sub = parser.add_subparsers(dest='command', required=True)

    # -- prepare --
    p_prep = sub.add_parser('prepare',
        help='Run Stockfish analysis, produce batch request + metadata files')
    p_prep.add_argument('--input', required=True,
                        help='Input JSONL (logical_chess.jsonl)')
    p_prep.add_argument('--batch-file', required=True,
                        help='Output batch request JSONL for OpenAI')
    p_prep.add_argument('--meta-file', required=True,
                        help='Output metadata JSONL (engine lines, wp_loss, etc.)')
    p_prep.add_argument('--model', default=DEFAULT_MODEL)
    p_prep.add_argument('--start', type=int, default=1,
                        help='1-indexed start position')
    p_prep.add_argument('--end', type=int, default=None,
                        help='1-indexed end position')
    p_prep.add_argument('--depth', type=int, default=22,
                        help='Stockfish depth')

    # -- submit --
    p_sub = sub.add_parser('submit',
        help='Upload batch file and submit to OpenAI Batch API')
    p_sub.add_argument('--batch-file', required=True,
                       help='Batch request JSONL from prepare step')

    # -- collect --
    p_col = sub.add_parser('collect',
        help='Check batch status and download results')
    p_col.add_argument('--batch-id', required=True,
                       help='Batch ID from submit step')
    p_col.add_argument('--output', required=True,
                       help='Where to save batch output JSONL')
    p_col.add_argument('--poll', action='store_true',
                       help='Poll until batch completes')
    p_col.add_argument('--poll-interval', type=int, default=60,
                       help='Seconds between polls (default: 60)')

    # -- process --
    p_proc = sub.add_parser('process',
        help='Join batch output + metadata, write included/excluded')
    p_proc.add_argument('--meta-file', required=True)
    p_proc.add_argument('--batch-output', required=True,
                        help='Batch output JSONL from collect step')
    p_proc.add_argument('--out-included', required=True)
    p_proc.add_argument('--out-excluded', required=True)
    p_proc.add_argument('--model', default=DEFAULT_MODEL,
                        help='Model name to record in output')
    p_proc.add_argument('--verify', action='store_true',
                        help='Run atom self-containment verification')
    p_proc.add_argument('--verify-model', default='gpt-4o-mini',
                        help='Model for verification (default: gpt-4o-mini)')

    # -- sync --
    p_sync = sub.add_parser('sync',
        help='Run extraction synchronously (no batch, direct API)')
    p_sync.add_argument('--input', required=True)
    p_sync.add_argument('--out-included', required=True)
    p_sync.add_argument('--out-excluded', required=True)
    p_sync.add_argument('--model', default=DEFAULT_MODEL)
    p_sync.add_argument('--start', type=int, default=1)
    p_sync.add_argument('--end', type=int, default=None)
    p_sync.add_argument('--depth', type=int, default=22)
    p_sync.add_argument('--verify', action='store_true')

    # -- filter-prepare --
    p_fp = sub.add_parser('filter-prepare',
        help='Build batch request JSONL for filter pass')
    p_fp.add_argument('--input', required=True,
                      help='Input JSONL (included.jsonl)')
    p_fp.add_argument('--batch-file', required=True,
                      help='Output batch request JSONL')
    p_fp.add_argument('--model', default=DEFAULT_MODEL)

    # -- filter-process --
    p_fproc = sub.add_parser('filter-process',
        help='Join filter batch output with original rows')
    p_fproc.add_argument('--input', required=True,
                         help='Original input JSONL (same as filter-prepare --input)')
    p_fproc.add_argument('--batch-output', required=True,
                         help='Batch output JSONL from collect step')
    p_fproc.add_argument('--out-included', required=True)
    p_fproc.add_argument('--out-excluded', required=True)

    # -- filter (sync) --
    p_filt = sub.add_parser('filter',
        help='Filter extracted atoms synchronously (no batch)')
    p_filt.add_argument('--input', required=True,
                        help='Input JSONL (included.jsonl from process step)')
    p_filt.add_argument('--out-included', required=True,
                        help='Output JSONL for positions with reasoning atoms')
    p_filt.add_argument('--out-excluded', required=True,
                        help='Output JSONL for positions with 0 reasoning atoms')
    p_filt.add_argument('--model', default=DEFAULT_MODEL,
                        help='Model for filter LLM (default: gpt-5.4)')

    args = parser.parse_args()

    if args.command == 'prepare':
        cmd_prepare(args)
    elif args.command == 'submit':
        cmd_submit(args)
    elif args.command == 'collect':
        cmd_collect(args)
    elif args.command == 'process':
        cmd_process(args)
    elif args.command == 'sync':
        cmd_sync(args)
    elif args.command == 'filter-prepare':
        cmd_filter_prepare(args)
    elif args.command == 'filter-process':
        cmd_filter_process(args)
    elif args.command == 'filter':
        cmd_filter(args)


if __name__ == '__main__':
    main()
