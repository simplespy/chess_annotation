"""
Chess Commentary Judge: Decomposition, Verification, and Evaluation Pipeline

This module provides comprehensive evaluation of chess move explanations through:
1. Decomposition: Breaking explanations into atomic, verifiable claims
2. Verification: Tool-augmented fact-checking of each claim
3. Matching: Comparing generated atoms against gold standard
4. Scoring: Multi-dimensional quality assessment (accuracy, recall, fluency, specificity)

The pipeline is designed to evaluate both factual correctness and coverage
of important chess concepts, with programmatic sanity checks to catch LLM errors.
"""

import json
import re
import chess
from typing import List, Dict, Tuple, Optional

from llm_client import call_llm, call_with_tools
from chess_tools import execute_tool
from scoring import score_fluency, score_specificity, compute_composite_score


# ============================================================================
# PROMPT CONSTANTS: DECOMPOSITION
# ============================================================================

DECOMPOSE_SYSTEM = """\
Extract atomic factual claims from a chess move explanation.

CRITICAL: ONE CLAIM PER ATOM. Each atom must contain exactly ONE verifiable fact. \
If a sentence contains multiple claims, split it into separate atoms.

EXAMPLE — given this text:
"Bf3 allows Black to play Bf5, skewering the queen on b1 and the knight on c3."

BAD (compound claims):
- "Bf3 allows Black to play Bf5, skewering the queen on b1 and the knight on c3."

GOOD (atomic claims - one fact each):
- "Bf3 would allow Black to play Bf5"
- "After Bf3 Bf5, the white queen is on b1"
- "After Bf3 Bf5, there is a white knight on c3"
- "After Bf3 Bf5, the bishop on f5 skewers the queen on b1 and knight on c3"

CONTEXTUAL ATOMS: Each atom must be self-contained with full positional context. \
If an atom references a position after a sequence of moves, include the FULL \
preceding move sequence.

EXAMPLE — given this text:
"Nxd7 removes one of the defenders of the knight on f6. After the expected Nh5, \
White can capture the bishop on g6."

BAD (later atoms lose context):
- "Nxd7 removes one of the defenders of the knight on f6."
- "After Nh5, White can capture the bishop on g6."  <- Nh5 from where?

GOOD (each carries full move sequence from root):
- "Nxd7 removes one of the defenders of the knight on f6."
- "After Nxd7 Nh5, White can capture the bishop on g6."

EXAMPLE 2 — given text about alternative moves:
"Instead, White should play Nd5, attacking the bishop and threatening Nxf6, which would disrupt Black's pawn structure."

BAD (loses context of the alternative move):
- "Nd5 attacks the bishop."
- "Nxf6 would disrupt Black's pawn structure."  <- From where? After which moves?

GOOD (includes full context):
- "If White plays Nd5 instead, the knight would attack the bishop on e7."
- "After Nd5 Nxf6, Black's pawn structure would be disrupted."

CRITICAL: For consequences of move sequences, ALWAYS include the complete sequence.

RULES:
1. ONE claim per atom - split compound claims into separate atoms
2. Be specific — include piece names, squares, and concrete assertions
3. Group logically connected setup+consequence ONLY if they're inseparable
4. Each atom must include the FULL move sequence from the root position
5. Ignore stylistic flourishes that make no factual claim
6. Ignore generic chess philosophy not applied to this position

Output valid JSON with claims grouped by logical assertions:
{
  "claims": [
    {
      "claim_text": "Complete sentence(s) making this assertion",
      "atoms": [
        "Atomic fact 1",
        "Atomic fact 2",
        "Atomic fact 3"
      ]
    },
    {
      "claim_text": "Another logical assertion",
      "atoms": ["Atomic fact 4"]
    }
  ]
}

EXAMPLE OUTPUT:
{
  "claims": [
    {
      "claim_text": "Bf3 is a blunder because it allows Black to play Bf5, skewering the queen on b1 and the rook on d1.",
      "atoms": [
        "Bf3 is a blunder",
        "Bf3 allows Black to play Bf5",
        "After Bf3 Bf5, the white queen is on b1",
        "After Bf3 Bf5, the white rook is on d1",
        "After Bf3 Bf5, the bishop on f5 skewers the queen on b1 and rook on d1"
      ]
    },
    {
      "claim_text": "Instead, White should have played Nd5, attacking the bishop on e7.",
      "atoms": [
        "If White plays Nd5 instead, the knight would attack the bishop on e7"
      ]
    }
  ]
}"""


# ============================================================================
# PROMPT CONSTANTS: VERIFICATION
# ============================================================================

VERIFY_SINGLE_SYSTEM = """\
You are a chess fact-checker verifying a SINGLE atomic claim about a chess position.

POSITION CONTEXT:
- Pre-move FEN: {pre_fen}
- Move played: {move_san} (UCI: {move_uci})
- Post-move FEN: {post_fen}
- Side that played the move: {side_to_move}

CLAIM TO VERIFY:
{atom}

VERIFICATION RULES:

1. MANDATORY TOOL USE: A verdict without any tool call is INVALID. You MUST call \
at least one tool to verify this claim. Do not rely on your own chess analysis.

2. EVAL_MOVE INTERPRETATION: When using eval_move, the wp_loss value represents \
the win-percentage lost BY THE SIDE PLAYING THE MOVE, not the other side.
   - wp_loss < 5: good move
   - wp_loss 5-20: suboptimal/inaccuracy
   - wp_loss > 20: blunder

3. WHICH FEN TO USE:
   - Claims about the position AFTER the move -> use post-move FEN
   - Claims about what existed BEFORE the move -> use pre-move FEN
   - Claims about alternative moves -> use pre-move FEN
   - Variations from the post-move position -> use post-move FEN

4. For COMPARISON claims ("X is better than Y", "instead of Y"):
   You MUST evaluate BOTH moves. Use eval_move or compare_moves on both.

5. For VARIATION claims ("after X Y Z..."):
   Use try_variation to test the full sequence.

6. For PIECE PLACEMENT claims ("piece on X"):
   Use get_piece_at to verify.

7. For claims about ALTERNATIVE MOVES (e.g., "Nd5 attacks..." when discussing what White should have played instead):
   - The atom may reference pieces or positions that don't exist in pre/post-move FENs
   - First try playing the alternative move from pre-move FEN using try_variation
   - Then verify the claim in the resulting position
   - Check that the alternative move is legal before verifying its consequences

8. For THREAT claims (e.g., "threatens Qg7 mate", "threatens to win the bishop"):
   - Use the check_threat tool to verify if the threat is real
   - The threat is a move the side NOT to move wants to play on their next turn
   - check_threat will test all opponent responses and report if threat works

9. For MULTI-MOVE PLAN claims (e.g., "plans c3, d4, d5" or "wants to play Nf3, Bg5, Qd3"):
   - These are strategic claims that cannot be verified with tools
   - Mark as unverifiable: {{"verified": false, "confidence": "n/a", "reasoning": "Multi-move plan: tool-unverifiable strategic claim"}}
   - Plans involving 2+ moves by the same side require strategic judgment, not tool verification

{type_specific_guidance}

OUTPUT: Respond with valid JSON only:
{{"verified": true/false, "confidence": "high"/"medium"/"low", "reasoning": "..."}}

Your reasoning MUST cite specific tool results."""


TYPE_GUIDANCE = {
    'quality': """\
QUALITY CHECK GUIDANCE:
- Use eval_move(pre_fen, move) to get wp_loss
- wp_loss < 5 = good, 5-20 = suboptimal, > 20 = blunder
- If the atom says "good move" but wp_loss > 15, it's WRONG
- If the atom says "blunder" but wp_loss < 10, it's WRONG""",

    'comparison': """\
COMPARISON CHECK GUIDANCE:
- You MUST evaluate BOTH moves being compared using eval_move or compare_moves
- Compare their wp_loss values to determine which is actually better
- Do NOT conclude one move is better without checking both""",

    'material': """\
MATERIAL CHECK GUIDANCE:
- Use get_material or get_piece_at to verify material claims
- For captures, compare material before and after using both FENs
- Use compare_positions to compare material between two positions""",

    'tactic': """\
TACTIC CHECK GUIDANCE:
- For pins: use is_pinned(fen, square)
- For forks: check get_attacks on the forking piece
- For skewers: use is_pinned (skewers show as pins on higher-value pieces)
- For discovered attacks: play the move with try_variation, then check attacks""",

    'piece_placement': """\
PIECE PLACEMENT GUIDANCE:
- Use get_piece_at(fen, square) to verify exact piece locations
- For "piece on [square]" claims, check the CORRECT fen (pre or post move)
- The UCI notation shows from->to: e.g. e2f3 means piece was on e2, now on f3
- If atom mentions a piece on a square that doesn't exist in pre/post FENs (e.g., "Nd5 attacks..." when no N on d5), it likely refers to an alternative move. Use try_variation to play that move from pre-move FEN, then verify in the result""",

    'threat': """\
THREAT CHECK GUIDANCE:
- Use check_threat(fen, threat_move) to verify if the threat is real
- The threat is a move the side NOT to move wants to play on their next turn
- check_threat will test all opponent responses and report success rate
- If check_threat shows threat_viable: true with high success rate, threat is real
- If check_threat shows many defenses, mention them in reasoning""",

    'plan': """\
PLAN CHECK GUIDANCE:
- Multi-move plans (e.g., "c3, d4, d5" or "Nf3, Bg5, Qd3") cannot be tool-verified
- These require strategic judgment beyond tool capabilities
- Mark as unverifiable: {{"verified": false, "confidence": "n/a", "reasoning": "Multi-move plan: tool-unverifiable strategic claim"}}
- Single-move plans (e.g., "plans Qg7 mate next") can use check_threat""",

    'defender': """\
DEFENDER CHECK GUIDANCE:
- Use count_attackers_defenders on BOTH pre-move and post-move FENs
- Compare defender counts to verify "removes a defender" claims
- Check both the claimed square and the defending piece's square""",

    'positional': """\
POSITIONAL CHECK GUIDANCE:
- For diagonal/file claims: use check_ray_alignment or get_attacks
- For pawn structure: use get_squares(fen, "pawn", color)
- For control claims: use get_attacks or count_attackers_defenders""",

    'move_sequence': """\
MOVE SEQUENCE GUIDANCE:
- Use try_variation with the FULL move sequence
- Check that all moves are legal
- Verify the claimed consequence in the resulting position""",
}


# ============================================================================
# PROMPT CONSTANTS: ALTERNATIVES
# ============================================================================

EXTRACT_ALTERNATIVES_SYSTEM = """Analyze a chess move explanation and extract any ALTERNATIVE MOVES mentioned.

For each alternative move mentioned:
1. Identify the move in SAN notation (e.g., "Nd5", "Bxf6")
2. Determine the STANCE: how is this alternative presented relative to the played move?
   - "better": explicitly stated as superior ("should have", "instead", "better")
   - "worse": presented as inferior ("but", "however", "not as good")
   - "neutral": just mentioned without comparison
3. Extract the variation if given (e.g., "Nd5 Nxf6 Bxf6")
4. Identify which atoms (by index) discuss this alternative

Output valid JSON:
{
  "alternatives": [
    {
      "move": "Nd5",
      "stance": "better",
      "variation": "Nd5 Nxf6",
      "atom_indices": [4, 5, 6, 7]
    }
  ]
}

If no alternatives mentioned, return {"alternatives": []}."""


# ============================================================================
# PROMPT CONSTANTS: MATCHING
# ============================================================================

MATCH_SYSTEM = """\
Compare two sets of atomic chess claims.

Given CANDIDATE atoms (from a generated explanation) and GOLD atoms \
(from an expert annotation), determine which gold atoms are COVERED \
by the candidate atoms.

A gold atom is "covered" if any candidate atom expresses the same \
factual insight, even if worded differently. Partial coverage counts \
if the core insight is present.

Output valid JSON:
{"results": [{"gold_atom": "...", "covered": true, \
"matching_candidate": "the matching candidate text"}, ...]}

Use null for matching_candidate when covered is false."""


# ============================================================================
# DECOMPOSITION FUNCTIONS
# ============================================================================

def _parse_judge_json(text: str) -> dict:
    """
    Parse JSON from judge LLM output, handling markdown fences.

    Args:
        text: Raw LLM response text

    Returns:
        Parsed dict, or empty dict if parsing fails
    """
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"  [JSON parse error] {text[:200]}...")
        return {}


def decompose_to_atoms(text: str, provider: str = "openai",
                      model: str = "gpt-4o",
                      base_url: Optional[str] = None,
                      api_key: Optional[str] = None) -> Tuple[List[str], List[dict]]:
    """
    Decompose NL explanation into claims and atoms.

    The function uses an LLM to break down complex explanations into:
    - Claims: High-level logical assertions
    - Atoms: Individual verifiable facts within each claim

    Args:
        text: Explanation text to decompose
        provider: LLM provider ('openai', 'anthropic', or 'qwen')
        model: Model name
        base_url: Base URL for OpenAI-compatible APIs
        api_key: API key (optional)

    Returns:
        Tuple of (atoms_list, claims_list)
        where claims_list = [
            {'claim_text': '...', 'atom_indices': [0, 1, 2], 'atoms': [...]},
            ...
        ]
    """
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM},
        {"role": "user", "content": text},
    ]
    resp, _ = call_llm(messages, provider=provider, model=model, temperature=0.0,
                       base_url=base_url, api_key=api_key)
    data = _parse_judge_json(resp.text)

    # Flatten atoms and build claims structure
    atoms = []
    claims = []

    for claim_data in data.get('claims', []):
        claim_atoms = claim_data.get('atoms', [])
        start_idx = len(atoms)
        atoms.extend(claim_atoms)
        end_idx = len(atoms)

        claims.append({
            'claim_text': claim_data.get('claim_text', ''),
            'atom_indices': list(range(start_idx, end_idx)),
            'atoms': claim_atoms,  # Keep for reference
        })

    # Fallback: if no claims structure, treat each atom as a separate claim
    if not claims and 'atoms' in data:
        atoms = data['atoms']
        claims = [
            {'claim_text': atom, 'atom_indices': [i], 'atoms': [atom]}
            for i, atom in enumerate(atoms)
        ]

    return atoms, claims


def classify_atom(atom_text: str) -> List[str]:
    """
    Regex-based classification of an atom into semantic types.

    Classifies atoms into categories like:
    - quality: Move quality assessment
    - comparison: Comparing alternatives
    - material: Material balance claims
    - tactic: Tactical motifs (pins, forks, etc.)
    - positional: Strategic concepts
    - threat: Threats and attacking moves
    - plan: Multi-move plans
    - etc.

    Args:
        atom_text: Single atomic claim text

    Returns:
        List of type strings (can have multiple types)
    """
    text = atom_text.lower()
    types = []

    if re.search(r'\b(good|bad|blunder|mistake|inaccuracy|dubious|best move|excellent|strong move|weak move|poor|brilliant)\b', text):
        types.append('quality')

    if re.search(r'\b(better|worse|instead|rather than|compared to|alternative|superior|inferior|preferable)\b', text):
        types.append('comparison')

    if re.search(r'\b(wins?\s+(material|a?\s*piece|a?\s*pawn|the exchange)|sacrifice|captures?|takes|recapture|material\s+(advantage|gain|loss))\b', text):
        types.append('material')

    if re.search(r'\b(after\s+\S+\s+\S+|variation|line|sequence|follows|continuation)\b', text) or text.count('...') > 0:
        types.append('move_sequence')

    if re.search(r'\b(fork|pins?|skewer|discovered|double attack|back rank|mate|checkmate|trap|tactic|combination)\b', text):
        types.append('tactic')

    if re.search(r'\b(outpost|weak square|pawn structure|open file|diagonal|fianchetto|isolated|doubled|passed|backward|space|control|center|centre)\b', text):
        types.append('positional')

    if re.search(r'\b(on\s+[a-h][1-8]|to\s+[a-h][1-8]|places|develops|moves?\s+to|lands?\s+on|from\s+[a-h][1-8])\b', text):
        types.append('piece_placement')

    if re.search(r'\b(threatens?|threatening|attacks?|targeting)\b', text):
        types.append('threat')

    # Detect multi-move plans
    if re.search(r'\b(plans?|wants? to play|intends?|idea|followed by)\b', text):
        if re.search(r'[a-h][1-8].*[a-h][1-8].*[a-h][1-8]', text) or \
           re.search(r'\b(then|followed by|and then|subsequently)\b', text):
            types.append('plan')

    if re.search(r'\b(defends?|defender|protects?|guards?|removes?\s.*defender|undefended|unprotected)\b', text):
        types.append('defender')

    if not types:
        types.append('general')

    return types


# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================

def verify_single_atom(atom: str, fen: str, move_san: str, move_uci: str,
                      provider: str = "openai", model: str = "gpt-4o",
                      verbose: bool = True,
                      base_url: Optional[str] = None,
                      api_key: Optional[str] = None) -> dict:
    """
    Verify a single atom with a fresh tool-calling conversation.

    This is the core verification function that:
    1. Classifies the atom type
    2. Builds a verification prompt with type-specific guidance
    3. Calls LLM with tool access to verify
    4. Returns structured verification result

    Args:
        atom: Atomic claim text to verify
        fen: Pre-move FEN position
        move_san: Move in SAN notation
        move_uci: Move in UCI notation
        provider: LLM provider ('openai', 'anthropic', or 'qwen')
        model: Model name
        verbose: Print progress
        base_url: Base URL for OpenAI-compatible APIs
        api_key: API key (optional)

    Returns:
        Dict with:
        - atom: Original atom text
        - verified: Boolean verification result
        - confidence: 'high'/'medium'/'low'/'n/a'
        - reasoning: Explanation from LLM
        - atom_types: List of classified types
        - tool_log: Complete tool call log
        - tool_summary: Abbreviated tool call summary
    """
    board = chess.Board(fen)
    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    board.push(chess.Move.from_uci(move_uci))
    post_fen = board.fen()

    atom_types = classify_atom(atom)

    # Build type-specific guidance
    guidance_parts = []
    for t in atom_types:
        if t in TYPE_GUIDANCE:
            guidance_parts.append(TYPE_GUIDANCE[t])
    type_guidance = "\n\n".join(guidance_parts) if guidance_parts else ""

    system_msg = VERIFY_SINGLE_SYSTEM.format(
        pre_fen=fen, post_fen=post_fen,
        move_san=move_san, move_uci=move_uci,
        side_to_move=side_to_move, atom=atom,
        type_specific_guidance=type_guidance,
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Verify this claim: {atom}"},
    ]

    if verbose:
        print(f"    Verifying: {atom[:80]}...")

    text, tool_log = call_with_tools(
        messages, provider=provider, model=model,
        temperature=0.0, max_rounds=10, verbose=verbose,
        base_url=base_url, api_key=api_key)

    data = _parse_judge_json(text)

    return {
        'atom': atom,
        'verified': data.get('verified', False),
        'confidence': data.get('confidence', 'low'),
        'reasoning': data.get('reasoning', text),
        'atom_types': atom_types,
        'tool_log': tool_log,
        'tool_summary': [{'tool': t['tool'], 'args': t['args']} for t in tool_log],
    }


def run_sanity_checks(verify_result: dict, fen: str, move_uci: str,
                     is_alternative_atom: bool = False) -> dict:
    """
    Programmatic post-hoc checks on a verified atom. Can override LLM verdict.

    Performs automated consistency checks:
    1. No tool calls -> force verified=False
    2. Quality wp_loss consistency (good move can't have high wp_loss)
    3. Comparison: both moves checked (SKIPPED if is_alternative_atom=True)
    4. Variation legality (illegal moves -> false)
    5. Piece placement contradiction

    Args:
        verify_result: Result dict from verify_single_atom
        fen: Pre-move FEN
        move_uci: Move in UCI notation
        is_alternative_atom: If True, skip comparison check (atom describes alternative position)

    Returns:
        Dict with:
        - overrides: List of override checks triggered
        - original_verified: Original LLM verdict
        - final_verified: Final verdict after overrides
        - was_overridden: Whether verdict changed
    """
    overrides = []
    atom = verify_result['atom']
    tool_log = verify_result['tool_log']
    atom_types = verify_result['atom_types']
    original_verified = verify_result['verified']

    # 1. No tool calls -> force False
    if not tool_log:
        overrides.append({
            'check': 'no_tool_calls',
            'message': 'No tool calls made — verdict unsupported',
            'action': 'override_false',
        })

    # 2. Quality wp_loss consistency
    if 'quality' in atom_types:
        for entry in tool_log:
            if entry['tool'] == 'eval_move':
                try:
                    result = json.loads(entry['result'])
                    wp_loss = result.get('wp_loss', 0)
                    atom_lower = atom.lower()

                    if any(w in atom_lower for w in ('good move', 'strong move', 'excellent', 'best move', 'brilliant')):
                        if wp_loss > 15:
                            overrides.append({
                                'check': 'quality_wp_loss',
                                'message': f'Atom claims good move but wp_loss={wp_loss}',
                                'action': 'override_false',
                            })

                    if any(w in atom_lower for w in ('blunder', 'mistake', 'bad move', 'weak move', 'poor')):
                        if wp_loss < 5:
                            overrides.append({
                                'check': 'quality_wp_loss',
                                'message': f'Atom claims bad move but wp_loss={wp_loss}',
                                'action': 'override_false',
                            })
                except (json.JSONDecodeError, KeyError):
                    pass

    # 3. Comparison: both moves checked (skip for alternative atoms)
    if 'comparison' in atom_types and not is_alternative_atom:
        eval_entries = [e for e in tool_log if e['tool'] in ('eval_move', 'compare_moves')]
        if len(eval_entries) < 1:
            overrides.append({
                'check': 'comparison_incomplete',
                'message': 'Comparison claim but no move evaluation performed',
                'action': 'override_false',
            })
        elif not any(e['tool'] == 'compare_moves' for e in eval_entries):
            eval_move_calls = [e for e in tool_log if e['tool'] == 'eval_move']
            evaluated_moves = set()
            for e in eval_move_calls:
                evaluated_moves.add(e['args'].get('move', ''))
            if len(evaluated_moves) < 2:
                overrides.append({
                    'check': 'comparison_one_side',
                    'message': f'Comparison claim but only {len(evaluated_moves)} move(s) evaluated',
                    'action': 'override_false',
                })

    # 4. Variation legality
    for entry in tool_log:
        if entry['tool'] == 'try_variation':
            try:
                result = json.loads(entry['result'])
                if not result.get('legal', True):
                    if original_verified:
                        overrides.append({
                            'check': 'variation_illegal',
                            'message': f'try_variation returned illegal: {result.get("error", "")}',
                            'action': 'override_false',
                        })
            except (json.JSONDecodeError, KeyError):
                pass

    # 5. Piece placement contradiction
    if 'piece_placement' in atom_types:
        for entry in tool_log:
            if entry['tool'] == 'get_piece_at':
                try:
                    result = json.loads(entry['result'])
                    if result.get('piece') == 'empty':
                        sq = result.get('square', '')
                        if sq and sq in atom.lower() and original_verified:
                            overrides.append({
                                'check': 'piece_not_found',
                                'message': f'get_piece_at({sq}) returned empty but atom claims piece there',
                                'action': 'override_false',
                            })
                except (json.JSONDecodeError, KeyError):
                    pass

    # Apply overrides
    final_verified = original_verified
    if any(o['action'] == 'override_false' for o in overrides):
        final_verified = False

    return {
        'overrides': overrides,
        'original_verified': original_verified,
        'final_verified': final_verified,
        'was_overridden': final_verified != original_verified,
    }


def verify_atoms_improved(atoms: List[str], fen: str, move_san: str,
                         move_uci: str, alternatives: Optional[List[dict]] = None,
                         provider: str = "openai", model: str = "gpt-4o",
                         verbose: bool = True,
                         base_url: Optional[str] = None,
                         api_key: Optional[str] = None) -> Tuple[List[dict], str, List[dict]]:
    """
    Per-atom verification with sanity checks.

    Orchestrates verification of multiple atoms:
    1. Identifies which atoms are about alternatives
    2. Verifies each atom individually
    3. Runs sanity checks on each verification
    4. Derives overall quality assessment

    Args:
        atoms: List of atomic claim texts
        fen: Pre-move FEN
        move_san: Move in SAN notation
        move_uci: Move in UCI notation
        alternatives: List of alternative moves (from extract_alternatives). Used to identify
                      which atoms are about alternatives (skip comparison sanity check).
        provider: LLM provider ('openai', 'anthropic', or 'qwen')
        model: Model name
        verbose: Print progress
        base_url: Base URL for OpenAI-compatible APIs
        api_key: API key (optional)

    Returns:
        Tuple of (results_list, assessment, all_tool_logs)
        - results_list: List of verification dicts for each atom
        - assessment: Overall quality assessment ('good'/'bad'/'inconclusive')
        - all_tool_logs: Combined tool logs from all verifications
    """
    if not atoms:
        return [], 'inconclusive', []

    # Build set of alternative-related atom indices
    alt_atom_indices = set()
    if alternatives:
        for alt in alternatives:
            alt_atom_indices.update(alt.get('atom_indices', []))

    results = []
    all_tool_logs = []

    for i, atom in enumerate(atoms):
        if verbose:
            print(f"  [{i+1}/{len(atoms)}]")

        v_result = verify_single_atom(
            atom, fen, move_san, move_uci,
            provider=provider, model=model, verbose=verbose,
            base_url=base_url, api_key=api_key)

        # Check if this atom is about an alternative (skip comparison check)
        is_alt_atom = i in alt_atom_indices
        sanity = run_sanity_checks(v_result, fen, move_uci, is_alternative_atom=is_alt_atom)

        v_result['verified'] = sanity['final_verified']
        v_result['sanity_overrides'] = sanity['overrides']
        v_result['was_overridden'] = sanity['was_overridden']

        results.append(v_result)
        all_tool_logs.extend(v_result['tool_log'])

        if verbose and sanity['was_overridden']:
            for o in sanity['overrides']:
                print(f"    SANITY OVERRIDE: {o['message']}")

    # Derive overall assessment from quality-type atoms
    quality_atoms = [r for r in results if 'quality' in r.get('atom_types', [])]
    if quality_atoms:
        verified_quality = [r for r in quality_atoms if r['verified']]
        if verified_quality:
            any_bad = any(
                any(w in r['atom'].lower() for w in ('bad', 'blunder', 'mistake', 'weak', 'poor', 'dubious'))
                for r in verified_quality)
            any_good = any(
                any(w in r['atom'].lower() for w in ('good', 'strong', 'excellent', 'best', 'brilliant'))
                for r in verified_quality)
            if any_bad:
                assessment = 'bad'
            elif any_good:
                assessment = 'good'
            else:
                assessment = 'inconclusive'
        else:
            assessment = 'inconclusive'
    else:
        assessment = 'inconclusive'

    return results, assessment, all_tool_logs


# ============================================================================
# MATCHING FUNCTIONS
# ============================================================================

def match_gold_atoms(candidate_atoms: List[str], gold_atoms: List[str],
                    provider: str = "openai", model: str = "gpt-4o",
                    base_url: Optional[str] = None,
                    api_key: Optional[str] = None) -> List[dict]:
    """
    Check which gold atoms are covered by candidate atoms.

    Uses LLM to compare generated atoms against gold standard atoms,
    determining semantic coverage (not just exact matching).

    Args:
        candidate_atoms: List of generated atomic claims
        gold_atoms: List of gold standard atomic claims
        provider: LLM provider ('openai', 'anthropic', or 'qwen')
        model: Model name
        base_url: Base URL for OpenAI-compatible APIs
        api_key: API key (optional)

    Returns:
        List of matching results, one per gold atom:
        [{'gold_atom': '...', 'covered': True/False, 'matching_candidate': '...'}, ...]
    """
    if not gold_atoms:
        return []
    if not candidate_atoms:
        return [{"gold_atom": g, "covered": False, "matching_candidate": None}
                for g in gold_atoms]

    user_msg = (
        f"Candidate atoms:\n{json.dumps(candidate_atoms, indent=2)}\n\n"
        f"Gold atoms:\n{json.dumps(gold_atoms, indent=2)}"
    )
    messages = [
        {"role": "system", "content": MATCH_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    resp, _ = call_llm(messages, provider=provider, model=model, temperature=0.0,
                       base_url=base_url, api_key=api_key)
    data = _parse_judge_json(resp.text)
    return data.get('results', [])


def check_quality_improved(atoms_results: List[dict], fen: str,
                          move_uci: str, wp_loss: float) -> dict:
    """
    Derive quality assessment from verified atoms and compare to ground truth.

    Determines:
    1. Actual quality from wp_loss (ground truth)
    2. Generated assessment from quality-type atoms
    3. Whether generated assessment matches actual quality

    Args:
        atoms_results: List of atom verification results
        fen: Position FEN (unused, for consistency)
        move_uci: Move UCI (unused, for consistency)
        wp_loss: Win percentage loss (ground truth)

    Returns:
        Dict with:
        - quality_correct: Boolean, does gen_assessment match actual?
        - actual_quality: Ground truth quality level
        - gen_assessment: What the explanation claimed
        - engine_wp_loss: wp_loss from verification tools (or input wp_loss)
    """
    # Ground truth
    if wp_loss > 30:     actual_quality = 'blunder'
    elif wp_loss > 20:   actual_quality = 'mistake'
    elif wp_loss > 10:   actual_quality = 'inaccuracy'
    else:                actual_quality = 'good'

    # Generated assessment from quality-type atoms
    # Look at ALL quality atoms (not just verified) to determine what the generator CLAIMED
    quality_atoms = [r for r in atoms_results if 'quality' in r.get('atom_types', [])]

    if not quality_atoms:
        gen_assessment = 'inconclusive'
    else:
        # Check what the generator claimed (regardless of verification)
        any_bad = any(
            any(w in r['atom'].lower() for w in ('bad', 'blunder', 'mistake', 'weak', 'poor'))
            for r in quality_atoms)
        any_good = any(
            any(w in r['atom'].lower() for w in ('good', 'strong', 'excellent', 'best'))
            for r in quality_atoms)
        gen_assessment = 'bad' if any_bad else ('good' if any_good else 'inconclusive')

    # Correctness
    if actual_quality in ('blunder', 'mistake'):
        quality_correct = (gen_assessment == 'bad')
    elif actual_quality == 'inaccuracy':
        quality_correct = True  # grey area
    else:
        quality_correct = (gen_assessment in ('good', 'inconclusive'))

    # Extract engine wp_loss from tool logs if available
    engine_wp_loss = None
    for r in quality_atoms:
        for t in r.get('tool_log', []):
            if t['tool'] == 'eval_move':
                try:
                    result = json.loads(t['result'])
                    engine_wp_loss = result.get('wp_loss')
                except Exception:
                    pass

    return {
        'quality_correct': quality_correct,
        'actual_quality': actual_quality,
        'gen_assessment': gen_assessment,
        'engine_wp_loss': engine_wp_loss or wp_loss,
    }


# ============================================================================
# ALTERNATIVE MOVES FUNCTIONS
# ============================================================================

def extract_alternatives(explanation: str, atoms: List[str], fen: str,
                        move_san: str, provider: str = "openai",
                        model: str = "gpt-4o",
                        base_url: Optional[str] = None,
                        api_key: Optional[str] = None) -> List[dict]:
    """
    Extract alternative moves mentioned in the explanation.

    Identifies and evaluates alternative moves discussed in the commentary:
    1. Parses explanation to find alternative move mentions
    2. Determines stance (better/worse/neutral)
    3. Maps atoms that discuss each alternative

    Note: This version returns basic alternatives without engine evaluation.
    For full engine enrichment, integrate with chess.engine module.

    Args:
        explanation: Full explanation text
        atoms: List of atomic claims
        fen: Position FEN
        move_san: Played move in SAN
        provider: LLM provider ('openai', 'anthropic', or 'qwen')
        model: Model name
        base_url: Base URL for OpenAI-compatible APIs
        api_key: API key (optional)

    Returns:
        List of dicts with:
        - move: SAN of alternative
        - stance: 'better' | 'worse' | 'neutral'
        - variation: full variation string if provided
        - atom_indices: which atoms discuss this alternative
    """
    # Build context for LLM
    atoms_text = "\n".join(f"{i}. {atom}" for i, atom in enumerate(atoms))
    user_msg = (
        f"Explanation: {explanation}\n\n"
        f"Atoms:\n{atoms_text}\n\n"
        f"Current position (FEN): {fen}\n"
        f"Move played: {move_san}"
    )

    messages = [
        {"role": "system", "content": EXTRACT_ALTERNATIVES_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    resp, _ = call_llm(messages, provider=provider, model=model, temperature=0.0,
                       base_url=base_url, api_key=api_key)
    data = _parse_judge_json(resp.text)
    alternatives_list = data.get('alternatives', [])

    if not alternatives_list:
        return []

    # Parse moves to UCI and add to alternatives
    board = chess.Board(fen)
    enriched = []

    for alt in alternatives_list:
        try:
            # Try to parse the move
            move_san = alt['move']
            move = board.parse_san(move_san)
            alt['move_uci'] = move.uci()
            alt['eval_diff'] = 0.0  # Placeholder
            enriched.append(alt)
        except (ValueError, KeyError):
            # Skip unparseable moves
            continue

    return enriched


def verify_alternative_atoms(alternative: dict, atoms: List[str], fen: str,
                            move_san: str, verification_results: List[dict],
                            provider: str = "openai", model: str = "gpt-4o",
                            verbose: bool = True,
                            base_url: Optional[str] = None,
                            api_key: Optional[str] = None):
    """
    Verify atoms that discuss an alternative move in that alternative's context.

    Re-verifies atoms using the position after the alternative move, which allows
    verification of claims like "Nd5 attacks the bishop" by actually playing Nd5.
    Updates verification_results in place with improved verdicts.

    Args:
        alternative: Alternative move dict from extract_alternatives
        atoms: Full list of atomic claims
        fen: Pre-move FEN
        move_san: Played move in SAN
        verification_results: List of verification dicts (modified in place)
        provider: LLM provider ('openai', 'anthropic', or 'qwen')
        model: Model name
        verbose: Print progress
        base_url: Base URL for OpenAI-compatible APIs
        api_key: API key (optional)
    """
    if not alternative.get('atom_indices'):
        return

    if verbose:
        print(f"  Re-verifying {len(alternative['atom_indices'])} atoms for alternative {alternative['move']}...")

    # Create the alternative position
    board = chess.Board(fen)
    alt_move = chess.Move.from_uci(alternative['move_uci'])
    board.push(alt_move)
    alt_fen = board.fen()
    alt_move_san = alternative['move']

    # Re-verify each atom in this context
    for idx in alternative['atom_indices']:
        if idx >= len(atoms):
            continue

        atom = atoms[idx]

        if verbose:
            print(f"    [{idx}] Re-verifying with alternative context...")

        # Verify in alternative context
        v_result = verify_single_atom(
            atom, fen, alt_move_san, alternative['move_uci'],
            provider=provider, model=model, verbose=False,
            base_url=base_url, api_key=api_key)

        sanity = run_sanity_checks(v_result, fen, alternative['move_uci'])

        # Update the verification result
        verification_results[idx]['verified_alt'] = sanity['final_verified']
        verification_results[idx]['alt_context'] = alternative['move']
        verification_results[idx]['alt_reasoning'] = v_result['reasoning']
        verification_results[idx]['alt_sanity_overrides'] = sanity['overrides']
        verification_results[idx]['alt_was_overridden'] = sanity['was_overridden']

        # If it's verified in alternative context but not in main context, note this
        if sanity['final_verified'] and not verification_results[idx]['verified']:
            verification_results[idx]['verified'] = True
            verification_results[idx]['reasoning'] = (
                f"Verified in alternative context ({alternative['move']}): " +
                v_result['reasoning'])
            if verbose:
                print(f"      ✓ Now verified in alternative context")
        elif sanity['was_overridden'] and verbose:
            print(f"      ✗ Alt context verification overridden by sanity check")


# ============================================================================
# MAIN JUDGING PIPELINE
# ============================================================================

def judge_explanation_improved(entry: dict, explanation: str,
                              gold_atoms: Optional[List[str]] = None,
                              provider: str = "openai", model: str = "gpt-4o",
                              verbose: bool = True,
                              reverify_alts: bool = False,
                              base_url: Optional[str] = None,
                              api_key: Optional[str] = None) -> dict:
    """
    Improved judge pipeline: decompose -> per-atom verify + sanity -> match gold -> quality -> scoring.

    This is the main evaluation pipeline that:
    1. Decomposes explanation into claims and atoms
    2. Extracts alternative moves mentioned
    3. Verifies each atom with tools and sanity checks
    4. Optionally re-verifies alternative-move atoms in their context
    5. Evaluates claim correctness
    6. Matches against gold atoms (recall)
    7. Checks quality assessment correctness
    8. Scores fluency and specificity
    9. Computes composite score

    Args:
        entry: Position dict with 'fen', 'move_uci', 'move_san', 'wp_loss'
        explanation: Generated explanation text to evaluate
        gold_atoms: List of gold standard atoms (or None to extract from entry)
        provider: LLM provider ('openai', 'anthropic', or 'qwen')
        model: Model name
        verbose: Print progress
        reverify_alts: Re-verify alternative-move atoms in alternative context
        base_url: Base URL for OpenAI-compatible APIs (for Qwen)
        api_key: API key (optional)

    Returns:
        Enriched dict with all dimensions:
        - claim_accuracy: Fraction of correct claims
        - n_claims, n_correct_claims: Claim counts
        - claims: List of claim dicts with correctness info
        - recall: Fraction of gold atoms covered
        - alternatives: List of alternative move dicts
        - quality_correct: Boolean quality assessment correctness
        - gen_assessment, actual_quality: Quality assessments
        - fluency, specificity: 1-5 scores
        - composite: Overall composite score
        - atoms: List of atomic claims
        - verification: List of verification results
        - matching: Gold atom matching results
        - n_sanity_overrides: Count of sanity check overrides
    """
    board = chess.Board(entry['fen'])
    move_san = entry.get('move_san') or board.san(chess.Move.from_uci(entry['move_uci']))
    wp_loss = entry.get('wp_loss', 0)

    # 1. Decompose into claims and atoms
    if verbose:
        print("  Decomposing into claims and atoms...")
    atoms, claims = decompose_to_atoms(explanation, provider=provider, model=model,
                                       base_url=base_url, api_key=api_key)
    if verbose:
        print(f"  -> {len(claims)} claims, {len(atoms)} atoms")

    # 2. Extract alternative moves
    if verbose:
        print("  Extracting alternative moves...")
    alternatives = extract_alternatives(
        explanation, atoms, entry['fen'], move_san,
        provider=provider, model=model,
        base_url=base_url, api_key=api_key)
    if verbose:
        if alternatives:
            print(f"  -> {len(alternatives)} alternative(s) found:")
            for alt in alternatives:
                stance_tag = {'better': '↑', 'worse': '↓', 'neutral': '→'}[alt['stance']]
                eval_diff = alt.get('eval_diff', 0.0)
                print(f"     {stance_tag} {alt['move']} (eval_diff: {eval_diff:+.2f}, "
                      f"{len(alt['atom_indices'])} atoms)")
        else:
            print(f"  -> No alternatives found")

    # 3. Per-atom verification with sanity checks
    if verbose:
        print("  Verifying atoms (per-atom)...")
    verification, gen_assessment, verify_log = verify_atoms_improved(
        atoms, entry['fen'], move_san, entry['move_uci'],
        alternatives=alternatives,
        provider=provider, model=model, verbose=verbose,
        base_url=base_url, api_key=api_key)

    n_verified = sum(1 for r in verification if r.get('verified'))
    n_overridden = sum(1 for r in verification if r.get('was_overridden'))
    if verbose:
        print(f"  -> {n_verified}/{len(atoms)} verified ({n_overridden} sanity overrides)")
        print(f"  -> assessment: {gen_assessment}")

    # 4. Two-pass verification: re-verify alternative-move atoms in their context
    if reverify_alts:
        for alt in alternatives:
            verify_alternative_atoms(
                alt, atoms, entry['fen'], move_san, verification,
                provider=provider, model=model, verbose=verbose,
                base_url=base_url, api_key=api_key)

        # Recount after alternative verification
        n_verified_final = sum(1 for r in verification if r.get('verified'))
        if n_verified_final > n_verified and verbose:
            print(f"  -> {n_verified_final - n_verified} additional atoms verified in alternative context")

    # 5. Evaluate claims (all verifiable atoms must be verified for claim to be correct)
    for claim in claims:
        claim['n_atoms'] = len(claim['atom_indices'])

        # Separate verifiable from unverifiable atoms
        verifiable_indices = [i for i in claim['atom_indices']
                             if verification[i].get('confidence') != 'n/a']
        claim['n_verifiable'] = len(verifiable_indices)
        claim['n_verified'] = sum(
            1 for i in verifiable_indices if verification[i]['verified'])
        claim['n_unverifiable'] = len(claim['atom_indices']) - len(verifiable_indices)

        # Claim is correct if all verifiable atoms are verified
        if claim['n_verifiable'] == 0:
            claim['correct'] = False
        else:
            claim['correct'] = all(
                verification[i]['verified'] for i in verifiable_indices)

    n_correct_claims = sum(1 for c in claims if c['correct'])
    claim_accuracy = n_correct_claims / len(claims) if claims else 0.0

    if verbose:
        print(f"  Claim evaluation: {n_correct_claims}/{len(claims)} correct")

    # 6. Match gold atoms
    if gold_atoms is None:
        gold_atoms = entry.get('extracted', {}).get('reasoning', [])
    if verbose:
        print(f"  Matching against {len(gold_atoms)} gold atoms...")
    matching = match_gold_atoms(atoms, gold_atoms, provider=provider, model=model,
                                base_url=base_url, api_key=api_key)
    n_covered = sum(1 for r in matching if r.get('covered'))
    if verbose:
        print(f"  -> {n_covered}/{len(gold_atoms)} covered")

    # 7. Quality check
    quality_result = check_quality_improved(verification, entry['fen'],
                                            entry['move_uci'], wp_loss)
    if verbose:
        print(f"  Quality: gen={quality_result['gen_assessment']}, "
              f"actual={quality_result['actual_quality']} -> "
              f"{'correct' if quality_result['quality_correct'] else 'WRONG'}")

    # 8. Scoring
    if verbose:
        print("  Scoring fluency & specificity...")
    fluency = score_fluency(explanation, provider=provider, model=model,
                           base_url=base_url, api_key=api_key)
    specificity = score_specificity(explanation, entry['fen'], move_san,
                                     provider=provider, model=model,
                                     base_url=base_url, api_key=api_key)
    if verbose:
        print(f"  -> fluency={fluency:.2f}, specificity={specificity:.2f}")

    # 9. Composite
    factual_precision = n_verified / len(atoms) if atoms else 0.0
    recall = n_covered / len(gold_atoms) if gold_atoms else 0.0

    composite = compute_composite_score(
        claim_accuracy, recall, quality_result['quality_correct'],
        specificity, fluency)
    if verbose:
        print(f"  -> composite={composite:.3f}")

    return {
        'claim_accuracy': claim_accuracy,
        'n_claims': len(claims),
        'n_correct_claims': n_correct_claims,
        'claims': claims,
        'recall': recall,
        'alternatives': alternatives,
        'quality_correct': quality_result['quality_correct'],
        'gen_assessment': quality_result['gen_assessment'],
        'actual_quality': quality_result['actual_quality'],
        'engine_wp_loss': quality_result['engine_wp_loss'],
        'fluency': fluency,
        'specificity': specificity,
        'composite': composite,
        'atoms': atoms,
        'verification': verification,
        'verify_tool_log': verify_log,
        'matching': matching,
        'gold_atoms': gold_atoms,
        'n_sanity_overrides': n_overridden,
    }
