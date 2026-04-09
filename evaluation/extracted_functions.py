"""
Extracted Functions from eval_tools_improved.ipynb
Organized by category with complete function definitions and imports.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import json
import os
import math
import random
import re
import numpy as np
import chess
import chess.engine
import chess.svg
from IPython.display import display, SVG, Markdown, HTML
from dotenv import load_dotenv
from dataclasses import dataclass

# From chess_tools module (assumed to be available)
from chess_tools import (TOOL_FUNCTIONS,
                         TOOL_SCHEMAS_OPENAI,
                         TOOL_SCHEMAS_ANTHROPIC,
                         execute_tool)


# ============================================================================
# CATEGORY 1: LLM API FUNCTIONS
# ============================================================================

@dataclass
class ToolCall:
    """Represents a tool call from an LLM."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Unified LLM response format."""
    text: str
    tool_calls: list


def call_openai(messages, tools=None, model="gpt-4o", temperature=0.3):
    """
    Call OpenAI API with unified interface.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tools: Optional list of tool schemas in OpenAI format
        model: Model name
        temperature: Sampling temperature

    Returns:
        Tuple of (LLMResponse, raw_message)
    """
    import openai
    client = openai.OpenAI()
    token_key = "max_completion_tokens" if any(m in model for m in ("gpt-4.1", "gpt-5", "o3", "o4")) else "max_tokens"
    kwargs = dict(model=model, messages=messages, temperature=temperature, **{token_key: 2048})
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    text = msg.content or ""
    tc = []
    if msg.tool_calls:
        for c in msg.tool_calls:
            tc.append(ToolCall(id=c.id, name=c.function.name,
                               arguments=json.loads(c.function.arguments)))
    return LLMResponse(text=text, tool_calls=tc), msg


def call_anthropic(messages, tools=None, model="claude-sonnet-4-5-20250929",
                   system=None, temperature=0.3):
    """
    Call Anthropic API with unified interface.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tools: Optional list of tool schemas in Anthropic format
        model: Model name
        system: System message (extracted from messages if not provided)
        temperature: Sampling temperature

    Returns:
        Tuple of (LLMResponse, raw_response)
    """
    import anthropic
    client = anthropic.Anthropic()
    kwargs = dict(model=model, max_tokens=2048, temperature=temperature)
    if system:
        kwargs["system"] = system
    api_messages = [m for m in messages if m["role"] != "system"]
    kwargs["messages"] = api_messages
    if tools:
        kwargs["tools"] = tools
    resp = client.messages.create(**kwargs)
    text = ""
    tc = []
    for block in resp.content:
        if block.type == "text":
            text += block.text
        elif block.type == "tool_use":
            tc.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
    return LLMResponse(text=text, tool_calls=tc), resp


def call_llm(messages, tools=None, provider="openai", model=None, temperature=0.3):
    """
    Unified LLM calling interface supporting multiple providers.

    Args:
        messages: List of message dicts
        tools: Tool schemas (provider-specific format)
        provider: 'openai' or 'anthropic'
        model: Model name (provider-specific default if None)
        temperature: Sampling temperature

    Returns:
        Tuple of (LLMResponse, raw_response)
    """
    if provider == "openai":
        return call_openai(messages, tools=tools, model=model or "gpt-4o", temperature=temperature)
    elif provider == "anthropic":
        m = model or "claude-sonnet-4-5-20250929"
        sys_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        return call_anthropic(messages, tools=tools, model=m, system=sys_msg, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _call_with_tools(messages, provider="openai", model="gpt-4o", temperature=0.3,
                     max_rounds=10, verbose=True, use_tools=True):
    """
    Run LLM with tool access, handling multi-round tool calls.

    This function orchestrates a multi-turn conversation where the LLM can:
    1. Call tools to gather information
    2. Receive tool results
    3. Continue reasoning with the results
    4. Repeat until it provides a final answer

    Args:
        messages: Initial conversation messages
        provider: 'openai' or 'anthropic'
        model: Model name
        temperature: Sampling temperature
        max_rounds: Maximum tool-calling rounds
        verbose: Print tool calls
        use_tools: Enable tool calling (if False, single-round without tools)

    Returns:
        Tuple of (final_text, tool_log)
        - final_text: LLM's final response text
        - tool_log: List of tool calls with args and results
    """
    if use_tools:
        tool_schemas = TOOL_SCHEMAS_OPENAI if provider == "openai" else TOOL_SCHEMAS_ANTHROPIC
    else:
        tool_schemas = []
        max_rounds = 1
    tool_log = []

    for _ in range(max_rounds):
        resp, raw = call_llm(messages, tools=tool_schemas, provider=provider,
                             model=model, temperature=temperature)

        if not resp.tool_calls:
            return resp.text.strip() if resp.text else "", tool_log

        if provider == "openai":
            messages.append({
                "role": "assistant", "content": resp.text or None,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.name,
                                             "arguments": json.dumps(tc.arguments)}}
                               for tc in resp.tool_calls],
            })
            for tc in resp.tool_calls:
                result = execute_tool(tc.name, tc.arguments)
                tool_log.append({"tool": tc.name, "args": tc.arguments, "result": result})
                if verbose:
                    print(f"    tool: {tc.name}({tc.arguments}) -> {result[:80]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        elif provider == "anthropic":
            content_blocks = []
            if resp.text:
                content_blocks.append({"type": "text", "text": resp.text})
            for tc in resp.tool_calls:
                content_blocks.append({"type": "tool_use", "id": tc.id,
                                       "name": tc.name, "input": tc.arguments})
            messages.append({"role": "assistant", "content": content_blocks})
            tool_results = []
            for tc in resp.tool_calls:
                result = execute_tool(tc.name, tc.arguments)
                tool_log.append({"tool": tc.name, "args": tc.arguments, "result": result})
                if verbose:
                    print(f"    tool: {tc.name}({tc.arguments}) -> {result[:80]}")
                tool_results.append({"type": "tool_result", "tool_use_id": tc.id,
                                     "content": result})
            messages.append({"role": "user", "content": tool_results})

    return resp.text.strip() if resp.text else "[max tool rounds]", tool_log


# ============================================================================
# CATEGORY 2: GENERATION FUNCTIONS
# ============================================================================

# Generation prompts
GEN_SYSTEM_PROMPT = """\
You are a chess instructor explaining moves to an intermediate player.

Given a position (FEN), the move played, and engine analysis, write a concise, \
instructive explanation of the move.

GUIDELINES:
- Explain the strategic or tactical purpose of the move.
- If the move is bad or a blunder: explain why it fails and what was better.
- Be specific: name squares, pieces, diagonals, and key variations.
- No generic chess philosophy or filler. Every sentence must be position-specific.
- Keep it concise. A simple forced move needs one sentence. \
A rich tactical position deserves a short paragraph.
- Output plain text only. No JSON, no bullet points, no headers."""

GEN_RAW_SYSTEM_PROMPT = """\
You are a chess instructor explaining moves to an intermediate player.

Given a position (FEN) and the move played, write a concise, instructive \
explanation of the move. You have access to chess analysis tools — use them \
to understand the position before writing your explanation.

GUIDELINES:
- Explain the strategic or tactical purpose of the move.
- If the move is bad or a blunder: explain why it fails and what was better.
- Be specific: name squares, pieces, diagonals, and key variations.
- No generic chess philosophy or filler. Every sentence must be position-specific.
- Keep it concise. A simple forced move needs one sentence. \
A rich tactical position deserves a short paragraph.
- Output plain text only. No JSON, no bullet points, no headers."""


def fen_to_ascii(fen):
    """Convert FEN string to ASCII board representation."""
    return str(chess.Board(fen))


def build_gen_user_prompt(entry, engine_lines, ascii=False):
    """
    Build user prompt for generation: FEN + move + engine analysis.

    Args:
        entry: Dict with 'fen', 'move_uci', 'wp_loss' (optional)
        engine_lines: List of engine analysis lines
        ascii: Include ASCII board representation

    Returns:
        Formatted prompt string
    """
    board = chess.Board(entry['fen'])
    move_san = board.san(chess.Move.from_uci(entry['move_uci']))

    top_lines = [l for l in engine_lines if l.get('is_top', True)]
    played_line = next((l for l in engine_lines if not l.get('is_top', True)), None)

    pv_text = []
    for i, line in enumerate(top_lines, 1):
        pv_text.append(f"  {i}. {line['move_san']} ({line['eval']}): {line['pv_san']}")

    wp_loss = entry.get('wp_loss', 0)
    if wp_loss > 20:   quality_hint = f" [BLUNDER — wp_loss: {wp_loss:.1f}%]"
    elif wp_loss > 5:  quality_hint = f" [SUBOPTIMAL — wp_loss: {wp_loss:.1f}%]"
    else:              quality_hint = f" [wp_loss: {wp_loss:.1f}%]"

    prompt = f"FEN: {entry['fen']}\n"
    if ascii:
        prompt += f"Board:\n{fen_to_ascii(entry['fen'])}\n"
    prompt += f"Move played: {move_san}{quality_hint}\nEngine top-3:\n" + "\n".join(pv_text) + "\n"
    if played_line:
        prompt += f"Played move eval:\n  {played_line['move_san']} ({played_line['eval']}): {played_line['pv_san']}\n"
    return prompt


def build_gen_user_prompt_raw(entry, ascii=False):
    """
    Build user prompt for raw generation: FEN + move only (no engine lines).

    Args:
        entry: Dict with 'fen', 'move_uci'
        ascii: Include ASCII board representation

    Returns:
        Formatted prompt string
    """
    board = chess.Board(entry['fen'])
    move_san = board.san(chess.Move.from_uci(entry['move_uci']))
    prompt = f"FEN: {entry['fen']}\n"
    if ascii:
        prompt += f"Board:\n{fen_to_ascii(entry['fen'])}\n"
    prompt += f"Move played: {move_san}\n"
    return prompt


def generate_commentary(entry, engine_lines, provider="openai", model="gpt-4o",
                       ascii=False, use_tools=True):
    """
    Generate NL move commentary with engine info.

    Args:
        entry: Position dict with 'fen', 'move_uci', 'wp_loss'
        engine_lines: Engine analysis lines
        provider: LLM provider
        model: Model name
        ascii: Include ASCII board
        use_tools: Enable tool calling

    Returns:
        Tuple of (generated_text, tool_log)
    """
    # Note: GEN_DEMOS would need to be built from demo data
    messages = [{"role": "system", "content": GEN_SYSTEM_PROMPT}]
    # messages.extend(GEN_DEMOS)  # Would be added if demos available
    messages.append({"role": "user", "content": build_gen_user_prompt(entry, engine_lines, ascii=ascii)})
    return _call_with_tools(messages, provider=provider, model=model, temperature=0.3, use_tools=use_tools)


def generate_commentary_raw(entry, provider="openai", model="gpt-4o", ascii=False, use_tools=True):
    """
    Generate commentary from FEN + move only (tool-augmented).

    Args:
        entry: Position dict with 'fen', 'move_uci'
        provider: LLM provider
        model: Model name
        ascii: Include ASCII board
        use_tools: Enable tool calling

    Returns:
        Tuple of (generated_text, tool_log)
    """
    # Note: GEN_RAW_DEMOS would need to be built from demo data
    messages = [{"role": "system", "content": GEN_RAW_SYSTEM_PROMPT}]
    # messages.extend(GEN_RAW_DEMOS)  # Would be added if demos available
    messages.append({"role": "user", "content": build_gen_user_prompt_raw(entry, ascii=ascii)})
    return _call_with_tools(messages, provider=provider, model=model, temperature=0.3, use_tools=use_tools)


# ============================================================================
# CATEGORY 3: DECOMPOSITION/CLASSIFICATION FUNCTIONS
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


def _parse_judge_json(text):
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


def decompose_to_atoms(text, provider="openai", model="gpt-4o"):
    """
    Decompose NL explanation into claims and atoms.

    The function uses an LLM to break down complex explanations into:
    - Claims: High-level logical assertions
    - Atoms: Individual verifiable facts within each claim

    Args:
        text: Explanation text to decompose
        provider: LLM provider
        model: Model name

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
    resp, _ = call_llm(messages, provider=provider, model=model, temperature=0.0)
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


def classify_atom(atom_text):
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
# CATEGORY 4: VERIFICATION FUNCTIONS
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


def verify_single_atom(atom, fen, move_san, move_uci,
                       provider="openai", model="gpt-4o", verbose=True):
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
        provider: LLM provider
        model: Model name
        verbose: Print progress

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

    text, tool_log = _call_with_tools(
        messages, provider=provider, model=model,
        temperature=0.0, max_rounds=10, verbose=verbose)

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


def run_sanity_checks(verify_result, fen, move_uci, is_alternative_atom=False):
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


def verify_atoms_improved(atoms, fen, move_san, move_uci, alternatives=None,
                          provider="openai", model="gpt-4o", verbose=True):
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
        provider: LLM provider
        model: Model name
        verbose: Print progress

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
            provider=provider, model=model, verbose=verbose)

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


def match_gold_atoms(candidate_atoms, gold_atoms, provider="openai", model="gpt-4o"):
    """
    Check which gold atoms are covered by candidate atoms.

    Uses LLM to compare generated atoms against gold standard atoms,
    determining semantic coverage (not just exact matching).

    Args:
        candidate_atoms: List of generated atomic claims
        gold_atoms: List of gold standard atomic claims
        provider: LLM provider
        model: Model name

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
    resp, _ = call_llm(messages, provider=provider, model=model, temperature=0.0)
    data = _parse_judge_json(resp.text)
    return data.get('results', [])


def check_quality_improved(atoms_results, fen, move_uci, wp_loss):
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
# CATEGORY 5: ALTERNATIVE MOVES FUNCTIONS
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


def extract_alternatives(explanation, atoms, fen, move_san,
                         provider="openai", model="gpt-4o"):
    """
    Extract alternative moves mentioned in the explanation.

    Identifies and evaluates alternative moves discussed in the commentary:
    1. Parses explanation to find alternative move mentions
    2. Determines stance (better/worse/neutral)
    3. Evaluates with engine to verify claims
    4. Maps atoms that discuss each alternative

    Args:
        explanation: Full explanation text
        atoms: List of atomic claims
        fen: Position FEN
        move_san: Played move in SAN
        provider: LLM provider
        model: Model name

    Returns:
        List of dicts with:
        - move: SAN of alternative
        - move_uci: UCI notation
        - stance: 'better' | 'worse' | 'neutral'
        - variation: full variation string if provided
        - atom_indices: which atoms discuss this alternative
        - eval: engine evaluation of the alternative
        - eval_diff: difference vs played move (negative = alternative is worse)
        - played_wp: win percentage after played move
        - alt_wp: win percentage after alternative
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

    resp, _ = call_llm(messages, provider=provider, model=model, temperature=0.0)
    data = _parse_judge_json(resp.text)
    alternatives_list = data.get('alternatives', [])

    if not alternatives_list:
        return []

    # Enrich each alternative with engine eval
    # Note: Requires STOCKFISH_PATH and K constant to be defined
    board = chess.Board(fen)
    enriched = []

    # This section requires chess.engine and STOCKFISH_PATH to be configured
    # Simplified version shown here - full version uses engine
    return alternatives_list  # Return basic version without engine enrichment


def verify_alternative_atoms(alternative, atoms, fen, move_san,
                             verification_results,
                             provider="openai", model="gpt-4o", verbose=True):
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
        provider: LLM provider
        model: Model name
        verbose: Print progress
    """
    if not alternative['atom_indices']:
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
            provider=provider, model=model, verbose=False)

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
# CATEGORY 6: SCORING FUNCTIONS
# ============================================================================

FLUENCY_SYSTEM = """\
Rate the fluency and clarity of this chess move explanation on a scale of 1-5.

1 = Incoherent, grammatically broken, or nonsensical
2 = Understandable but poorly structured, awkward phrasing
3 = Adequate clarity, some rough edges
4 = Well-written, clear, and well-structured
5 = Excellent prose, concise, engaging, and perfectly clear

Respond with ONLY a single digit (1-5)."""

SPECIFICITY_SYSTEM = """\
Rate the specificity of this chess move explanation on a scale of 1-5.

Context:
- FEN: {fen}
- Move: {move_san}

1 = Completely generic, could apply to any position (e.g., "this is a good move")
2 = Vaguely position-aware but lacks concrete details
3 = Mentions some specific squares/pieces but also has generic filler
4 = Mostly specific with concrete squares, pieces, and variations
5 = Fully specific — every claim references concrete board elements

Respond with ONLY a single digit (1-5)."""


def _score_with_logprobs(system, user, provider="openai", model="gpt-4o"):
    """
    Get a 1-5 score using logprob-weighted averaging (OpenAI) or parsed digit (Anthropic).

    For OpenAI: Uses logprobs to get probability distribution over 1-5,
    then computes weighted average for more nuanced scoring.

    For Anthropic: Parses single digit output (no logprobs available).

    Args:
        system: System prompt
        user: User prompt
        provider: LLM provider
        model: Model name

    Returns:
        Float score between 1.0 and 5.0
    """
    if provider == "openai":
        import openai
        client = openai.OpenAI()
        token_key = "max_completion_tokens" if any(m in (model or "") for m in ("gpt-4.1", "gpt-5", "o3", "o4")) else "max_tokens"
        resp = client.chat.completions.create(
            model=model or "gpt-4o",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
            logprobs=True,
            top_logprobs=10,
            **{token_key: 1},
        )
        choice = resp.choices[0]

        if choice.logprobs and choice.logprobs.content:
            top_lps = choice.logprobs.content[0].top_logprobs
            score_probs = {}
            for lp in top_lps:
                token = lp.token.strip()
                if token in ("1", "2", "3", "4", "5"):
                    score_probs[int(token)] = math.exp(lp.logprob)

            if score_probs:
                total_prob = sum(score_probs.values())
                return round(sum(s * p for s, p in score_probs.items()) / total_prob, 2)

        # Fallback: parse generated token
        token = choice.message.content.strip()
        for c in token:
            if c in "12345":
                return float(c)
        return 3.0

    else:
        # Anthropic: no logprobs, parse digit
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model or "claude-sonnet-4-5-20250929",
            max_tokens=1,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        token = resp.content[0].text.strip()
        for c in token:
            if c in "12345":
                return float(c)
        return 3.0


def score_fluency(explanation, provider="openai", model="gpt-4o"):
    """
    Rate explanation fluency 1-5 using logprob-weighted scoring.

    Args:
        explanation: Explanation text to score
        provider: LLM provider
        model: Model name

    Returns:
        Float score 1.0-5.0
    """
    return _score_with_logprobs(FLUENCY_SYSTEM, explanation,
                                provider=provider, model=model)


def score_specificity(explanation, fen, move_san, provider="openai", model="gpt-4o"):
    """
    Rate explanation specificity 1-5 using logprob-weighted scoring.

    Args:
        explanation: Explanation text to score
        fen: Position FEN
        move_san: Move in SAN notation
        provider: LLM provider
        model: Model name

    Returns:
        Float score 1.0-5.0
    """
    system = SPECIFICITY_SYSTEM.format(fen=fen, move_san=move_san)
    return _score_with_logprobs(system, explanation,
                                provider=provider, model=model)


def compute_composite_score(claim_accuracy, recall, quality_correct,
                            specificity, fluency):
    """
    Weighted combination of all metrics, normalized to 0-1.

    Weights:
    - ClaimAccuracy = 0.30 (factual correctness of claims)
    - Recall = 0.25 (coverage of gold atoms)
    - Quality = 0.20 (correct quality assessment)
    - Specificity = 0.15 (position-specific details)
    - Fluency = 0.10 (writing quality)

    Args:
        claim_accuracy: Fraction of correct claims (0-1)
        recall: Fraction of gold atoms covered (0-1)
        quality_correct: Boolean, is quality assessment correct
        specificity: Specificity score (1-5)
        fluency: Fluency score (1-5)

    Returns:
        Float composite score 0.0-1.0
    """
    specificity_norm = (specificity - 1) / 4   # 1-5 -> 0-1
    fluency_norm = (fluency - 1) / 4           # 1-5 -> 0-1
    quality_norm = 1.0 if quality_correct else 0.0

    composite = (
        0.30 * claim_accuracy +
        0.25 * recall +
        0.20 * quality_norm +
        0.15 * specificity_norm +
        0.10 * fluency_norm
    )
    return round(composite, 3)


# ============================================================================
# CATEGORY 7: MAIN JUDGING PIPELINE
# ============================================================================

def judge_explanation_improved(entry, explanation, gold_atoms=None,
                               provider="openai", model="gpt-4o", verbose=True,
                               reverify_alts=False):
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
        provider: LLM provider
        model: Model name
        verbose: Print progress
        reverify_alts: Re-verify alternative-move atoms in alternative context

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
    atoms, claims = decompose_to_atoms(explanation, provider=provider, model=model)
    if verbose:
        print(f"  -> {len(claims)} claims, {len(atoms)} atoms")

    # 2. Extract alternative moves
    if verbose:
        print("  Extracting alternative moves...")
    alternatives = extract_alternatives(
        explanation, atoms, entry['fen'], move_san,
        provider=provider, model=model)
    if verbose:
        if alternatives:
            print(f"  -> {len(alternatives)} alternative(s) found:")
            for alt in alternatives:
                stance_tag = {'better': '↑', 'worse': '↓', 'neutral': '→'}[alt['stance']]
                print(f"     {stance_tag} {alt['move']} (eval_diff: {alt['eval_diff']:+.2f}, "
                      f"{len(alt['atom_indices'])} atoms)")
        else:
            print(f"  -> No alternatives found")

    # 3. Per-atom verification with sanity checks
    if verbose:
        print("  Verifying atoms (per-atom)...")
    verification, gen_assessment, verify_log = verify_atoms_improved(
        atoms, entry['fen'], move_san, entry['move_uci'],
        alternatives=alternatives,
        provider=provider, model=model, verbose=verbose)

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
                provider=provider, model=model, verbose=verbose)

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
    matching = match_gold_atoms(atoms, gold_atoms, provider=provider, model=model)
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
    fluency = score_fluency(explanation, provider=provider, model=model)
    specificity = score_specificity(explanation, entry['fen'], move_san,
                                     provider=provider, model=model)
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


# ============================================================================
# CATEGORY 8: TESTING FUNCTIONS
# ============================================================================

def test_gen(idx, accepted_positions, provider="openai", model="gpt-4o",
             ascii=False, engine=True, use_tools=True):
    """
    Generate NL commentary for a test position and display comparisons.

    Args:
        idx: Index into accepted_positions list
        accepted_positions: List of test positions
        provider: LLM provider
        model: Model name
        ascii: Include ASCII board
        engine: Use engine lines in prompt
        use_tools: Enable tool calling

    Returns:
        Tuple of (generated_text, tool_log)
    """
    pos = accepted_positions[idx]
    # show_position(idx) would be called here if display functions available

    if engine:
        el = pos.get('engine_lines')  # Would call get_engine_analysis if None
    else:
        el = None
    entry = {'fen': pos['fen'], 'move_uci': pos['move_uci'],
             'annotation': pos.get('annotation', ''), 'wp_loss': pos.get('wp_loss', 0)}

    mode = "engine prompt" if engine else "raw prompt"
    print(f"\nGenerating ({provider}/{model}, {mode})...")
    if engine:
        text, tool_log = generate_commentary(entry, el, provider=provider, model=model,
                                            ascii=ascii, use_tools=use_tools)
    else:
        text, tool_log = generate_commentary_raw(entry, provider=provider, model=model,
                                                ascii=ascii, use_tools=use_tools)
    if tool_log:
        print(f'{len(tool_log)} tool call(s)')

    gold_atoms = pos['extracted'].get('reasoning', [])
    print(f'#### Generated\n> {text}')
    print(f'#### Gold atoms\n' + '\n'.join(f'- {a}' for a in gold_atoms))

    return text, tool_log


def test_gen_raw(idx, accepted_positions, provider="openai", model="gpt-4o", ascii=False):
    """
    Generate commentary without engine info (tool-augmented).

    Args:
        idx: Index into accepted_positions list
        accepted_positions: List of test positions
        provider: LLM provider
        model: Model name
        ascii: Include ASCII board

    Returns:
        Tuple of (generated_text, tool_log)
    """
    pos = accepted_positions[idx]
    entry = {'fen': pos['fen'], 'move_uci': pos['move_uci'],
             'annotation': pos.get('annotation', ''), 'wp_loss': pos.get('wp_loss', 0)}

    print(f"\nGenerating RAW ({provider}/{model})...")
    text, tool_log = generate_commentary_raw(entry, provider=provider, model=model, ascii=ascii)

    if tool_log:
        print(f'{len(tool_log)} tool call(s)')

    gold_atoms = pos['extracted'].get('reasoning', [])
    print(f'#### Generated (raw)\n> {text}')
    print(f'#### Gold atoms\n' + '\n'.join(f'- {a}' for a in gold_atoms))

    return text, tool_log


def test_judge(idx, accepted_positions, explanation, provider="openai", model="gpt-4o"):
    """
    Full improved judge pipeline for a test position.

    Args:
        idx: Index into accepted_positions list
        accepted_positions: List of test positions
        explanation: Explanation text to evaluate
        provider: LLM provider
        model: Model name

    Returns:
        Judge results dict
    """
    pos = accepted_positions[idx]
    gold_atoms = pos['extracted'].get('reasoning', [])

    results = judge_explanation_improved(pos, explanation, gold_atoms=gold_atoms,
                                         provider=provider, model=model)

    # show_judge_results_improved(results, gold_atoms) would be called here
    return results


def test_gen_judge(idx, accepted_positions, provider="openai", model="gpt-4o"):
    """
    Generate + judge in one call (uses improved pipeline).

    Args:
        idx: Index into accepted_positions list
        accepted_positions: List of test positions
        provider: LLM provider
        model: Model name

    Returns:
        Tuple of (generated_text, judge_results)
    """
    text, tool_log = test_gen(idx, accepted_positions, provider=provider, model=model)
    print("\n--- Judging (improved) ---")
    results = test_judge(idx, accepted_positions, text, provider=provider, model=model)
    return text, results


# ============================================================================
# CATEGORY 9: BATCH EVALUATION
# ============================================================================

def batch_evaluate(accepted_positions, indices=None, n=10, seed=42,
                   provider="openai", model="gpt-4o",
                   gen_model=None, output_path=None, use_engine=False, use_tools=True):
    """
    Run generation + improved judge on multiple positions.

    This function:
    1. Selects positions to evaluate (random sample or specified indices)
    2. For each position:
       - Generates commentary (with or without engine lines)
       - Judges with full improved pipeline
       - Saves results incrementally if output_path provided
    3. Computes aggregate metrics

    Args:
        accepted_positions: List of test positions
        indices: specific indices to evaluate (overrides n/seed)
        n: number of random positions
        seed: random seed
        provider: LLM provider
        model: judge model
        gen_model: generation model (defaults to model)
        output_path: optional JSONL path to save results
        use_engine: if False, build generation prompts without engine lines
        use_tools: Enable tool calling in generation

    Returns:
        List of result dicts with metrics for each position
    """
    gen_model = gen_model or model

    if indices is None:
        rng = random.Random(seed)
        indices = rng.sample(range(len(accepted_positions)), min(n, len(accepted_positions)))

    all_results = []

    for idx in indices:
        pos = accepted_positions[idx]
        board = chess.Board(pos['fen'])
        move_san = pos.get('move_san', board.san(chess.Move.from_uci(pos['move_uci'])))
        print(f"\n{'='*60}")
        print(f"[{len(all_results)+1}/{len(indices)}] idx={idx} — {pos.get('game','?')} — {move_san}")
        print(f"{'='*60}")

        el = pos.get('engine_lines') if use_engine else None
        entry = {'fen': pos['fen'], 'move_uci': pos['move_uci'],
                 'annotation': pos.get('annotation', ''), 'wp_loss': pos.get('wp_loss', 0),
                 'move_san': move_san, 'extracted': pos.get('extracted', {})}

        prompt_mode = "engine" if use_engine else "raw"
        print(f"  Generating ({provider}/{gen_model}, {prompt_mode})...")
        if use_engine:
            gen_text, gen_log = generate_commentary(entry, el, provider=provider,
                                                    model=gen_model, use_tools=use_tools)
        else:
            gen_text, gen_log = generate_commentary_raw(entry, provider=provider,
                                                        model=gen_model, use_tools=use_tools)
        print(f"  Generated: {gen_text[:100]}...")

        print(f"  Judging ({provider}/{model}, improved)...")
        gold_atoms = pos['extracted'].get('reasoning', [])
        judge_results = judge_explanation_improved(
            entry, gen_text, gold_atoms=gold_atoms,
            provider=provider, model=model)

        result_row = {
            'idx': idx,
            'fen': pos['fen'],
            'move_san': move_san,
            'game': pos.get('game', '?'),
            'wp_loss': pos.get('wp_loss', 0),
            'quality': pos.get('quality', '?'),
            'generated_text': gen_text,
            'gen_tool_calls': len(gen_log),
            'claim_accuracy': judge_results['claim_accuracy'],
            'n_claims': judge_results['n_claims'],
            'n_correct_claims': judge_results['n_correct_claims'],
            'recall': judge_results['recall'],
            'quality_correct': judge_results['quality_correct'],
            'gen_assessment': judge_results['gen_assessment'],
            'actual_quality': judge_results['actual_quality'],
            'fluency': judge_results['fluency'],
            'specificity': judge_results['specificity'],
            'composite': judge_results['composite'],
            'n_atoms': len(judge_results['atoms']),
            'n_verified': sum(1 for r in judge_results['verification'] if r.get('verified')),
            'n_gold': len(gold_atoms),
            'n_covered': sum(1 for r in judge_results['matching'] if r.get('covered')),
            'n_sanity_overrides': judge_results['n_sanity_overrides'],
            'n_alternatives': len(judge_results.get('alternatives', [])),
            'alternatives_summary': [
                {
                    'move': alt['move'],
                    'stance': alt['stance'],
                    'eval_diff': alt['eval_diff'],
                }
                for alt in judge_results.get('alternatives', [])
            ],
            'gen_model': gen_model,
            'judge_model': model,
            'provider': provider,
            'used_engine_prompt': use_engine,
        }
        all_results.append(result_row)

        if output_path:
            with open(output_path, 'a') as f:
                f.write(json.dumps(result_row) + '\n')

        n_alts = len(judge_results.get('alternatives', []))
        alt_str = f" | Alternatives: {n_alts}" if n_alts > 0 else ""
        print(f"  Claim Accuracy: {judge_results['claim_accuracy']:.0%} | "
              f"Recall: {judge_results['recall']:.0%} | "
              f"Quality: {'OK' if judge_results['quality_correct'] else 'WRONG'} | "
              f"Fluency: {judge_results['fluency']:.1f} | "
              f"Specificity: {judge_results['specificity']:.1f} | "
              f"Composite: {judge_results['composite']:.3f} | "
              f"Overrides: {judge_results['n_sanity_overrides']}{alt_str}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(all_results)} positions)")
    print(f"{'='*60}")
    avg_claim_acc = np.mean([r['claim_accuracy'] for r in all_results])
    avg_rec = np.mean([r['recall'] for r in all_results])
    quality_acc = np.mean([r['quality_correct'] for r in all_results])
    avg_fluency = np.mean([r['fluency'] for r in all_results])
    avg_spec = np.mean([r['specificity'] for r in all_results])
    avg_comp = np.mean([r['composite'] for r in all_results])
    total_overrides = sum(r['n_sanity_overrides'] for r in all_results)
    print(f"  Avg Claim Accuracy: {avg_claim_acc:.1%}")
    print(f"  Avg Recall: {avg_rec:.1%}")
    print(f"  Quality Accuracy: {quality_acc:.1%}")
    print(f"  Avg Fluency: {avg_fluency:.2f} / 5")
    print(f"  Avg Specificity: {avg_spec:.2f} / 5")
    print(f"  Avg Composite: {avg_comp:.3f}")
    print(f"  Total Sanity Overrides: {total_overrides}")
    avg_alts = np.mean([r.get('n_alternatives', 0) for r in all_results])
    print(f"  Avg Alternatives/Position: {avg_alts:.1f}")

    return all_results


def summarize_results(results):
    """
    Display summary table from batch_evaluate results.

    Args:
        results: List of result dicts from batch_evaluate

    Returns:
        pandas DataFrame with results
    """
    import pandas as pd
    df = pd.DataFrame(results)

    print('### Summary')
    print(f'Positions: {len(df)}')
    print(f'Avg Claim Accuracy: {df["claim_accuracy"].mean():.1%}')
    print(f'Avg Recall: {df["recall"].mean():.1%}')
    print(f'Quality Accuracy: {df["quality_correct"].mean():.1%}')
    print(f'Avg Fluency: {df["fluency"].mean():.2f} / 5')
    print(f'Avg Specificity: {df["specificity"].mean():.2f} / 5')
    print(f'Avg Composite: {df["composite"].mean():.3f}')
    print(f'Avg Atoms/Position: {df["n_atoms"].mean():.1f}')
    print(f'Total Sanity Overrides: {df["n_sanity_overrides"].sum()}')
    print(f'Avg Alternatives/Position: {df["n_alternatives"].mean():.1f}')

    # Per-quality breakdown
    if 'quality' in df.columns:
        print('\n### By Move Quality')
        grouped = df.groupby('quality').agg({
            'claim_accuracy': 'mean',
            'recall': 'mean',
            'quality_correct': 'mean',
            'fluency': 'mean',
            'specificity': 'mean',
            'composite': 'mean',
            'n_sanity_overrides': 'sum',
            'idx': 'count',
        }).rename(columns={'idx': 'count'})
        print(grouped)

    return df


# ============================================================================
# HELPER FUNCTIONS (UTILITY)
# ============================================================================

def cp_to_winpct(cp):
    """
    Lichess win% formula.

    Args:
        cp: Centipawn evaluation

    Returns:
        Win percentage (0-100)
    """
    K = 0.00368208
    return 50 + 50 * math.tanh(K * cp / 2)


def get_engine_analysis(fen, move_uci=None, depth=22, multipv=3, stockfish_path='/opt/homebrew/bin/stockfish'):
    """
    Get Stockfish top lines. If move_uci given and not in top-N, fetch separately.

    Args:
        fen: Position FEN
        move_uci: Optional move to analyze if not in top lines
        depth: Search depth
        multipv: Number of top lines to get
        stockfish_path: Path to Stockfish binary

    Returns:
        List of engine line dicts with:
        - move_uci, move_san: Move notation
        - eval: Evaluation string (e.g., '+1.23' or 'M5')
        - pv_san: Principal variation in SAN
        - cp, mate: Numeric evaluation
        - is_top: Whether this was a top-N line
    """
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
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

        if move_uci is not None:
            if not any(l['move_uci'] == move_uci for l in lines):
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
