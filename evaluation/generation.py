"""
Chess commentary generation functions.

Provides functions for generating chess move explanations using LLMs with:
- Engine analysis integration
- Tool-augmented generation (raw mode)
- Multiple provider support (OpenAI, Anthropic, Qwen)
"""

import chess
from llm_client import call_with_tools


# ============================================================================
# GENERATION PROMPTS
# ============================================================================

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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fen_to_ascii(fen):
    """Convert FEN string to ASCII board representation."""
    return str(chess.Board(fen))


def build_gen_user_prompt(entry, engine_lines, ascii=False):
    """
    Build user prompt for generation: FEN + move + engine analysis.

    Args:
        entry: Dict with 'fen', 'move_uci', 'wp_loss' (optional)
        engine_lines: List of engine analysis lines from eval_utils.get_engine_analysis()
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


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_commentary(entry, engine_lines, provider="openai", model="gpt-4o",
                       ascii=False, use_tools=True, base_url=None, api_key=None):
    """
    Generate NL move commentary with engine info.

    Args:
        entry: Position dict with 'fen', 'move_uci', 'wp_loss'
        engine_lines: Engine analysis lines from eval_utils.get_engine_analysis()
        provider: LLM provider ('openai', 'anthropic', or 'qwen')
        model: Model name
        ascii: Include ASCII board
        use_tools: Enable tool calling
        base_url: Base URL for OpenAI-compatible APIs (for Qwen)
        api_key: API key (optional)

    Returns:
        Tuple of (generated_text, tool_log)
    """
    messages = [{"role": "system", "content": GEN_SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": build_gen_user_prompt(entry, engine_lines, ascii=ascii)})
    return call_with_tools(messages, provider=provider, model=model, temperature=0.3,
                          use_tools=use_tools, base_url=base_url, api_key=api_key)


def generate_commentary_raw(entry, provider="openai", model="gpt-4o", ascii=False,
                           use_tools=True, base_url=None, api_key=None):
    """
    Generate commentary from FEN + move only (tool-augmented).

    Args:
        entry: Position dict with 'fen', 'move_uci'
        provider: LLM provider ('openai', 'anthropic', or 'qwen')
        model: Model name
        ascii: Include ASCII board
        use_tools: Enable tool calling
        base_url: Base URL for OpenAI-compatible APIs (for Qwen)
        api_key: API key (optional)

    Returns:
        Tuple of (generated_text, tool_log)
    """
    messages = [{"role": "system", "content": GEN_RAW_SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": build_gen_user_prompt_raw(entry, ascii=ascii)})
    return call_with_tools(messages, provider=provider, model=model, temperature=0.3,
                          use_tools=use_tools, base_url=base_url, api_key=api_key)
