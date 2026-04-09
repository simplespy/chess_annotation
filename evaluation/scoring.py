"""
Scoring functions for chess commentary evaluation.

Provides:
- Fluency scoring (1-5)
- Specificity scoring (1-5)
- Composite score computation
"""

import math
from llm_client import call_llm


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


def _score_with_logprobs(system, user, provider="openai", model="gpt-4o", base_url=None, api_key=None):
    """
    Get a 1-5 score using logprob-weighted averaging (OpenAI/Qwen) or parsed digit (Anthropic).

    For OpenAI: Uses logprobs to get probability distribution over 1-5,
    then computes weighted average for more nuanced scoring.

    For Anthropic: Parses single digit output (no logprobs available).

    Args:
        system: System prompt
        user: User prompt
        provider: LLM provider
        model: Model name
        base_url: Base URL for OpenAI-compatible APIs
        api_key: API key (optional)

    Returns:
        Float score between 1.0 and 5.0
    """
    if provider in ("openai", "qwen"):
        import openai

        kwargs = {}
        if base_url:
            kwargs['base_url'] = base_url
        if api_key:
            kwargs['api_key'] = api_key
        elif provider == "qwen":
            kwargs['base_url'] = base_url or "http://127.0.0.1:8000/v1"
            kwargs['api_key'] = api_key or 'not-needed'

        client = openai.OpenAI(**kwargs)
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


def score_fluency(explanation, provider="openai", model="gpt-4o", base_url=None, api_key=None):
    """
    Rate explanation fluency 1-5 using logprob-weighted scoring.

    Args:
        explanation: Explanation text to score
        provider: LLM provider ('openai', 'anthropic', or 'qwen')
        model: Model name
        base_url: Base URL for OpenAI-compatible APIs
        api_key: API key (optional)

    Returns:
        Float score 1.0-5.0
    """
    return _score_with_logprobs(FLUENCY_SYSTEM, explanation,
                                provider=provider, model=model, base_url=base_url, api_key=api_key)


def score_specificity(explanation, fen, move_san, provider="openai", model="gpt-4o",
                     base_url=None, api_key=None):
    """
    Rate explanation specificity 1-5 using logprob-weighted scoring.

    Args:
        explanation: Explanation text to score
        fen: Position FEN
        move_san: Move in SAN notation
        provider: LLM provider ('openai', 'anthropic', or 'qwen')
        model: Model name
        base_url: Base URL for OpenAI-compatible APIs
        api_key: API key (optional)

    Returns:
        Float score 1.0-5.0
    """
    system = SPECIFICITY_SYSTEM.format(fen=fen, move_san=move_san)
    return _score_with_logprobs(system, explanation,
                                provider=provider, model=model, base_url=base_url, api_key=api_key)


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
