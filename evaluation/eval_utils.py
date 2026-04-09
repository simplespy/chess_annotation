"""Generic utility functions for chess evaluation and display."""
import math
import chess
import chess.engine
import chess.svg
from IPython.display import display, SVG, Markdown
from config import K, STOCKFISH_PATH


def wp(cp):
    """Convert centipawn score to win percentage using Lichess formula."""
    return 50 + 50 * math.tanh(K * cp / 2)


def get_engine_analysis(fen, move_uci=None, depth=22, multipv=3):
    """Get Stockfish top lines with analysis.

    Args:
        fen: Position FEN string
        move_uci: If provided and not in top-N, analyze separately
        depth: Engine search depth
        multipv: Number of top lines to analyze

    Returns:
        List of analysis dicts with move_uci, move_san, eval, pv_san, cp, mate, is_top
    """
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


def fen_to_ascii(fen):
    """Convert FEN to ASCII board representation."""
    return str(chess.Board(fen))


def show_board(fen, move_uci=None, size=350, bad_move=False):
    """Display a board with optional move arrow.

    Args:
        fen: Position FEN string
        move_uci: Optional UCI move to highlight with arrow
        size: Board size in pixels
        bad_move: If True, use red arrow; otherwise green
    """
    board = chess.Board(fen)
    kwargs = {'size': size}
    if move_uci:
        move = chess.Move.from_uci(move_uci)
        arrow_color = '#cc0000' if bad_move else '#15781B'
        kwargs['arrows'] = [(move.from_square, move.to_square)]
        kwargs['colors'] = {'arrow': arrow_color}
    svg = chess.svg.board(board, **kwargs)
    display(SVG(svg))


def show_engine_lines(engine_lines, played_uci=None):
    """Display engine analysis lines.

    Args:
        engine_lines: List of engine line dicts with move_san, eval, pv_san, is_top
        played_uci: Optional UCI of played move to tag
    """
    for i, line in enumerate(engine_lines, 1):
        is_top = line.get('is_top', True)
        tag = ' <- played' if line.get('move_uci') == played_uci else ''
        prefix = f'{i}.' if is_top else '*Played:*'
        display(Markdown(f'{prefix} **{line["move_san"]}** ({line["eval"]}): {line["pv_san"]}{tag}'))


def show_judge_results(results, gold_atoms):
    """Display formatted judge results (original pipeline).

    Args:
        results: Judge results dict with verification, matching, quality fields
        gold_atoms: List of gold standard atom strings
    """
    n_ver = sum(1 for r in results['verification'] if r.get('verified'))
    n_cov = sum(1 for r in results['matching'] if r.get('covered'))
    q_label = 'correct' if results['quality_correct'] else 'WRONG'
    q_detail = f"gen={results['gen_assessment']}, actual={results['actual_quality']}"

    display(Markdown(
        f'| Metric | Score |\n|--------|-------|\n'
        f'| Factual Precision | **{results["factual_precision"]:.0%}** '
        f'({n_ver}/{len(results["atoms"])}) |\n'
        f'| Recall | **{results["recall"]:.0%}** '
        f'({n_cov}/{len(gold_atoms)}) |\n'
        f'| Quality | **{q_label}** ({q_detail}) |\n'
    ))

    display(Markdown('**Generated atoms:**\n' +
        '\n'.join(f'{i+1}. {a}' for i, a in enumerate(results['atoms']))))

    display(Markdown('**Candidate atoms (verified):**'))
    for r in results['verification']:
        v = 'Y' if r.get('verified') else 'N'
        reason = f'  \n  *{r["reasoning"]}*' if r.get('reasoning') else ''
        display(Markdown(f'- [{v}] {r.get("atom", "?")}{reason}'))

    display(Markdown('**Gold atoms (covered):**'))
    for r in results['matching']:
        v = 'Y' if r.get('covered') else 'N'
        match = f'  \n  *matched: {r["matching_candidate"]}*' if r.get('matching_candidate') else ''
        display(Markdown(f'- [{v}] {r.get("gold_atom", "?")}{match}'))


def show_judge_results_improved(results, gold_atoms):
    """Display formatted judge results (improved pipeline).

    Shows all 6 metrics, sanity override annotations, confidence + type tags.

    Args:
        results: Judge results dict with full improved pipeline metrics
        gold_atoms: List of gold standard atom strings
    """
    n_ver = sum(1 for r in results['verification'] if r.get('verified'))
    n_cov = sum(1 for r in results['matching'] if r.get('covered'))
    n_over = results.get('n_sanity_overrides', 0)
    q_label = 'correct' if results['quality_correct'] else 'WRONG'
    q_detail = f"gen={results['gen_assessment']}, actual={results['actual_quality']}"

    # Summary table with all metrics
    display(Markdown(
        f'| Metric | Score |\n|--------|-------|\n'
        f'| Claim Accuracy | **{results.get("claim_accuracy", 0):.0%}** '
        f'({results.get("n_correct_claims", 0)}/{results.get("n_claims", 0)}) |\n'
        f'| Recall | **{results["recall"]:.0%}** '
        f'({n_cov}/{len(gold_atoms)}) |\n'
        f'| Quality | **{q_label}** ({q_detail}) |\n'
        f'| Fluency | **{results.get("fluency", 0):.2f}** / 5 |\n'
        f'| Specificity | **{results.get("specificity", 0):.2f}** / 5 |\n'
        f'| **Composite** | **{results.get("composite", 0):.3f}** |\n'
        f'| Sanity overrides | {n_over} |\n'
    ))

    # Alternatives display
    if results.get('alternatives'):
        display(Markdown('**Alternative moves mentioned:**'))
        for alt in results['alternatives']:
            stance_icon = {'better': '↑ BETTER', 'worse': '↓ WORSE', 'neutral': '→ NEUTRAL'}[alt['stance']]
            eval_info = f"{alt['eval']}"
            wp_info = f"played: {alt.get('played_wp', 0):.1f}% → alt: {alt.get('alt_wp', 0):.1f}% (diff: {alt['eval_diff']:+.1f})"
            atom_count = len(alt['atom_indices'])
            variation = f" — variation: {alt['variation']}" if alt['variation'] else ""

            # Detect stance/eval mismatch (threshold 5.0 to avoid flagging minor differences)
            mismatch = ""
            eval_diff = alt['eval_diff']
            if alt['stance'] == 'better' and eval_diff < -5.0:
                mismatch = f" ⚠️ **CLAIMED BETTER (actually {eval_diff:.1f} worse)**"
            elif alt['stance'] == 'worse' and eval_diff > 5.0:
                mismatch = f" ⚠️ **CLAIMED WORSE (actually {eval_diff:+.1f} better)**"

            display(Markdown(
                f"- **{alt['move']}** {stance_icon}{mismatch} | eval: {eval_info} | "
                f"WP: {wp_info} | {atom_count} atoms{variation}"
            ))

    # Claims display
    if results.get('claims'):
        display(Markdown('**Claims:**'))
        for i, claim in enumerate(results['claims']):
            status = '✓' if claim['correct'] else '✗'
            display(Markdown(
                f'{i+1}. **[{status}]** {claim["claim_text"]}  \n'
                f'   *{claim["n_verified"]}/{claim["n_atoms"]} atoms verified*'
            ))

    # Per-atom verification with type tags and sanity annotations
    display(Markdown('**Candidate atoms (verified):**'))
    for r in results['verification']:
        v = 'Y' if r.get('verified') else 'N'
        types_str = ', '.join(r.get('atom_types', []))
        conf = r.get('confidence', '?')

        # Build annotation line
        parts = [f'[{v}]']
        parts.append(f'`[{types_str}]`')
        parts.append(f'({conf})')
        parts.append(r.get('atom', '?'))

        line = ' '.join(parts)

        # Alternative context annotation
        if r.get('alt_context'):
            alt_status = '✓' if r.get('verified_alt') else '✗'
            alt_info = f'{alt_status} {r.get("alt_reasoning", "verified")}'

            # Show sanity overrides if any
            if r.get('alt_sanity_overrides'):
                override_msgs = [o['message'] for o in r['alt_sanity_overrides']]
                alt_info += f' **[OVERRIDDEN: {"; ".join(override_msgs)}]**'

            line += f'  \n  **ALT CONTEXT ({r["alt_context"]}):** {alt_info}'
        # Sanity override annotation
        elif r.get('was_overridden'):
            overrides = r.get('sanity_overrides', [])
            override_msgs = [o['message'] for o in overrides]
            line += f'  \n  **OVERRIDDEN:** {"; ".join(override_msgs)}'
        elif r.get('reasoning'):
            line += f'  \n  *{r["reasoning"]}*'

        display(Markdown(f'- {line}'))

    # Gold atoms coverage
    display(Markdown('**Gold atoms (covered):**'))
    for r in results['matching']:
        v = 'Y' if r.get('covered') else 'N'
        match = f'  \n  *matched: {r["matching_candidate"]}*' if r.get('matching_candidate') else ''
        display(Markdown(f'- [{v}] {r.get("gold_atom", "?")}{match}'))


def summarize_results(results):
    """Display summary table from batch_evaluate results.

    Args:
        results: List of result dicts from batch evaluation

    Returns:
        DataFrame with aggregated results
    """
    import pandas as pd
    df = pd.DataFrame(results)
    display(Markdown('### Summary'))
    display(Markdown(
        f'| Metric | Value |\n|--------|-------|\n'
        f'| Positions | {len(df)} |\n'
        f'| Avg Claim Accuracy | {df["claim_accuracy"].mean():.1%} |\n'
        f'| Avg Recall | {df["recall"].mean():.1%} |\n'
        f'| Quality Accuracy | {df["quality_correct"].mean():.1%} |\n'
        f'| Avg Fluency | {df["fluency"].mean():.2f} / 5 |\n'
        f'| Avg Specificity | {df["specificity"].mean():.2f} / 5 |\n'
        f'| Avg Composite | {df["composite"].mean():.3f} |\n'
        f'| Avg Atoms/Position | {df["n_atoms"].mean():.1f} |\n'
        f'| Total Sanity Overrides | {df["n_sanity_overrides"].sum()} |\n'
        f'| Avg Alternatives/Position | {df["n_alternatives"].mean():.1f} |\n'
    ))

    # Per-quality breakdown
    if 'quality' in df.columns:
        display(Markdown('### By Move Quality'))
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
        display(grouped.style.format({
            'claim_accuracy': '{:.1%}',
            'recall': '{:.1%}',
            'quality_correct': '{:.1%}',
            'fluency': '{:.2f}',
            'specificity': '{:.2f}',
            'composite': '{:.3f}',
            'n_sanity_overrides': '{:.0f}',
        }))

    return df
