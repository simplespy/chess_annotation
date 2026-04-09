"""
ChessPublishing PGN → JSONL extraction.

Recursively walks the full game tree (mainline + all variations).
Every (position, move) pair that has a non-trivial comment becomes one JSONL
entry, ready for downstream atom extraction by an LLM.

Pipeline:  PGN file(s) → this script → JSONL (one entry per annotated move)

Each entry contains:
  fen            – board position *before* the move
  move_san/uci   – the move this annotation is about
  annotation     – cleaned-up commentary text (see transforms below)
  prev_moves     – last 6 half-moves as a numbered PGN string for context
  context_fen    – FEN of the position at the start of the prev_moves window
  is_mainline    – True if this node is on the game's main line
  variation_depth– how many levels deep into sidelines (0 = mainline)
  mainline_move  – (sidelines only) the SAN of the mainline move this is an
                   alternative to; None for mainline entries or for moves that
                   continue within an existing sideline
  parent_comment – the annotation of the nearest ancestor node that had a
                   comment, giving framing context ("Black has two options:")
  annotator      – from the PGN [Annotator] header
  game           – "White vs Black" label
  source_file    – which PGN file this came from
  metadata       – full PGN header dict

Text transforms applied during extraction:
  1. Fragment repair     – variation comments that start mid-sentence (lowercase)
                          get the move prepended: "is good" → "14...Bd6 is good"
  2. "This move" resolve – "this move is strong" → "Nf3 is strong" when the
                          move SAN isn't already in the text
  3. Continuation lines  – child variation moves after each annotated move:
       default:     regex detects trailing connectors and appends moves inline
       --no-stitch: stored as a separate `line` field (annotation stays clean)
  4. Content filter      – bare moves, single-word evaluations, name-only
                          attributions, and move+filler patterns are dropped

Usage:
    # Default: stitch continuation moves into annotation text
    python chesspublishing_to_jsonl.py \\
        --input chesspublishing*.pgn \\
        --output data/chesspublishing.jsonl \\
        [--max-games 100] [--min-length 15]

    # Line mode: clean annotations + separate "line" field
    python chesspublishing_to_jsonl.py \\
        --input chesspublishing*.pgn \\
        --output data/chesspublishing_line.jsonl \\
        --no-stitch
"""

import argparse
import chess.pgn
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
#  1. CONTINUATION HANDLING
#
#  Problem: chess annotators often end a comment expecting the reader to
#  see the continuation moves that follow in the PGN tree.  When we extract
#  one entry per comment, that context is lost.
#
#  Two modes (controlled by --stitch flag):
#
#  LINE mode (default):
#    Store the continuation moves as a separate `line` field on each entry
#    (and on each alternative).  Annotation text stays clean/unmodified.
#
#  STITCH mode (--stitch):
#    Detect trailing connectors via regex and append the continuation moves
#    inline into the annotation text.  No separate `line` field.
# ═══════════════════════════════════════════════════════════════════════════

TRAILING_RE = re.compile(
    r'(?:'
    # Explicit intro phrases (always stitch)
    r'for example|for instance|such as|including|as follows|namely'
    r'|as in|after which|e\.g\.'
    # Chess-annotation connectors that expect continuation moves
    r'|and if|but if|and after|but after|and now|but now'
    r'|and then|but then|allows|allowing|when|then'
    r'|as if|as now|as after'
    # Trailing "after" — comment says "better after" / "worse after" / etc.
    r'|after'
    # Causal / concessive connectors that expect a continuation line
    r'|although after|although|though after|even after|even though'
    r'|because after|because of|since after|since'
    # Intent / plan phrases (the continuation IS the idea)
    r'|intending|with the idea|preparing|threatening'
    r'|followed by|leads to|leading to|resulting in'
    # "Now" at end of sentence (e.g. "Now after")
    r'|now after|now'
    # Copula / linking verbs (e.g. "most common is")
    r'|is|was|are|were'
    # Bare conjunctions / comparatives at end (comment clearly continues)
    r'|,\s*and|,\s*but|,\s*as|,\s*or'
    r'|better than|worse than|less usual than|more common than'
    r')\s*:?\s*$',
    re.IGNORECASE,
)


def _comment_trails_off(text):
    """True when the comment ends mid-thought, expecting continuation moves."""
    text = text.rstrip()
    if text.endswith(':'):
        return True
    if TRAILING_RE.search(text):
        return True
    return False


def _stitch_continuations(comment, child_node, board):
    """Append child variation moves to a comment that trails off.

    Called after board.push(move), so `board` reflects the position after the
    annotated move.  `child_node` is the PGN node whose children we'll render.
    """
    if not _comment_trails_off(comment):
        return comment
    if not child_node.variations:
        return comment

    main_line = _render_variation_line(child_node.variations[0], board)
    stitched = comment.rstrip().rstrip(':').rstrip() + ': ' + main_line

    for var in child_node.variations[1:]:
        alt_line = _render_variation_line(var, board)
        stitched += '; or ' + alt_line

    return stitched


def _render_variation_line(node, board, max_moves=12):
    """Render up to `max_moves` from a PGN node as a human-readable string.

    Follows the mainline continuation of `node`, formatting each move with
    its number and including inline comments (truncated to 120 chars).

    Example output: "12. Nf3 Be7 (a solid choice) 13. O-O"
    """
    parts = []
    b = board.copy()
    count = 0
    while True:
        move = node.move
        san = b.san(move)

        # Format with move number (always for White, only first for Black)
        fullmove = b.fullmove_number
        if b.turn == chess.WHITE:
            parts.append(f'{fullmove}. {san}')
        elif count == 0:
            # Black's first move in a variation needs the "..." notation
            parts.append(f'{fullmove}... {san}')
        else:
            parts.append(san)

        # Inline comment (truncated to avoid bloat)
        if node.comment and node.comment.strip():
            c = node.comment.strip()
            if len(c) > 120:
                c = c[:117] + '...'
            parts.append(f'({c})')

        b.push(move)
        count += 1
        if count >= max_moves or not node.variations:
            break
        node = node.variations[0]

    return ' '.join(parts)



# ═══════════════════════════════════════════════════════════════════════════
#  2. CONTENT-QUALITY FILTER
#
#  Problem: many PGN comments are too bare to extract useful atoms from:
#  bare move notation ("19. Nd5+"), single-word evaluations ("Losing."),
#  name attributions ("Ribli"), or move + filler ("28...Rc8 wins").
#
#  Solution: a multi-layer filter that catches these patterns and drops the
#  entry (while still recursing into the node's children).
# ═══════════════════════════════════════════════════════════════════════════

# Matches pure SAN notation with optional move number, including castling
_MOVE_ONLY_RE = re.compile(
    r'^[\d.\s]*'
    r'(?:[KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8][=QRBN]?|O-O(?:-O)?)'
    r'[+#]?\s*$'
)

# Content-free evaluations and filler phrases that carry no extractable info.
# Checked after stripping parens, quotes, and trailing punctuation.
_CONTENTLESS = {
    # single-word evals
    'mate', 'instead', 'not', 'missing', 'losing', 'winning',
    'forced', 'best', 'naturally', 'obviously', 'the move',
    'simplest', 'desperation', 'ouch', 'finally', 'again',
    'here', 'anyway', 'combination', 'devastating', 'prophylaxis',
    'alternatively', 'zugzwang',
    # short phrases — generic evaluations without specific squares/pieces
    'white wins', 'black wins', 'black goes', 'white goes',
    'the only move', 'the simplest', 'too passive', 'too slow',
    'the best', 'a mistake', 'pretty desperate', 'a novelty',
    'a blunder', 'an oversight', 'too late', 'the point',
    'of course', 'what else', 'where else', 'why not',
    'not bad', 'not the best', 'back again', 'here we go',
    'white has', 'black has', 'i prefer',
    'white can play', 'black can play',
    'black is fine', 'black is solid', 'white is fine',
    'now it\'s over', 'bang', 'after', 'with the idea',
    'see next game', 'if instead',
}

# Matches name-only attributions: "Ribli", "(Keilhack)", "J. Watson"
_NAME_ONLY_RE = re.compile(
    r'^\(?[A-Z][a-z]+(?:\s*[A-Z]\.?)?\)?$'
)

# Matches a move (with number) at the start — used with _FILLER_WORDS
# to catch "28...Rc8 wins", "5. Bg2 etc."
_MOVE_PLUS_FILLER_RE = re.compile(
    r'^[\d.]+\s*(?:\.{3})?[KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8][=QRBN]?[+#]?'
    r'|^[\d.]+\s*(?:\.{3})?O-O(?:-O)?[+#]?'
)
_FILLER_WORDS = {
    'wins', 'loses', 'anyway', 'though', 'say', 'etc', 'but', 'as',
    'when', 'then', 'and', 'or',
}


def _has_content(text):
    """Return False if the annotation is too bare to be worth extracting.

    Applies four checks in order:
      1. Too short (< 4 chars after stripping)
      2. Pure move notation (e.g. "19. Nd5+", "O-O-O")
      3. Contentless phrase from the _CONTENTLESS set (also handles parens)
      4. Name-only attribution (e.g. "Ribli", "(Keilhack)")
      5. Move + single filler word (e.g. "28...Rc8 wins")
    """
    text = text.strip().rstrip('.,!;')
    if len(text) < 4:
        return False
    if _MOVE_ONLY_RE.match(text):
        return False
    # Strip surrounding parens/quotes for matching — catches "(forced)",
    # "(what else?)", etc.
    inner = text.strip('()"\'!? ').rstrip('.,!;:')
    if inner.lower() in _CONTENTLESS:
        return False
    if _NAME_ONLY_RE.match(text):
        return False
    # "move + single filler word" — e.g. "28...Rc8 wins", "5. Bg2 etc."
    words = text.split()
    if len(words) <= 3 and _MOVE_PLUS_FILLER_RE.match(text):
        tail = words[-1].lower().rstrip('.,!;:')
        if tail in _FILLER_WORDS:
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  3. FRAGMENT REPAIR
#
#  Problem: in PGN, variation comments are attached to the move node, but
#  the annotator writes them as continuations of a sentence whose subject
#  is the move itself.  E.g. after "(14...Bd6", the comment "is a solid
#  choice" is a fragment — the subject "14...Bd6" was the move, not text.
#
#  Quantified: ~88% of variation annotations start lowercase (fragments),
#  vs ~2% of mainline annotations.
#
#  Solution: for non-mainline entries, detect fragments (start lowercase /
#  leading punctuation) and prepend the formatted move string.
#    "is a solid choice" → "14...Bd6 is a solid choice"
# ═══════════════════════════════════════════════════════════════════════════

def _is_fragment(text):
    """True if the comment is a sentence fragment (starts lowercase or punctuation).

    Heuristic: complete sentences start with an uppercase letter.  Comments
    beginning with lowercase or with connective punctuation (,.; ) are
    fragments that need the move prepended.
    """
    if not text:
        return False
    return text[0].islower() or text[0] in (',', '.', ';')


def _format_move_str(fullmove, san, is_white):
    """Format a move with its number: '14. Nc3' (White) or '14...Bd6' (Black)."""
    if is_white:
        return f'{fullmove}. {san}'
    return f'{fullmove}...{san}'


def _repair_fragment(move_str, comment):
    """Prepend the move to a fragment comment to make it a complete sentence.

    Strips leading connective punctuation first so we don't get
    "14...Bd6 , is a solid choice".
    """
    comment = comment.lstrip(',;. ')
    return f'{move_str} {comment}'


# ═══════════════════════════════════════════════════════════════════════════
#  4. "THIS MOVE" RESOLUTION
#
#  Problem: ~1% of annotations say "this move is strong" without naming the
#  actual move.  The move SAN is available in the entry's move_san field,
#  but the annotation text itself is opaque to an LLM that reads only text.
#
#  Solution: when the annotation contains "this move" / "the text move" /
#  "the game move" and the SAN isn't already present in the text, replace
#  the first occurrence with the actual SAN.
#    "this move is strong" → "Nf3 is strong"
# ═══════════════════════════════════════════════════════════════════════════

_THIS_MOVE_RE = re.compile(
    r'\b(this move|the text move|the game move|the game continuation)\b',
    re.IGNORECASE,
)


def _resolve_this_move(comment, move_san):
    """Replace 'this move' with the actual move SAN when it's not already in the text.

    Only replaces the first occurrence to avoid mangling sentences like
    "this move is better than the game continuation" where both phrases appear.
    """
    if move_san and move_san not in comment and _THIS_MOVE_RE.search(comment):
        return _THIS_MOVE_RE.sub(move_san, comment, count=1)
    return comment


# ═══════════════════════════════════════════════════════════════════════════
#  5. RECURSIVE TREE WALKER
#
#  Core extraction logic.  Walks the python-chess game tree depth-first,
#  applying all transforms and collecting entries.
#
#  Key design decisions:
#    - board is mutated via push/pop (not copied) for performance
#    - move_history is a list of tuples, appended immutably per branch
#    - parent_comment propagates down so child entries know their framing
#    - mainline_san is computed once per node to avoid re-deriving it
#    - content-filtered entries still recurse into children (their subtree
#      may contain useful annotations even if this node's comment is bare)
# ═══════════════════════════════════════════════════════════════════════════

def walk_tree(node, board, entries, move_history, *,
              depth=0, is_mainline=True, parent_comment='',
              annotator='', game_label='', source_file='', metadata=None,
              stitch=False):
    """Recurse through all variations, collecting annotated entries.

    Args:
        node:           python-chess game node (has .variations, .comment, .move)
        board:          chess.Board — mutated via push/pop, reflects position
                        *before* processing this node's children
        entries:        list to append JSONL dicts to (accumulated output)
        move_history:   list of (san, fen, fullmove, is_white) tuples for the
                        path from root to this node
        depth:          variation nesting depth (0 = mainline)
        is_mainline:    True if we're still on the game's main continuation
        parent_comment: annotation text from the nearest ancestor with a comment,
                        providing framing context for this node's entries
        annotator/game_label/source_file/metadata: pass-through fields
    """

    # Compute the mainline move SAN *once* for this node's children.
    # variations[0] is always the mainline continuation in python-chess.
    # Sidelines (i > 0) use this to populate their mainline_move field,
    # telling the downstream LLM "the game played X, this sideline considers Y".
    mainline_san = None
    if node.variations:
        mainline_san = board.san(node.variations[0].move)

    # ── Build per-variation alternative info for branch points ──
    # At a branch point, every variation (mainline + sidelines) gets an
    # alternatives list containing all OTHER siblings that have non-trivial
    # annotations.  This means:
    #   - The mainline entry sees annotated sidelines as alternatives
    #   - Each sideline entry sees the mainline + other sidelines as alternatives
    # Siblings with no/minimal annotation are omitted from everyone's list.
    _branch_infos = []  # one dict per variation, indexed by position in node.variations
    if len(node.variations) > 1:
        fullmove_here = board.fullmove_number
        is_white_here = board.turn == chess.WHITE
        for var in node.variations:
            var_san = board.san(var.move)
            var_uci = var.move.uci()
            var_comment = var.comment.strip() if var.comment else None

            # Apply same text transforms
            if var_comment:
                if _is_fragment(var_comment):
                    move_str = _format_move_str(fullmove_here, var_san, is_white_here)
                    var_comment = _repair_fragment(move_str, var_comment)
                var_comment = _resolve_this_move(var_comment, var_san)

            # Continuation: stitch into annotation or store as separate field
            line = None
            if var.variations:
                board.push(var.move)
                if stitch and var_comment:
                    var_comment = _stitch_continuations(var_comment, var, board)
                else:
                    line = _render_variation_line(var.variations[0], board)
                board.pop()

            info = {
                'move_san': var_san,
                'move_uci': var_uci,
                'annotation': var_comment,
            }
            if not stitch:
                info['line'] = line
            _branch_infos.append(info)

    def _alternatives_for(idx):
        """Return alternatives list for variation at position `idx`, excluding
        siblings with no annotation."""
        if not _branch_infos:
            return None
        alts = [info for j, info in enumerate(_branch_infos)
                if j != idx and info['annotation']]
        return alts if alts else None

    for i, child in enumerate(node.variations):
        move = child.move
        fen_before = board.fen()
        san = board.san(move)
        uci = move.uci()
        fullmove = board.fullmove_number
        is_white = board.turn == chess.WHITE

        # i == 0 is the mainline continuation; i > 0 are sidelines.
        # A child is only mainline if the parent was mainline AND it's i==0.
        child_is_mainline = is_mainline and (i == 0)
        # Depth increments when we enter a sideline (i > 0), stays same for
        # mainline continuation (i == 0).
        child_depth = depth if i == 0 else depth + 1

        # Build prev_moves: last 6 half-moves leading to (and including) this
        # move, formatted as a numbered PGN string for the LLM's context.
        current_history = move_history + [(san, fen_before, fullmove, is_white)]
        prev_window = current_history[-6:]
        prev_moves_str = _format_prev_moves(current_history)
        # context_fen: position at the *start* of the prev_moves window,
        # so the LLM can replay the window from this FEN.
        context_fen = prev_window[0][1]

        if child.comment and child.comment.strip():
            comment = child.comment.strip()

            # ── Transform 1: Fragment repair ──
            # Variation comments often start mid-sentence because the move
            # itself is the implicit subject in PGN.
            # Only applied to non-mainline entries (sidelines + moves within
            # sidelines), since mainline comments are usually complete sentences.
            if not child_is_mainline and _is_fragment(comment):
                move_str = _format_move_str(fullmove, san, is_white)
                comment = _repair_fragment(move_str, comment)

            # ── Transform 2: "This move" resolution ──
            comment = _resolve_this_move(comment, san)

            # ── Transform 3: Continuation handling ──
            # --stitch: append continuation moves into annotation text
            # default:  store as separate `line` field
            line = None
            if child.variations:
                board.push(move)
                if stitch:
                    comment = _stitch_continuations(comment, child, board)
                else:
                    line = _render_variation_line(child.variations[0], board)
                board.pop()

            # ── Transform 4: Content filter ──
            # Drop bare/contentless entries, but still recurse into children
            # (a bare parent may have useful annotated descendants).
            if not _has_content(comment):
                board.push(move)
                walk_tree(child, board, entries, current_history,
                          depth=child_depth, is_mainline=child_is_mainline,
                          parent_comment=comment,
                          annotator=annotator, game_label=game_label,
                          source_file=source_file, metadata=metadata,
                          stitch=stitch)
                board.pop()
                continue

            # ── Build the entry ──

            # mainline_move: for sidelines (i > 0), record what the mainline
            # played instead. This tells the downstream LLM: "the game played
            # mainline_move, but this sideline discusses move_san as an
            # alternative."  None for mainline entries and for continuation
            # moves within an existing sideline (those aren't branch points).
            mainline_move = None
            if i > 0 and mainline_san:
                mainline_move = mainline_san

            entry = {
                'fen': fen_before,
                'move_uci': uci,
                'move_san': san,
                'annotation': comment,
            }
            if not stitch:
                entry['line'] = line
            entry.update({
                'prev_moves': prev_moves_str,
                'context_fen': context_fen,
                'is_mainline': child_is_mainline,
                'variation_depth': child_depth,
                'mainline_move': mainline_move,
                'alternatives': _alternatives_for(i),
                'parent_comment': parent_comment or None,
                'annotator': annotator,
                'game': game_label,
                'source_file': source_file,
                'metadata': metadata or {},
            })
            entries.append(entry)

        # Recurse into this child's subtree.
        # Pass this node's comment as parent_comment so descendants have
        # framing context.  Falls back to the inherited parent_comment if
        # this node has no comment of its own.
        child_comment = child.comment.strip() if child.comment else ''
        board.push(move)
        walk_tree(child, board, entries, current_history,
                  depth=child_depth, is_mainline=child_is_mainline,
                  parent_comment=child_comment or parent_comment,
                  annotator=annotator, game_label=game_label,
                  source_file=source_file, metadata=metadata,
                  stitch=stitch)
        board.pop()


def _format_prev_moves(history, n=6):
    """Format the last N entries of move_history as a numbered PGN string.

    Example: "10. b3 Be7 11. Bb2 Rd8 12. h4 g6"

    The first Black move gets "..." notation (e.g. "10... Be7") since there's
    no preceding White move in the window to establish the move number.
    """
    recent = history[-n:]
    parts = []
    for san, _fen, fullmove, is_white in recent:
        if is_white:
            parts.append(f'{fullmove}. {san}')
        elif not parts:
            # First entry in window is Black — needs explicit number
            parts.append(f'{fullmove}... {san}')
        else:
            parts.append(san)
    return ' '.join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  6. GAME PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def process_game(game, source_file, entries, *, stitch=False):
    """Extract all annotated (position, move) pairs from one game.

    Handles the game-level comment (before move 1) separately, then delegates
    to walk_tree for the recursive extraction.
    """
    annotator = game.headers.get('Annotator', '')
    white = game.headers.get('White', '?')
    black = game.headers.get('Black', '?')
    game_label = f'{white} vs {black}'
    metadata = dict(game.headers)

    # Game-level comment (before move 1) — often an intro paragraph about
    # the opening or the players.  No move is associated with it.
    game_comment = game.comment.strip() if game.comment else ''
    if game_comment:
        entry = {
            'fen': game.board().fen(),
            'move_uci': None,
            'move_san': None,
            'annotation': game_comment,
        }
        if not stitch:
            entry['line'] = None
        entry.update({
            'prev_moves': '',
            'context_fen': game.board().fen(),
            'is_mainline': True,
            'variation_depth': 0,
            'mainline_move': None,
            'alternatives': None,
            'parent_comment': None,
            'annotator': annotator,
            'game': game_label,
            'source_file': source_file,
            'metadata': metadata,
        })
        entries.append(entry)

    board = game.board()
    walk_tree(game, board, entries, [],
              annotator=annotator, game_label=game_label,
              source_file=source_file, metadata=metadata,
              stitch=stitch)


# ═══════════════════════════════════════════════════════════════════════════
#  7. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Extract annotated (position, move) pairs from ChessPublishing PGNs.')
    parser.add_argument('--input', nargs='+', required=True,
                        help='PGN file(s) to process (supports globs)')
    parser.add_argument('--output', required=True,
                        help='Output JSONL path')
    parser.add_argument('--max-games', type=int, default=0,
                        help='Max games per file (0 = all)')
    parser.add_argument('--min-length', type=int, default=0,
                        help='Drop entries with annotation shorter than N chars '
                             '(0 = keep all, 15 is a reasonable backstop)')
    parser.add_argument('--no-stitch', dest='stitch', action='store_false',
                        help='Store continuation moves as a separate "line" field '
                             'instead of stitching into annotation text (default: stitch).')
    parser.set_defaults(stitch=True)
    args = parser.parse_args()

    all_entries = []
    file_stats = {}

    for pgn_path in args.input:
        pgn_path = Path(pgn_path)
        if not pgn_path.exists():
            print(f'Skipping {pgn_path} (not found)')
            continue

        print(f'Processing {pgn_path.name}...')
        entries = []
        n_games = 0

        with open(pgn_path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                n_games += 1
                process_game(game, pgn_path.name, entries, stitch=args.stitch)
                if n_games % 1000 == 0:
                    print(f'  {n_games} games, {len(entries)} entries so far')
                if args.max_games and n_games >= args.max_games:
                    break

        mainline_count = sum(1 for e in entries if e['is_mainline'])
        variation_count = len(entries) - mainline_count

        file_stats[pgn_path.name] = {
            'games': n_games,
            'entries': len(entries),
            'mainline': mainline_count,
            'variation': variation_count,
        }

        print(f'  Done: {n_games} games → {len(entries)} entries '
              f'(mainline: {mainline_count}, variation: {variation_count})')

        all_entries.extend(entries)

    # Optional min-length filter — a blunt backstop after all the smarter
    # content filters have run.  Useful for dropping the long tail of
    # <15-char entries that slip through pattern matching.
    if args.min_length > 0:
        before = len(all_entries)
        all_entries = [e for e in all_entries if len(e['annotation']) >= args.min_length]
        print(f'\nMin-length filter ({args.min_length}): {before} → {len(all_entries)} '
              f'({before - len(all_entries)} dropped)')

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Summary
    total_mainline = sum(1 for e in all_entries if e['is_mainline'])
    total_variation = len(all_entries) - total_mainline

    print(f'\n{"="*60}')
    print(f'TOTAL: {len(all_entries)} entries from '
          f'{sum(s["games"] for s in file_stats.values())} games')
    print(f'  Mainline: {total_mainline}  |  Variation: {total_variation}')
    print(f'Written to {out_path}')


if __name__ == '__main__':
    main()
