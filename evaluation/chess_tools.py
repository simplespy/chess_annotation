import json as _json
import math
import chess
import chess.engine
from config import STOCKFISH_PATH, K 

def tool_get_legal_moves(fen):
    board = chess.Board(fen)
    moves = [board.san(m) for m in board.legal_moves]
    return _json.dumps({"moves": moves, "count": len(moves)})

def tool_get_piece_at(fen, square):
    board = chess.Board(fen)
    sq = chess.parse_square(square)
    p = board.piece_at(sq)
    if p is None:
        return _json.dumps({"square": square, "piece": "empty"})
    color = "white" if p.color == chess.WHITE else "black"
    return _json.dumps({"square": square, "piece": f"{color} {chess.piece_name(p.piece_type)}"})

def tool_get_attacks(fen, square):
    board = chess.Board(fen)
    sq = chess.parse_square(square)
    p = board.piece_at(sq)
    if p is None:
        return _json.dumps({"error": f"no piece on {square}"})
    attacks = sorted([chess.square_name(a) for a in board.attacks(sq)])
    return _json.dumps({"piece": p.symbol(), "square": square, "attacks": attacks})

def tool_get_attackers(fen, square, color):
    board = chess.Board(fen)
    sq = chess.parse_square(square)
    c = chess.WHITE if color == "white" else chess.BLACK
    attackers = [[chess.piece_name(board.piece_at(a).piece_type), chess.square_name(a)]
                 for a in board.attackers(c, sq)]
    return _json.dumps({"square": square, "color": color, "attackers": attackers})

def tool_count_attackers_defenders(fen, square):
    """Count attackers and defenders of a square for both colors."""
    board = chess.Board(fen)
    sq = chess.parse_square(square)
    piece = board.piece_at(sq)
    
    white_attackers = [[chess.piece_name(board.piece_at(a).piece_type), 
                        chess.square_name(a)]
                       for a in board.attackers(chess.WHITE, sq)]
    black_attackers = [[chess.piece_name(board.piece_at(a).piece_type), 
                        chess.square_name(a)]
                       for a in board.attackers(chess.BLACK, sq)]
    
    result = {
        "square": square,
        "occupant": f"{('white' if piece.color == chess.WHITE else 'black')} {chess.piece_name(piece.piece_type)}" if piece else "empty",
        "white_attackers": {"count": len(white_attackers), "pieces": white_attackers},
        "black_attackers": {"count": len(black_attackers), "pieces": black_attackers},
    }
    
    if piece:
        if piece.color == chess.WHITE:
            result["defenders"] = {"count": len(white_attackers), "pieces": white_attackers}
            result["attackers"] = {"count": len(black_attackers), "pieces": black_attackers}
        else:
            result["defenders"] = {"count": len(black_attackers), "pieces": black_attackers}
            result["attackers"] = {"count": len(white_attackers), "pieces": white_attackers}
    
    return _json.dumps(result)

def tool_check_ray_alignment(square_a, square_b):
    """Check if two squares are aligned on a ray (rank/file/diagonal)."""
    sq_a = chess.parse_square(square_a)
    sq_b = chess.parse_square(square_b)
    
    result = {"square_a": square_a, "square_b": square_b, "aligned": False}
    
    if sq_a == sq_b:
        result["aligned"] = True
        result["ray_type"] = "same_square"
        return _json.dumps(result)
    
    if chess.square_rank(sq_a) == chess.square_rank(sq_b):
        result["aligned"] = True
        result["ray_type"] = "rank"
        result["rank"] = chess.square_rank(sq_a)
        return _json.dumps(result)
    
    if chess.square_file(sq_a) == chess.square_file(sq_b):
        result["aligned"] = True
        result["ray_type"] = "file"
        result["file"] = chr(ord('a') + chess.square_file(sq_a))
        return _json.dumps(result)
    
    if abs(chess.square_file(sq_a) - chess.square_file(sq_b)) == \
       abs(chess.square_rank(sq_a) - chess.square_rank(sq_b)):
        result["aligned"] = True
        result["ray_type"] = "diagonal"
        return _json.dumps(result)
    
    return _json.dumps(result)

def tool_is_pinned(fen, square):
    """Check if a piece is pinned. Returns pinner and pinned-to piece details."""
    board = chess.Board(fen)
    sq = chess.parse_square(square)
    p = board.piece_at(sq)
    if p is None:
        return _json.dumps({"error": f"no piece on {square}"})
    pinned = board.is_pinned(p.color, sq)
    result = {"square": square, "pinned": pinned}
    if pinned:
        pin_mask = board.pin(p.color, sq)
        pin_squares = [chess.square_name(s) for s in chess.SquareSet(pin_mask)]
        result["pin_ray"] = pin_squares

        pinner = None
        pinned_to = None
        for s in chess.SquareSet(pin_mask):
            sq_name = chess.square_name(s)
            piece = board.piece_at(s)
            if piece and s != sq:
                if piece.color != p.color:
                    pinner = {"square": sq_name, "piece": f"{chess.piece_name(piece.piece_type)}"}
                else:
                    pinned_to = {"square": sq_name, "piece": f"{chess.piece_name(piece.piece_type)}"}

        if pinner:
            result["pinner"] = pinner
        if pinned_to:
            result["pinned_to"] = pinned_to
    return _json.dumps(result)

def tool_is_check(fen):
    board = chess.Board(fen)
    in_check = board.is_check()
    result = {"in_check": in_check}
    if in_check:
        result["checkers"] = [chess.square_name(sq) for sq in board.checkers()]
    return _json.dumps(result)

def tool_try_variation(fen, moves):
    board = chess.Board(fen)
    played = []
    for san in moves:
        try:
            move = board.parse_san(san)
            played.append(board.san(move))
            board.push(move)
        except Exception as e:
            return _json.dumps({"legal": False, "error": f"{san}: {str(e)}", "played": played})
    result = {"legal": True, "played": played, "resulting_fen": board.fen()}
    if board.is_checkmate():
        result["eval"] = "checkmate"
    elif board.is_stalemate():
        result["eval"] = "stalemate"
    else:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=18))
            sc = info['score'].white()
            mate = sc.mate()
            result["eval"] = f"M{mate}" if mate is not None else f"{sc.score(mate_score=10000)/100:+.2f}"
    return _json.dumps(result)

def tool_get_engine_eval(fen, depth=20):
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        info = engine.analyse(board, chess.engine.Limit(depth=int(depth)))
        sc = info['score'].white()
        pv = info.get('pv', [])
        mate = sc.mate()
        ev = f"M{mate}" if mate is not None else f"{sc.score(mate_score=10000)/100:+.2f}"
        best = board.san(pv[0]) if pv else "?"
    return _json.dumps({"eval": ev, "best_move": best})

def tool_get_top_moves(fen, n=3, depth=20):
    """Get top N engine moves with evals and PV lines."""
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        result = engine.analyse(board, chess.engine.Limit(depth=int(depth)), multipv=int(n))
        moves = []
        for info in result:
            score = info['score'].white()
            pv = info.get('pv', [])
            if pv:
                b = board.copy()
                san_moves = []
                for m in pv[:6]:
                    san_moves.append(b.san(m))
                    b.push(m)
                cp = score.score(mate_score=10000)
                mate = score.mate()
                ev = f"M{mate}" if mate is not None else f"{cp/100:+.2f}"
                moves.append({"move": board.san(pv[0]), "eval": ev, "pv": " ".join(san_moves)})
    return _json.dumps({"top_moves": moves})

def tool_eval_move(fen, move, depth=20):
    """Evaluate a specific move: eval, best move, wp_loss, quality."""
    board = chess.Board(fen)
    def wp(cp):
        return 50 + 50 * math.tanh(K * cp / 2)
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        root_info = engine.analyse(board, chess.engine.Limit(depth=int(depth)))
        root_cp = root_info['score'].white().score(mate_score=10000)
        root_pv = root_info.get('pv', [])
        best_move_san = board.san(root_pv[0]) if root_pv else "?"

        m = board.parse_san(move)
        move_san = board.san(m)
        b2 = board.copy()
        b2.push(m)
        move_info = engine.analyse(b2, chess.engine.Limit(depth=int(depth)))
        move_cp = move_info['score'].white().score(mate_score=10000)
        move_mate = move_info['score'].white().mate()
        move_eval_str = f"M{move_mate}" if move_mate is not None else f"{move_cp/100:+.2f}"

    if board.turn == chess.WHITE:
        wp_loss_val = wp(root_cp) - wp(move_cp)
    else:
        wp_loss_val = wp(move_cp) - wp(root_cp)
    wp_loss_val = round(max(0, wp_loss_val), 1)
    quality = "good" if wp_loss_val < 5 else ("bad" if wp_loss_val < 20 else "blunder")

    return _json.dumps({
        "move": move_san, "move_eval": move_eval_str,
        "best_move": best_move_san, "wp_loss": wp_loss_val, "quality": quality,
    })

def tool_compare_moves(fen, move_a, move_b, depth=20):
    board = chess.Board(fen)
    def wp(cp):
        return 50 + 50 * math.tanh(K * cp / 2)
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        ba = board.copy(); ba.push(board.parse_san(move_a))
        ia = engine.analyse(ba, chess.engine.Limit(depth=int(depth)))
        sa = ia['score'].white().score(mate_score=10000)
        bb = board.copy(); bb.push(board.parse_san(move_b))
        ib = engine.analyse(bb, chess.engine.Limit(depth=int(depth)))
        sb = ib['score'].white().score(mate_score=10000)
    return _json.dumps({
        "move_a": move_a, "eval_a": f"{sa/100:+.2f}",
        "move_b": move_b, "eval_b": f"{sb/100:+.2f}",
        "wp_diff": round(abs(wp(sa) - wp(sb)), 1)
    })

def tool_make_move(fen, move):
    board = chess.Board(fen)
    m = board.parse_san(move)
    gives_check = board.gives_check(m)
    board.push(m)
    return _json.dumps({"fen": board.fen(), "gives_check": gives_check})

_PIECE_NAMES = {"pawn": chess.PAWN, "knight": chess.KNIGHT, "bishop": chess.BISHOP,
                "rook": chess.ROOK, "queen": chess.QUEEN, "king": chess.KING}

def tool_get_squares(fen, piece, color=None):
    """Find all squares of a piece type, optionally by color."""
    board = chess.Board(fen)
    pt = _PIECE_NAMES.get(piece.lower())
    if pt is None:
        return _json.dumps({"error": f"unknown piece: {piece}"})
    colors = []
    if color is None:
        colors = [chess.WHITE, chess.BLACK]
    elif color == "white":
        colors = [chess.WHITE]
    elif color == "black":
        colors = [chess.BLACK]
    else:
        return _json.dumps({"error": f"unknown color: {color}"})
    results = []
    for c in colors:
        cname = "white" if c == chess.WHITE else "black"
        for sq in board.pieces(pt, c):
            results.append({"square": chess.square_name(sq), "color": cname})
    return _json.dumps({"piece": piece, "squares": results})

def tool_get_material(fen):
    """Get material count for both sides."""
    board = chess.Board(fen)
    result = {}
    for color_name, color in [("white", chess.WHITE), ("black", chess.BLACK)]:
        pieces = {}
        for pt, name in [(chess.PAWN, "pawns"), (chess.KNIGHT, "knights"),
                         (chess.BISHOP, "bishops"), (chess.ROOK, "rooks"),
                         (chess.QUEEN, "queens")]:
            count = len(board.pieces(pt, color))
            if count > 0:
                pieces[name] = count
        result[color_name] = pieces
    return _json.dumps(result)

def tool_compare_positions(fen_a, fen_b, depth=18):
    """Compare material counts and engine eval between two FENs."""
    _piece_vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                   chess.ROOK: 5, chess.QUEEN: 9}

    def _mat(board, color):
        return sum(len(board.pieces(pt, color)) * v for pt, v in _piece_vals.items())

    board_a = chess.Board(fen_a)
    board_b = chess.Board(fen_b)

    w_a, b_a = _mat(board_a, chess.WHITE), _mat(board_a, chess.BLACK)
    w_b, b_b = _mat(board_b, chess.WHITE), _mat(board_b, chess.BLACK)

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        info_a = engine.analyse(board_a, chess.engine.Limit(depth=int(depth)))
        sc_a = info_a['score'].white().score(mate_score=10000)
        info_b = engine.analyse(board_b, chess.engine.Limit(depth=int(depth)))
        sc_b = info_b['score'].white().score(mate_score=10000)

    return _json.dumps({
        "position_a": {"white_material": w_a, "black_material": b_a,
                        "eval": f"{sc_a/100:+.2f}"},
        "position_b": {"white_material": w_b, "black_material": b_b,
                        "eval": f"{sc_b/100:+.2f}"},
        "material_change": {"white": w_b - w_a, "black": b_b - b_a},
        "eval_change": f"{(sc_b - sc_a)/100:+.2f}",
    })


def tool_check_threat(fen, threat_move, max_defenses=5):
    """
    Check if a move is a genuine threat in the given position.

    A threat is a move that the side NOT currently to move wants to play on their
    NEXT turn (after the opponent moves). This function:
    1. Generates all legal moves for the side to move (opponent of threatening side)
    2. For each opponent move, checks if threat_move becomes legal and effective
    3. Returns whether the threat works and what defenses exist

    Args:
        fen: Position where opponent is to move
        threat_move: The threatened move (SAN or UCI) by the side NOT to move
        max_defenses: Maximum number of defensive moves to report

    Returns JSON with:
        - threat_viable: Can threat be played after opponent moves?
        - n_positions_checked: How many opponent moves were tried
        - n_threat_works: In how many positions does threat achieve its goal?
        - n_threat_fails: In how many positions does threat fail?
        - defenses: List of moves that defend against the threat
        - sample_success: Example continuation where threat works
        - threat_type: What the threat accomplishes (mate/material/positional)
    """
    board = chess.Board(fen)
    opponent_moves = list(board.legal_moves)

    if not opponent_moves:
        return _json.dumps({"error": "No legal moves for opponent (position is checkmate or stalemate)"})

    # Threatening side is opposite of current turn
    threatening_side = not board.turn

    threat_works_count = 0
    threat_fails_count = 0
    defenses = []
    sample_success = None
    threat_type = None

    for opp_move in opponent_moves:
        # Play opponent's move
        board_after = board.copy()
        board_after.push(opp_move)

        # Try to parse and play the threat move
        try:
            # Try SAN first, then UCI
            threat = None
            try:
                threat = board_after.parse_san(threat_move)
            except:
                try:
                    threat = chess.Move.from_uci(threat_move)
                except:
                    pass

            if threat is None:
                # Could not parse the threat move
                threat_fails_count += 1
                if len(defenses) < max_defenses:
                    defenses.append({
                        'move': board.san(opp_move),
                        'reason': f'Could not parse threat move: {threat_move}'
                    })
                continue

            if threat not in board_after.legal_moves:
                # Threat is not legal in this position
                threat_fails_count += 1
                defenses.append({
                    'move': board.san(opp_move),
                    'reason': f'Makes {threat_move} illegal'
                })
                continue

            # Play the threat move
            board_after.push(threat)

            # Evaluate what the threat accomplishes
            is_checkmate = board_after.is_checkmate()
            is_check = board_after.is_check()

            # Use engine to evaluate if needed
            threat_accomplishment = None
            if is_checkmate:
                threat_accomplishment = "checkmate"
                threat_works_count += 1
                if not threat_type:
                    threat_type = "mate"
                if not sample_success:
                    sample_success = f"{board.san(opp_move)} {board_after.san(threat)}"
            else:
                # Evaluate the position
                with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
                    info = engine.analyse(board_after, chess.engine.Limit(depth=18))
                    score = info.get("score")

                    if score and score.relative:
                        # Convert to centipawns from threatening side's perspective
                        cp = score.relative.score(mate_score=10000)
                        if threatening_side == chess.BLACK:
                            cp = -cp

                        # Check if threat wins material or gives strong advantage
                        if cp is not None:
                            if cp > 300:  # Winning advantage
                                threat_accomplishment = f"wins material/position (+{cp/100:.2f})"
                                threat_works_count += 1
                                if not threat_type:
                                    threat_type = "material" if cp > 200 else "positional"
                                if not sample_success:
                                    sample_success = f"{board.san(opp_move)} {board_after.san(threat)}"
                            elif cp > 100:  # Strong advantage
                                threat_accomplishment = f"strong advantage (+{cp/100:.2f})"
                                threat_works_count += 1
                                if not threat_type:
                                    threat_type = "positional"
                                if not sample_success:
                                    sample_success = f"{board.san(opp_move)} {board_after.san(threat)}"
                            else:
                                # Threat doesn't accomplish much
                                threat_fails_count += 1
                                if len(defenses) < max_defenses:
                                    defenses.append({
                                        'move': board.san(opp_move),
                                        'reason': f'Threat gives only +{cp/100:.2f}'
                                    })

        except Exception as e:
            # Threat move invalid or error in evaluation
            threat_fails_count += 1
            if len(defenses) < max_defenses:
                defenses.append({
                    'move': board.san(opp_move),
                    'reason': f'Error: {str(e)[:50]}'
                })

    # Determine if threat is viable
    threat_viable = threat_works_count > 0
    threat_percentage = (threat_works_count / len(opponent_moves) * 100) if opponent_moves else 0

    return _json.dumps({
        "threat_viable": threat_viable,
        "n_positions_checked": len(opponent_moves),
        "n_threat_works": threat_works_count,
        "n_threat_fails": threat_fails_count,
        "threat_percentage": f"{threat_percentage:.1f}%",
        "defenses": defenses[:max_defenses],
        "sample_success": sample_success,
        "threat_type": threat_type,
        "summary": f"Threat {'works' if threat_viable else 'fails'}: accomplishes goal in {threat_works_count}/{len(opponent_moves)} positions"
    })


TOOL_FUNCTIONS = {
    "get_legal_moves": tool_get_legal_moves, "get_piece_at": tool_get_piece_at,
    "get_attacks": tool_get_attacks, "get_attackers": tool_get_attackers,
    "count_attackers_defenders": tool_count_attackers_defenders,
    "check_ray_alignment": tool_check_ray_alignment,
    "is_pinned": tool_is_pinned, "is_check": tool_is_check,
    "try_variation": tool_try_variation, "get_engine_eval": tool_get_engine_eval,
    "get_top_moves": tool_get_top_moves, "eval_move": tool_eval_move,
    "compare_moves": tool_compare_moves, "make_move": tool_make_move,
    "get_squares": tool_get_squares, "get_material": tool_get_material,
    "compare_positions": tool_compare_positions,
    "check_threat": tool_check_threat,
}

def _schema(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}

_fen_prop = {"fen": {"type": "string", "description": "FEN of the position"}}
_sq_prop = {"square": {"type": "string", "description": "Square name, e.g. 'd3'"}}

TOOL_SCHEMAS_OPENAI = [
    _schema("get_legal_moves", "Get all legal moves in SAN.", _fen_prop, ["fen"]),
    _schema("get_piece_at", "Get the piece on a square.", {**_fen_prop, **_sq_prop}, ["fen", "square"]),
    _schema("get_attacks", "Get squares attacked by piece on a square.", {**_fen_prop, **_sq_prop}, ["fen", "square"]),
    _schema("get_attackers", "Get pieces of a color attacking a square.",
            {**_fen_prop, **_sq_prop, "color": {"type": "string", "enum": ["white", "black"]}},
            ["fen", "square", "color"]),
    _schema("count_attackers_defenders", "Count attackers and defenders of a square (both colors). Use this to verify 'removes a defender' or 'adds an attacker' claims.",
            {**_fen_prop, **_sq_prop}, ["fen", "square"]),
    _schema("check_ray_alignment", "Check if two squares are on the same rank/file/diagonal. Use to verify 'on the same diagonal', 'in line with' claims.",
            {"square_a": {"type": "string"}, "square_b": {"type": "string"}}, ["square_a", "square_b"]),
    _schema("is_pinned", "Check if a piece is pinned. Returns the pinner (attacking piece) and pinned-to piece (protected piece, usually king/queen).", {**_fen_prop, **_sq_prop}, ["fen", "square"]),
    _schema("is_check", "Check if side to move is in check.", _fen_prop, ["fen"]),
    _schema("try_variation", "Try a sequence of SAN moves. Returns resulting FEN.",
            {**_fen_prop, "moves": {"type": "array", "items": {"type": "string"}}}, ["fen", "moves"]),
    _schema("get_engine_eval", "Get Stockfish eval + best move.",
            {**_fen_prop, "depth": {"type": "integer"}}, ["fen"]),
    _schema("get_top_moves", "Get top N engine moves with evals and PV lines.",
            {**_fen_prop, "n": {"type": "integer"}, "depth": {"type": "integer"}}, ["fen"]),
    _schema("eval_move", "Evaluate a specific move: eval, best move, wp_loss, quality.",
            {**_fen_prop, "move": {"type": "string"}, "depth": {"type": "integer"}}, ["fen", "move"]),
    _schema("compare_moves", "Compare eval of two moves.",
            {**_fen_prop, "move_a": {"type": "string"}, "move_b": {"type": "string"},
             "depth": {"type": "integer"}}, ["fen", "move_a", "move_b"]),
    _schema("compare_positions", "Compare material and eval between two positions.",
            {"fen_a": {"type": "string", "description": "FEN of position A"},
             "fen_b": {"type": "string", "description": "FEN of position B"},
             "depth": {"type": "integer", "description": "Engine depth (default 18)"}},
            ["fen_a", "fen_b"]),
    _schema("make_move", "Make a move, return new FEN.",
            {**_fen_prop, "move": {"type": "string"}}, ["fen", "move"]),
    _schema("get_squares", "Find all squares of a piece type, optionally by color. Use when square not specified (e.g., 'a knight is pinned').",
            {**_fen_prop, "piece": {"type": "string", "enum": ["pawn","knight","bishop","rook","queen","king"]},
             "color": {"type": "string", "enum": ["white", "black"]}}, ["fen", "piece"]),
    _schema("get_material", "Get material count for both sides.", _fen_prop, ["fen"]),
    _schema("check_threat", "Check if a move is a genuine threat. Use this to verify claims like 'threatens Qg7 mate' or 'threatens to win the bishop'. The threat is a move the side NOT to move wants to play on their next turn.",
            {**_fen_prop,
             "threat_move": {"type": "string", "description": "The threatened move in SAN (e.g., 'Qg7') or UCI"},
             "max_defenses": {"type": "integer", "description": "Max number of defensive moves to report (default 5)"}},
            ["fen", "threat_move"]),
]

TOOL_SCHEMAS_ANTHROPIC = [
    {"name": s["function"]["name"], "description": s["function"]["description"],
     "input_schema": s["function"]["parameters"]} for s in TOOL_SCHEMAS_OPENAI
]

def execute_tool(name, arguments):
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return _json.dumps({"error": f"unknown tool: {name}"})
    try:
        return fn(**arguments)
    except Exception as e:
        return _json.dumps({"error": str(e)})