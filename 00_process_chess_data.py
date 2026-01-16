import io
import json
import re
import chess
import chess.pgn
from datasets import load_dataset
from tqdm import tqdm
import random
# --- Configuration ---
OUTPUT_FILE = "data/chess_annotation_filtered.jsonl"
MIN_WORD_COUNT = 5 
BAD_KEYWORDS = [
    "click", "subscribe", "video", "chapter", "lichess", "study", 
    "heart", "donate", "check out", "http", "www."
]
ENGLISH_STOP_WORDS = {"the", "is", "of", "to", "and", "a", "in", "that", "it", "for", "with", "as", "but", "on", "are", "this", "by", "an", "be", "at", 'move', 'piece', 'king', 'queen', 'rook', 'bishop', 'knight', 'pawn', 'check', 'mate', "my", "you", "he", "she", "they", "we", "his", "her", "its", "their", "yes", "no", 'game'}

# NEW: Phrases that are useless on their own
GENERIC_PHRASES = [
    "white is better", "black is better", "white is winning", "black is winning",
    "position is equal", "draw", "white has an advantage", "black has an advantage",
    "white resigns", "black resigns", "mate in", "checkmate"
]

TACTICAL_KEYWORDS = {
    "pin", "fork", "skewer", "hanging", "mate", "checkmate", "threat", 
    "tactic", "sacrifice", "blunder", "trap", "discovery", "discovered", 
    "double attack", "capture", "exchange", "forcing", "combination", 
    "intermezzo", "zwischenzug", "deflect", "decoy", "overloaded", 
    "perpetual", "stalemate", "promotion", "remove the defender", 
    "win material", "loose piece"
}

POSITIONAL_KEYWORDS = {
    "weakness", "weak", "outpost", "space", "structure", "control", "file", 
    "diagonal", "open file", "rank", "bishop pair", "develop", "development", 
    "prophylaxis", "prophylactic", "initiative", "maneuver", "pawn chain", 
    "isolated", "doubled", "backward", "hole", "square", "mobility", 
    "activity", "passive", "active", "coordination", "blockade", "block", 
    "compensation", "majority", "minority", "color complex", 
    "good bishop", "bad bishop", "tempo", "restrict", "bind"
}

def get_tags(comment):
    comment_lower = comment.lower()
    found_tags = []
    
    if any(re.search(rf"\b{kw}\b", comment_lower) for kw in TACTICAL_KEYWORDS):
        found_tags.append("Tactical")
        
    if any(re.search(rf"\b{kw}\b", comment_lower) for kw in POSITIONAL_KEYWORDS):
        found_tags.append("Positional")
        
    return found_tags

def is_high_quality(comment):

    # 1. Clean PGN artifacts (e.g., [%eval 0.3] [%clk 0:05:00])
    comment = re.sub(r"\[%.*?\]", "", comment).strip()
    
    # 2. Length Check
    words = comment.split()
    if len(words) < MIN_WORD_COUNT:
        return False
    
    # 3. Meta-data/Spam Check
    comment_lower = comment.lower()
    if any(kw in comment_lower for kw in BAD_KEYWORDS):
        return False
        
    # 4. Filter out pure move lists (e.g., "1. e4 e5 2. Nf3")
    # If > 50% of the text looks like algebraic notation, skip it.
    move_pattern = r"\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8]\b"
    moves_found = re.findall(move_pattern, comment)
    if len(moves_found) > len(words) * 0.5:
        return False
    
    #words_lower = set(w.lower() for w in words)
    #common_count = len(words_lower.intersection(ENGLISH_STOP_WORDS))
    #if common_count < 1:
    #    print(f"Skipped due to Not English/Garbage: {comment.strip()[:60]}...")
    #    return False
        
    return True

def clean_comment(comment):
    """Removes braces and PGN tags to leave just the text."""
    # Remove [%...] tags
    comment = re.sub(r"\[%.*?\]", "", comment)
    # Remove recursive braces if any
    comment = comment.replace("{", "").replace("}", "")
    return " ".join(comment.split())

def process_dataset():
    print("Loading dataset...")
    ds = load_dataset(
        "json",
        data_files="annotated_pgn/*.jsonl",
        split="train",
        streaming=True
    )
    #ds = ds.shuffle(buffer_size=10, seed=42)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        total_comments = 0
        high_quality_comments = 0
        comments_with_tags = 0
        for idx, row in enumerate(tqdm(ds, desc="Processing Games")):
            #if idx > 10: break
            pgn_text = row.get('text')

            if not pgn_text: 
                continue
            try:
                pgn_io = io.StringIO(pgn_text)
                game = chess.pgn.read_game(pgn_io)
                
                if not game: continue

                board = game.board()
                node = game
                
                while node.variations:
                    next_node = node.variations[0] # Follow main line
                    move = next_node.move
                    comment = next_node.comment
                    
                    if comment:
                        total_comments += 1
                    
                        if is_high_quality(comment):
                            high_quality_comments += 1
                            clean_text = clean_comment(comment)
                            tags = get_tags(clean_text)
                            if len(tags) > 0:
                                comments_with_tags += 1
                            
                            entry = {
                                "id": f"game{idx}_move{board.fullmove_number}_{'w' if board.turn else 'b'}",
                                "game_idx": idx,
                                "fen": board.fen(),
                                "move": board.san(move),
                                "move_uci": move.uci(),
                                "explanation": clean_text,
                                "tags": tags,
                                "event": game.headers.get("Event", "?"),
                                "annotator": game.headers.get("Annotator", "?"),
                                "link": game.headers.get("Site", "?")
                            }
                            f_out.write(json.dumps(entry) + "\n")
                        else:
                            if random.random() < 0.0:  # Log a few skipped comments for debugging
                                print(f"Skipped comment: {comment.strip()[:60]}...")
                    # Advance board and node
                    board.push(move)
                    node = next_node

            except Exception as e:
                # parsing errors are common in dirty PGNs, skip bad games
                continue

    print(f"Finished. High-quality data saved to {OUTPUT_FILE}")
    print(f"Total comments processed: {total_comments}")
    print(f"High-quality comments retained: {high_quality_comments}")
    print(f"Comments with tags: {comments_with_tags}")

def keep_tagged_dataset():
    input_file = OUTPUT_FILE
    output_file = "chess_reasoning_tagged.jsonl"
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            entry = json.loads(line)
            if entry.get("tags") and len(entry["tags"]) > 0:
                f_out.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    process_dataset()
    #keep_tagged_dataset()