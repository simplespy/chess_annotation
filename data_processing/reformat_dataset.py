import json
data_path = 'data/1k_other_move.jsonl'
save_path = 'data/online1k_other_move.jsonl'
move_list = []
with open(data_path) as f:
    for line in f:
        try:
            entry = json.loads(line)
            metadata = {
                "Event": entry.get("event", "?"), 
                "Annotator": entry.get("annotator", "?"), 
                "Link": entry.get("link", "?"),
                "SLM_tag": entry.get("slm_tag", []),
                "SLM_score": entry.get("slm_score", 0)
            }
            entry_formatted = {
                "game_id": entry["id"],
                "fen": entry["fen"],
                "move_uci": entry["move_uci"],
                "move_san": entry["move"],
                "annotation": entry["annotation"],
                "best_move_uci": entry["best_move_uci"],
                "best_move_san": entry["best_move_san"],
                "wp_loss": entry["wp_loss"],
                "is_top_engine_move": entry["is_top_engine_move"],
                "metadata": metadata
            }
            move_list.append(entry_formatted)
        except:
            print(line)


with open(save_path, "w") as f:
    for e in move_list:
        f.write(json.dumps(e) + "\n")
