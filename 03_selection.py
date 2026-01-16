import pandas as pd
import ast
def select_best():
    df = pd.read_json("data/step2_tagged.jsonl", lines=True)
    
    # 1. Filter: Keep only high scores
    high_quality = df[df["slm_score"] >= 5]
    
    # 2. Stratify
    tactical = high_quality[high_quality["slm_tag"] == "Tactical"]
    positional = high_quality[high_quality["slm_tag"] == "Positional"]
    
    # 3. Sample (e.g. 5k each)
    # Using 'sample' randomly picks rows if we have more than needed
    best_tactical = tactical.sample(n=min(len(tactical), 500), random_state=42)
    best_positional = positional.sample(n=min(len(positional), 500), random_state=42)
    
    final_set = pd.concat([best_tactical, best_positional])
    
    final_set.to_json("data/platinum_1k.jsonl", orient="records", lines=True)
    print(f"Created dataset with {len(final_set)} records.")

def select_best_qwen3():
    df = pd.read_json("data/step2_qwen3_scored.jsonl", lines=True)
    
    # 1. Filter: Keep only high scores
    high_quality = df[df["slm_score"] >= 4]
    
    # 2. Stratify
    #tactical = high_quality[high_quality["slm_tags"] == ["Tactical"]]
    #positional = high_quality[high_quality["slm_tags"] == ["Positional"]]
    #both = high_quality[high_quality["slm_tags"] == ["Tactical", "Positional"]]


    def normalize_tags(x):
        # Handle NaN / None
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ()

        # If it's a stringified list, parse it
        if isinstance(x, str):
            x = x.strip()
            if x.startswith("[") and x.endswith("]"):
                try:
                    x = ast.literal_eval(x)
                except Exception:
                    # Fall back: treat as a single tag string
                    x = [x]
            else:
                # Plain string tag
                x = [x]

        # Now x should be list-like or something iterable; normalize
        if isinstance(x, (list, tuple, set)):
            return tuple(sorted(x))
        else:
            return (str(x),)

    # 1) Create a normalized key column
    high_quality = high_quality.copy()
    high_quality["slm_tags_key"] = high_quality["slm_tags"].apply(normalize_tags)

    # 2) Check what exists
    print(high_quality["slm_tags_key"].value_counts())

    # (Optional) see all unique combos
    print(sorted(high_quality["slm_tags_key"].unique()))

    # 3) Select the three you want
    tactical   = high_quality[high_quality["slm_tags_key"].apply(lambda t: t == ("Tactical",))]
    positional = high_quality[high_quality["slm_tags_key"].apply(lambda t: t == ("Positional",))]
    both       = high_quality[high_quality["slm_tags_key"].apply(lambda t: t == ("Positional", "Tactical"))]


    
    # 3. Sample (e.g. 5k each)
    # Using 'sample' randomly picks rows if we have more than needed
    best_tactical = tactical.sample(n=min(len(tactical), 300), random_state=42)
    best_positional = positional.sample(n=min(len(positional), 300), random_state=42)
    best_both = both.sample(n=min(len(both), 400), random_state=42)
    
    final_set = pd.concat([best_tactical, best_positional, best_both])
    
    final_set.to_json("data/platinum_qwen3_1k.jsonl", orient="records", lines=True)
    print(f"Created dataset with {len(final_set)} records.")

if __name__ == "__main__":
    #select_best()
    select_best_qwen3() 