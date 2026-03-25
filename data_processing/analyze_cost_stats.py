import argparse
import json
import math
import statistics as stats
from collections import Counter, defaultdict


def pct(values, p):
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values[int(k)])
    return float(values[f] * (c - k) + values[c] * (k - f))


def fmt_money(x):
    return f"${x:,.4f}"


def fmt_num(x):
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:,.2f}"


def summarize_group(rows, name):
    costs = [r["usage"].get("cost_usd", 0.0) for r in rows if r.get("usage")]
    prompt = [r["usage"].get("prompt_tokens", 0) for r in rows if r.get("usage")]
    completion = [r["usage"].get("completion_tokens", 0) for r in rows if r.get("usage")]
    rounds = [r["usage"].get("rounds", 0) for r in rows if r.get("usage")]
    print(f"\n== {name} ==")
    print(f"rows with usage: {len(costs):,}")
    if not costs:
        return
    print(f"total cost: {fmt_money(sum(costs))}")
    print(f"avg cost/position: {fmt_money(stats.mean(costs))}")
    print(f"median cost/position: {fmt_money(stats.median(costs))}")
    print(f"p90 cost/position: {fmt_money(pct(costs, 90))}")
    print(f"p99 cost/position: {fmt_money(pct(costs, 99))}")
    print(f"avg prompt tokens: {fmt_num(stats.mean(prompt))}")
    print(f"avg completion tokens: {fmt_num(stats.mean(completion))}")
    print(f"avg total tokens: {fmt_num(stats.mean([p + c for p, c in zip(prompt, completion)]))}")
    print(f"avg rounds: {fmt_num(stats.mean(rounds))}")


def main():
    ap = argparse.ArgumentParser(description="Analyze cost statistics from generate_atoms output JSONL.")
    ap.add_argument("path", help="Path to output JSONL file")
    ap.add_argument("--top", type=int, default=10, help="Show top N most expensive positions")
    args = ap.parse_args()

    rows = []
    with open(args.path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping bad JSON at line {lineno}: {e}")
                continue
            rows.append(row)

    with_usage = [r for r in rows if isinstance(r.get("usage"), dict)]
    without_usage = len(rows) - len(with_usage)

    print(f"Total rows: {len(rows):,}")
    print(f"Rows with usage: {len(with_usage):,}")
    print(f"Rows without usage: {without_usage:,}")

    if not with_usage:
        print("No usage field found. This file may have been generated before cost tracking was added.")
        return

    models = Counter((r["usage"].get("provider"), r["usage"].get("model")) for r in with_usage)
    print("\nModels/providers seen:")
    for (provider, model), cnt in models.most_common():
        print(f"  {provider}/{model}: {cnt:,}")

    total_prompt = sum(r["usage"].get("prompt_tokens", 0) for r in with_usage)
    total_completion = sum(r["usage"].get("completion_tokens", 0) for r in with_usage)
    total_tokens = sum(r["usage"].get("total_tokens", 0) for r in with_usage)
    total_rounds = sum(r["usage"].get("rounds", 0) for r in with_usage)
    total_cost = sum(r["usage"].get("cost_usd", 0.0) for r in with_usage)

    print("\n== Overall ==")
    print(f"total prompt tokens: {total_prompt:,}")
    print(f"total completion tokens: {total_completion:,}")
    print(f"total tokens: {total_tokens:,}")
    print(f"total rounds: {total_rounds:,}")
    print(f"estimated total cost: {fmt_money(total_cost)}")
    print(f"avg cost/position: {fmt_money(total_cost / len(with_usage))}")
    print(f"avg rounds/position: {total_rounds / len(with_usage):.2f}")

    summarize_group(with_usage, "All positions")

    by_quality = defaultdict(list)
    by_include = defaultdict(list)
    for r in with_usage:
        by_quality[r.get("quality", "unknown")].append(r)
        extracted = r.get("extracted") or {}
        by_include[bool(extracted.get("include", True))].append(r)

    for q in sorted(by_quality):
        summarize_group(by_quality[q], f"quality={q}")

    summarize_group(by_include.get(True, []), "include=true")
    summarize_group(by_include.get(False, []), "include=false")

    top = sorted(
        with_usage,
        key=lambda r: r["usage"].get("cost_usd", 0.0),
        reverse=True,
    )[: args.top]

    print(f"\n== Top {len(top)} most expensive positions ==")
    for i, r in enumerate(top, 1):
        u = r["usage"]
        pos = r.get("position_number")
        move = r.get("move_san") or r.get("move_uci")
        game = r.get("game", "?")
        quality = r.get("quality", "?")
        include = (r.get("extracted") or {}).get("include", True)
        print(
            f"{i:>2}. pos={pos} cost={fmt_money(u.get('cost_usd', 0.0))} "
            f"tokens={u.get('total_tokens', 0):,} rounds={u.get('rounds', 0)} "
            f"quality={quality} include={include} move={move} game={game}"
        )


if __name__ == "__main__":
    main()
