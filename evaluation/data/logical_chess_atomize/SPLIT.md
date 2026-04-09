# Train / Test Split

Game-wise split of `included_filtered.jsonl` (1283 positions, 32 games).
No game appears in both splits — zero data leakage.

- **Seed:** `random.Random(99)`
- **Algorithm:** Shuffle games, take first 6 as test
- **Source:** `included_filtered.jsonl` (post-filter extraction output)

## Summary

| Split | Games | Positions |
|-------|------:|----------:|
| Train | 26    | 1054      |
| Test  | 6     | 229       |
| Total | 32    | 1283      |

## Test Games (filter these out of training)

| Game | Positions |
|------|----------:|
| Flohr – Pitschak | 30 |
| Janowsky – Alapin | 66 |
| Liubarski – Soultanbeieff | 21 |
| Pillsbury – Marco | 35 |
| Rubinstein – Maroczy | 55 |
| Scheve – Teichmann | 22 |

## Train Games

| Game | Positions |
|------|----------:|
| Alekhine – Poindle | 37 |
| Bernstein – Mieses | 64 |
| Blackburne – Blanchard | 24 |
| Canal – Capablanca | 75 |
| Capablanca – Mattison | 29 |
| Capablanca – Villegas | 45 |
| Chekhover – Rudakowsky | 54 |
| Chernev – Hahlbohm | 30 |
| Colle – Delvaux | 30 |
| Dobias – Podgorny | 27 |
| Grunfeld – Schenkein | 31 |
| Havasi – Capablanca | 44 |
| Marshall – Tarrasch | 49 |
| Noteboom – Doesburgh | 46 |
| Phillsbury – Mason | 51 |
| Pitschak – Flohr | 33 |
| Przepiorka – Prokes | 32 |
| Rubinstein – Salwe | 54 |
| Ruger – Gebhard | 21 |
| Spielmann – Wahle | 24 |
| Tarrasch – Eckart | 23 |
| Tarrasch – Kurschner | 31 |
| Tarrasch – Mieses | 89 |
| Vliet – Znosko-Borovsky | 46 |
| Zeissl – Walthoffen | 19 |
| Znosko-Borovsky – Mackenzie | 46 |

## Reproducing the Split

```python
import json, random
from collections import defaultdict

with open("included_filtered.jsonl") as f:
    rows = [json.loads(l) for l in f]

by_game = defaultdict(list)
for r in rows:
    by_game[r["game"]].append(r)

games = list(by_game.keys())
rng = random.Random(99)
rng.shuffle(games)

test_games = set(games[:6])
```
