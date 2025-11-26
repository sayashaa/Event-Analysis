import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================
# 1. CONFIG
# =========================================
INPUT_CSV = "output/all_questions_with_8responses.csv"  # run from scripts/
SAVE_PLOT = "output/stall_score_chart_q017.png"

# =========================================
# 2. LOAD Q017 RESPONSES
# =========================================
df = pd.read_csv(INPUT_CSV)

q17_rows = df[df["Question_id"] == "Q017"].copy()
q17_rows = q17_rows.sort_values("Type")  # QUESTION, RESPONSE_1..8

responses = q17_rows[q17_rows["Type"].str.startswith("RESPONSE")]["Text"].tolist()

print("Found", len(responses), "responses for Q017\n")

# =========================================
# 3. KEYWORD → STALL DICTIONARY (FROM RESPONSES)
# =========================================
keyword_to_stalls = {
    "flower shop": [1], "flowers": [1], "flower": [1],

    "shop in the middle": [4, 5], "middle shop": [4, 5],
    "food shop": [6, 7, 8],

    "fisherman’s marinated rice bowl": [8],
    "fisherman's marinated rice bowl": [8],
    "marinated rice bowl": [8], "marinated fish bowl": [8],
    "rice bowl": [8], "hot sandwiches": [8],
    "hot sandwich": [8], "hot snacks": [8],

    "bread and fish": [8], "bread": [8],
    "fish shops": [8], "fish shop": [8], "fish": [8],

    "kakigori": [4], "shaved ice shop": [4], "shaved ice": [4],

    "vegetable stalls": [2, 3], "vegetable stall": [2, 3],
    "vegetable stand": [2, 3], "vegetable shop": [2, 3],
    "vegetables": [2, 3],

    "sweets": [5], "dessert": [5], "cookies": [5],

    "bento": [6], "food stalls": [6, 7],

    # ignore
    "coffee shop": [], "popcorn": [],
}

print("=== Keyword Dictionary ===")
for k, v in keyword_to_stalls.items():
    print(f"{k:25s} -> {v}")
print()

# =========================================
# 4. MAP ONE RESPONSE TO STALL ORDER
# =========================================
def extract_stall_order(text: str):
    t = str(text).lower()
    hits = []

    for kw, stalls in keyword_to_stalls.items():
        if not stalls:
            continue
        idx = t.find(kw)
        if idx != -1:
            hits.append((idx, kw, stalls))

    hits.sort(key=lambda x: x[0])

    order = []
    for idx, kw, stalls in hits:
        for s in stalls:
            if s not in order:
                order.append(s)

    return order

# =========================================
# 5. GET 8 ORDERS
# =========================================
stall_orders = []
print("=== Stall order per response ===")
for i, txt in enumerate(responses, start=1):
    order = extract_stall_order(txt)
    stall_orders.append(order)
    print(f"Response {i}: {order}")

# =========================================
# 6. SCORE & RANK
# =========================================
def score_order(order):
    s_dict = {}
    n = len(order)
    for rank, stall in enumerate(order, start=1):
        s_dict[stall] = s_dict.get(stall, 0) + (n - rank + 1)
    return s_dict

scores = {s: 0 for s in range(1, 9)}
first_choice = {s: 0 for s in range(1, 9)}

for order in stall_orders:
    sc = score_order(order)
    for st, val in sc.items():
        scores[st] += val
    if order:
        first_choice[order[0]] += 1

print("\n=== Total Scores ===")
for s in range(1, 9):
    print(f"Stall {s}: score={scores[s]}, first-choice={first_choice[s]}")

ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("\n=== Ranking ===")
for i, (stall, sc) in enumerate(ranking, start=1):
    print(f"{i}. Stall {stall} (score={sc}, first-choice={first_choice[stall]})")

# =========================================
# 7. BAR CHART (DESCENDING BY SCORE) + SAVE
# =========================================
import os

# sort stalls by score (high → low)
ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
sorted_stalls = [s for s, sc in ranking]
sorted_scores = [sc for s, sc in ranking]

plt.figure(figsize=(8, 4))
plt.bar(range(len(sorted_stalls)), sorted_scores)
plt.xticks(range(len(sorted_stalls)), sorted_stalls)
plt.xlabel("Stall ID")
plt.ylabel("Score")
plt.title("Stall Preference Ranking")

plt.tight_layout()

# make sure output folder exists
os.makedirs("output", exist_ok=True)
plt.savefig("output/stall_score_chart_q017.png", dpi=300)

plt.show()
print("Saved bar chart to: output/stall_score_chart_q017.png")

