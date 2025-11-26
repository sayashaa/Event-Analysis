# ============================================================
# scripts/main.py
# Stall-based text analysis + spider charts
# ============================================================
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 0. PATH SETTINGS  (data/ and output/ are inside scripts/)
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(THIS_DIR, "data")
OUTPUT_DIR = os.path.join(THIS_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STALL_FILE = os.path.join(DATA_DIR, "stall_keyword_dictionary.csv")
RESPONSES_FILE = os.path.join(DATA_DIR, "ev1_phase2_local_final.csv")
QUESTIONS_FILE = os.path.join(DATA_DIR, "question_list_ev1_phase2_local.csv")

# ============================================================
# 1. LOAD FILES
# ============================================================
stall_df = pd.read_csv(STALL_FILE)
responses_df = pd.read_csv(RESPONSES_FILE)
questions_df = pd.read_csv(QUESTIONS_FILE)

# Keep only participant responses (P1, P2, ...)
responses_df = responses_df[responses_df["Speaker"].str.startswith("P")].copy()
responses_df["Content_clean"] = responses_df["Content"].str.lower()

# ============================================================
# 2. CATEGORY DEFINITIONS (STALL-BASED ONLY)
# ============================================================
# Stall-based categories:
# SB01 ... Visibility
# SB02 ... Attractiveness
# SB03 ... Family Friendliness & Comfiness
# SB04 ... Improvement Priority Level
STALL_CATEGORY_IDS = ["SB01", "SB02", "SB03", "SB04"]

# Map Category_ID -> Category label from question list
cat_label_map = (
    questions_df[["Category_ID", "Category"]]
    .drop_duplicates()
    .set_index("Category_ID")["Category"]
    .to_dict()
)

# Keep only questions that belong to stall-based categories
stall_q = questions_df[questions_df["Category_ID"].isin(STALL_CATEGORY_IDS)].copy()

# Attach Category_ID to each response
responses_sb = responses_df.merge(
    stall_q[["Question_id", "Category_ID"]],
    on="Question_id",
    how="inner",
)

# ============================================================
# 3. PARSE STALL KEYWORDS
# ============================================================
def parse_keyword_cell(cell):
    """Split 'a;b;c' into ['a','b','c'] and lowercase."""
    import pandas as _pd
    if _pd.isna(cell):
        return []
    return [kw.strip().lower() for kw in str(cell).split(";") if kw.strip()]

stall_keywords = {}
for _, row in stall_df.iterrows():
    stall_id = row["stall_id"]
    stall_keywords[stall_id] = {
        "name": row["stall_name"],
        "strong": parse_keyword_cell(row["strong_keywords"]),
        "context": parse_keyword_cell(row["context_keywords"]),
    }

# Generic words indicating a stall context
SHOP_WORDS = ["shop", "stall", "stand", "store", "place", "booth"]

def score_response_for_stall(text, stall_info):
    """
    Base scoring:
      - strong keyword: +1.0
      - context keyword: +0.5 (only if some 'shop' word present)
    """
    if not isinstance(text, str):
        return 0.0

    t = text.lower()
    score = 0.0

    # Strong keywords
    for kw in stall_info["strong"]:
        if kw in t:
            score += 1.0

    # Context keywords (only when clearly discussing a stall/shop)
    has_shop_word = any(sw in t for sw in SHOP_WORDS)
    if has_shop_word:
        for kw in stall_info["context"]:
            if kw in t:
                score += 0.5

    return score

# ============================================================
# 4. SPECIAL LOGIC FOR SB04 (IMPROVEMENT PRIORITY LEVEL)
# ============================================================
IMPROVEMENT_PHRASES = [
    "need to ", "needs to ", "need more", "needs more",
    "need to be more", "there need to be more", "there needs to be more",
    "could make this place", "could make the place", "could be more",
    "more attractive", "more appealing",
    "it would be better", "would be better", "better if",
    "it would be more",
    "i would add", "i would change", "i would increase",
    "i would put", "i would choose", "i would place", "i would space",
    "add more", "would add", "should add", "should be",
    "i think it would be better", "i think it would be more",
    "they could make it", "i would make", "i’d make",
]

NEGATIVE_PHRASES = [
    "dangerous", "unsafe", "risky", "risk", "worried", "concerned", "concerning",
    "could be dangerous",
    "hard to tell", "hard to see", "bit hard to see", "hard to understand",
    "not enough", "not many", "not much", "isn't much", "isn’t much",
    "aren't many", "aren’t many",
    "boring", "wouldn't go", "would not go",
    "i probably wouldn’t go", "i probably wouldn't go",
    "difficult", "tough", "hard to stay",
]

def is_improvement_related(text: str) -> bool:
    """Return True if the response clearly talks about issues or improvements."""
    if not isinstance(text, str):
        return False
    t = text.lower()
    for phrase in IMPROVEMENT_PHRASES + NEGATIVE_PHRASES:
        if phrase in t:
            return True
    return False

# ============================================================
# 5. BUILD RAW SCORE TABLE (stall_id x Category_ID)
# ============================================================
stall_ids = stall_df["stall_id"].tolist()
category_ids = STALL_CATEGORY_IDS

index = pd.MultiIndex.from_product(
    [stall_ids, category_ids],
    names=["stall_id", "Category_ID"],
)
scores_df = pd.DataFrame(0.0, index=index, columns=["raw_score"])

for _, row in responses_sb.iterrows():
    text = row["Content_clean"]
    cat_id = row["Category_ID"]

    # For SB04, only when clearly improvement/issue related
    if cat_id == "SB04" and not is_improvement_related(text):
        continue

    for stall_id, info in stall_keywords.items():
        s = score_response_for_stall(text, info)
        if s > 0:
            scores_df.loc[(stall_id, cat_id), "raw_score"] += s

scores_df = scores_df.reset_index()

# ============================================================
# 6. NORMALIZE SCORES PER CRITERION (0–5 SCALE)
# ============================================================
for cid in category_ids:
    mask = scores_df["Category_ID"] == cid
    max_val = scores_df.loc[mask, "raw_score"].max()
    if max_val == 0:
        scores_df.loc[mask, "norm_score"] = 0.0
    else:
        scores_df.loc[mask, "norm_score"] = (
            scores_df.loc[mask, "raw_score"] / max_val * 5.0
        )

# Pivot to one row per stall_id
stall_scores = scores_df.pivot_table(
    index="stall_id",
    columns="Category_ID",
    values="norm_score",
    fill_value=0.0,
).reset_index()

# Attach stall names
stall_scores = stall_scores.merge(
    stall_df[["stall_id", "stall_name"]],
    on="stall_id",
    how="left",
)

# --------- identify "poor performance" stalls (no data) BEFORE merging -----
numeric_cols = [cid for cid in STALL_CATEGORY_IDS if cid in stall_scores.columns]
poor_mask = stall_scores[numeric_cols].sum(axis=1) == 0
poor_stalls = stall_scores.loc[poor_mask, "stall_name"].tolist()

# Save scores table (original, before merge)
scores_out_path = os.path.join(OUTPUT_DIR, "stall_scores.csv")
stall_scores.to_csv(scores_out_path, index=False, encoding="utf-8-sig")

# ============================================================
# 6b. PREPARE DATA FOR PLOTTING (MERGE STALL 2 & 3, REMOVE ZERO STALLS)
# ============================================================
plot_df = stall_scores.copy()

# helper: convert stall_id to int where possible
def _to_int_safe(x):
    try:
        return int(x)
    except Exception:
        return None

plot_df["stall_id_int"] = plot_df["stall_id"].apply(_to_int_safe)

# merge stall 2 and 3 into Vegetable Stall (2 & 3)
has2 = (plot_df["stall_id_int"] == 2).any()
has3 = (plot_df["stall_id_int"] == 3).any()

if has2 and has3:
    subset_23 = plot_df[plot_df["stall_id_int"].isin([2, 3])]
    if not subset_23.empty:
        combined = {"stall_id": "2_3", "stall_name": "Vegetable Stall (Stalls 2 & 3)"}
        for cid in STALL_CATEGORY_IDS:
            if cid in plot_df.columns:
                combined[cid] = subset_23[cid].mean()
        combined_row = pd.DataFrame([combined])

        # drop original 2 & 3
        plot_df = plot_df[~plot_df["stall_id_int"].isin([2, 3])]
        # drop helper col before concat
        plot_df = plot_df.drop(columns=["stall_id_int"])
        plot_df = pd.concat([plot_df, combined_row], ignore_index=True)
else:
    # if not both present, just drop helper col
    plot_df = plot_df.drop(columns=["stall_id_int"])

# keep only stalls that actually have some score (data only)
numeric_cols_plot = [cid for cid in STALL_CATEGORY_IDS if cid in plot_df.columns]
plot_df = plot_df[plot_df[numeric_cols_plot].sum(axis=1) > 0].copy()

# ============================================================
# 7. SPIDER (RADAR) CHART GENERATOR
# ============================================================
def plot_spider_for_stall(row, category_ids, cat_label_map, outdir):
    """
    Draw a spider chart for a single stall.
    category_ids: ["SB01","SB02","SB03","SB04"]
    cat_label_map: maps SBxx -> human label ("Visibility", ...)
    """
    # Values in fixed order
    values = [float(row.get(cid, 0.0)) for cid in category_ids]
    values += [values[0]]  # close loop

    # Angles
    angles = [
        n / float(len(category_ids)) * 2 * math.pi
        for n in range(len(category_ids))
    ]
    angles += [angles[0]]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    # Axis labels = Category names (Visibility, Attractiveness, ...)
    labels = [cat_label_map.get(cid, cid) for cid in category_ids]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 5)
    # Title uses only stall_name (no stall number)
    title = str(row["stall_name"])
    ax.set_title(title, y=1.1)

    sid = str(row["stall_id"])
    outpath = os.path.join(outdir, f"stall_{sid}_spider.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# 8. GENERATE SPIDER CHARTS FOR STALLS WITH DATA ONLY
# ============================================================
for _, row in plot_df.iterrows():
    plot_spider_for_stall(row, STALL_CATEGORY_IDS, cat_label_map, OUTPUT_DIR)

print("Processing completed.")
print(f"- Scores CSV: {scores_out_path}")
print(f"- Spider charts: PNGs in {OUTPUT_DIR}")

# Print poor performance stalls (no data in stall-based questions)
if poor_stalls:
    print("\nStalls with no detected mentions in stall-based questions (potentially poor impression):")
    for name in poor_stalls:
        print(f" - {name}")
else:
    print("\nAll stalls had at least some mentions in stall-based questions.")
