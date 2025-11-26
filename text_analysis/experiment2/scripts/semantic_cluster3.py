import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# =========================================
# 1. CONFIG
# =========================================
RESP_CSV = "output/all_questions_with_8responses.csv"
QLIST_CSV = "data/question_list_ev1_phase2_local.csv"
REP_CSV  = "data/ov0n_representative_comments_keywords.csv"  # put the CSV here
OUTPUT_DIR = "output/ov0n_top3_bars_horizontal"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# 2. LOAD DATA
# =========================================
df_resp = pd.read_csv(RESP_CSV)
df_q    = pd.read_csv(QLIST_CSV)
df_rep  = pd.read_csv(REP_CSV)

# df_resp: Type, Question_id, Text
# df_q:    Question_id, Question_list, Total Response, Category_ID, Category
# df_rep:  Category_ID, Category_Name, Question_ID, Representative_Comment, Keywords

# =========================================
# 3. FILTER TO OV0N QUESTIONS ONLY
# =========================================
df_ovq = df_q[df_q["Category_ID"].str.startswith("OV0")].copy()

# Merge category info into responses
df_join = df_resp.merge(
    df_ovq[["Question_id", "Category_ID", "Category"]],
    on="Question_id",
    how="inner"
)

# Keep only RESPONSE rows
df_join = df_join[df_join["Type"].str.startswith("RESPONSE")].copy()

# =========================================
# 4. PREP REPRESENTATIVE COMMENT → KEYWORDS MAP
# =========================================
rep_map = {}

for _, row in df_rep.iterrows():
    cat_id = row["Category_ID"]
    qid    = row["Question_ID"]
    rep_c  = row["Representative_Comment"]

    kw_raw = str(row["Keywords"]).split(";")
    keywords = [k.strip() for k in kw_raw if k.strip()]

    key = (cat_id, qid)
    rep_map.setdefault(key, []).append({
        "rep_comment": rep_c,
        "keywords": keywords
    })

# =========================================
# 5. LABEL RESPONSES WITH REPRESENTATIVE COMMENTS
# =========================================
def label_response(category_id, question_id, text):
    """
    Return list of representative comments that match the response text,
    based on keyword substring matching (case-insensitive).
    """
    key = (category_id, question_id)
    if key not in rep_map:
        return []

    t = str(text).lower()
    matched_comments = []

    for item in rep_map[key]:
        rep_c = item["rep_comment"]
        kws   = item["keywords"]

        for kw in kws:
            if not kw:
                continue
            if kw.lower() in t:
                matched_comments.append(rep_c)
                break

    return matched_comments

labels = []
for _, row in df_join.iterrows():
    cat_id = row["Category_ID"]
    qid    = row["Question_id"]
    txt    = row["Text"]
    labels.append(label_response(cat_id, qid, txt))

df_join["Matched_Comments"] = labels

# Optional: inspect labeling
os.makedirs("output", exist_ok=True)
df_join.to_csv("output/ov0n_labeled_responses.csv", index=False)

# =========================================
# 6. COUNT FREQUENCY PER (Category_ID, Question_ID, Representative_Comment)
# =========================================
freq = Counter()
cat_name_map = {}
qtext_map = {}

for _, row in df_join.iterrows():
    cat_id = row["Category_ID"]
    qid    = row["Question_id"]

    # category name (from question list)
    cat_name = df_ovq[df_ovq["Category_ID"] == cat_id]["Category"].iloc[0]
    cat_name_map[cat_id] = cat_name

    # question text
    qtext = df_ovq[df_ovq["Question_id"] == qid]["Question_list"].iloc[0]
    qtext_map[(cat_id, qid)] = qtext

    matched = row["Matched_Comments"]
    if not isinstance(matched, list):
        continue

    for rep_c in set(matched):
        freq[(cat_id, qid, rep_c)] += 1

# =========================================
# 7. PLOT HORIZONTAL BARS PER CATEGORY & QUESTION
# =========================================

cmap = cm.OrRd  # heatmap-style orange → red
categories = sorted({cat_id for (cat_id, _, _) in freq.keys()})

for cat_id in categories:
    # all questions in this category
    qids = sorted(df_ovq[df_ovq["Category_ID"] == cat_id]["Question_id"].unique())
    if not qids:
        continue

    # --- shared colour scale for this category ---
    cat_counts = [
        cnt for (c_id, _, _), cnt in freq.items()
        if c_id == cat_id
    ]
    if not cat_counts:
        continue

    cat_min = min(cat_counts)
    cat_max = max(cat_counts)

    # avoid zero-range
    vmin_cat = cat_min
    vmax_cat = cat_max
    if vmin_cat == vmax_cat:
        vmin_cat -= 0.5
        vmax_cat += 0.5

    norm_cat = mcolors.Normalize(vmin=vmin_cat, vmax=vmax_cat)

    n_q = len(qids)

    fig, axes = plt.subplots(
        nrows=n_q,
        ncols=1,
        figsize=(11, 1.5 * n_q + 1.8),
        constrained_layout=True
    )

    if n_q == 1:
        axes = [axes]

    cat_name = cat_name_map.get(cat_id, cat_id)
    fig.suptitle(cat_name, fontsize=22)

    # ------------ main bars -------------
    for ax, qid in zip(axes, qids):

        # Collect (rep_comment, count) for this category+question
        items = [
            (rep_c, cnt)
            for (c_id, q, rep_c), cnt in freq.items()
            if c_id == cat_id and q == qid
        ]

        if not items:
            ax.axis("off")
            continue

        # Sort by count DESCENDING so first is largest
        items.sort(key=lambda x: x[1], reverse=True)
        # Take top-3, still in DESC order: [high, mid, low]
        top3 = items[:3]

        rep_comments = [rc for rc, _ in top3]
        counts       = [cnt for _, cnt in top3]

        # colours from shared category norm
        colors = [cmap(norm_cat(c)) for c in counts]

        # ---- bars (slim, almost no vertical gap) ----
        n_bar = len(top3)
        y_pos = np.arange(n_bar)     # 0 = bottom, 1 = middle, 2 = top
        bar_height = 0.9             # nearly full height → minimal gap

        # Because y=0 is bottom, we want index 0 = highest frequency
        ax.barh(
            y_pos,
            [1.0] * n_bar,           # equal visual length
            color=colors,
            edgecolor="black",
            height=bar_height
        )

        ax.set_ylim(-0.5, n_bar - 0.5)
        ax.set_yticks([])

        # ---- question text ----
        qtext = qtext_map.get((cat_id, qid), "")
        ax.set_title(qtext, fontsize=13, pad=5)


        ax.set_xlim(0, 1)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

        # ---- label text: one line, centered inside bar ----
        mid = (cat_min + cat_max) / 2.0
        for i, (cnt, label) in enumerate(zip(counts, rep_comments)):
            text_color = "white" if cnt >= mid else "black"
            ax.text(
                0.5,            # middle of bar in x
                y_pos[i],
                label,
                ha="center",
                va="center",
                fontsize=10,
                color=text_color
            )

    # x-label only on bottom axis
    axes[-1].set_xlabel("Response")

    # ------------ single colourbar for whole category -------------
    sm_cat = cm.ScalarMappable(cmap=cmap, norm=norm_cat)
    sm_cat.set_array([])

    cbar = fig.colorbar(
        sm_cat,
        ax=axes,
        orientation="vertical",
        fraction=0.035,
        pad=0.02
    )
    cbar.set_label("Frequency")

    # ticks from 1 up to cat_max, but not below cat_min
    tick_start = max(1, cat_min)
    ticks = list(range(tick_start, cat_max + 1))
    if not ticks:  # safety
        ticks = [cat_min]
    cbar.set_ticks(ticks)

    # 1 at top, larger at bottom (keep your current colour scale direction)
    cbar.ax.invert_yaxis()

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"{cat_id}_top3_comments_horizontal.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved bar chart for {cat_id} ({cat_name}) -> {out_path}")
