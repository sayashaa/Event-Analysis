import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from japanese_font_setup import ensure_japanese_font

# ==========================================================
# ✅ FONT SETUP
# ==========================================================
ensure_japanese_font()

# ==========================================================
# ✅ CONFIG
# ==========================================================
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Automatically find keyword summary file
keyword_summary_file = [
    f for f in os.listdir(OUTPUT_FOLDER)
    if "keyword_summary" in f.lower() and f.endswith(".csv")
]
if not keyword_summary_file:
    raise FileNotFoundError("❌ 'keyword_summary' ファイルが output フォルダ内に見つかりません。")
keyword_summary_file = os.path.join(OUTPUT_FOLDER, keyword_summary_file[0])

df = pd.read_csv(keyword_summary_file)
print(f"📄 Loaded: {keyword_summary_file}")
print("🧾 Columns detected:", list(df.columns))

# ==========================================================
# ✅ COLUMN NORMALIZATION
# ==========================================================
colmap = {
    "Category": next((c for c in df.columns if "category" in c.lower()), None),
    "Keyword": next((c for c in df.columns if c.lower() in ["keyword", "parent"]), None),
    "Participant": next((c for c in df.columns if "participant" in c.lower()), None),
    "Total": next((c for c in df.columns if "total" in c.lower() and "weight" not in c.lower()), None),
    "Weighted": next((c for c in df.columns if "weight" in c.lower()), None),
}

for key, col in colmap.items():
    if col is None:
        raise KeyError(f"❌ 必要な列 '{key}' がCSVに見つかりません。")

df.rename(columns={
    colmap["Category"]: "Category",
    colmap["Keyword"]: "Keyword",
    colmap["Participant"]: "Participant",
    colmap["Total"]: "Total_Count",
    colmap["Weighted"]: "Weighted_Count"
}, inplace=True)

# ==========================================================
# ✅ TOP-7 KEYWORDS PER CATEGORY
# ==========================================================
top_keywords_by_cat = (
    df.groupby(["Category", "Keyword"], as_index=False)[["Total_Count", "Weighted_Count"]]
    .sum()
    .sort_values(["Category", "Total_Count"], ascending=[True, False])
    .groupby("Category")
    .head(7)
)

# ==========================================================
# ✅ COMBINE TOP + BOTTOM INTO ONE FIGURE
# ==========================================================
for category in top_keywords_by_cat["Category"].unique():
    cat_top = top_keywords_by_cat[top_keywords_by_cat["Category"] == category].copy()
    cat_df = df[df["Category"] == category].copy()

    # Sort descending by Total_Count
    cat_top = cat_top.sort_values("Total_Count", ascending=False)
    keywords = cat_top["Keyword"].tolist()

    # ======================================================
    # TOP GRAPH DATA
    # ======================================================
    x = np.arange(len(keywords))
    width = 0.35
    total_vals = cat_top["Total_Count"].tolist()
    weighted_vals = cat_top["Weighted_Count"].tolist()

    # Determine gray for zero-weight bars
    bar_colors = ["#3A78C2" if w > 0 else "gray" for w in weighted_vals]

    # ======================================================
    # BOTTOM GRAPH DATA (Top speaker per keyword)
    # ======================================================
    bottom_records = []
    for kw in keywords:
        kw_df = cat_df[cat_df["Keyword"] == kw]
        if kw_df.empty:
            continue

        # find top total count
        max_total = kw_df["Total_Count"].max()
        top_speakers = kw_df[kw_df["Total_Count"] == max_total]

        # remove duplicate participant names while keeping order
        unique_speakers = list(dict.fromkeys(top_speakers["Participant"].tolist()))
        speakers = "+".join(unique_speakers)

        total_sum = top_speakers["Total_Count"].sum()
        weighted_sum = top_speakers["Weighted_Count"].sum()

        bottom_records.append({
            "Keyword": kw,
            "Speakers": speakers,
            "Total_Top": total_sum,
            "Weighted_Top": weighted_sum
        })

    bottom_df = pd.DataFrame(bottom_records)

    # ======================================================
    # CREATE COMBINED FIGURE
    # ======================================================
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi=150, sharex=False)
    top_ax, bottom_ax = axes

    # ---- TOP GRAPH ----
    top_ax.bar(
        x - width/2, total_vals, width,
        label="実際の出現数", color="#3A78C2", edgecolor="black"
    )
    top_ax.bar(
        x + width/2, weighted_vals, width,
        label="重み付き出現数", color="#C23A3A", edgecolor="black"
    )
    for i, color in enumerate(bar_colors):
        if color == "gray":
            top_ax.bar(x[i] - width/2, total_vals[i], width,
                       color="gray", edgecolor="black", label=None)

    top_ax.set_ylabel("出現数", fontweight="bold")
    top_ax.set_xticks(x)
    top_ax.set_xticklabels(keywords, rotation=25, ha="right", fontsize=10)
    top_ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.18),
        ncol=2, frameon=False, fontsize=11
    )
    top_ax.set_title(f"{category}カテゴリ｜上位キーワードと上位発言者貢献度", pad=60, fontweight="bold")

    # ---- BOTTOM GRAPH ----
    bx = np.arange(len(bottom_df))
    bottom_ax.bar(
        bx - width/2, bottom_df["Total_Top"], width,
        label="上位発言者の出現数", color="#3A78C2", edgecolor="black"
    )
    bottom_ax.bar(
        bx + width/2, bottom_df["Weighted_Top"], width,
        label="上位発言者の重み付き出現数", color="#C23A3A", edgecolor="black"
    )

    # Bottom axis = top speakers (unique, with '+')
    bottom_ax.set_xticks(bx)
    bottom_ax.set_xticklabels(bottom_df["Speakers"], rotation=25, ha="right", fontsize=10)

    bottom_ax.set_ylabel("出現数", fontweight="bold")
    bottom_ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.15),
        ncol=2, frameon=False, fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(OUTPUT_FOLDER, f"{category}_combined.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved combined final graph: {out_path}")

print("🎉 全カテゴリの上下統合グラフを出力フォルダに保存しました（重複発言者修正済み・表示最適化）。")
