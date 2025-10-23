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
    "Total": next((c for c in df.columns if "total" in c.lower() and "weight" not in c.lower()), None),
    "Weighted": next((c for c in df.columns if "weight" in c.lower()), None),
}

for key, col in colmap.items():
    if col is None:
        raise KeyError(f"❌ 必要な列 '{key}' がCSVに見つかりません。")

df.rename(columns={
    colmap["Category"]: "Category",
    colmap["Keyword"]: "Keyword",
    colmap["Total"]: "Total_Count",
    colmap["Weighted"]: "Weighted_Count"
}, inplace=True)

# ==========================================================
# ✅ GROUP DUPLICATES BY CATEGORY + KEYWORD
# ==========================================================
grouped_df = (
    df.groupby(["Category", "Keyword"], as_index=False)[["Total_Count", "Weighted_Count"]]
    .sum()
)

print("📊 集計後（重複キーワード統合）:\n", grouped_df.head())

# ==========================================================
# ✅ VISUALIZATION PER CATEGORY
# ==========================================================
categories = grouped_df["Category"].unique()

for category in categories:
    cat_df = grouped_df[grouped_df["Category"] == category]
    if cat_df.empty:
        print(f"⚠️ {category} にデータがありません。スキップします。")
        continue

    # ---- 上位7件 ----
    cat_df = cat_df.sort_values("Total_Count", ascending=False).head(7)
    keywords = cat_df["Keyword"].tolist()
    total_counts = cat_df["Total_Count"].tolist()
    weighted_counts = cat_df["Weighted_Count"].tolist()

    x = np.arange(len(keywords))
    width = 0.35

    # ---- 色設定 ----
    colors_total = [
        "#3A78C2" if w > 0 else "gray" for w in weighted_counts
    ]  # gray if no weighted count
    colors_weight = [
        "#C23A3A" if w > 0 else "none" for w in weighted_counts
    ]

    # ---- 図設定 ----
    plt.figure(figsize=(10, 6), dpi=150)
    plt.bar(
        x - width/2, total_counts, width,
        label="実際の出現数", color=colors_total, edgecolor="black"
    )
    plt.bar(
        x + width/2, weighted_counts, width,
        label="重み付き出現数", color="#C23A3A", edgecolor="black"
    )

    plt.title(f"{category}カテゴリの上位キーワード）", pad=40, fontweight="bold")
    plt.ylabel("出現数", fontweight="bold")
    plt.xticks(x, keywords, rotation=25, ha="right")

    # ---- 凡例設定 ----
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(OUTPUT_FOLDER, f"{category}_keywords_count.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")

print("🎉 すべてのカテゴリのグラフ（最大7件、重み0は灰色）を output フォルダに保存しました。")
