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

# Find keyword summary automatically
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
# ✅ CREATE CATEGORY SUBFOLDERS
# ==========================================================
categories = df["Category"].unique()
for cat in categories:
    os.makedirs(os.path.join(OUTPUT_FOLDER, cat), exist_ok=True)

# ==========================================================
# ✅ TOP 7 KEYWORDS PER CATEGORY
# ==========================================================
top_keywords_by_cat = (
    df.groupby(["Category", "Keyword"], as_index=False)["Total_Count"].sum()
    .sort_values(["Category", "Total_Count"], ascending=[True, False])
    .groupby("Category")
    .head(7)
)

print("📊 上位キーワード抽出完了:\n", top_keywords_by_cat.head())

# ==========================================================
# ✅ VISUALIZE PER CATEGORY → PER KEYWORD (participant breakdown)
# ==========================================================
for category in categories:
    cat_keywords = top_keywords_by_cat[top_keywords_by_cat["Category"] == category]["Keyword"].tolist()
    if not cat_keywords:
        continue

    for keyword in cat_keywords:
        sub_df = df[(df["Category"] == category) & (df["Keyword"] == keyword)]
        if sub_df.empty:
            continue

        # Sort participants by total count
        sub_df = sub_df.sort_values("Total_Count", ascending=False)

        participants = sub_df["Participant"].tolist()
        total_counts = sub_df["Total_Count"].tolist()
        weighted_counts = sub_df["Weighted_Count"].tolist()

        x = np.arange(len(participants))
        width = 0.35

        # Bar colors (gray if weight=0)
        colors_total = ["#3A78C2" if w > 0 else "gray" for w in weighted_counts]

        plt.figure(figsize=(8, 5), dpi=150)
        plt.bar(
            x - width/2, total_counts, width,
            label="実際の出現数", color=colors_total, edgecolor="black"
        )
        plt.bar(
            x + width/2, weighted_counts, width,
            label="重み付き出現数", color="#C23A3A", edgecolor="black"
        )

        plt.title(f"{category}｜『{keyword}』キーワード別参加者出現数", pad=35, fontweight="bold")
        plt.ylabel("出現数", fontweight="bold")
        plt.xticks(x, participants)

        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.10),
            ncol=2,
            frameon=False,
            fontsize=10
        )

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        save_path = os.path.join(OUTPUT_FOLDER, category, f"{keyword}_participant_breakdown.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {save_path}")

print("🎉 全カテゴリ・キーワードごとの参加者別グラフを output フォルダに保存しました。")
