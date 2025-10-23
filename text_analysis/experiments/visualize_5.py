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
    "Participant": next((c for c in df.columns if "participant" in c.lower()), None),
    "Total": next((c for c in df.columns if "total" in c.lower() and "weight" not in c.lower()), None),
    "Weighted": next((c for c in df.columns if "weight" in c.lower()), None),
}
for key, col in colmap.items():
    if col is None:
        raise KeyError(f"❌ 必要な列 '{key}' がCSVに見つかりません。")

df.rename(columns={
    colmap["Category"]: "Category",
    colmap["Participant"]: "Participant",
    colmap["Total"]: "Total_Count",
    colmap["Weighted"]: "Weighted_Count"
}, inplace=True)

# ==========================================================
# ✅ AGGREGATE TOTAL COUNT & WEIGHTED COUNT BY CATEGORY & PARTICIPANT
# ==========================================================
cat_participant_df = (
    df.groupby(["Category", "Participant"], as_index=False)[["Total_Count", "Weighted_Count"]]
    .sum()
    .sort_values(["Category", "Total_Count"], ascending=[True, False])
)

# ==========================================================
# ✅ VISUALIZATION PER CATEGORY
# ==========================================================
categories = cat_participant_df["Category"].unique()

for category in categories:
    cat_df = cat_participant_df[cat_participant_df["Category"] == category]
    if cat_df.empty:
        continue

    # Sort by total descending
    cat_df = cat_df.sort_values("Total_Count", ascending=False)

    participants = cat_df["Participant"].tolist()
    total_counts = cat_df["Total_Count"].tolist()
    weighted_counts = cat_df["Weighted_Count"].tolist()

    x = np.arange(len(participants))
    width = 0.35

    # ---- Plot ----
    plt.figure(figsize=(10, 6), dpi=150)
    plt.bar(
        x - width/2, total_counts, width,
        label="総出現数", color="#3A78C2", edgecolor="black"
    )
    plt.bar(
        x + width/2, weighted_counts, width,
        label="重み付き出現数", color="#C23A3A", edgecolor="black"
    )

    # Title and labels
    plt.title(f"{category}カテゴリ｜参加者別 出現数と重み付き出現数", pad=30, fontweight="bold")
    plt.ylabel("出現数", fontweight="bold")
    plt.xticks(x, participants, rotation=25, ha="right")

    # Legend above
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        frameon=False,
        fontsize=11
    )

    # Value labels (optional)
    for i, (t, w) in enumerate(zip(total_counts, weighted_counts)):
        plt.text(x[i] - width/2, t + 0.3, str(int(t)), ha='center', va='bottom', fontsize=9, color="#3A78C2")
        plt.text(x[i] + width/2, w + 0.3, str(int(w)), ha='center', va='bottom', fontsize=9, color="#C23A3A")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(OUTPUT_FOLDER, f"{category}_participant_total_weight.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")

print("🎉 各カテゴリごとの参加者別 総出現数＋重み付き出現数グラフを output フォルダに保存しました。")
