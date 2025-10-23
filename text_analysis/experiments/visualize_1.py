import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import the shared font setup
from japanese_font_setup import ensure_japanese_font

# ======================================
# ✅ FONT SETUP
# ======================================
ensure_japanese_font()

# ======================================
# ✅ CONFIG
# ======================================
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

# ======================================
# ✅ COLUMN NORMALIZATION
# ======================================
colmap = {
    "Category": next((c for c in df.columns if "category" in c.lower()), None),
    "Total": next((c for c in df.columns if "total" in c.lower() and "weight" not in c.lower()), None),
    "Weighted": next((c for c in df.columns if "weight" in c.lower()), None),
}

for key, col in colmap.items():
    if col is None:
        raise KeyError(f"❌ 必要な列 '{key}' がCSVに見つかりません。")

df.rename(columns={
    colmap["Category"]: "Category",
    colmap["Total"]: "Total_Count",
    colmap["Weighted"]: "Weighted_Count"
}, inplace=True)

# ======================================
# ✅ AGGREGATE BY CATEGORY
# ======================================
agg_df = df.groupby("Category")[["Total_Count", "Weighted_Count"]].sum().reset_index()
print("📊 集計結果:\n", agg_df)

# ======================================
# ✅ VISUALIZATION
# ======================================
plt.figure(figsize=(10, 6), dpi=150)

x = np.arange(len(agg_df))
width = 0.35

plt.bar(
    x - width/2, agg_df["Total_Count"], width,
    label="実際の出現数", color="#3A78C2", edgecolor="black"
)
plt.bar(
    x + width/2, agg_df["Weighted_Count"], width,
    label="重み付き出現数", color="#C23A3A", edgecolor="black"
)

plt.title("カテゴリ別 出現数と重み付き出現数", pad=40, fontweight="bold")
plt.ylabel("出現数", fontweight="bold")
plt.xticks(x, agg_df["Category"], rotation=20, ha="right")

# Legend below title, not overlapping
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.10),
    ncol=2,
    frameon=False,
    fontsize=11
)

# Layout & save
plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = os.path.join(OUTPUT_FOLDER, "category_total_vs_weighted.png")
plt.savefig(out_path, bbox_inches="tight")
plt.close()

print(f"✅ Saved visualization: {out_path}")
