import os
import time
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ==================================================
# CONFIG
# ==================================================
DATA_FOLDER = "data"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==================================================
# SENTIMENT MODEL SETUP (with timing and progress)
# ==================================================
print("🧠 Step 1: Initializing Japanese sentiment model...")
MODEL_NAME = "ku-nlp/deberta-v2-base-japanese"

start_time = time.time()
print(f"⏳ Loading tokenizer for {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"✅ Tokenizer loaded. ({time.time() - start_time:.1f}s)")

print("⏳ Loading model weights (this can take a few minutes the first time)...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
labels = ["negative", "neutral", "positive"]
print(f"✅ Model fully loaded in {time.time() - start_time:.1f}s total.\n")

# ==================================================
# SENTIMENT DICTIONARIES
# ==================================================
positive_words = ["良い", "いい", "楽しい", "見やすい", "好き", "素敵", "印象的", "便利", "カラフル", "嬉しい"]
negative_words = ["ない", "悪い", "難しい", "微妙", "興味ない", "いまいち", "ちょっと", "嫌い", "しづらい"]
sentiment_weights = {"positive": 1.0, "neutral": 0.5, "negative": 0}

# ==================================================
# RULE-BASED DETECTION
# ==================================================
def rule_based_sentiment(text):
    text = str(text)
    pos = sum(w in text for w in positive_words)
    neg = sum(w in text for w in negative_words)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    else:
        return "neutral"

# ==================================================
# MODEL-BASED DETECTION (with live print)
# ==================================================
def model_based_sentiment(text):
    if not text or not isinstance(text, str) or text.strip() == "":
        return "neutral"
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        result = labels[torch.argmax(probs).item()]
        return result
    except Exception as e:
        print(f"⚠️ Model error on text: {text[:30]}... ({e})")
        return "neutral"

# ==================================================
# HYBRID SENTIMENT DETECTION (with progress)
# ==================================================
def detect_sentiment(text):
    """
    1️⃣ Use rule-based for short/simple sentences.
    2️⃣ If result is 'neutral' or text > 10 chars → use model-based refinement.
    """
    rule_result = rule_based_sentiment(text)
    if rule_result == "neutral" or len(str(text)) > 10:
        model_result = model_based_sentiment(text)
        if model_result != rule_result:
            print(f"🔁 Refined by model: {rule_result} → {model_result}")
        return model_result
    return rule_result

# ==================================================
# MAIN ANALYSIS FUNCTION
# ==================================================
def analyze_pair(base_name, final_path, keyword_path):
    print(f"\n📄 Step 2: Processing file → {base_name}")
    print("   Reading data and keywords...")

    df = pd.read_csv(final_path)
    kw = pd.read_csv(keyword_path)

    if not {"Speaker", "Question_id", "Session", "Content"}.issubset(df.columns):
        print(f"⚠️ Skipping {base_name}: missing required columns.")
        return

    # Expand keyword hierarchy
    print("   Expanding keyword hierarchy...")
    expanded = []
    for _, row in kw.iterrows():
        children = str(row["Child"]).split(",") if pd.notna(row["Child"]) else []
        for c in children:
            c = c.strip()
            if c and c != "なし":
                expanded.append({"Category": row["Category"], "Parent": row["Parent"], "Child": c})
        if str(row["Child"]).strip() == "なし" or not row["Child"]:
            expanded.append({"Category": row["Category"], "Parent": row["Parent"], "Child": row["Parent"]})
    expanded = pd.DataFrame(expanded)

    df_i = df[df["Speaker"] == "I"].copy()
    df_p = df[df["Speaker"].str.startswith("P")].copy()

    # ==================================================
    # SENTIMENT per (Session, Question_id)
    # ==================================================
    print("   🔍 Analyzing sentiment for each Session × Question...")
    session_question_sentiments = {}
    total_groups = len(df_p.groupby(["Session", "Question_id"]))
    for idx, ((sid, qid), group) in enumerate(df_p.groupby(["Session", "Question_id"]), 1):
        print(f"      [{idx}/{total_groups}] Analyzing Session={sid}, Question={qid}")
        p_text = " ".join(group["Content"].astype(str))
        session_question_sentiments[(sid, qid)] = detect_sentiment(p_text)

    # ==================================================
    # COUNT KEYWORDS
    # ==================================================
    print("\n   🧮 Counting keyword appearances...")
    records = []
    total_kw = len(expanded)
    start_time_kw = time.time()

    for idx, kw_row in enumerate(expanded.itertuples(index=False), 1):
        cat, parent, kw = kw_row.Category, kw_row.Parent, kw_row.Child
        if idx % 10 == 0 or idx == total_kw:
            print(f"      → {idx}/{total_kw} keywords processed")

        for (sid, qid) in df[["Session", "Question_id"]].drop_duplicates().itertuples(index=False):
            subset_i = df_i[(df_i["Session"] == sid) & (df_i["Question_id"] == qid)]
            subset_p = df_p[(df_p["Session"] == sid) & (df_p["Question_id"] == qid)]

            participant = next((s for s in subset_p["Speaker"].unique() if str(s).startswith("P")), None)
            if not participant:
                continue

            count_i = subset_i["Content"].astype(str).apply(lambda x: x.count(kw)).sum()
            count_p = subset_p["Content"].astype(str).apply(lambda x: x.count(kw)).sum()

            sentiment_context = session_question_sentiments.get((sid, qid), "neutral")
            sentiment_participant = detect_sentiment(" ".join(subset_p["Content"].astype(str)))

            if count_i > 0:
                records.append({
                    "Category": cat, "Parent": parent, "Child": kw,
                    "Participant": participant, "Session": sid, "Question_id": qid,
                    "Source": "Interviewer", "Sentiment": sentiment_context, "Count": count_i
                })
            if count_p > 0:
                records.append({
                    "Category": cat, "Parent": parent, "Child": kw,
                    "Participant": participant, "Session": sid, "Question_id": qid,
                    "Source": "Participant", "Sentiment": sentiment_participant, "Count": count_p
                })

    print(f"   ✅ Keyword counting complete in {time.time() - start_time_kw:.1f}s.")

    if not records:
        print(f"⚠️ No keyword matches found in {base_name}")
        return

    # ==================================================
    # SAVE RESULTS
    # ==================================================
    print("   💾 Saving keyword-level and summary files...")
    counts = pd.DataFrame(records)
    counts["Weighted_Count"] = counts["Sentiment"].map(sentiment_weights).fillna(0) * counts["Count"]

    count_out = os.path.join(OUTPUT_FOLDER, f"keyword_count_{base_name}.csv")
    counts.to_csv(count_out, index=False, encoding="utf-8-sig")

    summary = (
        counts.groupby(["Category", "Parent", "Child", "Participant", "Sentiment"], as_index=False)["Count"].sum()
        .pivot_table(index=["Category", "Parent", "Child", "Participant"],
                     columns="Sentiment", values="Count", fill_value=0)
        .reset_index()
    )

    for col in ["positive", "negative", "neutral"]:
        if col not in summary.columns:
            summary[col] = 0

    summary["Total"] = summary["positive"] + summary["negative"] + summary["neutral"]
    summary["Net_Sentiment"] = summary["positive"] - summary["negative"]

    weighted_total = (
        counts.groupby(["Category", "Parent", "Child", "Participant"], as_index=False)["Weighted_Count"].sum()
    )
    summary = summary.merge(weighted_total, on=["Category", "Parent", "Child", "Participant"], how="left")
    summary.rename(columns={"Weighted_Count": "Weighted_Total"}, inplace=True)

    summary_out = os.path.join(OUTPUT_FOLDER, f"keyword_summary_{base_name}.csv")
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print(f"✅ Completed {base_name}")
    print(f"   ↳ Saved: {os.path.basename(count_out)}")
    print(f"   ↳ Saved: {os.path.basename(summary_out)}")

# ==================================================
# SCAN ALL PAIRS IN DATA FOLDER
# ==================================================
print("\n🔎 Step 3: Scanning data folder...")
final_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith("_final.csv")]

if not final_files:
    print("⚠️ No *_final.csv files found in 'data/' folder.")
else:
    print(f"📦 Found {len(final_files)} dataset(s): {final_files}")
    for file in final_files:
        base_name = file.replace("_final.csv", "")
        final_path = os.path.join(DATA_FOLDER, file)
        keyword_path = os.path.join(DATA_FOLDER, f"keyword_hierarchy_{base_name}.csv")

        if os.path.exists(keyword_path):
            analyze_pair(base_name, final_path, keyword_path)
        else:
            print(f"⚠️ Skipping {base_name}: keyword file not found ({keyword_path})")

print("\n🏁 All processing complete.")
