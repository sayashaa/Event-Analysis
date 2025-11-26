import os
import pandas as pd

# ============================================================
# 1. FILE PATHS
# ============================================================
DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

responses_file = os.path.join(DATA_DIR, "ev1_phase2_local_final.csv")
questions_file = os.path.join(DATA_DIR, "question_list_ev1_phase2_local.csv")

# Load CSV files
responses_df = pd.read_csv(responses_file)
questions_df = pd.read_csv(questions_file)

# Keep only participant responses (Speaker starts with P)
responses_df = responses_df[responses_df["Speaker"].str.startswith("P")]

# Merge questions into responses
merged = responses_df.merge(
    questions_df[["Question_id", "Question_list"]],
    on="Question_id",
    how="left"
)

# ============================================================
# 2. BUILD ONE LARGE CSV
# ============================================================
rows = []

for qid, subset in merged.groupby("Question_id"):

    question_text = subset["Question_list"].iloc[0]
    responses = subset["Content"].tolist()

    # Add the question row
    rows.append({
        "Type": "QUESTION",
        "Question_id": qid,
        "Text": question_text
    })

    # Add each response
    for i, resp in enumerate(responses):
        rows.append({
            "Type": f"RESPONSE_{i+1}",
            "Question_id": qid,
            "Text": resp
        })

# Convert to DataFrame
out_df = pd.DataFrame(rows)

# Save as one CSV
output_path = os.path.join(OUTPUT_DIR, "all_questions_with_8responses.csv")
out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("DONE: Saved to output/all_questions_with_8responses.csv")
