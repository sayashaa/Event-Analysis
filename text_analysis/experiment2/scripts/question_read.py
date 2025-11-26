import pandas as pd

# =========================================
# 1. CONFIG
# =========================================
RESP_CSV = "output/all_questions_with_8responses.csv"
QLIST_CSV = "data/question_list_ev1_phase2_local.csv"

# =========================================
# 2. LOAD DATA
# =========================================
df_resp = pd.read_csv(RESP_CSV)
df_q = pd.read_csv(QLIST_CSV)

# df_resp columns: Type, Question_id, Text
# df_q columns: Question_id, Question_list, Total Response, Category_ID, Category

# =========================================
# 3. FILTER OV0N CATEGORIES (Category_ID starting with 'OV0')
# =========================================
df_ov = df_q[df_q["Category_ID"].str.startswith("OV0")].copy()

# Merge to get Category info into the response table
df_join = df_resp.merge(
    df_ov[["Question_id", "Category_ID", "Category", "Question_list"]],
    on="Question_id",
    how="inner"
)

# =========================================
# 4. PRINT STRUCTURED OUTPUT
#    Group by Category_ID, then by Question_id
# =========================================

df_join = df_join.sort_values(["Category_ID", "Question_id", "Type"])

categories = df_join["Category_ID"].unique()

for cat_id in categories:

    # Get category name
    cat_name = df_join[df_join["Category_ID"] == cat_id]["Category"].iloc[0]

    print("=" * 70)
    print(f"Category ID:   {cat_id}")
    print(f"Category Name: {cat_name}")
    print("=" * 70)

    qids = df_join[df_join["Category_ID"] == cat_id]["Question_id"].unique()

    for qid in qids:

        # Fetch question text
        q_text = df_join[df_join["Question_id"] == qid]["Question_list"].iloc[0]

        print(f"\n  Question ID:   {qid}")
        print(f"  Question_list: {q_text}")

        # Get all responses for that question
        sub = df_join[
            (df_join["Category_ID"] == cat_id) &
            (df_join["Question_id"] == qid) &
            (df_join["Type"].str.startswith("RESPONSE"))
        ].sort_values("Type")

        for _, row in sub.iterrows():
            print(f"    {row['Type']}: {row['Text']}")

    print("\n")  # space between categories
