import os
import re
import pandas as pd

# =====================================
# CONFIG
# =====================================
RAW_FOLDER = "raw"
DATA_FOLDER = "data"

# Ensure data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

# Regex pattern for question sentences
QUESTION_PATTERN = r"([^？?]*[？?])"

# =====================================
# FUNCTION: Extract questions (Interviewer == "Yes")
# =====================================
def extract_questions_from_file(file_path):
    df = pd.read_excel(file_path)
    
    if not {"Interviewer", "Content"}.issubset(df.columns):
        print(f"⚠️ Skipping {file_path} (missing required columns)")
        return []
    
    # Filter interviewer questions only
    df = df[df["Interviewer"].astype(str).str.lower().eq("yes")]
    
    questions = []
    for content in df["Content"].dropna():
        found = re.findall(QUESTION_PATTERN, str(content))
        questions.extend([q.strip() for q in found if len(q.strip()) > 1])
    
    return questions

# =====================================
# MAIN PROCESS
# =====================================
for file in os.listdir(RAW_FOLDER):
    if file.lower().endswith(".xlsx"):
        file_path = os.path.join(RAW_FOLDER, file)
        base_name = os.path.splitext(file)[0].lower()  # lowercase base name
        
        print(f"📄 Processing: {file_path}")
        questions = extract_questions_from_file(file_path)
        
        if not questions:
            print(f"⚠️ No questions found in {file}")
            continue
        
        # Count duplicates
        question_series = pd.Series([q.replace("\n", " ").strip() for q in questions if q.strip()])
        question_counts = question_series.value_counts().reset_index()
        question_counts.columns = ["Question_list", "Total"]
        question_counts = question_counts.sort_values("Question_list").reset_index(drop=True)
        
        # Assign IDs
        question_counts.insert(0, "Question_id", [f"Q{i+1:03d}" for i in range(len(question_counts))])
        
        # Save as CSV
        output_file = os.path.join(DATA_FOLDER, f"question_list_{base_name}.csv")
        question_counts.to_csv(output_file, index=False, encoding="utf-8-sig")
        
        print(f"✅ Saved: {output_file} ({len(question_counts)} unique questions)")
        print(question_counts.head())
