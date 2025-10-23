import os
import pandas as pd

# =====================================
# CONFIG
# =====================================
RAW_FOLDER = "raw"
DATA_FOLDER = "data"

os.makedirs(DATA_FOLDER, exist_ok=True)

# Helper: robust yes/no → True/False
def is_yes(value):
    s = str(value).strip().lower()
    return s in {"yes", "y", "true", "1"}

for file in os.listdir(RAW_FOLDER):
    if not file.lower().endswith(".xlsx"):
        continue

    file_path = os.path.join(RAW_FOLDER, file)
    base_name = os.path.splitext(file)[0].lower()  # for output filename
    print(f"📄 Processing: {file_path}")

    df = pd.read_excel(file_path)

    # Required columns
    required = {"Interviewer", "Participant", "Male", "Marital status", "Having kids"}
    missing = required - set(df.columns)
    if missing:
        print(f"⚠️ Skipping {file} — missing columns: {missing}")
        continue

    # Filter Interviewer == "No"
    df = df[df["Interviewer"].astype(str).str.strip().str.lower().eq("no")].copy()
    if df.empty:
        print(f"⚠️ No participant rows found in {file}")
        continue

    # Normalize participant key for grouping; keep original text for Name
    # We preserve the original Participant value as Name, but use a normalized string as the key
    df["__participant_key__"] = df["Participant"].astype(str).str.strip()

    # Deduplicate to one row per participant (first appearance order)
    # Keep first non-null values for Male / Marital status / Having kids
    # Use drop_duplicates to preserve first-seen ordering, then merge aggregates if needed
    first_order = df.drop_duplicates(subset="__participant_key__", keep="first").copy()

    # Aggregate preferred values in case there are multiple rows per participant
    agg = (
        df.groupby("__participant_key__", sort=False)
          .agg({
              "Participant": "first",
              "Male": "first",
              "Marital status": "first",
              "Having kids": "first"
          })
          .reset_index()
    )

    # Reorder by first appearance
    key_order = first_order["__participant_key__"].tolist()
    agg = agg.set_index("__participant_key__").loc[key_order].reset_index()

    # Map gender: Male = Yes → Male, else Female
    agg["Gender"] = agg["Male"].apply(lambda x: "Male" if is_yes(x) else "Female")

    # Assign stable Participant IDs by first appearance
    agg.insert(0, "Participant_id", [f"P{i+1}" for i in range(len(agg))])

    # Prepare output columns
    out = agg.rename(columns={"Participant": "Name"})[
        ["Participant_id", "Name", "Gender", "Marital status", "Having kids"]
    ]

    # Save
    output_file = os.path.join(DATA_FOLDER, f"participant_{base_name}.csv")
    out.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"✅ Saved: {output_file} ({len(out)} unique participants)")
    print(out.head())
