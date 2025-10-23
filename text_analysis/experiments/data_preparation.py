import os
import re
import pandas as pd

# =====================================
# CONFIG
# =====================================
RAW_FOLDER = "raw"
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# =====================================
# LOAD QUESTION + PARTICIPANT MAPS
# =====================================
def load_mapping(base_name):
    q_file = os.path.join(DATA_FOLDER, f"question_list_{base_name}.csv")
    p_file = os.path.join(DATA_FOLDER, f"participant_{base_name}.csv")

    if not os.path.exists(q_file):
        raise FileNotFoundError(f"❌ Missing question file: {q_file}")
    if not os.path.exists(p_file):
        raise FileNotFoundError(f"❌ Missing participant file: {p_file}")

    q_df = pd.read_csv(q_file)
    p_df = pd.read_csv(p_file)

    # Map exact question text -> Question_id
    question_map = {str(q).strip(): str(qid).strip()
                    for qid, q in zip(q_df["Question_id"], q_df["Question_list"])}

    # Map participant Name -> Participant_id (e.g., "1"→"P1", "Tanaka"→"P3")
    participant_map = {str(name).strip(): str(pid).strip()
                       for pid, name in zip(p_df["Participant_id"], p_df["Name"].astype(str))}
    return question_map, participant_map

# =====================================
# SPEAKER ASSIGNMENT
# =====================================
def assign_speakers(df, participant_map):
    """
    Returns a Series of speakers: 'I' for interviewer, 'P#' for participant (using participant_map),
    'Unknown' if no match.
    """
    speakers = []
    for _, row in df.iterrows():
        if str(row["Interviewer"]).strip().lower() == "yes":
            speakers.append("I")
        else:
            name = str(row["Participant"]).strip()
            speakers.append(participant_map.get(name, "Unknown"))
    return pd.Series(speakers, index=df.index)

# =====================================
# SESSION ASSIGNMENT (SEGMENT-BASED)
# =====================================
def assign_sessions_segment_based(speakers):
    """
    Rule:
      - Sessions are contiguous segments per participant.
      - A segment *starts* when we first see its participant (P#), and all immediately preceding
        consecutive 'I' lines (since the previous participant) belong to that same session.
      - When the participant changes (new P# encountered), start a new session number.
      - Interviewer 'I' lines after a participant continue the current session
        until a different participant appears.

    Example sequence: I, P1, I, P1, I, P2, P2  ->  1,1,1,1,2,2,2
    """
    n = len(speakers)
    sessions = [None] * n
    current_session = 0
    current_pid = None

    i = 0
    while i < n:
        spk = speakers[i]

        if isinstance(spk, str) and spk.startswith("P") and spk != "Unknown":
            # New segment if participant changes
            if spk != current_pid:
                current_session += 1
                current_pid = spk

                # Back-fill preceding consecutive 'I' lines (always overwrite to new session)
                j = i - 1
                while j >= 0 and speakers[j] == "I":
                    sessions[j] = current_session  # ← changed: always overwrite
                    j -= 1

            # Assign session for this participant line
            sessions[i] = current_session

            # Forward-assign consecutive 'I' lines until next participant change
            k = i + 1
            while k < n and speakers[k] == "I":
                sessions[k] = current_session
                k += 1

            i += 1
        else:
            # 'I' or 'Unknown' before we have seen any participant yet -> leave as None for now;
            # it will be back-filled when the first P# appears.
            i += 1

    # If there are leading or trailing 'I' lines that never got a session
    # (e.g., no participant ever appeared), set them to nearest known session (or 1).
    if any(s is not None for s in sessions):
        first_assigned = next((s for s in sessions if s is not None), 1)
        for idx in range(n):
            if sessions[idx] is None:
                prev = next((sessions[j] for j in range(idx - 1, -1, -1) if sessions[j] is not None), None)
                nxt  = next((sessions[j] for j in range(idx + 1,  n) if sessions[j] is not None), None)
                sessions[idx] = prev if prev is not None else (nxt if nxt is not None else first_assigned)
    else:
        sessions = [1] * n

    return pd.Series(sessions, index=speakers.index)

# =====================================
# QUESTION ID PROPAGATION
# =====================================
def assign_question_ids(df, question_map):
    """
    For each row, if its Content contains any question text (exact substring match),
    set current_qid to that Question_id; otherwise carry forward the last seen Question_id.
    """
    qids = []
    current_qid = None
    for content in df["Content"].astype(str):
        matched = None
        for q_text, qid in question_map.items():
            if q_text and q_text in content:
                matched = qid
                break
        if matched:
            current_qid = matched
        qids.append(current_qid)
    return pd.Series(qids, index=df.index)

# =====================================
# MAIN
# =====================================
for file in os.listdir(RAW_FOLDER):
    if not file.lower().endswith(".xlsx"):
        continue

    file_path = os.path.join(RAW_FOLDER, file)
    base_name = os.path.splitext(file)[0].lower()
    print(f"📄 Processing: {file_path}")

    df = pd.read_excel(file_path)

    # Required columns
    need = {"Content", "Interviewer", "Participant"}
    if not need.issubset(df.columns):
        print(f"⚠️ Skipping {file} (missing required columns: {need - set(df.columns)})")
        continue

    try:
        question_map, participant_map = load_mapping(base_name)
    except FileNotFoundError as e:
        print(e)
        continue

    # Speaker
    speakers = assign_speakers(df, participant_map)

    # Session (numeric) with the corrected segment rule
    sessions = assign_sessions_segment_based(speakers)

    # Question Ids (carry forward)
    question_ids = assign_question_ids(df, question_map)

    # Final output
    final = pd.DataFrame({
        "Speaker": speakers,
        "Question_id": question_ids,
        "Session": sessions,
        "Content": df["Content"].astype(str)
    })

    out_file = os.path.join(DATA_FOLDER, f"{base_name}_final.csv")
    final.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {out_file} ({len(final)} rows)")
    print(final.head(10))
