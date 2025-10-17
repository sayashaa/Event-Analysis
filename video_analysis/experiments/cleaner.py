#cleaner.py
import pandas as pd
import numpy as np
import math
import os


# Load the movement log
# Ensure your CSV has columns: frame, id, x, y
input_csv = "../../apps/data/movement_log_GX010011_final.csv"
output_csv = "../../apps/data/movement_log_GX010011.csv"

# Parameters
merge_distance_threshold = 20      # pixels: merge based on overall-centroid proximity
outlier_distance_threshold = 50   # pixels: max allowed frame-to-frame jump

# Read data
orig_df = pd.read_csv(input_csv)
df = orig_df.copy()

# Phase 1: Reassign all unknowns → nearest known centroid
df['old_id'] = df['id']

# Treat any id starting with 'unknown' as unknown
def is_unknown(id_val):
    return str(id_val).startswith('unknown')

def compute_centroids(data):
    # Only known IDs contribute to centroids
    known = data[~data['old_id'].apply(is_unknown)]
    return known.groupby('old_id')[['x','y']].mean().to_dict('index')

centroids = compute_centroids(df)

total_unknown = df['old_id'].apply(is_unknown).sum()

def assign_nearest(row, cents):
    if is_unknown(row['old_id']) and cents:
        best, min_d = None, float('inf')
        for k, coord in cents.items():
            d = math.hypot(row['x'] - coord['x'], row['y'] - coord['y'])
            if d < min_d:
                best, min_d = k, d
        return best if best is not None else row['old_id']
    return row['old_id']

df['id_assigned'] = df.apply(lambda r: assign_nearest(r, centroids), axis=1)

# Remove any rows still unassigned (optional, unlikely here)
df = df[~df['id_assigned'].apply(is_unknown)].copy()

assigned_unknown = total_unknown - df['old_id'].apply(is_unknown).sum()
print(f"Phase 1: {total_unknown} unknown → {assigned_unknown} reassigned, {total_unknown - assigned_unknown} removed.")

# Phase 2: Continuity correction
all_frames = sorted(orig_df['frame'].unique())
x_min, x_max = df['x'].min(), df['x'].max()
y_min, y_max = df['y'].min(), df['y'].max()
continuity_rows = []
for pid, group in df.groupby('id_assigned'):
    positions = dict(zip(group['frame'], zip(group['x'], group['y'])))
    for f in all_frames:
        if f not in positions:
            prev_frames = [pf for pf in positions if pf < f]
            next_frames = [nf for nf in positions if nf > f]
            if prev_frames and next_frames:
                pf = max(prev_frames)
                nf = min(next_frames)
                px, py = positions[pf]
                nx, ny = positions[nf]
                if x_min <= px <= x_max and y_min <= py <= y_max and x_min <= nx <= x_max and y_min <= ny <= y_max:
                    continuity_rows.append({
                        'frame': f,
                        'old_id': pid,
                        'id_assigned': pid,
                        'x': px,
                        'y': py
                    })
inserted = len(continuity_rows)
if continuity_rows:
    df = pd.concat([df, pd.DataFrame(continuity_rows)], ignore_index=True)
print(f"Phase 2: Inserted {inserted} continuity rows.")

# Phase 3: Merge close ID clusters based on overall centroids
centroids2 = df.groupby('id_assigned')[['x','y']].mean().to_dict('index')
ids = list(centroids2.keys())
merge_map = {}
for i, id1 in enumerate(ids):
    for id2 in ids[i+1:]:
        d = math.hypot(centroids2[id1]['x'] - centroids2[id2]['x'], 
                       centroids2[id1]['y'] - centroids2[id2]['y'])
        if d < merge_distance_threshold:
            merge_map[id2] = id1

df['id_merged'] = df['id_assigned'].apply(lambda x: merge_map.get(x, x))
if merge_map:
    print("Phase 3: Merged the following IDs:")
    for src, dst in merge_map.items():
        print(f"  - ID {src} → ID {dst}")
else:
    print("Phase 3: No IDs merged.")

# Phase 4: Remove outliers by frame-to-frame jump
cleaned = []
removed_outliers = 0
for pid, group in df.groupby('id_merged'):
    grp = group.sort_values('frame')
    keep = [True] * len(grp)
    coords = list(zip(grp['x'], grp['y']))
    for i in range(1, len(coords)):
        if math.hypot(coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1]) > outlier_distance_threshold:
            keep[i] = False
            removed_outliers += 1
    cleaned.append(grp[keep])
clean_df = pd.concat(cleaned)
print(f"Phase 4: Removed {removed_outliers} outlier jumps.")

# Final output to (frame, old_id, id, x, y)
def try_int(v):
    try: return int(v)
    except: return v

final_df = clean_df.copy()
final_df['old_id'] = final_df['old_id'].apply(try_int)
final_df['id']     = final_df['id_merged'].apply(try_int)

final_df = ( final_df
    .loc[:, ['frame','old_id','id','x','y']]
    .sort_values(['frame','id'])
    .reset_index(drop=True)
)

final_df.to_csv(output_csv, index=False)
print(f"Final cleaned log saved to '{output_csv}'")
