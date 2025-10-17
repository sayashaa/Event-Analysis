import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Parameters
video_path = '../../apps/videos/GX010011.MP4'
csv_path = '../../apps/data/movement_log_GX010011.csv'

# Output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
output_frame_image = os.path.join(output_dir, 'heatmap_overlay_frame.png')
output_white_image = os.path.join(output_dir, 'heatmap_overlay_white.png')
output_bar_chart = os.path.join(output_dir, 'stall_visits_bar_chart.png')

# Grid parameters
grid_rows = 20
grid_cols = 20
alpha = 0.6
beta = 0.4

def create_heatmap_overlay():
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise IOError(f"Cannot read frame 1 from {video_path}")
    height, width = frame.shape[:2]

    # Square grid
    cell_size = min(height // grid_rows, width // grid_cols)
    grid_rows_sq = (height + cell_size - 1) // cell_size
    grid_cols_sq = (width + cell_size - 1) // cell_size

    df = pd.read_csv(csv_path)
    counts = np.zeros((grid_rows_sq, grid_cols_sq), dtype=int)

    for _, row in df.iterrows():
        x, y = int(row['x']), int(row['y'])
        r = min(y // cell_size, grid_rows_sq - 1)
        c = min(x // cell_size, grid_cols_sq - 1)
        counts[r, c] += 1

    max_count = counts.max()
    norm_counts = (counts / max_count * 255).astype(np.uint8) if max_count > 0 else counts.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(norm_counts, cv2.COLORMAP_JET)
    heatmap_full = np.zeros((grid_rows_sq*cell_size, grid_cols_sq*cell_size, 3), dtype=np.uint8)
    for r in range(grid_rows_sq):
        for c in range(grid_cols_sq):
            y1, y2 = r*cell_size, (r+1)*cell_size
            x1, x2 = c*cell_size, (c+1)*cell_size
            heatmap_full[y1:y2, x1:x2] = heatmap_color[r, c]

    heatmap_full = heatmap_full[:height, :width]
    overlay_frame = cv2.addWeighted(frame, alpha, heatmap_full, beta, 0)
    overlay_white = cv2.addWeighted(np.full_like(frame, 255), alpha, heatmap_full, beta, 0)

    # Draw grid and numbers
    for overlay in (overlay_frame, overlay_white):
        for i in range(grid_rows_sq + 1):
            cv2.line(overlay, (0, i*cell_size), (width, i*cell_size), (255,255,255), 1)
        for j in range(grid_cols_sq + 1):
            cv2.line(overlay, (j*cell_size, 0), (j*cell_size, height), (255,255,255), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(cell_size / 80.0, 0.25)
        thickness = 1

        for r in range(grid_rows_sq):
            for c in range(grid_cols_sq):
                zone_id = r * grid_cols_sq + c + 1
                text = str(zone_id)
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                tx = c*cell_size + (cell_size - text_size[0]) // 2
                ty = r*cell_size + (cell_size + text_size[1]) // 2
                color = (0,0,0) if overlay is overlay_white else (255,255,255)
                cv2.putText(overlay, text, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imwrite(output_frame_image, overlay_frame)
    cv2.imwrite(output_white_image, overlay_white)

    # Heatmap with colorbar
    norm = Normalize(vmin=0, vmax=max_count)
    mapper = cm.ScalarMappable(norm=norm, cmap='jet')
    mapper.set_array([])

    for img, title, fname in [
        (overlay_frame, 'Heatmap on Frame Background', os.path.join(output_dir, 'heatmap_frame_colorbar.png')),
        (overlay_white, 'Heatmap on White Background', os.path.join(output_dir, 'heatmap_white_colorbar.png'))
    ]:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img_rgb)
        ax.set_title(title)
        ax.axis('off')
        cbar = fig.colorbar(mapper, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Visit Frequency')
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)

def analyze_stalls_and_table():
    stall_zones = {
        'Stall 1': [337, 373, 409],
        'Stall 2': [338, 339, 374, 375, 410, 411, 446, 447],
        'Stall 5': [304, 340, 376, 412,448],
        'Stall 7': [306, 307, 342, 343, 378, 379, 414, 415, 450, 451, 486, 487],
        'Stall 8': [524, 525, 526, 527, 528, 308, 309, 310, 311, 312, 344, 345, 346, 347, 348, 380, 381, 382, 383, 384, 416, 417, 418, 419, 420, 452, 453, 454, 455, 456, 488, 489, 490, 491, 492],
    }
    staff_zones = {338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 416, 417, 418, 419, 420}

    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    success, frame = cap.read()
    cap.release()

    height, width = frame.shape[:2]
    cell_size = min(height // grid_rows, width // grid_cols)
    grid_rows_sq = (height + cell_size - 1) // cell_size
    grid_cols_sq = (width + cell_size - 1) // cell_size

    df['zone'] = df.apply(
        lambda row: (min(int(row['y']) // cell_size, grid_rows_sq - 1)) * grid_cols_sq +
                    min(int(row['x']) // cell_size, grid_cols_sq - 1) + 1,
        axis=1
    )
    df['role'] = df['zone'].apply(lambda z: 'staff' if z in staff_zones else 'customer')

    table = []
    for stall, zones in stall_zones.items():
        d = df[df['zone'].isin(zones)]
        for role in ['customer', 'staff']:
            grp = d[d['role'] == role]
            total = grp['id'].nunique()
            stops = grp.groupby('id').size()
            threshold_frames = fps * 10 if role == 'customer' else fps * 20
            stops_cnt = (stops >= threshold_frames).sum()
            table.append({'stall': stall, 'role': role, 'total': total, 'stops_threshold': stops_cnt})
    table_df = pd.DataFrame(table)
    print(table_df)

    # Bar chart - grouped bars per stall
    stalls = list(stall_zones.keys())
    x = np.arange(len(stalls))
    width_bar = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    # Customer and staff side by side
    for i, role in enumerate(['customer', 'staff']):
        dfr = table_df[table_df['role'] == role].sort_values('stall')
        totals = dfr['total'].values
        stops = dfr['stops_threshold'].values
        color_total = 'skyblue' if role == 'customer' else 'lightcoral'
        color_stops = 'blue' if role == 'customer' else 'red'
        # position for grouped bars
        pos = x - width_bar/2 + i*(width_bar)
        ax.bar(pos, totals, width=width_bar, label=f'{role} total', color=color_total)
        ax.bar(pos, stops, width=width_bar, label=f'{role} ≥{"10s" if role=="customer" else "20s"}', color=color_stops)

    ax.set_xticks(x)
    ax.set_xticklabels(stalls)
    ax.set_xlabel('Stall')
    ax.set_ylabel('Unique ID Count')
    ax.set_title('Visits per Stall by Role Overlay')
    ax.legend()
    fig.savefig(output_bar_chart, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    create_heatmap_overlay()
    analyze_stalls_and_table()
