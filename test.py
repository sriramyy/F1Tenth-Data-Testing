import pandas as pd
import numpy as np

def analyze_opponent_freezes(csv_path, s_col='opp_s', d_col='opp_d', ego_s_col='ego_s', min_consecutive_rows=5):
    """
    Scans a telemetry CSV for segments where the opponent's s and d are frozen
    while the ego vehicle is actively moving.
    """
    print(f"Analyzing {csv_path}...")
    
    # Read only the required columns to save memory
    cols = [s_col, d_col, ego_s_col]
    df = pd.read_csv(csv_path, usecols=lambda c: c in cols)
    
    # Calculate step-to-step deltas
    opp_s_diff = df[s_col].diff().fillna(0)
    opp_d_diff = df[d_col].diff().fillna(0)
    ego_s_diff = df[ego_s_col].diff().fillna(0)
    
    # Detect exact freezes (opp static, ego moving)
    # Using a tight epsilon for floating-point safety
    eps = 1e-6
    is_opp_frozen = (opp_s_diff.abs() < eps) & (opp_d_diff.abs() < eps)
    is_ego_moving = ego_s_diff.abs() > eps
    
    freeze_mask = is_opp_frozen & is_ego_moving
    
    # Group consecutive frozen frames
    freeze_groups = (~freeze_mask).cumsum()[freeze_mask]
    group_counts = freeze_groups.value_counts()
    
    # Filter by minimum duration/rows
    significant_freezes = group_counts[group_counts >= min_consecutive_rows]
    
    total_rows = len(df)
    total_frozen_rows = freeze_mask.sum()
    pct_frozen = (total_frozen_rows / total_rows) * 100
    
    print("\n--- Summary Statistics ---")
    print(f"Total frames: {total_rows}")
    print(f"Frozen opp frames (while ego moving): {total_frozen_rows} ({pct_frozen:.2f}%)")
    print(f"Freeze streaks >= {min_consecutive_rows} consecutive frames: {len(significant_freezes)}")
    
    if not significant_freezes.empty:
        print(f"Longest continuous freeze: {significant_freezes.max()} frames")
        
    return {
        "total_rows": total_rows,
        "frozen_rows": total_frozen_rows,
        "pct_frozen": pct_frozen,
        "streak_counts": significant_freezes
    }

# Run the analysis
analyze_opponent_freezes("data/richard_existing.csv")