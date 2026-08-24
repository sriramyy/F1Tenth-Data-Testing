import pandas as pd
import numpy as np
import glob
import os

def analyze_perception_vs_throttling(csv_path, eps=1e-5, lidar_range_threshold=10.0):
    print("=" * 70)
    print(f"ANALYSIS REPORT: {os.path.basename(csv_path)}")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    
    # 1. Calculate Euclidean separation distance (Ego to Opp)
    df['dist'] = np.hypot(df['opp_x'] - df['ego_x'], df['opp_y'] - df['ego_y'])
    
    # 2. Movement & Frenet Update deltas
    df['d_opp_s'] = df['opp_s'].diff().abs()
    df['d_opp_d'] = df['opp_d'].diff().abs()
    df['d_ego_s'] = df['ego_s'].diff().abs()
    
    ego_moving = df['d_ego_s'] > eps
    
    # Is Frenet actively updating vs frozen?
    frenet_updating = (df['d_opp_s'] > eps) | (df['d_opp_d'] > eps)
    frenet_frozen = (df['d_opp_s'] < eps) & (df['d_opp_d'] < eps) & ego_moving
    
    # 3. Test Cases
    # Case A: Updating while OUTSIDE standard LiDAR range (> lidar_range_threshold)
    updating_far = frenet_updating & (df['dist'] > lidar_range_threshold)
    
    # Case B: Frozen while INSIDE close LiDAR range (< 5.0m, dead-ahead/close)
    frozen_close = frenet_frozen & (df['dist'] < 5.0)
    
    print("\n[1] Overall Distance vs Frenet State Summary:")
    print(f"  - Total Frames: {total_rows}")
    print(f"  - Min Distance: {df['dist'].min():.2f} m | Max Distance: {df['dist'].max():.2f} m | Median: {df['dist'].median():.2f} m")
    
    print("\n[2] Key Hypothesis Test Results:")
    print(f"  A. Frames UPDATING at long range (dist > {lidar_range_threshold}m):")
    print(f"     -> Count: {updating_far.sum()} frames (Max distance observed while updating: {df.loc[frenet_updating, 'dist'].max():.2f} m)")
    
    print(f"\n  B. Frames FROZEN at point-blank range (dist < 5.0m):")
    print(f"     -> Count: {frozen_close.sum()} frames")
    
    # 4. Detailed Distance Bins Breakdown
    bins = [0, 2, 5, 8, 12, 18, 25, 50, 100]
    df['dist_bin'] = pd.cut(df['dist'], bins=bins)
    
    bin_summary = df[ego_moving].groupby('dist_bin', observed=False).agg(
        total_frames=('dist', 'count'),
        frenet_frozen_frames=('d_opp_s', lambda s: ((s < eps) & (df.loc[s.index, 'd_opp_d'] < eps)).sum()),
        frenet_active_frames=('d_opp_s', lambda s: ((s > eps) | (df.loc[s.index, 'd_opp_d'] > eps)).sum())
    )
    bin_summary['pct_frozen'] = (bin_summary['frenet_frozen_frames'] / bin_summary['total_frames']) * 100
    
    print("\n[3] Distance Bins vs Freeze Rates (while Ego is moving):")
    print(f"  {'Distance Range (m)':<20} | {'Total':<8} | {'Active (s,d)':<14} | {'Frozen (s,d)':<14} | {'% Frozen'}")
    print("  " + "-" * 72)
    for idx, row in bin_summary.iterrows():
        print(f"  {str(idx):<20} | {int(row['total_frames']):<8} | {int(row['frenet_active_frames']):<14} | {int(row['frenet_frozen_frames']):<14} | {row['pct_frozen']:.1f}%")
        
    # 5. Consecutive streak inspection at point-blank range (if any)
    if frozen_close.sum() > 0:
        print("\n[4] Sample of Point-Blank Freezes (< 5.0m):")
        sample_rows = df[frozen_close].head(5)[['timestamp', 'ego_s', 'opp_s', 'opp_x', 'opp_y', 'dist']]
        print(sample_rows.to_string())

    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    csv_files = glob.glob("data/*.csv")
    if not csv_files:
        print("No CSV files found.")
    else:
        for f in csv_files:
            analyze_perception_vs_throttling(f)