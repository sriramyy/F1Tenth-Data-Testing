import os
import glob
import csv
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree

class ContinuousFrenetConverter:
    def __init__(self, csv_path, num_dense_points=20000):
        self.centerline = self._load_centerline(csv_path)
        
        # Fit a periodic cubic B-spline to the closed track centerline
        tck, _ = splprep([self.centerline[:, 0], self.centerline[:, 1]], s=0, per=True, k=3)
        self.tck = tck
        
        # Dense interpolation for fast KDTree local search
        self.u_fine = np.linspace(0, 1, num_dense_points, endpoint=False)
        pts = np.array(splev(self.u_fine, self.tck)).T  # (M, 2)
        
        # Dense arc-length computation along spline
        diffs = np.diff(pts, axis=0)
        seg_dists = np.hypot(diffs[:, 0], diffs[:, 1])
        # Closing segment distance back to index 0
        last_seg = np.hypot(pts[0, 0] - pts[-1, 0], pts[0, 1] - pts[-1, 1])
        
        self.s_dense = np.insert(np.cumsum(seg_dists), 0, 0.0)
        self.total_track_length = self.s_dense[-1] + last_seg
        
        self.kdtree = KDTree(pts)
        self.dense_pts = pts
        
        # Spline 1st derivatives (tangents) at dense points
        derivatives = np.array(splev(self.u_fine, self.tck, der=1)).T
        tangent_norms = np.linalg.norm(derivatives, axis=1, keepdims=True) + 1e-12
        self.tangents = derivatives / tangent_norms
        
        print(f"Loaded centerline: {len(self.centerline)} points | Fitted Continuous Spline: {self.total_track_length:.2f}m")

    def _load_centerline(self, path):
        points = []
        with open(path, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    try:
                        points.append([float(row[0]), float(row[1])])
                    except ValueError:
                        continue
        
        points = np.array(points, dtype=np.float64)
        if len(points) > 0 and np.allclose(points[0], points[-1]):
            points = points[:-1]  # Spline with per=True automatically closes the loop
        return points

    def transform_batch(self, x_arr, y_arr):
        """
        Fast KDTree + local continuous projection.
        Guarantees strictly continuous (s, d) with zero discrete clamping plateaus.
        """
        pts = np.column_stack((x_arr, y_arr))
        
        # 1. Query nearest spline waypoint
        dists, nearest_idxs = self.kdtree.query(pts)
        
        # 2. Local continuous refinement along tangent
        nearest_pts = self.dense_pts[nearest_idxs]
        tangents = self.tangents[nearest_idxs]
        
        vec_to_point = pts - nearest_pts
        
        # Continuous projection onto tangent
        dt = np.sum(vec_to_point * tangents, axis=1)
        
        # Continuous s (with modulo track length wraparound)
        s_arr = (self.s_dense[nearest_idxs] + dt) % self.total_track_length
        
        # 3. Signed lateral deviation d (2D cross product with tangent)
        cross = tangents[:, 0] * vec_to_point[:, 1] - tangents[:, 1] * vec_to_point[:, 0]
        sign = np.where(cross >= 0.0, 1.0, -1.0)
        d_arr = dists * sign
        
        return s_arr, d_arr

def process_directory(input_dir="data/input", output_dir="data/output", map_csv="Spielberg_map.csv", suffix="_fixed"):
    os.makedirs(output_dir, exist_ok=True)
    converter = ContinuousFrenetConverter(map_csv)
    
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in '{input_dir}'.")
        return
        
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        fixed_filename = f"{name}{suffix}{ext}"
        out_path = os.path.join(output_dir, fixed_filename)
        print(f"Processing {filename} -> {fixed_filename}...")
        
        df = pd.read_csv(file_path)
        
        # 1. Recompute Ego Frenet
        df['ego_s'], df['ego_d'] = converter.transform_batch(df['ego_x'].to_numpy(), df['ego_y'].to_numpy())
        
        # 2. Recompute Opponent Frenet
        df['opp_s'], df['opp_d'] = converter.transform_batch(df['opp_x'].to_numpy(), df['opp_y'].to_numpy())
        
        # 3. Recompute Relative Metrics
        df['rel_x'] = df['opp_x'] - df['ego_x']
        df['rel_y'] = df['opp_y'] - df['ego_y']
        df['rel_d'] = df['opp_d'] - df['ego_d']
        
        raw_rel_s = df['opp_s'] - df['ego_s']
        track_len = converter.total_track_length
        df['rel_s'] = (raw_rel_s + track_len / 2.0) % track_len - (track_len / 2.0)
        
        df.to_csv(out_path, index=False)
        print(f" -> Successfully saved {out_path}")

if __name__ == "__main__":
    MAP_FILE = "Spielberg_map.csv"
    process_directory("data/input", "data/output", MAP_FILE)