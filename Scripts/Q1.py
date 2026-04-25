import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. File Path Handling ---
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "lines.csv")

# Define the path to the existing results folder as per file structure 
results_dir = os.path.join(script_dir, "..", "Images", "Results")

def total_least_squares(x, y):
    """Fits a line using Total Least Squares (Orthogonal Regression)"""
    x_mean, y_mean = np.mean(x), np.mean(y)
    # Center the data
    X_centered = x - x_mean
    Y_centered = y - y_mean
    data = np.vstack([X_centered, Y_centered]).T
    
    # SVD to find the normal vector (smallest singular value)
    _, _, Vh = np.linalg.svd(data)
    a, b = Vh[-1] 
    d = -(a * x_mean + b * y_mean)
    
    return a, b, d

def main():
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please place lines.csv in: {script_dir}")
        return

    # --- 2. Load Data ---
    # Using parameters defined in the assignment code snippet [cite: 10, 11, 12]
    D = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    
    # --- Question 1(a): TLS on the first line only ---
    x1, y1 = D[:, 0], D[:, 3] 
    a1, b1, d1 = total_least_squares(x1, y1)
    
    print("--- Question 1(a) Results ---")
    print(f"Parameters: a={a1:.4f}, b={b1:.4f}, d={d1:.4f}")
    print(f"Line Equation: {a1:.4f}x + {b1:.4f}y + {d1:.4f} = 0")
    print(f"Slope-Intercept: y = {(-a1/b1):.4f}x + {(-d1/b1):.4f}\n")

    # --- Question 1(b): Sequential RANSAC ---
    # Flattening data as per assignment hint [cite: 8, 13, 14]
    X_cols = D[:, :3]
    Y_cols = D[:, 3:]
    X_all = X_cols.flatten()
    Y_all = Y_cols.flatten()
    points = np.column_stack((X_all, Y_all))
    
    remaining_points = points.copy()
    threshold = 0.25  
    iterations = 2000
    
    line_results = [] 
    
    print("--- Question 1(b) Results (Sequential RANSAC) ---")
    for i in range(3):
        best_inliers_mask = np.zeros(len(remaining_points), dtype=bool)
        best_params = None
        
        for _ in range(iterations):
            sample_idx = np.random.choice(len(remaining_points), 2, replace=False)
            p1, p2 = remaining_points[sample_idx]
            
            v = p2 - p1
            if np.linalg.norm(v) < 1e-6: continue
            n = np.array([-v[1], v[0]]) 
            n /= np.linalg.norm(n)
            c = -np.dot(n, p1)
            
            distances = np.abs(np.dot(remaining_points, n) + c)
            inliers = distances < threshold
            
            if np.sum(inliers) > np.sum(best_inliers_mask):
                best_inliers_mask = inliers
                best_params = (n[0], n[1], c)
        
        if best_params:
            inlier_pts = remaining_points[best_inliers_mask]
            # Refit with TLS for precision
            ra, rb, rd = total_least_squares(inlier_pts[:,0], inlier_pts[:,1])
            
            m = -ra/rb
            c_int = -rd/rb
            print(f"Line {i+1}: y = {m:.4f}x + {c_int:.4f}")
            
            line_results.append({
                'points': inlier_pts,
                'm': m,
                'c': c_int
            })
            
            # Masking: Remove consensus to find next line [cite: 9]
            remaining_points = remaining_points[~best_inliers_mask]

    # --- 3. Plotting the Results ---
    plt.figure(figsize=(10, 7))
    colors = ['#e41a1c', '#377eb8', '#4daf4a'] 
    
    for i, res in enumerate(line_results):
        pts = res['points']
        plt.scatter(pts[:, 0], pts[:, 1], color=colors[i], s=12, alpha=0.6, label=f'Line {i+1} Inliers')
        
        x_range = np.array([X_all.min(), X_all.max()])
        plt.plot(x_range, res['m']*x_range + res['c'], color=colors[i], linewidth=2.5, linestyle='--')

    plt.title('Question 1: Sequential RANSAC Line Fitting', fontsize=14)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # --- 4. Saving the Results ---
    output_plot_path = os.path.join(results_dir, "q1_results_plot.png")
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close() 
    
    print(f"\nVisual results saved to: {output_plot_path}")

if __name__ == "__main__":
    main()