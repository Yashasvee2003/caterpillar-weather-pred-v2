import os
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis

# Function to read and parse the custom VTK file
def read_custom_vtk(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

        points_data = []
        intensity_data = []
        parsing_points = False

        for line in lines:
            line = line.strip()
            if line.startswith('POINTS'):
                parsing_points = True
                continue

            if parsing_points:
                values = line.split()
                if len(values) == 4:
                    x, y, z, intensity = map(float, values)
                    points_data.append([x, y, z])
                    intensity_data.append(intensity)
                else:
                    break
        return np.array(points_data), np.array(intensity_data)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None

# Function to compute stats for one file
def compute_stats(filename, intensities):
    return {
        'File': filename,
        'Count': len(intensities),
        'Mean': np.mean(intensities),
        'Median': np.median(intensities),
        'Std Dev': np.std(intensities),
        'Max': np.max(intensities),
        'Min': np.min(intensities),
        'Range': np.ptp(intensities),
        'Skewness': skew(intensities),
        'Kurtosis': kurtosis(intensities),
        'Sum': np.sum(intensities)
    }

# Main execution
if __name__ == "__main__":
    vtk_folder = r"./Data"

    stats_rows = []      # Will hold stats per file
    file_feature_matrix = []  # For clustering
    file_names = []

    for filename in os.listdir(vtk_folder):
        if filename.endswith(".vtk"):
            file_path = os.path.join(vtk_folder, filename)
            points, intensities = read_custom_vtk(file_path)

            if points is not None and intensities is not None:
                stats = compute_stats(filename, intensities)
                stats_rows.append(stats)
                file_feature_matrix.append(list(stats.values())[1:])  # Skip 'File' in feature vector
                file_names.append(filename)

    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_rows)
    print("\n=== Per-File Statistics ===")
    print(stats_df.head())

    # Clustering based on stats
    num_clusters = 3  # Low / Medium / High snow intensity
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(file_feature_matrix)

    cluster_labels = ["Low", "Medium", "High"]
    output_folder = os.path.join(vtk_folder, "snowy_intensity_cluster")
    os.makedirs(output_folder, exist_ok=True)

    for label in cluster_labels:
        os.makedirs(os.path.join(output_folder, label), exist_ok=True)

    # Copy files and add cluster info to DataFrame
    cluster_column = []
    for filename, cluster_id in zip(file_names, clusters):
        cluster_label = cluster_labels[cluster_id]
        src = os.path.join(vtk_folder, filename)
        dst = os.path.join(output_folder, cluster_label, filename)
        shutil.copy2(src, dst)
        print(f"Copied {filename} to {cluster_label} folder")
        cluster_column.append(cluster_label)

    stats_df['Cluster'] = cluster_column

    # Save stats summary CSV
    stats_csv_path = os.path.join(output_folder, "file_stats_with_clusters.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"\nPer-file statistics with cluster labels saved to: {stats_csv_path}")
