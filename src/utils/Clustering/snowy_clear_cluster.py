import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, classification_report, adjusted_rand_score
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
                if len(values) == 4:  # x, y, z, intensity
                    x, y, z, intensity = map(float, values)
                    points_data.append([x, y, z])
                    intensity_data.append(intensity)
                else:
                    break

        return np.array(points_data), np.array(intensity_data)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None

# Function to compute features from intensity values
def compute_features(intensities):
    features = {
        'mean': np.mean(intensities),
        'median': np.median(intensities),
        'max': np.max(intensities),
        'std': np.std(intensities),
        'skewness': skew(intensities),
        'kurtosis': kurtosis(intensities),
        'range': np.ptp(intensities),
        'sum': np.sum(intensities)
    }
    return np.array(list(features.values()))

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=["Clear", "Snowy"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Clear", "Snowy"], yticklabels=["Clear", "Snowy"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    vtk_folder = r"./Data"

    # Load all VTK files and compute features
    feature_vectors = []
    file_names = []

    for filename in os.listdir(vtk_folder):
        if filename.endswith(".vtk"):
            file_path = os.path.join(vtk_folder, filename)
            points, intensities = read_custom_vtk(file_path)

            if points is not None and intensities is not None:
                feature_vector = compute_features(intensities)
                feature_vectors.append(feature_vector)
                file_names.append(filename)

    feature_vectors = np.array(feature_vectors)

    # Perform KMeans clustering with 2 clusters
    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(feature_vectors)

    # Output folders for clusters
    cluster_labels = ["Clear_Day", "Snowy_Data"]
    output_folder = os.path.join(vtk_folder, "snowy_clear_cluster")
    os.makedirs(output_folder, exist_ok=True)

    for label in cluster_labels:
        os.makedirs(os.path.join(output_folder, label), exist_ok=True)

    # Copy clustered files into respective folders
    for filename, cluster in zip(file_names, clusters):
        label = cluster_labels[cluster]
        src = os.path.join(vtk_folder, filename)
        dst = os.path.join(output_folder, label, filename)
        shutil.copy2(src, dst)
        print(f"Copied {filename} to {label} folder")

    # Cluster-wise summary statistics
    cluster_data = defaultdict(list)
    for filename, cluster in zip(file_names, clusters):
        _, intensities = read_custom_vtk(os.path.join(vtk_folder, filename))
        if intensities is not None:
            cluster_data[cluster].extend(intensities)

    print("\n=== Cluster Summary Statistics ===")
    summary_rows = []
    for cluster_id, intensities in cluster_data.items():
        intensities = np.array(intensities)
        stats = {
            'Cluster': cluster_labels[cluster_id],
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
        summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.sort_values(by="Mean"))

    # Evaluation using ground truth
    print("\n=== Evaluation Against Ground Truth ===")
    ground_truth = []
    for fname in file_names:
        if os.path.exists(os.path.join(vtk_folder, "Clear", fname)):
            ground_truth.append("Clear")
        elif os.path.exists(os.path.join(vtk_folder, "Snowy", fname)):
            ground_truth.append("Snowy")
        else:
            ground_truth.append("Unknown")

    # Map clusters to majority ground truth
    cluster_to_label = {}
    for i in range(num_clusters):
        labels = [ground_truth[j] for j in range(len(clusters)) if clusters[j] == i]
        majority = max(set(labels), key=labels.count)
        cluster_to_label[i] = majority

    predicted_labels = [cluster_to_label[c] for c in clusters]

    print(confusion_matrix(ground_truth, predicted_labels, labels=["Clear", "Snowy"]))
    print(classification_report(ground_truth, predicted_labels))
    print(f"\nClassification and comparison completed! Results saved in: {output_folder}")

    # Plot confusion matrix
    plot_confusion_matrix(ground_truth, predicted_labels, title="KMeans Clustering Confusion Matrix")


