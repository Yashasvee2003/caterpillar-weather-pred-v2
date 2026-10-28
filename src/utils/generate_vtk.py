def load_vtk_data(folder_path):
    from utils.conversions import VTKToPyVistaPolyDataObj
    import os
    import numpy as np
    """Load VTK files from folder and extract features and labels."""
    X, y, point_counts = [], [], []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".vtk"):
            file_path = os.path.join(folder_path, file_name)
            polydata = VTKToPyVistaPolyDataObj(file_path)
            points = polydata.points
            intensity = polydata['intensity']
            distances = np.linalg.norm(points, axis=1)

            X.extend(np.column_stack((points, distances)))
            y.extend(intensity)
            point_counts.append(len(points))

    avg_point_count = int(np.mean(point_counts))
    return np.array(X), np.array(y), avg_point_count


def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    """Train a RandomForest regressor."""
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_features="sqrt")
    model.fit(X_train, y_train)
    return model


def predict_and_augment(model, input_vtk, avg_point_count, add_noise=True):
    from utils.conversions import VTKToPyVistaPolyDataObj
    import numpy as np
    """Predict intensity for new VTK points with optional point augmentation."""
    dataobj = VTKToPyVistaPolyDataObj(input_vtk)
    points = dataobj.points.copy()
    distances = np.linalg.norm(points, axis=1)

    std_dev = int(0.1 * avg_point_count)
    sampled_point_count = int(np.random.normal(loc=avg_point_count, scale=std_dev))

    if len(points) < sampled_point_count:
        num_extra = sampled_point_count - len(points)
        extra_indices = np.random.choice(len(points), num_extra, replace=True)
        extra_points = points[extra_indices]
        extra_distances = distances[extra_indices]

        if add_noise:
            jitter = np.random.normal(scale=1.0, size=extra_points.shape)
            extra_points += jitter

        points = np.vstack([points, extra_points])
        distances = np.hstack([distances, extra_distances])

    X_new = np.column_stack((points, distances))
    predictions = model.predict(X_new)

    dataobj.points = points
    dataobj['intensity'] = predictions
    return dataobj


def generate_frames(dataobj, output_dir, num_frames=40, position_noise_std=0.01,
                    intensity_noise_std=0.02, jitter_std=0.001, temporal_noise_std=0.002,
                    occlusion_prob=0.1, intensity_dropout_prob=0.05,
                    temporal_clustering_prob=0.1,
                    x_limits=(0, 1), y_limits=(0, 1), z_limits=(0, 1), uniform = False):
    import os
    import numpy as np
    import pandas as pd
    import pyvista as pv
    from utils.scaling_run import scale_vtk_with_limits
    from utils.scaling_run import uniform_scale_vtk_within_limits

    os.makedirs(output_dir, exist_ok=True)
    points = dataobj.points.copy()
    intensity = dataobj['intensity'].copy()

    for i in range(num_frames):
        noisy_points = points + np.random.normal(0, position_noise_std, points.shape)
        jitter = np.random.normal(0, jitter_std, noisy_points.shape)
        noisy_points += jitter
        temporal_noise = np.random.normal(0, temporal_noise_std * i, noisy_points.shape)
        noisy_points += temporal_noise

        noisy_intensity = intensity + np.random.normal(0, intensity_noise_std, intensity.shape)
        dropout_mask = np.random.rand(noisy_intensity.shape[0]) < intensity_dropout_prob
        noisy_intensity[dropout_mask] = 0

        occlusion_mask = np.random.rand(noisy_points.shape[0]) > occlusion_prob
        noisy_points = noisy_points[occlusion_mask]
        noisy_intensity = noisy_intensity[occlusion_mask]

        if np.random.rand() < temporal_clustering_prob:
            cluster_radius = 0.05
            noisy_points += np.random.normal(0, cluster_radius, noisy_points.shape)

        # Create PyVista PolyData from noisy points and intensity
        cloud = pv.PolyData(noisy_points)
        cloud.point_data["intensity"] = noisy_intensity

        # Scale the cloud to fit within limits
        if uniform:
            scaled_cloud = uniform_scale_vtk_within_limits(cloud, x_limits, y_limits, z_limits)
        else:
            scaled_cloud = scale_vtk_with_limits(cloud, x_limits, y_limits, z_limits)


        def save_structured_vtk(points, intensity, output_path):
            with open(output_path, 'w') as f:         
                f.write("DATASET STRUCTURED_POINTS\n")
                f.write("DIMENSIONS 1 1 1\n")
                f.write("ORIGIN 0 0 0\n")
                f.write("SPACING 1 1 1\n")
                f.write(f"POINTS {len(points)} float\n")
                for pt, val in zip(points, intensity):
                    f.write(f"{pt[0]} {pt[1]} {pt[2]} {val}\n")

        
        # Extract the scaled data
        scaled_points = scaled_cloud.points
        scaled_intensity = scaled_cloud['intensity']

        # Save as STRUCTURED_POINTS .vtk
        output_path = os.path.join(output_dir, f"frame_{i:03d}.vtk")
        save_structured_vtk(scaled_points, scaled_intensity, output_path)
        print(f"✅ Saved frame {i+1}/{num_frames} to {output_path}")

    print("✅ Frame generation complete!")