import argparse
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from utils.generate_vtk import load_vtk_data, train_random_forest, predict_and_augment, generate_frames


def get_next_run_dir(base_dir):
    """Create a unique run directory based on current date and next available number."""
    today = datetime.now().strftime("%Y%m%d")
    run_base = os.path.join(base_dir, f"{today}_run_")
    run_num = 1

    while os.path.exists(f"{run_base}{run_num}"):
        run_num += 1

    final_dir = f"{run_base}{run_num}"
    os.makedirs(final_dir)
    return final_dir


def main(args):
    # Resolve absolute paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    train_dir = os.path.join(project_root, args.train_dir)
    input_vtk = os.path.join(project_root, args.input_vtk)
    output_base = os.path.join(project_root, args.output_dir)
    output_dir = get_next_run_dir(output_base)
    model_path_abs = os.path.join(project_root, args.model_path)

    print(f"📂 Training data directory: {train_dir}")
    print(f"📂 Input VTK: {input_vtk}")
    print(f"📂 Output frames will be saved to: {output_dir}")

    if args.train:
        # Load and train
        X, y, avg_point_count = load_vtk_data(train_dir)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_random_forest(X_train, y_train)
        
        # Save model and avg_point_count
        model_dir = os.path.dirname(model_path_abs)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump((model, avg_point_count), model_path_abs)
        print(f"✅ Model saved to {model_path_abs}")

        # Evaluation
        y_pred = model.predict(X_test)
        print("📊 Evaluation:")
        print("  - Mean Squared Error:", mean_squared_error(y_test, y_pred))
        print("  - Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
        print("  - R² Score:", r2_score(y_test, y_pred))
    else:
        print(f"📦 Loading model from {model_path_abs}")
        model, avg_point_count = joblib.load(model_path_abs)
        print("✅ Model loaded successfully")

    # Compute number of frames
    num_frames = int(args.duration_s * args.frequency_hz)
    print(f"🕒 Generating {num_frames} frames for {args.duration_s} seconds at {args.frequency_hz} Hz")

    # Predict and generate
    dataobj = predict_and_augment(model, input_vtk, avg_point_count)
    generate_frames(
        dataobj,
        output_dir=output_dir,
        num_frames=num_frames,
        position_noise_std=args.position_noise_std,
        intensity_noise_std=args.intensity_noise_std,
        jitter_std=args.jitter_std,
        temporal_noise_std=args.temporal_noise_std,
        occlusion_prob=args.occlusion_prob,
        intensity_dropout_prob=args.intensity_dropout_prob,
        temporal_clustering_prob=args.temporal_clustering_prob,
        x_limits=tuple(args.x_limits),
        y_limits=tuple(args.y_limits),
        z_limits=tuple(args.z_limits),
        uniform=args.uniform
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on VTK files and generate noisy LiDAR frames.")
    parser.add_argument("--train_dir", type=str, default="caterpillar-weather-pred/data/snow_extracted", help="Relative path to training VTK files")
    parser.add_argument("--input_vtk", type=str, default="caterpillar-weather-pred/data/input.vtk", help="Relative path to input VTK file")
    parser.add_argument("--output_dir", type=str, default="caterpillar-weather-pred/outputs", help="Base output directory (subfolder created with date & run number)")
    
    # duration and frequency
    parser.add_argument("--duration_s", type=float, default=2.0, help="Duration in seconds for frame generation")
    parser.add_argument("--frequency_hz", type=float, default=20.0, help="Frame frequency (Hz)")

    # Noise and augmentation params
    parser.add_argument("--position_noise_std", type=float, default=0.01)
    parser.add_argument("--intensity_noise_std", type=float, default=0.02)
    parser.add_argument("--jitter_std", type=float, default=0.001)
    parser.add_argument("--temporal_noise_std", type=float, default=0.002)    
    parser.add_argument("--occlusion_prob", type=float, default=0.1)
    parser.add_argument("--intensity_dropout_prob", type=float, default=0.05)
    parser.add_argument("--temporal_clustering_prob", type=float, default=0.1)
    
    # New coordinate limits
    parser.add_argument("--x_limits", type=float, nargs=2, default=[0.0, 1.0], help="Min and max values for X axis")
    parser.add_argument("--y_limits", type=float, nargs=2, default=[0.0, 1.0], help="Min and max values for Y axis")
    parser.add_argument("--z_limits", type=float, nargs=2, default=[0.0, 1.0], help="Min and max values for Z axis")
    
    # Using action="store_true" means this argument defaults to False, but if passed (--uniform), it becomes True
    parser.add_argument("--uniform", action="store_true", help="Apply uniform scaling instead of axis-wise scaling")
    
    parser.add_argument("--train", action="store_true", help="Train a new model. If not set, loads model from model_path.")
    parser.add_argument("--model_path", type=str, default="caterpillar-weather-pred/outputs/model/model.pkl", help="Path to save/load the model")



    args = parser.parse_args()
    main(args)
