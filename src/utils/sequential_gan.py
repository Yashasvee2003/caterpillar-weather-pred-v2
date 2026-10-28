from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import re
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KDTree
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Tuple
from tqdm import tqdm
import pyvista as pv
def scale_vtk_with_limits(mesh, x_limits, y_limits, z_limits):
    """
    Scales the coordinates based on given limits.
    
    Args:
        mesh: PyVista PolyData mesh object
        x_limits (tuple): (min_x, max_x) desired limits for x coordinates
        y_limits (tuple): (min_y, max_y) desired limits for y coordinates
        z_limits (tuple): (min_z, max_z) desired limits for z coordinates
    """

    points = mesh.points
    
    current_x_min, current_x_max = np.min(points[:, 0]), np.max(points[:, 0])
    current_y_min, current_y_max = np.min(points[:, 1]), np.max(points[:, 1])
    current_z_min, current_z_max = np.min(points[:, 2]), np.max(points[:, 2])
    
    x_scale = (x_limits[1] - x_limits[0]) / (current_x_max - current_x_min)
    y_scale = (y_limits[1] - y_limits[0]) / (current_y_max - current_y_min)
    z_scale = (z_limits[1] - z_limits[0]) / (current_z_max - current_z_min)
    
    # Scale points
    scaled_points = points.copy()
    scaled_points[:, 0] = (points[:, 0] - current_x_min) * x_scale + x_limits[0]
    scaled_points[:, 1] = (points[:, 1] - current_y_min) * y_scale + y_limits[0]
    scaled_points[:, 2] = (points[:, 2] - current_z_min) * z_scale + z_limits[0]
    
    # Create new PyVista PolyData with scaled points
    scaled_mesh = pv.PolyData(scaled_points)
    
    # Copy intensity data from original mesh
    scaled_mesh.point_data['intensity'] = mesh.point_data['intensity']
    
    return scaled_mesh

def uniform_scale_vtk_within_limits(mesh, x_limits, y_limits, z_limits):
    """
    Uniformly scales the coordinates to fit within the given x/y/z limits.
    Preserves aspect ratio (no distortion).
    
    Args:
        mesh: PyVista PolyData mesh object
        x_limits (tuple): (min_x, max_x) desired limits for x coordinates
        y_limits (tuple): (min_y, max_y) desired limits for y coordinates
        z_limits (tuple): (min_z, max_z) desired limits for z coordinates
    """
    # Get original coordinates
    points = mesh.points
    original_min = np.min(points, axis=0)
    original_max = np.max(points, axis=0)
    original_range = original_max - original_min

    # Get target limits and ranges
    target_min = np.array([x_limits[0], y_limits[0], z_limits[0]])
    target_max = np.array([x_limits[1], y_limits[1], z_limits[1]])
    target_range = target_max - target_min

    # Compute uniform scale factor based on the largest original dimension and smallest target range
    scale_factor = np.min(target_range / original_range)

    # Shift to origin, scale, then offset into target box (centered)
    shifted_points = points - original_min
    scaled_points = shifted_points * scale_factor

    scaled_range = np.max(scaled_points, axis=0) - np.min(scaled_points, axis=0)
    offset = target_min + (target_range - scaled_range) / 2
    scaled_points += offset

    # Create new PyVista PolyData with scaled points
    scaled_mesh = pv.PolyData(scaled_points)

    # Copy intensity data
    scaled_mesh.point_data['intensity'] = mesh.point_data['intensity']

    return scaled_mesh

def parse_custom_vtk(vtk_path):
    coordinates = []
    intensity = []
    
    with open(vtk_path, 'r') as f:
        lines = f.readlines()
        
        points_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("POINTS"):
                points_start = i + 1
                break

        if points_start is not None:
            for line in lines[points_start:]:
                if line.strip() and not line.strip().startswith(("SCALARS", "LOOKUP")):
                    values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                    if len(values) >= 4:
                        coordinates.append(values[:3])
                        intensity.append(values[3])
    
    # Create PyVista PolyData object
    coords_array = np.array(coordinates)
    intensity_array = np.array(intensity)
    
    mesh = pv.PolyData(coords_array)
    mesh.point_data['intensity'] = intensity_array
    
    return mesh

def compute_surface_normals(points, k=10):
    tree = KDTree(points)
    normals = np.zeros(points.shape)
    
    for i, point in enumerate(points):
        _, indices = tree.query([point], k=k)
        neighbors = points[indices[0]]
        cov = np.cov(neighbors.T)
        U, S, Vt = np.linalg.svd(cov)
        normals[i] = Vt[2]
        
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)

def create_range_image(coords, intensity, img_size=(64, 2048)):
    normals = compute_surface_normals(coords)
    ray_dirs = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    incidence_angles = np.arccos(np.sum(normals * ray_dirs, axis=1))
        
    coords_min = coords.min(0)
    coords_max = coords.max(0)
    coords_norm = 2 * ((coords - coords_min) / (coords_max - coords_min)) - 1
    intensity_norm = (intensity / intensity.max()) * 2 - 1
    incidence_norm = incidence_angles / (np.pi/2)
    
    azimuth = np.arctan2(coords[:,1], coords[:,0])
    elevation = np.arcsin(coords[:,2] / np.linalg.norm(coords, axis=1))
    
    range_image = np.zeros((img_size[0], img_size[1], 5))
    indices = []
    
    for i in range(len(coords)):
        u = int((azimuth[i] + np.pi) / (2*np.pi) * (img_size[1]-1))
        v = int((elevation[i] + np.pi/2) / np.pi * (img_size[0]-1))
        
        if 0 <= u < img_size[1] and 0 <= v < img_size[0]:
            range_image[v,u,0:3] = coords_norm[i]
            range_image[v,u,3] = incidence_norm[i]
            range_image[v,u,4] = intensity_norm[i]
            indices.append((v, u, i))
    
    return range_image, indices, (coords_min, coords_max, intensity.max())

class WeatherCondition(Enum):
    CLEAR = 0
    LIGHT_SNOW = 1
    MODERATE_SNOW = 2
    HEAVY_SNOW = 3

class WeatherAwareGenerator(nn.Module):
    def __init__(self, latent_dim=64, weather_dim=4, temporal_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.weather_dim = weather_dim
        self.temporal_dim = temporal_dim
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        # Weather condition embedding
        self.weather_embedding = nn.Sequential(
            nn.Linear(weather_dim, 128),
            nn.LeakyReLU(0.2)
        )
        
        # Temporal variation embedding
        self.temporal_embedding = nn.Sequential(
            nn.Linear(temporal_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2)
        )
        
        # Latent noise integration
        self.noise_mapper = nn.Sequential(
            nn.Linear(latent_dim + 128 + 128, 128 * 16 * 512),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x, noise=None, weather_condition=None, temporal_noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), self.latent_dim, device=x.device)
            
        if weather_condition is None:
            weather_condition = torch.zeros(x.size(0), self.weather_dim, device=x.device)
            
        if temporal_noise is None:
            temporal_noise = torch.randn(x.size(0), self.temporal_dim, device=x.device)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        # Process weather condition
        weather_features = self.weather_embedding(weather_condition)
        
        # Process temporal variation
        temporal_features = self.temporal_embedding(temporal_noise)
        
        # Combine noise with weather and temporal features
        combined_noise = torch.cat([noise, weather_features, temporal_features], 1)
        noise_feature = self.noise_mapper(combined_noise).view(x.size(0), 128, 16, 512)
        
        # Combine encoded features with noise
        combined = torch.cat([e2, noise_feature], 1)
        
        # Decoder
        d1 = self.dec1(combined)
        d1 = torch.cat([d1, e1], 1)
        out = self.dec2(d1)
        return out

class WeatherAwareDiscriminator(nn.Module):
    def __init__(self, weather_dim=4):
        super().__init__()
        self.weather_dim = weather_dim
        
        self.model = nn.Sequential(
            nn.Conv2d(5 + weather_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, y, weather_condition):
        # Expand weather condition to match spatial dimensions
        weather_expanded = weather_condition.view(weather_condition.size(0), -1, 1, 1)
        weather_expanded = weather_expanded.expand(-1, -1, x.size(2), x.size(3))
        
        combined = torch.cat([x, y, weather_expanded], 1)
        return self.model(combined)

class SequentialPointCloudGenerator:
    def __init__(self, clear_vtk_path: str, snowy_vtk_path: str, output_dir="generated_data", pd_dir='generated_pd', 
                 x_limits=(0.0, 1.0), y_limits=(0.0, 1.0), z_limits=(0.0, 1.0), scaling_mode='independent',
                 load_existing_model=True, model_path="trained_gan_model.pth"):
        self.clear_vtk_path = clear_vtk_path
        self.snowy_vtk_path = snowy_vtk_path
        self.output_dir = output_dir
        self.pd_dir = pd_dir
        
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.z_limits = z_limits
        self.scaling_mode = scaling_mode
        self.model_path = model_path

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.clear_mesh = parse_custom_vtk(clear_vtk_path)
        self.snowy_mesh = parse_custom_vtk(snowy_vtk_path)
        
        # Extract coordinates and intensity for backward compatibility
        self.clear_coords = self.clear_mesh.points
        self.clear_intensity = self.clear_mesh.point_data['intensity']
        self.snowy_coords = self.snowy_mesh.points
        self.snowy_intensity = self.snowy_mesh.point_data['intensity']
        
        os.makedirs(output_dir, exist_ok=True)
        self.generator = WeatherAwareGenerator().to(self.device)
        self.discriminator = WeatherAwareDiscriminator().to(self.device)
        
        # Try to load existing model, otherwise train new one
        if load_existing_model and os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            self.load_model(model_path)
        else:
            print("Training new model...")
            self._train_generator()
        
    def save_model(self, model_path=None):
        """Save the trained generator and discriminator"""
        if model_path is None:
            model_path = self.model_path
        
        # Ensure the model directory exists
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path=None):
        """Load a pre-trained model"""
        if model_path is None:
            model_path = self.model_path
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        print(f"Model loaded from {model_path}")
        
    def _get_weather_condition_tensor(self, condition: WeatherCondition) -> torch.Tensor:
        weather_tensor = torch.zeros(1, 4, device=self.device)
        weather_tensor[0, condition.value] = 1
        return weather_tensor
    
    def _apply_scaling(self, mesh):
        """Apply scaling to PyVista mesh using the existing scaling functions"""
        if self.scaling_mode == 'independent':
            return scale_vtk_with_limits(mesh, self.x_limits, self.y_limits, self.z_limits)
        elif self.scaling_mode == 'uniform':
            return uniform_scale_vtk_within_limits(mesh, self.x_limits, self.y_limits, self.z_limits)
        else:
            raise ValueError(f"Unknown scaling mode: {self.scaling_mode}")
        
    def _train_generator(self, epochs=100, batch_size=1, save_model=True):
        criterion_GAN = nn.BCELoss()
        criterion_L1 = nn.L1Loss()
        
        opt_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        print(f"Device being used for training: {self.device}")
        
        for epoch in tqdm(range(epochs), desc="GAN Training"):
            clear_ri, _, _ = create_range_image(self.clear_coords, self.clear_intensity)
            snowy_ri, _, _ = create_range_image(self.snowy_coords, self.snowy_intensity)
            
            clear_inputs = torch.FloatTensor(clear_ri[..., :4].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            snowy_inputs = torch.FloatTensor(snowy_ri[..., :4].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            
            clear_targets = torch.FloatTensor(clear_ri[..., 4:5].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            snowy_targets = torch.FloatTensor(snowy_ri[..., 4:5].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            
            batch_size = clear_inputs.size(0)
            valid = torch.ones((batch_size, 1, 8, 256), device=self.device)
            fake = torch.zeros((batch_size, 1, 8, 256), device=self.device)
            
            # Train Generator
            opt_G.zero_grad()
            
            # Generate multiple variations
            num_variations = 2
            loss_GAN_clear = 0
            loss_GAN_snowy = 0
            loss_L1_clear = 0
            loss_L1_snowy = 0
            
            for _ in range(num_variations):
                noise = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
                temporal_noise = torch.randn(batch_size, self.generator.temporal_dim, device=self.device)
                
                clear_weather = self._get_weather_condition_tensor(WeatherCondition.CLEAR)
                snowy_weather = self._get_weather_condition_tensor(WeatherCondition.HEAVY_SNOW)
                
                # Generate for both conditions
                fake_clear = self.generator(clear_inputs, noise, clear_weather, temporal_noise)
                fake_snowy = self.generator(snowy_inputs, noise, snowy_weather, temporal_noise)
                
                # Accumulate losses
                loss_GAN_clear += criterion_GAN(self.discriminator(clear_inputs, fake_clear, clear_weather), valid)
                loss_GAN_snowy += criterion_GAN(self.discriminator(snowy_inputs, fake_snowy, snowy_weather), valid)
                loss_L1_clear += criterion_L1(fake_clear, clear_targets)
                loss_L1_snowy += criterion_L1(fake_snowy, snowy_targets)
                
                # Clear cache after each variation
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Average the losses
            loss_GAN_clear /= num_variations
            loss_GAN_snowy /= num_variations
            loss_L1_clear /= num_variations
            loss_L1_snowy /= num_variations
            
            loss_G = (loss_GAN_clear + loss_GAN_snowy) * 0.5 + (loss_L1_clear + loss_L1_snowy) * 12.5
            loss_G.backward()
            opt_G.step()
            
            # Train Discriminator
            opt_D.zero_grad()
            
            # Real losses
            loss_real_clear = criterion_GAN(self.discriminator(clear_inputs, clear_targets, clear_weather), valid)
            loss_real_snowy = criterion_GAN(self.discriminator(snowy_inputs, snowy_targets, snowy_weather), valid)
            
            # Generate new fake samples
            noise = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
            temporal_noise = torch.randn(batch_size, self.generator.temporal_dim, device=self.device)
            
            fake_clear = self.generator(clear_inputs, noise, clear_weather, temporal_noise)
            fake_snowy = self.generator(snowy_inputs, noise, snowy_weather, temporal_noise)
            
            # Fake losses
            loss_fake_clear = criterion_GAN(self.discriminator(clear_inputs, fake_clear.detach(), clear_weather), fake)
            loss_fake_snowy = criterion_GAN(self.discriminator(snowy_inputs, fake_snowy.detach(), snowy_weather), fake)
            
            loss_D = (loss_real_clear + loss_real_snowy + loss_fake_clear + loss_fake_snowy) * 0.25
            loss_D.backward()
            opt_D.step()
            
            # Clear cache after each epoch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Free up memory
            del fake_clear, fake_snowy, loss_G, loss_D
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        # Save model after training
        if save_model:
            self.save_model()
    
    def generate_weather_sequence(self, weather_sequence: List[Tuple[WeatherCondition, int]], frame_interval_seconds: int = 10):
        total_frames = sum(duration // frame_interval_seconds for _, duration in weather_sequence)
        
        # Start with clear day data
        current_mesh = self.clear_mesh.copy()
        coords = current_mesh.points
        intensity = current_mesh.point_data['intensity']
        ri, indices, (coords_min, coords_max, int_max) = create_range_image(coords, intensity)
        inputs = torch.FloatTensor(ri[..., :4].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        # Create a timestamp for sequence start
        start_time = datetime.now()
        frame_count = 0
        
        # Create progress bar for generation
        pbar = tqdm(total=total_frames, desc="Generating VTK Files")
        
        for weather_condition, duration in weather_sequence:
            frames_in_phase = duration // frame_interval_seconds
            
            for frame in range(frames_in_phase):
                current_time = start_time + timedelta(seconds=frame_count * frame_interval_seconds)
                timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
                
                # Generate unique noise vectors for each frame
                base_noise = torch.randn(1, self.generator.latent_dim, device=self.device)
                temporal_noise = torch.randn(1, self.generator.temporal_dim, device=self.device)
                
                weather_tensor = self._get_weather_condition_tensor(weather_condition)
                
                with torch.no_grad():
                    generated = self.generator(
                        inputs, 
                        base_noise, 
                        weather_tensor,
                        temporal_noise
                    ).cpu().numpy().squeeze()
                
                base_intensity = np.zeros(len(coords))
                for v, u, orig_idx in indices:
                    base_intensity[orig_idx] = (generated[v, u] + 1) / 2
                
                # Get weather-adjusted intensity and coordinates
                weather_intensity, final_coords = self._apply_weather_intensity_model(coords, base_intensity, weather_condition)
                
                # Create temporary mesh for scaling
                temp_mesh = pv.PolyData(final_coords)
                temp_mesh.point_data['intensity'] = weather_intensity * int_max
                
                # Apply scaling using PyVista functions
                scaled_mesh = self._apply_scaling(temp_mesh)
                
                # Extract scaled data
                scaled_coords = scaled_mesh.points
                new_intensity = scaled_mesh.point_data['intensity']
                
                # Create output data with scaled coordinates
                output_data = np.column_stack((scaled_coords, new_intensity))
                
                weather_name = weather_condition.name.lower()
                
                # Create weather-specific subfolder
                weather_output_dir = os.path.join(self.output_dir, weather_name)
                os.makedirs(weather_output_dir, exist_ok=True)
                
                output_file = os.path.join(weather_output_dir, f"{weather_name}_{timestamp_str}.vtk")
                self._save_vtk(output_file, output_data)
                
                # Create DataFrame
                # df = pd.DataFrame(output_data, columns=["x", "y", "z", "intensity"])
                # csv_filename = os.path.join(self.pd_dir, f"{weather_name}_frame_{frame_count:03d}.csv")
                # df.to_csv(csv_filename, index=False)
                
                # Update inputs for next frame using scaled coordinates
                current_mesh = scaled_mesh.copy()
                coords = current_mesh.points
                intensity = current_mesh.point_data['intensity']
                ri, indices, _ = create_range_image(coords, intensity)
                inputs = torch.FloatTensor(ri[..., :4].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                
                frame_count += 1
                pbar.update(1)
        
        pbar.close()
    
    def _apply_weather_intensity_model(self, coords, base_intensity, weather_condition):
        # Reuse the same weather intensity model from lotsoftime.py
        distances = np.linalg.norm(coords, axis=1)
        max_distance = np.max(distances)
        distances_norm = distances / max_distance
        
        # Get base intensity characteristics
        clear_min = np.min(self.clear_intensity)
        clear_max = np.max(self.clear_intensity)
        snowy_min = np.min(self.snowy_intensity)
        snowy_max = np.max(self.snowy_intensity)
        
        # Define exponential distance decay factor
        def distance_decay(dist_norm, strength):
            return np.exp(-strength * dist_norm)
        
        # Define weather-specific parameters
        if weather_condition == WeatherCondition.CLEAR:
            base_scale = 1.0
            decay_strength = 1.5
            intensity_range = (clear_min, clear_max)
            noise_scale = 0.15
            point_density_factor = 1.0
            
        elif weather_condition == WeatherCondition.LIGHT_SNOW:
            base_scale = 1.0
            decay_strength = 2.0  # Reduced from 2.5
            intensity_range = (clear_min, clear_max * 0.85)  # Increased from 80% to 85%
            noise_scale = 0.05  # Reduced from 0.1
            point_density_factor = 1.02  # Reduced from 1.05
            
        elif weather_condition == WeatherCondition.MODERATE_SNOW:
            base_scale = 1.0
            decay_strength = 2.2  # Reduced from 3.0
            intensity_range = (clear_min, clear_max * 0.7)  # Increased from 60% to 70%
            noise_scale = 0.08  # Reduced from 0.15
            point_density_factor = 1.03  # Reduced from 1.1
            
        elif weather_condition == WeatherCondition.HEAVY_SNOW:
            base_scale = 1.0
            decay_strength = 2.3  # Further reduced from 2.5
            intensity_range = (clear_min, clear_max * 0.65)  # Increased from 60% to 65%
            noise_scale = 0.1  # Further reduced from 0.15
            point_density_factor = 1.04  # Further reduced from 1.05
            
        else:
            base_scale = 1.0
            decay_strength = 2.0
            intensity_range = (clear_min, clear_max)
            noise_scale = 0.1
            point_density_factor = 1.0
        
        # Calculate distance-based decay
        decay = distance_decay(distances_norm, decay_strength)
        
        scaled_intensity = base_intensity * base_scale
        decayed_intensity = scaled_intensity * decay
        
        # Add distance-dependent noise
        noise = np.random.normal(0, noise_scale, size=len(base_intensity)) * (1 - distances_norm)
        noisy_intensity = decayed_intensity + noise
        
        # Normalize to weather-specific intensity range
        normalized_intensity = np.clip(noisy_intensity, 0, 1)
        final_intensity = (normalized_intensity * (intensity_range[1] - intensity_range[0]) + 
                         intensity_range[0])

        if weather_condition == WeatherCondition.HEAVY_SNOW:
            close_points = distances_norm < 0.15  # Further reduced from 0.2 to 0.15
            final_intensity[close_points] *= 1.2  # Further reduced from 1.3 to 1.2
            
            num_drifts = 3  # Further reduced from 4 to 3
            for _ in range(num_drifts):
                center_idx = np.random.choice(len(coords))
                center = coords[center_idx]
                dist_to_center = np.linalg.norm(coords - center, axis=1)
                drift_mask = dist_to_center < max_distance * 0.05  # Further reduced from 0.08 to 0.05
                final_intensity[drift_mask] *= 1.1  # Further reduced from 1.2 to 1.1
                
        elif weather_condition == WeatherCondition.MODERATE_SNOW:
            close_points = distances_norm < 0.15  # Reduced from 0.2
            final_intensity[close_points] *= 1.15  # Reduced from 1.3
            
            num_drifts = 3  # Reduced from 5
            for _ in range(num_drifts):
                center_idx = np.random.choice(len(coords))
                center = coords[center_idx]
                dist_to_center = np.linalg.norm(coords - center, axis=1)
                drift_mask = dist_to_center < max_distance * 0.05  # Reduced from 0.08
                final_intensity[drift_mask] *= 1.1  # Reduced from 1.2
                
        elif weather_condition == WeatherCondition.LIGHT_SNOW:
            close_points = distances_norm < 0.1  # Reduced from 0.15
            final_intensity[close_points] *= 1.05  # Reduced from 1.1
            
            num_drifts = 2  # Reduced from 3
            for _ in range(num_drifts):
                center_idx = np.random.choice(len(coords))
                center = coords[center_idx]
                dist_to_center = np.linalg.norm(coords - center, axis=1)
                drift_mask = dist_to_center < max_distance * 0.03  # Reduced from 0.05
                final_intensity[drift_mask] *= 1.05  # Reduced from 1.1
        
        # Apply point density sampling
        final_coords, final_intensity = self._apply_point_density_sampling(
            coords, final_intensity, point_density_factor, weather_condition
        )
        
        return np.clip(final_intensity, clear_min, snowy_max * 1.5), final_coords
    
    def _apply_point_density_sampling(self, coords, intensity, density_factor, weather_condition):
        # Reuse the same point density sampling from lotsoftime.py
        original_count = len(coords)
        target_count = int(original_count * density_factor)
        
        # For snowy conditions, ensure we never go below original count
        if weather_condition in [WeatherCondition.LIGHT_SNOW, WeatherCondition.MODERATE_SNOW, WeatherCondition.HEAVY_SNOW]:
            target_count = max(target_count, original_count)
        
        if abs(density_factor - 1.0) < 0.01:
            return coords, intensity
        
        if density_factor < 1.0 and weather_condition == WeatherCondition.CLEAR:
            # Only allow point reduction for clear conditions
            distances = np.linalg.norm(coords, axis=1)
            distances_norm = distances / distances.max()
            
            keep_prob = np.exp(-2 * distances_norm)
            keep_prob = keep_prob / keep_prob.max()
            keep_prob = keep_prob * (1 / density_factor)
            keep_prob = np.clip(keep_prob, 0, 1)
            
            keep_mask = np.random.random(len(coords)) < keep_prob
            
            if np.sum(keep_mask) < target_count:
                shortfall = target_count - np.sum(keep_mask)
                additional_indices = np.random.choice(
                    np.where(~keep_mask)[0], 
                    size=shortfall, 
                    replace=False
                )
                keep_mask[additional_indices] = True
            
            # Keep original points but allow some variation for weather effects
            sampled_coords = coords[keep_mask]
            sampled_intensity = intensity[keep_mask]
            
        else:
            # For snowy conditions or when density_factor >= 1.0, only add points
            sampled_coords = coords.copy()
            sampled_intensity = intensity.copy()
            
            additional_count = target_count - original_count
            if additional_count > 0:
                tree = KDTree(coords)
                
                for _ in range(additional_count):
                    idx = np.random.randint(0, original_count)
                    point = coords[idx]
                    
                    distances, indices = tree.query(point.reshape(1, -1), k=5)
                    neighbor_idx = np.random.choice(indices[0][1:])
                    neighbor = coords[neighbor_idx]
                    
                    alpha = np.random.random()
                    new_point = alpha * point + (1 - alpha) * neighbor
                    
                    current_z_min = np.min(coords[:, 2])
                    current_z_max = np.max(coords[:, 2])
                    z_range = current_z_max - current_z_min
                    min_safe_z = max(current_z_min + (z_range * 0.15), 0.1)
                    
                    if new_point[2] < min_safe_z:
                        new_point[2] = min_safe_z + np.random.random() * (z_range * 0.1)
                    elif new_point[2] > current_z_max + (z_range * 0.2):
                        new_point[2] = current_z_max + (z_range * 0.1) + np.random.random() * (z_range * 0.1)
                    
                    new_intensity = alpha * intensity[idx] + (1 - alpha) * intensity[neighbor_idx]
                    
                    if weather_condition == WeatherCondition.HEAVY_SNOW:
                        xy_noise_scale = 0.015  # Further reduced from 0.02
                        z_noise_scale = 0.03   # Further reduced from 0.04
                    elif weather_condition == WeatherCondition.MODERATE_SNOW:
                        xy_noise_scale = 0.02  # Reduced from 0.05
                        z_noise_scale = 0.04  # Reduced from 0.10
                    elif weather_condition == WeatherCondition.LIGHT_SNOW:
                        xy_noise_scale = 0.015  # Reduced from 0.03
                        z_noise_scale = 0.03  # Reduced from 0.06
                    else:
                        xy_noise_scale = 0.005  # Reduced from 0.01
                        z_noise_scale = 0.01  # Reduced from 0.02
                    
                    # Add distance-based noise scaling for more controlled variations
                    distance_factor = np.linalg.norm(new_point) / np.linalg.norm(coords, axis=1).max()
                    scaled_xy_noise = xy_noise_scale * (0.5 + 0.5 * distance_factor)
                    scaled_z_noise = z_noise_scale * (0.5 + 0.5 * distance_factor)
                    
                    position_noise_xy = np.random.normal(0, scaled_xy_noise, size=2)
                    position_noise_z = np.random.normal(0, scaled_z_noise, size=1)
                    
                    new_point[:2] += position_noise_xy
                    new_point[2] += position_noise_z[0]
                    
                    # Add coordinate validation to prevent excessive displacement
                    if weather_condition in [WeatherCondition.HEAVY_SNOW, WeatherCondition.MODERATE_SNOW, WeatherCondition.LIGHT_SNOW]:
                        max_displacement = 0.02  # Reduced from 0.05 to 2% of coordinate range
                        original_point = coords[idx]
                        displacement = np.linalg.norm(new_point - original_point)
                        coord_range = np.max(coords) - np.min(coords)
                        max_allowed = max_displacement * coord_range
                        
                        if displacement > max_allowed:
                            # Scale down the noise to stay within bounds
                            scale_factor = max_allowed / displacement
                            new_point = original_point + (new_point - original_point) * scale_factor
                    
                    if new_point[2] < 0:
                        new_point[2] = 0.1
                    
                    intensity_noise = np.random.normal(0, xy_noise_scale)
                    new_intensity += intensity_noise
                    
                    sampled_coords = np.vstack((sampled_coords, new_point))
                    sampled_intensity = np.append(sampled_intensity, new_intensity)
        
        current_z_min = np.min(coords[:, 2])
        current_z_max = np.max(coords[:, 2])
        z_range = current_z_max - current_z_min
        
        min_safe_z = max(current_z_min + (z_range * 0.2), 0.1)
        
        if weather_condition == WeatherCondition.HEAVY_SNOW:
            max_z_buffer = z_range * 0.3
        elif weather_condition == WeatherCondition.MODERATE_SNOW:
            max_z_buffer = z_range * 0.25
        elif weather_condition == WeatherCondition.LIGHT_SNOW:
            max_z_buffer = z_range * 0.2
        else:  # CLEAR
            max_z_buffer = z_range * 0.1
        
        sampled_coords[:, 2] = np.clip(sampled_coords[:, 2], 
                                      min_safe_z, 
                                      current_z_max + max_z_buffer)
        
        current_xy_min = np.min(coords[:, :2], axis=0)
        current_xy_max = np.max(coords[:, :2], axis=0)
        xy_range = current_xy_max - current_xy_min
        
        if weather_condition == WeatherCondition.HEAVY_SNOW:
            xy_buffer = xy_range * 0.2
        elif weather_condition == WeatherCondition.MODERATE_SNOW:
            xy_buffer = xy_range * 0.15
        elif weather_condition == WeatherCondition.LIGHT_SNOW:
            xy_buffer = xy_range * 0.1
        else:  # CLEAR
            xy_buffer = xy_range * 0.05
        
        sampled_coords[:, 0] = np.clip(sampled_coords[:, 0], 
                                      current_xy_min[0] - xy_buffer[0], 
                                      current_xy_max[0] + xy_buffer[0])
        sampled_coords[:, 1] = np.clip(sampled_coords[:, 1], 
                                      current_xy_min[1] - xy_buffer[1], 
                                      current_xy_max[1] + xy_buffer[1])
        
        # Final validation: ensure no z-coordinates are below 0
        sampled_coords[:, 2] = np.maximum(sampled_coords[:, 2], 0.1)
        
        return sampled_coords, sampled_intensity
    
    def _save_vtk(self, filename, data):
        header = (
            "DATASET STRUCTURED_POINTS\n"
            "DIMENSIONS 1 1 1\n"
            "ORIGIN 0 0 0\n"
            "SPACING 1 1 1\n"
            f"POINTS {len(data)} float\n"
        )
        
        coords = data[:, :3]
        intensity = data[:, 3]
        
        intensity_min = np.min(intensity)
        intensity_max = np.max(intensity)
        normalized_intensity = (intensity - intensity_min) / (intensity_max - intensity_min)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(header)
            
            for i in range(len(coords)):
                f.write(f"{coords[i,0]} {coords[i,1]} {coords[i,2]} {normalized_intensity[i]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate weather sequence with specified conditions and durations')
    parser.add_argument('--weather-conditions', nargs='+', required=True,
                       choices=['CLEAR', 'LIGHT_SNOW', 'MODERATE_SNOW', 'HEAVY_SNOW'],
                       help='Weather conditions to generate')
    parser.add_argument('--durations', nargs='+', type=int, required=True,
                       help='Duration in seconds for each weather condition (must match number of weather conditions)')
    parser.add_argument('--frame-interval', type=int, default=10,
                       help='Frame interval in seconds (default: 10)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for generated data (default: outputs/gan)')
    parser.add_argument('--x-limits', nargs=2, type=float, default=[0.0, 1.0],
                       help='X coordinate limits (min_x max_x) (default: 0.0 1.0)')
    parser.add_argument('--y-limits', nargs=2, type=float, default=[0.0, 1.0],
                       help='Y coordinate limits (min_y max_y) (default: 0.0 1.0)')
    parser.add_argument('--z-limits', nargs=2, type=float, default=[0.0, 1.0],
                       help='Z coordinate limits (min_z max_z) (default: 0.0 1.0)')
    parser.add_argument('--scaling-mode', type=str, choices=['independent', 'uniform'], default='independent',
                       help='Scaling mode: independent (stretch to fit) or uniform (preserve aspect ratio) (default: independent)')
    parser.add_argument('--model-path', type=str, default='./outputs/model/trained_gan_model.pth',
                       help='Path to save/load the trained model (default: trained_gan_model.pth)')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if model exists')
    
    args = parser.parse_args()
    
    # Validate that number of weather conditions matches number of durations
    if len(args.weather_conditions) != len(args.durations):
        parser.error("Number of weather conditions must match number of durations")
    
    # Set default paths
    curdir = os.getcwd()
    clear_vtk_path = os.path.join(curdir, "data/clear_day.vtk")
    snowy_vtk_path = os.path.join(curdir, "data/snowwy.vtk")
    output_dir = args.output_dir if args.output_dir else os.path.join(curdir, "outputs/gan")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create weather sequence from command line arguments
    weather_sequence = []
    for condition_str, duration in zip(args.weather_conditions, args.durations):
        condition = WeatherCondition[condition_str]
        weather_sequence.append((condition, duration))
    
    generator = SequentialPointCloudGenerator(
        clear_vtk_path=clear_vtk_path,
        snowy_vtk_path=snowy_vtk_path,
        output_dir=output_dir,
        pd_dir='generated_pd',  # Keep for compatibility but not used
        x_limits=tuple(args.x_limits),
        y_limits=tuple(args.y_limits),
        z_limits=tuple(args.z_limits),
        scaling_mode=args.scaling_mode,
        load_existing_model=not args.force_retrain,
        model_path=args.model_path
    )

    generator.generate_weather_sequence(weather_sequence, frame_interval_seconds=args.frame_interval)
