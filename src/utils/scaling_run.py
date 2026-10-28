import numpy as np
import pyvista as pv
from utils.conversions import VTKToPyVistaPolyDataObj
from utils.visualisation import visualisePolyDataObject, visualisePolyDataObjectInDynamic3D

def scale_vtk_with_limits(mesh, x_limits, y_limits, z_limits):
    """
    Reads a VTK file, scales the coordinates based on given limits, and visualizes the result.
    
    Args:
        vtk_path (str): Path to the input VTK file
        x_limits (tuple): (min_x, max_x) desired limits for x coordinates
        y_limits (tuple): (min_y, max_y) desired limits for y coordinates
        z_limits (tuple): (min_z, max_z) desired limits for z coordinates
    """
    
    
    # Get current coordinates
    points = mesh.points
    
    # Get current min and max for each dimension
    current_x_min, current_x_max = np.min(points[:, 0]), np.max(points[:, 0])
    current_y_min, current_y_max = np.min(points[:, 1]), np.max(points[:, 1])
    current_z_min, current_z_max = np.min(points[:, 2]), np.max(points[:, 2])
    
    # Calculate scaling factors
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
    
    # Visualize the scaled mesh
    # visualisePolyDataObjectInDynamic3D(scaled_mesh)
    
    return scaled_mesh


def uniform_scale_vtk_within_limits(mesh, x_limits, y_limits, z_limits):
    """
    Reads a VTK file, uniformly scales the coordinates to fit within the given x/y/z limits,
    and visualizes the result. Preserves aspect ratio (no distortion).
    
    Args:
        vtk_path (str): Path to the input VTK file
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

    # Visualize
    # visualisePolyDataObjectInDynamic3D(scaled_mesh)

    return scaled_mesh

#NOTE:  Example input lines to use the scaling function
# vtk_file_path = "data/input.vtk"
# x_limits = (-10.0, 10.0)
# y_limits = (-5.0, 5.0)
# z_limits = (0.0, 15.0)

# scaled_mesh = scale_vtk_with_limits(vtk_file_path, x_limits, y_limits, z_limits)


# NOTE: Example input lines to use the uniform scaling function
# vtk_file_path = "data/input.vtk"
# x_limits = (-10.0, 10.0)
# y_limits = (-5.0, 5.0)
# z_limits = (0.0, 15.0)

# scaled_mesh = uniform_scale_vtk_within_limits(vtk_file_path, x_limits, y_limits, z_limits)



