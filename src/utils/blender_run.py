import pyvista as pv              # For 3D visualization and mesh processing
import numpy as np               # For numerical operations
from pathlib import Path         # For cross-platform file path handling
import conversions              # Local module for data conversion utilities
import visualisation            # Local module for visualization utilities

# NOTE: below is for converting the vtk file to obj file

# # Get the project root directory
# project_root = Path(__file__).parent.parent.parent

# # Define input and output paths
# vtk_path = project_root / "data" / "input.vtk"
# output_path = project_root / "outputs" / "pointcloud_colored2.obj"

# # Convert VTK to OBJ
# conversions.VTKToOBJ(vtk_path, output_path)



# NOTE: below is for visualising the vtk file in 3D

project_root = Path(__file__).parent.parent.parent
obj_path = project_root / "outputs" / "pointcloud_colored2.obj"
visualisation.visualiseBlenderObjFile(obj_path)

