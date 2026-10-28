def VTKToPyVistaPolyDataObj(VTKFileName):
    import pyvista as pv
    import numpy as np
    def read_custom_vtk(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        for i, line in enumerate(lines):
            if line.startswith('POINTS'):
                points_start = i + 1
                break
        
        points_data = []
        intensity_data = []
        for line in lines[points_start:]:
            values = line.split()
            if len(values) >= 4:  # Ensure we have x, y, z, and intensity
                points_data.append([float(values[0]), float(values[1]), float(values[2])])
                intensity_data.append(float(values[3]))
        
        return np.array(points_data), np.array(intensity_data)

    # Read the points and intensity from the file
    points, intensities = read_custom_vtk(VTKFileName)

    # Create a PyVista PolyData object from the points
    cloud = pv.PolyData(points)

    # Add the intensity data to the PolyData
    cloud.point_data['intensity'] = intensities

    return cloud

def VTKToOBJ(vtk_path, output_path):
    """
    Convert VTK file to OBJ file with color information preserved
    Args:
        vtk_path: Path to input VTK file
        output_path: Path where OBJ file will be saved
    """
    import numpy as np
    from pathlib import Path
    
    # Load the .vtk file
    polydata = VTKToPyVistaPolyDataObj(str(vtk_path))

    # Get point coordinates and convert to float32
    points = polydata.points.astype(np.float32)

    # Get intensity values
    intensity = polydata['intensity']

    # Convert intensity to grayscale RGB (same value for R, G, B) as float32
    normalized_intensity = (intensity / np.max(intensity)).astype(np.float32)
    colors = np.array([[i, i, i] for i in normalized_intensity], dtype=np.float32)

    # Create MTL file alongside OBJ
    mtl_path = Path(output_path).parent / "colors.mtl"
    with open(mtl_path, "w") as f:
        f.write("newmtl VertexColors\n")
        f.write("Ka 1.0 1.0 1.0\n")  # Ambient color
        f.write("Kd 1.0 1.0 1.0\n")  # Diffuse color
        f.write("Ks 0.0 0.0 0.0\n")  # Specular color
        f.write("d 1.0\n")           # Opacity
        f.write("illum 1\n")         # Illumination model

    # Write to .obj file with vertex colors
    with open(output_path, "w") as f:
        # Write MTL library reference
        f.write("mtllib colors.mtl\n")
        
        # Write vertices with their colors
        for i, (pt, (r, g, b)) in enumerate(zip(points, colors)):
            f.write(f"v {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
            f.write(f"vc {r:.6f} {g:.6f} {b:.6f}\n")  # Vertex color command
        
        # Write vertex color data
        f.write("\n# Vertex colors\n")
        f.write("usemtl VertexColors\n")
        for i in range(len(points)):
            f.write(f"p {i+1}\n")  # Point primitive with vertex index


