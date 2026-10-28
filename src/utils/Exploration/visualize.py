import pyvista as pv
import numpy as np
import pandas as pd
import plotly.express as px

# === 1. Read StructuredPoints VTK File ===
def read_structured_points_as_polydata(vtk_path):
    with open(vtk_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.startswith('POINTS'):
            points_start = i + 1
            break

    points = []
    intensities = []
    for line in lines[points_start:]:
        values = line.strip().split()
        if len(values) >= 4:
            x, y, z, intensity = map(float, values[:4])
            points.append([x, y, z])
            intensities.append(intensity)

    points = np.array(points, dtype=np.float32)
    intensities = np.array(intensities, dtype=np.float32)

    cloud = pv.PolyData(points)
    cloud.point_data['intensity'] = intensities
    return cloud

# === 2. Top View ===
def visualisingStructuredPointsVTKFileTopView(VTKFileName, plot_title="Top View"):
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
            if len(values) >= 4:
                points_data.append([float(values[0]), float(values[1]), float(values[2])])
                intensity_data.append(float(values[3]))

        return np.array(points_data), np.array(intensity_data)

    points, intensities = read_custom_vtk(VTKFileName)

    resolution = 500
    x = np.linspace(points[:, 0].min(), points[:, 0].max(), resolution)
    y = np.linspace(points[:, 1].min(), points[:, 1].max(), resolution)
    z = np.zeros((resolution, resolution))

    for point, intensity in zip(points, intensities):
        i = np.argmin(np.abs(x - point[0]))
        j = np.argmin(np.abs(y - point[1]))
        z[j, i] = intensity

    image = pv.ImageData(dimensions=(resolution, resolution, 1))
    image.point_data['intensity'] = z.ravel()

    plotter = pv.Plotter(title=plot_title)
    plotter.add_mesh(image, scalars='intensity', cmap='viridis')
    plotter.view_xy()
    plotter.add_text(plot_title, font_size=14)
    plotter.show()

# === 3. Static 3D Point Cloud ===
def visualisePolyDataObject(polydataObject, title="Static 3D View"):
    plotter = pv.Plotter(title=title)
    plotter.add_mesh(polydataObject, render_points_as_spheres=True, point_size=5,
                     scalars='intensity', cmap='viridis')
    plotter.add_text(title, font_size=14)
    plotter.show()

# === 4. Interactive Dynamic 3D View ===
def visualisePolyDataObjectInDynamic3D(polydata, title="Interactive 3D View"):
    x = polydata.points[:, 0]
    y = polydata.points[:, 1]
    z = polydata.points[:, 2]
    intensity = polydata['intensity']

    df = pd.DataFrame({'X': x, 'Y': y, 'Z': z, 'Intensity': intensity})

    fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Intensity',
                        color_continuous_scale='Viridis', title=title)

    fig.add_scatter3d(x=[0], y=[0], z=[0], mode='markers',
                      marker=dict(size=5, color='red', symbol='x'),
                      name='Origin')

    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        width=1000, height=800,
        margin=dict(l=0, r=100, b=0, t=30),
        legend=dict(x=0.02, y=0.95, title_text='Legend')
    )
    # fig.show()
    
    fig.write_html("visualisePolyDataObjectInDynamic3D.html", auto_open=True)
    
def visualiseBlenderObjFile(blenderObjFileName):
    import pyvista as pv
    import numpy as np

    # Read the OBJ file and parse vertices and colors
    points = []
    colors = []
    
    with open(blenderObjFileName, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('v '):
                # Parse vertex coordinates
                _, x, y, z = line.split()
                points.append([float(x), float(y), float(z)])
                
                # Get next line which should be vertex color
                i += 1
                if i < len(lines) and lines[i].strip().startswith('vc'):
                    _, r, g, b = lines[i].strip().split()
                    colors.append([float(r), float(g), float(b)])
            i += 1

    # Convert to numpy arrays
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)

    # Create PyVista PolyData
    cloud = pv.PolyData(points)
    
    # Add colors to the point data
    cloud.point_data['colors'] = colors

    # Create plotter
    plotter = pv.Plotter()
    
    # Add points with colors
    plotter.add_mesh(
        cloud, 
        render_points_as_spheres=True, 
        point_size=5, 
        scalars='colors',
        rgb=True,  # Use RGB mode for direct color mapping
    )
    
    # Show the plot
    plotter.show()

# === File Paths ===
clear_day_vtk = r'../../../data/clear_day.vtk'
obj_path = "../../../outputs/pointcloud_colored2.obj"
snowy_day_vtk = r'../../../data/snowwy.vtk'

# === Visualize Clear Day ===
print("=== CLEAR DAY ===")
visualisingStructuredPointsVTKFileTopView(clear_day_vtk, plot_title="Top View - Clear Day")
clear_polydata = read_structured_points_as_polydata(clear_day_vtk)
visualisePolyDataObject(clear_polydata, title="Static 3D - Clear Day")
visualisePolyDataObjectInDynamic3D(clear_polydata, title="Interactive 3D - Clear Day")

# === Visualize Snowy Day ===
print("=== SNOWY DAY ===")
visualisingStructuredPointsVTKFileTopView(snowy_day_vtk, plot_title="Top View - Snowy Day")
snowy_polydata = read_structured_points_as_polydata(snowy_day_vtk)
visualisePolyDataObject(snowy_polydata, title="Static 3D - Snowy Day")
visualisePolyDataObjectInDynamic3D(snowy_polydata, title="Interactive 3D - Snowy Day")


# visualiseBlenderObjFile(obj_path)
