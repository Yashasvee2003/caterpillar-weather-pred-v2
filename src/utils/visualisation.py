def visualisingStructuredPointsVTKFile(VTKFileName):
    # visualising directly with the input.vtk no need for conversion
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

    # Create a plotter
    plotter = pv.Plotter()

    # Add the point cloud to the plotter, coloring by intensity
    plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=5, scalars='intensity', cmap='viridis')

    # Show the plot
    plotter.show()



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
    

def visualisingStructuredPointsVTKFileTopView(VTKFileName):
    # visualising directly with the input.vtk no need for conversion# Vis on 2d, top view
    import pyvista as pv
    import numpy as np

    # (Use the same read_custom_vtk function from above)
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


    # Create a 2D image
    resolution = 500
    x = np.linspace(points[:, 0].min(), points[:, 0].max(), resolution)
    y = np.linspace(points[:, 1].min(), points[:, 1].max(), resolution)
    z = np.zeros((resolution, resolution))

    # Project points onto the 2D grid, using intensity for the value
    for point, intensity in zip(points, intensities):
        i = np.argmin(np.abs(x - point[0]))
        j = np.argmin(np.abs(y - point[1]))
        z[j, i] = intensity

    # Create a PyVista image
    image = pv.ImageData(dimensions=(resolution, resolution, 1))
    image.point_data['intensity'] = z.ravel()

    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(image, scalars='intensity', cmap='viridis')
    plotter.view_xy()
    plotter.show()


def visualisePolyDataObject(polydataObject):
    import pyvista as pv
    import numpy as np

    # Create a plotter
    plotter = pv.Plotter()

    # Add the point cloud to the plotter, coloring by intensity
    plotter.add_mesh(polydataObject, render_points_as_spheres=True, point_size=5, scalars='intensity', cmap='viridis')

    # Show the plot
    plotter.show()



def visualisePolyDataObjectInDynamic3D(polydata):

    import pyvista as pv
    import numpy as np
    import pandas as pd
    import plotly.express as px

    # Step 1: Load your PolyData object
    # polydata = VTKToPyVistaPolyDataObj("dustPointCloud.vtk")

    # Step 2: Extract x, y, z coordinates and intensity from the PolyData object
    x = polydata.points[:, 0]  # X coordinates
    y = polydata.points[:, 1]  # Y coordinates
    z = polydata.points[:, 2]  # Z coordinates
    intensity = polydata['intensity']  # Intensity values

    # Step 3: Create a DataFrame
    data = pd.DataFrame({'X': x, 'Y': y, 'Z': z, 'Intensity': intensity})

    # Step 4: Create a 3D scatter plot with fixed size for points, colored by intensity
    fig = px.scatter_3d(
        data, 
        x='X', 
        y='Y', 
        z='Z', 
        color='Intensity',  # Color based on intensity
        color_continuous_scale='Viridis', 
        title='3D Scatter Plot with Intensity',
    )

    # Step 5: Mark the origin (0, 0, 0)
    fig.add_scatter3d(
        x=[0], y=[0], z=[0], 
        mode='markers', 
        marker=dict(size=5, color='red', symbol='x'),  # Origin marker properties
        name='Origin'
    )

    # Step 6: Update layout for a larger figure size and smaller points
    fig.update_traces(marker=dict(size=3))  # Adjust the size of all points

    # Step 7: Adjust margins and layout for legends and color bars
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1000,  # Width of the plot
        height=800,  # Height of the plot
        margin=dict(l=0, r=100, b=0, t=30),  # Adjust the right margin for the color bar
        legend=dict(
            x=0.02,  # Move the legend away from the color bar
            y=0.95,  # Place the legend on the top-left
            title_text='Legend',  # Title for the legend
            itemsizing='constant',  # Ensure marker sizes in the legend are constant
        )
    )

    # Step 8: Show the plot
    fig.show()


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

