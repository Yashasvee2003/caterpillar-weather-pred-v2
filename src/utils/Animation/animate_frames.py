import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def read_structured_points_vtk(vtk_path):
    with open(vtk_path, 'r') as f:
        lines = f.readlines()

    points = []
    reading_points = False
    for line in lines:
        line = line.strip()
        if line.startswith("POINTS"):
            reading_points = True
            continue
        if reading_points:
            # Each line may contain multiple float values
            values = list(map(float, line.split()))
            for i in range(0, len(values), 4):  # Expecting x, y, z, intensity
                if i + 3 < len(values):
                    x, y, z, intensity = values[i:i+4]
                    points.append((x, y, z, intensity))
    df = pd.DataFrame(points, columns=["x", "y", "z", "intensity"])
    return df

def vtk_to_dataframe(vtk_path):
    return read_structured_points_vtk(vtk_path)

def visualize_vtk_sequence(vtk_folder):
    frames = []
    vtk_files = sorted([f for f in os.listdir(vtk_folder) if f.endswith('.vtk')])

    for i, filename in enumerate(vtk_files):
        vtk_path = os.path.join(vtk_folder, filename)
        try:
            df = vtk_to_dataframe(vtk_path)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        frame = go.Frame(
            data=[go.Scatter3d(
                x=df['x'], y=df['y'], z=df['z'],
                mode='markers',
                marker=dict(
                    size=2, color=df['intensity'],
                    colorscale='Viridis', opacity=0.8
                )
            )],
            name=f"Frame {i}"
        )
        frames.append(frame)

    fig = go.Figure(
        data=[go.Scatter3d(x=[], y=[], z=[], mode='markers',
                           marker=dict(size=2, color=[], colorscale='Viridis', opacity=0.8))],
        layout=go.Layout(
            scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')),
            title="VTK Frame Animation",
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play", method="animate",
                              args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True)])]
            )]
        ),
        frames=frames
    )

    fig.write_html("vtk_animation.html", auto_open=True)

# Run it
vtk_folder = r'../../../outputs/20250619_run_1'
visualize_vtk_sequence(vtk_folder)
