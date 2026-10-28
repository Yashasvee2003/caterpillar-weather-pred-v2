# NOTE: just a sample run to visualise the vtk file in 3D to compare with some other visualisation
import numpy as np
import pyvista as pv
from conversions import VTKToPyVistaPolyDataObj
from visualisation import visualisePolyDataObject, visualisePolyDataObjectInDynamic3D

# NOTE: visualisng a sample .vtk for comparing with above
polydata_obj = VTKToPyVistaPolyDataObj("data/input.vtk")
visualisePolyDataObjectInDynamic3D(polydata_obj)