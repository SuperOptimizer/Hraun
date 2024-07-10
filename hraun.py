import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSplitter, QLabel, QLineEdit, QPushButton, QCheckBox
from PyQt6.QtCore import Qt
from skimage import measure
import matplotlib.pyplot as plt
from volman import VolMan
from skimage.exposure import equalize_adapthist
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat
from vtkmodules.util import numpy_support
import zarr

def rescale_array(arr):
    min_val = arr.min()
    max_val = arr.max()
    rescaled_arr = (arr - min_val) / (max_val - min_val)
    return rescaled_arr


class CustomQVTKRenderWindowInteractor(QVTKRenderWindowInteractor):
    def __init__(self, parent=None, **kw):
        super().__init__(parent, **kw)

    def CreateFrame(self):
        super().CreateFrame()
        self.GetRenderWindow().SetOffScreenRendering(True)  # Enable off-screen rendering


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setWindowTitle("3D Mesh Visualizer")

        format = QSurfaceFormat()
        format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
        format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        format.setVersion(4, 5)  # Specify OpenGL version
        QSurfaceFormat.setDefaultFormat(format)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.splitter)

        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout()
        self.control_panel.setLayout(self.control_layout)
        self.splitter.addWidget(self.control_panel)

        self.vtk_widget = CustomQVTKRenderWindowInteractor(self.splitter)
        self.splitter.addWidget(self.vtk_widget)

        self.splitter.setStretchFactor(1, 1)

        self.init_params_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        self.voxel_data = None
        self.mesh = None
        self.zarray = None

        self.volman = VolMan('D:/vesuvius.volman')

    def init_params_ui(self):
        self.label_vol = QLabel("Volume ID:")
        self.control_layout.addWidget(self.label_vol)
        self.vol_id = QLineEdit("Scroll3")
        self.control_layout.addWidget(self.vol_id)

        self.label_timestamp = QLabel("Timestamp:")
        self.control_layout.addWidget(self.label_timestamp)
        self.vol_timestamp = QLineEdit("20231117143551")
        self.control_layout.addWidget(self.vol_timestamp)

        self.label_dim = QLabel("Dimensions (z,y,x):")
        self.control_layout.addWidget(self.label_dim)
        self.dim_z = QLineEdit("2000")
        self.control_layout.addWidget(self.dim_z)
        self.dim_y = QLineEdit("2000")
        self.control_layout.addWidget(self.dim_y)
        self.dim_x = QLineEdit("2000")
        self.control_layout.addWidget(self.dim_x)

        self.label_chunk = QLabel("Chunk size (z,y,x):")
        self.control_layout.addWidget(self.label_chunk)
        self.chunk_z = QLineEdit("128")
        self.control_layout.addWidget(self.chunk_z)
        self.chunk_y = QLineEdit("128")
        self.control_layout.addWidget(self.chunk_y)
        self.chunk_x = QLineEdit("128")
        self.control_layout.addWidget(self.chunk_x)

        self.label_isolevel = QLabel("Isolevel:")
        self.control_layout.addWidget(self.label_isolevel)
        self.isolevel_input = QLineEdit("100")
        self.control_layout.addWidget(self.isolevel_input)

        self.label_downscale = QLabel("Downscaling factor:")
        self.control_layout.addWidget(self.label_downscale)
        self.downscale_input = QLineEdit("1")
        self.control_layout.addWidget(self.downscale_input)


        self.color_source_checkbox = QCheckBox("Use Zarr for Coloring")
        self.color_source_checkbox.setChecked(True)  # Default to using Zarr
        self.control_layout.addWidget(self.color_source_checkbox)

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_voxel_data)
        self.control_layout.addWidget(self.load_button)



    def load_voxel_data(self):
        vol_id = self.vol_id.text()
        vol_timestamp = self.vol_timestamp.text()
        dim_x, dim_y, dim_z = map(int, [self.dim_x.text(), self.dim_y.text(), self.dim_z.text()])
        chunk_x, chunk_y, chunk_z = map(int, [self.chunk_x.text(), self.chunk_y.text(), self.chunk_z.text()])

        print("Chunking")
        self.voxel_data = self.volman.chunk(vol_id, vol_timestamp, [dim_z, dim_y, dim_x],
                                            [chunk_z, chunk_y, chunk_x])
        #self.voxel_data = np.rot90(self.voxel_data, k=3, axes=(0, 2))
        isolevel = int(self.isolevel_input.text())
        downscale = int(self.downscale_input.text())

        print(f"Using isolevel: {isolevel}, Downscaling factor: {downscale}")
        mask = self.voxel_data > 0
        verts, faces, normals, values = measure.marching_cubes(self.voxel_data, level=isolevel, step_size=downscale,
                                                               mask=mask)
        print("Marching cubes completed successfully.")

        # Create initial vtkPolyData efficiently
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(verts, deep=True))

        cells = vtk.vtkCellArray()
        cells.SetCells(len(faces), numpy_support.numpy_to_vtkIdTypeArray(
            np.hstack((np.ones(len(faces), dtype=np.int64)[:, np.newaxis] * 3,
                       faces)).ravel(),
            deep=True
        ))

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)

        # Perform triangle stripping
        strip_per = vtk.vtkStripper()
        strip_per.SetInputData(poly_data)
        strip_per.Update()
        self.mesh = strip_per.GetOutput()

        print("VTK mesh created and triangle strips generated")

        # Print statistics
        print(f"Number of points: {self.mesh.GetNumberOfPoints()}")
        print(f"Number of cells: {self.mesh.GetNumberOfCells()}")

        # Coloring logic
        if self.color_source_checkbox.isChecked() and vol_id == "Scroll1":
            # Use Zarr for coloring
            self.zarray = zarr.open(r'D:\dl.ash2txt.org\community-uploads\ryan\3d_predictions_scroll1.zarr', 'r')

            # Get vertex coordinates
            n_points = self.mesh.GetNumberOfPoints()
            vertices = np.array([self.mesh.GetPoint(i) for i in range(n_points)])

            # Convert vertex coordinates to zarr indices
            # Adjust for the different coordinate systems and apply offsets
            zarr_indices = np.round(vertices).astype(int)

            # Map voxel space (z, y, x) to zarr space (y, x, z) and apply offsets
            zarr_y = zarr_indices[:, 1] + dim_y  # y in voxel space maps to y (dim 0) in zarr
            zarr_x = zarr_indices[:, 2] + dim_x  # x in voxel space maps to x (dim 1) in zarr
            zarr_z = zarr_indices[:, 0] + dim_z  # z in voxel space maps to z (dim 2) in zarr

            # Clip indices to valid range
            zarr_shape = np.array(self.zarray.shape)
            zarr_y = np.clip(zarr_y, 0, zarr_shape[0] - 1)
            zarr_x = np.clip(zarr_x, 0, zarr_shape[1] - 1)
            zarr_z = np.clip(zarr_z, 0, zarr_shape[2] - 1)

            # Get zarr values for vertices
            color_values = self.zarray[zarr_y, zarr_x, zarr_z]

            print(f"Zarr shape: {self.zarray.shape}")
            print(f"Chunk dimensions (z, y, x): {chunk_z}, {chunk_y}, {chunk_x}")
            print(f"Chunk start position (z, y, x): {dim_z}, {dim_y}, {dim_x}")
            print(f"Vertex coordinate ranges: X({vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}), "
                  f"Y({vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}), "
                  f"Z({vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f})")
            print(f"Zarr index ranges: X({zarr_x.min()}-{zarr_x.max()}), "
                  f"Y({zarr_y.min()}-{zarr_y.max()}), Z({zarr_z.min()}-{zarr_z.max()})")
        else:
            # Use marching cubes values for coloring
            color_values = values

        # Normalize color values
        normalized_values = (color_values - color_values.min()) / (color_values.max() - color_values.min())
        normalized_values = equalize_adapthist(normalized_values)

        # Add colors to the mesh
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")

        cmap = plt.get_cmap("viridis")
        rgb_values = (cmap(normalized_values)[:, :3] * 255).astype(np.uint8)
        colors.SetNumberOfTuples(len(rgb_values))
        colors.SetArray(rgb_values.ravel(), len(rgb_values) * 3, 1)

        self.mesh.GetPointData().SetScalars(colors)

        print('Colorizing completed')

        # Create mapper and actor
        mapper = vtk.vtkOpenGLPolyDataMapper()
        mapper.SetInputData(self.mesh)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set up material properties for simple ambient lighting
        actor.GetProperty().SetAmbient(1.0)  # Full ambient lighting
        actor.GetProperty().SetDiffuse(0.0)  # No diffuse lighting
        actor.GetProperty().SetSpecular(0.0)  # No specular lighting

        print('Adding mesh to renderer')
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)

        # Set background color
        self.renderer.SetBackground(0.1, 0.1, 0.1)  # Dark gray background

        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Elevation(20)
        self.renderer.GetActiveCamera().Azimuth(20)
        self.renderer.GetActiveCamera().Dolly(1.2)
        self.renderer.ResetCameraClippingRange()

        self.vtk_widget.GetRenderWindow().Render()

        print("Mesh rendering completed")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())