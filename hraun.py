import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSplitter, QLabel, QLineEdit, QPushButton
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

        isolevel = int(self.isolevel_input.text())
        downscale = int(self.downscale_input.text())

        print(f"Using isolevel: {isolevel}, Downscaling factor: {downscale}")
        mask = self.voxel_data > 0
        verts, faces, normals, values = measure.marching_cubes(self.voxel_data, level=isolevel, step_size=downscale, mask=mask)
        print(f"got {len(verts)} verts, {len(faces)} aces, {len(normals)} normals, {len(values)} values")
        values = rescale_array(values)
        values = equalize_adapthist(values)
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

        # Convert and add normals to the mesh
        #normals_array = vtk.vtkSignedCharArray()
        #normals_array.SetNumberOfComponents(3)
        #normals_array.SetName("Normals")

        # Scale and convert normals to signed char
        #scaled_normals = np.clip(normals * 127, -127, 127).astype(np.int8)
        #normals_array.SetNumberOfTuples(len(scaled_normals))
        #normals_array.SetArray(scaled_normals.ravel(), len(scaled_normals) * 3, 1)

        #self.mesh.GetPointData().SetNormals(normals_array)

        if vol_id == "Scroll1":
            self.zarray = zarr.open(r'D:\dl.ash2txt.org\community-uploads\ryan\3d_predictions_scroll1.zarr','r')

        # Add colors to the mesh
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")

        cmap = plt.get_cmap("viridis")
        color_values = (cmap(values / values.max())[:, :3] * 255).astype(np.uint8)
        colors.SetNumberOfTuples(len(color_values))
        colors.SetArray(color_values.ravel(), len(color_values) * 3, 1)

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