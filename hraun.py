import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSplitter, QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox
from PyQt6.QtCore import Qt
from skimage import measure
import matplotlib.pyplot as plt
from volman import VolMan
from skimage.exposure import equalize_adapthist
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt6.QtGui import QSurfaceFormat
from vtkmodules.util import numpy_support
import zarr

def rescale_array(arr):
    min_val = arr.min()
    max_val = arr.max()
    rescaled_arr = (arr - min_val) / (max_val - min_val)
    return rescaled_arr


class PickingInteractorStyle(vtk.vtkInteractorStyleRubberBandPick):
    def __init__(self, parent=None, renderer=None):
        super().__init__()
        self.parent = parent
        self.renderer = renderer
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.start_position = None
        self.end_position = None
        self.rubber_band_actor = None

    def left_button_press_event(self, obj, event):
        self.start_position = self.GetInteractor().GetEventPosition()
        self.end_position = None
        self.remove_rubber_band()
        self.OnLeftButtonDown()

    def on_mouse_move(self, obj, event):
        if self.start_position:
            self.end_position = self.GetInteractor().GetEventPosition()
            self.draw_rubber_band()
        self.OnMouseMove()

    def left_button_release_event(self, obj, event):
        self.end_position = self.GetInteractor().GetEventPosition()
        self.perform_vertex_pick()
        self.start_position = None
        self.end_position = None
        self.remove_rubber_band()
        self.OnLeftButtonUp()

    def draw_rubber_band(self):
        if not self.start_position or not self.end_position or not self.renderer:
            return

        if self.rubber_band_actor:
            self.renderer.RemoveActor(self.rubber_band_actor)

        points = vtk.vtkPoints()
        points.InsertNextPoint(self.start_position[0], self.start_position[1], 0)
        points.InsertNextPoint(self.end_position[0], self.start_position[1], 0)
        points.InsertNextPoint(self.end_position[0], self.end_position[1], 0)
        points.InsertNextPoint(self.start_position[0], self.end_position[1], 0)

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(5)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        lines.InsertCellPoint(2)
        lines.InsertCellPoint(3)
        lines.InsertCellPoint(0)

        rubber_band = vtk.vtkPolyData()
        rubber_band.SetPoints(points)
        rubber_band.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(rubber_band)

        self.rubber_band_actor = vtk.vtkActor2D()
        self.rubber_band_actor.SetMapper(mapper)
        self.rubber_band_actor.GetProperty().SetColor(1, 1, 1)
        self.rubber_band_actor.GetProperty().SetLineWidth(2)

        self.renderer.AddActor(self.rubber_band_actor)
        self.GetInteractor().GetRenderWindow().Render()

    def remove_rubber_band(self):
        if self.rubber_band_actor and self.renderer:
            self.renderer.RemoveActor(self.rubber_band_actor)
            self.rubber_band_actor = None
            self.GetInteractor().GetRenderWindow().Render()

    def perform_vertex_pick(self):
        if not self.start_position or not self.end_position or not self.renderer:
            return

        picker = vtk.vtkAreaPicker()
        picker.AreaPick(min(self.start_position[0], self.end_position[0]),
                        min(self.start_position[1], self.end_position[1]),
                        max(self.start_position[0], self.end_position[0]),
                        max(self.start_position[1], self.end_position[1]),
                        self.renderer)

        frustum = picker.GetFrustum()

        selected_points = vtk.vtkPoints()

        if self.parent.picking_mode == "Surface":
            self.perform_surface_vertex_pick(selected_points)
        else:
            self.perform_through_vertex_pick(selected_points, frustum)

        print(f"Number of selected vertices: {selected_points.GetNumberOfPoints()}")

        self.highlight_selected_points(selected_points)

    def perform_through_vertex_pick(self, selected_points, frustum):
        for i in range(self.parent.mesh.GetNumberOfPoints()):
            point = self.parent.mesh.GetPoint(i)
            if frustum.EvaluateFunction(point[0], point[1], point[2]) < 0:
                selected_points.InsertNextPoint(point)

    def perform_surface_vertex_pick(self, selected_points):
        renderer = self.GetCurrentRenderer()

        hw_selector = vtk.vtkHardwareSelector()
        hw_selector.SetRenderer(renderer)
        hw_selector.SetArea(int(min(self.start_position[0], self.end_position[0])),
                            int(min(self.start_position[1], self.end_position[1])),
                            int(max(self.start_position[0], self.end_position[0])),
                            int(max(self.start_position[1], self.end_position[1])))

        hw_selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS)

        selection = hw_selector.Select()

        selection_node = selection.GetNode(0)
        id_array = selection_node.GetSelectionList()

        if id_array:
            for i in range(id_array.GetNumberOfTuples()):
                point_id = id_array.GetValue(i)
                point = self.parent.mesh.GetPoint(point_id)
                selected_points.InsertNextPoint(point)

        print(f"Surface picking selected {selected_points.GetNumberOfPoints()} points")

    def highlight_selected_points(self, selected_points):
        point_polydata = vtk.vtkPolyData()
        point_polydata.SetPoints(selected_points)

        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(point_polydata)
        vertex_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex_filter.GetOutputPort())

        if self.parent.selected_vertex_actor:
            self.renderer.RemoveActor(self.parent.selected_vertex_actor)

        self.parent.selected_vertex_actor = vtk.vtkActor()
        self.parent.selected_vertex_actor.SetMapper(mapper)
        self.parent.selected_vertex_actor.GetProperty().SetColor(1, 0, 0)
        self.parent.selected_vertex_actor.GetProperty().SetPointSize(5)

        self.renderer.AddActor(self.parent.selected_vertex_actor)
        self.GetInteractor().GetRenderWindow().Render()


class CustomQVTKRenderWindowInteractor(QVTKRenderWindowInteractor):
    def __init__(self, parent=None, **kw):
        super().__init__(parent, **kw)

    def CreateFrame(self):
        super().CreateFrame()
        self.GetRenderWindow().SetOffScreenRendering(True)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setWindowTitle("3D Mesh Visualizer")

        format = QSurfaceFormat()
        format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
        format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        format.setVersion(4, 5)
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
        self.init_picking_mode_selector()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        self.camera_style = vtk.vtkInteractorStyleTrackballCamera()
        self.picking_style = PickingInteractorStyle(self, self.renderer)
        self.interactor.SetInteractorStyle(self.camera_style)

        self.voxel_data = None
        self.mesh = None
        self.zarray = None
        self.selected_vertex_actor = None

        self.volman = VolMan('D:/vesuvius.volman')

        self.picking_enabled = False

    def init_picking_mode_selector(self):
        self.picking_mode_selector = QComboBox()
        self.picking_mode_selector.addItems(["None", "Through", "Surface"])
        self.picking_mode_selector.currentTextChanged.connect(self.change_picking_mode)
        self.control_layout.addWidget(QLabel("Picking Mode:"))
        self.control_layout.addWidget(self.picking_mode_selector)

    def change_picking_mode(self, mode):
        self.picking_mode = mode
        if mode == "None":
            self.interactor.SetInteractorStyle(self.camera_style)
        else:
            self.interactor.SetInteractorStyle(self.picking_style)
        self.vtk_widget.GetRenderWindow().Render()

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
        self.color_source_checkbox.setChecked(True)
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

        isolevel = int(self.isolevel_input.text())
        downscale = int(self.downscale_input.text())

        print(f"Using isolevel: {isolevel}, Downscaling factor: {downscale}")
        mask = self.voxel_data > 0
        verts, faces, normals, values = measure.marching_cubes(self.voxel_data, level=isolevel, step_size=downscale,
                                                               mask=mask)
        print("Marching cubes completed successfully.")

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

        strip_per = vtk.vtkStripper()
        strip_per.SetInputData(poly_data)
        strip_per.Update()
        self.mesh = strip_per.GetOutput()

        print("VTK mesh created and triangle strips generated")

        print(f"Number of points: {self.mesh.GetNumberOfPoints()}")
        print(f"Number of cells: {self.mesh.GetNumberOfCells()}")

        if self.color_source_checkbox.isChecked() and vol_id == "Scroll1":
            self.zarray = zarr.open(r'D:\dl.ash2txt.org\community-uploads\ryan\3d_predictions_scroll1.zarr', 'r')

            n_points = self.mesh.GetNumberOfPoints()
            vertices = np.array([self.mesh.GetPoint(i) for i in range(n_points)])

            zarr_indices = np.round(vertices).astype(int)

            zarr_y = zarr_indices[:, 1] + dim_y
            zarr_x = zarr_indices[:, 2] + dim_x
            zarr_z = zarr_indices[:, 0] + dim_z

            zarr_shape = np.array(self.zarray.shape)
            zarr_y = np.clip(zarr_y, 0, zarr_shape[0] - 1)
            zarr_x = np.clip(zarr_x, 0, zarr_shape[1] - 1)
            zarr_z = np.clip(zarr_z, 0, zarr_shape[2] - 1)

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
            color_values = values

        normalized_values = (color_values - color_values.min()) / (color_values.max() - color_values.min())
        normalized_values = equalize_adapthist(normalized_values)

        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")

        cmap = plt.get_cmap("viridis")
        rgb_values = (cmap(normalized_values)[:, :3] * 255).astype(np.uint8)
        colors.SetNumberOfTuples(len(rgb_values))
        colors.SetArray(rgb_values.ravel(), len(rgb_values) * 3, 1)

        self.mesh.GetPointData().SetScalars(colors)

        print('Colorizing completed')

        mapper = vtk.vtkOpenGLPolyDataMapper()
        mapper.SetInputData(self.mesh)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        actor.GetProperty().SetSpecular(0.0)

        print('Adding mesh to renderer')
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)

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