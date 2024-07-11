import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QMessageBox, QWidget, QSlider, QSplitter, QLabel, QLineEdit, QPushButton, QCheckBox, QHBoxLayout, QComboBox
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

from volman import the_index

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
        self.is_picking = False

    def left_button_press_event(self, obj, event):
        self.start_position = self.GetInteractor().GetEventPosition()
        self.end_position = None
        self.remove_rubber_band()
        self.is_picking = True
        self.GetInteractor().GetRenderWindow().SetDesiredUpdateRate(30.0)
        self.InvokeEvent(vtk.vtkCommand.StartInteractionEvent)

    def on_mouse_move(self, obj, event):
        if self.is_picking:
            self.end_position = self.GetInteractor().GetEventPosition()
            self.draw_rubber_band()
            self.InvokeEvent(vtk.vtkCommand.InteractionEvent)

    def left_button_release_event(self, obj, event):
        if self.is_picking:
            self.end_position = self.GetInteractor().GetEventPosition()
            self.perform_vertex_pick()
            self.start_position = None
            self.end_position = None
            self.remove_rubber_band()
            self.is_picking = False
            self.GetInteractor().GetRenderWindow().SetDesiredUpdateRate(0.001)
            self.InvokeEvent(vtk.vtkCommand.EndInteractionEvent)

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

        selected_points = vtk.vtkPoints()

        if self.parent.picking_mode == "Surface":
            self.perform_surface_vertex_pick(selected_points, picker)
        else:
            self.perform_through_vertex_pick(selected_points, picker.GetFrustum())

        print(f"Number of selected vertices: {selected_points.GetNumberOfPoints()}")

        self.highlight_selected_points(selected_points)

    def perform_surface_vertex_pick(self, selected_points, picker):
        if not self.renderer:
            print("No renderer available for surface picking")
            return

        hw_selector = vtk.vtkHardwareSelector()
        hw_selector.SetRenderer(self.renderer)
        hw_selector.SetArea(int(min(self.start_position[0], self.end_position[0])),
                            int(min(self.start_position[1], self.end_position[1])),
                            int(max(self.start_position[0], self.end_position[0])),
                            int(max(self.start_position[1], self.end_position[1])))

        hw_selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS)

        selection = hw_selector.Select()

        if selection and selection.GetNumberOfNodes() > 0:
            selection_node = selection.GetNode(0)
            id_array = selection_node.GetSelectionList()

            if id_array:
                for i in range(id_array.GetNumberOfTuples()):
                    point_id = id_array.GetValue(i)
                    point = self.parent.mesh.GetPoint(point_id)
                    selected_points.InsertNextPoint(point)

        print(f"Surface picking selected {selected_points.GetNumberOfPoints()} points")

    def perform_through_vertex_pick(self, selected_points, frustum):
        for i in range(self.parent.mesh.GetNumberOfPoints()):
            point = self.parent.mesh.GetPoint(i)
            if frustum.EvaluateFunction(point[0], point[1], point[2]) < 0:
                selected_points.InsertNextPoint(point)

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
        # Volume ID selection
        self.control_layout.addWidget(QLabel("Volume ID:"))
        self.volume_combo = QComboBox()
        self.volume_combo.addItems(the_index.keys())
        self.volume_combo.currentTextChanged.connect(self.update_timestamp_combo)
        self.control_layout.addWidget(self.volume_combo)

        # Timestamp selection
        self.control_layout.addWidget(QLabel("Timestamp:"))
        self.timestamp_combo = QComboBox()
        self.control_layout.addWidget(self.timestamp_combo)
        self.timestamp_combo.currentTextChanged.connect(self.update_dimensions)

        # Volume dimensions display
        self.vol_dim_label = QLabel("Volume Dimensions (z,y,x):")
        self.control_layout.addWidget(self.vol_dim_label)

        # Offset dimensions input
        self.control_layout.addWidget(QLabel("Offset Dimensions (z,y,x):"))
        offset_layout = QHBoxLayout()
        self.offset_z = QLineEdit("0")
        self.offset_y = QLineEdit("0")
        self.offset_x = QLineEdit("0")
        for offset in [self.offset_z, self.offset_y, self.offset_x]:
            #offset.setValidator(QIntValidator(0, 999999))
            offset_layout.addWidget(offset)
        self.control_layout.addLayout(offset_layout)

        # Chunk size input
        self.control_layout.addWidget(QLabel("Chunk size (z,y,x):"))
        chunk_layout = QHBoxLayout()
        self.chunk_z = QLineEdit("128")
        self.chunk_y = QLineEdit("128")
        self.chunk_x = QLineEdit("128")
        for chunk in [self.chunk_z, self.chunk_y, self.chunk_x]:
            #chunk.setValidator(QIntValidator(1, 999999))
            chunk_layout.addWidget(chunk)
        self.control_layout.addLayout(chunk_layout)

        # Isolevel slider and value display
        iso_layout = QHBoxLayout()
        iso_layout.addWidget(QLabel("Isolevel:"))
        self.isolevel_slider = QSlider(Qt.Orientation.Horizontal)
        self.isolevel_slider.setRange(0, 255)
        self.isolevel_slider.setValue(100)
        self.isolevel_slider.valueChanged.connect(self.update_isolevel_label)
        iso_layout.addWidget(self.isolevel_slider)
        self.isolevel_label = QLabel("100")
        iso_layout.addWidget(self.isolevel_label)
        self.control_layout.addLayout(iso_layout)

        # Downscaling factor slider and value display
        downscale_layout = QHBoxLayout()
        downscale_layout.addWidget(QLabel("Downscaling factor:"))
        self.downscale_slider = QSlider(Qt.Orientation.Horizontal)
        self.downscale_slider.setRange(1, 8)
        self.downscale_slider.setValue(1)
        self.downscale_slider.valueChanged.connect(self.update_downscale_label)
        downscale_layout.addWidget(self.downscale_slider)
        self.downscale_label = QLabel("1")
        downscale_layout.addWidget(self.downscale_label)
        self.control_layout.addLayout(downscale_layout)

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_voxel_data)
        self.control_layout.addWidget(self.load_button)

        self.color_source_checkbox = QCheckBox("Use Zarr for Coloring")
        self.color_source_checkbox.setChecked(True)
        self.control_layout.addWidget(self.color_source_checkbox)


        # Initialize combos
        self.update_timestamp_combo(self.volume_combo.currentText())

    def update_isolevel_label(self, value):
        self.isolevel_label.setText(str(value))

    def update_downscale_label(self, value):
        self.downscale_label.setText(str(value))

    def update_timestamp_combo(self, volume_id):
        self.timestamp_combo.clear()
        self.timestamp_combo.addItems(the_index[volume_id].keys())
        self.update_dimensions()

    def update_dimensions(self):
        volume_id = self.volume_combo.currentText()
        timestamp = self.timestamp_combo.currentText()
        if volume_id and timestamp:
            dimensions = the_index[volume_id][timestamp]
            self.vol_dim_label.setText(f"Volume Dimensions (z,y,x): {dimensions['depth']}, {dimensions['height']}, {dimensions['width']}")

    def validate_dimensions(self):
        volume_id = self.volume_combo.currentText()
        timestamp = self.timestamp_combo.currentText()
        if not volume_id or not timestamp:
            return False

        vol_dims = the_index[volume_id][timestamp]
        offsets = [int(self.offset_z.text()), int(self.offset_y.text()), int(self.offset_x.text())]
        chunks = [int(self.chunk_z.text()), int(self.chunk_y.text()), int(self.chunk_x.text())]

        for i, (offset, chunk, vol_dim) in enumerate(zip(offsets, chunks, [vol_dims['depth'], vol_dims['height'], vol_dims['width']])):
            if offset + chunk > vol_dim:
                QMessageBox.warning(self, "Invalid Dimensions", 
                                    f"The selected chunk exceeds the volume boundaries in dimension {['z', 'y', 'x'][i]}.")
                return False

        return True

    def load_voxel_data(self):
        if not self.validate_dimensions():
            return

        volume_id = self.volume_combo.currentText()
        timestamp = self.timestamp_combo.currentText()
        offset_dims = [int(self.offset_z.text()), int(self.offset_y.text()), int(self.offset_x.text())]
        chunk_dims = [int(self.chunk_z.text()), int(self.chunk_y.text()), int(self.chunk_x.text())]
        isolevel = self.isolevel_slider.value()
        downscale = self.downscale_slider.value()

        # Here you would use these values to load and process your data
        print(f"Loading data for {volume_id}, timestamp {timestamp}")
        print(f"Offsets: {offset_dims}")
        print(f"Chunk size: {chunk_dims}")
        print(f"Isolevel: {isolevel}")
        print(f"Downscale factor: {downscale}")
        self.voxel_data = self.volman.chunk(volume_id, timestamp, offset_dims, chunk_dims)

        print(f"Using isolevel: {isolevel}, Downscaling factor: {downscale}")
        mask = self.voxel_data > 0
        try:
            verts, faces, normals, values = measure.marching_cubes(self.voxel_data, level=isolevel, step_size=downscale,
                                                               mask=mask)
        except ValueError:
            QMessageBox.warning(self, "Invalid marching cubes data", "The given chunk did not yield any triangles")
            return
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

        if self.color_source_checkbox.isChecked() and volume_id == "Scroll1":
            self.zarray = zarr.open(r'D:\dl.ash2txt.org\community-uploads\ryan\3d_predictions_scroll1.zarr', 'r')

            n_points = self.mesh.GetNumberOfPoints()
            vertices = np.array([self.mesh.GetPoint(i) for i in range(n_points)])

            zarr_indices = np.round(vertices).astype(int)

            zarr_y = zarr_indices[:, 1] + offset_dims[1]
            zarr_x = zarr_indices[:, 2] + offset_dims[2]
            zarr_z = zarr_indices[:, 0] + offset_dims[0]

            zarr_shape = np.array(self.zarray.shape)
            zarr_y = np.clip(zarr_y, 0, zarr_shape[0] - 1)
            zarr_x = np.clip(zarr_x, 0, zarr_shape[1] - 1)
            zarr_z = np.clip(zarr_z, 0, zarr_shape[2] - 1)

            color_values = self.zarray[zarr_y, zarr_x, zarr_z]

            print(f"Zarr shape: {self.zarray.shape}")
            print(f"Chunk dimensions (z, y, x): {chunk_dims[0]}, {chunk_dims[1]}, {chunk_dims[2]}")
            print(f"Chunk start position (z, y, x): {offset_dims[0]}, {offset_dims[1]}, {offset_dims[2]}")
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

        # Configure actor properties for lighting and shadows
        actor.GetProperty().SetAmbient(0.1)
        actor.GetProperty().SetDiffuse(0.7)
        actor.GetProperty().SetSpecular(0.2)
        actor.GetProperty().SetSpecularPower(10)

        # Enable shadow casting for the actor
        #actor.GetProperty().SetShadowIntensity(0.5)

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