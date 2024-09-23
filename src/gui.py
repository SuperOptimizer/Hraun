import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QSplitter, QLabel, \
  QLineEdit, QPushButton, QHBoxLayout, QComboBox, QSpinBox
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt6.QtGui import QSurfaceFormat, QIntValidator
from vtkmodules.util import numpy_support
import numpy as np
import skimage
from scipy import ndimage

import common, preprocessing, numbamath, segment
from src.common import timing_decorator



class CustomQVTKRenderWindowInteractor(QVTKRenderWindowInteractor):
  def __init__(self, parent=None, **kw):
    super().__init__(parent, **kw)

  def CreateFrame(self):
    super().CreateFrame()
    self.GetRenderWindow().SetOffScreenRendering(True)


class MainWindow(QMainWindow):
  def __init__(self, parent=None):
    super(MainWindow, self).__init__(parent)

    self.setWindowTitle("3D Volume Visualizer")

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

    self.renderer = vtk.vtkRenderer()
    self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
    self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

    self.camera_style = vtk.vtkInteractorStyleTrackballCamera()
    self.interactor.SetInteractorStyle(self.camera_style)

    self.lights = []

    self.voxel_data = None
    self.volume = None
    self.iso_value = 0
    self.opacity_cap = 255
    self.volume_mapper = None
    self.volume_property = None
    self.color_transfer_function = None
    self.opacity_transfer_function = None

    self.scroll_timestamps = {
      '1': ['20230205180739', '20230206171837'],
      '2': ['20230206082907', '20230210143520', '20230212125146'],
      '3': ['20231027191953', '20231117143551'],
      '4': ['20231107190228']
    }

    # self.picked_points = vtk.vtkPoints()
    # self.points_actor = None
    # self.setup_point_picking()

    self.setup_control_panel()

  @timing_decorator
  def setup_control_panel(self):
    self.control_layout.addWidget(QLabel("Scroll Number:"))
    self.scroll_combo = QComboBox()
    self.scroll_combo.addItems(['1', '2', '3', '4'])
    self.scroll_combo.currentTextChanged.connect(self.update_timestamp_options)
    self.control_layout.addWidget(self.scroll_combo)

    self.control_layout.addWidget(QLabel("Timestamp:"))
    self.timestamp_combo = QComboBox()
    self.control_layout.addWidget(self.timestamp_combo)

    self.control_layout.addWidget(QLabel("Coordinates (z,y,x):"))
    coord_layout = QHBoxLayout()
    self.coord_z = QLineEdit("9")
    self.coord_y = QLineEdit("9")
    self.coord_x = QLineEdit("9")
    for coord in [self.coord_z, self.coord_y, self.coord_x]:
      coord_layout.addWidget(coord)
    self.control_layout.addLayout(coord_layout)

    self.load_button = QPushButton("Load Data")
    self.load_button.clicked.connect(self.load_voxel_data)
    self.control_layout.addWidget(self.load_button)

    self.iso_slider = QSlider(Qt.Orientation.Horizontal)
    self.iso_slider.setRange(0, 255)
    self.iso_slider.setValue(self.iso_value)
    self.iso_slider.valueChanged.connect(self.update_iso_value)
    self.control_layout.addWidget(QLabel("Iso Value:"))
    self.control_layout.addWidget(self.iso_slider)

    segment_layout = QHBoxLayout()
    self.segment_input = QLineEdit()
    self.segment_input.setPlaceholderText("Enter segment ID")
    self.segment_input.setValidator(QIntValidator())  # Ensure only integers are entered
    segment_layout.addWidget(self.segment_input)

    self.display_segment_button = QPushButton("Display Segment")
    self.display_segment_button.clicked.connect(self.display_segment)
    segment_layout.addWidget(self.display_segment_button)

    self.control_layout.addLayout(segment_layout)

    self.update_timestamp_options(self.scroll_combo.currentText())

    self.control_layout.addWidget(QLabel("Remove Connected Components:"))

    # Small components removal layout
    small_cc_layout = QHBoxLayout()
    self.small_size_spinbox = QSpinBox()
    self.small_size_spinbox.setRange(1, 1000000)
    self.small_size_spinbox.setValue(10)
    small_cc_layout.addWidget(QLabel("Remove smaller:"))
    small_cc_layout.addWidget(self.small_size_spinbox)

    self.small_count_spinbox = QSpinBox()
    self.small_count_spinbox.setRange(1, 1000000)
    self.small_count_spinbox.setValue(1000000)
    small_cc_layout.addWidget(QLabel("Regions:"))
    small_cc_layout.addWidget(self.small_count_spinbox)

    self.remove_small_cc_button = QPushButton("Remove Small")
    self.remove_small_cc_button.clicked.connect(self.remove_smallest_components)
    small_cc_layout.addWidget(self.remove_small_cc_button)

    # Large components removal layout
    large_cc_layout = QHBoxLayout()
    self.large_size_spinbox = QSpinBox()
    self.large_size_spinbox.setRange(1, 10000000)
    self.large_size_spinbox.setValue(1000)
    large_cc_layout.addWidget(QLabel("Remove larger:"))
    large_cc_layout.addWidget(self.large_size_spinbox)

    self.large_count_spinbox = QSpinBox()
    self.large_count_spinbox.setRange(1, 1000000)
    self.large_count_spinbox.setValue(1000000)
    large_cc_layout.addWidget(QLabel("Regions:"))
    large_cc_layout.addWidget(self.large_count_spinbox)

    self.remove_large_cc_button = QPushButton("Remove Large")
    self.remove_large_cc_button.clicked.connect(self.remove_largest_components)
    large_cc_layout.addWidget(self.remove_large_cc_button)

    cc_layout = QVBoxLayout()
    cc_layout.addLayout(small_cc_layout)
    cc_layout.addLayout(large_cc_layout)
    self.control_layout.addLayout(cc_layout)

  @timing_decorator
  def display_segment(self, *args, **kwargs):
    segment_id = int(self.segment_input.text()) if self.segment_input.text() else None
    if segment_id is not None:
      print(f"Displaying segment: {segment_id}")  # Placeholder, replace with actual method call
      segment.segment(self.voxel_data)
      # Call your method here with the segment_id
      # For example: self.your_display_segment_method(segment_id)
    else:
      print("Please enter a valid segment ID")


  def remove_largest_components(self):
    self.voxel_data = segment.analyze_and_process_components(self.voxel_data, self.iso_value, self.large_count_spinbox.value(),True,True)
    self.render_volume(self.voxel_data)
    print(f"Removed the {self.large_count_spinbox.value()} largest components.")


  def remove_smallest_components(self):
    self.voxel_data = segment.analyze_and_process_components(self.voxel_data, self.iso_value, self.small_count_spinbox.value(),False,True)
    self.render_volume(self.voxel_data)
    print(f"Removed the {self.small_count_spinbox.value()} smallest components.")

  @timing_decorator
  def update_timestamp_options(self, scroll_number, *args, **kwargs):
    self.timestamp_combo.clear()
    self.timestamp_combo.addItems(self.scroll_timestamps[scroll_number])

  @timing_decorator
  def setup_vtk_pipeline(self, *args, **kwargs):
    self.volume_mapper = vtk.vtkSmartVolumeMapper()

    self.volume_property = vtk.vtkVolumeProperty()
    self.volume_property.SetInterpolationTypeToLinear()
    self.volume_property.ShadeOn()
    self.volume_property.SetAmbient(0.8)
    self.volume_property.SetDiffuse(0.3)
    self.volume_property.SetSpecular(0.1)
    self.volume_property.SetSpecularPower(10)

    self.color_transfer_function = self.create_viridis_color_function()
    self.volume_property.SetColor(self.color_transfer_function)

    self.opacity_transfer_function = vtk.vtkPiecewiseFunction()
    self.volume_property.SetScalarOpacity(self.opacity_transfer_function)

    self.volume = vtk.vtkVolume()
    self.volume.SetMapper(self.volume_mapper)
    self.volume.SetProperty(self.volume_property)

    self.renderer.AddVolume(self.volume)

    self.renderer.ResetCamera()

  @timing_decorator
  def load_voxel_data(self, *args, **kwargs):

    volume_id = int(self.scroll_combo.currentText())
    timestamp = int(self.timestamp_combo.currentText())
    offset_dims = [int(self.coord_z.text()), int(self.coord_y.text()), int(self.coord_x.text())]

    print(f"Loading data for {volume_id}, source timestamp {timestamp}")
    print(f"Offsets: {offset_dims}")
    data = common.get_chunk(volume_id, timestamp, offset_dims[0], offset_dims[1], offset_dims[2])
    print("loaded data")
    #data = numbamath.sumpool(data, (2, 2, 2), (2, 2, 2), (1, 1, 1))
    data = data[0:257,0:257,0:257]
    data = data.astype(np.float32)
    data = numbamath.rescale_array(data)
    data = skimage.restoration.denoise_tv_chambolle(data, weight=0.2)
    data = preprocessing.global_local_contrast_3d(data)
    data *= 255.
    self.voxel_data = data
    print("preprocessed data")

    if self.volume_mapper is None:
      self.setup_vtk_pipeline()
    print("pipeline has been setup")

    self.render_volume(data)

  @timing_decorator
  def render_volume(self, voxel_data, *args, **kwargs):
    image_data = self.numpy_to_vtk(voxel_data)
    self.volume_mapper.SetInputData(image_data)

    if not self.volume:
      self.volume = vtk.vtkVolume()
      self.volume.SetMapper(self.volume_mapper)
      self.volume.SetProperty(self.volume_property)
      self.renderer.AddVolume(self.volume)

    self.renderer.ResetCamera()
    self.vtk_widget.GetRenderWindow().Render()

  @timing_decorator
  def update_opacity_transfer_function(self, *args, **kwargs):
    if self.opacity_transfer_function is not None:
      self.opacity_transfer_function.RemoveAllPoints()

      self.opacity_transfer_function.AddPoint(0, 0.0)
      self.opacity_transfer_function.AddPoint(self.iso_value - 1, 0.0)

      if self.opacity_cap > self.iso_value:
        for i in range(self.iso_value, self.opacity_cap + 1):
          normalized_value = (i - self.iso_value) / (self.opacity_cap - self.iso_value)
          self.opacity_transfer_function.AddPoint(i, normalized_value)

      self.opacity_transfer_function.AddPoint(self.opacity_cap, 1.0)
      self.opacity_transfer_function.AddPoint(255, 1.0)

      if self.vtk_widget.GetRenderWindow():
        self.vtk_widget.GetRenderWindow().Render()

  @timing_decorator
  def update_iso_value(self, value, *args, **kwargs):
    self.iso_value = value
    self.opacity_cap = value
    self.update_opacity_transfer_function()

  @timing_decorator
  def numpy_to_vtk(self, numpy_array, *args, **kwargs):
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(numpy_array.shape)
    vtk_image.SetSpacing(1.0, 1.0, 1.0)
    vtk_image.SetOrigin(0.0, 0.0, 0.0)
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    if numpy_array.shape[0] == numpy_array.shape[1] and numpy_array.shape[1] == numpy_array.shape[2]:
      order = 'C'
    else:
      order = 'F'
    vtk_array = numpy_support.numpy_to_vtk(
      numpy_array.ravel(order=order),
      deep=False,
      array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_image.GetPointData().SetScalars(vtk_array)

    return vtk_image

  @timing_decorator
  def create_viridis_color_function(self, *args, **kwargs):
    color_function = vtk.vtkColorTransferFunction()
    cmap = plt.get_cmap("viridis")
    for i in range(256):
      color = cmap(i / 255.0)
      color_function.AddRGBPoint(i, color[0], color[1], color[2])
    return color_function

  @timing_decorator
  def setup_point_picking(self, *args, **kwargs):
    self.picker = vtk.vtkVolumePicker()

    self.points = vtk.vtkPoints()
    self.point_source = vtk.vtkPointSource()
    self.point_source.SetNumberOfPoints(0)
    self.point_source.SetRadius(0)

    self.points_mapper = vtk.vtkPolyDataMapper()
    self.points_mapper.SetInputConnection(self.point_source.GetOutputPort())

    self.points_actor = vtk.vtkActor()
    self.points_actor.SetMapper(self.points_mapper)
    self.points_actor.GetProperty().SetColor(1, 0, 0)
    self.points_actor.GetProperty().SetPointSize(10)
    self.points_actor.GetProperty().SetRenderPointsAsSpheres(True)
    self.renderer.AddActor(self.points_actor)

    self.interactor.SetPicker(self.picker)
    self.interactor.AddObserver("KeyPressEvent", self.keypress_callback)

  @timing_decorator
  def world_to_voxel(self, world_coords, *args, **kwargs):
    vtk_image = self.volume_mapper.GetInput()
    spacing = vtk_image.GetSpacing()
    origin = vtk_image.GetOrigin()

    voxel_coords = [
      int(round((world_coords[i] - origin[i]) / spacing[i]))
      for i in range(3)
    ]

    dimensions = vtk_image.GetDimensions()
    voxel_coords = [
      max(0, min(voxel_coords[i], dimensions[i] - 1))
      for i in range(3)
    ]

    return voxel_coords

  @timing_decorator
  def keypress_callback(self, obj, event, *args, **kwargs):
    key = obj.GetKeySym().lower()
    if key == 'p':
      click_pos = self.interactor.GetEventPosition()

      self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
      picked_position = self.picker.GetPickPosition()

      if self.picker.GetVolume() is not None:
        # Convert world coordinates to voxel coordinates
        voxel_coords = self.world_to_voxel(picked_position)
        point_id = self.points.InsertNextPoint(picked_position)
        self.point_source.SetNumberOfPoints(self.points.GetNumberOfPoints())
        self.point_source.SetCenter(picked_position)
        self.point_source.Update()

        self.points_mapper.Update()
        self.vtk_widget.GetRenderWindow().Render()
        print(f"Added point at world coordinates: {picked_position}")
        print(f"Voxel coordinates: {voxel_coords}")
      else:
        print("No volume picked at this position.")


@timing_decorator
def guimain():
  app = QApplication(sys.argv)
  window = MainWindow()
  window.show()
  window.interactor.Initialize()
  sys.exit(app.exec())


if __name__ == "__main__":
  guimain()
