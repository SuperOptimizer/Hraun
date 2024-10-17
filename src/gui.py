import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QSplitter, QLabel, \
  QLineEdit, QPushButton, QHBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox
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
    self.segments = None
    self.segmented_data = None
    self.num_segments=0
    self.display_segments = False
    self.data_nbins = 16

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

    downscale_layout = QHBoxLayout()
    downscale_layout.addWidget(QLabel("Downscale Factor:"))
    self.downscale_spinbox = QSpinBox()
    self.downscale_spinbox.setRange(1, 8)
    self.downscale_spinbox.setValue(1)
    downscale_layout.addWidget(self.downscale_spinbox)
    self.control_layout.addLayout(downscale_layout)

    # Add compactness input
    compactness_layout = QHBoxLayout()
    compactness_layout.addWidget(QLabel("Superpixel Compactness:"))
    self.compactness_input = QDoubleSpinBox()
    self.compactness_input.setRange(0, 10000)
    self.compactness_input.setValue(10)
    self.compactness_input.setDecimals(2)
    compactness_layout.addWidget(self.compactness_input)
    self.control_layout.addLayout(compactness_layout)

    # Add density (d_seed) input
    density_layout = QHBoxLayout()
    density_layout.addWidget(QLabel("Superpixel Density:"))
    self.density_spinbox = QSpinBox()
    self.density_spinbox.setRange(2, 16)
    self.density_spinbox.setValue(8)  # Default value, adjust if needed
    density_layout.addWidget(self.density_spinbox)
    self.control_layout.addLayout(density_layout)

    self.load_button = QPushButton("Load Data")
    self.load_button.clicked.connect(self.load_voxel_data)
    self.control_layout.addWidget(self.load_button)

    self.iso_slider = QSlider(Qt.Orientation.Horizontal)
    self.iso_slider.setRange(0, 255)
    self.iso_slider.setValue(self.iso_value)
    self.iso_slider.valueChanged.connect(self.update_iso_value)
    self.control_layout.addWidget(QLabel("Iso Value:"))
    self.control_layout.addWidget(self.iso_slider)

    self.segment_slider = QSlider(Qt.Orientation.Horizontal)
    self.segment_slider.setRange(0,  self.num_segments)
    self.segment_slider.setValue(0)
    self.segment_slider.valueChanged.connect(self.update_segment_display)
    self.control_layout.addWidget(QLabel("Segment Selection:"))
    self.control_layout.addWidget(self.segment_slider)

    self.do_segmentation_button = QPushButton("Do Segmentation")
    self.do_segmentation_button.clicked.connect(self.do_segmentation)
    self.control_layout.addWidget(self.do_segmentation_button)

    self.segment_checkbox = QCheckBox("Display Segments")
    self.segment_checkbox.setChecked(True)
    self.segment_checkbox.stateChanged.connect(self.toggle_segment_display)
    self.control_layout.addWidget(self.segment_checkbox)

    self.update_timestamp_options(self.scroll_combo.currentText())

    self.control_layout.addWidget(QLabel("Remove Connected Components:"))

    # Small components removal layout
    small_cc_layout = QHBoxLayout()
    self.small_size_spinbox = QSpinBox()
    self.small_size_spinbox.setRange(1, 10000)
    self.small_size_spinbox.setValue(10)
    small_cc_layout.addWidget(QLabel("Remove smaller:"))
    small_cc_layout.addWidget(self.small_size_spinbox)

    self.small_count_spinbox = QSpinBox()
    self.small_count_spinbox.setRange(1, 10000)
    self.small_count_spinbox.setValue(10)
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
  def do_segmentation(self, *args, **kwargs):
    compactness = float(self.compactness_input.text())
    density = self.density_spinbox.value()
    data = self.voxel_data
    mask = data < self.iso_value
    data[mask] = 0.0
    #data = numbamath.rescale_array(data)

    superpixels, labels, segs, sp_to_seg = segment.segment(data, compactness, density, self.iso_value, self.data_nbins)
    self.segmented_data = segment.label_data(labels,superpixels,sp_to_seg,len(segs))
    self.num_segments = len(segs)
    self.segment_slider.setRange(0,  self.num_segments)



    if self.volume_mapper is None:
      self.setup_vtk_pipeline()

    self.update_display()

  def toggle_segment_display(self, state):
    self.display_segments = state == Qt.CheckState.Checked.value
    self.update_display()

  def update_display(self):
    if self.display_segments:
      self.update_segment_display(self.segment_slider.value())
    else:
      self.update_voxel_display()

  def update_segment_display(self, segment_id):
    if self.segmented_data is None:
      return

    display_data = np.zeros_like(self.segmented_data)
    display_data[self.segmented_data == segment_id] = 255

    self.update_segment_color_transfer_function(segment_id)
    self.render_volume(display_data)

  def update_voxel_display(self):
    self.update_voxel_color_transfer_function()
    self.render_volume(self.voxel_data)

  def update_segment_color_transfer_function(self, segment_id):
    self.color_transfer_function.RemoveAllPoints()
    cmap = plt.get_cmap("viridis")
    color = cmap(segment_id / 255)
    self.color_transfer_function.AddRGBPoint(0, 0, 0, 0)  # Transparent for non-selected segments
    self.color_transfer_function.AddRGBPoint(1, color[0], color[1], color[2])  # Colored for selected segment
    self.volume_property.SetColor(self.color_transfer_function)

  def update_voxel_color_transfer_function(self):
    self.color_transfer_function.RemoveAllPoints()
    cmap = plt.get_cmap("viridis")
    for i in range(256):
      color = cmap(i / 255.0)
      self.color_transfer_function.AddRGBPoint(i, color[0], color[1], color[2])
    self.volume_property.SetColor(self.color_transfer_function)

  def create_viridis_color_function(self):
    color_function = vtk.vtkColorTransferFunction()
    cmap = plt.get_cmap("viridis")
    for i in range(256):
      color = cmap(i / 255.0)
      color_function.AddRGBPoint(i, color[0], color[1], color[2])
    return color_function

  def update_opacity_transfer_function(self):
    if self.opacity_transfer_function is not None:
      self.opacity_transfer_function.RemoveAllPoints()
      if self.display_segments:
        self.opacity_transfer_function.AddPoint(0, 0.0)
        self.opacity_transfer_function.AddPoint(1, 1.0)
      else:
        self.opacity_transfer_function.AddPoint(0, 0.0)
        self.opacity_transfer_function.AddPoint(self.iso_value - 1, 0.0)
        self.opacity_transfer_function.AddPoint(self.iso_value, 1.0)
        self.opacity_transfer_function.AddPoint(255, 1.0)
      if self.vtk_widget.GetRenderWindow():
        self.vtk_widget.GetRenderWindow().Render()

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
    self.volume_property.SetAmbient(0.6)
    self.volume_property.SetDiffuse(0.6)
    self.volume_property.SetSpecular(0.6)
    self.volume_property.SetSpecularPower(100)

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
    downscale_factor = self.downscale_spinbox.value()

    print(f"Loading data for {volume_id}, source timestamp {timestamp}")
    print(f"Offsets: {offset_dims}")
    data = common.get_chunk(volume_id, timestamp, offset_dims[0], offset_dims[1], offset_dims[2])
    print("loaded data")

    data = data[0:256, 0:256, 0:256]
    if downscale_factor > 1:
      data = numbamath.sumpool(data,
                               (downscale_factor, downscale_factor, downscale_factor),
                               (downscale_factor, downscale_factor, downscale_factor), (1, 1, 1))
    #data = skimage.restoration.denoise_tv_chambolle(data, weight=.25)
    data = preprocessing.global_local_contrast_3d(data)
    data = skimage.filters.gaussian(data, sigma=2)
    data = skimage.exposure.equalize_adapthist(data,nbins=self.data_nbins)
    data = data.astype(np.float32)
    data = numbamath.rescale_array(data)

    data *= 255.
    self.voxel_data = data
    print("preprocessed data")

    if self.volume_mapper is None:
      self.setup_vtk_pipeline()

    self.update_display()

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

  #@timing_decorator
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

  #@timing_decorator
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
