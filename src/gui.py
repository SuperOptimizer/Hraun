import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QMessageBox, QWidget, QSlider, QSplitter, QLabel, \
    QLineEdit, QPushButton, QHBoxLayout, QComboBox, QSpinBox
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt6.QtGui import QSurfaceFormat
from vtkmodules.util import numpy_support
import numpy as np
import skimage


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

        self.volman = ZVol(create=False, overwrite=False, modify=True)
        self.voxel_data = None
        self.volume = None
        self.iso_value = 0
        self.opacity_cap = 255
        self.volume_mapper = None
        self.volume_property = None
        self.color_transfer_function = None
        self.opacity_transfer_function = None

        self.picked_points = vtk.vtkPoints()
        self.points_actor = None
        self.setup_point_picking()

        self.setup_control_panel()

    def setup_control_panel(self):

        # Add username input
        self.control_layout.addWidget(QLabel("ash2txt.org Username:"))
        self.username_input = QLineEdit()
        self.control_layout.addWidget(self.username_input)

        # Add password input
        self.control_layout.addWidget(QLabel("ash2txt.org Password:"))
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)  # Hide password characters
        self.control_layout.addWidget(self.password_input)

        self.control_layout.addWidget(QLabel("Volume ID:"))
        self.volume_combo = QComboBox()
        self.volume_combo.addItems(the_index.keys())
        self.volume_combo.currentTextChanged.connect(self.update_timestamp_combo)
        self.control_layout.addWidget(self.volume_combo)

        self.vol_dim_label = QLabel("Volume Dimensions (z,y,x):")
        self.control_layout.addWidget(self.vol_dim_label)

        self.control_layout.addWidget(QLabel("Timestamp:"))
        self.timestamp_combo = QComboBox()
        self.control_layout.addWidget(self.timestamp_combo)
        self.timestamp_combo.currentTextChanged.connect(self.update_dimensions)

        self.control_layout.addWidget(QLabel("Offset Dimensions (z,y,x):"))
        offset_layout = QHBoxLayout()
        self.offset_z = QLineEdit("0")
        self.offset_y = QLineEdit("0")
        self.offset_x = QLineEdit("0")
        for offset in [self.offset_z, self.offset_y, self.offset_x]:
            offset_layout.addWidget(offset)
        self.control_layout.addLayout(offset_layout)

        self.control_layout.addWidget(QLabel("Chunk size (z,y,x):"))
        chunk_layout = QHBoxLayout()
        self.chunk_z = QLineEdit("128")
        self.chunk_y = QLineEdit("128")
        self.chunk_x = QLineEdit("128")
        for chunk in [self.chunk_z, self.chunk_y, self.chunk_x]:
            chunk_layout.addWidget(chunk)
        self.control_layout.addLayout(chunk_layout)

        self.iso_slider = QSlider(Qt.Orientation.Horizontal)
        self.iso_slider.setRange(0, 255)
        self.iso_slider.setValue(self.iso_value)
        self.iso_slider.valueChanged.connect(self.update_iso_value)
        self.control_layout.addWidget(QLabel("Iso Value:"))
        self.control_layout.addWidget(self.iso_slider)

        self.noise_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_size_slider.setRange(0, 1024)
        self.noise_size_slider.setValue(self.iso_value)
        #self.noise_size_slider.valueChanged.connect(self.update_iso_value)
        self.control_layout.addWidget(QLabel("Noise Size Value:"))
        self.control_layout.addWidget(self.noise_size_slider)

        self.control_layout.addWidget(QLabel("Downscale Factor:"))
        self.downscale_factor = QSpinBox()
        self.downscale_factor.setRange(1, 8)  # Allow downscaling from 1x to 8x
        self.downscale_factor.setValue(1)
        self.control_layout.addWidget(self.downscale_factor)

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_voxel_data)
        self.control_layout.addWidget(self.load_button)

        self.update_timestamp_combo()

    def setup_vtk_pipeline(self):
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


    def update_timestamp_combo(self):
        self.timestamp_combo.clear()
        volume_id = self.volume_combo.currentText()
        if volume_id in the_index:
            self.timestamp_combo.addItems(the_index[volume_id].keys())
        self.update_dimensions()

    def update_dimensions(self):
        volume_id = self.volume_combo.currentText()
        timestamp = self.timestamp_combo.currentText()
        if volume_id and timestamp:
            dimensions = the_index[volume_id][timestamp]
            if 'depth' in dimensions and 'height' in dimensions and 'width' in dimensions:
                self.vol_dim_label.setText(
                    f"Volume Dimensions (z,y,x): {dimensions['depth']}, {dimensions['height']}, {dimensions['width']}")
            else:
                self.vol_dim_label.setText("Volume Dimensions: Not available")

    def validate_dimensions(self):
        volume_id = self.volume_combo.currentText()
        timestamp = self.timestamp_combo.currentText()
        if not volume_id or not timestamp:
            return False

        vol_dims = the_index[volume_id][timestamp]
        if 'depth' not in vol_dims or 'height' not in vol_dims or 'width' not in vol_dims:
            QMessageBox.warning(self, "Invalid Dimensions",
                                "Dimension information is not available for the selected item.")
            return False

        offsets = [int(self.offset_z.text()), int(self.offset_y.text()), int(self.offset_x.text())]
        chunks = [int(self.chunk_z.text()), int(self.chunk_y.text()), int(self.chunk_x.text())]

        for i, (offset, chunk, vol_dim) in enumerate(
                zip(offsets, chunks, [vol_dims['depth'], vol_dims['height'], vol_dims['width']])):
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

        print(f"Loading data for {volume_id}, source timestamp {timestamp}")
        print(f"Offsets: {offset_dims}")
        print(f"Chunk size: {chunk_dims}")

        #voxel_data = self.volman.chunk(volume_id, timestamp, offset_dims, chunk_dims)
        #voxel_data = skimage.util.apply_parallel(skimage.measure.block_reduce, voxel_data,
        #                                         extra_keywords={'block_size': self.downscale_factor.value(),
        #                                                         'func': np.max})
        print("loaded data")
        #self.voxel_data = preprocessing.global_local_contrast_3d(self.voxel_data)
        #self.voxel_data = (skimage.exposure.equalize_adapthist(self.voxel_data) * 255).astype(np.uint8)
        #self.voxel_data = (skimage.restoration.denoise_tv_chambolle(self.voxel_data,weight=0.1)*255).astype(np.uint8)
        print("preprocessed data")

        if volume_id == 'Scroll1':
            #self.zarray = zarr.open(r'D:\dl.ash2txt.org\community-uploads\ryan\3d_predictions_scroll1.zarr', 'r')
            self.zarray = zarr.open(r'C:\vesuvius_scroll1_downscaled.zarr', 'r')

            myiso = 120
        elif volume_id == 'Scroll2':
            self.zarray = zarr.open(r'D:\dl.ash2txt.org\community-uploads\ryan\3d_predictions_scroll2.zarr', 'r')
            myiso = 128
        elif volume_id == 'Scroll3':
            #self.zarray = zarr.open(r'D:\dl.ash2txt.org\community-uploads\ryan\3d_predictions_scroll3_invariant.zarr', 'r')
            self.zarray = zarr.open(r'C:\vesuvius_scroll3_downscaled.zarr', 'r')

            myiso = 100
        elif volume_id == 'Scroll4':
            #self.zarray = zarr.open(r'D:\dl.ash2txt.org\community-uploads\ryan\3d_predictions_scroll4.zarr', 'r')
            self.zarray = zarr.open(r'C:\vesuvius_scroll4_downscaled.zarr', 'r')
            myiso = 100
        print("got color data")
        #color_chunk = self.zarray[offset_dims[1]:offset_dims[1] + chunk_dims[1],
        #              offset_dims[2]:offset_dims[2] + chunk_dims[2],
        #              offset_dims[0]:offset_dims[0] + chunk_dims[0]]
        color_chunk = self.zarray[0:self.zarray.shape[0], 0:self.zarray.shape[1], 0:self.zarray.shape[2]]
        #color_chunk = skimage.util.apply_parallel(skimage.measure.block_reduce, color_chunk,
        #                                          extra_keywords={'block_size': self.downscale_factor.value(),
        #                                                          'func': np.max})


        print("transposing color data")
        color_chunk = np.transpose(color_chunk, (2,0,1))
        color_chunk = (skimage.util.apply_parallel(skimage.filters.gaussian, color_chunk, extra_keywords={'sigma':2})*255).astype(np.uint8)
        color_chunk = (skimage.util.apply_parallel(skimage.filters.unsharp_mask, color_chunk,extra_keywords={'radius':1.0,'amount':1.0})*255).astype(np.uint8)
        print("masking color data")

        print("removing small connected noise")
        #labels = label(color_chunk > self.iso_value, connectivity=1, return_num=False)
        #component_sizes = np.bincount(labels.ravel())
        #mask = np.isin(labels, np.where(component_sizes >= self.noise_size_slider.value())[0])
        #color_chunk *= mask


        #color_chunk[voxel_data < myiso] = 0
        print("reducing color data")

        if self.volume_mapper is None:
            self.setup_vtk_pipeline()
        print("pipeline has been setup")

        self.render_volume(color_chunk)

    def render_volume(self, voxel_data):
        image_data = self.numpy_to_vtk(voxel_data)
        self.volume_mapper.SetInputData(image_data)

        if not self.volume:
            self.volume = vtk.vtkVolume()
            self.volume.SetMapper(self.volume_mapper)
            self.volume.SetProperty(self.volume_property)
            self.renderer.AddVolume(self.volume)

        self.update_opacity_transfer_function()

        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def update_opacity_transfer_function(self):
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

    def update_iso_value(self, value):
        self.iso_value = value
        self.opacity_cap = value
        self.update_opacity_transfer_function()

    def update_opacity_cap(self, value):
        self.opacity_cap = value
        self.update_opacity_transfer_function()

    def numpy_to_vtk(self, numpy_array):
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

    def create_grayscale_color_function(self):
        color_function = vtk.vtkColorTransferFunction()
        for i in range(256):
            color_function.AddRGBPoint(i, i / 255.0, i / 255.0, i / 255.0)
        return color_function

    def create_viridis_color_function(self):
        color_function = vtk.vtkColorTransferFunction()
        cmap = plt.get_cmap("viridis")
        for i in range(256):
            color = cmap(i / 255.0)
            color_function.AddRGBPoint(i, color[0], color[1], color[2])
        return color_function

    def setup_point_picking(self):
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

    def world_to_voxel(self, world_coords):
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

    def keypress_callback(self, obj, event):
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

def guimain():
  app = QApplication(sys.argv)
  window = MainWindow()
  window.show()
  window.interactor.Initialize()
  sys.exit(app.exec())

if __name__ == "__main__":
    guimain()