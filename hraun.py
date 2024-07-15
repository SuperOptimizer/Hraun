import sys
from scipy.spatial import ConvexHull
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QMessageBox, QWidget, QSlider, QSplitter, QLabel, QLineEdit, QPushButton, QCheckBox, QHBoxLayout, QComboBox
from PyQt6.QtCore import Qt
from skimage import measure
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt6.QtGui import QSurfaceFormat
from vtkmodules.util import numpy_support
import zarr
import requests
import os
import io
import tifffile
import numpy as np
from PIL import Image
from skimage.measure import label


USER = os.environ.get('SCROLLPRIZE_USER')
PASS = os.environ.get('SCROLLPRIZE_PASS')


the_index = {
    'Scroll1': {
        '20230205180739': {'depth': 14376, 'height': 7888, 'width': 8096, 'ext': 'tif', 'url': 'https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_masked/20230205180739/'},
        '20230206171837': {'depth': 10532, 'height': 7812, 'width': 8316, 'ext': 'tif', 'url': 'https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes/20230206171837/'}
    },
    'Scroll2': {
        '20230210143520': {'depth': 14428, 'height': 10112, 'width': 11984, 'ext': 'tif', 'url': 'https://dl.ash2txt.org/full-scrolls/Scroll2.volpkg/volumes_masked/20230210143520/'},
        '20230212125146': {'depth': 1610, 'height': 8480, 'width': 11136},
    },
    'Scroll3': {
        '20231027191953': {'depth': 22941, 'height': 9414, 'width': 9414, 'ext':'jpg', 'url': 'https://dl.ash2txt.org/community-uploads/james/PHerc0332/volumes_masked/20231027191953_jpg/'},
        '20231117143551': {'depth': 9778,  'height': 3550, 'width': 3400, 'ext':'tif', 'url': 'https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes/20231117143551/'},
        '20231201141544': {'depth': 22932, 'height': 9414, 'width': 9414, 'ext':'tif', 'url': 'https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes/20231201141544/'},
    },
    'Scroll4': {
        '20231107190228': {'depth': 26391, 'height': 7960, 'width': 8120, 'ext': 'jpg', 'url': 'https://dl.ash2txt.org/community-uploads/james-darby/PHerc1667/volumes_masked/20231107190228_jpg/'},
        '20231117161658': {'depth': 11174, 'height': 3340, 'width': 3440},
    },
}


def get_color_for_selection_type(selection_type):
    colors = [
        (1, 0, 0),  # Red
        (0, 1, 0),  # Green
        (0, 0, 1),  # Blue
        (1, 1, 0),  # Yellow
        (1, 0, 1),  # Magenta
        (0, 1, 1),  # Cyan
        (1, 0.5, 0),  # Orange
        (0.5, 0, 1),  # Purple
        (0, 0.5, 0),  # Dark Green
        (0.5, 0.5, 0),  # Olive
        (0.5, 0, 0.5),  # Plum
        (0, 0.5, 0.5),  # Teal
        (1, 0.5, 0.5),  # Pink
        (0.5, 1, 0.5),  # Light Green
        (0.5, 0.5, 1),  # Light Blue
        (0.7, 0.7, 0.7),  # Light Gray
    ]
    return colors[selection_type]

def rescale_array(arr):
    min_val = arr.min()
    max_val = arr.max()
    rescaled_arr = (arr - min_val) / (max_val - min_val)
    return rescaled_arr

def _download(url):
    response = requests.get(url, auth=(USER, PASS))
    if response.status_code == 200:
        filedata = io.BytesIO(response.content)
        if url.endswith('.tif'):
            with tifffile.TiffFile(filedata) as tif:
                data = tif.asarray()
                if data.dtype == np.uint16:
                    return ((data >> 8) & 0xf0).astype(np.uint8)
                else:
                    raise
        elif url.endswith('.jpg'):
            data = np.array(Image.open(filedata))
            return data & 0xf0
        elif url.endswith('.png'):
            data = np.array(Image.open(filedata))
            return data
    else:
        raise Exception(f'Cannot download {url}')

class VolMan:
    def __init__(self, cachedir='D:/vesuvius.volman'):
        self.cachedir = cachedir
        for scroll, id in [
            ['Scroll1', '20230205180739'],
            ['Scroll1', '20230206171837'],
            ['Scroll2', '20230210143520'],
            ['Scroll2', '20230206082907'],
            ['Scroll2', '20230212125146'],
            ['Scroll3', '20231027191953'],
            ['Scroll3', '20231117143551'],
            ['Scroll3', '20231201141544'],
            ['Scroll4', '20231107190228'],
            ['Scroll4', '20231117161658'],]:
            os.makedirs(f'{cachedir}/{scroll}/{id}', exist_ok=True)

    def load_cropped_tiff_slices(self, scroll, idnum, start_slice, end_slice, crop_start, crop_end, padlen):
        slices_data = []
        for slice_index in range(start_slice, end_slice):
            tiff_filename = f"{slice_index:0{padlen}d}.tif"
            tiff_path = os.path.join(f'{self.cachedir}/{scroll}/{idnum}', tiff_filename)
            tiff_data = tifffile.memmap(tiff_path)
            if tiff_data.dtype != np.uint8:
                raise ValueError("invalid input dtype from tiff files, must be uint8")
            slices_data.append(tiff_data[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]])
        return slices_data

    def download(self, scroll, id, start, end):
        """downloads 2d tiff slices from the vesuvius challenge and converts them into a
        zarr array"""

        depth = the_index[scroll][id]['depth']
        url = the_index[scroll][id]['url']
        ext = the_index[scroll][id]['ext']

        for x in range(start, end):
            src_filename = f"{x:0{len(str(depth))}d}.{ext}"

            dst_filename = f"{x:0{len(str(depth))}d}.tif"
            if os.path.exists(f'{self.cachedir}/{scroll}/{id}/{dst_filename}'):
                #print(f"skipped {url}{filename}")
                continue
            print(f"Downloading {url}{src_filename}")
            data = _download(url + src_filename)

            if id == '20231201141544':
                maskname = src_filename.replace('tif','png')
                mask = _download('https://dl.ash2txt.org/community-uploads/james/PHerc0332/volumes_masked/20231027191953_unapplied_masks/' + maskname)
                data[mask == 0] = 0

            print(f"Downloaded {url}{src_filename}")
            tifffile.imwrite(f'{self.cachedir}/{scroll}/{id}/{dst_filename}', data)
            print(f"wrote {url}{src_filename}")

    def _get_pad_and_len(self, scroll,idnum):
        depth = the_index[scroll][idnum]['depth']
        return len(str(depth)), depth

    def get_mask(self, scroll, idnum, start, size):
        """ a mask is a segmentation mask, where we label each pixel* in the volume
            as belonging to one of 65536 unique segments.
            0 indicates that there is no papyrus
            65535 indicates papyrus of an indeterminate segment label
            1-65534 are unique segment labels """
        zoff, yoff, xoff = start
        zsize, ysize, xsize = size

        start = zoff
        end = zoff + zsize

        padlen, numtiffs = self._get_pad_and_len(scroll, idnum)
        if start > numtiffs or end > numtiffs:
            raise ValueError(
                f'start:{start} or end:{end} is greater than {numtiffs} tiffs in {scroll}/{idnum}')

        mask_path = f'{self.cachedir}/{scroll}/{idnum}_masks'

        mask_data = []
        for idx in range(start, end):
            filename = f"{idx:0{padlen}d}.tif"
            mask_file = os.path.join(mask_path, filename)

            full_mask_slice = tifffile.memmap(mask_file)
            mask_slice = full_mask_slice[yoff:yoff + ysize, xoff:xoff + xsize]
            mask_data.append(mask_slice)

        mask_data = np.stack(mask_data, axis=0)
        return mask_data

    def set_mask(self, scroll, idnum, start, mask):
        zoff, yoff, xoff = start
        zsize, ysize, xsize = mask.shape

        start = zoff
        end = zoff + zsize

        padlen, numtiffs = self._get_pad_and_len(scroll, idnum)
        if start > numtiffs or end > numtiffs:
            raise ValueError(
                f'start:{start} or end:{end} is greater than {numtiffs} tiffs in {scroll}/{idnum}')

        mask_path = f'{self.cachedir}/{scroll}/{idnum}_masks'

        for idx in range(start, end):
            filename = f"{idx:0{padlen}d}.tif"
            mask_file = os.path.join(mask_path, filename)

            full_mask_slice = tifffile.memmap(mask_file)
            full_mask_slice[yoff:yoff + ysize, xoff:xoff + xsize] = mask[idx - start]

    def chunk(self, scroll, idnum, start, size):
        ''' get a 3d chunk of data. Download the sources if necessary, otherwise pull them from the cache directory'''

        if scroll not in the_index.keys():
            raise ValueError(f'{scroll} is not a valid scroll')
        if idnum not in the_index[scroll]:
            raise ValueError(f'{idnum} is not a valid id for in {scroll}')

        zoff, yoff, xoff = start
        zsize, ysize, xsize = size

        start = zoff
        end = start + zsize
        padlen, numtiffs = self._get_pad_and_len(scroll, idnum)
        if start > numtiffs or end > numtiffs:
            raise ValueError(f'start:{start} or end:{end} is greater than {numtiffs} tiffs in {scroll}/{idnum}')
        dl_path = f'{self.cachedir}/{scroll}/{idnum}'

        self.download(scroll, idnum, start, end)

        crop_start = (yoff, xoff)
        crop_end = (yoff + ysize, xoff + xsize)
        data = self.load_cropped_tiff_slices(scroll, idnum, start, end, crop_start, crop_end, padlen)
        data = np.stack(data, axis=0)

        mask_path = f'{self.cachedir}/{scroll}/{idnum}_masks'
        os.makedirs(mask_path, exist_ok=True)

        for idx in range(start, end):
            filename = f"{idx:0{padlen}d}.tif"
            mask_file = os.path.join(mask_path, filename)

            if not os.path.exists(mask_file):
                print(f"creating mask {mask_file}")
                mask_slice = np.zeros_like(tifffile.memmap(os.path.join(dl_path, filename)), dtype=np.uint8)
                tifffile.imwrite(mask_file, mask_slice)

        return data

class PickingInteractorStyle(vtk.vtkInteractorStyleRubberBandPick):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.start_position = None
        self.end_position = None
        self.is_picking = False

    def left_button_press_event(self, obj, event):
        self.start_position = self.GetInteractor().GetEventPosition()
        self.end_position = None
        self.parent.remove_rubber_band()
        self.is_picking = True
        self.GetInteractor().GetRenderWindow().SetDesiredUpdateRate(30.0)
        self.InvokeEvent(vtk.vtkCommand.StartInteractionEvent)

    def on_mouse_move(self, obj, event):
        if self.is_picking:
            self.end_position = self.GetInteractor().GetEventPosition()
            self.parent.draw_rubber_band(self.start_position, self.end_position)
            self.InvokeEvent(vtk.vtkCommand.InteractionEvent)

    def left_button_release_event(self, obj, event):
        if self.is_picking:
            self.end_position = self.GetInteractor().GetEventPosition()
            selected_points = self.parent.perform_vertex_pick(self.start_position, self.end_position)
            self.parent.add_points(selected_points)
            self.start_position = None
            self.end_position = None
            self.parent.remove_rubber_band()
            self.is_picking = False
            self.GetInteractor().GetRenderWindow().SetDesiredUpdateRate(0.001)
            self.InvokeEvent(vtk.vtkCommand.EndInteractionEvent)

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
            # offset.setValidator(QIntValidator(0, 999999))
            offset_layout.addWidget(offset)
        self.control_layout.addLayout(offset_layout)

        # Chunk size input
        self.control_layout.addWidget(QLabel("Chunk size (z,y,x):"))
        chunk_layout = QHBoxLayout()
        self.chunk_z = QLineEdit("128")
        self.chunk_y = QLineEdit("128")
        self.chunk_x = QLineEdit("128")
        for chunk in [self.chunk_z, self.chunk_y, self.chunk_x]:
            # chunk.setValidator(QIntValidator(1, 999999))
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

        self.control_layout.addWidget(QLabel("Scaling Factors (z,y,x):"))
        scale_layout = QHBoxLayout()
        self.scale_z = QLineEdit("1")
        self.scale_y = QLineEdit("1")
        self.scale_x = QLineEdit("1")
        for scale in [self.scale_z, self.scale_y, self.scale_x]:
            scale_layout.addWidget(scale)
        self.control_layout.addLayout(scale_layout)

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_voxel_data)
        self.control_layout.addWidget(self.load_button)

        self.color_source_checkbox = QCheckBox("Use Zarr for Coloring")
        self.color_source_checkbox.setChecked(True)
        self.control_layout.addWidget(self.color_source_checkbox)

        self.update_timestamp_combo(self.volume_combo.currentText())
        self.picking_mode_selector = QComboBox()
        self.picking_mode_selector.addItems(["None", "Through", "Surface"])
        self.picking_mode_selector.currentTextChanged.connect(self.change_picking_mode)
        self.control_layout.addWidget(QLabel("Picking Mode:"))
        self.control_layout.addWidget(self.picking_mode_selector)

        self.clear_all_selections_button = QPushButton("Clear All Selections")
        self.clear_all_selections_button.clicked.connect(self.clear_all_selections)
        self.control_layout.addWidget(self.clear_all_selections_button)

        self.clear_last_selection_button = QPushButton("Clear Last Selection")
        self.clear_last_selection_button.clicked.connect(self.clear_last_selection)
        self.control_layout.addWidget(self.clear_last_selection_button)

        self.write_selection_button = QPushButton("Write Selection")
        self.write_selection_button.clicked.connect(self.write_selection)
        self.control_layout.addWidget(self.write_selection_button)

        self.delete_selection_button = QPushButton("Delete Selected Points")
        self.delete_selection_button.clicked.connect(self.delete_selected_points)
        self.control_layout.addWidget(self.delete_selection_button)

        self.expand_selection_button = QPushButton("Expand Current Selection")
        self.expand_selection_button.clicked.connect(self.expand_selection_to_connected)
        self.control_layout.addWidget(self.expand_selection_button)

        self.expand_all_selections_button = QPushButton("Expand All Selections")
        self.expand_all_selections_button.clicked.connect(self.expand_all_selections)
        self.control_layout.addWidget(self.expand_all_selections_button)

        self.selection_type_selector = QComboBox()
        self.selection_type_selector.addItems([f"Selection {i}" for i in range(16)])
        self.selection_type_selector.currentIndexChanged.connect(self.set_current_selection_type)
        self.control_layout.addWidget(QLabel("Selection Type:"))
        self.control_layout.addWidget(self.selection_type_selector)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        self.camera_style = vtk.vtkInteractorStyleTrackballCamera()
        self.picking_style = PickingInteractorStyle(self)
        self.interactor.SetInteractorStyle(self.camera_style)

        self.volman = VolMan('D:/vesuvius.volman')

        self.voxel_data = None
        self.label_data = None
        self.mesh = None
        self.zarray = None
        self.selected_vertex_actor = None
        self.delete_mask = None

        self.picking_enabled = False

        self.labeled_points = {i: vtk.vtkPoints() for i in range(16)}
        self.selected_vertex_actors = {i: None for i in range(16)}
        self.current_selection_type = 0
        self.rubber_band_actor = None
        self.picking_mode = "None"

    def reset(self):
        self.voxel_data = None
        self.label_data = None
        self.mesh = None
        self.zarray = None
        self.selected_vertex_actor = None
        self.delete_mask = None

        self.picking_enabled = False

        self.labeled_points = {i: vtk.vtkPoints() for i in range(16)}
        self.selected_vertex_actors = {i: None for i in range(16)}
        self.current_selection_type = 0
        self.rubber_band_actor = None
        self.picking_mode = "None"

        self.renderer.RemoveAllViewProps()
        self.renderer.RemoveAllLights()

    def add_points(self, new_points):
        num_new_points = new_points.GetNumberOfPoints()
        if num_new_points == 0:
            return
        current_points = self.labeled_points[self.current_selection_type]
        num_existing_points = current_points.GetNumberOfPoints()
        current_points.InsertPoints(num_existing_points, num_new_points, 0, new_points)
        self.update_visualization()

    def update_visualization(self):
        for selection_type, points in self.labeled_points.items():
            if points.GetNumberOfPoints() > 0:
                self.visualize_selection(selection_type, points)

    def clear_all_selections(self):
        pass

    def clear_last_selection(self):
        pass

    def visualize_selection(self, selection_type, points):
        point_polydata = vtk.vtkPolyData()
        point_polydata.SetPoints(points)

        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(point_polydata)
        vertex_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex_filter.GetOutputPort())

        if self.selected_vertex_actors[selection_type]:
            self.renderer.RemoveActor(self.selected_vertex_actors[selection_type])

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(get_color_for_selection_type(selection_type))
        actor.GetProperty().SetPointSize(5)

        self.selected_vertex_actors[selection_type] = actor
        self.renderer.AddActor(actor)
        self.interactor.GetRenderWindow().Render()

    def get_current_selection_type(self):
        return self.current_selection_type

    def set_current_selection_type(self, selection_type):
        self.current_selection_type = selection_type

    def add_coordinate_system(self, mesh_bounds):

        # Create bounding box with scales
        cube_axes = vtk.vtkCubeAxesActor()
        cube_axes.SetBounds(mesh_bounds)
        cube_axes.SetCamera(self.renderer.GetActiveCamera())
        cube_axes.SetXTitle("X")
        cube_axes.SetYTitle("Y")
        cube_axes.SetZTitle("Z")
        cube_axes.SetFlyMode(vtk.vtkCubeAxesActor.VTK_FLY_OUTER_EDGES)
        cube_axes.SetGridLineLocation(vtk.vtkCubeAxesActor.VTK_GRID_LINES_FURTHEST)
        cube_axes.XAxisMinorTickVisibilityOff()
        cube_axes.YAxisMinorTickVisibilityOff()
        cube_axes.ZAxisMinorTickVisibilityOff()
        self.renderer.AddActor(cube_axes)

        # Create checkered grid
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(mesh_bounds[0], mesh_bounds[2], mesh_bounds[4])
        plane.SetPoint1(mesh_bounds[1], mesh_bounds[2], mesh_bounds[4])
        plane.SetPoint2(mesh_bounds[0], mesh_bounds[3], mesh_bounds[4])
        plane.SetResolution(10, 10)

        # Create a custom checkered texture
        texture_size = 64
        texture = vtk.vtkImageData()
        texture.SetDimensions(texture_size, texture_size, 1)
        texture.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)

        for i in range(texture_size):
            for j in range(texture_size):
                if (i // (texture_size // 8) + j // (texture_size // 8)) % 2 == 0:
                    texture.SetScalarComponentFromFloat(i, j, 0, 0, 200)
                    texture.SetScalarComponentFromFloat(i, j, 0, 1, 200)
                    texture.SetScalarComponentFromFloat(i, j, 0, 2, 200)
                else:
                    texture.SetScalarComponentFromFloat(i, j, 0, 0, 100)
                    texture.SetScalarComponentFromFloat(i, j, 0, 1, 100)
                    texture.SetScalarComponentFromFloat(i, j, 0, 2, 100)

        texture_object = vtk.vtkTexture()
        texture_object.SetInputData(texture)
        texture_object.InterpolateOn()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetTexture(texture_object)
        self.renderer.AddActor(actor)

        # Adjust camera and add light
        self.renderer.GetActiveCamera().Elevation(20)
        self.renderer.GetActiveCamera().Azimuth(20)
        self.renderer.ResetCamera()

        if self.renderer.GetLights().GetNumberOfItems() == 0:
            light = vtk.vtkLight()
            light.SetFocalPoint((mesh_bounds[1] + mesh_bounds[0]) / 2,
                                (mesh_bounds[3] + mesh_bounds[2]) / 2,
                                (mesh_bounds[5] + mesh_bounds[4]) / 2)
            light.SetPosition(mesh_bounds[1], mesh_bounds[3], mesh_bounds[5])
            self.renderer.AddLight(light)


    def change_picking_mode(self, mode):
        self.picking_mode = mode
        if mode == "None":
            self.interactor.SetInteractorStyle(self.camera_style)
        else:
            self.interactor.SetInteractorStyle(self.picking_style)
        self.vtk_widget.GetRenderWindow().Render()


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

    def expand_all_selections(self):
        total_expanded = 0

        # First, collect all currently labeled points
        #for selection_type in range(16):
        #    current_selection = self.labeled_points[selection_type]
        #    for i in range(current_selection.GetNumberOfPoints()):
        #        point = current_selection.GetPoint(i)
        #        point = (int(round(point[0])), int(round(point[1])), int(round(point[2])))
        #        found_point = self.mesh.FindPoint(point)
        #        self.all_labeled_points.add(found_point)

        for selection_type in range(16):
            current_selection = self.labeled_points[selection_type]

            if current_selection.GetNumberOfPoints() == 0:
                continue  # Skip empty selections

            # Create a set to store the IDs of selected points
            selected_point_ids = set()
            for i in range(current_selection.GetNumberOfPoints()):
                point = current_selection.GetPoint(i)
                point_id = self.mesh.FindPoint(point)
                selected_point_ids.add(point_id)

            # Create a set to store the new point IDs
            new_point_ids = set()

            # Iterate through all cells in the mesh
            for cell_id in range(self.mesh.GetNumberOfCells()):
                cell = self.mesh.GetCell(cell_id)
                cell_point_ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]

                # If any point of the cell is in the current selection, add all points of the cell
                # that are not already labeled
                if any(point_id in selected_point_ids for point_id in cell_point_ids):
                    for point_id in cell_point_ids:
                        new_point_ids.add(point_id)

            # Create a new vtkPoints object for the expanded selection
            expanded_points = vtk.vtkPoints()
            for point_id in selected_point_ids.union(new_point_ids):
                expanded_points.InsertNextPoint(self.mesh.GetPoint(point_id))

            # Update the selection for the current selection type
            self.labeled_points[selection_type] = expanded_points

            total_expanded += len(new_point_ids)

        self.update_visualization()

        print(f"All selections expanded. Total new points added: {total_expanded}")

    def expand_selection_to_connected(self):
        pass

    def load_voxel_data(self, voxel_data=None):
        if not self.validate_dimensions():
            return

        self.reset()

        volume_id = self.volume_combo.currentText()
        timestamp = self.timestamp_combo.currentText()
        offset_dims = [int(self.offset_z.text()), int(self.offset_y.text()), int(self.offset_x.text())]
        chunk_dims = [int(self.chunk_z.text()), int(self.chunk_y.text()), int(self.chunk_x.text())]
        scaling_factors = [float(self.scale_z.text()), float(self.scale_y.text()), float(self.scale_x.text())]

        print(f"Loading data for {volume_id}, timestamp {timestamp}")
        print(f"Offsets: {offset_dims}")
        print(f"Chunk size: {chunk_dims}")
        print(f"Scaling factors: {scaling_factors}")

        if voxel_data is False:
            self.voxel_data = self.volman.chunk(volume_id, timestamp, offset_dims, chunk_dims)
        else:
            self.voxel_data = voxel_data

        isolevel = self.isolevel_slider.value()
        downscale = self.downscale_slider.value()
        print(f"Using isolevel: {isolevel}, Downscaling factor: {downscale}")
        mask = self.voxel_data > 0
        self.voxel_data[self.voxel_data < isolevel] = 0

        labels = label(self.voxel_data > isolevel)
        component_sizes = np.bincount(labels.flatten())
        mask = np.isin(labels, np.where(component_sizes >= 32)[0])
        self.voxel_data = self.voxel_data * mask

        self.label_data = np.where(self.voxel_data < isolevel, 0, 255).astype(np.uint8)

        try:
            verts, faces, normals, values = measure.marching_cubes(self.voxel_data, level=isolevel,
                                                                   step_size=downscale, mask=mask)
        except ValueError:
            QMessageBox.warning(self, "Invalid marching cubes data", "The given chunk did not yield any triangles")
            return
        print("Marching cubes completed successfully.")

        # Apply scaling factors to vertices
        verts[:, 0] *= scaling_factors[2]  # x
        verts[:, 1] *= scaling_factors[1]  # y
        verts[:, 2] *= scaling_factors[0]  # z

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
        actor.GetProperty().SetAmbient(0.3)
        actor.GetProperty().SetDiffuse(0.7)
        actor.GetProperty().SetSpecular(0.2)
        actor.GetProperty().SetSpecularPower(10)

        print('Adding mesh to renderer')
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)

        self.renderer.SetBackground(0.05, 0.05, 0.05)  # Dark gray background

        # Set up the camera for consistent orientation
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, 0, 1)  # Camera position
        camera.SetFocalPoint(0, 0, 0)  # Look at point
        camera.SetViewUp(0, 1, 0)  # Up direction

        self.renderer.ResetCamera()

        self.renderer.GetActiveCamera().Elevation(20)
        self.renderer.GetActiveCamera().Azimuth(20)
        self.renderer.ResetCamera()

        #light = vtk.vtkLight()
        #light.SetFocalPoint(0, 0, 0)
        #light.SetPosition(1, 1, 1)
        #self.renderer.AddLight(light)

        self.renderer.ResetCameraClippingRange()

        self.add_coordinate_system(self.mesh.GetBounds())

        self.vtk_widget.GetRenderWindow().Render()

        print("Mesh rendering completed")

    def convert_to_world_space(self, voxel_coord):
        z, y, x = voxel_coord
        bounds = self.mesh.GetBounds()
        scaling_factors = [float(self.scale_z.text()), float(self.scale_y.text()), float(self.scale_x.text())]
        world_x = bounds[0] + (x / (self.voxel_data.shape[2] - 1)) * (bounds[1] - bounds[0]) / scaling_factors[2]
        world_y = bounds[2] + (y / (self.voxel_data.shape[1] - 1)) * (bounds[3] - bounds[2]) / scaling_factors[1]
        world_z = bounds[4] + (z / (self.voxel_data.shape[0] - 1)) * (bounds[5] - bounds[4]) / scaling_factors[0]
        return (world_x, world_y, world_z)

    def convert_to_voxel_space(self, world_coord):
        z, y, x = world_coord
        bounds = self.mesh.GetBounds()
        scaling_factors = [float(self.scale_z.text()), float(self.scale_y.text()), float(self.scale_x.text())]
        voxel_x = int(round((x - bounds[0]) / (bounds[1] - bounds[0]) * (self.voxel_data.shape[2] - 1) * scaling_factors[2]))
        voxel_y = int(round((y - bounds[2]) / (bounds[3] - bounds[2]) * (self.voxel_data.shape[1] - 1) * scaling_factors[1]))
        voxel_z = int(round((z - bounds[4]) / (bounds[5] - bounds[4]) * (self.voxel_data.shape[0] - 1) * scaling_factors[0]))
        return (voxel_z, voxel_y, voxel_x)

    def draw_rubber_band(self, start_position, end_position):
        if not start_position or not end_position or not self.renderer:
            return

        if self.rubber_band_actor:
            self.renderer.RemoveActor(self.rubber_band_actor)

        points = vtk.vtkPoints()
        points.InsertNextPoint(start_position[0], start_position[1], 0)
        points.InsertNextPoint(end_position[0], start_position[1], 0)
        points.InsertNextPoint(end_position[0], end_position[1], 0)
        points.InsertNextPoint(start_position[0], end_position[1], 0)

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
        self.interactor.GetRenderWindow().Render()

    def remove_rubber_band(self):
        if self.rubber_band_actor and self.renderer:
            self.renderer.RemoveActor(self.rubber_band_actor)
            self.rubber_band_actor = None
            self.interactor.GetRenderWindow().Render()

    def perform_vertex_pick(self, start_position, end_position):
        if not start_position or not end_position or not self.renderer:
            return

        picker = vtk.vtkAreaPicker()
        picker.AreaPick(min(start_position[0], end_position[0]),
                        min(start_position[1], end_position[1]),
                        max(start_position[0], end_position[0]),
                        max(start_position[1], end_position[1]),
                        self.renderer)


        if self.picking_mode == "Surface":
            selected_points = self.perform_surface_vertex_pick(picker, start_position, end_position)
        else:
            selected_points = self.perform_through_vertex_pick(picker.GetFrustum())

        print(f"Number of selected vertices: {selected_points.GetNumberOfPoints()}")
        return selected_points


    def perform_surface_vertex_pick(self, picker, start_position, end_position):
        if not self.renderer:
            print("No renderer available for surface picking")
            return
        selected_points = vtk.vtkPoints()

        hw_selector = vtk.vtkHardwareSelector()
        hw_selector.SetRenderer(self.renderer)
        hw_selector.SetArea(int(min(start_position[0], end_position[0])),
                            int(min(start_position[1], end_position[1])),
                            int(max(start_position[0], end_position[0])),
                            int(max(start_position[1], end_position[1])))

        hw_selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS)

        selection = hw_selector.Select()

        if selection and selection.GetNumberOfNodes() > 0:
            selection_node = selection.GetNode(0)
            id_array = selection_node.GetSelectionList()

            if id_array:
                for i in range(id_array.GetNumberOfTuples()):
                    point_id = id_array.GetValue(i)
                    point = self.mesh.GetPoint(point_id)
                    selected_points.InsertNextPoint(point)

        print(f"Surface picking selected {selected_points.GetNumberOfPoints()} points")
        return selected_points

    def perform_through_vertex_pick(self, frustum):
        selected_points = vtk.vtkPoints()
        for i in range(self.mesh.GetNumberOfPoints()):
            point = self.mesh.GetPoint(i)
            if frustum.EvaluateFunction(point[0], point[1], point[2]) < 0:
                selected_points.InsertNextPoint(point)

        return selected_points

    def write_selection(self):
        pass

    def delete_selected_points(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())