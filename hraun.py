import sys
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
VOLMAN_PATH = 'D:/vesuvius.volman'

the_index = {
    'Scroll1': {
        'volumes': {
            '20230205180739': {'depth': 14376, 'height': 7888, 'width': 8096, 'ext': 'tif', 'url': 'https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_masked/20230205180739/'},
            '20230206171837': {'depth': 10532, 'height': 7812, 'width': 8316, 'ext': 'tif', 'url': 'https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes/20230206171837/'}
        },
        'segments': {
            '20230503225234': {'depth': 65, 'height': 1962, 'width': 7920, 'ext': 'tif','url':'https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20240116164433/layers/'},
            '20230504093154': {},
            '20230504094316': {},
            '20230504125349': {},
            '20230504171956': {},
            '20230504223647': {},
            '20230504225948': {},
            '20230504231922': {},
            '20230505093556': {},
            '20230505113642': {},
            '20230505131816': {},
            '20230505135219': {},
            '20230505141722': {},
            '20230505164332': {},
            '20230505175240': {},
            '20230506133355': {},
            '20230507172452': {},
            '20230507175928': {},
            '20230508164013': {},
            '20230508220213': {},
            '20230509160956': {},
            '20230509182749': {},
            '20230510153006': {},
            '20230510153843': {},
            '20230510170242': {},
            '20230511085916': {},
            '20230511094040': {},
            '20230511201612': {},
            '20230511204029': {},
            '20230511211540': {},
            '20230511215040': {},
            '20230511224701': {},
            '20230512094635': {},
            '20230512105719': {},
            '20230512111225': {},
            '20230512112647': {},
            '20230512120728': {},
            '20230512123446': {},
            '20230512123540': {},
            '20230512170431': {},
            '20230513092954': {},
            '20230513095916': {},
            '20230513164153': {},
            '20230514173038': {},
            '20230514182829': {},
            '20230515162442': {},
            '20230516112444': {},
            '20230516114341': {},
            '20230516115453': {},
            '20230517021606': {},
            '20230517024455': {},
            '20230517025833': {},
            '20230517180019': {},
            '20230517204451': {},
            '20230517205601': {},
            '20230518012543': {},
            '20230518075340': {},
            '20230518104908': {},
            '20230518130337': {},
            '20230518135715': {},
            '20230518181521': {},
            '20230518191548': {},
            '20230518223227': {},
            '20230519031042': {},
            '20230519140147': {},
            '20230519195952': {},
            '20230519202000': {},
            '20230519212155': {},
            '20230519213404': {},
            '20230519215753': {},
            '20230520132429': {},
            '20230520175435': {},
            '20230520191415': {},
            '20230520192625': {},
            '20230521093501': {},
            '20230521104548': {},
            '20230521113334': {},
            '20230521114306': {},
            '20230521155616': {},
            '20230521182226': {},
            '20230521193032': {},
            '20230522055405': {},
            '20230522151031': {},
            '20230522152820': {},
            '20230522181603': {},
            '20230522210033': {},
            '20230522215721': {},
            '20230523002821': {},
            '20230523020515': {},
            '20230523034033': {},
            '20230523043449': {},
            '20230523182629': {},
            '20230523191325': {},
            '20230523233708': {},
            '20230524004853': {},
            '20230524005636': {},
            '20230524092434': {},
            '20230524163814': {},
            '20230524173051': {},
            '20230524200918': {},
            '20230525051821': {},
            '20230525115626': {},
            '20230525121901': {},
            '20230525190724': {},
            '20230525194033': {},
            '20230525200512': {},
            '20230525212209': {},
            '20230525234349': {},
            '20230526002441': {},
            '20230526015925': {},
            '20230526154635': {},
            '20230526164930': {},
            '20230526175622': {},
            '20230526183725': {},
            '20230526205020': {},
            '20230527020406': {},
            '20230528112855': {},
            '20230529203721': {},
            '20230530025328': {},
            '20230530164535': {},
            '20230530172803': {},
            '20230530212931': {},
            '20230531101257': {},
            '20230531121653': {},
            '20230531193658': {},
            '20230531211425': {},
            '20230601192025': {},
            '20230601193301': {},
            '20230601201143': {},
            '20230601204340': {},
            '20230602092221': {},
            '20230602213452': {},
            '20230603153221': {},
            '20230604111512': {},
            '20230604112252': {},
            '20230604161948': {},
            '20230605065957': {},
            '20230606105130': {},
            '20230606222610': {},
            '20230608150300': {},
            '20230608200454': {},
            '20230608222722': {},
            '20230609123853': {},
            '20230611014200': {},
            '20230611145109': {},
            '20230612195231': {},
            '20230613144727': {},
            '20230613204956': {},
            '20230619113941': {},
            '20230619163051': {},
            '20230620230617': {},
            '20230620230619': {},
            '20230621111336': {},
            '20230621122303': {},
            '20230621182552': {},
            '20230623123730': {},
            '20230623160629': {},
            '20230624144604': {},
            '20230624160816': {},
            '20230624190349': {},
            '20230625171244': {},
            '20230625194752': {},
            '20230626140105': {},
            '20230626151618': {},
            '20230627122904': {},
            '20230627170800': {},
            '20230627202005': {},
            '20230629215956': {},
            '20230701020044': {},
            '20230701115953': {},
            '20230702182347': {},
            '20230702185752_superseded': {},
            '20230702185753': {},
            '20230705142414': {},
            '20230706165709': {},
            '20230707113838': {},
            '20230709211458': {},
            '20230711201157': {},
            '20230711210222': {},
            '20230711222033': {},
            '20230712124010': {},
            '20230712210014': {},
            '20230713152725': {},
            '20230717092556': {},
            '20230719103041': {},
            '20230719214603': {},
            '20230720215300': {},
            '20230721122533': {},
            '20230721143008': {},
            '20230801193640': {},
            '20230806094533': {},
            '20230806132553': {},
            '20230808163057': {},
            '20230812170020': {},
            '20230819093803': {},
            '20230819210052': {},
            '20230820091651': {},
            '20230820174948': {},
            '20230820203112': {},
            '20230826135043': {},
            '20230826170124': {},
            '20230826211400': {},
            '20230827161846_superseded': {},
            '20230827161847': {},
            '20230828154913': {},
            '20230901184804': {},
            '20230901234823': {},
            '20230902141231': {},
            '20230903193206': {},
            '20230904020426': {},
            '20230904135535': {},
            '20230905134255': {},
            '20230909121925': {},
            '20230918021838': {},
            '20230918022237': {},
            '20230918023430': {},
            '20230918024753': {},
            '20230918140728': {},
            '20230918143910': {},
            '20230918145743': {},
            '20230919113918': {},
            '20230922174128': {},
            '20230925002745': {},
            '20230925090314': {},
            '20230926164631': {},
            '20230926164853': {},
            '20230929220920_superseded': {},
            '20230929220921_superseded': {},
            '20230929220923_superseded': {},
            '20230929220924_superseded': {},
            '20230929220925_superseded': {},
            '20230929220926': {},
            '20231001164029': {},
            '20231004222109': {},
            '20231005123333_superseded': {},
            '20231005123334_superseded': {},
            '20231005123335_superseded': {},
            '20231005123336': {},
            '20231007101615_superseded': {},
            '20231007101616_superseded': {},
            '20231007101617_superseded': {},
            '20231007101618_superseded': {},
            '20231007101619': {},
            '20231011111857': {},
            '20231011144857': {},
            '20231012085431': {},
            '20231012173610': {},
            '20231012184420': {},
            '20231012184421_superseded': {},
            '20231012184422_superseded': {},
            '20231012184423_superseded': {},
            '20231012184424': {},
            '20231016151000_superseded': {},
            '20231016151001_superseded': {},
            '20231016151002': {},
            '20231022170900_superseded': {},
            '20231022170901': {},
            '20231024093300': {},
            '20231031143850_superseded': {},
            '20231031143851_superseded': {},
            '20231031143852': {},
            '20231106155350_superseded': {},
            '20231106155351': {},
            '20231205141500': {},
            '20231206155550': {},
            '20231210121320_superseded': {},
            '20231210121321': {},
            '20231221180250_superseded': {},
            '20231221180251': {},
            '20231231235900_GP': {},
            '20240101215220': {},
            '20240102231959': {},
            '20240107134630': {},
            '20240109095720': {},
            '20240110113230': {},
            '20240116164433': {},
        }
    },
    'Scroll2': {
        'volumes':{
            '20230210143520': {'depth': 14428, 'height': 10112, 'width': 11984, 'ext': 'tif', 'url': 'https://dl.ash2txt.org/full-scrolls/Scroll2.volpkg/volumes_masked/20230210143520/'},
            '20230212125146': {'depth': 1610, 'height': 8480, 'width': 11136},
        },
        'segments': {
            '20230421192746': {},
            '20230421204550': {},
            '20230421215232': {},
            '20230421235552': {},
            '20230422011040': {},
            '20230422213203': {},
            '20230424181417': {},
            '20230424213608': {},
            '20230425163721': {},
            '20230425200944': {},
            '20230426114804': {},
            '20230426144221': {},
            '20230427171131': {},
            '20230501040514': {},
            '20230501042136': {},
            '20230503120034': {},
            '20230503213852': {},
            '20230504151750': {},
            '20230505142626': {},
            '20230505150348': {},
            '20230506111616': {},
            '20230506141535': {},
            '20230506142341': {},
            '20230506145035': {},
            '20230506145829': {},
            '20230506151750': {},
            '20230507064642': {},
            '20230507125513': {},
            '20230507175344': {},
            '20230508032834': {},
            '20230508080928': {},
            '20230508131616': {},
            '20230508171353': {},
            '20230508181757': {},
            '20230509144225': {},
            '20230509163359': {},
            '20230509173534': {},
            '20230511150730': {},
            '20230512192835': {},
            '20230512211850': {},
            '20230515151114': {},
            '20230516154633': {},
            '20230517000306': {},
            '20230517104414': {},
            '20230517151648': {},
            '20230517153958': {},
            '20230517164827': {},
            '20230517171727': {},
            '20230517193901': {},
            '20230517214715': {},
            '20230518210035': {},
            '20230519033308': {},
            '20230520080703': {},
            '20230520105602': {},
            '20230522172834': {},
            '20230522182853': {},
            '20230709155141': {},
            '20230801194757': {},
            '20240516205750': {},
        },
    },
    'Scroll3': {
        'volumes': {
            '20231027191953': {'depth': 22941, 'height': 9414, 'width': 9414, 'ext':'jpg', 'url': 'https://dl.ash2txt.org/community-uploads/james/PHerc0332/volumes_masked/20231027191953_jpg/'},
            '20231117143551': {'depth': 9778,  'height': 3550, 'width': 3400, 'ext':'tif', 'url': 'https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes/20231117143551/'},
            '20231201141544': {'depth': 22932, 'height': 9414, 'width': 9414, 'ext':'tif', 'url': 'https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes/20231201141544/'},
        },
        'segments': {
            '20231030220150': {},
            '20231031231220': {},
            '20240618142020': {},
            '20240702133100_thaumato_20231027191953': {},
            '20240702133100_thaumato_20231117143551': {},
            '20240712064330': {},
            '20240712071520': {},
            '20240712074250': {},
            '20240715203740': {},
        }
    },
    'Scroll4': {
        'volumes': {
            '20231107190228': {'depth': 26391, 'height': 7960, 'width': 8120, 'ext': 'jpg', 'url': 'https://dl.ash2txt.org/community-uploads/james-darby/PHerc1667/volumes_masked/20231107190228_jpg/'},
            '20231117161658': {'depth': 11174, 'height': 3340, 'width': 3440},
        },
        'segments': {
            '20231111135340': {},
            '20231122192640': {},
            '20231210132040': {},
            '20240304141530': {},
            '20240304141531': {},
            '20240304144030': {},
            '20240304144031': {},
            '20240304161940': {},
            '20240304161941': {},
        }
    },
}


def get_color_for_selection_type(selection_type):
    # Create a color map with 32 distinct colors
    cmap = plt.get_cmap('inferno')
    base_colors = cmap(np.linspace(0, 1, 16))


    all_colors = list(base_colors)
    return all_colors[selection_type][:3]  # Return only RGB values

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
        for volume, sources in the_index.items():
            for source, timestamps in sources.items():
                for timestamp in timestamps.keys():
                    os.makedirs(f'{cachedir}/{volume}/{source}/{timestamp}', exist_ok=True)
                    print(f"Created directory: {cachedir}/{volume}/{source}/{timestamp}")

    def load_cropped_tiff_slices(self, scroll, source, idnum, start_slice, end_slice, crop_start, crop_end, padlen):
        slices_data = []
        for slice_index in range(start_slice, end_slice):
            tiff_filename = f"{slice_index:0{padlen}d}.tif"
            tiff_path = os.path.join(f'{self.cachedir}/{scroll}/{source}/{idnum}', tiff_filename)
            tiff_data = tifffile.memmap(tiff_path)
            if tiff_data.dtype != np.uint8:
                raise ValueError("invalid input dtype from tiff files, must be uint8")
            slices_data.append(tiff_data[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]])
        return slices_data

    def download(self, scroll, source, id, start, end):
        """downloads 2d tiff slices from the vesuvius challenge and converts them into a
        zarr array"""

        depth = the_index[scroll][source][id]['depth']
        url = the_index[scroll][source][id]['url']
        ext = the_index[scroll][source][id]['ext']

        for x in range(start, end):
            src_filename = f"{x:0{len(str(depth))}d}.{ext}"

            dst_filename = f"{x:0{len(str(depth))}d}.tif"
            if os.path.exists(f'{self.cachedir}/{scroll}/{source}/{id}/{dst_filename}'):
                #print(f"skipped {url}{filename}")
                continue
            print(f"Downloading {url}{src_filename}")
            data = _download(url + src_filename)

            if id == '20231201141544':
                maskname = src_filename.replace('tif','png')
                mask = _download('https://dl.ash2txt.org/community-uploads/james/PHerc0332/volumes_masked/20231027191953_unapplied_masks/' + maskname)
                data[mask == 0] = 0

            print(f"Downloaded {url}{src_filename}")
            outpath = f'{self.cachedir}/{scroll}/{source}/{id}/{dst_filename}'
            print(f"wrote {url}{src_filename} to {outpath}")
            tifffile.imwrite(outpath, data)

    def get_mask(self, scroll, source, idnum, start, size):
        """ a mask is a segmentation mask, where we label each pixel* in the volume
            as belonging to one of 65536 unique segments.
            0 indicates that there is no papyrus
            65535 indicates papyrus of an indeterminate segment label
            1-65534 are unique segment labels """
        zoff, yoff, xoff = start
        zsize, ysize, xsize = size

        start = zoff
        end = zoff + zsize

        padlen, numtiffs = self._get_pad_and_len(scroll, source, idnum)
        if start > numtiffs or end > numtiffs:
            raise ValueError(
                f'start:{start} or end:{end} is greater than {numtiffs} tiffs in {scroll}/{idnum}')

        mask_path = f'{self.cachedir}/{scroll}/{source}/{idnum}_masks'

        mask_data = []
        for idx in range(start, end):
            filename = f"{idx:0{padlen}d}.tif"
            mask_file = os.path.join(mask_path, filename)

            full_mask_slice = tifffile.memmap(mask_file)
            mask_slice = full_mask_slice[yoff:yoff + ysize, xoff:xoff + xsize]
            mask_data.append(mask_slice)

        mask_data = np.stack(mask_data, axis=0)
        return mask_data

    def set_mask(self, scroll, source, idnum, start, mask):
        zoff, yoff, xoff = start
        zsize, ysize, xsize = mask.shape

        start = zoff
        end = zoff + zsize

        padlen, numtiffs = self._get_pad_and_len(scroll, idnum)
        if start > numtiffs or end > numtiffs:
            raise ValueError(
                f'start:{start} or end:{end} is greater than {numtiffs} tiffs in {scroll}/{idnum}')

        mask_path = f'{self.cachedir}/{scroll}/{source}/{idnum}_masks'

        for idx in range(start, end):
            filename = f"{idx:0{padlen}d}.tif"
            mask_file = os.path.join(mask_path, filename)

            full_mask_slice = tifffile.memmap(mask_file)
            full_mask_slice[yoff:yoff + ysize, xoff:xoff + xsize] = mask[idx - start]

    def chunk(self, scroll, source, idnum, start, size):
        ''' get a 3d chunk of data. Download the sources if necessary, otherwise pull them from the cache directory'''

        if scroll not in the_index.keys():
            raise ValueError(f'{scroll} is not a valid scroll')
        if source not in the_index[scroll]:
            raise ValueError(f'{source} is not a valid source for {scroll}')
        if idnum not in the_index[scroll][source]:
            raise ValueError(f'{idnum} is not a valid id for {source} in {scroll}')

        zoff, yoff, xoff = start
        zsize, ysize, xsize = size

        start = zoff
        end = start + zsize
        padlen, numtiffs = self._get_pad_and_len(scroll, source, idnum)
        if start > numtiffs or end > numtiffs:
            raise ValueError(f'start:{start} or end:{end} is greater than {numtiffs} tiffs in {scroll}/{source}/{idnum}')
        dl_path = f'{self.cachedir}/{scroll}/{source}/{idnum}'

        self.download(scroll, source, idnum, start, end)

        crop_start = (yoff, xoff)
        crop_end = (yoff + ysize, xoff + xsize)
        data = self.load_cropped_tiff_slices(scroll, source, idnum, start, end, crop_start, crop_end, padlen)
        data = np.stack(data, axis=0)

        mask_path = f'{self.cachedir}/{scroll}/{source}/{idnum}_masks'
        os.makedirs(mask_path, exist_ok=True)

        for idx in range(start, end):
            filename = f"{idx:0{padlen}d}.tif"
            mask_file = os.path.join(mask_path, filename)

            if not os.path.exists(mask_file):
                print(f"creating mask {mask_file}")
                mask_slice = np.zeros_like(tifffile.memmap(os.path.join(dl_path, filename)), dtype=np.uint8)
                tifffile.imwrite(mask_file, mask_slice)

        return data

    def _get_pad_and_len(self, scroll, source, idnum):
        depth = the_index[scroll][source][idnum]['depth']
        return len(str(depth)), depth

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
        self.volume_combo.currentTextChanged.connect(self.update_source_combo)
        self.control_layout.addWidget(self.volume_combo)

        # Source selection
        self.control_layout.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.currentTextChanged.connect(self.update_timestamp_combo)
        self.control_layout.addWidget(self.source_combo)
        self.vol_dim_label = QLabel("Volume Dimensions (z,y,x):")

        # Timestamp selection
        self.control_layout.addWidget(QLabel("Timestamp:"))
        self.timestamp_combo = QComboBox()
        self.control_layout.addWidget(self.timestamp_combo)
        self.timestamp_combo.currentTextChanged.connect(self.update_dimensions)

        # Initialize the combos
        self.update_source_combo(self.volume_combo.currentText())

        # Volume dimensions display
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
        self.color_source_checkbox.setChecked(False)
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

        self.control_layout.addWidget(QLabel("Expansion Steps:"))
        self.expansion_steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.expansion_steps_slider.setRange(1, 512)
        self.expansion_steps_slider.setValue(1)
        self.expansion_steps_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.expansion_steps_slider.setTickInterval(1)
        self.expansion_steps_slider.valueChanged.connect(self.update_expansion_steps_label)
        self.control_layout.addWidget(self.expansion_steps_slider)
        self.expansion_steps_label = QLabel("1")
        self.control_layout.addWidget(self.expansion_steps_label)

        # Replace combo box with slider for selection type
        self.control_layout.addWidget(QLabel("Selection Type:"))
        self.selection_type_slider = QSlider(Qt.Orientation.Horizontal)
        self.selection_type_slider.setRange(0, 16)
        self.selection_type_slider.setValue(0)
        self.selection_type_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.selection_type_slider.setTickInterval(1)
        self.selection_type_slider.valueChanged.connect(self.set_current_selection_type)
        self.control_layout.addWidget(self.selection_type_slider)
        self.selection_type_label = QLabel("0")
        self.control_layout.addWidget(self.selection_type_label)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        self.camera_style = vtk.vtkInteractorStyleTrackballCamera()
        self.picking_style = PickingInteractorStyle(self)
        self.interactor.SetInteractorStyle(self.camera_style)

        self.volman = VolMan(VOLMAN_PATH)

        self.voxel_data = None
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

        self.processed_points = set()
        self.processed_cells = set()
        self.frontier = set()

        self.mesh_version = 0


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

        self.processed_points = set()
        self.processed_cells = set()
        self.frontier = set()
        self.mesh_version += 1

        self.renderer.RemoveAllViewProps()
        self.renderer.RemoveAllLights()

    def update_source_combo(self, volume_id):
        self.source_combo.clear()
        if volume_id in the_index:
            self.source_combo.addItems(the_index[volume_id].keys())
        self.update_timestamp_combo()

    def update_timestamp_combo(self, source=None):
        self.timestamp_combo.clear()
        volume_id = self.volume_combo.currentText()
        if source is None:
            source = self.source_combo.currentText()
        if volume_id in the_index and source in the_index[volume_id]:
            self.timestamp_combo.addItems(the_index[volume_id][source].keys())
        self.update_dimensions()

    def update_dimensions(self):
        volume_id = self.volume_combo.currentText()
        source = self.source_combo.currentText()
        timestamp = self.timestamp_combo.currentText()
        if volume_id and source and timestamp:
            dimensions = the_index[volume_id][source][timestamp]
            if 'depth' in dimensions and 'height' in dimensions and 'width' in dimensions:
                self.vol_dim_label.setText(f"Volume Dimensions (z,y,x): {dimensions['depth']}, {dimensions['height']}, {dimensions['width']}")
            else:
                self.vol_dim_label.setText("Volume Dimensions: Not available")

    def update_expansion_steps_label(self, value):
        self.expansion_steps_label.setText(str(value))

    def set_current_selection_type(self, value):
        self.current_selection_type = value
        self.selection_type_label.setText(str(value))

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


    def add_coordinate_system(self, mesh_bounds):
        # Create bounding box with scales
        cube_axes = vtk.vtkCubeAxesActor()
        cube_axes.SetBounds(mesh_bounds)
        cube_axes.SetCamera(self.renderer.GetActiveCamera())
        cube_axes.SetXTitle("Z")
        cube_axes.SetYTitle("Y")
        cube_axes.SetZTitle("X")
        cube_axes.SetFlyMode(vtk.vtkCubeAxesActor.VTK_FLY_OUTER_EDGES)
        cube_axes.SetGridLineLocation(vtk.vtkCubeAxesActor.VTK_GRID_LINES_FURTHEST)
        cube_axes.XAxisMinorTickVisibilityOff()
        cube_axes.YAxisMinorTickVisibilityOff()
        cube_axes.ZAxisMinorTickVisibilityOff()
        self.renderer.AddActor(cube_axes)

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

    def validate_dimensions(self):
        volume_id = self.volume_combo.currentText()
        source = self.source_combo.currentText()
        timestamp = self.timestamp_combo.currentText()
        if not volume_id or not source or not timestamp:
            return False

        vol_dims = the_index[volume_id][source][timestamp]
        if 'depth' not in vol_dims or 'height' not in vol_dims or 'width' not in vol_dims:
            QMessageBox.warning(self, "Invalid Dimensions", "Dimension information is not available for the selected item.")
            return False

        offsets = [int(self.offset_z.text()), int(self.offset_y.text()), int(self.offset_x.text())]
        chunks = [int(self.chunk_z.text()), int(self.chunk_y.text()), int(self.chunk_x.text())]

        for i, (offset, chunk, vol_dim) in enumerate(zip(offsets, chunks, [vol_dims['depth'], vol_dims['height'], vol_dims['width']])):
            if offset + chunk > vol_dim:
                QMessageBox.warning(self, "Invalid Dimensions", 
                                    f"The selected chunk exceeds the volume boundaries in dimension {['z', 'y', 'x'][i]}.")
                return False

        return True

    def get_triangle_strip_neighbors(self, point_id):
        neighbors = set()

        # Get all cells (triangle strips) that use this point
        cell_ids = vtk.vtkIdList()
        self.mesh.GetPointCells(point_id, cell_ids)

        for i in range(cell_ids.GetNumberOfIds()):
            cell_id = cell_ids.GetId(i)
            strip = self.mesh.GetCell(cell_id)

            # Get the point ids of the strip
            strip_points = strip.GetPointIds()
            n_points = strip_points.GetNumberOfIds()

            # Find the position(s) of our point in the strip
            for j in range(n_points):
                if strip_points.GetId(j) == point_id:
                    # Add the previous point in the strip (if it exists)
                    if j > 0:
                        neighbors.add(strip_points.GetId(j - 1))

                    # Add the next point in the strip (if it exists)
                    if j < n_points - 1:
                        neighbors.add(strip_points.GetId(j + 1))

        return list(neighbors)


    def expand_selection_to_connected(self, selection_type=None):
        print()
        if not selection_type:
            #when called via the gui, selection_type is False. not none, not an it, False. ???
            selection_type = self.selection_type_slider.value()

        # calculate the frontier from looking through all selection points
        current_selection = self.labeled_points[selection_type]
        for i in range(current_selection.GetNumberOfPoints()):
            point = current_selection.GetPoint(i)
            point_id = self.mesh.FindPoint(point)
            neighbors = self.get_triangle_strip_neighbors(point_id)
            for n in neighbors:
                if n in self.processed_points:
                    continue
                self.frontier.add((n, selection_type))

        if len(self.frontier) == 0:
            print("could not find any points for expansion")
            # we didn't add any points, which means we probably have already processed all the points
            return

        num_added = 0
        next_frontier = set()
        frontier_points = set(f[0] for f in self.frontier)
        for point_id, selection_type_ in self.frontier:
            if selection_type_ != selection_type:
                next_frontier.add((point_id, selection_type_))
                #we aren't processing the given selection type for this point id so skip
                continue

            neighbors = self.get_triangle_strip_neighbors(point_id)
            for n in neighbors:
                if n not in self.processed_points and n not in frontier_points:
                    next_frontier.add((n, selection_type))
            point = self.mesh.GetPoint(point_id)
            self.labeled_points[selection_type].InsertNextPoint(point)
            self.processed_points.add(point_id)
            num_added += 1
        self.frontier = next_frontier
        print(f"added {num_added} point for selection label {selection_type}")
        self.update_visualization()
        QApplication.processEvents()

    def expand_all_selections(self):
        steps=self.expansion_steps_slider.value()

        for selection_type in range(16):
            current_selection = self.labeled_points[selection_type]
            for i in range(current_selection.GetNumberOfPoints()):
                point = current_selection.GetPoint(i)
                point_id = self.mesh.FindPoint(point)
                neighbors = self.get_triangle_strip_neighbors(point_id)
                for n in neighbors:
                    if n in self.processed_points:
                        continue
                    self.frontier.add((n, selection_type))

        if len(self.frontier) == 0:
            print("could not find any points for expansion")
            # we didn't add any points, which means we probably have already processed all the points
            return

        for step in range(steps):
            num_added = 0
            next_frontier = set()
            frontier_points = set(f[0] for f in self.frontier)
            for point_id, selection_type in self.frontier:
                neighbors = self.get_triangle_strip_neighbors(point_id)
                for n in neighbors:
                    if n not in self.processed_points and n not in frontier_points:
                        next_frontier.add((n,selection_type))
                point = self.mesh.GetPoint(point_id)
                self.labeled_points[selection_type].InsertNextPoint(point)
                self.processed_points.add(point_id)
                num_added += 1
                # print('qwer')
            self.frontier = next_frontier
            print(f"added {num_added} points to different selections")

            self.update_visualization()
            QApplication.processEvents()

    def load_voxel_data(self, voxel_data=None):
        if not self.validate_dimensions():
            return

        self.reset()

        volume_id = self.volume_combo.currentText()
        source = self.source_combo.currentText()
        timestamp = self.timestamp_combo.currentText()
        offset_dims = [int(self.offset_z.text()), int(self.offset_y.text()), int(self.offset_x.text())]
        chunk_dims = [int(self.chunk_z.text()), int(self.chunk_y.text()), int(self.chunk_x.text())]
        scaling_factors = [float(self.scale_z.text()), float(self.scale_y.text()), float(self.scale_x.text())]

        print(f"Loading data for {volume_id}, source {source}, timestamp {timestamp}")
        print(f"Offsets: {offset_dims}")
        print(f"Chunk size: {chunk_dims}")
        print(f"Scaling factors: {scaling_factors}")

        if voxel_data is False:
            print("getting voxel data")
            self.voxel_data = self.volman.chunk(volume_id, source, timestamp, offset_dims, chunk_dims)
            print("got voxel data")
        else:
            self.voxel_data = voxel_data

        isolevel = self.isolevel_slider.value()
        downscale = self.downscale_slider.value()
        print(f"Using isolevel: {isolevel}, Downscaling factor: {downscale}")
        print("masking below iso data")
        self.voxel_data = np.where(self.voxel_data < isolevel, 0, self.voxel_data)
        print("done masking below iso data")
        print("removing small connected noise")
        labels = label(self.voxel_data > isolevel, connectivity=1, return_num=False)
        component_sizes = np.bincount(labels.ravel())
        mask = np.isin(labels, np.where(component_sizes >= 32)[0])
        self.voxel_data *= mask
        print("done removing small connected noise")

        try:
            print("marching cubes")
            verts, faces, normals, values = measure.marching_cubes(self.voxel_data, level=isolevel,
                                                                   step_size=downscale, mask=mask, allow_degenerate=False)
            print("done marching cubes")
        except ValueError:
            QMessageBox.warning(self, "Invalid marching cubes data", "The given chunk did not yield any triangles")
            return
        print("creating vtk mesh")

        # Apply scaling factors to vertices
        verts[:, 0] *= scaling_factors[2]  # x
        verts[:, 1] *= scaling_factors[1]  # y
        verts[:, 2] *= scaling_factors[0]  # z

        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(verts, deep=False))

        cells = vtk.vtkCellArray()
        cells.SetCells(len(faces), numpy_support.numpy_to_vtkIdTypeArray(
            np.hstack((np.ones(len(faces), dtype=np.int64)[:, np.newaxis] * 3,
                       faces)).ravel(),
            deep=False
        ))

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)

        # Add normals to the polydata
        vtk_normals = numpy_support.numpy_to_vtk(normals, deep=False)
        vtk_normals.SetName("Normals")
        poly_data.GetPointData().SetNormals(vtk_normals)

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(poly_data)
        cleaner.Update()

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
        actor.GetProperty().SetAmbient(0.7)
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