import os
import re
import requests
import tifffile
import numpy as np
import io
import concurrent.futures
from PIL import Image

the_index = {
    'PHerc0332': {
        'segments': {
            '20231030220150': {'depth': 157, 'height': 9417, 'width': 10745},
            '20231031231220': {'depth': 157, 'height': 9523, 'width': 3599},
        },
        'volumes': {
            '20231027191953': {'depth': 22941, 'height': 9414, 'width': 9414},
            '20231117143551': {'depth': 9778,  'height': 3550, 'width': 3400},
            '20231201141544': {'depth': 22932, 'height': 9414, 'width': 9414},
        },
    },
    'PHerc1667': {
        'segments': {
            '20231111135340': {},
            '20231122192640': {},
            '20231210132040': {},
            '20240304141530': {},
            '20240304144030': {},
        },
        'volumes': {
            '20231107190228': {'depth': 26391, 'height': 7960, 'width': 8120},
            '20231117161658': {'depth': 11174, 'height': 3340, 'width': 3440},
        },
    },
    'Scroll1':{
        'segments': {
            '20230503225234': {},
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
            '20230929220926': {},
            '20231001164029': {},
            '20231004222109': {},
            '20231005123336': {},
            '20231007101619': {},
            '20231011111857': {},
            '20231011144857': {},
            '20231012085431': {},
            '20231012173610': {},
            '20231012184420': {},
            '20231012184424': {},
            '20231016151002': {},
            '20231022170901': {},
            '20231024093300': {},
            '20231031143852': {},
            '20231106155351': {},
            '20231205141500': {},
            '20231206155550': {},
            '20231210121321': {},
            '20231221180251': {},
        },
        'volumes': {
            '20230205180739': {'depth':14376},
        },
    },
    'Scroll2':{
        'segments': [
            '20230421192746',
            '20230421204550',
            '20230421215232',
            '20230421235552',
            '20230422011040',
            '20230422213203',
            '20230424181417',
            '20230424213608',
            '20230425163721',
            '20230425200944',
            '20230426114804',
            '20230426144221',
            '20230427171131',
            '20230501040514',
            '20230501042136',
            '20230503120034',
            '20230503213852',
            '20230504151750',
            '20230505142626',
            '20230505150348',
            '20230506111616',
            '20230506141535',
            '20230506142341',
            '20230506145035',
            '20230506145829',
            '20230506151750',
            '20230507064642',
            '20230507125513',
            '20230507175344',
            '20230508032834',
            '20230508080928',
            '20230508131616',
            '20230508171353',
            '20230508181757',
            '20230509144225',
            '20230509163359',
            '20230509173534',
            '20230511150730',
            '20230512192835',
            '20230512211850',
            '20230515151114',
            '20230516154633',
            '20230517000306',
            '20230517104414',
            '20230517151648',
            '20230517153958',
            '20230517164827',
            '20230517171727',
            '20230517193901',
            '20230517214715',
            '20230518210035',
            '20230519033308',
            '20230520080703',
            '20230520105602',
            '20230522172834',
            '20230522182853',
            '20230709155141',
            '20230801194757',
        ],
        'volumes': {
            '20230210143520': {},
            '20230212125146': {},
        },
    },
}

def load_cropped_tiff_slices(tiff_directory, start_slice, end_slice, crop_start, crop_end, padlen):
    slices_data = []
    for slice_index in range(start_slice, end_slice):
        tiff_filename = f"{slice_index:0{padlen}d}.tif"
        tiff_path = os.path.join(tiff_directory, tiff_filename)
        tiff_data = tifffile.memmap(tiff_path)
        if tiff_data.dtype != np.uint8:
            raise ValueError("invalid input dtype from tiff files, must be uint8")
        slices_data.append(tiff_data[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]])
    return slices_data


class VolMan:
    def __init__(self, cachedir='D:/vesuvius.volman'):
        self.cachedir = cachedir
        pass

    def _make_url(self,scroll,source,idnum,file=None):

        if scroll == 'PHerc0332' and source == 'volumes' and idnum == '20231027191953':
            url = f'https://dl.ash2txt.org/community-uploads/james-darby/PHerc0332/volumes_masked/20231027191953_jpg/'
            if file:
                file = file.replace('.tif','.jpg')
        elif scroll == 'PHerc1667' and source == 'volumes' and idnum == '20231107190228':
            url = f'https://dl.ash2txt.org/community-uploads/james-darby/PHerc1667/volumes_masked/20231107190228_jpg/'
            if file:
                file = file.replace('.tif','.jpg')
        elif scroll == 'Scroll1' and source  == 'volumes' and idnum == '20230205180739':
            url = 'https://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/volumes_masked/20230205180739/'
        elif scroll == 'Scroll2' and source == 'volumes' and idnum == '20230210143520':
            url = 'https://dl.ash2txt.org/full-scrolls/Scroll2.volpkg/volumes_masked/20230210143520/'
        else:
            if source == 'segments':
                source = 'paths'
            url = f'https://dl.ash2txt.org/full-scrolls/{scroll}.volpkg/{source}/{idnum}/'
            if source == 'paths':
                url += 'layers/'
        if file is not None:
            url += file
        return url

    def _get_pad_and_len(self,scroll,source,idnum):
        depth = the_index[scroll][source][idnum]['depth']
        return len(str(depth)), depth

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
                f'start:{start} or end:{end} is greater than {numtiffs} tiffs in {scroll}/{source}/{idnum}')

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

        padlen, numtiffs = self._get_pad_and_len(scroll, source, idnum)
        if start > numtiffs or end > numtiffs:
            raise ValueError(
                f'start:{start} or end:{end} is greater than {numtiffs} tiffs in {scroll}/{source}/{idnum}')

        mask_path = f'{self.cachedir}/{scroll}/{source}/{idnum}_masks'

        for idx in range(start, end):
            filename = f"{idx:0{padlen}d}.tif"
            mask_file = os.path.join(mask_path, filename)

            full_mask_slice = tifffile.memmap(mask_file)
            full_mask_slice[yoff:yoff + ysize, xoff:xoff + xsize] = mask[idx - start]
            #tifffile.imwrite(mask_file, full_mask_slice)

    def chunk(self, scroll, source, idnum, start, size):
        ''' get a 3d chunk of data. Download the sources if necessary, otherwise pull them from the cache directory'''

        user = os.environ['SCROLLPRIZE_USER']
        password = os.environ['SCROLLPRIZE_PASS']

        if scroll not in the_index.keys():
            raise ValueError(f'{scroll} is not a valid scroll')
        if source not in the_index[scroll].keys():
            raise ValueError(f'{source} is not a valid source')
        if idnum not in the_index[scroll][source]:
            raise ValueError(f'{idnum} is not a valid id for {source} in {scroll}')
        elif source in ['segments', 'volumes']:

            zoff, yoff, xoff = start
            zsize, ysize, xsize = size

            start = zoff
            end = start + zsize
            padlen, numtiffs = self._get_pad_and_len(scroll, source, idnum)
            if start > numtiffs or end > numtiffs:
                raise ValueError(
                    f'start:{start} or end:{end} is greater than {numtiffs} tiffs in {scroll}/{source}/{idnum}')
            dl_path = f'{self.cachedir}/{scroll}/{source}/{idnum}'

            def download_file(idx, start, end, padlen, scroll, source, idnum, dl_path, user, password):
                filename = f"{idx:0{padlen}d}.tif"
                url = self._make_url(scroll, source, idnum, filename)

                os.makedirs(dl_path, exist_ok=True)
                local_file = os.path.join(dl_path, filename)
                if os.path.exists(local_file):
                    return

                print(f"downloading {url}")
                response = requests.get(url, auth=(user, password))
                if response.status_code == 200:
                    if url.endswith('.tif'):
                        data = tifffile.imread(io.BytesIO(response.content))
                        data //= 256
                        data = data.astype(np.uint8)
                    elif url.endswith('.jpg'):
                        data = np.array(Image.open(io.BytesIO(response.content)))
                    else:
                        raise ValueError("Unsupported file format")
                    data &= 0xf0
                    tifffile.imwrite(local_file, data)
                    print(f'Downloaded {url} to {local_file}')
                else:
                    print(f'Failed to download {url}. Status code: {response.status_code}')

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for idx in range(start, end):
                    future = executor.submit(download_file, idx, start, end, padlen, scroll, source, idnum, dl_path,
                                             user, password)
                    futures.append(future)

                concurrent.futures.wait(futures)
            crop_start = (yoff, xoff)
            crop_end = (yoff + ysize, xoff + xsize)
            data = load_cropped_tiff_slices(dl_path, start, end, crop_start, crop_end, padlen)
            data = np.stack(data, axis=0)

            # Check for the existence of mask files and create them if they don't exist
            mask_path = f'{self.cachedir}/{scroll}/{source}/{idnum}_masks'
            os.makedirs(mask_path, exist_ok=True)

            for idx in range(start, end):
                filename = f"{idx:0{padlen}d}.tif"
                mask_file = os.path.join(mask_path, filename)

                if not os.path.exists(mask_file):
                    print(f"creating mask {mask_file}")
                    mask_slice = np.zeros_like(tifffile.memmap(os.path.join(dl_path, filename)), dtype=np.uint16)
                    tifffile.imwrite(mask_file, mask_slice)

            return data


