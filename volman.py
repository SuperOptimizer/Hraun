import requests
import os
import io
import tifffile
import numpy as np
from PIL import Image
import shutil
import concurrent


the_index = {
    'Scroll1': {
        '20230205180739': {'depth': 14376, 'height': 7888, 'width': 8096, 'ext': 'tif', 'url': 'https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes/20230205180739/'},
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

USER = os.environ.get('SCROLLPRIZE_USER')
PASS = os.environ.get('SCROLLPRIZE_PASS')

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
                mask_slice = np.zeros_like(tifffile.memmap(os.path.join(dl_path, filename)), dtype=np.uint16)
                tifffile.imwrite(mask_file, mask_slice)

        return data
