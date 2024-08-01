import random

import numcodecs
import requests
import os
import io
import tifffile
import numpy as np
from PIL import Image
import zarr
from numcodecs import Blosc

VOLMAN_PATH = 'D:/vesuvius.volman'

the_index = {
  'Scroll1': {
      '20230205180739': {'depth': 14376, 'height': 7888, 'width': 8096, 'ext': 'tif',
                         'url': 'https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_masked/20230205180739/'},
      '20230206171837': {'depth': 10532, 'height': 7812, 'width': 8316, 'ext': 'tif',
                         'url': 'https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes/20230206171837/'},
      '20230503225234': {'depth': 65, 'height': 1962, 'width': 7920, 'ext': 'tif',
                         'url': 'https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20240116164433/layers/'},
      #'20230504093154': {},
      #'20230504094316': {},
      #'20230504125349': {},
      #'20230504171956': {},
      #'20230504223647': {},
      #'20230504225948': {},
      #'20230504231922': {},
      #'20230505093556': {},
      #'20230505113642': {},
      #'20230505131816': {},
      #'20230505135219': {},
      #'20230505141722': {},
      #'20230505164332': {},
      #'20230505175240': {},
      #'20230506133355': {},
      #'20230507172452': {},
      #'20230507175928': {},
      #'20230508164013': {},
      #'20230508220213': {},
      #'20230509160956': {},
      #'20230509182749': {},
      #'20230510153006': {},
      #'20230510153843': {},
      #'20230510170242': {},
      #'20230511085916': {},
      #'20230511094040': {},
      #'20230511201612': {},
      #'20230511204029': {},
      #'20230511211540': {},
      #'20230511215040': {},
      #'20230511224701': {},
      #'20230512094635': {},
      #'20230512105719': {},
      #'20230512111225': {},
      #'20230512112647': {},
      #'20230512120728': {},
      #'20230512123446': {},
      #'20230512123540': {},
      #'20230512170431': {},
      #'20230513092954': {},
      #'20230513095916': {},
      #'20230513164153': {},
      #'20230514173038': {},
      #'20230514182829': {},
      #'20230515162442': {},
      #'20230516112444': {},
      #'20230516114341': {},
      #'20230516115453': {},
      #'20230517021606': {},
      #'20230517024455': {},
      #'20230517025833': {},
      #'20230517180019': {},
      #'20230517204451': {},
      #'20230517205601': {},
      #'20230518012543': {},
      #'20230518075340': {},
      #'20230518104908': {},
      #'20230518130337': {},
      #'20230518135715': {},
      #'20230518181521': {},
      #'20230518191548': {},
      #'20230518223227': {},
      #'20230519031042': {},
      #'20230519140147': {},
      #'20230519195952': {},
      #'20230519202000': {},
      #'20230519212155': {},
      #'20230519213404': {},
      #'20230519215753': {},
      #'20230520132429': {},
      #'20230520175435': {},
      #'20230520191415': {},
      #'20230520192625': {},
      #'20230521093501': {},
      #'20230521104548': {},
      #'20230521113334': {},
      #'20230521114306': {},
      #'20230521155616': {},
      #'20230521182226': {},
      #'20230521193032': {},
      #'20230522055405': {},
      #'20230522151031': {},
      #'20230522152820': {},
      #'20230522181603': {},
      #'20230522210033': {},
      #'20230522215721': {},
      #'20230523002821': {},
      #'20230523020515': {},
      #'20230523034033': {},
      #'20230523043449': {},
      #'20230523182629': {},
      #'20230523191325': {},
      #'20230523233708': {},
      #'20230524004853': {},
      #'20230524005636': {},
      #'20230524092434': {},
      #'20230524163814': {},
      #'20230524173051': {},
      #'20230524200918': {},
      #'20230525051821': {},
      #'20230525115626': {},
      #'20230525121901': {},
      #'20230525190724': {},
      #'20230525194033': {},
      #'20230525200512': {},
      #'20230525212209': {},
      #'20230525234349': {},
      #'20230526002441': {},
      #'20230526015925': {},
      #'20230526154635': {},
      #'20230526164930': {},
      #'20230526175622': {},
      #'20230526183725': {},
      #'20230526205020': {},
      #'20230527020406': {},
      #'20230528112855': {},
      #'20230529203721': {},
      #'20230530025328': {},
      #'20230530164535': {},
      #'20230530172803': {},
      #v'20230530212931': {},
      #'20230531101257': {},
      #'20230531121653': {},
      #'20230531193658': {},
      #'20230531211425': {},
      #'20230601192025': {},
      #'20230601193301': {},
      #'20230601201143': {},
      #'20230601204340': {},
      #'20230602092221': {},
      #'20230602213452': {},
      #'20230603153221': {},
      #'20230604111512': {},
      #'20230604112252': {},
      #'20230604161948': {},
      #'20230605065957': {},
      #'20230606105130': {},
      #'20230606222610': {},
      #'20230608150300': {},
      #'20230608200454': {},
      #'20230608222722': {},
      #'20230609123853': {},
      #'20230611014200': {},
      #'20230611145109': {},
      #'20230612195231': {},
      #'20230613144727': {},
      #'20230613204956': {},
      #'20230619113941': {},
      #'20230619163051': {},
      #'20230620230617': {},
      #'20230620230619': {},
      #'20230621111336': {},
      #'20230621122303': {},
      #'20230621182552': {},
      #'20230623123730': {},
      #'20230623160629': {},
      #'20230624144604': {},
      #'20230624160816': {},
      #v'20230624190349': {},
      #'20230625171244': {},
      #'20230625194752': {},
      #'20230626140105': {},
      #v'20230626151618': {},
      #'20230627122904': {},
      #'20230627170800': {},
      #'20230627202005': {},
      #'20230629215956': {},
      #'20230701020044': {},
      #'20230701115953': {},
      #'20230702182347': {},
      #'20230702185752_superseded': {},
      #'20230702185753': {},
      #'20230705142414': {},
      #'20230706165709': {},
      #'20230707113838': {},
      #'20230709211458': {},
      #'20230711201157': {},
      #'20230711210222': {},
      #'20230711222033': {},
      #'20230712124010': {},
      #'20230712210014': {},
      #'20230713152725': {},
      #'20230717092556': {},
      #'20230719103041': {},
      #'20230719214603': {},
      #'20230720215300': {},
      #'20230721122533': {},
      #'20230721143008': {},
      #'20230801193640': {},
      #'20230806094533': {},
      #'20230806132553': {},
      #'20230808163057': {},
      #'20230812170020': {},
      #'20230819093803': {},
      #'20230819210052': {},
      #'20230820091651': {},
      #'20230820174948': {},
      #'20230820203112': {},
      #'20230826135043': {},
      #'20230826170124': {},
      #'20230826211400': {},
      #'20230827161846_superseded': {},
      #'20230827161847': {},
      #'20230828154913': {},
      #'20230901184804': {},
      #'20230901234823': {},
      #'20230902141231': {},
      #'20230903193206': {},
      #'20230904020426': {},
      #'20230904135535': {},
      #'20230905134255': {},
      #'20230909121925': {},
      #'20230918021838': {},
      #'20230918022237': {},
      #'20230918023430': {},
      #'20230918024753': {},
      #'20230918140728': {},
      #v'20230918143910': {},
      #'20230918145743': {},
      #'20230919113918': {},
      #'20230922174128': {},
      #'20230925002745': {},
      #'20230925090314': {},
      #'20230926164631': {},
      #'20230926164853': {},
      #v'20230929220920_superseded': {},
      #'20230929220921_superseded': {},
      #v'20230929220923_superseded': {},
      #'20230929220924_superseded': {},
      #'20230929220925_superseded': {},
      #'20230929220926': {},
      #'20231001164029': {},
      #'20231004222109': {},
      #v'20231005123333_superseded': {},
      #'20231005123334_superseded': {},
      #'20231005123335_superseded': {},
      #'20231005123336': {},
      #'20231007101615_superseded': {},
      #'20231007101616_superseded': {},
      #'20231007101617_superseded': {},
      #'20231007101618_superseded': {},
      #'20231007101619': {},
      #'20231011111857': {},
      #'20231011144857': {},
      #'20231012085431': {},
      #'20231012173610': {},
      #'20231012184420': {},
      #'20231012184421_superseded': {},
      #'20231012184422_superseded': {},
      #'20231012184423_superseded': {},
      #'20231012184424': {},
      #'20231016151000_superseded': {},
      #'20231016151001_superseded': {},
      #'20231016151002': {},
      #'20231022170900_superseded': {},
      #'20231022170901': {},
      #'20231024093300': {},
      #'20231031143850_superseded': {},
      #'20231031143851_superseded': {},
      #'20231031143852': {},
      #'20231106155350_superseded': {},
      #'20231106155351': {},
      #'20231205141500': {},
      #'20231206155550': {},
      #'20231210121320_superseded': {},
      #'20231210121321': {},
      #'20231221180250_superseded': {},
      #'20231221180251': {},
      #'20231231235900_GP': {},
      #'20240101215220': {},
      #'20240102231959': {},
      #'20240107134630': {},
      #'20240109095720': {},
      #'20240110113230': {},
      #'20240116164433': {},
  },
  'Scroll2': {
      '20230210143520': {'depth': 14428, 'height': 10112, 'width': 11984, 'ext': 'tif',
                         'url': 'https://dl.ash2txt.org/full-scrolls/Scroll2.volpkg/volumes_masked/20230210143520/'},
      '20230212125146': {'depth': 1610, 'height': 8480, 'width': 11136},
      #'20230421192746': {},
      #'20230421204550': {},
      #'20230421215232': {},
      #'20230421235552': {},
      #'20230422011040': {},
      #'20230422213203': {},
      #'20230424181417': {},
      #'20230424213608': {},
      #'20230425163721': {},
      #'20230425200944': {},
      #'20230426114804': {},
      #'20230426144221': {},
      #'20230427171131': {},
      #'20230501040514': {},
      #'20230501042136': {},
      #'20230503120034': {},
      #'20230503213852': {},
      #'20230504151750': {},
      #'20230505142626': {},
      #'20230505150348': {},
      #'20230506111616': {},
      #'20230506141535': {},
      #'20230506142341': {},
      #v'20230506145035': {},
      #'20230506145829': {},
      #'20230506151750': {},
      #'20230507064642': {},
      #'20230507125513': {},
      #'20230507175344': {},
      #'20230508032834': {},
      #'20230508080928': {},
      # '20230508131616': {},
      #'20230508171353': {},
      #'20230508181757': {},
      #'20230509144225': {},
      #'20230509163359': {},
      #'20230509173534': {},
      #'20230511150730': {},
      #'20230512192835': {},
      #'20230512211850': {},
      #'20230515151114': {},
      #'20230516154633': {},
      #'20230517000306': {},
      #'20230517104414': {},
      #'20230517151648': {},
      #'20230517153958': {},
      #'20230517164827': {},
      #'20230517171727': {},
      #v'20230517193901': {},
      #'20230517214715': {},
      #'20230518210035': {},
      #'20230519033308': {},
      #'20230520080703': {},
      #'20230520105602': {},
      #'20230522172834': {},
      #'20230522182853': {},
      #'20230709155141': {},
      #'20230801194757': {},
      #'20240516205750': {},
  },
  'Scroll3': {
      '20231027191953': {'depth': 22941, 'height': 9414, 'width': 9414, 'ext': 'jpg',
                         'url': 'https://dl.ash2txt.org/community-uploads/james/PHerc0332/volumes_masked/20231027191953_jpg/'},
      '20231117143551': {'depth': 9778, 'height': 3550, 'width': 3400, 'ext': 'tif',
                         'url': 'https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes/20231117143551/'},
      '20231201141544': {'depth': 22932, 'height': 9414, 'width': 9414, 'ext': 'tif',
                         'url': 'https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes/20231201141544/'},
      #'20231030220150': {},
      #'20231031231220': {},
      #'20240618142020': {},
      #'20240702133100_thaumato_20231027191953': {},
      #v'20240702133100_thaumato_20231117143551': {},
      #'20240712064330': {},
      #'20240712071520': {},
      #'20240712074250': {},
      #'20240715203740': {},
  },
  'Scroll4': {
      '20231107190228': {'depth': 26391, 'height': 7960, 'width': 8120, 'ext': 'jpg',
                         'url': 'https://dl.ash2txt.org/community-uploads/james-darby/PHerc1667/volumes_masked/20231107190228_jpg/'},
      '20231117161658': {'depth': 11174, 'height': 3340, 'width': 3440},
      #'20231111135340': {},
      #'20231122192640': {},
      #'20231210132040': {},
      #'20240304141530': {},
      #'20240304141531': {},
      #'20240304144030': {},
      #'20240304144031': {},
      #'20240304161940': {},
      #'20240304161941': {},
  },
}

def _download(url, user, password):
  response = requests.get(url, auth=(user, password))
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
      data = np.asarray(Image.open(filedata))
      return data & 0xf0
    elif url.endswith('.png'):
      data = np.asarray(Image.open(filedata))
      return data
  else:
    raise Exception(f'Cannot download {url}')


class ZVol:
  def __init__(self, zroot='C:/vesuvius.zarr', create=False, overwrite=False, modify=False):
    self.username = None
    self.password = None
    synchronizer = zarr.ProcessSynchronizer(zroot + '.sync')
    compressor = numcodecs.Blosc(cname='blosclz', clevel=9, shuffle=Blosc.BITSHUFFLE)
    if create and not overwrite and os.path.exists(zroot):
      raise ValueError(f'{zroot} already exists. Cannot overwrite it')
    if not create and overwrite:
      raise ValueError(f'cannot _not_ create zarr but also overwrite it')
    if not create and not overwrite:
      mode = 'r+' if modify else 'r'
      self.root = zarr.open(zroot, mode=mode, synchronizer=synchronizer)
      return

    # now we're creating the zarr
    self.root = zarr.group(zroot, synchronizer=synchronizer)
    for scroll, timestamps in the_index.items():
        self.root.create_group(f'{scroll}')
        for timestamp in timestamps.keys():
          entry = the_index[scroll][timestamp]
          depth, height, width = entry['depth'], entry['height'], entry['width']
          self.root[f'{scroll}'].zeros(timestamp, shape=(depth, height, width), chunks=(256, 256, 256),
                                                dtype='u1', compressor=compressor, write_empty_chunks=False)
          self.root[f'{scroll}'].zeros(timestamp + '_dlmask', shape=(depth), dtype='u1')

  def update_userpass(self, username, password):
      self.username = username
      self.password = password

  def chunk(self, scroll, timestamp, start, size):
    z,y,x = start
    endz, endy, endx = z + size[0], y + size[1], x + size[2]

    depth = the_index[scroll][timestamp]['depth']
    url = the_index[scroll][timestamp]['url']
    ext = the_index[scroll][timestamp]['ext']

    for i in range(z,endz):
      if self.root[f'{scroll}/{timestamp}_dlmask'][i] == 0:
        src_filename = f"{i:0{len(str(depth))}d}.{ext}"
        print(f"Downloading {url}{src_filename}")
        data = _download(url + src_filename, self.username, self.password)

        if timestamp == '20231201141544':
          maskname = src_filename.replace('tif', 'png')
          mask = _download(
                      'https://dl.ash2txt.org/community-uploads/james/PHerc0332/volumes_masked/20231027191953_unapplied_masks/' + maskname)
          data[mask == 0] = 0
        self.root[f'{scroll}/{timestamp}_dlmask'][i] = 1
        self.root[f'{scroll}/{timestamp}'][i] = data
    return self.root[f'{scroll}/{timestamp}'][z: endz, y: endy, x: endx]

import tiffs2zarr

if __name__ == '__main__':
    zvol = ZVol(create=True, overwrite=True, modify=True)

    input_folder = r'D:\vesuvius.volman\Scroll1\volumes\20230205180739'
    output_file = r'c:\vesuvius.volman\20230205180739.zarr'
    z_size = 14376
    y_size = 7888
    x_size = 8096


    tiffs2zarr.create_3d_array_from_tiffs(zvol.root, input_folder, output_file, z_size, y_size, x_size)
    #zvol.chunk('Scroll3','20231117143551',(1024,1024,1024),(4,256,256))
    #zvol.chunk('Scroll3','20231117143551',(1024,1024,1024),(1,256,256))