# This is adapted from https://github.com/spelufo/stabia/tree/main
# licensed under the MIT license
# Modified by VerditeLabs for usage with Hraun

'''
MIT License

Copyright (c) 2023 Santiago Pelufo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


import ctypes
import numpy as np
import platform
import os
import subprocess
import numba

from common import timing_decorator, ROOTDIR

SUPERPIXEL_MAX_NEIGHS = 56*2  # Replace with the actual value from your C co cde

spec = [
    ('x', numba.float32),
    ('y', numba.float32),
    ('z', numba.float32),
    ('c', numba.float32),               # a simple scalar field
    ('n', numba.uint32),               # a simple scalar field
    ('nlow', numba.uint32),               # a simple scalar field
    ('nmid', numba.uint32),               # a simple scalar field
    ('nhig', numba.uint32),
    ('neighs', numba.uint32[:]),
    ('label', numba.uint32)
]

#@numba.experimental.jitclass(spec)
class Superpixel:
    def __init__(self, x,y,z,c,n,nlow,nmid,nhigh,neighs,label):
        self.x = x
        self.y = y
        self.z = z
        self.c = c
        self.n = n
        self.nlow = nlow
        self.nmid = nmid
        self.nhig = nhigh
        self.neighs = neighs
        self.label = label

# Define the Superpixel structure
class SuperpixelCType(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('z', ctypes.c_float),
        ('c', ctypes.c_float),
        ('n', ctypes.c_uint),
        ('nlow', ctypes.c_uint),
        ('nmid', ctypes.c_uint),
        ('nhig', ctypes.c_uint),
        ('neighs', ctypes.c_uint * SUPERPIXEL_MAX_NEIGHS),
    ]


@timing_decorator
def snic(img, d_seed, compactness, lowmid, midhig):
    if platform.system() == 'Windows':
        os.makedirs(f"{ROOTDIR}/bin", exist_ok=True)
        if not os.path.exists(f"{ROOTDIR}/bin/snic.dll"):
            asdf = subprocess.run(r"clang.exe c/snic.c -shared -o {}/bin/snic.dll -O3 -g3 -march=native -ffast-math -fopenmp -DNDEBUG".format(ROOTDIR).split(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(asdf)
        snic_lib = ctypes.CDLL(f'{ROOTDIR}/bin/snic.dll')
    else:
        snic_lib = ctypes.CDLL('path/to/libsnic.so')

    snic_lib.snic.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32),  # img
        ctypes.c_int,  # lx
        ctypes.c_int,  # ly
        ctypes.c_int,  # lz
        ctypes.c_int,  # d_seed
        ctypes.c_float,  # compactness
        ctypes.c_float,  # lowmid
        ctypes.c_float,  # midhig
        np.ctypeslib.ndpointer(dtype=np.uint32),  # labels
        ctypes.POINTER(SuperpixelCType),  # superpixels
    ]
    snic_lib.snic.restype = ctypes.c_int

    snic_lib.snic_superpixel_count.argtypes = [
        ctypes.c_int,  # lx
        ctypes.c_int,  # ly
        ctypes.c_int,  # lz
        ctypes.c_int,  # d_seed
    ]
    snic_lib.snic_superpixel_count.restype = ctypes.c_int

    img = np.ascontiguousarray(img, dtype=np.float32)
    lx, ly, lz = img.shape
    labels = np.zeros((lx, ly, lz), dtype=np.uint32)
    superpixels_count = snic_lib.snic_superpixel_count(lx, ly, lz, d_seed)
    superpixels_ctype = (SuperpixelCType * (superpixels_count + 1))()

    snic_lib.snic.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, shape=(lx, ly, lz)),  # img
        ctypes.c_int,  # lx
        ctypes.c_int,  # ly
        ctypes.c_int,  # lz
        ctypes.c_int,  # d_seed
        ctypes.c_float,  # compactness
        ctypes.c_float,  # lowmid
        ctypes.c_float,  # midhig
        np.ctypeslib.ndpointer(dtype=np.uint32),  # labels
        ctypes.POINTER(SuperpixelCType),  # superpixels_ctype
    ]

    neigh_overflow = snic_lib.snic(
        img,
        lx, ly, lz,
        d_seed,
        compactness, lowmid, midhig,
        labels,
        ctypes.cast(superpixels_ctype, ctypes.POINTER(SuperpixelCType))
    )

    superpixels = []

    for i in range(superpixels_count+1):
        x = superpixels_ctype[i].x
        y = superpixels_ctype[i].y
        z = superpixels_ctype[i].z
        c = superpixels_ctype[i].c
        n = superpixels_ctype[i].n
        nlow = superpixels_ctype[i].nlow
        nmid = superpixels_ctype[i].nmid
        nhigh = superpixels_ctype[i].nhig
        label = i
        sp = Superpixel(x,y,z,c,n,nlow,nmid,nhigh,set(),label)
        superpixels.append(sp)

    #fixup neighbors now since we wouldnt while creating the array because we needed future superpixels
    #that exist now
    for sp in superpixels:
        for n in superpixels_ctype[sp.label].neighs:
            if n == 0:
                break
            sp.neighs.add(superpixels[n])

    return neigh_overflow, labels, superpixels
