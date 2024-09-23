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
    ('neighs', numba.uint32[:])
]

@numba.experimental.jitclass(spec)
class Superpixel:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.c = 0
        self.n = 0
        self.nlow = 0
        self.nmid = 0
        self.nhig = 0
        self.neighs = np.zeros(SUPERPIXEL_MAX_NEIGHS, dtype=np.uint32)

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
            asdf = subprocess.run(r"clang.exe c/snic.c -shared -o {}/bin/snic.dll -O3 -g3 -march=native -ffast-math -fopenmp".format(ROOTDIR).split(),
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
    for i in range(superpixels_count + 1):
        sp = Superpixel()
        sp.x = superpixels_ctype[i].x
        sp.y = superpixels_ctype[i].y
        sp.z = superpixels_ctype[i].z
        sp.c = superpixels_ctype[i].c
        sp.n = superpixels_ctype[i].n
        sp.nlow = superpixels_ctype[i].nlow
        sp.nmid = superpixels_ctype[i].nmid
        sp.nhig = superpixels_ctype[i].nhig
        sp.neighs = np.ctypeslib.as_array(superpixels_ctype[i].neighs, shape=(SUPERPIXEL_MAX_NEIGHS,))
        superpixels.append(sp)

    return neigh_overflow, labels, superpixels
