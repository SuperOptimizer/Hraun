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

import numpy as np

# HEAP ////////////////////////////////////////////////////////////////////////

class HeapNode:
    def __init__(self, d, k, x, y, z):
        self.d = d
        self.k = k
        self.x = x
        self.y = y
        self.z = z

    def val(self):
        return -self.d

class Heap:
    def __init__(self, size):
        self.len = 0
        self.size = size
        self.nodes = np.empty(size + 1, dtype=HeapNode)

    def left(self, i):
        return 2 * i

    def right(self, i):
        return 2 * i + 1

    def parent(self, i):
        return i // 2

    def fix_edge(self, i, j):
        if self.nodes[j].val() > self.nodes[i].val():
            self.nodes[j], self.nodes[i] = self.nodes[i], self.nodes[j]

    @staticmethod
    def alloc(size):
        return Heap(size*2)

    def free(self):
        pass

    def push(self, node):
        assert self.len <= self.size

        self.len += 1
        self.nodes[self.len] = node
        i = self.len
        while i > 1:
            j = self.parent(i)
            self.fix_edge(j, i)
            if self.nodes[j].val() <= self.nodes[i].val():
                break
            i = j

    def pop(self):
        assert self.len > 0

        node = self.nodes[1]
        self.len -= 1
        self.nodes[1] = self.nodes[self.len + 1]
        i = 1
        while i <= self.len:
            l = self.left(i)
            r = self.right(i)
            if l > self.len:
                break
            j = l
            if r <= self.len and self.nodes[l].val() < self.nodes[r].val():
                j = r
            self.fix_edge(i, j)
            if self.nodes[i].val() >= self.nodes[j].val():
                break
            i = j

        return node

# SNIC ////////////////////////////////////////////////////////////////////////

SUPERPIXEL_MAX_NEIGHS = 56

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

def snic_superpixel_max_neighs():
    return SUPERPIXEL_MAX_NEIGHS

def superpixel_add_neighbors(superpixels, k1, k2):
    for i in range(SUPERPIXEL_MAX_NEIGHS):
        if superpixels[k1].neighs[i] == 0:
            superpixels[k1].neighs[i] = k2
            return 0
        elif superpixels[k1].neighs[i] == k2:
            return 0
    return 1

def snic_superpixel_count(lx, ly, lz, d_seed):
    cz = (lz - d_seed // 2 + d_seed - 1) // d_seed
    cy = (ly - d_seed // 2 + d_seed - 1) // d_seed
    cx = (lx - d_seed // 2 + d_seed - 1) // d_seed
    return cx * cy * cz

def snic(img, d_seed, compactness, lowmid, midhig):
    lx, ly, lz = img.shape
    neigh_overflow = 0  # Number of neighbors that couldn't be added.
    lylx = ly * lx

    def idx(y, x, z):
        return (z * lylx + x * ly + y)

    def sqr(x):
        return x * x

    pq = Heap.alloc(img.size)
    numk = 0
    for iz in range(d_seed // 2, lz, d_seed):
        for ix in range(d_seed // 2, lx, d_seed):
            for iy in range(d_seed // 2, ly, d_seed):
                numk += 1
                grad = np.inf
                x, y, z = ix, iy, iz
                for dz in range(-1, 2):
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            jx, jy, jz = ix + dx, iy + dy, iz + dz
                            if 0 < jx < lx - 1 and 0 < jy < ly - 1 and 0 < jz < lz - 1:
                                gy = img[jy + 1, jx, jz] - img[jy - 1, jx, jz]
                                gx = img[jy, jx + 1, jz] - img[jy, jx - 1, jz]
                                gz = img[jy, jx, jz + 1] - img[jy, jx, jz - 1]
                                jgrad = sqr(gx) + sqr(gy) + sqr(gz)
                                if jgrad < grad:
                                    x, y, z = jx, jy, jz
                                    grad = jgrad
                pq.push(HeapNode(0.0, numk, x, y, z))

    if numk == 0:
        return 0

    invwt = (compactness * compactness * numk) / img.size

    labels = np.zeros((lx, ly, lz), dtype=np.uint32)
    superpixels = np.array([Superpixel() for _ in range(numk + 1)])  # Initialize superpixels array

    while pq.len > 0:
        n = pq.pop()
        i = idx(n.y, n.x, n.z)
        if labels[n.x, n.y, n.z] > 0:
            continue

        k = n.k
        labels[n.x, n.y, n.z] = k
        superpixels[k].c += img[n.x, n.y, n.z]
        superpixels[k].x += n.x
        superpixels[k].y += n.y
        superpixels[k].z += n.z
        superpixels[k].n += 1
        if img[n.x, n.y, n.z] <= lowmid:
            superpixels[k].nlow += 1
        elif img[n.x, n.y, n.z] <= midhig:
            superpixels[k].nmid += 1
        else:
            superpixels[k].nhig += 1

        def do_neigh(ndy, ndx, ndz, ioffset):
            nonlocal neigh_overflow
            xx, yy, zz = n.x + ndx, n.y + ndy, n.z + ndz
            if 0 <= xx < lx and 0 <= yy < ly and 0 <= zz < lz:
                ii = i + ioffset
                if labels[xx, yy, zz] <= 0:
                    ksize = superpixels[k].n
                    dc = sqr(100.0 * (superpixels[k].c - (img[xx, yy, zz] * ksize)))
                    dx = superpixels[k].x - xx * ksize
                    dy = superpixels[k].y - yy * ksize
                    dz = superpixels[k].z - zz * ksize
                    dpos = sqr(dx) + sqr(dy) + sqr(dz)
                    d = (dc + dpos * invwt) / (ksize * ksize)
                    pq.push(HeapNode(d, k, xx, yy, zz))
                elif k != labels[xx, yy, zz]:
                    neigh_overflow += superpixel_add_neighbors(superpixels, k, labels[xx, yy, zz])
                    neigh_overflow += superpixel_add_neighbors(superpixels, labels[xx, yy, zz], k)

        do_neigh(1, 0, 0, 1)
        do_neigh(-1, 0, 0, -1)
        do_neigh(0, 1, 0, ly)
        do_neigh(0, -1, 0, -ly)
        do_neigh(0, 0, 1, lylx)
        do_neigh(0, 0, -1, -lylx)

    for k in range(1, numk + 1):
        ksize = superpixels[k].n
        if ksize != 0:
            superpixels[k].c /= ksize
            superpixels[k].x /= ksize
            superpixels[k].y /= ksize
            superpixels[k].z /= ksize
        else:
            # Handle the case when ksize is zero
            # You can assign default values or handle it based on your requirements
            superpixels[k].c = 0
            superpixels[k].x = 0
            superpixels[k].y = 0
            superpixels[k].z = 0

    pq.free()
    return neigh_overflow, labels, superpixels