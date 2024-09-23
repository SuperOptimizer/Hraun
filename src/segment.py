import numpy as np
from numba import jit
import random
from snic import snic
from scipy.spatial import cKDTree

from common import timing_decorator, CACHEDIR, get_chunk
from numbamath import argmaxpool, argminpool, sumpool, avgpool, minpool, maxpool, index_to_offset_3d, rescale_array
from preprocessing import global_local_contrast_3d


class KDTree3D:
  def __init__(self):
    self.points = []
    self.superpixels = []
    self.tree = None

  def add_point(self, z, y, x, superpixel):
    self.points.append([z, y, x])
    self.superpixels.append(superpixel)
    self._rebuild_tree()

  @timing_decorator
  def add_points(self, points, superpixels):
    self.points.extend(points)
    self.superpixels.extend(superpixels)
    self._rebuild_tree()

  @timing_decorator
  def _rebuild_tree(self):
    if self.points:
      self.tree = cKDTree(np.array(self.points))
    else:
      self.tree = None

  @timing_decorator
  def find_nearest_neighbors(self, query_point, k=1, max_distance=50.0):
    if self.tree is None:
      return []

    distances, indices = self.tree.query(query_point, k=k, distance_upper_bound=max_distance)

    if k == 1:
      distances = [distances]
      indices = [indices]

    results = []
    for dist, idx in zip(distances, indices):
      if idx < len(self.points):  # Check if the index is valid
        results.append((dist, self.points[idx], self.superpixels[idx]))

    return results

  @timing_decorator
  def find_points_within_radius(self, query_point, radius):
    if self.tree is None:
      return []

    indices = self.tree.query_ball_point(query_point, radius)
    return [(np.linalg.norm(np.array(self.points[i]) - np.array(query_point)),
             self.points[i],
             self.superpixels[i]) for i in indices]


  def get_all_points_with_superpixels(self):
    return list(zip(self.points, self.superpixels))

  def clear(self):
    self.points = []
    self.superpixels = []
    self.tree = None

@timing_decorator
@jit(nopython=True)
def find_seed_points(arr, nseeds):
  maxiter = 10

  maxes = argmaxpool(arr, (3, 3, 3), (1, 1, 1), (1, 1, 1))
  maxima = set()
  for seed in range(nseeds):
    z, y, x = (random.randint(0, arr.shape[0] - 1),
               random.randint(0, arr.shape[1] - 1),
               random.randint(0, arr.shape[2] - 1))
    for i in range(maxiter):
      offset = index_to_offset_3d(maxes[z, y, x], (3, 3, 3))
      if offset[0] == 0 and offset[1] == 0 and offset[2] == 0:
        maxima.add((z, y, x))
        break
      z += offset[0]
      y += offset[1]
      x += offset[2]
    else:
      maxima.add((z, y, x))

  mins = argminpool(arr, (3, 3, 3), (1, 1, 1), (1, 1, 1))
  minima = set()
  for seed in range(nseeds):
    z, y, x = (random.randint(0, arr.shape[0] - 1),
               random.randint(0, arr.shape[1] - 1),
               random.randint(0, arr.shape[2] - 1))
    for i in range(maxiter):
      offset = index_to_offset_3d(mins[z, y, x], (3, 3, 3))
      if offset[0] == 0 and offset[1] == 0 and offset[2] == 0:
        minima.add((z, y, x))
        break
      z += offset[0]
      y += offset[1]
      x += offset[2]
    else:
      minima.add((z, y, x))
  return maxima, minima


@timing_decorator
@jit(nopython=True)
def walk(arr, seeds):
  labels = np.zeros(arr.shape, dtype=np.uint16)
  max_iter = 10
  for label, seed in enumerate(seeds):
    z, y, x = seed
    for i in range(max_iter):
      tries = 0
      nextval = 0.0
      candidates = []
      while tries < 10:
        tries += 1
        for zi in range(-1, 2):
          for yi in range(-1, 2):
            for xi in range(-1, 2):
              if zi == yi == xi == 0:
                continue
              if (z + zi < 0 or z + zi >= arr.shape[0] or
                  y + yi < 0 or y + yi >= arr.shape[1] or
                  x + xi < 0 or x + xi >= arr.shape[2]):
                continue
              if (arr[z + zi, y + yi, x + xi] > nextval and
                  labels[z + zi, y + yi, x + xi] == 0):
                candidates.insert(0,(z + zi, y + yi, x + xi))
                nextval = arr[z + zi, y + yi, x + xi]
        if len(candidates) == 0:
          print()

      assert len(candidates)> 0
      z, y, x = candidates[0]
      labels[z, y, x] = label
  return labels


@timing_decorator
@jit(nopython=True)
def merge_all_superpixels(superpixels, labels, iso):
    num_superpixels = len(superpixels)
    sp_to_segment = np.zeros(num_superpixels, dtype=np.int32)

    for i in range(num_superpixels):
        if superpixels[i].c >= iso:
            sp_to_segment[i] = i + 1
        else:
            sp_to_segment[i] = 0

    changed = True
    while changed:
        changed = False
        for i in range(num_superpixels):
            if sp_to_segment[i] > 0:
                min_label = sp_to_segment[i]
                for n in superpixels[i].neighs:
                    if n == 0:
                        break
                    if sp_to_segment[n] > 0:
                        min_label = min(min_label, sp_to_segment[n])
                if min_label < sp_to_segment[i]:
                    sp_to_segment[i] = min_label
                    changed = True

    final_labels = np.zeros(num_superpixels, dtype=np.int32)
    next_label = 1
    for i in range(num_superpixels):
        if sp_to_segment[i] > 0:
            if final_labels[sp_to_segment[i] - 1] == 0:
                final_labels[sp_to_segment[i] - 1] = next_label
                next_label += 1
            sp_to_segment[i] = final_labels[sp_to_segment[i] - 1]

    new_labels = np.zeros_like(labels)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                new_labels[i, j, k] = sp_to_segment[labels[i, j, k]]

    return new_labels, next_label - 1



@timing_decorator
@jit(nopython=True)
def get_neighbors(labels, num_labels):
  neighbors = np.zeros((num_labels, num_labels), dtype=np.uint16)
  num_neighbors = np.zeros((num_labels,), dtype=np.uint16)
  for z in range(labels.shape[0]):
    for y in range(labels.shape[1]):
      for x in range(labels.shape[2]):
        l = labels[z, y, x]
        if l == 0:
          continue
        for zi in range(-1, 2):
          for yi in range(-1, 2):
            for xi in range(-1, 2):
              if zi == 0 and yi == 0 and xi == 0:
                continue
              if (z + zi < 0 or z + zi >= labels.shape[0] or
                  y + yi < 0 or y + yi >= labels.shape[1] or
                  x + xi < 0 or x + xi >= labels.shape[2]):
                continue
              if (labels[z + zi, y + yi, x + xi] > 0 and
                  labels[z + zi, y + yi, x + xi] != l):
                neighbors[l][num_neighbors[l]] = labels[z + zi, y + yi, x + xi]
                num_neighbors[l] += 1
  return neighbors


@timing_decorator
def segment(data):
  neigh_overflow, labels, superpixels = snic(data, 8, 5.0, 80, 160)
  tree = KDTree3D()
  tree.add_points([[sp.x, sp.y, sp.z] for sp in superpixels], superpixels)
  asdf = tree.find_nearest_neighbors((6.0,11.5,11.2),50)
  print(asdf)
  #labels, next_label = merge_all_superpixels(superpixels, labels, 0.8)
  print()

@timing_decorator
def main():
  data = get_chunk(1, 20230205180739, 10,8,8)
  data = sumpool(data, (2, 2, 2), (2, 2, 2), (1, 1, 1))
  data = data.astype(np.float32)
  data = rescale_array(data)
  data = global_local_contrast_3d(data)
  neigh_overflow, labels, superpixels = snic(data, 8, 5.0, 80, 160)
  labels, next_label = merge_all_superpixels(superpixels, labels, 0.8)
  print()
  #segment(data)


if __name__ == "__main__":
  main()
