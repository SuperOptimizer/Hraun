import numpy as np
from networkx.classes import neighbors
from numba import jit
import random
from snic import snic
from scipy.spatial import cKDTree
import skimage
from scipy import ndimage
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import collections
import copy

from common import timing_decorator, CACHEDIR, get_chunk
from numbamath import argmaxpool, argminpool, sumpool, avgpool, minpool, maxpool, index_to_offset_3d, rescale_array
from preprocessing import global_local_contrast_3d
import snic


@timing_decorator
def analyze_and_process_components(voxel_data, iso_value, size_threshold, largest=True, remove=False):
    """
    Analyze connected components, optionally remove them, and return component information.

    :param voxel_data: 3D numpy array of voxel data
    :param iso_value: Threshold for creating binary mask
    :param size_threshold: Size threshold for components to process
    :param largest: If True, process largest components; if False, process smallest
    :param remove: If True, remove the selected components from voxel_data
    :return: If remove is True, return updated voxel_data. Otherwise, return component_info
    """
    binary_mask = voxel_data > iso_value
    labels, num_labels = skimage.measure.label(binary_mask, return_num=True)
    component_sizes = np.bincount(labels.ravel())

    # Select components based on size threshold
    if largest:
        selected_components = np.where(component_sizes > size_threshold)[0]
    else:
        selected_components = np.where((component_sizes <= size_threshold) & (component_sizes > 0))[0]

    if remove:
        if largest:
            # Remove large components
            mask = np.isin(labels, selected_components, invert=True)
        else:
            # Remove small components
            mask = np.isin(labels, selected_components, invert=True)
        voxel_data = voxel_data * mask
        return voxel_data
    else:
        component_info = []
        for label in selected_components:
            points = np.argwhere(labels == label)
            size = component_sizes[label]
            centroid = ndimage.center_of_mass(labels == label)
            component_info.append({
                'label': label,
                'sample_point': tuple(points[0]),
                'size': size,
                'centroid': centroid
            })
        return component_info


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


def get_superpixel_seeds(superpixels, nseeds):
  return list(reversed(sorted(superpixels, key=lambda x: x.c)))[:nseeds]


@timing_decorator
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
def get_flow_order(superpixels, seeds):
  order = []
  to_process = copy.copy(seeds)
  to_process_set = set(to_process)

  processed = set()
  sp_to_candidates = dict()

  for sp in superpixels:
    candidates = list(reversed(sorted(sp.neighs, key=lambda x: x.c)))
    sp_to_candidates[sp] = candidates

  while len(to_process) > 0:
    sp = to_process.pop(0)
    to_process_set.remove(sp)
    candidates = sp_to_candidates[sp]
    for n in candidates:
      if n in processed or n in to_process_set:
        continue
      to_process.append(sp)
      to_process.append(n)
      to_process_set.add(sp)
      to_process_set.add(n)
      break
    else:
      processed.add(sp)
    order.append(sp)
  new_order = []
  new_order_set = set()
  for sp in order:
    if sp not in new_order_set:
      new_order.append(sp)
      new_order_set.add(sp)
  return new_order

@timing_decorator
def superpixel_flow(superpixels: snic.Superpixel, labels, seeds, nsteps, iso):
  segments = dict()
  to_process = list()
  for seed in seeds:
    to_process.append(seed)
  order = get_flow_order(superpixels, seeds)
  label = 1
  for sp in order:
    assert sp not in segments
    #if >= 1 neighbors belong to exactly one segment, the sp joins that segment
    #if exactly zero neighbors are in a segment, then this superpixel becomes a new segment
    #if more than one neighbor belongs to more than one segment, ???
    # should we merge the segments?
    possible_segments = list()
    if sp.c < iso:
      segments[sp] = 0
      continue
    for n in sp.neighs:
      if s := segments.get(n):
        possible_segments.append(s)
    if len(possible_segments) == 0:
      segments[sp] = label
      label +=1
    elif len(set(possible_segments)) == 1:
      #we can get duplicate possible segments which we may need to determine boundaries
      #so remove duplicates for this check
      segments[sp] = possible_segments[0]
    else:
      l = 0
      count = 0
      for s in set(possible_segments):
        c = possible_segments.count(s)
        if c > count:
          count = c
          l = s
      segments[sp] = l

  seg_to_sp = dict()
  for sp, seg in segments.items():
    if seg in seg_to_sp:
      seg_to_sp[seg].add(sp)
    else:
      seg_to_sp[seg] = set()
      seg_to_sp[seg].add(sp)
  return list(reversed(sorted(seg_to_sp.values(), key= lambda x: len(x))))


@timing_decorator
def segment(data):
  neigh_overflow, labels, superpixels = snic.snic(data, 8, 10.0, 80, 160)
  seeds = get_superpixel_seeds(superpixels, len(superpixels) // 1000)
  segments = superpixel_flow(superpixels, labels, seeds, 10, 0.5)
  return superpixels, labels, segments

@timing_decorator
def main():
  data = get_chunk(1, 20230205180739, 10,8,8)
  data = sumpool(data, (4, 4, 4), (4, 4, 4), (1, 1, 1))
  data = data.astype(np.float32)
  data = rescale_array(data)
  data = global_local_contrast_3d(data)
  data = skimage.exposure.equalize_adapthist(data, nbins=16)
  neigh_overflow, labels, superpixels = snic.snic(data, 8, 10.0, 80, 160)
  seeds = get_superpixel_seeds(superpixels, len(superpixels)//1000)
  segments = superpixel_flow(superpixels, labels, seeds, 10, 0.5)
  print(segments)
  print(seeds)
  print(neigh_overflow)
  print(labels)
  print(superpixels)


if __name__ == "__main__":
  main()
