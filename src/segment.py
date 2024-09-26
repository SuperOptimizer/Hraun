import numpy as np
from numba import jit
import random
from snic import snic
from scipy.spatial import cKDTree
import skimage
from scipy import ndimage
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

from common import timing_decorator, CACHEDIR, get_chunk
from numbamath import argmaxpool, argminpool, sumpool, avgpool, minpool, maxpool, index_to_offset_3d, rescale_array
from preprocessing import global_local_contrast_3d


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


@timing_decorator
def find_superpixel_seeds(superpixels, nseeds):
  return reversed(sorted(superpixels, key= lambda x: x.c))[:nseeds]

@timing_decorator
def greedy_walk(superpixels, iso):
  nextlabel = 1
  label = 0
  sp_to_segment = dict()
  for i,sp in enumerate(superpixels):
    if sp in sp_to_segment:
      continue
    if sp.c > iso:
      label = nextlabel
      if label >= 256:
        raise Exception("cant label more than 256 within one chunk")
      nextlabel+=1
    else:
      sp_to_segment[i] = 0
    to_be_processed = [i]
    while len(to_be_processed) > 0:
      cur = to_be_processed.pop()
      if superpixels[cur].c < iso:
        sp_to_segment[cur] = 0
      else:
        sp_to_segment[cur] = label
      for n in superpixels[cur].neighs:
        if n == 0:
          break
        if n not in to_be_processed and n not in sp_to_segment:
          to_be_processed.append(n)
  return sp_to_segment

#@jit(nopython=True)
def label_voxel_data(superpixels, sp_to_segment, labels):
  ret = np.zeros_like(labels,dtype=np.uint8)
  for z in range(labels.shape[0]):
    for y in range(labels.shape[1]):
      for x in range(labels.shape[2]):
        l = labels[z,y,x]
        v = sp_to_segment[l]

        ret[z,y,x] = v
  return ret



@timing_decorator
def segment(data, iso):
  if any([data.shape[0] > 256,data.shape[1] > 256 ,data.shape[2] > 256]):
    raise ValueError("can only do snic segmentation on regions of less than or equal to 256x256x256")
  neigh_overflow, labels, superpixels = snic(data, 8, 5.0, 80, 160)
  sp_to_segment = greedy_walk(superpixels, iso)
  sp_to_segment_numpy = np.zeros(len(superpixels),dtype=np.uint8)
  for sp,label in sp_to_segment.items():
    sp_to_segment_numpy[sp] = label
  data = label_voxel_data(superpixels, sp_to_segment_numpy, labels)

  return data
  #find_superpixel_seeds(superpixels, 100)
  #tree = KDTree3D()
  #tree.add_points([[sp.x, sp.y, sp.z] for sp in superpixels], superpixels)
  #asdf = tree.find_nearest_neighbors((6.0,11.5,11.2),50)
  #print(asdf)
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
