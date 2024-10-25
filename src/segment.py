import random
from snic import snic
from scipy.spatial import cKDTree
import skimage
from scipy import ndimage
import numpy as np
from scipy.spatial import cKDTree
import copy
import bisect



from common import timing_decorator, get_chunk
from numbamath import sumpool, rescale_array, get_superpixel_connectivity
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
             self.superpixels[i]) for i in indices]

  def get_all_points_with_superpixels(self):
    return list(zip(self.points, self.superpixels))

  def clear(self):
    self.points = []
    self.superpixels = []
    self.tree = None

@timing_decorator
def bin_superpixels(superpixels, nbins):
  bins = []
  for x in range(nbins):
    bins.append(set())
  for sp in superpixels:
    bins[int(sp.c*255.0 / nbins)].add(sp)
  return bins

#@timing_decorator
#def segment_superpixel_connectivity(superpixel_connectivity, segment, superpixel):



@timing_decorator
def superpixel_flow(superpixels: [snic.Superpixel], labels, iso, density, compactness, nbins, data):
  segments = []
  frontiers = []
  processed = set()

  #kdtree = KDTree3D()
  #points = [(sp.z,sp.y,sp.x) for sp in superpixels]
  #kdtree.add_points(points, superpixels)

  # we want seed values that are both high and fairly regularly distributed throughout the
  # chunk
  for x in range(256):
    segments.append(set())
    frontiers.append(set())

  bins = bin_superpixels(superpixels, nbins)
  n = 0
  seeds = set()
  for bin in reversed(bins):
    if n == 256:
      break
    for sp in bin:
      seeds.add(sp)
      n+=1
      if n == 256:
        break

  for i, sp in enumerate(seeds):
    segments[i].add(sp)
    frontiers[i] = list(reversed(sorted(sp.neighs, key=lambda x: x.c)))

  superpixel_connectivity = get_superpixel_connectivity(len(superpixels), labels, data)

  for _ in range(256):
    segments.append(set())
    frontiers.append(list())

  cur = 1
  len_at_last_check = 0
  while len(processed) < len(superpixels) -1:
    if cur == 256:
      if len_at_last_check == len(processed):
        #we went through the entire list and havent added to anything so quit i guess
        break
      len_at_last_check = len(processed)
      cur = 1
    segment = segments[cur]
    frontier = frontiers[cur]
    for candidate in frontier:
      if candidate in processed:
        continue
      if candidate.c < iso:
        segments[0].add(candidate)
        processed.add(candidate)
        continue
      segment.add(candidate)
      processed.add(candidate)

      connectivity  = np.nonzero(superpixel_connectivity[candidate.label])[0]

      connectedness = [superpixel_connectivity[candidate.label][i] for i in connectivity]
      print()
      for n in candidate.neighs:
        if n.c < iso:
          segments[0].add(n)
          processed.add(n)
        if n not in processed:
          bisect.insort(frontier,n, key=lambda x: x.c)
      break
    else:
      pass
      #print()
    cur+=1

  sp_to_seg = dict()
  segments = list(filter(lambda x: len(x) > 0,segments))
  segmentssorted = list(sorted(segments,key=lambda x: sum(_.c for _ in x)/len(x)))

  for i,segment in enumerate(segmentssorted):
    for sp in segment:
      sp_to_seg[sp] = i

  for sp in superpixels:
    if sp not in sp_to_seg:
      #for whatever reason we didnt process this superpixel so just add it to the void segment
      sp_to_seg[sp] = 0
      segments[0].add(sp)

  return segmentssorted, sp_to_seg

@timing_decorator
def get_segment_connectivity(superpixels, labels, segments, data):
  connectivity = get_superpixel_connectivity(len(superpixels), labels)

@timing_decorator
def label_data(labels, superpixels, sp_to_segment, nsegments):
  ret = np.zeros_like(labels)
  for z in range(labels.shape[0]):
    for y in range(labels.shape[1]):
      for x in range(labels.shape[2]):
        l = labels[z,y,x] #need to get actual superpixel not superpixel number
        sp = superpixels[l]
        v = sp_to_segment[sp]
        ret[z,y,x] = int(v*256//nsegments)
  return ret


@timing_decorator
def segment(data, compactness, density, iso, nbins):
  if iso > 1:
    iso /= 256
  # we dont know how data is scaled so just rescale to 0 - 1
  data = rescale_array(data)

  neigh_overflow, labels, superpixels = snic.snic(data, density, compactness, 80, 160)
  print(f"neighbor overflow {neigh_overflow}")

  segs, sp_to_seg = superpixel_flow(superpixels, labels, iso, density, compactness, nbins, data)
  return superpixels, labels, segs, sp_to_seg


@timing_decorator
def main():
  data = get_chunk(1, 20230205180739, 10, 8, 8)
  data = sumpool(data, (2, 2, 2), (2, 2, 2), (1, 1, 1))
  data = data.astype(np.float32)
  data = rescale_array(data)
  data = global_local_contrast_3d(data)
  data = skimage.exposure.equalize_adapthist(data, nbins=16)
  neigh_overflow, labels, superpixels = snic.snic(data, 4, 10.0, 80, 160)
  seeds = get_superpixel_seeds(superpixels, len(superpixels) // 1000)
  segments = superpixel_flow(superpixels, labels, seeds, 10, 0.5)
  print(segments)
  print(seeds)
  print(neigh_overflow)
  print(labels)
  print(superpixels)


if __name__ == "__main__":
  main()
