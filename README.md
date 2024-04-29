# Hraun 

Hraun is the Old Norse word for lava.

Hraun is a collection of python tools for handling volumetric data, specifically for the [Vesuvius Challenge](https://scrollprize.org/). The main functionality includes:

- Loading and managing volumetric data
- Preprocessing data using techniques such as clipping, rescaling, and contrast enhancement
- TODO: Superpixel segmentation using the SNIC algorithm
- TODO: Unwrapping and flattening
- Generating 3D meshes
- Colorizing and postprocessing the generated meshes
- Saving the resulting meshes

## Main Scripts

1. `hraun.py`: The main script that orchestrates the entire pipeline. It loads the data, applies preprocessing, superpixel segmentation (optional), generates the mesh using Marching Cubes, colorizes and postprocesses the mesh, and saves it to a PLY file.

2. `preprocessing.py`: Contains various preprocessing functions such as clipping, rescaling, global and local contrast enhancement, and applying ink labels to the volumetric data.

3. `snic.py`: Implements the SNIC (Simple Non-Iterative Clustering) superpixel segmentation algorithm. 

4. `volman.py`: Provides a class `VolMan` for managing and loading volumetric data from a remote server. It handles downloading, caching, and chunking the data based on specified scroll, source, and ID.

5. `unwrap.py`: Contains functions for unwrapping scroll images. It calculates the angle between points and performs the unwrapping process to flatten the scroll image.

## Usage

1. Set (`SCROLLPRIZE_USER` and `SCROLLPRIZE_PASS`) environment variables to access [dl.ash2txt.org](dl.ash2txt.org)

2. Modify the `main` function in `hraun.py` to specify the desired scroll, source, ID, chunk size, chunk offset, and output directory.

3. Run `hraun.py` to execute the pipeline.

The resulting PLY file will be saved in the specified output directory.

## Dependencies

The code requires the following Python libraries:

- NumPy
- scikit-image
- Matplotlib
- ctypes
- tifffile
- OpenCV (cv2)
- Pillow (PIL)

## License

The code in this repository is licensed under the [AGPL License](LICENSE) unless noted otherwise or if that code is adapted from an external source

## Acknowledgments
- [@spelufo](https://github.com/spelufo) for his [stabia](https://github.com/spelufo/stabia) repository. Stabia is licensed under the MIT license
- @polytrope on discord for her [updated ink labels](https://discord.com/channels/1079907749569237093/1208440488437350500/1208440488437350500)
- The SNIC algorithm implementation is adapted from the [stabia](https://github.com/spelufo/stabia) repository, which is itself adapted from [achanta/SNIC](https://github.com/achanta/SNIC)
- The global and local contrast enhancement algorithm is adapted from the [glcae](https://github.com/pengyan510/glcae) repository.

## Sources

```bibtex
@inproceedings{snic_cvpr17, 
author = {Achanta, Radhakrishna and Susstrunk, Sabine}, 
title = {Superpixels and Polygons using Simple Non-Iterative Clustering}, 
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
year = {2017} }
```

```bibtex
@inproceedings{tian2017glcae,
  title={Global and Local Contrast Adaptive Enhancement for Non-uniform Illumination Color Images},
  author={Tian, Qi-Chong and Cohen, Laurent D.},
  booktitle={2017 IEEE International Conference on Computer Vision Workshop (ICCVW)},
  year={2017},
  month={Oct},
  }
```




