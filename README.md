# ObjectRemovalDetection

ObjectRemovalDetection is C++ library used to detect the changes that occurred
in a scene between an outdated 3D mesh of said-scene and a sequence of
up-to-date images. Particular attention is paid to the detection of object
removals.

## Installation

ObjectRemovalDetection expends on the work conducted by E. Palazzolo
at https://github.com/PRBonn/fast_change_detection. Download said project and
extract it into a `workfolder` and install the dependencies. If you are running
the library on windows, you do not need to install `catkin`. `Qt` is not
required by ObjectRemovalDetection.

To install and use ObjectRemovalDetection, download the project and extract it
into the same `workfolder` (replacing the headers and adding new source files).

Replace the existing `CMakeLists.txt` by either `CMakeLists-ubuntu.txt` or
`CMakeLists-windows.txt`. On Ubuntu, the installation procedure is the same as
https://github.com/PRBonn/fast_change_detection. On Windows you must edit the
`CMakeLists.txt` to find the dependencies and run the following commands:

```bash
cd workfolder
mkdir build
cd build
cmake ..
```

And then build the Visual Studio solution generated by cmake.

## Usage

### Datasets

E. Palazzolo's dataset is downloadable at http://www.ipb.uni-bonn.de/html/projects/changedetection2017/changedetection2017.zip

The ScanNet dataset is available at http://www.scan-net.org/

The 3D models used in our dataset are downloadable at:
* `shelf`- https://www.turbosquid.com/3d-models/3d-shelf-model-1548060
* `car`- https://www.turbosquid.com/3d-models/simple-car-model-1330846
* `robot`
* `stone`- https://www.turbosquid.com/3d-models/mountain-rock-pbr-8k-3d-model-1300107
* `plant`- https://www.turbosquid.com/3d-models/3d-plants-1528072
* `box`- https://www.turbosquid.com/3d-models/sci-fi-military-container-3d-model-1369994
* `statue`- https://www.turbosquid.com/3d-models/3d-statuette-sheep-barrel-model-1335035
* `dollhouse`- https://www.turbosquid.com/3d-models/3d-cartoon-house-1576949
* `table`- https://www.turbosquid.com/3d-models/table-04-model-1578760
* `chair`- https://www.turbosquid.com/3d-models/3d-leather-chair-black-model-1551213
* `extinguisher`- https://www.turbosquid.com/3d-models/3d-extinguisher-model-1447524
* `cat`- https://www.turbosquid.com/3d-models/low-polycatanimal-model-1340490
* `desklamp`- https://www.turbosquid.com/3d-models/3d-table-lamp-lights-v-ray-model-1522080
* `ghost`- https://www.turbosquid.com/3d-models/3d-ghost-model-1419900
* `bucket`- https://www.turbosquid.com/3d-models/3d-bucket-low-poly-1288741
* `pitcher`- https://www.turbosquid.com/3d-models/adid-porcelain-milk-jug-3d-1368224
* `lamp`- https://www.turbosquid.com/3d-models/lighting-fixtures-3d-1428616

### Examples

The `DATASET_PATH` work folder's contents must follow the conventions of E.
Palazzolo's dataset. It should contain:
* a `model.obj` file for the outdated mesh
* an `images` folder with `Image*.JPG` files for the up-to-date image sequence
* a `cameras.xml` file that contain the camera's intrisincs and the images' poses

Usage:
```bash
./change_detection[_shaders] DATASET_PATH [kernel_size] [max_comparisons] [rescale_width] [threshold_change_area] [threshold_change_value]
```

`kernel_size` is the windows size for the median filter used, 3 by default.

`max_comparisons` is the number of images in the sequence minus 1, 4 by default.

`rescale_width` is the width chosen to process the images, 500 by default, -1
keeps the images in their original size.

`threshold_change_area` is the area threshold under which 2D changes are
discarded, 50 by default (should be coherent with `rescale_width`).

`threshold_change_value` is the pixel value (1-255) threshold under which 2D
changes are discarded, -1 by default to use the automatic triangle threshold.

The `change_detection_shaders` example uses shaders instead of CPU code, only
for insertion detection.

## License

ObjectRemovalDetection is licensed under the Apache License, Version 2.0

## Authors

* Olivier Roupin, Matthieu Fradet, Caroline Baillard, Guillaume Moreau

## Citation

If you use this project, please cite the relevant original publications for the
models and datasets, and cite this project as:

```
@article{roupin2020ordetection,
  title={Detection of Removed Objects in 3D Meshes Using Up-to-Date Images for Mixed Reality Applications},
  author={Roupin, Olivier and Fradet, Matthieu and Baillard, Caroline and Moreau, Guillaume},
  year={2021},
  journal={Electronics},
  volume={10},
  number={4},
  article-number={377},
  url={https://www.mdpi.com/2079-9292/10/4/377},
  ISSN={2079-9292},
  DOI={10.3390/electronics10040377}
}
```

## Related links
