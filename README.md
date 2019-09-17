# Prystine 3D object detection

Project derived from  https://github.com/kujason/avod and  https://github.com/traveller59/second.pytorch.

Tested with python 3.6+, pytorch 1.1.0+.

## Install

### 1. Clone code

```bash
git clone https://github.com/michelecancilla/prystine.git
```
<!--cd ./second.pytorch/second-->

### 2. Install dependence python packages


```bash
pip install -r requirements.txt
```

Install spconv following the instructions [here](prystine_detection/spconv/README.md).
 

### 3. Setup cuda for numba

You have to add following environment variables for numba.cuda, you can add them to `~/.bashrc`:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

### 4. Add prystine directory to PYTHONPATH

## How to inference

### Dataset preparation

Get a [KITTI Raw](http://www.cvlibs.net/datasets/kitti/raw_data.php) recording, downloading sync, tracklets and calib (i.e. 2011_09_26_drive_0059) and extract it.
The directory should have the following structure:
```plain
└── KITTI_RAW_DATASET_ROOT
       ├── 2011_09_26_drive_0059_sync
       |   ├── image_00
       |   ├── image_01
       |   ├── image_02
       |   ├── image_03
       |   ├── oxts
       |   ├── velodyne_points
       |   └── tracklets_labels.txt
       ├── calib_cam_to_cam.txt
       ├── calib_imu_to_velo.txt
       └── calib_velo_to_cam.txt
```

### Test the detection network

- Download the pretrained weights [here](https://drive.google.com/file/d/1Ru1RO7JXeQ8sMQ5qAa1xih5AKWx-qzjh) and set 
the `ckpt_path` variable into `prystine_detection/prystine_inference.py` file.

- Change the following lines in  `prystine_detection/create_video.py`:
 ```
basedir = '/home/user/data/KittiRaw'
date = '2011_09_26'
drive = '0059'
```

#### Output example

![](prystine_detection/imgs/inference_example.gif)

### Test the tracking network
- Change the following lines in  `prystine_tracking/main.py`:
 ```
basedir = '/home/user/data/KittiRaw'
date = '2011_09_26'
drive = '0059'
```

- The tracking algorithm will call the detection network for every frame in the scene,
 calculating the trajectory of every object.

#### Output example

![](prystine_tracking/output/tracking.gif)

## Concepts

#### Kitti lidar box

A kitti lidar box consists of 7 elements: `[x, y, z, w, l, h, rz]` (see figure).

![Kitti Box Image](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/kittibox.png)

Both training and inference code use kitti box format. So we need to convert other format to KITTI format before training.

#### Kitti camera box

A kitti camera box consists of 7 elements: `[x, y, z, l, h, w, ry]`.
