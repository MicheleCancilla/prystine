import datetime

import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from moviepy.editor import ImageSequenceClip
from mpl_toolkits.mplot3d import Axes3D

from prystine_detection.prystine.kitti_utils import *
from prystine_detection.prystine.plot_utils import *
from prystine_detection.prystine.prystine_inference import DetectorInference

colors = {
    'Car': 'b',
    'Tram': 'r',
    'Cyclist': 'g',
    'Van': 'c',
    'Truck': 'm',
    'Pedestrian': 'y',
    'Sitter': 'k'
}

CLASS_NAMES = [
    'Car', 'Cyclist', 'Pedestrian', 'Van', 'Sitter', 'car', 'tractor', 'trailer',
]

axes_limits = [
    [-20, 80],  # X axis range
    [-20, 20],  # Y axis range
    [-3, 10]  # Z axis range
]


def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.

    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]

    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=1.5)


def draw_3d_plot(frame, dataset, velo, img, boxes_lidar, boxes_lidar_camera, boxes_type, points=0.2, path='video'):
    """
    Saves a single frame for an animation: a 2D/3D plot of the network predictions.
    Parameters
    ----------
    frame               : Absolute number of the frame.
    dataset             : `raw` dataset.
    velo                : Point cloud
    img                 : Input image
    boxes_lidar         : List with bounding boxes coordinates.
    boxes_lidar_camera  : List with bounding boxes in the camera coordinates.
    boxes_type          : List with objects types.
    points              : Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.

    Returns
    -------
    Saved frame filename.
    """

    f = plt.figure(figsize=(19.2, 10.8), dpi=100)
    gs = GridSpec(2, 1, figure=f, height_ratios=[.3, 1], wspace=0., hspace=0.)
    ax_im = f.add_subplot(gs[0])

    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    velo_range = range(0, velo.shape[0], points_step)
    velo_frame = velo[velo_range, :]

    img = np.array(img)

    ax = f.add_subplot(gs[1], projection='3d', xticks=[], yticks=[], zticks=[])

    ax.scatter(*np.transpose(velo_frame[:, [0, 1, 2]]), s=point_size, c=velo_frame[:, 3], cmap='gray')
    ax.set_xlim3d(*axes_limits[0])
    ax.set_ylim3d(*axes_limits[1])
    ax.set_zlim3d(*axes_limits[2])
    ax.view_init(elev=60, azim=180)
    ax.patch.set_visible(False)

    for box, box_cam, box_type in zip(boxes_lidar, boxes_lidar_camera, boxes_type):
        box = box.T  # for inference
        box_cam = box_cam.T  # for inference

        draw_box(ax, box, axes=[0, 1, 2], color=colors[box_type])

        # In order to transform a homogeneous point X = [x y z 1]' from the velodyne
        # coordinate system to a homogeneous point Y = [u v 1]' on image plane of
        # camera xx, the following transformation has to be applied:
        #
        # Y = P_rect_xx * R_rect_00 * (R|T)_velo_to_cam * X

        # Don't draw in image bbox behind the camera
        if not all(box[0, :] > 0):
            continue
        t_rects_2d = dataset.calib.P_rect_20 @ np.row_stack((box_cam, np.ones(box_cam.shape[1])))
        # To cartesian coordinate
        t_rects_2d = (t_rects_2d / t_rects_2d[2, :])[:-1, :]
        if all(np.logical_and(t_rects_2d[0] <= img.shape[1], t_rects_2d[0] >= 0)):
            # Draw only if all points relies on image plane
            draw_box(ax_im, t_rects_2d, axes=[0, 1], color=colors[box_type])
            # ax_im.text(*t_rects_2d[:, 1], 'test', color='green')

    ax_im.imshow(img)
    ax_im.axis('off')
    # plt.show()
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'{path}/frame_{frame:0>4}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close(f)
    return filename


if __name__ == '__main__':
    # Change this to the directory where you store KITTI data
    basedir = '/homes/mcancilla/data/KittiRaw'
    date = '2011_09_26'
    drive = '0059'
    dataset = load_dataset(basedir, date, drive)

    frames = []
    n_frames = len(list(dataset.velo))
    time = datetime.datetime.now().strftime("%H_%M_%S")
    out_path = f'{basedir}/frames_{date}_{drive}_{time}'
    dataset_velo = list(dataset.velo)
    dataset_rgb = list(dataset.rgb)

    print('Preparing animation frames...')
    detector = DetectorInference()
    # for i in tqdm(range(n_frames)):
    for i in (range(n_frames)):
        velo = dataset_velo[i]
        img = dataset_rgb[i][0]  # Left camera rgb

        # boxes_lidar, boxes_lidar_camera, labels = inference(velo, img, dataset.calib, postprocessing=False)
        boxes_lidar, boxes_lidar_camera, labels = detector.inference(velo, img, dataset.calib)
        print(f'Inference {i + 1} done.')
        types = [CLASS_NAMES[i] for i in labels]
        filename = draw_3d_plot(i, dataset, velo, img, boxes_lidar, boxes_lidar_camera, types, path=out_path)
        frames += [filename]

    print('...Animation frames ready.')
    clip = ImageSequenceClip(frames, fps=5)
    clip.write_gif(f'{out_path}/pcl_data.gif', fps=5)
    # clip.write_gif(f'{out_path}/pcl_data_20.gif', fps=20)
    clip.write_videofile(f'{out_path}/pcl_data.mp4', fps=5)
