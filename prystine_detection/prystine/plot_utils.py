import numpy as np


def compute_box_corners_3d(object_label):
    """Computes the 3D bounding box corner positions from an ObjectLabel

    :param object_label: ObjectLabel to compute corners from
    :return: a numpy array of 3D corners if the box is in front of the camera,
             an empty array otherwise
    """

    # Compute rotational matrix
    rot = np.array([[+np.cos(object_label[6]), 0, +np.sin(object_label[6])],
                    [0, 1, 0],
                    [-np.sin(object_label[6]), 0, +np.cos(object_label[6])]])

    l = object_label[3]
    w = object_label[4]
    h = object_label[5]

    # 3D BB corners
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))

    corners_3d[0, :] = corners_3d[0, :] + object_label[0]
    corners_3d[1, :] = corners_3d[1, :] + object_label[1]
    corners_3d[2, :] = corners_3d[2, :] + object_label[2]

    return corners_3d


def project_to_image(point_cloud, p):
    """ Projects a 3D point cloud to 2D points for plotting

    :param point_cloud: 3D point cloud (3, N)
    :param p: Camera matrix (3, 4)

    :return: pts_2d: the image coordinates of the 3D points in the shape (2, N)
    """

    pts_2d = np.dot(p, np.append(point_cloud, np.ones((1, point_cloud.shape[1])), axis=0))

    pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
    pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
    pts_2d = np.delete(pts_2d, 2, 0)
    return pts_2d


def compute_orientation_3d(obj, p):
    """Computes the orientation given object and camera matrix

    Keyword arguments:
    obj -- object file to draw bounding box
    p -- transform matrix
    """

    # compute rotational matrix
    rot = np.array([[+np.cos(obj[6]), 0, +np.sin(obj[6])],
                    [0, 1, 0],
                    [-np.sin(obj[6]), 0, +np.cos(obj[6])]])

    orientation3d = np.array([0.0, obj[3], 0.0, 0.0, 0.0, 0.0]).reshape(3, 2)
    orientation3d = np.dot(rot, orientation3d)

    orientation3d[0, :] = orientation3d[0, :] + obj[0]
    orientation3d[1, :] = orientation3d[1, :] + obj[1]
    orientation3d[2, :] = orientation3d[2, :] + obj[2]

    # only draw for boxes that are in front of the camera
    for idx in np.arange(orientation3d.shape[1]):
        if orientation3d[2, idx] < 0.1:
            return None

    return project_to_image(orientation3d, p)


def draw_box_3d(ax, obj, p, show_orientation=True,
                color_table=None, line_width=3, double_line=True,
                box_color=None, img_shape=None):
    """Draws the 3D boxes given the subplot, object label,
    and frame transformation matrix

    :param ax: subplot handle
    :param obj: object file to draw bounding box
    :param p:stereo frame transformation matrix

    :param show_orientation: optional, draw a line showing orientaion
    :param color_table: optional, a custom table for coloring the boxes,
        should have 4 values to match the 4 truncation values. This color
        scheme is used to display boxes colored based on difficulty.
    :param line_width: optional, custom line width to draw the box
    :param double_line: optional, overlays a thinner line inside the box lines
    :param box_color: optional, use a custom color for box (instead of
        the default color_table.
    """

    face_idx = np.array([0, 1, 5, 4,  # front face
                         1, 2, 6, 5,  # left face
                         2, 3, 7, 6,  # back face
                         3, 0, 4, 7]).reshape((4, 4))  # right face

    # corners3d = compute_box_corners_3d(obj)
    # corners = project_to_image(corners3d, p)
    corners = project_to_image(obj, p)
    # if not (0 < corners[0, :] < img_shape[0]) or not (0 < corners[1, :] < img_shape[1]):
    # if any(0 < corners[0, :] < img_shape[0]) or not (0 < corners[1, :] < img_shape[1]):
    #     return

    image_filter = (corners[0, :] >= 0) & \
                   (corners[0, :] <= img_shape[1]) & \
                   (corners[1, :] >= 0) & \
                   (corners[1, :] <= img_shape[0])
    # corners = corners[:, image_filter]
    if corners.shape[1] != 8:
        return
    # define colors
    if color_table:
        if len(color_table) != 4:
            raise ValueError('Invalid color table length, must be 4')
    else:
        color_table = ["#00cc00", 'y', 'r', 'w']

    trun_style = ['solid', 'dashed']
    # trc = int(obj.truncation > 0.1)
    trc = 0

    if len(corners) > 0:
        for i in range(4):
            x = np.append(corners[0, face_idx[i]], corners[0, face_idx[i, 0]])
            y = np.append(corners[1, face_idx[i]], corners[1, face_idx[i, 0]])

            # Draw the boxes
            if box_color is None:
                box_color = color_table[2]

            ax.plot(x, y, linewidth=line_width,
                    color=box_color,
                    linestyle=trun_style[trc])

            # Draw a thinner second line inside
            if double_line:
                ax.plot(x, y, linewidth=line_width / 3.0, color='b')

    if show_orientation:
        # Compute orientation 3D
        orientation = compute_orientation_3d(obj, p)

        if orientation is not None:
            x = np.append(orientation[0], orientation[0])
            y = np.append(orientation[1], orientation[1])

            # draw the boxes
            ax.plot(x, y, linewidth=4, color='w')
            ax.plot(x, y, linewidth=2, color='k')


def lidar_to_camera(points, r_rect, velo2cam):
    # points_shape = list(points.shape[:-1])
    # if points.shape[-1] == 3:
    #     points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    # camera_points = points @ velo2cam.T
    # return camera_points[..., :3]

    points_shape = list(points.shape[:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = points @ (r_rect @ velo2cam).T
    return camera_points[..., :3]


def box_lidar_to_camera(data, r_rect, velo2cam):
    xyz_lidar = data[:, 0:3]
    w, l, h = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return np.concatenate([xyz, l, h, w, r], axis=1)


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=(0.5, 0.5, 0.5),
                           axis=2):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners
