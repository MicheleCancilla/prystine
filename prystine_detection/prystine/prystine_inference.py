import os
import torch
from google.protobuf import text_format

from prystine_detection.prystine.plot_utils import *
from prystine_detection.prystine.protos import pipeline_pb2
from prystine_detection.prystine.pytorch.train import build_network
from prystine_detection.prystine.utils import config_tool


class DetectorInference:
    def __init__(self):
        # Read Config file
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.config_path = f'{current_path}/configs/all.fhd.config'
        self.config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(self.config_path, 'r') as f:
            proto_str = f.read()
            text_format.Merge(proto_str, self.config)

        self.model_cfg = self.config.model.second

        config_tool.change_detection_range(self.model_cfg, [-50, -50, 50, 50], verbose=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        # Build Network, Target Assigner and Voxel Generator
        ckpt_path = '/nas/softechict-nas-1/mcancilla/projects/second1.5.1/all_fhd_backup/voxelnet-148480.tckpt'
        # ckpt_path = 'path/to/prystine-148480.tckpt'
        self.net = build_network(self.model_cfg).to(self.device).eval()
        self.net.load_state_dict(torch.load(ckpt_path))
        self.target_assigner = self.net.target_assigner
        self.voxel_generator = self.net.voxel_generator

    def inference(self, points, img, calib, postprocessing=True):
        # Generate Anchors
        grid_size = self.voxel_generator.grid_size
        feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(self.model_cfg)
        feature_map_size = [*feature_map_size, 1][::-1]

        anchors = self.target_assigner.generate_anchors(feature_map_size)['anchors']
        anchors = torch.tensor(anchors, dtype=torch.float32, device=self.device)
        anchors = anchors.view(1, -1, 7)

        # Keep only points in front of the camera
        points = points[points[:, 0] > 0, :]

        voxels, coords, num_points = self.voxel_generator.generate(points, max_voxels=90000)

        # add batch idx to coords
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)

        # Detection
        example = {
            'anchors': anchors,
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coords,
            'image': img,
        }
        pred = self.net(example)[0]
        boxes_lidar = pred['box3d_lidar'].detach().cpu().numpy()
        # Filter boxes with confidence <= 0.6
        boxes_lidar = boxes_lidar[pred['scores'].detach().cpu().numpy() > 0.6, :]
        labels = pred['label_preds'].detach().cpu().numpy()

        rect = calib.R_rect_00
        Trv2c = calib.T_cam0_velo

        boxes_lidar_camera = boxes_lidar.copy()
        boxes_lidar_camera = box_lidar_to_camera(boxes_lidar_camera, rect, Trv2c)

        if not postprocessing:
            return boxes_lidar, boxes_lidar_camera, labels

        locs = boxes_lidar[:, :3]
        dims = boxes_lidar[:, 3:6]
        angles = boxes_lidar[:, 6]
        camera_box_origin = [0.5, 0.5, 0.5]
        boxes_lidar = center_to_corner_box3d(locs, dims, angles, camera_box_origin, axis=2)
        locs = boxes_lidar_camera[:, :3]
        dims = boxes_lidar_camera[:, 3:6]
        angles = boxes_lidar_camera[:, 6]
        boxes_lidar_camera = center_to_corner_box3d(locs, dims, angles, camera_box_origin, axis=1)
        return boxes_lidar, boxes_lidar_camera, labels
