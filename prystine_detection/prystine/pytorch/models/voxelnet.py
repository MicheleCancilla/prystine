import time

import numpy as np
import torch
from enum import Enum
from torch import nn
from torch.nn import functional as F

from prystine_detection.prystine.pytorch.core import box_torch_ops
from prystine_detection.prystine.pytorch.core.losses import (WeightedSigmoidClassificationLoss,
                                                             WeightedSmoothL1LocalizationLoss,
                                                             WeightedSoftmaxClassificationLoss)
from prystine_detection.prystine.pytorch.models import middle, rpn, voxel_encoder, resnet_50
from prystine_detection import torchplus
from prystine_detection.torchplus import metrics


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"
    DontNorm = "dont_norm"


class VoxelNet(nn.Module):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_class_name="VoxelFeatureExtractor",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="SparseMiddleExtractor",
                 middle_num_input_features=-1,
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 rpn_class_name="RPN",
                 rpn_num_input_features=-1,
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_sparse_rpn=False,
                 use_voxel_classifier=False,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_threshold=0.5,
                 nms_pre_max_size=1000,
                 nms_post_max_size=20,
                 nms_iou_threshold=0.1,
                 target_assigner=None,
                 use_bev=False,
                 use_fextractor=False,
                 use_rc_net=False,
                 lidar_only=False,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 measure_time=False,
                 voxel_generator=None,
                 name='voxelnet'):
        super().__init__()
        self.name = name
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_threshold = nms_score_threshold
        self._nms_pre_max_size = nms_pre_max_size
        self._nms_post_max_size = nms_post_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_sparse_rpn = use_sparse_rpn
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._num_input_features = num_input_features
        self._box_coder = target_assigner.box_coder
        self._use_rc_net = use_rc_net
        self._lidar_only = lidar_only
        self.target_assigner = target_assigner
        self.voxel_generator = voxel_generator
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()
        self._vox_loss_ftor = WeightedSoftmaxClassificationLoss()
        self._rc_loss_ftor = WeightedSigmoidClassificationLoss()
        self._diff_loc_loss_ftor = WeightedSmoothL1LocalizationLoss()
        self._use_fextractor = use_fextractor
        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight
        self.measure_time = measure_time
        vfe_class_dict = {
            "VoxelFeatureExtractor": voxel_encoder.VoxelFeatureExtractor,
            "VoxelFeatureExtractorV2": voxel_encoder.VoxelFeatureExtractorV2,
            "VoxelFeatureExtractorV3": voxel_encoder.VoxelFeatureExtractorV3,
            "SimpleVoxel": voxel_encoder.SimpleVoxel
        }
        # self.fextractor = resnet_50.resnet101(True, False)
        vfe_class = vfe_class_dict[vfe_class_name]
        self.voxel_feature_extractor = vfe_class(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance)
        if len(middle_num_filters_d2) == 0:
            if len(middle_num_filters_d1) == 0:
                num_rpn_input_filters = vfe_num_filters[-1]
            else:
                num_rpn_input_filters = middle_num_filters_d1[-1]
        else:
            num_rpn_input_filters = middle_num_filters_d2[-1]

        if use_sparse_rpn:
            pass
        else:
            mid_class_dict = {
                "SpMiddleFHD": middle.SpMiddleFHD,
            }
            mid_class = mid_class_dict[middle_class_name]
            self.middle_feature_extractor = mid_class(
                output_shape,
                use_norm,
                num_input_features=middle_num_input_features,
                num_filters_down1=middle_num_filters_d1,
                num_filters_down2=middle_num_filters_d2)
            rpn_class_dict = {
                "RPNV2": rpn.RPNV2,
            }
            rpn_class = rpn_class_dict[rpn_class_name]
            self.rpn = rpn_class(
                use_norm=True,
                num_class=num_class,
                layer_nums=rpn_layer_nums,
                layer_strides=rpn_layer_strides,
                num_filters=rpn_num_filters,
                upsample_strides=rpn_upsample_strides,
                num_upsample_filters=rpn_num_upsample_filters,
                num_input_features=rpn_num_input_features,
                num_anchor_per_loc=target_assigner.num_anchors_per_location,
                encode_background_as_zeros=encode_background_as_zeros,
                use_direction_classifier=use_direction_classifier,
                use_bev=use_bev,
                use_groupnorm=use_groupnorm,
                num_groups=num_groups,
                box_code_size=target_assigner.box_coder.code_size,
                use_rc_net=use_rc_net)
        if use_voxel_classifier:
            pass
        else:
            self.voxel_classifier = None

        self.rpn_acc = metrics.Accuracy(dim=-1, encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=use_sigmoid_score,
            encode_background_as_zeros=encode_background_as_zeros)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.register_buffer("global_step", torch.LongTensor(1).zero_())

        self._time_dict = {}
        self._time_total_dict = {}
        self._time_count_dict = {}

    def start_timer(self, *names):
        if not self.measure_time:
            return
        for name in names:
            self._time_dict[name] = time.time()
        torch.cuda.synchronize()

    def end_timer(self, name):
        if not self.measure_time:
            return
        torch.cuda.synchronize()
        time_elapsed = time.time() - self._time_dict[name]
        if name not in self._time_count_dict:
            self._time_count_dict[name] = 1
            self._time_total_dict[name] = time_elapsed
        else:
            self._time_count_dict[name] += 1
            self._time_total_dict[name] += time_elapsed
        self._time_dict[name] = 0

    def clear_timer(self):
        self._time_count_dict.clear()
        self._time_dict.clear()
        self._time_total_dict.clear()

    def get_avg_time_dict(self):
        ret = {}
        for name, val in self._time_total_dict.items():
            count = self._time_count_dict[name]
            ret[name] = val / max(1, count)
        return ret

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def forward(self, example):
        """module's forward should always accept dict and return loss.
        """
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        batch_anchors = example["anchors"]
        batch_size_dev = batch_anchors.shape[0]
        # t = time.time()
        # features: [num_voxels, max_num_points_per_voxel, 7]
        # num_points: [num_voxels]
        # coors: [num_voxels, 4]
        self.start_timer("voxel_feature_extractor")
        voxel_features = self.voxel_feature_extractor(voxels, num_points)
        # if self._use_fextractor:
        #     spatial_features = self.fextractor(example['image'])
        self.end_timer("voxel_feature_extractor")

        if self._use_sparse_rpn:
            preds_dict = self.sparse_rpn(voxel_features, coors, batch_size_dev)
        else:
            self.start_timer("middle forward")
            spatial_features = self.middle_feature_extractor(voxel_features, coors, batch_size_dev)
            self.end_timer("middle forward")
            self.start_timer("rpn forward")
            if self._use_bev:
                preds_dict = self.rpn(spatial_features, example["bev_map"])
            else:
                preds_dict = self.rpn(spatial_features)
            self.end_timer("rpn forward")
        if self.voxel_classifier is not None:
            preds_dict["voxel_logits"] = self.voxel_classifier(voxel_features, coors, batch_size_dev)
        # preds_dict["voxel_features"] = voxel_features
        # preds_dict["spatial_features"] = spatial_features
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]
        if self.training:
            pass
        else:
            self.start_timer("predict")
            with torch.no_grad():
                res = self.predict(example, preds_dict)
            self.end_timer("predict")
            return res

    def predict(self, example, preds_dict):
        batch_size = example['anchors'].shape[0]
        if "metadata" in example:
            meta_list = example["metadata"]
        else:
            meta_list = [None] * batch_size
        batch_anchors = example["anchors"].view(batch_size, -1, 7)
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)

        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1

        batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds, batch_anchors)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        for box_preds, cls_preds, dir_preds, a_mask, meta in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds,
                batch_anchors_mask, meta_list):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            if self._use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms

            if self._multiclass_nms:
                pass
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long)

                else:
                    top_scores, top_labels = torch.max(total_scores, dim=-1)

                if self._nms_score_threshold > 0.0:
                    thresh = torch.tensor(
                        [self._nms_score_threshold],
                        device=total_scores.device).type_as(total_scores)
                    top_scores_keep = (top_scores >= thresh)
                    top_scores = top_scores.masked_select(top_scores_keep)

                if top_scores.shape[0] != 0:
                    if self._nms_score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    if not self._use_rotate_nms:
                        box_preds_corners = box_torch_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4])
                        boxes_for_nms = box_torch_ops.corner_to_standup_nd(box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_size,
                        post_max_size=self._nms_post_max_size,
                        iou_threshold=self._nms_iou_threshold,
                    )
                else:
                    selected = []
                # if selected is not None:
                selected_boxes = box_preds[selected]
                if self._use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                # predictions
                predictions_dict = {
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "metadata": meta,
                }
            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "box3d_lidar": torch.zeros([0, 7], dtype=dtype, device=device),
                    "scores": torch.zeros([0], dtype=dtype, device=device),
                    "label_preds": torch.zeros([0, 4], dtype=top_labels.dtype, device=device),
                    "metadata": meta,
                }
            predictions_dicts.append(predictions_dict)
        if "calib" in example:
            batch_rect = example["calib"]["rect"]
            batch_Trv2c = example["calib"]["Trv2c"]
            batch_P2 = example["calib"]["P2"]
            for pred_dict, rect, Trv2c, P2 in zip(predictions_dicts,
                                                  batch_rect, batch_Trv2c, batch_P2):
                final_box_preds = pred_dict["box3d_lidar"]
                if final_box_preds.shape[0] != 0:
                    final_box_preds[:, 2] -= final_box_preds[:, 5] / 2
                    final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                        final_box_preds, rect, Trv2c)

                    locs = final_box_preds_camera[:, :3]
                    dims = final_box_preds_camera[:, 3:6]
                    angles = final_box_preds_camera[:, 6]
                    camera_box_origin = [0.5, 1.0, 0.5]
                    box_corners = box_torch_ops.center_to_corner_box3d(
                        locs, dims, angles, camera_box_origin, axis=1)
                    box_corners_in_image = box_torch_ops.project_to_image(
                        box_corners, P2)
                    # box_corners_in_image: [N, 8, 2]
                    minxy = torch.min(box_corners_in_image, dim=1)[0]
                    maxxy = torch.max(box_corners_in_image, dim=1)[0]
                    box_2d_preds = torch.cat([minxy, maxxy], dim=1)
                    pred_dict["bbox"] = box_2d_preds
                    pred_dict["box3d_camera"] = final_box_preds_camera
                else:
                    pred_dict["bbox"] = torch.zeros([0, 4], dtype=dtype, device=device)
                    pred_dict["box3d_camera"] = torch.zeros([0, 7], dtype=dtype, device=device)

        return predictions_dicts

    def metrics_to_float(self):
        self.rpn_acc.float()
        self.rpn_metrics.float()
        self.rpn_cls_loss.float()
        self.rpn_loc_loss.float()
        self.rpn_total_loss.float()
        if self.voxel_classifier is not None:
            self.vox_acc.float()
            self.vox_precision.float()
            self.vox_recall.float()

    def update_metrics(self,
                       cls_loss,
                       loc_loss,
                       cls_preds,
                       labels,
                       sampled,
                       vox_preds=None,
                       vox_labels=None,
                       vox_weights=None):
        batch_size = cls_preds.shape[0]
        num_class = self._num_class
        if not self._encode_background_as_zeros:
            num_class += 1
        cls_preds = cls_preds.view(batch_size, -1, num_class)
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
        prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        prec = prec.numpy()
        recall = recall.numpy()
        rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
        rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
        ret = {
            "cls_loss": float(rpn_cls_loss),
            "cls_loss_rt": float(cls_loss.data.cpu().numpy()),
            'loc_loss': float(rpn_loc_loss),
            "loc_loss_rt": float(loc_loss.data.cpu().numpy()),
            "rpn_acc": float(rpn_acc),
        }
        for i, thresh in enumerate(self.rpn_metrics.thresholds):
            ret[f"prec@{int(thresh * 100)}"] = float(prec[i])
            ret[f"rec@{int(thresh * 100)}"] = float(recall[i])
        if self.voxel_classifier is not None:
            vox_acc = self.vox_acc(
                vox_labels, vox_preds.view(-1, 2),
                weights=vox_weights > 0).numpy()[0]
            vox_prec = self.vox_precision(
                vox_labels, vox_preds.view(-1, 2),
                weights=vox_weights > 0).numpy()[0]
            vox_recall = self.vox_recall(
                vox_labels, vox_preds.view(-1, 2),
                weights=vox_weights > 0).numpy()[0]

            ret.update({
                "vox_acc": float(vox_acc),
                "vox_prec": float(vox_prec),
                "vox_recall": float(vox_recall),
            })
        return ret

    def clear_metrics(self):
        self.rpn_acc.clear()
        self.rpn_metrics.clear()
        self.rpn_cls_loss.clear()
        self.rpn_loc_loss.clear()
        self.rpn_total_loss.clear()
        if self.voxel_classifier is not None:
            self.vox_acc.clear()
            self.vox_precision.clear()
            self.vox_recall.clear()
        pass

    @staticmethod
    def convert_norm_to_float(net):
        '''
        BatchNorm layers to have parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        '''
        if isinstance(net, torch.nn.modules.batchnorm._BatchNorm):
            net.float()
        for child in net.children():
            VoxelNet.convert_norm_to_float(child)
        return net


#
def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                box_code_size=7):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = torchplus.nn.one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses


def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         loss_norm_type=LossNormType.NormByNumPositives,
                         dtype=torch.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.type(dtype).sum(1, keepdim=True)
        num_examples = torch.clamp(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    elif loss_norm_type == LossNormType.DontNorm:  # support ghm loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        # pass
    else:
        raise ValueError(
            f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared


#
# def assign_weight_to_each_class(labels,
#                                 weight_per_class,
#                                 norm_by_num=True,
#                                 dtype=torch.float32):
#     weights = torch.zeros(labels.shape, dtype=dtype, device=labels.device)
#     for label, weight in weight_per_class:
#         positives = (labels == label).type(dtype)
#         weight_class = weight * positives
#         if norm_by_num:
#             normalizer = positives.sum()
#             normalizer = torch.clamp(normalizer, min=1.0)
#             weight_class /= normalizer
#         weights += weight_class
#     return weights
#
#
def get_direction_target(anchors, reg_targets, one_hot=True):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, 7)
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = (rot_gt > 0).long()
    if one_hot:
        dir_cls_targets = torchplus.nn.one_hot(dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets
