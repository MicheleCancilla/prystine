# Copyright 2017 yanyan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
VoxelNet builder.
"""

from prystine_detection.prystine.protos import prystine_pb2
from prystine_detection.prystine.pytorch.builder import losses_builder
from prystine_detection.prystine.pytorch.models.voxelnet import LossNormType, VoxelNet


def build(model_cfg: prystine_pb2.VoxelNet, voxel_generator,
          target_assigner, measure_time=False) -> VoxelNet:
    """build second pytorch instance.
    """
    if not isinstance(model_cfg, prystine_pb2.VoxelNet):
        raise ValueError('model_cfg not of type ' 'prystine_pb2.VoxelNet.')
    vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)
    vfe_with_distance = model_cfg.voxel_feature_extractor.with_distance
    grid_size = voxel_generator.grid_size
    dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    num_class = len(target_assigner.classes)

    num_input_features = model_cfg.num_point_features
    if model_cfg.without_reflectivity:
        num_input_features = 3
    loss_norm_type_dict = {
        # 0: LossNormType.NormByNumExamples,
        1: LossNormType.NormByNumPositives,
        # 2: LossNormType.NormByNumPosNeg,
        # 3: LossNormType.DontNorm,
    }
    loss_norm_type = loss_norm_type_dict[model_cfg.loss_norm_type]

    losses = losses_builder.build(model_cfg.loss)
    encode_rad_error_by_sin = model_cfg.encode_rad_error_by_sin
    cls_loss_ftor, loc_loss_ftor, cls_weight, loc_weight, _ = losses
    pos_cls_weight = model_cfg.pos_class_weight
    neg_cls_weight = model_cfg.neg_class_weight
    direction_loss_weight = model_cfg.direction_loss_weight

    net = VoxelNet(
        dense_shape,
        num_class=num_class,
        vfe_class_name=model_cfg.voxel_feature_extractor.module_class_name,
        vfe_num_filters=vfe_num_filters,
        middle_class_name=model_cfg.middle_feature_extractor.module_class_name,
        middle_num_input_features=model_cfg.middle_feature_extractor.num_input_features,
        middle_num_filters_d1=list(
            model_cfg.middle_feature_extractor.num_filters_down1),
        middle_num_filters_d2=list(
            model_cfg.middle_feature_extractor.num_filters_down2),
        rpn_class_name=model_cfg.rpn.module_class_name,
        rpn_num_input_features=model_cfg.rpn.num_input_features,
        rpn_layer_nums=list(model_cfg.rpn.layer_nums),
        rpn_layer_strides=list(model_cfg.rpn.layer_strides),
        rpn_num_filters=list(model_cfg.rpn.num_filters),
        rpn_upsample_strides=list(model_cfg.rpn.upsample_strides),
        rpn_num_upsample_filters=list(model_cfg.rpn.num_upsample_filters),
        use_norm=True,
        use_voxel_classifier=model_cfg.use_aux_classifier,
        use_rotate_nms=model_cfg.use_rotate_nms,
        multiclass_nms=model_cfg.use_multi_class_nms,
        nms_score_threshold=model_cfg.nms_score_threshold,
        nms_pre_max_size=model_cfg.nms_pre_max_size,
        nms_post_max_size=model_cfg.nms_post_max_size,
        nms_iou_threshold=model_cfg.nms_iou_threshold,
        use_sigmoid_score=model_cfg.use_sigmoid_score,
        use_sparse_rpn=False,
        encode_background_as_zeros=model_cfg.encode_background_as_zeros,
        use_direction_classifier=model_cfg.use_direction_classifier,
        use_bev=model_cfg.use_bev,
        num_input_features=num_input_features,
        num_groups=model_cfg.rpn.num_groups,
        use_groupnorm=model_cfg.rpn.use_groupnorm,
        with_distance=vfe_with_distance,
        cls_loss_weight=cls_weight,
        loc_loss_weight=loc_weight,
        pos_cls_weight=pos_cls_weight,
        neg_cls_weight=neg_cls_weight,
        direction_loss_weight=direction_loss_weight,
        loss_norm_type=loss_norm_type,
        encode_rad_error_by_sin=encode_rad_error_by_sin,
        loc_loss_ftor=loc_loss_ftor,
        cls_loss_ftor=cls_loss_ftor,
        target_assigner=target_assigner,
        measure_time=measure_time,
        voxel_generator=voxel_generator,
    )
    return net
