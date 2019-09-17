from prystine_detection.prystine.builder import target_assigner_builder, voxel_builder
from prystine_detection.prystine.pytorch.builder import (box_coder_builder, second_builder)


def build_network(model_cfg, measure_time=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)
    # class_names = target_assigner.classes
    net = second_builder.build(model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    return net
