# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: prystine_detection/prystine/protos/input_reader.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from prystine_detection.prystine.protos import target_pb2 as prystine__detection_dot_prystine_dot_protos_dot_target__pb2
from prystine_detection.prystine.protos import preprocess_pb2 as prystine__detection_dot_prystine_dot_protos_dot_preprocess__pb2
from prystine_detection.prystine.protos import sampler_pb2 as prystine__detection_dot_prystine_dot_protos_dot_sampler__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='prystine_detection/prystine/protos/input_reader.proto',
  package='prystine_detection.prystine.protos',
  syntax='proto3',
  serialized_pb=_b('\n5prystine_detection/prystine/protos/input_reader.proto\x12\"prystine_detection.prystine.protos\x1a/prystine_detection/prystine/protos/target.proto\x1a\x33prystine_detection/prystine/protos/preprocess.proto\x1a\x30prystine_detection/prystine/protos/sampler.proto\"\xec\x07\n\x0bInputReader\x12\x12\n\nbatch_size\x18\x03 \x01(\r\x12\x16\n\x0emax_num_epochs\x18\x04 \x01(\r\x12\x15\n\rprefetch_size\x18\x05 \x01(\r\x12\x1c\n\x14max_number_of_voxels\x18\x06 \x01(\r\x12K\n\x0ftarget_assigner\x18\x07 \x01(\x0b\x32\x32.prystine_detection.prystine.protos.TargetAssigner\x12\x17\n\x0fkitti_info_path\x18\x08 \x01(\t\x12\x17\n\x0fkitti_root_path\x18\t \x01(\t\x12\x16\n\x0eshuffle_points\x18\n \x01(\x08\x12*\n\"groundtruth_localization_noise_std\x18\x0b \x03(\x02\x12*\n\"groundtruth_rotation_uniform_noise\x18\x0c \x03(\x02\x12%\n\x1dglobal_rotation_uniform_noise\x18\r \x03(\x02\x12$\n\x1cglobal_scaling_uniform_noise\x18\x0e \x03(\x02\x12\x1f\n\x17remove_unknown_examples\x18\x0f \x01(\x08\x12\x13\n\x0bnum_workers\x18\x10 \x01(\r\x12\x1d\n\x15\x61nchor_area_threshold\x18\x11 \x01(\x02\x12\"\n\x1aremove_points_after_sample\x18\x12 \x01(\x08\x12*\n\"groundtruth_points_drop_percentage\x18\x13 \x01(\x02\x12(\n groundtruth_drop_max_keep_points\x18\x14 \x01(\r\x12\x1a\n\x12remove_environment\x18\x15 \x01(\x08\x12\x1a\n\x12unlabeled_training\x18\x16 \x01(\x08\x12/\n\'global_random_rotation_range_per_object\x18\x17 \x03(\x02\x12Z\n\x13\x64\x61tabase_prep_steps\x18\x18 \x03(\x0b\x32=.prystine_detection.prystine.protos.DatabasePreprocessingStep\x12\x45\n\x10\x64\x61tabase_sampler\x18\x19 \x01(\x0b\x32+.prystine_detection.prystine.protos.Sampler\x12\x14\n\x0cuse_group_id\x18\x1a \x01(\x08\x12O\n\x1aunlabeled_database_sampler\x18\x1b \x01(\x0b\x32+.prystine_detection.prystine.protos.Samplerb\x06proto3')
  ,
  dependencies=[prystine__detection_dot_prystine_dot_protos_dot_target__pb2.DESCRIPTOR,prystine__detection_dot_prystine_dot_protos_dot_preprocess__pb2.DESCRIPTOR,prystine__detection_dot_prystine_dot_protos_dot_sampler__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_INPUTREADER = _descriptor.Descriptor(
  name='InputReader',
  full_name='prystine_detection.prystine.protos.InputReader',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='prystine_detection.prystine.protos.InputReader.batch_size', index=0,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='max_num_epochs', full_name='prystine_detection.prystine.protos.InputReader.max_num_epochs', index=1,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='prefetch_size', full_name='prystine_detection.prystine.protos.InputReader.prefetch_size', index=2,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='max_number_of_voxels', full_name='prystine_detection.prystine.protos.InputReader.max_number_of_voxels', index=3,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='target_assigner', full_name='prystine_detection.prystine.protos.InputReader.target_assigner', index=4,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='kitti_info_path', full_name='prystine_detection.prystine.protos.InputReader.kitti_info_path', index=5,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='kitti_root_path', full_name='prystine_detection.prystine.protos.InputReader.kitti_root_path', index=6,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='shuffle_points', full_name='prystine_detection.prystine.protos.InputReader.shuffle_points', index=7,
      number=10, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='groundtruth_localization_noise_std', full_name='prystine_detection.prystine.protos.InputReader.groundtruth_localization_noise_std', index=8,
      number=11, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='groundtruth_rotation_uniform_noise', full_name='prystine_detection.prystine.protos.InputReader.groundtruth_rotation_uniform_noise', index=9,
      number=12, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='global_rotation_uniform_noise', full_name='prystine_detection.prystine.protos.InputReader.global_rotation_uniform_noise', index=10,
      number=13, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='global_scaling_uniform_noise', full_name='prystine_detection.prystine.protos.InputReader.global_scaling_uniform_noise', index=11,
      number=14, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='remove_unknown_examples', full_name='prystine_detection.prystine.protos.InputReader.remove_unknown_examples', index=12,
      number=15, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_workers', full_name='prystine_detection.prystine.protos.InputReader.num_workers', index=13,
      number=16, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='anchor_area_threshold', full_name='prystine_detection.prystine.protos.InputReader.anchor_area_threshold', index=14,
      number=17, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='remove_points_after_sample', full_name='prystine_detection.prystine.protos.InputReader.remove_points_after_sample', index=15,
      number=18, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='groundtruth_points_drop_percentage', full_name='prystine_detection.prystine.protos.InputReader.groundtruth_points_drop_percentage', index=16,
      number=19, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='groundtruth_drop_max_keep_points', full_name='prystine_detection.prystine.protos.InputReader.groundtruth_drop_max_keep_points', index=17,
      number=20, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='remove_environment', full_name='prystine_detection.prystine.protos.InputReader.remove_environment', index=18,
      number=21, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='unlabeled_training', full_name='prystine_detection.prystine.protos.InputReader.unlabeled_training', index=19,
      number=22, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='global_random_rotation_range_per_object', full_name='prystine_detection.prystine.protos.InputReader.global_random_rotation_range_per_object', index=20,
      number=23, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='database_prep_steps', full_name='prystine_detection.prystine.protos.InputReader.database_prep_steps', index=21,
      number=24, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='database_sampler', full_name='prystine_detection.prystine.protos.InputReader.database_sampler', index=22,
      number=25, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='use_group_id', full_name='prystine_detection.prystine.protos.InputReader.use_group_id', index=23,
      number=26, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='unlabeled_database_sampler', full_name='prystine_detection.prystine.protos.InputReader.unlabeled_database_sampler', index=24,
      number=27, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=246,
  serialized_end=1250,
)

_INPUTREADER.fields_by_name['target_assigner'].message_type = prystine__detection_dot_prystine_dot_protos_dot_target__pb2._TARGETASSIGNER
_INPUTREADER.fields_by_name['database_prep_steps'].message_type = prystine__detection_dot_prystine_dot_protos_dot_preprocess__pb2._DATABASEPREPROCESSINGSTEP
_INPUTREADER.fields_by_name['database_sampler'].message_type = prystine__detection_dot_prystine_dot_protos_dot_sampler__pb2._SAMPLER
_INPUTREADER.fields_by_name['unlabeled_database_sampler'].message_type = prystine__detection_dot_prystine_dot_protos_dot_sampler__pb2._SAMPLER
DESCRIPTOR.message_types_by_name['InputReader'] = _INPUTREADER

InputReader = _reflection.GeneratedProtocolMessageType('InputReader', (_message.Message,), dict(
  DESCRIPTOR = _INPUTREADER,
  __module__ = 'prystine_detection.prystine.protos.input_reader_pb2'
  # @@protoc_insertion_point(class_scope:prystine_detection.prystine.protos.InputReader)
  ))
_sym_db.RegisterMessage(InputReader)


# @@protoc_insertion_point(module_scope)
