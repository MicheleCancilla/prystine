# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: prystine_detection/prystine/protos/optimizer.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='prystine_detection/prystine/protos/optimizer.proto',
  package='prystine_detection.prystine.protos',
  syntax='proto3',
  serialized_pb=_b('\n2prystine_detection/prystine/protos/optimizer.proto\x12\"prystine_detection.prystine.protos\"\xe4\x02\n\tOptimizer\x12R\n\x12rms_prop_optimizer\x18\x01 \x01(\x0b\x32\x34.prystine_detection.prystine.protos.RMSPropOptimizerH\x00\x12S\n\x12momentum_optimizer\x18\x02 \x01(\x0b\x32\x35.prystine_detection.prystine.protos.MomentumOptimizerH\x00\x12K\n\x0e\x61\x64\x61m_optimizer\x18\x03 \x01(\x0b\x32\x31.prystine_detection.prystine.protos.AdamOptimizerH\x00\x12\x1a\n\x12use_moving_average\x18\x04 \x01(\x08\x12\x1c\n\x14moving_average_decay\x18\x05 \x01(\x02\x12\x1a\n\x12\x66ixed_weight_decay\x18\x06 \x01(\x08\x42\x0b\n\toptimizer\"\xb3\x01\n\x10RMSPropOptimizer\x12G\n\rlearning_rate\x18\x01 \x01(\x0b\x32\x30.prystine_detection.prystine.protos.LearningRate\x12 \n\x18momentum_optimizer_value\x18\x02 \x01(\x02\x12\r\n\x05\x64\x65\x63\x61y\x18\x03 \x01(\x02\x12\x0f\n\x07\x65psilon\x18\x04 \x01(\x02\x12\x14\n\x0cweight_decay\x18\x05 \x01(\x02\"\x94\x01\n\x11MomentumOptimizer\x12G\n\rlearning_rate\x18\x01 \x01(\x0b\x32\x30.prystine_detection.prystine.protos.LearningRate\x12 \n\x18momentum_optimizer_value\x18\x02 \x01(\x02\x12\x14\n\x0cweight_decay\x18\x03 \x01(\x02\"\x7f\n\rAdamOptimizer\x12G\n\rlearning_rate\x18\x01 \x01(\x0b\x32\x30.prystine_detection.prystine.protos.LearningRate\x12\x14\n\x0cweight_decay\x18\x02 \x01(\x02\x12\x0f\n\x07\x61msgrad\x18\x03 \x01(\x08\"\xa9\x01\n\x0cLearningRate\x12\x45\n\x0bmulti_phase\x18\x01 \x01(\x0b\x32..prystine_detection.prystine.protos.MultiPhaseH\x00\x12\x41\n\tone_cycle\x18\x02 \x01(\x0b\x32,.prystine_detection.prystine.protos.OneCycleH\x00\x42\x0f\n\rlearning_rate\"U\n\x11LearningRatePhase\x12\r\n\x05start\x18\x01 \x01(\x02\x12\x13\n\x0blambda_func\x18\x02 \x01(\t\x12\x1c\n\x14momentum_lambda_func\x18\x03 \x01(\t\"S\n\nMultiPhase\x12\x45\n\x06phases\x18\x01 \x03(\x0b\x32\x35.prystine_detection.prystine.protos.LearningRatePhase\"O\n\x08OneCycle\x12\x0e\n\x06lr_max\x18\x01 \x01(\x02\x12\x0c\n\x04moms\x18\x02 \x03(\x02\x12\x12\n\ndiv_factor\x18\x03 \x01(\x02\x12\x11\n\tpct_start\x18\x04 \x01(\x02\x62\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_OPTIMIZER = _descriptor.Descriptor(
  name='Optimizer',
  full_name='prystine_detection.prystine.protos.Optimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rms_prop_optimizer', full_name='prystine_detection.prystine.protos.Optimizer.rms_prop_optimizer', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='momentum_optimizer', full_name='prystine_detection.prystine.protos.Optimizer.momentum_optimizer', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='adam_optimizer', full_name='prystine_detection.prystine.protos.Optimizer.adam_optimizer', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='use_moving_average', full_name='prystine_detection.prystine.protos.Optimizer.use_moving_average', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='moving_average_decay', full_name='prystine_detection.prystine.protos.Optimizer.moving_average_decay', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='fixed_weight_decay', full_name='prystine_detection.prystine.protos.Optimizer.fixed_weight_decay', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
    _descriptor.OneofDescriptor(
      name='optimizer', full_name='prystine_detection.prystine.protos.Optimizer.optimizer',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=91,
  serialized_end=447,
)


_RMSPROPOPTIMIZER = _descriptor.Descriptor(
  name='RMSPropOptimizer',
  full_name='prystine_detection.prystine.protos.RMSPropOptimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='prystine_detection.prystine.protos.RMSPropOptimizer.learning_rate', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='momentum_optimizer_value', full_name='prystine_detection.prystine.protos.RMSPropOptimizer.momentum_optimizer_value', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='decay', full_name='prystine_detection.prystine.protos.RMSPropOptimizer.decay', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='epsilon', full_name='prystine_detection.prystine.protos.RMSPropOptimizer.epsilon', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weight_decay', full_name='prystine_detection.prystine.protos.RMSPropOptimizer.weight_decay', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=450,
  serialized_end=629,
)


_MOMENTUMOPTIMIZER = _descriptor.Descriptor(
  name='MomentumOptimizer',
  full_name='prystine_detection.prystine.protos.MomentumOptimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='prystine_detection.prystine.protos.MomentumOptimizer.learning_rate', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='momentum_optimizer_value', full_name='prystine_detection.prystine.protos.MomentumOptimizer.momentum_optimizer_value', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weight_decay', full_name='prystine_detection.prystine.protos.MomentumOptimizer.weight_decay', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=632,
  serialized_end=780,
)


_ADAMOPTIMIZER = _descriptor.Descriptor(
  name='AdamOptimizer',
  full_name='prystine_detection.prystine.protos.AdamOptimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='prystine_detection.prystine.protos.AdamOptimizer.learning_rate', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weight_decay', full_name='prystine_detection.prystine.protos.AdamOptimizer.weight_decay', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='amsgrad', full_name='prystine_detection.prystine.protos.AdamOptimizer.amsgrad', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=782,
  serialized_end=909,
)


_LEARNINGRATE = _descriptor.Descriptor(
  name='LearningRate',
  full_name='prystine_detection.prystine.protos.LearningRate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='multi_phase', full_name='prystine_detection.prystine.protos.LearningRate.multi_phase', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='one_cycle', full_name='prystine_detection.prystine.protos.LearningRate.one_cycle', index=1,
      number=2, type=11, cpp_type=10, label=1,
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
    _descriptor.OneofDescriptor(
      name='learning_rate', full_name='prystine_detection.prystine.protos.LearningRate.learning_rate',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=912,
  serialized_end=1081,
)


_LEARNINGRATEPHASE = _descriptor.Descriptor(
  name='LearningRatePhase',
  full_name='prystine_detection.prystine.protos.LearningRatePhase',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='start', full_name='prystine_detection.prystine.protos.LearningRatePhase.start', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='lambda_func', full_name='prystine_detection.prystine.protos.LearningRatePhase.lambda_func', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='momentum_lambda_func', full_name='prystine_detection.prystine.protos.LearningRatePhase.momentum_lambda_func', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=1083,
  serialized_end=1168,
)


_MULTIPHASE = _descriptor.Descriptor(
  name='MultiPhase',
  full_name='prystine_detection.prystine.protos.MultiPhase',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='phases', full_name='prystine_detection.prystine.protos.MultiPhase.phases', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=1170,
  serialized_end=1253,
)


_ONECYCLE = _descriptor.Descriptor(
  name='OneCycle',
  full_name='prystine_detection.prystine.protos.OneCycle',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lr_max', full_name='prystine_detection.prystine.protos.OneCycle.lr_max', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='moms', full_name='prystine_detection.prystine.protos.OneCycle.moms', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='div_factor', full_name='prystine_detection.prystine.protos.OneCycle.div_factor', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pct_start', full_name='prystine_detection.prystine.protos.OneCycle.pct_start', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=1255,
  serialized_end=1334,
)

_OPTIMIZER.fields_by_name['rms_prop_optimizer'].message_type = _RMSPROPOPTIMIZER
_OPTIMIZER.fields_by_name['momentum_optimizer'].message_type = _MOMENTUMOPTIMIZER
_OPTIMIZER.fields_by_name['adam_optimizer'].message_type = _ADAMOPTIMIZER
_OPTIMIZER.oneofs_by_name['optimizer'].fields.append(
  _OPTIMIZER.fields_by_name['rms_prop_optimizer'])
_OPTIMIZER.fields_by_name['rms_prop_optimizer'].containing_oneof = _OPTIMIZER.oneofs_by_name['optimizer']
_OPTIMIZER.oneofs_by_name['optimizer'].fields.append(
  _OPTIMIZER.fields_by_name['momentum_optimizer'])
_OPTIMIZER.fields_by_name['momentum_optimizer'].containing_oneof = _OPTIMIZER.oneofs_by_name['optimizer']
_OPTIMIZER.oneofs_by_name['optimizer'].fields.append(
  _OPTIMIZER.fields_by_name['adam_optimizer'])
_OPTIMIZER.fields_by_name['adam_optimizer'].containing_oneof = _OPTIMIZER.oneofs_by_name['optimizer']
_RMSPROPOPTIMIZER.fields_by_name['learning_rate'].message_type = _LEARNINGRATE
_MOMENTUMOPTIMIZER.fields_by_name['learning_rate'].message_type = _LEARNINGRATE
_ADAMOPTIMIZER.fields_by_name['learning_rate'].message_type = _LEARNINGRATE
_LEARNINGRATE.fields_by_name['multi_phase'].message_type = _MULTIPHASE
_LEARNINGRATE.fields_by_name['one_cycle'].message_type = _ONECYCLE
_LEARNINGRATE.oneofs_by_name['learning_rate'].fields.append(
  _LEARNINGRATE.fields_by_name['multi_phase'])
_LEARNINGRATE.fields_by_name['multi_phase'].containing_oneof = _LEARNINGRATE.oneofs_by_name['learning_rate']
_LEARNINGRATE.oneofs_by_name['learning_rate'].fields.append(
  _LEARNINGRATE.fields_by_name['one_cycle'])
_LEARNINGRATE.fields_by_name['one_cycle'].containing_oneof = _LEARNINGRATE.oneofs_by_name['learning_rate']
_MULTIPHASE.fields_by_name['phases'].message_type = _LEARNINGRATEPHASE
DESCRIPTOR.message_types_by_name['Optimizer'] = _OPTIMIZER
DESCRIPTOR.message_types_by_name['RMSPropOptimizer'] = _RMSPROPOPTIMIZER
DESCRIPTOR.message_types_by_name['MomentumOptimizer'] = _MOMENTUMOPTIMIZER
DESCRIPTOR.message_types_by_name['AdamOptimizer'] = _ADAMOPTIMIZER
DESCRIPTOR.message_types_by_name['LearningRate'] = _LEARNINGRATE
DESCRIPTOR.message_types_by_name['LearningRatePhase'] = _LEARNINGRATEPHASE
DESCRIPTOR.message_types_by_name['MultiPhase'] = _MULTIPHASE
DESCRIPTOR.message_types_by_name['OneCycle'] = _ONECYCLE

Optimizer = _reflection.GeneratedProtocolMessageType('Optimizer', (_message.Message,), dict(
  DESCRIPTOR = _OPTIMIZER,
  __module__ = 'prystine_detection.prystine.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:prystine_detection.prystine.protos.Optimizer)
  ))
_sym_db.RegisterMessage(Optimizer)

RMSPropOptimizer = _reflection.GeneratedProtocolMessageType('RMSPropOptimizer', (_message.Message,), dict(
  DESCRIPTOR = _RMSPROPOPTIMIZER,
  __module__ = 'prystine_detection.prystine.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:prystine_detection.prystine.protos.RMSPropOptimizer)
  ))
_sym_db.RegisterMessage(RMSPropOptimizer)

MomentumOptimizer = _reflection.GeneratedProtocolMessageType('MomentumOptimizer', (_message.Message,), dict(
  DESCRIPTOR = _MOMENTUMOPTIMIZER,
  __module__ = 'prystine_detection.prystine.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:prystine_detection.prystine.protos.MomentumOptimizer)
  ))
_sym_db.RegisterMessage(MomentumOptimizer)

AdamOptimizer = _reflection.GeneratedProtocolMessageType('AdamOptimizer', (_message.Message,), dict(
  DESCRIPTOR = _ADAMOPTIMIZER,
  __module__ = 'prystine_detection.prystine.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:prystine_detection.prystine.protos.AdamOptimizer)
  ))
_sym_db.RegisterMessage(AdamOptimizer)

LearningRate = _reflection.GeneratedProtocolMessageType('LearningRate', (_message.Message,), dict(
  DESCRIPTOR = _LEARNINGRATE,
  __module__ = 'prystine_detection.prystine.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:prystine_detection.prystine.protos.LearningRate)
  ))
_sym_db.RegisterMessage(LearningRate)

LearningRatePhase = _reflection.GeneratedProtocolMessageType('LearningRatePhase', (_message.Message,), dict(
  DESCRIPTOR = _LEARNINGRATEPHASE,
  __module__ = 'prystine_detection.prystine.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:prystine_detection.prystine.protos.LearningRatePhase)
  ))
_sym_db.RegisterMessage(LearningRatePhase)

MultiPhase = _reflection.GeneratedProtocolMessageType('MultiPhase', (_message.Message,), dict(
  DESCRIPTOR = _MULTIPHASE,
  __module__ = 'prystine_detection.prystine.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:prystine_detection.prystine.protos.MultiPhase)
  ))
_sym_db.RegisterMessage(MultiPhase)

OneCycle = _reflection.GeneratedProtocolMessageType('OneCycle', (_message.Message,), dict(
  DESCRIPTOR = _ONECYCLE,
  __module__ = 'prystine_detection.prystine.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:prystine_detection.prystine.protos.OneCycle)
  ))
_sym_db.RegisterMessage(OneCycle)


# @@protoc_insertion_point(module_scope)