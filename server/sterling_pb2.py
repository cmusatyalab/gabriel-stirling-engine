# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sterling.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='sterling.proto',
  package='sterling',
  syntax='proto3',
  serialized_options=_b('\n\023edu.cmu.cs.sterlingB\006Protos'),
  serialized_pb=_b('\n\x0esterling.proto\x12\x08sterling\"\xf6\x01\n\x0eToServerExtras\x12\x1c\n\x04step\x18\x01 \x01(\x0e\x32\x0e.sterling.Step\x12\x18\n\x10\x66rames_same_hash\x18\x02 \x01(\x05\x12\x1d\n\x15\x66rames_completed_step\x18\x03 \x01(\x05\x12\x35\n\x11viewfinder_status\x18\x04 \x01(\x0e\x32\x1a.sterling.ViewfinderStatus\x12\x17\n\x0f\x64\x65tected_frames\x18\x05 \x01(\x05\x12\x19\n\x11undetected_frames\x18\x06 \x01(\x05\x12\x0f\n\x07go_back\x18\x07 \x01(\x08\x12\x11\n\tlast_hash\x18\x08 \x01(\t\"\x84\x02\n\x0eToClientExtras\x12\x1c\n\x04step\x18\x01 \x01(\x0e\x32\x0e.sterling.Step\x12\r\n\x05image\x18\x02 \x01(\x0c\x12\x0e\n\x06speech\x18\x03 \x01(\t\x12\x18\n\x10\x66rames_same_hash\x18\x04 \x01(\x05\x12\x1d\n\x15\x66rames_completed_step\x18\x05 \x01(\x05\x12\x35\n\x11viewfinder_change\x18\x06 \x01(\x0e\x32\x1a.sterling.ViewfinderChange\x12\x17\n\x0f\x64\x65tected_frames\x18\x07 \x01(\x05\x12\x19\n\x11undetected_frames\x18\x08 \x01(\x05\x12\x11\n\tlast_hash\x18\t \x01(\t*\xf6\x02\n\x04Step\x12\t\n\x05START\x10\x00\x12\x0e\n\nFOURSCREWS\x10\x01\x12\x0f\n\x0bTHREESCREWS\x10\x02\x12\r\n\tTWOSCREWS\x10\x03\x12\x14\n\x10TWOSCREWSVISIBLE\x10\x04\x12\x0c\n\x08ONESCREW\x10\x05\x12\x0c\n\x08NOSCREWS\x10\x06\x12\t\n\x05NOPAD\x10\x07\x12\n\n\x06NORING\x10\x08\x12\x0e\n\nNOCYLINDER\x10\t\x12\x0c\n\x08NOPISTON\x10\n\x12\x0b\n\x07TWORODS\x10\x0b\x12\x0f\n\x0b\x46IRSTRODOFF\x10\x0c\x12\n\n\x06ONEROD\x10\r\x12\x0f\n\x0bSECONDRODON\x10\x0e\x12\t\n\x05NOROD\x10\x0f\x12\r\n\tTWOWHEELS\x10\x10\x12\x0c\n\x08ONEWHEEL\x10\x11\x12\x0b\n\x07NOWHEEL\x10\x12\x12\x0b\n\x07NOSHAFT\x10\x13\x12\x12\n\x0eTWOSCREWS_BASE\x10\x14\x12\x11\n\rONESCREW_BASE\x10\x15\x12\x10\n\x0cNOSCREW_BASE\x10\x16\x12\x0c\n\x08\x46INISHED\x10\x17\x12\x08\n\x04\x44ONE\x10\x18*:\n\x10ViewfinderChange\x12\n\n\x06TurnOn\x10\x00\x12\x0b\n\x07TurnOff\x10\x01\x12\r\n\tDoNothing\x10\x02*\'\n\x10ViewfinderStatus\x12\x08\n\x04IsOn\x10\x00\x12\t\n\x05IsOff\x10\x01\x42\x1d\n\x13\x65\x64u.cmu.cs.sterlingB\x06Protosb\x06proto3')
)

_STEP = _descriptor.EnumDescriptor(
  name='Step',
  full_name='sterling.Step',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='START', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FOURSCREWS', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='THREESCREWS', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TWOSCREWS', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TWOSCREWSVISIBLE', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ONESCREW', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOSCREWS', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOPAD', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NORING', index=8, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOCYLINDER', index=9, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOPISTON', index=10, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TWORODS', index=11, number=11,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FIRSTRODOFF', index=12, number=12,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ONEROD', index=13, number=13,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SECONDRODON', index=14, number=14,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOROD', index=15, number=15,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TWOWHEELS', index=16, number=16,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ONEWHEEL', index=17, number=17,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOWHEEL', index=18, number=18,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOSHAFT', index=19, number=19,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TWOSCREWS_BASE', index=20, number=20,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ONESCREW_BASE', index=21, number=21,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOSCREW_BASE', index=22, number=22,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FINISHED', index=23, number=23,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DONE', index=24, number=24,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=541,
  serialized_end=915,
)
_sym_db.RegisterEnumDescriptor(_STEP)

Step = enum_type_wrapper.EnumTypeWrapper(_STEP)
_VIEWFINDERCHANGE = _descriptor.EnumDescriptor(
  name='ViewfinderChange',
  full_name='sterling.ViewfinderChange',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TurnOn', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TurnOff', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DoNothing', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=917,
  serialized_end=975,
)
_sym_db.RegisterEnumDescriptor(_VIEWFINDERCHANGE)

ViewfinderChange = enum_type_wrapper.EnumTypeWrapper(_VIEWFINDERCHANGE)
_VIEWFINDERSTATUS = _descriptor.EnumDescriptor(
  name='ViewfinderStatus',
  full_name='sterling.ViewfinderStatus',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='IsOn', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IsOff', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=977,
  serialized_end=1016,
)
_sym_db.RegisterEnumDescriptor(_VIEWFINDERSTATUS)

ViewfinderStatus = enum_type_wrapper.EnumTypeWrapper(_VIEWFINDERSTATUS)
START = 0
FOURSCREWS = 1
THREESCREWS = 2
TWOSCREWS = 3
TWOSCREWSVISIBLE = 4
ONESCREW = 5
NOSCREWS = 6
NOPAD = 7
NORING = 8
NOCYLINDER = 9
NOPISTON = 10
TWORODS = 11
FIRSTRODOFF = 12
ONEROD = 13
SECONDRODON = 14
NOROD = 15
TWOWHEELS = 16
ONEWHEEL = 17
NOWHEEL = 18
NOSHAFT = 19
TWOSCREWS_BASE = 20
ONESCREW_BASE = 21
NOSCREW_BASE = 22
FINISHED = 23
DONE = 24
TurnOn = 0
TurnOff = 1
DoNothing = 2
IsOn = 0
IsOff = 1



_TOSERVEREXTRAS = _descriptor.Descriptor(
  name='ToServerExtras',
  full_name='sterling.ToServerExtras',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='step', full_name='sterling.ToServerExtras.step', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frames_same_hash', full_name='sterling.ToServerExtras.frames_same_hash', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frames_completed_step', full_name='sterling.ToServerExtras.frames_completed_step', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='viewfinder_status', full_name='sterling.ToServerExtras.viewfinder_status', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detected_frames', full_name='sterling.ToServerExtras.detected_frames', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='undetected_frames', full_name='sterling.ToServerExtras.undetected_frames', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='go_back', full_name='sterling.ToServerExtras.go_back', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='last_hash', full_name='sterling.ToServerExtras.last_hash', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=29,
  serialized_end=275,
)


_TOCLIENTEXTRAS = _descriptor.Descriptor(
  name='ToClientExtras',
  full_name='sterling.ToClientExtras',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='step', full_name='sterling.ToClientExtras.step', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image', full_name='sterling.ToClientExtras.image', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='speech', full_name='sterling.ToClientExtras.speech', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frames_same_hash', full_name='sterling.ToClientExtras.frames_same_hash', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frames_completed_step', full_name='sterling.ToClientExtras.frames_completed_step', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='viewfinder_change', full_name='sterling.ToClientExtras.viewfinder_change', index=5,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detected_frames', full_name='sterling.ToClientExtras.detected_frames', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='undetected_frames', full_name='sterling.ToClientExtras.undetected_frames', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='last_hash', full_name='sterling.ToClientExtras.last_hash', index=8,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=278,
  serialized_end=538,
)

_TOSERVEREXTRAS.fields_by_name['step'].enum_type = _STEP
_TOSERVEREXTRAS.fields_by_name['viewfinder_status'].enum_type = _VIEWFINDERSTATUS
_TOCLIENTEXTRAS.fields_by_name['step'].enum_type = _STEP
_TOCLIENTEXTRAS.fields_by_name['viewfinder_change'].enum_type = _VIEWFINDERCHANGE
DESCRIPTOR.message_types_by_name['ToServerExtras'] = _TOSERVEREXTRAS
DESCRIPTOR.message_types_by_name['ToClientExtras'] = _TOCLIENTEXTRAS
DESCRIPTOR.enum_types_by_name['Step'] = _STEP
DESCRIPTOR.enum_types_by_name['ViewfinderChange'] = _VIEWFINDERCHANGE
DESCRIPTOR.enum_types_by_name['ViewfinderStatus'] = _VIEWFINDERSTATUS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ToServerExtras = _reflection.GeneratedProtocolMessageType('ToServerExtras', (_message.Message,), {
  'DESCRIPTOR' : _TOSERVEREXTRAS,
  '__module__' : 'sterling_pb2'
  # @@protoc_insertion_point(class_scope:sterling.ToServerExtras)
  })
_sym_db.RegisterMessage(ToServerExtras)

ToClientExtras = _reflection.GeneratedProtocolMessageType('ToClientExtras', (_message.Message,), {
  'DESCRIPTOR' : _TOCLIENTEXTRAS,
  '__module__' : 'sterling_pb2'
  # @@protoc_insertion_point(class_scope:sterling.ToClientExtras)
  })
_sym_db.RegisterMessage(ToClientExtras)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
