# Lint as: python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions used in mp_run_executor.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Text
from ml_metadata.proto import metadata_store_pb2
from tfx.types import artifact
from tfx.types import artifact_utils


def parse_raw_artifact_dict(inputs_dict) -> Dict[Text, List[artifact.Artifact]]:
  """Parses a map from key to a list of a single Artifact from pb objects."""
  result = {}
  for k, v in inputs_dict.items():
    result[k] = [
        _parse_raw_artifact(single_artifact)
        for single_artifact in v.artifacts
    ]
  return result


def parse_execution_properties(exec_properties) -> Dict[Text, Any]:
  """Parses a map from key to Value proto as execution properties."""
  result = {}
  for k, v in exec_properties.items():
    # Translate each field from Value pb to plain value.
    result[k] = getattr(v, v.WhichOneof('value'))
    if result[k] is None:
      raise TypeError('Unrecognized type encountered at field %s of execution'
                      ' properties %s' % (k, exec_properties))

  return result


def _parse_raw_artifact(artifact_pb) -> artifact.Artifact:
  """Parses json serialized version of artifact without artifact_type."""
  # This parser can only reserve what's inside artifact pb message.
  # TODO(b/152444458): For compatibility, current TFX serialization assumes
  # there is no type field in Artifact pb message.
  type_name = artifact_pb.type
  artifact_pb.ClearField('type')

  # Make an ArtifactType pb according to artifact_pb
  type_pb = metadata_store_pb2.ArtifactType()
  type_pb.name = type_name
  for k, v in artifact_pb.properties.items():
    if v.HasField('int_value'):
      type_pb.properties[k] = metadata_store_pb2.PropertyType.INT
    elif v.HasField('string_value'):
      type_pb.properties[k] = metadata_store_pb2.PropertyType.STRING
    elif v.HasField('double_value'):
      type_pb.properties[k] = metadata_store_pb2.PropertyType.DOUBLE
    else:
      raise ValueError('Unrecognized type encountered at field %s' % (k))

  result = artifact_utils.deserialize_artifact(type_pb, artifact_pb)
  return result
