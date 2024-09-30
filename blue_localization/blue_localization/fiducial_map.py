#!/usr/bin/env python3
#
# Copyright (c) 2024 University of Washington, All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# Neither the name of the copyright holder nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Stores a fiducial map."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import yaml
from apriltag_msgs.msg import ApriltagDetectionArray

FindDetectionResult = namedtuple("FindDetectionResult", ["matched", "unmatched"])
MatchedPair = namedtuple("MatchedPair", ["object", "detection"])


class FiducialMap:
    """Stores a map of fiducials."""

    @classmethod
    def load_yaml(yaml_file: Path) -> FiducialMap:
        """Generate a FiducialMap from a YAML input file."""
        with open(yaml_file) as fp:
            yaml_in = yaml.load(fp)

        if "maps" not in yaml_in:
            return None

        return FiducialMap(yaml_in["maps"])

    def __init__(self, maps: dict):
        """Initialize a FiducialMap."""
        self.maps = maps

        self.fiducials = {}
        # Create an internal hash table
        for m in maps:
            if "tags" not in m:
                continue

            tag_type = m["type"]
            tag_family = m["family"]

            for tag in m["tags"]:
                uuid = f"{tag_type}_{tag_family}_{tag.id}"

                self.fiducials[uuid] = tag["world"]

    def match_detections(self, detection_array: ApriltagDetectionArray):
        """Match detected tags with tags in map."""
        result = FindDetectionResult()

        result.matched = []
        result.unmatched = []

        for detection in detection_array:
            uuid = f"apriltags_{detection.family}_{detection.id}"

            if uuid in self.fiducials:
                result.matched.append(MatchedPair(self.fiducials[uuid], detection))
            else:
                result.unmatched.append(detection)

        return result
