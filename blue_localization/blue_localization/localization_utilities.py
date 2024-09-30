from __future__ import annotations

import time
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from tf2_geometry_msgs import PoseStamped

MatchedDetections = namedtuple("FindDetectionResult", ["matched", "unmatched"])
MatchedPair = namedtuple("MatchedPair", ["gt_tag", "detected_tag"])


def tags_from_msg(msg: AprilTagDetectionArray) -> List[Tag]:
    """
    Convert AprilTagDetectionArray to ROS-Agnostic dataclass

    Args:
        msg (AprilTagDetectionArray): ROS Message

    Returns:
        List[Tag]: List of Tag classes
    """
    tag_list = []
    for dt in msg.detections:
        corners = [
            [dt.corners[0].x, dt.corners[0].y],
            [dt.corners[1].x, dt.corners[1].y],
            [dt.corners[2].x, dt.corners[2].y],
            [dt.corners[3].x, dt.corners[3].y],
        ]
        tag = Tag(
            tag_family=dt.family,
            tag_id=dt.id,
            hamming=dt.hamming,
            decision_margin=dt.decision_margin,
            homography=dt.homography,
            center=[dt.centre.x, dt.centre.y],
            corners=corners,
            pose_R=None,
            pose_t=None,
            pose_err=None,
        )
        tag_list.append(tag)
    return tag_list


def load_yaml(yaml_file: Path) -> dict:
    """
    Generate a dict from a YAML input file formatted as TagSLAM output.
    """
    tag_dict = {}
    try:
        with open(yaml_file, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
            tag_list = data["bodies"][0]["osb"]["tags"]
    except yaml.YAMLError as err:
        print(err)

    tag_dict = {}
    for tag in tag_list:
        try:
            roll_x, roll_y, roll_z = (
                tag["pose"]["rotation"]["x"],
                tag["pose"]["rotation"]["y"],
                tag["pose"]["rotation"]["z"],
            )
            rot_matrix = R.from_euler("xyz", [roll_x, roll_y, roll_z]).as_matrix()

            tag_dict[tag["id"]] = {
                "size": tag["size"],
                "position": np.array(
                    [
                        tag["pose"]["position"]["x"],
                        tag["pose"]["position"]["y"],
                        tag["pose"]["position"]["z"],
                    ]
                ),
                "rotation": rot_matrix,
            }
        except KeyError as ex:
            print(f"Error with tag {tag['id']} - {ex}")
    print(f"Loaded ground truth data for {len(tag_dict)} tags.")
    return tag_dict


@dataclass
class LocalizationResult:
    """
    Dataclass to store localization result
    """

    rvec: np.array
    tvec: np.array
    cam_rot: np.array
    cam_trans: np.array
    tag_count: int
    reprojection_error: float


@dataclass
class Tag:
    """
    Tag class. Mirrors structure of Apriltag Detection messages
    """

    tag_family: str
    tag_id: int
    hamming: int
    decision_margin: float
    homography: List[List[float]]
    center: List[float]
    corners: List[List[float]]
    pose_R: Optional[List[List[float]]]
    pose_t: Optional[List[float]]
    pose_err: Optional[float]


class FiducialMap:
    """
    Stores a ground truth map of fiducials.
    Ground Truth data is loaded from a YAML (from TagSLAM)
    """

    def __init__(self, yaml_file: str, verbose: bool = True):
        """Initialize a FiducialMap."""
        self.gt_dict = load_yaml(yaml_file)
        self.verbose = verbose

    def match_detections(self, detected_tags: List[Tag]):
        """Match detected tags with tags in map."""

        matched = []
        unmatched = []

        for tag in detected_tags:
            if tag.tag_id in self.gt_dict:
                matched.append(MatchedPair(self.gt_dict[tag.tag_id], tag))
            else:
                unmatched.append(tag)

        if self.verbose:
            print(
                f"Found {len(matched)} matched tags and {len(unmatched)} unmatched tags."
            )
        result = MatchedDetections(matched=matched, unmatched=unmatched)
        return result


class PNPLocalizer:
    """PNP Based Localization Class"""

    def __init__(
        self,
        fiducial_map: FiducialMap,
        camera_matrix: np.array,
        dist_coeffs: np.array,
        use_ransac: bool = False,
        min_tags: int = 5,
        verbose: bool = False,
    ) -> None:
        self.fiducial_map = fiducial_map
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.verbose = verbose
        self.min_tags = min_tags
        self.use_ransac = use_ransac

    def compute_pose(self, matched_tags: MatchedDetections) -> LocalizationResult:
        """
        Compute pose based on detected tags

        Args:
            matched_tags (MatchedDetections): named tuple of matched tags

        Returns:
            LocalizationResult: Dataclass representing a localization result
        """
        results = None
        if len(matched_tags[0]) <= 1:
            if self.verbose:
                print("Not enough tags detected! Skipping")
            return results

        object_points = []
        image_points = []
        for gt_tag, detected_tag in matched_tags[0]:
            # obj_pts_world = gt_tag["position"]
            tz = gt_tag["size"]
            obj_pts_local = np.array(
                [
                    [-tz / 2, -tz / 2, 0],
                    [tz / 2, -tz / 2, 0],
                    [tz / 2, tz / 2, 0],
                    [-tz / 2, tz / 2, 0],
                ],
                dtype=np.float32,
            )
            obj_pts_rot = gt_tag["rotation"] @ obj_pts_local.T
            obj_pts_world = obj_pts_rot.T + gt_tag["position"]
            object_points.extend(obj_pts_world)
            # object_points.append(obj_pts_world)
            image_points.append(detected_tag.corners)

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32).reshape(-1, 2)

        if len(matched_tags[0]) >= self.min_tags:
            try:
                t0 = time.time()
                if not self.use_ransac:
                    retval, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                    if self.verbose:
                        print(f"Solved PnP in {1000*(time.time() - t0):.3f} ms")
                else:
                    t0 = time.time()
                    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                        object_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        reprojectionError=1.0,
                        confidence=0.99,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                    if self.verbose:
                        print(f"Solved PnP in {1000*(time.time() - t0):.3f} ms")
            except Exception:
                return results

            if retval:
                cam_rot, _ = cv2.Rodrigues(rvec)
                cam_to_world = cam_rot.T
                cam_trans = -np.dot(cam_to_world, tvec).squeeze()

                if not self.use_ransac:
                    projected_points, _ = cv2.projectPoints(
                        object_points,
                        rvec,
                        tvec,
                        self.camera_matrix,
                        self.dist_coeffs,
                    )

                    reprojection_errors = np.linalg.norm(
                        image_points - projected_points.squeeze(), axis=1
                    )
                else:
                    projected_points, _ = cv2.projectPoints(
                        object_points[inliers],
                        rvec,
                        tvec,
                        self.camera_matrix,
                        self.dist_coeffs,
                    )

                    reprojection_errors = np.linalg.norm(
                        image_points[inliers] - projected_points.squeeze(), axis=1
                    )
                # Calculate mean reprojection error
                error = np.mean(reprojection_errors)

                if not self.use_ransac:
                    if error > 100:
                        return None

                results = LocalizationResult(
                    rvec=rvec,
                    tvec=tvec,
                    cam_rot=cam_rot,
                    cam_trans=cam_trans,
                    tag_count=len(matched_tags[0]),
                    reprojection_error=error,
                )

            return results


def pose_to_matrix(pose_msg: PoseStamped):
    """Convert PoseStamped to 4x4 transformation matrix."""
    # Extract translation
    trans = np.array(
        [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
    )

    rot = R.from_quat(
        [
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w,
        ]
    ).as_matrix()

    # Construct the 4x4 transformation matrix
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = trans
    return mat


def transform_to_matrix(transform: TransformStamped):
    """Convert TransformStamped to 4x4 transformation matrix."""
    # Extract translation
    trans = np.array(
        [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ]
    )

    rot = R.from_quat(
        [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]
    ).as_matrix()

    # Construct the 4x4 transformation matrix
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = trans
    return mat
