# Copyright 2023, Evan Palmer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from abc import ABC
from typing import Any

import cv2
import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    TwistStamped,
    TwistWithCovarianceStamped,
)
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import FluidPressure, Image, Imu
from tf2_ros import TransformException  # type: ignore
from tf2_ros import Time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class Localizer(Node, ABC):
    """Base class for implementing a localization interface.

    The solution obtained from the localization platform can go to two places: the
    ArduPilot EKF via the external vision interface and/or the Blue EKF which uses the
    Charles River Analytics, Inc. nonlinear state estimation interface. Below are some
    factors to consider when determining which interface to use:

    When to use the ArduPilot EKF:
        - When you only need a single external pose and/or velocity state provider
        - When you need to use GUIDED mode (or the other autonomous flight modes)
        - When working in simulation (there is less setup involved)
        - When you are only using onboard sensors that already have direct integration
          into ArduSub

    When to use the Blue EKF:
        - When you have more external state providers than ArduPilot provides interfaces
          for
        - When you want to integrate support for additional sensors (e.g., IMU)
        - When you want more control over the EKF parameters
        - When you want to use UKF (EKF is the default, but the state estimation
          interface also supports UKF)
        - When you don't want to fight 9 rounds in the octogon with ArduPilot and MAVROS
          just to get state information to your controller/planner

    As previously indicated, both estimators may be used at the same time, but the
    controller should only read state information from one estimator.
    """

    MAP_FRAME = "map"
    MAP_NED_FRAME = "map_ned"
    BASE_LINK_FRAME = "base_link"
    BASE_LINK_FRD_FRAME = "base_link_frd"
    CAMERA_FRAME = "camera"

    def __init__(self, node_name: str) -> None:
        """Create a new localizer.

        Args:
            node_name: The name of the ROS 2 node.
        """
        Node.__init__(self, node_name)
        ABC.__init__(self)

        # Provide access to TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


class PoseLocalizer(Localizer):
    """Interface for publishing pose states.

    This is supported both by the Blue EKF and the ArduPilot EKF.
    """

    def __init__(self, node_name: str) -> None:
        """Create a new pose localization interface.

        Args:
            node_name: The pose interface name.
        """
        super().__init__(node_name)

        self.declare_parameter("estimator", "ardusub")

        self.estimator = (
            self.get_parameter("estimator").get_parameter_value().string_value
        )

        if self.estimator not in ["ardusub", "blue", "both"]:
            raise ValueError(f"Invalid estimator provided: {self.estimator}")

        self.vision_pose_pub = self.create_publisher(
            PoseStamped, "/mavros/vision_pose/pose", 1
        )
        self.vision_pose_cov_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/mavros/vision_pose/pose_cov", 1
        )
        self.robot_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, f"/blue/ekf/pose/{node_name}/in", 1
        )

    def publish(self, pose: PoseStamped | PoseWithCovarianceStamped) -> None:
        """Publish a pose message to the desired estimator.

        This method is simply a wrapper for the state publishers to abstract away the
        need to think about which state gets published where.

        Args:
            pose: The state message to send.
        """
        match self.estimator:
            case "ardusub":
                if isinstance(pose, PoseStamped):
                    self.vision_pose_pub.publish(pose)
                else:
                    self.vision_pose_cov_pub.publish(pose)
            case "blue":
                self.robot_pose_pub.publish(pose)
            case "both":
                if isinstance(pose, PoseStamped):
                    self.vision_pose_pub.publish(pose)
                else:
                    self.vision_pose_cov_pub.publish(pose)
                self.robot_pose_pub.publish(pose)


class TwistLocalizer(Localizer):
    """Interface for publishing twist states.

    This is useful when integrating external velocity measurement sensors (e.g., a
    DVL).

    This is supported both by the Blue EKF and the ArduPilot EKF.
    """

    def __init__(self, node_name: str) -> None:
        """Create a new twist localization interface.

        Args:
            node_name: The name of the velocity localizer.
        """
        super().__init__(node_name)

        self.declare_parameter("estimator", "ardusub")

        self.estimator = (
            self.get_parameter("estimator").get_parameter_value().string_value
        )

        if self.estimator not in ["ardusub", "blue", "both"]:
            raise ValueError(f"Invalid estimator provided: {self.estimator}")

        self.vision_speed_pub = self.create_publisher(
            TwistStamped, "/mavros/vision_speed/speed", 1
        )
        self.vision_speed_cov_pub = self.create_publisher(
            TwistWithCovarianceStamped, "/mavros/vision_speed/speed_cov", 1
        )
        self.robot_twist_pub = self.create_publisher(
            TwistWithCovarianceStamped, f"/blue/ekf/twist/{node_name}/in", 1
        )

    def publish(self, twist: TwistStamped | TwistWithCovarianceStamped) -> None:
        """Publish a twist message to the desired estimator.

        This method is simply a wrapper for the state publishers to abstract away the
        need to think about which state gets published where.

        Args:
            twist: The state message to send.
        """
        match self.estimator:
            case "ardusub":
                if isinstance(twist, TwistStamped):
                    self.vision_speed_pub.publish(twist)
                else:
                    self.vision_speed_cov_pub.publish(twist)
            case "blue":
                self.robot_twist_pub.publish(twist)
            case "both":
                if isinstance(twist, TwistStamped):
                    self.robot_twist_pub.publish(twist)
                else:
                    self.vision_speed_cov_pub.publish(twist)
                self.robot_twist_pub.publish(twist)


class ImuLocalizer(Localizer):
    """Interface for publishing IMU readings.

    The IMU localization framework primarily targets platforms with multiple IMU
    sensors that need to be fused together.

    This is supported only by the Blue localization framework.
    """

    def __init__(self, node_name: str) -> None:
        """Create a new IMU localizer.

        Args:
            node_name: The name of the IMU localizer.
        """
        super().__init__(node_name)

        # IMU published to the external Blue EKF.
        self.robot_twist_pub = self.create_publisher(
            Imu, f"/blue/ekf/imu/{node_name}/in", 1
        )


class OdometryLocalizer(Localizer):
    """Interface for publishing odometry measurements.

    This is only supported by the Blue EKF.
    """

    def __init__(self, node_name: str) -> None:
        """Create a new odometry localization interface.

        Args:
            node_name: The name of the odometry localizer.
        """
        super().__init__(node_name)

        self.robot_odom_pub = self.create_publisher(
            Odometry, f"/blue/ekf/odom/{node_name}/in", 1
        )


class ArucoMarkerLocalizer(PoseLocalizer):
    """Performs localization using ArUco markers."""

    ARUCO_MARKER_TYPES = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_4X4_100,
        cv2.aruco.DICT_4X4_250,
        cv2.aruco.DICT_4X4_1000,
        cv2.aruco.DICT_5X5_50,
        cv2.aruco.DICT_5X5_100,
        cv2.aruco.DICT_5X5_250,
        cv2.aruco.DICT_5X5_1000,
        cv2.aruco.DICT_6X6_50,
        cv2.aruco.DICT_6X6_100,
        cv2.aruco.DICT_6X6_250,
        cv2.aruco.DICT_6X6_1000,
        cv2.aruco.DICT_7X7_50,
        cv2.aruco.DICT_7X7_100,
        cv2.aruco.DICT_7X7_250,
        cv2.aruco.DICT_7X7_1000,
        cv2.aruco.DICT_ARUCO_ORIGINAL,
    ]

    def __init__(self) -> None:
        """Create a new ArUco marker localizer."""
        super().__init__("aruco_marker_localizer")

        self.bridge = CvBridge()

        self.declare_parameter("camera_matrix", list(np.zeros(9)))
        self.declare_parameter("projection_matrix", list(np.zeros(12)))
        self.declare_parameter("distortion_coefficients", list(np.zeros(5)))
        self.declare_parameter("covariance", list(np.zeros(36)))

        # Get the camera intrinsics
        self.camera_matrix = np.array(
            self.get_parameter("camera_matrix")
            .get_parameter_value()
            .double_array_value,
            np.float32,
        ).reshape(3, 3)

        self.projection_matrix = np.array(
            self.get_parameter("projection_matrix")
            .get_parameter_value()
            .double_array_value,
            np.float32,
        ).reshape(3, 4)

        self.distortion_coefficients = np.array(
            self.get_parameter("distortion_coefficients")
            .get_parameter_value()
            .double_array_value,
            np.float32,
        ).reshape(1, 5)

        self.covariance = np.array(
            self.get_parameter("covariance").get_parameter_value().double_array_value
        ).reshape((6, 6))

        self.camera_sub = self.create_subscription(
            Image, "/camera", self.extract_and_publish_pose_cb, 1
        )

    def detect_markers(self, frame: np.ndarray) -> tuple[Any, Any] | None:
        """Detect any ArUco markers in the frame.

        All markers in a frame should be the same type of ArUco marker
        (e.g., 4x4 50) if multiple are expected to be in-frame.

        Args:
            frame: The video frame containing ArUco markers.

        Returns:
            A list of marker corners and IDs. If no markers were found, returns None.
        """
        # Check each tag type, breaking when we find one that works
        for tag_type in self.ARUCO_MARKER_TYPES:
            aruco_dict = cv2.aruco.Dictionary_get(tag_type)
            aruco_params = cv2.aruco.DetectorParameters_create()

            try:
                # Return the corners and ids if we find the correct tag type
                corners, ids, _ = cv2.aruco.detectMarkers(
                    frame, aruco_dict, parameters=aruco_params
                )

                if len(ids) > 0:
                    return corners, ids

            except Exception:
                continue

        # Nothing was found
        return None

    def get_camera_pose(self, frame: np.ndarray) -> tuple[Any, Any, int] | None:
        """Get the pose of the camera relative to any ArUco markers detected.

        If multiple markers are detected, then the "largest" marker will be used to
        determine the pose of the camera.

        Args:
            frame: The camera frame containing ArUco markers.

        Returns:
            The rotation vector and translation vector of the camera in the marker
            frame and the ID of the marker detected. If no marker was detected,
            returns None.
        """
        # Convert to greyscale image then try to detect the tag(s)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = self.detect_markers(gray)

        if detection is None:
            return None

        corners, ids = detection

        # If there are multiple markers, get the marker with the "longest" side, where
        # "longest" should be interpreted as the relative size in the image
        side_lengths = [
            abs(corner[0][0][0] - corner[0][2][0])
            + abs(corner[0][0][1] - corner[0][2][1])
            for corner in corners
        ]

        min_side_idx = side_lengths.index(max(side_lengths))
        min_marker_id = ids[min_side_idx]

        # Get the estimated pose
        rot_vec, trans_vec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[min_side_idx],
            min_marker_id,
            self.camera_matrix,
            self.distortion_coefficients,
        )

        return rot_vec, trans_vec, min_marker_id

    def extract_and_publish_pose_cb(self, frame: Image) -> None:
        """Get the camera pose relative to the marker and send to the ArduSub EKF.

        Args:
            frame: The BlueROV2 camera frame.
        """
        # Get the pose of the camera in the `marker` frame
        camera_pose = self.get_camera_pose(self.bridge.imgmsg_to_cv2(frame))

        # If there was no marker in the image, exit early
        if camera_pose is None:
            self.get_logger().debug(
                "An ArUco marker could not be detected in the current image"
            )
            return

        rot_vec, trans_vec, marker_id = camera_pose

        # Convert the pose into a PoseStamped message
        pose_cov = PoseWithCovarianceStamped()

        pose_cov.header.frame_id = f"marker_{marker_id}"
        pose_cov.header.stamp = self.get_clock().now().to_msg()

        (
            pose_cov.pose.pose.position.x,
            pose_cov.pose.pose.position.y,
            pose_cov.pose.pose.position.z,
        ) = trans_vec.squeeze()

        rot_mat, _ = cv2.Rodrigues(rot_vec)

        (
            pose_cov.pose.pose.orientation.x,
            pose_cov.pose.pose.orientation.y,
            pose_cov.pose.pose.orientation.z,
            pose_cov.pose.pose.orientation.w,
        ) = R.from_matrix(rot_mat).as_quat()

        # Transform the pose from the `marker` frame to the `map` frame
        try:
            pose_cov = self.tf_buffer.transform(pose_cov, self.MAP_FRAME)
        except TransformException as e:
            self.get_logger().warning(
                f"Could not transform from frame marker_{marker_id} to map: {e}"
            )
            return

        # The pose now represents the transformation from the map frame to the
        # camera frame, but we need to publish the transformation from the map frame
        # to the base_link frame

        # Start by getting the camera to base_link transform
        try:
            tf_camera_to_base = self.tf_buffer.lookup_transform(
                self.CAMERA_FRAME, self.BASE_LINK_FRAME, Time()
            )
        except TransformException as e:
            self.get_logger().warning(f"Could not access transform: {e}")
            return

        # Convert the tf into a homogeneous tf matrix
        tf_camera_to_base_mat = np.eye(4)
        tf_camera_to_base_mat[:3, :3] = R.from_quat(
            [
                tf_camera_to_base.transform.rotation.x,
                tf_camera_to_base.transform.rotation.y,
                tf_camera_to_base.transform.rotation.z,
                tf_camera_to_base.transform.rotation.w,
            ]
        ).as_matrix()
        tf_camera_to_base_mat[:3, 3] = np.array(
            [
                tf_camera_to_base.transform.translation.x,
                tf_camera_to_base.transform.translation.y,
                tf_camera_to_base.transform.translation.z,
            ]
        )

        # Convert the pose back into a matrix
        tf_map_to_camera_mat = np.eye(4)
        tf_map_to_camera_mat[:3, :3] = R.from_quat(
            [
                pose_cov.pose.orientation.x,  # type: ignore
                pose_cov.pose.orientation.y,  # type: ignore
                pose_cov.pose.orientation.z,  # type: ignore
                pose_cov.pose.orientation.w,  # type: ignore
            ]
        ).as_matrix()
        tf_map_to_camera_mat[:3, 3] = np.array(
            [
                pose_cov.pose.position.x,  # type: ignore
                pose_cov.pose.position.y,  # type: ignore
                pose_cov.pose.position.z,  # type: ignore
            ]
        )

        # Calculate the new transform
        tf_map_to_base_mat = tf_camera_to_base_mat @ tf_map_to_camera_mat

        # Update the pose using the new transform
        (
            pose_cov.pose.position.x,  # type: ignore
            pose_cov.pose.position.y,  # type: ignore
            pose_cov.pose.position.z,  # type: ignore
        ) = tf_map_to_base_mat[3:, 3]

        (
            pose_cov.pose.orientation.x,  # type: ignore
            pose_cov.pose.orientation.y,  # type: ignore
            pose_cov.pose.orientation.z,  # type: ignore
            pose_cov.pose.orientation.w,  # type: ignore
        ) = R.from_matrix(tf_map_to_base_mat[:3, :3]).as_quat()

        self.publish(pose_cov)  # type: ignore


class QualisysLocalizer(PoseLocalizer):
    """Localize the BlueROV2 using the Qualisys motion capture system."""

    def __init__(self) -> None:
        """Create a new Qualisys motion capture localizer."""
        super().__init__("qualisys_localizer")

        self.declare_parameter("body", "bluerov")

        body = self.get_parameter("body").get_parameter_value().string_value

        self.mocap_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            f"/blue/mocap/qualisys/{body}",
            self.proxy_pose_cb,
            1,
        )

    @staticmethod
    def check_isnan(pose_cov: PoseWithCovarianceStamped) -> bool:
        """Check if a pose message has NaN values.

        NaN values are not uncommon when dealing with MoCap data.

        Args:
            pose: The message to check for NaN values.

        Returns:
            Whether or not the message has any NaN values.
        """
        # Check the position
        if np.isnan(
            np.min(
                np.array(
                    [
                        pose_cov.pose.pose.position.x,
                        pose_cov.pose.pose.position.y,
                        pose_cov.pose.pose.position.z,
                    ]
                )
            )
        ):
            return False

        # Check the orientation
        if np.isnan(
            np.min(
                np.array(
                    [
                        pose_cov.pose.pose.orientation.x,
                        pose_cov.pose.pose.orientation.y,
                        pose_cov.pose.pose.orientation.z,
                        pose_cov.pose.pose.orientation.w,
                    ]
                )
            )
        ):
            return False

        return True

    def proxy_pose_cb(self, pose_cov: PoseWithCovarianceStamped) -> None:
        """Proxy the pose to the ArduSub EKF.

        Args:
            pose_cov: The pose of the BlueROV2 identified by the motion capture system.
        """
        # Check if any of the values in the array are NaN; if they are, then
        # discard the reading
        if not self.check_isnan(pose_cov):
            return

        self.robot_pose_pub.publish(pose_cov)


class GazeboLocalizer(PoseLocalizer):
    """Localize the BlueROV2 using the Gazebo ground-truth data."""

    def __init__(self) -> None:
        """Create a new Gazebo localizer."""
        super().__init__("gazebo_localizer")

        # We need to know the topic to stream from
        self.declare_parameter("gazebo_odom_topic", "")

        # Subscribe to that topic so that we can proxy messages to the ArduSub EKF
        # and the Blue EKF
        self.create_subscription(
            Odometry,
            self.get_parameter("gazebo_odom_topic").get_parameter_value().string_value,
            self.proxy_odom_cb,
            1,
        )

    def proxy_odom_cb(self, odom: Odometry) -> None:
        """Proxy the pose data from the Gazebo odometry ground-truth data.

        Args:
            odom: The Gazebo ground-truth odometry for the BlueROV2.
        """
        pose_cov = PoseWithCovarianceStamped()
        pose_cov.header.frame_id = odom.header.frame_id
        pose_cov.header.stamp = odom.header.stamp
        pose_cov.pose = odom.pose

        self.publish(pose_cov)


class Bar30Localizer(PoseLocalizer):
    """Interface for localizing the vehicle using the Blue Robotics Bar30 sensor."""

    def __init__(self) -> None:
        """Create a new localizer for the Bar30 sensor."""
        super().__init__("bar30_localizer")

        self.declare_parameter("water_density", 998.0)  # kg / m^3
        self.declare_parameter("atmospheric_pressure", 101100.0)  # Pa

        water_density = (
            self.get_parameter("water_density").get_parameter_value().double_value
        )
        atmospheric_pressure = (
            self.get_parameter("atmospheric_pressure")
            .get_parameter_value()
            .double_value
        )

        # Subscribe to topic for getting barometer reading
        self.create_subscription(
            FluidPressure,
            "/blue/bar30/pressure",
            lambda msg: self.calculate_and_publish_depth_cb(
                pressure=msg,  # type: ignore
                water_density=water_density,
                atmospheric_pressure=atmospheric_pressure,
            ),
            qos_profile_sensor_data,
        )

    def calculate_and_publish_depth_cb(
        self, pressure: FluidPressure, water_density: float, atmospheric_pressure: float
    ) -> None:
        """Calculate the current depth from the pressure sensor and publish.

        Args:
            pressure: The current pressure reading.
            water_density: The density of the fluid that the vehicle is submerged in.
            atmospheric_pressure: The current atmospheric pressure.
        """
        depth = pressure.fluid_pressure - atmospheric_pressure
        depth /= water_density * 9.85

        pose_cov = PoseWithCovarianceStamped()
        pose_cov.header = pressure.header
        pose_cov.pose.pose.position.z = depth

        # Measurements were evaluated
        cov = np.zeros((6, 6))
        cov[3, 3] = 0.2

        self.publish(pose_cov)


def main_aruco(args: list[str] | None = None):
    """Run the ArUco marker detector."""
    rclpy.init(args=args)

    node = ArucoMarkerLocalizer()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


def main_qualisys(args: list[str] | None = None):
    """Run the Qualisys localizer."""
    rclpy.init(args=args)

    node = QualisysLocalizer()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


def main_bar30(args: list[str] | None = None):
    """Run the Bar30 localizer."""
    rclpy.init(args=args)

    node = Bar30Localizer()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


def main_gazebo(args: list[str] | None = None):
    """Run the Gazebo localizer."""
    rclpy.init(args=args)

    node = GazeboLocalizer()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
