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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate a launch description for the localization package.

    Returns:
        The localization ROS 2 launch description.
    """
    args = [
        DeclareLaunchArgument(
            "config_filepath",
            default_value=None,
            description="The path to the localization configuration YAML file.",
        ),
        DeclareLaunchArgument(
            "sources",
            default_value="['qualisys_mocap']",
            description=(
                "The localization sources to stream from. Multiple sources can"
                " be loaded at once. If no sources need to be launched, then this can"
                " be set to an empty list."
            ),
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use the simulated Gazebo clock.",
        ),
    ]

    sources = LaunchConfiguration("sources")
    use_sim_time = LaunchConfiguration("use_sim_time")

    nodes = [
        Node(
            package="blue_localization",
            executable="camera",
            name="camera",
            output="screen",
            parameters=[
                LaunchConfiguration("config_filepath"),
                {"use_sim_time": use_sim_time},
            ],
            condition=IfCondition(PythonExpression(["'camera' in ", sources])),
        ),
        Node(
            package="blue_localization",
            executable="qualisys_mocap",
            name="qualisys_mocap",
            output="screen",
            parameters=[
                LaunchConfiguration("config_filepath"),
                {"use_sim_time": use_sim_time},
            ],
            condition=IfCondition(PythonExpression(["'qualisys_mocap' in ", sources])),
        ),
        Node(
            package="blue_localization",
            executable="bar30",
            name="bar30",
            output="screen",
            parameters=[
                LaunchConfiguration("config_filepath"),
                {"use_sim_time": use_sim_time},
            ],
            condition=IfCondition(PythonExpression(["'bar30' in ", sources])),
        ),
    ]

    return LaunchDescription(args + nodes)
