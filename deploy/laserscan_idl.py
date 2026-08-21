"""Local sensor_msgs/LaserScan IDL (the SDK ships PointCloud2 but not LaserScan). Matches the ROS2 field
layout so DDS can deserialize rt/slamware_ros_sdk_server_node/scan."""
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class LaserScan_(idl.IdlStruct, typename="sensor_msgs.msg.dds_.LaserScan_"):
    header: Header_
    angle_min: types.float32
    angle_max: types.float32
    angle_increment: types.float32
    time_increment: types.float32
    scan_time: types.float32
    range_min: types.float32
    range_max: types.float32
    ranges: types.sequence[types.float32]
    intensities: types.sequence[types.float32]
