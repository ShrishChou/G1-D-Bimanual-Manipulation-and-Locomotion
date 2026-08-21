"""Relay the slamware map (VOLATILE, slow) to /map as TRANSIENT_LOCAL (latched) so Nav2's static layer gets it
immediately and reliably. rclpy node."""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid

SRC = "/slamware_ros_sdk_server_node/map"
DST = "/map"


class MapRelay(Node):
    def __init__(self):
        super().__init__("map_relay")
        pub_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sub_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(OccupancyGrid, DST, pub_qos)
        self.create_subscription(OccupancyGrid, SRC, self._cb, sub_qos)
        self.n = 0
        self.get_logger().info(f"relaying {SRC} (volatile) -> {DST} (transient_local latched)")

    def _cb(self, m):
        self.pub.publish(m)
        self.n += 1
        if self.n % 5 == 1:
            self.get_logger().info(f"relayed map #{self.n} ({m.info.width}x{m.info.height})")


def main():
    rclpy.init()
    rclpy.spin(MapRelay())


if __name__ == "__main__":
    main()
