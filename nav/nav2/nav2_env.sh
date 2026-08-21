# Source this to talk to the robot's SLAM over ROS2:  source nav2_env.sh
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI="file://$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../deploy/cyclonedds.xml"
echo "[nav2_env] RMW=$RMW_IMPLEMENTATION DOMAIN=$ROS_DOMAIN_ID LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"
echo "[nav2_env] CYCLONEDDS_URI=$CYCLONEDDS_URI"
