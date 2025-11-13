#!/usr/bin/env python3
"""
Piper Joint State Relay
=======================

Relays joint states from robot hardware to MoveIt:
- Subscribes to /joint_states_feedback (from piper_single_ctrl)
- Maps joint names: 'gripper' -> 'joint7'
- Publishes to /joint_states (what MoveIt expects)

This ensures MoveIt receives valid joint state feedback with proper timestamps.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class PiperJointStateRelay(Node):
    """
    Relays joint states from robot hardware to MoveIt.
    """

    def __init__(self):
        super().__init__("piper_joint_state_relay")

        # Subscribe to robot feedback
        self.create_subscription(
            JointState, "/joint_states_feedback", self.joint_state_callback, 10
        )

        # Also try /joint_states_single as fallback
        self.create_subscription(
            JointState, "/joint_states_single", self.joint_state_callback, 10
        )

        # Publish to /joint_states (MoveIt listens here)
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)

        self.get_logger().info("✅ Joint state relay started")
        self.get_logger().info("   - Subscribing to /joint_states_feedback and /joint_states_single")
        self.get_logger().info("   - Publishing to /joint_states")
        self.get_logger().info("   - Mapping: 'gripper' -> 'joint7'")

    def joint_state_callback(self, msg: JointState):
        """
        Relay joint states with joint name mapping.
        """
        # Create new message
        relay_msg = JointState()
        relay_msg.header = msg.header
        relay_msg.header.frame_id = "base_link"  # Standard frame for MoveIt

        # Map joint names: gripper -> joint7
        relay_msg.name = []
        relay_msg.position = []
        relay_msg.velocity = []
        relay_msg.effort = []

        for i, name in enumerate(msg.name):
            # Map 'gripper' to 'joint7' for MoveIt
            if name == "gripper":
                relay_msg.name.append("joint7")
            else:
                relay_msg.name.append(name)

            # Copy position, velocity, effort
            if i < len(msg.position):
                relay_msg.position.append(msg.position[i])
            if i < len(msg.velocity):
                relay_msg.velocity.append(msg.velocity[i])
            if i < len(msg.effort):
                relay_msg.effort.append(msg.effort[i])

        # Ensure arrays are same length
        while len(relay_msg.position) < len(relay_msg.name):
            relay_msg.position.append(0.0)
        while len(relay_msg.velocity) < len(relay_msg.name):
            relay_msg.velocity.append(0.0)
        while len(relay_msg.effort) < len(relay_msg.name):
            relay_msg.effort.append(0.0)

        # Ensure timestamp is valid (use current time if invalid)
        if relay_msg.header.stamp.sec == 0 and relay_msg.header.stamp.nanosec == 0:
            relay_msg.header.stamp = self.get_clock().now().to_msg()
            # Only warn once (use a flag if you want to limit warnings)

        # Publish to /joint_states
        self.joint_state_pub.publish(relay_msg)


def main():
    rclpy.init()
    node = PiperJointStateRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Joint state relay shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

