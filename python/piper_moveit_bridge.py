#!/usr/bin/env python3
"""
Piper MoveIt Bridge
===================

Bridge between MoveIt controllers and piper_single_ctrl.

- Exposes:
    /arm_controller/follow_joint_trajectory
    /gripper_controller/follow_joint_trajectory

- For each trajectory point, publishes a JointState to /joint_states.
  piper_single_ctrl already listens on /joint_states and sends CAN.

Joint name mapping:
    - MoveIt uses 'joint7' for gripper
    - piper_single_ctrl expects 'gripper' for gripper
    - Bridge handles the conversion automatically
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState


class PiperMoveItBridge(Node):
    """
    Bridge between MoveIt controllers and piper_single_ctrl.
    """

    def __init__(self):
        super().__init__("piper_moveit_bridge")

        # This is the *command* topic for piper_single_ctrl
        # The launch file remaps joint_ctrl_single -> /joint_states
        self.joint_cmd_pub = self.create_publisher(JointState, "/joint_states", 10)

        # Arm trajectory controller (joints 1-6)
        self.arm_server = ActionServer(
            self,
            FollowJointTrajectory,
            "arm_controller/follow_joint_trajectory",
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
            execute_callback=self.execute_arm_cb,
        )

        # Gripper trajectory controller (joint7 in MoveIt -> gripper in piper)
        self.gripper_server = ActionServer(
            self,
            FollowJointTrajectory,
            "gripper_controller/follow_joint_trajectory",
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
            execute_callback=self.execute_gripper_cb,
        )

        self.get_logger().info("✅ Piper MoveIt bridge is up")
        self.get_logger().info("   - /arm_controller/follow_joint_trajectory")
        self.get_logger().info("   - /gripper_controller/follow_joint_trajectory")
        self.get_logger().info("   - Publishing commands to /joint_states")

    # ---- basic callbacks ----

    def goal_cb(self, goal_request):
        """Accept all trajectory goals"""
        self.get_logger().info("Received FollowJointTrajectory goal")
        return GoalResponse.ACCEPT

    def cancel_cb(self, goal_handle):
        """Accept all cancel requests"""
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    # ---- helpers ----

    def _map_joint_names(self, joint_names):
        """
        Map MoveIt joint names to piper_single_ctrl joint names.
        - joint7 (MoveIt) -> gripper (piper)
        - All other joints stay the same
        """
        mapped = []
        for name in joint_names:
            if name == "joint7":
                mapped.append("gripper")
            else:
                mapped.append(name)
        return mapped

    def _publish_point(self, joint_names, positions):
        """
        Publish a JointState command to /joint_states.
        piper_single_ctrl subscribes to this and sends CAN commands.
        """
        # Map joint names (joint7 -> gripper)
        mapped_names = self._map_joint_names(joint_names)

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = mapped_names
        js.position = list(positions)
        # velocities/effort can be left empty or zero; piper_single_ctrl only uses positions

        self.joint_cmd_pub.publish(js)
        self.get_logger().debug(f"Published joint command: {dict(zip(mapped_names, positions))}")

    # ---- arm trajectory execution ----

    def execute_arm_cb(self, goal_handle):
        """Execute arm trajectory (joints 1-6)"""
        traj = goal_handle.request.trajectory
        self.get_logger().info(
            f"Executing arm trajectory with {len(traj.points)} points "
            f"(joints: {traj.joint_names})"
        )

        last_t = 0.0
        for i, pt in enumerate(traj.points):
            # Calculate time from start
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            dt = max(t - last_t, 0.0)
            last_t = t

            # Sleep for the duration until this point
            if dt > 0.0:
                time.sleep(dt)

            # Publish joint command
            if pt.positions:
                self._publish_point(traj.joint_names, pt.positions)

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = 0  # SUCCESS
        self.get_logger().info("✓ Arm trajectory execution completed")
        return result

    # ---- gripper trajectory execution ----

    def execute_gripper_cb(self, goal_handle):
        """Execute gripper trajectory (joint7 -> gripper)"""
        traj = goal_handle.request.trajectory
        self.get_logger().info(
            f"Executing gripper trajectory with {len(traj.points)} points "
            f"(joints: {traj.joint_names})"
        )

        # For gripper, just take the final point (simpler and faster)
        if traj.points:
            pt = traj.points[-1]
            if pt.positions:
                self._publish_point(traj.joint_names, pt.positions)
                self.get_logger().info(f"✓ Gripper moved to position: {pt.positions[0]:.3f}")

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = 0  # SUCCESS
        return result


def main():
    rclpy.init()
    node = PiperMoveItBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Bridge shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


