#!/usr/bin/env python3
"""
ROS2 Service Node for Arm Planning on Real Robot
Wraps plan_6d_real_robot.py functionality as a service server
"""

import math
import time
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from builtin_interfaces.msg import Duration

from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from moveit_msgs.msg import Constraints, JointConstraint
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from piper_msgs.srv import GraspFromPose

# Configuration constants
ARM_GROUP = "arm"
EE_LINK = "link8"
BASE_FRAME = "base_link"
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT = "joint7"


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert Euler angles to quaternion (x, y, z, w)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return (qx, qy, qz, qw)


class ArmPlannerRealRobotNode(Node):
    """ROS2 Service Node for arm planning on real robot."""

    def __init__(self):
        super().__init__("arm_planner_real_robot_node")

        self.get_logger().info("🤖 Real Robot Mode: Applying safety optimizations")

        # Declare ROS parameters with defaults
        self.declare_parameter('via_above', 0.0)
        self.declare_parameter('ompl_planner', 'RRTConnect')
        self.declare_parameter('allowed_planning_time', 8.0)
        self.declare_parameter('speed', 0.1)
        self.declare_parameter('planning_attempts', 8)
        self.declare_parameter('ee_link', EE_LINK)
        self.declare_parameter('base_frame', BASE_FRAME)
        self.declare_parameter('arm_group', ARM_GROUP)
        self.declare_parameter('gripper_joint', GRIPPER_JOINT)

        # Get parameters
        self.via_above = self.get_parameter('via_above').get_parameter_value().double_value
        self.ompl_planner = self.get_parameter('ompl_planner').get_parameter_value().string_value
        self.allowed_planning_time = self.get_parameter('allowed_planning_time').get_parameter_value().double_value
        self.speed = self.get_parameter('speed').get_parameter_value().double_value
        self.planning_attempts = self.get_parameter('planning_attempts').get_parameter_value().integer_value
        self.ee_link = self.get_parameter('ee_link').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.arm_group = self.get_parameter('arm_group').get_parameter_value().string_value
        self.gripper_joint = self.get_parameter('gripper_joint').get_parameter_value().string_value

        # Apply real robot optimizations
        self._apply_real_robot_optimizations()

        self.callback_group = ReentrantCallbackGroup()

        # Latest joint state
        self._last_js: Optional[JointState] = None
        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_js,
            10,
            callback_group=self.callback_group,
        )

        # MoveGroup & gripper action clients
        self.move_ac = ActionClient(
            self, MoveGroup, "/move_action", callback_group=self.callback_group
        )
        self.grip_ac = ActionClient(
            self,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
            callback_group=self.callback_group,
        )

        # IK & FK services
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik", callback_group=self.callback_group)
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk", callback_group=self.callback_group)

        # Wait for servers/services
        self.get_logger().info("Waiting for MoveIt move_action...")
        self.move_ac.wait_for_server()
        self.get_logger().info("Waiting for gripper controller...")
        self.grip_ac.wait_for_server()
        self.get_logger().info("Waiting for /compute_ik service...")
        self.ik_client.wait_for_service()
        if self.fk_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("'/compute_fk' service is available")
        else:
            self.get_logger().warn("'/compute_fk' service NOT available - will skip FK-based orientation seed")

        self.get_logger().info("Waiting for joint states...")
        start_time = time.time()
        timeout = 10.0
        while self._last_js is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._last_js is None:
            self.get_logger().warn("⚠️ No joint state received after waiting - IK may fail")
        else:
            self.get_logger().info(f"✅ Received joint state with {len(self._last_js.position)} positions")

        # Create service server
        self.service = self.create_service(
            GraspFromPose,
            'grasp_from_pose',
            self.grasp_from_pose_callback,
            callback_group=self.callback_group
        )
        self.get_logger().info("✅ Arm planner real robot service ready on 'grasp_from_pose'")

    def _apply_real_robot_optimizations(self):
        """Clamp parameters for safe real-robot usage."""
        self.speed = min(self.speed, 0.15)  # max 15% speed
        self.allowed_planning_time = min(self.allowed_planning_time, 8.0)
        self.planning_attempts = max(self.planning_attempts, 8)

        if self.ompl_planner == "RRTstar":
            self.ompl_planner = "RRTConnect"
            self.get_logger().info("Switching to RRTConnect for real robot safety")

        self.get_logger().info(
            f"Real robot settings: speed={self.speed:.2f}, "
            f"planning_time={self.allowed_planning_time:.1f}s, "
            f"planner={self.ompl_planner}"
        )

    def _on_js(self, msg: JointState):
        """Store latest joint state."""
        self._last_js = msg

    def _wait_until_settled(self, still_time: float = 2.0):
        """Just wait a bit between segments."""
        if self._last_js is None:
            self.get_logger().warn("No joint state received, skipping settle wait")
            return
        self.get_logger().info(f"Waiting {still_time}s for robot to settle...")
        time.sleep(still_time)

    def move_gripper(self, position: float) -> bool:
        """Move gripper (joint7) to position [m]."""
        if not self.grip_ac.server_is_ready():
            self.get_logger().error("Gripper controller not available")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = [self.gripper_joint]

        pt = JointTrajectoryPoint()
        pt.positions = [position]
        pt.time_from_start.sec = 2
        goal.trajectory.points = [pt]

        self.get_logger().info(f"Gripper -> {position:.3f} m ({self.gripper_joint})")

        future = self.grip_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            self.get_logger().error("Gripper goal timeout")
            return False

        gh = future.result()
        if not gh.accepted:
            self.get_logger().error("Gripper goal rejected")
            return False

        result_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=10.0)
        if result_future.result() is None:
            self.get_logger().error("Gripper execution timeout")
            return False

        result = result_future.result().result
        if result.error_code != 0:
            self.get_logger().error(f"Gripper execution failed: {result.error_code}")
            return False

        self.get_logger().info("  ✓ gripper moved")
        return True

    def _get_current_ee_orientation(self) -> Optional[Tuple[float, float, float, float]]:
        """Use /compute_fk to get current EE orientation (if service available)."""
        if self._last_js is None:
            return None
        if not self.fk_client.service_is_ready():
            return None

        req = GetPositionFK.Request()
        req.header.frame_id = self.base_frame
        req.fk_link_names = [self.ee_link]
        req.robot_state.joint_state = self._last_js

        future = self.fk_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        res = future.result()
        if res is None or len(res.pose_stamped) == 0:
            return None

        pose = res.pose_stamped[0].pose
        return (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)

    def _compute_ik_single(
        self,
        x: float,
        y: float,
        z: float,
        q_xyzw: Tuple[float, float, float, float],
        timeout: float = 0.05,
    ) -> Optional[JointState]:
        """Single IK call for a given pose. Returns JointState or None."""
        if self._last_js is None:
            self.get_logger().error("No joint state available to seed IK")
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = Point(x=x, y=y, z=z)
        pose.pose.orientation = Quaternion(
            x=q_xyzw[0], y=q_xyzw[1], z=q_xyzw[2], w=q_xyzw[3]
        )

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.arm_group
        req.ik_request.pose_stamped = pose
        req.ik_request.robot_state.joint_state = self._last_js
        req.ik_request.timeout = Duration(sec=0, nanosec=int(timeout * 1e9))
        req.ik_request.avoid_collisions = True

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout + 0.5)
        res = future.result()
        if res is None:
            self.get_logger().error("IK service call failed or timed out")
            return None

        if res.error_code.val != 1:
            self.get_logger().debug(f"IK failed with code {res.error_code.val}")
            return None

        return res.solution.joint_state

    def _compute_ik_position_only(self, x: float, y: float, z: float) -> Optional[JointState]:
        """Position-only mode: Try many orientations until one IK solution is found."""
        self.get_logger().info(
            "Position-only mode: orientation is NOT constrained; sampling multiple orientations for IK."
        )

        candidates: List[Tuple[float, float, float, float]] = []

        # 1) Try current EE orientation first (if we can get it)
        cur_q = self._get_current_ee_orientation()
        if cur_q is not None:
            self.get_logger().info("  Adding current EE orientation as first IK seed")
            candidates.append(cur_q)

        # 2) Some "nice" default orientations
        candidates.append((0.0, 0.0, 0.0, 1.0))  # identity

        # 3) Sample yaw around Z, with no tilt
        yaw_samples = [0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi]
        for yaw in yaw_samples:
            candidates.append(quaternion_from_euler(0.0, 0.0, yaw))

        # 4) Mild tilt in pitch at a few yaws
        tilt = math.radians(30)
        for yaw in [0.0, math.pi / 2, -math.pi / 2]:
            for pitch in (-tilt, tilt):
                candidates.append(quaternion_from_euler(0.0, pitch, yaw))

        # Try each candidate
        for idx, q in enumerate(candidates, start=1):
            self.get_logger().info(f"  Trying IK sample {idx}/{len(candidates)} ...")
            js = self._compute_ik_single(x, y, z, q, timeout=0.05)
            if js is not None:
                self.get_logger().info("  -> IK sample succeeded")
                return js

        self.get_logger().error(
            "IK failed for all sampled orientations in position-only mode.\n"
            "  → Either the point is outside the workspace or in unavoidable collision."
        )
        return None

    def _compute_ik_oriented(
        self, x: float, y: float, z: float, q_xyzw: Tuple[float, float, float, float]
    ) -> Optional[JointState]:
        """Orientation-constrained IK: just call once with the requested orientation."""
        self.get_logger().info(
            f"Orientation mode: using quaternion ({q_xyzw[0]:.3f}, {q_xyzw[1]:.3f}, {q_xyzw[2]:.3f}, {q_xyzw[3]:.3f})"
        )
        js = self._compute_ik_single(x, y, z, q_xyzw, timeout=0.2)
        if js is None:
            self.get_logger().error("IK failed for the requested orientation.")
        return js

    def _make_joint_goal_from_ik(self, target_js: JointState) -> Optional[MoveGroup.Goal]:
        """Create a MoveGroup joint-space goal from an IK joint_state."""
        goal = MoveGroup.Goal()
        req = goal.request

        req.group_name = self.arm_group
        req.num_planning_attempts = self.planning_attempts
        req.allowed_planning_time = self.allowed_planning_time
        req.max_velocity_scaling_factor = self.speed
        req.max_acceleration_scaling_factor = self.speed

        if self.ompl_planner:
            req.planner_id = self.ompl_planner

        c = Constraints()
        joint_map = dict(zip(target_js.name, target_js.position))

        for j in ARM_JOINTS:
            if j not in joint_map:
                self.get_logger().warn(f"Joint {j} not in IK result, skipping constraint for it")
                continue
            jc = JointConstraint()
            jc.joint_name = j
            jc.position = float(joint_map[j])
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)

        if not c.joint_constraints:
            self.get_logger().error("No joint constraints created from IK result")
            return None

        req.goal_constraints = [c]

        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2

        return goal

    def _send_moveit_goal(self, goal: MoveGroup.Goal) -> bool:
        """Send MoveIt goal and wait for execution."""
        self.get_logger().info(f"Sending MoveGroup goal (v={self.speed:.2f}) ...")

        future = self.move_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if future.result() is None:
            self.get_logger().error("MoveIt goal timeout")
            return False

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("MoveIt goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        if result_future.result() is None:
            self.get_logger().error("MoveIt execution timeout")
            return False

        result = result_future.result().result
        if result.error_code.val != 1:
            self.get_logger().error(f"MoveIt planning/execution failed: {result.error_code.val}")
            return False

        self.get_logger().info("  ✓ MoveIt planning + execution succeeded")
        return True

    def grasp_from_pose_callback(self, request: GraspFromPose.Request, response: GraspFromPose.Response):
        """Service callback to handle grasp from pose requests."""
        self.get_logger().info("=" * 60)
        self.get_logger().info("Service callback started")
        try:
            # Extract pose from request
            pose_stamped = request.pose
            pose = pose_stamped.pose
            x = pose.position.x
            y = pose.position.y
            z = pose.position.z

            # Extract orientation
            qx = pose.orientation.x
            qy = pose.orientation.y
            qz = pose.orientation.z
            qw = pose.orientation.w

            self.get_logger().info(
                f"Received grasp request: pos=({x:.3f}, {y:.3f}, {z:.3f}), "
                f"orient=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}), "
                f"grasp={request.grasp}, no_orientation={request.no_orientation}"
            )

            # Set gripper values based on grasp flag
            gripper_before = 0.08 if request.grasp else 0.0
            gripper_after = 0.0 if request.grasp else 0.0

            # Move gripper before
            if gripper_before is not None:
                self.get_logger().info(f"Moving gripper before motion to {gripper_before:.3f} m")
                if not self.move_gripper(gripper_before):
                    response.success = False
                    response.message = "Failed to move gripper before motion"
                    return response
                self.get_logger().info("Gripper moved successfully, proceeding to planning")

            # Decide if we are in position-only mode
            position_only = request.no_orientation

            # Orientation (if not position-only)
            q_xyzw: Optional[Tuple[float, float, float, float]] = None
            if not position_only:
                q_xyzw = (qx, qy, qz, qw)
                self.get_logger().info(
                    f"Using quaternion from pose: ({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})"
                )
            else:
                self.get_logger().info(
                    "Position-only mode enabled. Orientation will be chosen automatically by IK search."
                )

            # Build segments (optional via-above)
            segments: List[Tuple[float, float, float]] = []
            if self.via_above > 1e-6:
                segments.append((x, y, z + self.via_above))
            segments.append((x, y, z))

            self.get_logger().info(f"Planning {len(segments)} segment(s) with via_above={self.via_above:.3f}")

            for i, (tx, ty, tz) in enumerate(segments, start=1):
                self.get_logger().info(f"[{i}/{len(segments)}] Planning to ({tx:.3f}, {ty:.3f}, {tz:.3f})")

                if i > 1:
                    self._wait_until_settled()

                if position_only:
                    ik_js = self._compute_ik_position_only(tx, ty, tz)
                else:
                    ik_js = self._compute_ik_oriented(tx, ty, tz, q_xyzw)  # type: ignore

                if ik_js is None:
                    response.success = False
                    response.message = f"IK failed for target pose at segment {i}/{len(segments)}"
                    return response

                goal = self._make_joint_goal_from_ik(ik_js)
                if goal is None:
                    response.success = False
                    response.message = f"Failed to build joint-space goal from IK at segment {i}/{len(segments)}"
                    return response

                self.get_logger().info(f"Sending MoveIt goal for segment {i}/{len(segments)}...")
                if not self._send_moveit_goal(goal):
                    self.get_logger().error(f"MoveIt planning/execution failed at segment {i}/{len(segments)}")
                    response.success = False
                    response.message = f"MoveIt planning/execution failed at segment {i}/{len(segments)}"
                    return response
                self.get_logger().info(f"Segment {i}/{len(segments)} completed successfully")

            # Move gripper after
            if gripper_after is not None:
                self.get_logger().info("Moving gripper at final position...")
                if not self.move_gripper(gripper_after):
                    response.success = False
                    response.message = "Failed to move gripper after motion"
                    return response

            # Build success message with final pose
            response.success = True
            response.message = (
                f"Position and orientation reached: "
                f"x={x:.3f}, y={y:.3f}, z={z:.3f}, "
                f"qx={qx:.3f}, qy={qy:.3f}, qz={qz:.3f}, qw={qw:.3f}"
            )
            self.get_logger().info(f"✅ {response.message}")
            return response

        except Exception as e:
            import traceback
            self.get_logger().error(f"Error in grasp_from_pose_callback: {str(e)}")
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")
            response.success = False
            response.message = f"Error: {str(e)}"
            return response


def main(args=None):
    rclpy.init(args=args)
    node = ArmPlannerRealRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

