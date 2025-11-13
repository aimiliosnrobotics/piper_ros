#!/usr/bin/env python3
"""
Real Robot 6D Pose Planner for Piper Robot
==========================================

Example:
    python3 python/plan_6d_real_robot.py \
        --x 0.25 --y 0.00 --z 0.35 \
        --position-only \
        --speed 0.1 \
        --gripper-before 0.08 \
        --gripper-after 0.0

Launch sequence:
    1) bash can_activate.sh can0 1000000
    2) ros2 launch piper start_single_piper.launch.py \
           can_port:=can0 auto_enable:=false gripper_exist:=true gripper_val_mutiple:=2
    3) ros2 service call /enable_srv piper_msgs/srv/Enable "{enable_request: true}"
    4) ros2 launch piper_with_gripper_moveit piper_moveit_safe.launch.py use_sim_time:=false
    5) python3 python/plan_6d_real_robot.py [...]
"""

import math
import time
import argparse
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


# ---------- CONFIGURE THESE IF NEEDED ----------
ARM_GROUP   = "arm"        # MoveIt planning group name
EE_LINK     = "link8"      # End-effector link name (check in RViz!)
BASE_FRAME  = "base_link"  # Planning frame (as in MoveIt RViz config)
ARM_JOINTS  = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT = "joint7"
# ------------------------------------------------


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


class RealRobotSixDPlanner(Node):
    """
    Real Robot 6D Pose Planner with safety optimizations.
    Uses:
      - /compute_ik  → to get joint goals
      - MoveGroup action → to plan & execute (like RViz Plan&Execute)
    """

    def __init__(self, args):
        super().__init__("real_robot_6d_planner")
        self.args = args

        self.get_logger().info("🤖 Real Robot Mode: Applying safety optimizations")
        self._apply_real_robot_optimizations()

        self.callback_group = ReentrantCallbackGroup()

        # Latest joint state (from real robot / MoveIt)
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
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")

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

        self.get_logger().info("✅ Real robot planner ready!")

    # -------------------------------------------------
    # Safety + common helpers
    # -------------------------------------------------
    def _apply_real_robot_optimizations(self):
        """Clamp parameters for safe real-robot usage."""
        self.args.speed = min(self.args.speed, 0.15)              # max 15% speed
        self.args.allowed_planning_time = min(self.args.allowed_planning_time, 8.0)
        self.args.planning_attempts = max(self.args.planning_attempts, 8)

        if self.args.ompl_planner == "RRTstar":
            self.args.ompl_planner = "RRTConnect"
            self.get_logger().info("Switching to RRTConnect for real robot safety")

        self.get_logger().info(
            f"Real robot settings: speed={self.args.speed:.2f}, "
            f"planning_time={self.args.allowed_planning_time:.1f}s, "
            f"planner={self.args.ompl_planner}"
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

    # -------------------------------------------------
    # Gripper control
    # -------------------------------------------------
    def move_gripper(self, position: float) -> bool:
        """Move gripper (joint7) to position [m]."""
        if not self.grip_ac.server_is_ready():
            self.get_logger().error("Gripper controller not available")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = [GRIPPER_JOINT]

        pt = JointTrajectoryPoint()
        pt.positions = [position]
        pt.time_from_start.sec = 2
        goal.trajectory.points = [pt]

        self.get_logger().info(f"Gripper -> {position:.3f} m ({GRIPPER_JOINT})")

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

    # -------------------------------------------------
    # FK helper (for position-only mode)
    # -------------------------------------------------
    def _get_current_ee_orientation(self) -> Optional[Tuple[float, float, float, float]]:
        """Use /compute_fk to get current EE orientation (if service available)."""
        if self._last_js is None:
            return None
        if not self.fk_client.service_is_ready():
            return None

        req = GetPositionFK.Request()
        req.header.frame_id = BASE_FRAME
        req.fk_link_names = [EE_LINK]
        req.robot_state.joint_state = self._last_js

        future = self.fk_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        res = future.result()
        if res is None or len(res.pose_stamped) == 0:
            return None

        pose = res.pose_stamped[0].pose
        return (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)

    # -------------------------------------------------
    # IK helpers
    # -------------------------------------------------
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
        pose.header.frame_id = BASE_FRAME
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = Point(x=x, y=y, z=z)
        pose.pose.orientation = Quaternion(
            x=q_xyzw[0], y=q_xyzw[1], z=q_xyzw[2], w=q_xyzw[3]
        )

        req = GetPositionIK.Request()
        req.ik_request.group_name = ARM_GROUP
        # Leave ik_link_name empty to use group's tip by default.
        # If you are 100% sure, you can set: req.ik_request.ik_link_name = EE_LINK
        req.ik_request.pose_stamped = pose
        req.ik_request.robot_state.joint_state = self._last_js
        req.ik_request.timeout = Duration(sec=0, nanosec=int(timeout * 1e9))
        # For safety, keep collision checking on
        req.ik_request.avoid_collisions = True

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout + 0.5)
        res = future.result()
        if res is None:
            self.get_logger().error("IK service call failed or timed out")
            return None

        if res.error_code.val != 1:
            # Typically "no IK solution" → target unreachable for that orientation
            self.get_logger().debug(f"IK failed with code {res.error_code.val}")
            return None

        return res.solution.joint_state

    def _compute_ik_position_only(self, x: float, y: float, z: float) -> Optional[JointState]:
        """
        Position-only mode:
        Try many orientations until one IK solution is found.
        """
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
            "  → Either the point is outside the workspace or in unavoidable collision.\n"
            "  → Please verify this (x, y, z) is reachable by dragging the EE in RViz to exactly the same coordinates."
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

    # -------------------------------------------------
    # Build and send MoveGroup joint-space goal
    # -------------------------------------------------
    def _make_joint_goal_from_ik(self, target_js: JointState) -> Optional[MoveGroup.Goal]:
        """Create a MoveGroup joint-space goal from an IK joint_state."""
        goal = MoveGroup.Goal()
        req = goal.request

        req.group_name = ARM_GROUP
        req.num_planning_attempts = self.args.planning_attempts
        req.allowed_planning_time = self.args.allowed_planning_time
        req.max_velocity_scaling_factor = self.args.speed
        req.max_acceleration_scaling_factor = self.args.speed

        if self.args.ompl_planner:
            req.planner_id = self.args.ompl_planner

        c = Constraints()
        joint_map = dict(zip(target_js.name, target_js.position))

        for j in ARM_JOINTS:
            if j not in joint_map:
                self.get_logger().warn(f"Joint {j} not in IK result, skipping constraint for it")
                continue
            jc = JointConstraint()
            jc.joint_name = j
            jc.position = float(joint_map[j])
            # Reasonable tolerance (~0.5 deg)
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
        self.get_logger().info(f"Sending MoveGroup goal (v={self.args.speed:.2f}) ...")

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

    # -------------------------------------------------
    # Main run
    # -------------------------------------------------
    def run(self):
        a = self.args

        # Gripper before move
        if a.gripper_before is not None:
            self.move_gripper(a.gripper_before)

        # Decide if we are in position-only mode
        position_only = a.position_only

        # Orientation (if not position-only)
        q_xyzw: Optional[Tuple[float, float, float, float]] = None
        if not position_only:
            if a.qx is not None or a.qy is not None or a.qz is not None or a.qw is not None:
                qx = a.qx if a.qx is not None else 0.0
                qy = a.qy if a.qy is not None else 0.0
                qz = a.qz if a.qz is not None else 0.0
                qw = a.qw if a.qw is not None else 1.0
                q_xyzw = (qx, qy, qz, qw)
                self.get_logger().info(
                    f"Using direct quaternion: ({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})"
                )
            elif a.roll is not None or a.pitch is not None or a.yaw is not None:
                rr = a.roll if a.roll is not None else 0.0
                pp = a.pitch if a.pitch is not None else 0.0
                yy = a.yaw if a.yaw is not None else 0.0
                q_xyzw = quaternion_from_euler(rr, pp, yy)
                self.get_logger().info(
                    f"Using Euler angles: roll={rr:.3f}, pitch={pp:.3f}, yaw={yy:.3f}"
                )
            else:
                raise RuntimeError(
                    "No orientation specified. Either:\n"
                    "  * use --position-only, OR\n"
                    "  * supply quaternion (qx qy qz qw) or Euler (roll/pitch/yaw)."
                )
        else:
            self.get_logger().info(
                "Position-only mode enabled. Orientation will be chosen automatically by IK search."
            )

        # Build segments (optional via-above)
        segments: List[Tuple[float, float, float]] = []
        if a.via_above > 1e-6:
            segments.append((a.x, a.y, a.z + a.via_above))
        segments.append((a.x, a.y, a.z))

        for i, (tx, ty, tz) in enumerate(segments, start=1):
            self.get_logger().info(f"[{i}/{len(segments)}] Planning to ({tx:.3f}, {ty:.3f}, {tz:.3f})")

            if i > 1:
                self._wait_until_settled()

            if position_only:
                ik_js = self._compute_ik_position_only(tx, ty, tz)
            else:
                ik_js = self._compute_ik_oriented(tx, ty, tz, q_xyzw)  # type: ignore

            if ik_js is None:
                raise RuntimeError("IK failed for target pose")

            goal = self._make_joint_goal_from_ik(ik_js)
            if goal is None:
                raise RuntimeError("Failed to build joint-space goal from IK")

            if not self._send_moveit_goal(goal):
                raise RuntimeError("MoveIt planning/execution failed")

        # Gripper after move
        if a.gripper_after is not None:
            self.get_logger().info("Moving gripper at final position...")
            self.move_gripper(a.gripper_after)

        self.get_logger().info("✅ Real robot motion completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Real Robot 6D Pose Planner for Piper")

    # Position (required)
    parser.add_argument("--x", type=float, required=True, help="Target X position (m)")
    parser.add_argument("--y", type=float, required=True, help="Target Y position (m)")
    parser.add_argument("--z", type=float, required=True, help="Target Z position (m)")

    # Orientation options
    parser.add_argument("--qx", type=float, help="Quaternion X")
    parser.add_argument("--qy", type=float, help="Quaternion Y")
    parser.add_argument("--qz", type=float, help="Quaternion Z")
    parser.add_argument("--qw", type=float, help="Quaternion W")
    parser.add_argument("--roll", type=float, help="Roll (rad)")
    parser.add_argument("--pitch", type=float, help="Pitch (rad)")
    parser.add_argument("--yaw", type=float, help="Yaw (rad)")

    # Motion options
    parser.add_argument(
        "--via-above",
        type=float,
        default=0.0,
        help="Approach from above (m), e.g. 0.05",
    )
    parser.add_argument(
        "--gripper-before",
        type=float,
        help="Gripper before motion (0.0=closed, ~0.08=open)",
    )
    parser.add_argument(
        "--gripper-after",
        type=float,
        help="Gripper after motion (0.0=closed, ~0.08=open)",
    )

    # Position-only flag
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="Don't constrain orientation; search over many orientations in IK.",
    )

    # Real robot safety settings
    parser.add_argument(
        "--speed",
        type=float,
        default=0.1,
        help="Velocity scaling (0..1, default: 0.1)",
    )
    parser.add_argument(
        "--planning-attempts",
        type=int,
        default=8,
        help="Number of planning attempts",
    )
    parser.add_argument(
        "--allowed-planning-time",
        type=float,
        default=8.0,
        help="Max planning time (s)",
    )
    parser.add_argument(
        "--ompl-planner",
        type=str,
        default="RRTConnect",
        choices=["RRTConnect", "RRTstar", "PRMstar"],
        help="OMPL planner to use",
    )

    args = parser.parse_args()

    rclpy.init()
    try:
        planner = RealRobotSixDPlanner(args)
        planner.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
