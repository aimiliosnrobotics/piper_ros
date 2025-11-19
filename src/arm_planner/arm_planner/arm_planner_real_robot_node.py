#!/usr/bin/env python3
"""
ROS2 Service Node for Arm Planning on Real Robot
Wraps plan_6d_real_robot.py functionality as a service server

New:
- request.place   -> inverse gripper logic (close before, open after)
- request.vertical -> after reaching the final pose, rotate ONLY joint6 so that
                      link6.x becomes parallel to base_link.y (±), then perform
                      the "after" gripper move.
- Grasp-only sequence (grasp=True, place=False):
    open -> (x-0.10, y, z) -> align link6.x || base_link.y -> (x, y, z) -> close
"""

import math
import time
from typing import Optional, List, Tuple, Dict

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
EE_LINK = "link6"
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

        # IK / planning params
        self.declare_parameter('ik_timeout', 0.3)              # seconds
        self.declare_parameter('ik_attempts', 3)               # kept for future use
        self.declare_parameter('ik_avoid_collisions', True)

        # Joint tolerance (either per-side or a single deg value)
        self.declare_parameter('joint_tolerance_above', 0.02)  # radians
        self.declare_parameter('joint_tolerance_below', 0.02)  # radians
        self.declare_parameter('joint_tolerance_deg', 0.0)     # if >0, overrides both above/below

        # Gripper open/close distances (meters)
        self.declare_parameter('gripper_open', 0.035)
        self.declare_parameter('gripper_close', 0.00)

        # Vertical snap setpoint for joint6 (legacy, unused in new alignment)
        self.declare_parameter('vertical_joint6', 1.528)

        # Approach distance (m) for grasp-only sequence
        self.declare_parameter('approach_offset_x', 0.10)

        # --------- NEW: faster, tunable waits ----------
        # settle time between segments (seconds)
        self.declare_parameter('settle_time', 0.8)
        # geometric alignment tolerance (degrees) for link6.x ∥ base_link.y
        self.declare_parameter('align_tol_deg', 6.0)
        # timeouts (seconds)
        self.declare_parameter('align_wait_timeout', 2.5)
        self.declare_parameter('segment_wait_timeout', 4.0)
        self.declare_parameter('final_wait_timeout', 1.5)
        # -----------------------------------------------

        # Cache a few params that rarely change; others read live in callback
        self.ik_timeout = self.get_parameter('ik_timeout').get_parameter_value().double_value
        self.ik_attempts = int(self.get_parameter('ik_attempts').get_parameter_value().integer_value)
        self.ik_avoid_collisions = self.get_parameter('ik_avoid_collisions').get_parameter_value().bool_value

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

    # ---------- utilities ----------

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

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _on_js(self, msg: JointState):
        """Store latest joint state."""
        self._last_js = msg

    def _wait_until_reached_joint_map(self, target_map: dict, tol: float = 0.02, timeout: float = 1.0) -> bool:
        """Wait until current joints are within tol [rad] of target_map for all ARM_JOINTS."""
        start = time.time()
        last_max_err = None
        while time.time() - start < timeout:
            if self._last_js:
                cur = dict(zip(self._last_js.name, self._last_js.position))
                errs = []
                ok = True
                for j in ARM_JOINTS:
                    if j not in cur or j not in target_map:
                        ok = False
                        break
                    e = abs(cur[j] - float(target_map[j]))
                    errs.append(e)
                    if e > tol:
                        ok = False
                if ok:
                    self.get_logger().info(f"Reached joint target (max err={max(errs):.4f} rad)")
                    return True
                if errs:
                    last_max_err = max(errs)
            rclpy.spin_once(self, timeout_sec=0.02)
        if last_max_err is not None:
            self.get_logger().warn(f"Timed out waiting for joints to reach target (max err≈{last_max_err:.3f} rad)")
        else:
            self.get_logger().warn("Timed out waiting for joints to reach target")
        return False

    def _wait_until_settled(self, still_time: Optional[float] = None):
        """Just wait a bit between segments."""
        if self._last_js is None:
            self.get_logger().warn("No joint state received, skipping settle wait")
            return
        t = still_time if still_time is not None else \
            self.get_parameter('settle_time').get_parameter_value().double_value
        self.get_logger().info(f"Waiting {t:.2f}s for robot to settle...")
        time.sleep(t)

    # ---------- tiny math helpers (kept minimal to avoid side-effects) ----------

    def _wrap_to_pi(self, a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    def _quat_to_rot(self, x: float, y: float, z: float, w: float) -> List[List[float]]:
        # Normalize defensively
        n = math.sqrt(x*x + y*y + z*z + w*w)
        if n < 1e-12:
            return [[1,0,0],[0,1,0],[0,0,1]]
        x, y, z, w = x/n, y/n, z/n, w/n
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        return [
            [1-2*(yy+zz),     2*(xy-wz),     2*(xz+wy)],
            [    2*(xy+wz), 1-2*(xx+zz),     2*(yz-wx)],
            [    2*(xz-wy),     2*(yz+wx), 1-2*(xx+yy)]
        ]

    def _R_mul_v(self, R: List[List[float]], v: List[float]) -> List[float]:
        return [
            R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
            R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
            R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2]
        ]

    def _v_dot(self, a: List[float], b: List[float]) -> float:
        return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

    def _v_sub(self, a: List[float], b: List[float]) -> List[float]:
        return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

    def _v_scale(self, a: List[float], s: float) -> List[float]:
        return [a[0]*s, a[1]*s, a[2]*s]

    def _v_norm(self, a: List[float]) -> float:
        return math.sqrt(self._v_dot(a, a))

    def _v_unit(self, a: List[float]) -> List[float]:
        n = self._v_norm(a)
        return [v/n for v in a] if n > 1e-12 else [0.0, 0.0, 0.0]

    # ---------- gripper ----------

    def move_gripper(self, position: float) -> bool:
        """Move gripper (joint7) to position [m]."""
        if not self.grip_ac.server_is_ready():
            self.get_logger().error("Gripper controller not available")
            return False

        # Safeguard: clamp to a sensible hardware range
        position = self._clamp(position, 0.0, 0.035)  # adjust upper bound to your hardware

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

    # ---------- FK / IK ----------

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
        req.ik_request.ik_link_name = self.ee_link  # ensure IK is solved for EE
        req.ik_request.pose_stamped = pose
        req.ik_request.robot_state.joint_state = self._last_js
        req.ik_request.timeout = Duration(sec=0, nanosec=int(timeout * 1e9))
        req.ik_request.avoid_collisions = self.ik_avoid_collisions

        # NOTE: Many MoveIt2 builds do NOT expose PositionIKRequest.attempts
        # Do NOT set req.ik_request.attempts here.

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
        self.get_logger().info("Position-only mode: sampling orientations for IK.")

        candidates: List[Tuple[float, float, float, float]] = []

        # Prefer useful real-world orientations first (tool down + yaw sweep)
        for yaw in [0.0, math.pi/2, -math.pi/2, math.pi]:
            candidates.append(quaternion_from_euler(0.0, math.pi/2, yaw))

        # Current EE orientation (if FK available)
        cur_q = self._get_current_ee_orientation()
        if cur_q is not None:
            candidates.insert(0, cur_q)

        # Identity and yaw-only (no tilt)
        candidates.append((0.0, 0.0, 0.0, 1.0))
        for yaw in [0.0, math.pi/2, -math.pi/2, math.pi]:
            candidates.append(quaternion_from_euler(0.0, 0.0, yaw))

        # Broaden coverage with extra tilts/rolls
        for pitch in [math.radians(a) for a in (-90, -60, -30, 30, 60, 90)]:
            for yaw in [0.0, math.pi/2, -math.pi/2, math.pi]:
                candidates.append(quaternion_from_euler(0.0, pitch, yaw))
        for roll in [math.pi/2, -math.pi/2]:
            for yaw in [0.0, math.pi/2, -math.pi/2, math.pi]:
                candidates.append(quaternion_from_euler(roll, 0.0, yaw))

        # Try each, with a slightly longer per-sample timeout (scaled from ik_timeout)
        per_sample_timeout = max(0.1, self.ik_timeout * 0.6)

        seen = set()
        for idx, q in enumerate(candidates, start=1):
            # de-duplicate near-identical quats
            key = tuple(round(v, 3) for v in q)
            if key in seen:
                continue
            seen.add(key)

            self.get_logger().info(f"  Trying IK sample {idx}/{len(candidates)} ...")
            js = self._compute_ik_single(x, y, z, q, timeout=per_sample_timeout)
            if js is not None:
                self.get_logger().info("  -> IK sample succeeded")
                return js

        self.get_logger().error(
            "IK failed for all sampled orientations in position-only mode "
            "(likely collision or pose outside reachable set)."
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

    # ---------- MoveIt goal building & sending ----------

    def _make_joint_goal_from_ik(self, target_js: JointState) -> Optional[MoveGroup.Goal]:
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.arm_group
        req.num_planning_attempts = self.planning_attempts
        req.allowed_planning_time = self.allowed_planning_time
        req.max_velocity_scaling_factor = self.speed
        req.max_acceleration_scaling_factor = self.speed
        if self.ompl_planner:
            req.planner_id = self.ompl_planner

        # fetch latest tolerances (supports live tuning via ros2 param set)
        tol_deg = self.get_parameter('joint_tolerance_deg').get_parameter_value().double_value
        if tol_deg > 0.0:
            tol_above = tol_below = math.radians(tol_deg)
        else:
            tol_above = self.get_parameter('joint_tolerance_above').get_parameter_value().double_value
            tol_below = self.get_parameter('joint_tolerance_below').get_parameter_value().double_value

        c = Constraints()
        joint_map = dict(zip(target_js.name, target_js.position))

        for j in ARM_JOINTS:
            if j not in joint_map:
                self.get_logger().warn(f"Joint {j} not in IK result, skipping constraint for it")
                continue
            jc = JointConstraint()
            jc.joint_name = j
            jc.position = float(joint_map[j])
            jc.tolerance_above = float(tol_above)
            jc.tolerance_below = float(tol_below)
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

    def _make_joint_goal_from_positions(self, joint_map: Dict[str, float]) -> Optional[MoveGroup.Goal]:
        """Build a MoveGroup goal from a dict of joint -> target position (rad)."""
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.arm_group
        req.num_planning_attempts = self.planning_attempts
        req.allowed_planning_time = self.allowed_planning_time
        req.max_velocity_scaling_factor = self.speed
        req.max_acceleration_scaling_factor = self.speed
        if self.ompl_planner:
            req.planner_id = self.ompl_planner

        tol_deg = self.get_parameter('joint_tolerance_deg').get_parameter_value().double_value
        if tol_deg > 0.0:
            tol_above = tol_below = math.radians(tol_deg)
        else:
            tol_above = self.get_parameter('joint_tolerance_above').get_parameter_value().double_value
            tol_below = self.get_parameter('joint_tolerance_below').get_parameter_value().double_value

        c = Constraints()
        for j in ARM_JOINTS:
            if j not in joint_map:
                self.get_logger().warn(f"Joint {j} not provided in joint_map, skipping")
                continue
            jc = JointConstraint()
            jc.joint_name = j
            jc.position = float(joint_map[j])
            jc.tolerance_above = float(tol_above)
            jc.tolerance_below = float(tol_below)
            jc.weight = 1.0
            c.joint_constraints.append(jc)

        if not c.joint_constraints:
            self.get_logger().error("No joint constraints created from joint_map")
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

    # ---------- alignment helpers (fast, geometric) ----------

    def _current_link6x_angle_to_base_y(self) -> Optional[float]:
        """
        Returns the smallest angle (rad) between link6.x and ±base_link.y (None if FK unavailable).
        """
        q = self._get_current_ee_orientation()
        if q is None:
            return None
        qx, qy, qz, qw = q
        R = self._quat_to_rot(qx, qy, qz, qw)  # base_link <- link6
        x6 = self._R_mul_v(R, [1, 0, 0])       # link6.x in base_link
        # compare to ±Y: use |dot| to be indifferent to sign
        dot = abs(self._v_dot(self._v_unit(x6), [0, 1, 0]))
        dot = max(-1.0, min(1.0, dot))
        return math.acos(dot)

    def _wait_until_link6x_parallel_base_y(self, tol_deg: float, timeout: float) -> bool:
        """
        Wait until link6.x is within tol_deg of parallel to base_link.y (either +Y or -Y).
        Falls back to success if FK isn't available (to avoid blocking).
        """
        tol_rad = math.radians(tol_deg)
        start = time.time()
        last_ang = None
        while time.time() - start < timeout:
            ang = self._current_link6x_angle_to_base_y()
            if ang is None:
                self.get_logger().warn("FK not available during alignment wait; skipping geometric check.")
                return True  # don't block if FK is down
            last_ang = ang
            if ang <= tol_rad:
                self.get_logger().info(f"link6.x ∥ base_link.y within {tol_deg:.1f}° (Δ={math.degrees(ang):.2f}°)")
                return True
            rclpy.spin_once(self, timeout_sec=0.02)
        if last_ang is not None:
            self.get_logger().warn(f"Alignment timeout (Δ≈{math.degrees(last_ang):.2f}° > {tol_deg:.1f}°)")
        else:
            self.get_logger().warn("Alignment timeout (no FK data)")
        return False

    # ---------- specialized steps ----------

    def _align_joint6_x_to_base_y(self) -> Optional[float]:
        """
        Rotate ONLY joint6 so that link6's X-axis is parallel to base_link's Y-axis (±).
        Returns the target joint6 angle (rad) if a motion was sent (or already aligned),
        otherwise None on failure.
        """
        if self._last_js is None:
            self.get_logger().error("No joint state for alignment")
            return None

        cur_map = dict(zip(self._last_js.name, self._last_js.position))
        if 'joint6' not in cur_map:
            self.get_logger().error("joint6 not in current joint state")
            return None

        # Get current link6 orientation in base_link
        q = self._get_current_ee_orientation()
        if q is None:
            self.get_logger().error("FK not available to get link6 orientation")
            return None
        qx, qy, qz, qw = q
        R = self._quat_to_rot(qx, qy, qz, qw)  # base_link <- link6

        # link6 axes expressed in base_link
        x6 = self._R_mul_v(R, [1,0,0])

        # Assume joint6 axis is z of link6 (typical wrist roll); express in base_link
        a = self._v_unit(self._R_mul_v(R, [0,0,1]))

        # Targets: +Y and -Y of base_link
        t1 = [0, 1, 0]
        t2 = [0,-1, 0]

        # Project onto plane orthogonal to a
        def proj_on_plane(v, n):
            return self._v_sub(v, self._v_scale(n, self._v_dot(v, n)))

        v  = proj_on_plane(x6, a);  v_n  = self._v_unit(v)
        u1 = proj_on_plane(t1, a);  u1_n = self._v_unit(u1)
        u2 = proj_on_plane(t2, a);  u2_n = self._v_unit(u2)

        # If projection is tiny, rotation about a cannot help -> treat as aligned
        if self._v_norm(v) < 1e-6 or self._v_norm(u1) < 1e-6:
            self.get_logger().info("link6.x is nearly parallel to joint6 axis; nothing to align.")
            return cur_map['joint6']

        # Signed angle around axis a from v -> u : atan2(a·(v×u), v·u)
        def signed_angle(vn, un, axis):
            c = self._v_dot(vn, un)
            s = axis[0]*(vn[1]*un[2]-vn[2]*un[1]) + axis[1]*(vn[2]*un[0]-vn[0]*un[2]) + axis[2]*(vn[0]*un[1]-vn[1]*un[0])
            return math.atan2(s, c)

        d1 = self._wrap_to_pi(signed_angle(v_n, u1_n, a))
        d2 = self._wrap_to_pi(signed_angle(v_n, u2_n, a))
        d  = d1 if abs(d1) <= abs(d2) else d2

        # Small tolerance to avoid micro motions (5 deg)
        tol_rad = math.radians(5.0)
        if abs(d) <= tol_rad:
            self.get_logger().info(f"link6.x is already within {math.degrees(tol_rad):.1f}° of base Y (Δ={math.degrees(d):.2f}°).")
            return cur_map['joint6']

        target_j6 = self._wrap_to_pi(cur_map['joint6'] + d)
        arm_joint_map = {j: cur_map[j] for j in ARM_JOINTS if j in cur_map}
        arm_joint_map['joint6'] = float(target_j6)

        self.get_logger().info(f"Aligning link6.x to base Y: Δj6={math.degrees(d):.2f}°, target {target_j6:.3f} rad")
        goal = self._make_joint_goal_from_positions(arm_joint_map)
        if goal is None:
            return None

        if not self._send_moveit_goal(goal):
            return None

        return target_j6

    def _snap_joint6_vertical(self) -> bool:
        """After reaching the target pose, rotate ONLY joint6 to the vertical setpoint."""
        if self._last_js is None:
            self.get_logger().error("No joint state available to snap joint6 vertical")
            return False

        target_j6 = self.get_parameter('vertical_joint6').get_parameter_value().double_value

        # Build a joint map from the latest state and override joint6
        joint_map = dict(zip(self._last_js.name, self._last_js.position))
        if 'joint6' not in joint_map:
            self.get_logger().error("joint6 not found in current joint state")
            return False

        joint_map['joint6'] = float(target_j6)

        # Only constrain the arm joints; ignore gripper joint here
        arm_joint_map = {j: joint_map[j] for j in ARM_JOINTS if j in joint_map}

        self.get_logger().info(f"Snapping joint6 to vertical: {target_j6:.3f} rad")
        goal = self._make_joint_goal_from_positions(arm_joint_map)
        if goal is None:
            return False

        return self._send_moveit_goal(goal)

    # ---------- service ----------

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

            # New flags (require updated .srv with 'place' and 'vertical')
            place = getattr(request, 'place', False)
            vertical = getattr(request, 'vertical', False)

            self.get_logger().info(
                f"Received grasp request: pos=({x:.3f}, {y:.3f}, {z:.3f}), "
                f"orient=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}), "
                f"grasp={request.grasp}, place={place}, vertical={vertical}, "
                f"no_orientation={request.no_orientation}"
            )

            # Read params live (tunable at runtime)
            gripper_open  = self.get_parameter('gripper_open').get_parameter_value().double_value
            gripper_close = self.get_parameter('gripper_close').get_parameter_value().double_value
            approach_dx   = self.get_parameter('approach_offset_x').get_parameter_value().double_value

            align_tol_deg     = self.get_parameter('align_tol_deg').get_parameter_value().double_value
            align_wait_timeout= self.get_parameter('align_wait_timeout').get_parameter_value().double_value
            segment_wait_to   = self.get_parameter('segment_wait_timeout').get_parameter_value().double_value
            final_wait_to     = self.get_parameter('final_wait_timeout').get_parameter_value().double_value

            # Decide gripper strategy
            if place and request.grasp:
                self.get_logger().warn("Both 'place' and 'grasp' are true; prioritizing 'place' behavior.")

            if place:
                # place: start closed, open at target
                gripper_before = gripper_close
                gripper_after  = gripper_open
            elif request.grasp:
                # grasp: start open, close at target
                gripper_before = gripper_open
                gripper_after  = gripper_close
            else:
                gripper_before = None
                gripper_after  = None

            # Move gripper before (if requested)
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

            # ------------- Special grasp-only sequence -------------
            if request.grasp and not place:
                # Step 1: approach to (x - approach_dx, y, z)
                ax = x - approach_dx
                self.get_logger().info(f"[approach] Planning to ({ax:.3f}, {y:.3f}, {z:.3f}) (dx={approach_dx:.3f})")
                if position_only:
                    ik_js = self._compute_ik_position_only(ax, y, z)
                else:
                    ik_js = self._compute_ik_oriented(ax, y, z, q_xyzw)  # type: ignore

                if ik_js is None:
                    response.success = False
                    response.message = "IK failed for approach point"
                    return response

                goal = self._make_joint_goal_from_ik(ik_js)
                if goal is None:
                    response.success = False
                    response.message = "Failed to build goal for approach point"
                    return response
                if not self._send_moveit_goal(goal):
                    response.success = False
                    response.message = "MoveIt failed for approach point"
                    return response

                # Ensure fully reached and settled (short)
                self._wait_until_reached_joint_map(dict(zip(ik_js.name, ik_js.position)), tol=0.02, timeout=segment_wait_to)
                self._wait_until_settled()

                # Step 2: rotate link6.x || base_link.y (alignment)
                target_j6 = self._align_joint6_x_to_base_y()
                if target_j6 is None:
                    response.success = False
                    response.message = "Failed to align joint6 (link6.x || base_link.y)"
                    return response

                # FAST geometric wait instead of joint6 angle wait
                self._wait_until_link6x_parallel_base_y(align_tol_deg, align_wait_timeout)
                self._wait_until_settled()

                # Step 3: final short move to (x, y, z)
                self.get_logger().info(f"[final] Planning to ({x:.3f}, {y:.3f}, {z:.3f})")
                if position_only:
                    ik_js = self._compute_ik_position_only(x, y, z)
                else:
                    ik_js = self._compute_ik_oriented(x, y, z, q_xyzw)  # type: ignore

                if ik_js is None:
                    response.success = False
                    response.message = "IK failed for final target"
                    return response

                goal = self._make_joint_goal_from_ik(ik_js)
                if goal is None:
                    response.success = False
                    response.message = "Failed to build goal for final target"
                    return response
                if not self._send_moveit_goal(goal):
                    response.success = False
                    response.message = "MoveIt failed for final target"
                    return response

                self._wait_until_reached_joint_map(dict(zip(ik_js.name, ik_js.position)), tol=0.02, timeout=segment_wait_to)
                self._wait_until_reached_joint_map(dict(zip(ik_js.name, ik_js.position)), tol=0.3, timeout=final_wait_to)

            # ------------- Default behavior (place or neutral) -------------
            else:
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
                    self._wait_until_reached_joint_map(dict(zip(ik_js.name, ik_js.position)), tol=0.02, timeout=segment_wait_to)

                # Optional post-target "vertical" alignment (same functional intent; faster wait)
                if vertical:
                    target_j6 = self._align_joint6_x_to_base_y()
                    if target_j6 is None:
                        response.success = False
                        response.message = "Failed to align joint6 (link6.x) with base_link.y"
                        return response
                    # geometric wait here too (faster & robust)
                    self._wait_until_link6x_parallel_base_y(align_tol_deg, align_wait_timeout)
                else:
                    self._wait_until_reached_joint_map(dict(zip(ik_js.name, ik_js.position)), tol=0.3, timeout=final_wait_to)

            # Move gripper after (if requested)
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
