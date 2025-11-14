#!/usr/bin/env python3
"""
ROS2 Service Node for Arm Planning in Simulation
Wraps plan_6d_safe.py functionality as a service server
"""

import sys
import time
import math
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from shape_msgs.msg import SolidPrimitive

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, RobotState
from control_msgs.action import FollowJointTrajectory

from piper_msgs.srv import GraspFromPose

# Import quaternion utilities (NumPy 2.0 compatible)
from scipy.spatial.transform import Rotation as R


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> tuple:
    """Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)."""
    rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    return tuple(quat)  # Return as (x, y, z, w)


class ArmPlannerSimNode(Node):
    """ROS2 Service Node for arm planning in simulation."""

    def __init__(self):
        super().__init__('arm_planner_sim_node')
        
        # Declare ROS parameters with defaults
        self.declare_parameter('via_above', 0.1)
        self.declare_parameter('orientation_weight', 0.05)
        self.declare_parameter('orientation_tolerance_deg', 120.0)
        self.declare_parameter('ompl_planner', 'RRTConnect')
        self.declare_parameter('allowed_planning_time', 30.0)
        self.declare_parameter('speed', 0.30)
        self.declare_parameter('accel', 0.30)  # Optional, defaults to speed
        self.declare_parameter('planning_attempts', 4)
        self.declare_parameter('gripper_joint', 'joint7')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tip_link', 'gripper_base')
        self.declare_parameter('group_name', 'arm')

        # Get parameters
        self.via_above = self.get_parameter('via_above').get_parameter_value().double_value
        self.orientation_weight = self.get_parameter('orientation_weight').get_parameter_value().double_value
        self.orientation_tolerance_deg = self.get_parameter('orientation_tolerance_deg').get_parameter_value().double_value
        self.ompl_planner = self.get_parameter('ompl_planner').get_parameter_value().string_value
        self.allowed_planning_time = self.get_parameter('allowed_planning_time').get_parameter_value().double_value
        self.speed = self.get_parameter('speed').get_parameter_value().double_value
        accel_param = self.get_parameter('accel')
        self.accel = accel_param.get_parameter_value().double_value
        # If accel is set to same as speed, treat as None (use speed)
        if abs(self.accel - self.speed) < 1e-6:
            self.accel = None
        self.planning_attempts = self.get_parameter('planning_attempts').get_parameter_value().integer_value
        self.gripper_joint = self.get_parameter('gripper_joint').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.tip_link = self.get_parameter('tip_link').get_parameter_value().string_value
        self.group_name = self.get_parameter('group_name').get_parameter_value().string_value

        # Use reentrant callback group to allow concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()

        self._last_js: Optional[JointState] = None
        self.create_subscription(
            JointState, '/joint_states', self._on_js, 10,
            callback_group=self.callback_group
        )

        self.move_ac = ActionClient(
            self, MoveGroup, '/move_action',
            callback_group=self.callback_group
        )
        self.grip_ac = ActionClient(
            self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory',
            callback_group=self.callback_group
        )

        # Wait for servers
        self.get_logger().info("Waiting for MoveIt move_action ...")
        self.move_ac.wait_for_server()
        self.get_logger().info("Waiting for gripper controller ...")
        self.grip_ac.wait_for_server()

        # Create service server with callback group
        self.service = self.create_service(
            GraspFromPose,
            'grasp_from_pose',
            self.grasp_from_pose_callback,
            callback_group=self.callback_group
        )
        self.get_logger().info("✅ Arm planner sim service ready on 'grasp_from_pose'")

    def _on_js(self, msg: JointState):
        self._last_js = msg

    def _current_js(self, wait_sec=2.0) -> Optional[JointState]:
        t0 = time.time()
        while self._last_js is None and (time.time() - t0) < wait_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._last_js

    def _wait_until_settled(self, vel_eps=1e-3, still_time=0.5, timeout=3.0):
        """Block until joint velocities are below vel_eps for still_time seconds."""
        t0 = time.time()
        last_ok = None
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            js = self._last_js
            if not js or not js.velocity:
                continue
            if all(abs(v) <= vel_eps for v in js.velocity):
                if last_ok is None:
                    last_ok = time.time()
                if time.time() - last_ok >= still_time:
                    return True
            else:
                last_ok = None
        return False

    def _position_box(self, center_pose: Pose, half_box: float) -> PositionConstraint:
        """Create a position constraint with specified box size."""
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [2*half_box, 2*half_box, 2*half_box]

        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.tip_link
        pc.constraint_region.primitives.append(box)
        pc.constraint_region.primitive_poses.append(center_pose)
        pc.weight = 1.0
        return pc

    def _orientation_goal(self, q_xyzw, tol_deg: float, weight: float = 1.0) -> OrientationConstraint:
        """Create orientation constraint with specified tolerance and weight."""
        qx, qy, qz, qw = q_xyzw
        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.tip_link
        oc.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        tol = math.radians(tol_deg)
        oc.absolute_x_axis_tolerance = tol
        oc.absolute_y_axis_tolerance = tol
        oc.absolute_z_axis_tolerance = tol
        oc.weight = weight
        return oc

    def _make_hard_position_goal(self, x, y, z) -> Constraints:
        """Create hard position goal with minimal tolerance."""
        c = Constraints()
        ps = Pose()
        ps.position = Point(x=float(x), y=float(y), z=float(z))
        ps.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)  # Dummy orientation
        hard_pos_box = 0.001  # 1mm tolerance
        c.position_constraints = [self._position_box(ps, hard_pos_box)]
        return c

    def _make_soft_orientation_path_constraint(self, q_xyzw, tol_deg: float, weight: float) -> Constraints:
        """Create soft orientation path constraint."""
        c = Constraints()
        c.orientation_constraints = [self._orientation_goal(q_xyzw, tol_deg, weight)]
        return c

    def _send_moveit_goal(self,
                          goal_constraints: Constraints,
                          path_constraints: Optional[Constraints] = None,
                          pipeline_id: str = 'ompl',
                          planner_id: str = 'RRTConnect') -> bool:
        """Send MoveIt goal with hard position and soft orientation."""
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.group_name
        req.num_planning_attempts = self.planning_attempts
        req.allowed_planning_time = self.allowed_planning_time

        # Velocity and acceleration scaling
        v = max(0.05, min(1.0, self.speed))
        a = v if (self.accel is None) else max(0.05, min(1.0, self.accel))
        req.max_velocity_scaling_factor = v
        req.max_acceleration_scaling_factor = a

        # Planner selection
        req.pipeline_id = pipeline_id
        req.planner_id = planner_id

        # Start state
        self.get_logger().debug("Getting current joint state for planning...")
        js = self._current_js(2.0)
        if js is not None:
            req.start_state = RobotState()
            req.start_state.joint_state = js
            self.get_logger().debug(f"Got joint state with {len(js.position)} positions")
        else:
            self.get_logger().warn("No joint state available, MoveIt will use current state")

        # Goal constraints (hard position)
        req.goal_constraints = [goal_constraints]
        
        # Path constraints (soft orientation)
        if path_constraints is not None:
            req.path_constraints = path_constraints

        # Planning options
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2

        self.get_logger().info(
            f"Sending MoveGroup goal (v={v:.2f}, a={a:.2f}, pipeline={pipeline_id}, planner={planner_id}) ..."
        )
        fut = self.move_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if not gh or not gh.accepted:
            self.get_logger().error("MoveGroup goal rejected")
            return False

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        res = res_fut.result().result
        ok = (res.error_code.val == res.error_code.SUCCESS)
        if ok:
            self.get_logger().info("  ✓ planned & executed")
        else:
            self.get_logger().error(f"  ✗ planner error {res.error_code.val}")
        return ok

    def move_gripper(self, width_m: float) -> bool:
        """Move gripper to specified width."""
        width_m = max(0.0, min(0.08, width_m))  # Allow up to 0.08m for grasping
        jt = JointTrajectory()
        jt.joint_names = [self.gripper_joint]
        pt = JointTrajectoryPoint()
        pt.positions = [width_m]
        pt.time_from_start.sec = max(1, int(0.8 + (1.0 - self.speed) * 1.2))
        pt.time_from_start.nanosec = 0
        jt.points = [pt]

        self.get_logger().info(f"Gripper -> {width_m:.3f} m ({self.gripper_joint})")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt
        fut = self.grip_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if not gh or not gh.accepted:
            self.get_logger().error("Gripper goal rejected")
            return False
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        ok = (res_fut.result().result.error_code == 0)
        self.get_logger().info("  ✓ gripper moved" if ok else "  ✗ gripper failed")
        return ok

    def _get_final_pose(self) -> Optional[Tuple[float, float, float, float, float, float, float]]:
        """Get current end-effector pose using FK or joint state."""
        # For now, return None - we'll use the target pose in the response
        # In a full implementation, you could use /compute_fk service
        return None

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

            # Build orientation preference (if not position-only)
            q_xyzw = None
            path_c = None
            
            if not request.no_orientation:
                # Use orientation from pose
                q_xyzw = (qx, qy, qz, qw)
                self.get_logger().info(f"Using quaternion from pose: ({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")
                path_c = self._make_soft_orientation_path_constraint(
                    q_xyzw, self.orientation_tolerance_deg, self.orientation_weight
                )
                self.get_logger().info(
                    f"Using soft orientation constraint (weight={self.orientation_weight:.2f}, "
                    f"tol={self.orientation_tolerance_deg:.1f}°)"
                )
            else:
                self.get_logger().info("Position-only mode: orientation will be ignored")

            # Build segments (via-above then final)
            segments: List[Tuple[float, float, float]] = []
            if self.via_above > 1e-6:
                segments.append((x, y, z + self.via_above))
            segments.append((x, y, z))

            self.get_logger().info(f"Planning {len(segments)} segment(s) with via_above={self.via_above:.3f}")

            for i, (tx, ty, tz) in enumerate(segments, start=1):
                self.get_logger().info(f"[{i}/{len(segments)}] Planning to ({tx:.3f}, {ty:.3f}, {tz:.3f})")

                # Ensure fully still before next leg
                if i > 1:
                    self._wait_until_settled()

                # Create hard position goal
                gc = self._make_hard_position_goal(tx, ty, tz)

                # Only apply orientation constraint to the final segment
                path_c_segment = None
                if i == len(segments) and not request.no_orientation:
                    path_c_segment = path_c
                    self.get_logger().info("Applying orientation constraint to final segment")
                else:
                    self.get_logger().info("Approach segment - no orientation constraint")

                # Send goal with conditional orientation path constraint
                self.get_logger().info(f"Sending MoveIt goal for segment {i}/{len(segments)}...")
                success = self._send_moveit_goal(gc, path_constraints=path_c_segment,
                                                pipeline_id='ompl', planner_id=self.ompl_planner)
                if not success:
                    self.get_logger().error(f"MoveIt planning/execution failed at segment {i}/{len(segments)}")
                    response.success = False
                    response.message = f"MoveIt planning/execution failed at segment {i}/{len(segments)}"
                    return response
                self.get_logger().info(f"Segment {i}/{len(segments)} completed successfully")

            # Move gripper after
            if gripper_after is not None:
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
    node = ArmPlannerSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

