#!/usr/bin/env python3
"""
6D Pose Planning with Hard Position + Soft Orientation
Based on plan_pose_safe.py but optimized for OMPL with clear 6D targets
"""

import sys
import time
import math
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from shape_msgs.msg import SolidPrimitive

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, RobotState
from control_msgs.action import FollowJointTrajectory

# Import quaternion utilities (NumPy 2.0 compatible)
from scipy.spatial.transform import Rotation as R

def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> tuple:
    """Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)."""
    rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    return tuple(quat)  # Return as (x, y, z, w)

def euler_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> tuple:
    """Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)."""
    rot = R.from_quat([qx, qy, qz, qw])
    euler = rot.as_euler('xyz', degrees=False)
    return tuple(euler)  # Return as (roll, pitch, yaw)


# ---------- quaternion utilities ----------
def axis_vec_from_flag(flag: str):
    """Convert tool axis flag to unit vector."""
    return {
        '+x': (1, 0, 0), '-x': (-1, 0, 0),
        '+y': (0, 1, 0), '-y': (0, -1, 0),
        '+z': (0, 0, 1), '-z': (0, 0, -1)
    }[flag]

def quat_align_tool_axis_to(world_dir, tool_axis, roll_about_axis=0.0):
    """Compute quaternion to align tool_axis with world_dir."""
    # Normalize inputs
    wd = [x / math.sqrt(sum(x*x for x in world_dir)) for x in world_dir]
    ta = [x / math.sqrt(sum(x*x for x in tool_axis)) for x in tool_axis]
    
    # Compute rotation axis (cross product)
    axis = [wd[1]*ta[2] - wd[2]*ta[1], wd[2]*ta[0] - wd[0]*ta[2], wd[0]*ta[1] - wd[1]*ta[0]]
    axis_norm = math.sqrt(sum(x*x for x in axis))
    
    if axis_norm < 1e-6:  # Parallel vectors
        if sum(wd[i]*ta[i] for i in range(3)) > 0:  # Same direction
            return (0, 0, 0, 1)  # Identity
        else:  # Opposite direction
            # Find perpendicular vector for rotation
            if abs(wd[0]) < 0.9:
                perp = [1, 0, 0]
            else:
                perp = [0, 1, 0]
            axis = [wd[1]*perp[2] - wd[2]*perp[1], wd[2]*perp[0] - wd[0]*perp[2], wd[0]*perp[1] - wd[1]*perp[0]]
            axis_norm = math.sqrt(sum(x*x for x in axis))
            axis = [x / axis_norm for x in axis]
            angle = math.pi
    else:
        axis = [x / axis_norm for x in axis]
        angle = math.acos(max(-1, min(1, sum(wd[i]*ta[i] for i in range(3)))))
    
    # Convert to quaternion
    s = math.sin(angle/2)
    c = math.cos(angle/2)
    qx, qy, qz, qw = axis[0]*s, axis[1]*s, axis[2]*s, c
    
    # Apply roll about the aligned axis
    if abs(roll_about_axis) > 1e-6:
        roll_q = quaternion_from_euler(roll_about_axis, 0, 0)
        # Compose quaternions: roll_q * align_q
        qx_new = roll_q[3]*qx + roll_q[0]*qw + roll_q[1]*qz - roll_q[2]*qy
        qy_new = roll_q[3]*qy + roll_q[1]*qw + roll_q[2]*qx - roll_q[0]*qz
        qz_new = roll_q[3]*qz + roll_q[2]*qw + roll_q[0]*qy - roll_q[1]*qx
        qw_new = roll_q[3]*qw - roll_q[0]*qx - roll_q[1]*qy - roll_q[2]*qz
        qx, qy, qz, qw = qx_new, qy_new, qz_new, qw_new
    
    return (qx, qy, qz, qw)


# ---------- main node -----------------
class SixDPlanner(Node):
    def __init__(self, args):
        super().__init__('six_d_pose_controller_moveit_safe')
        self.args = args

        # Optimize for real robot if requested
        if args.real_robot:
            self.get_logger().info("Real robot mode: optimizing for safety and reliability")
            # Reduce speed for safety
            args.speed = min(args.speed, 0.2)
            # Increase planning time
            args.allowed_planning_time = max(args.allowed_planning_time, 10.0)
            # Increase planning attempts
            args.planning_attempts = max(args.planning_attempts, 6)

        self._last_js: Optional[JointState] = None
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)

        self.move_ac  = ActionClient(self, MoveGroup, '/move_action')
        self.grip_ac  = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')

        self.group_name = 'arm'
        self.tip_link   = args.tip_link
        self.base_frame = args.base_frame

        # Wait servers
        self.get_logger().info("Waiting for MoveIt move_action ...")
        self.move_ac.wait_for_server()
        self.get_logger().info("Waiting for gripper controller ...")
        self.grip_ac.wait_for_server()

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

    # ---------- constraint builders ----------
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
        
        # Hard position constraint with very small tolerance
        ps = Pose()
        ps.position = Point(x=float(x), y=float(y), z=float(z))
        ps.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)  # Dummy orientation
        
        # Use very small box for hard position target
        hard_pos_box = 0.001  # 1mm tolerance
        c.position_constraints = [self._position_box(ps, hard_pos_box)]
        
        return c

    def _make_soft_orientation_path_constraint(self, q_xyzw, tol_deg: float, weight: float) -> Constraints:
        """Create soft orientation path constraint."""
        c = Constraints()
        c.orientation_constraints = [self._orientation_goal(q_xyzw, tol_deg, weight)]
        return c

    # ---------- planning & execution ----------
    def _send_moveit_goal(self,
                          goal_constraints: Constraints,
                          path_constraints: Optional[Constraints] = None,
                          pipeline_id: str = 'ompl',
                          planner_id: str = 'RRTstar') -> bool:
        """Send MoveIt goal with hard position and soft orientation."""
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.group_name
        req.num_planning_attempts = self.args.planning_attempts
        req.allowed_planning_time = self.args.allowed_planning_time

        # Velocity and acceleration scaling
        v = max(0.05, min(1.0, self.args.speed))
        a = v if (self.args.accel is None) else max(0.05, min(1.0, self.args.accel))
        req.max_velocity_scaling_factor = v
        req.max_acceleration_scaling_factor = a

        # Planner selection
        req.pipeline_id = pipeline_id
        req.planner_id = planner_id

        # Start state
        js = self._current_js(2.0)
        if js is not None:
            req.start_state = RobotState()
            req.start_state.joint_state = js

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

    # ---------- gripper ----------
    def move_gripper(self, width_m: float) -> bool:
        """Move gripper to specified width."""
        width_m = max(0.0, min(0.035, width_m))
        jt = JointTrajectory()
        jt.joint_names = [self.args.gripper_joint]
        pt = JointTrajectoryPoint()
        pt.positions = [width_m]
        pt.time_from_start.sec = max(1, int(0.8 + (1.0 - self.args.speed) * 1.2))
        pt.time_from_start.nanosec = 0
        jt.points = [pt]

        self.get_logger().info(f"Gripper -> {width_m:.3f} m ({self.args.gripper_joint})")
        goal = FollowJointTrajectory.Goal(); goal.trajectory = jt
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

    # ---------- main flow ----------
    def run(self):
        a = self.args

        # Optional: gripper before
        if a.gripper_before is not None:
            self.move_gripper(a.gripper_before)

        # Build orientation preference (if requested)
        q_xyzw = None
        path_c = None
        
        # Check for direct quaternion input
        if a.qx is not None or a.qy is not None or a.qz is not None or a.qw is not None:
            qx = a.qx if a.qx is not None else 0.0
            qy = a.qy if a.qy is not None else 0.0
            qz = a.qz if a.qz is not None else 0.0
            qw = a.qw if a.qw is not None else 1.0
            q_xyzw = (qx, qy, qz, qw)
            self.get_logger().info(f"Using direct quaternion: ({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")
        # Check for euler angles
        elif a.roll is not None or a.pitch is not None or a.yaw is not None:
            rr = a.roll  if a.roll  is not None else 0.0
            pp = a.pitch if a.pitch is not None else 0.0
            yy = a.yaw   if a.yaw   is not None else 0.0
            q_xyzw = quaternion_from_euler(rr, pp, yy)
            self.get_logger().info(f"Using euler angles: roll={rr:.3f}, pitch={pp:.3f}, yaw={yy:.3f}")

        # Create soft orientation path constraint if orientation is specified
        if q_xyzw is not None:
            orientation_weight = a.orientation_weight
            orientation_tolerance = a.orientation_tolerance_deg
            path_c = self._make_soft_orientation_path_constraint(q_xyzw, orientation_tolerance, orientation_weight)
            self.get_logger().info(f"Using soft orientation constraint (weight={orientation_weight:.2f}, tol={orientation_tolerance:.1f}°)")

        # Build segments (via-above then final)
        segments: List[Tuple[float, float, float]] = []
        if a.via_above > 1e-6:
            segments.append((a.x, a.y, a.z + a.via_above))
        segments.append((a.x, a.y, a.z))

        for i, (tx, ty, tz) in enumerate(segments, start=1):
            self.get_logger().info(f"[{i}/{len(segments)}] Plan to ({tx:.3f}, {ty:.3f}, {tz:.3f})")

            # Ensure fully still before next leg
            if i > 1:
                self._wait_until_settled()

            # Create hard position goal
            gc = self._make_hard_position_goal(tx, ty, tz)

            # OPTIMIZATION: Only apply orientation constraint to the final segment
            path_c_segment = None
            if i == len(segments):  # Apply soft constraint ONLY to the final segment
                path_c_segment = path_c
                self.get_logger().info("Applying orientation constraint to final segment")
            else:
                self.get_logger().info("Approach segment - no orientation constraint")

            # Send goal with conditional orientation path constraint
            if not self._send_moveit_goal(gc, path_constraints=path_c_segment,
                                          pipeline_id='ompl', planner_id=a.ompl_planner):
                raise RuntimeError("MoveIt planning/execution failed.")

        # Optional: gripper after
        if a.gripper_after is not None:
            self.move_gripper(a.gripper_after)

        self.get_logger().info("Done.")


# ----------------- CLI -----------------
def build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="6D Pose Planning with Hard Position + Soft Orientation.")
    
    # Position (required)
    p.add_argument('--x', type=float, required=True)
    p.add_argument('--y', type=float, required=True)
    p.add_argument('--z', type=float, required=True)

    # Orientation (optional) - Simple quaternion input
    p.add_argument('--qx', type=float, default=None, help='Quaternion X component')
    p.add_argument('--qy', type=float, default=None, help='Quaternion Y component')
    p.add_argument('--qz', type=float, default=None, help='Quaternion Z component')
    p.add_argument('--qw', type=float, default=None, help='Quaternion W component')
    
    # Alternative: Euler angles (simpler)
    p.add_argument('--roll',  type=float, default=None, help='Roll (rad)')
    p.add_argument('--pitch', type=float, default=None, help='Pitch (rad)')
    p.add_argument('--yaw',   type=float, default=None, help='Yaw (rad)')

    # Soft orientation parameters
    p.add_argument('--orientation-weight', type=float, default=0.1,
                   help='Weight for soft orientation constraint (0.0=ignore, 1.0=strict)')
    p.add_argument('--orientation-tolerance-deg', type=float, default=90.0,
                   help='Tolerance for soft orientation constraint (degrees)')

    # Path shaping
    p.add_argument('--via-above', type=float, default=0.0, help='Z meters above final, then descend.')

    # Gripper
    p.add_argument('--gripper-before', type=float, default=None, help='0..0.035 m')
    p.add_argument('--gripper-after',  type=float, default=None, help='0..0.035 m')
    p.add_argument('--gripper-joint',  type=str, default='joint7')

    # Frames / links
    p.add_argument('--base-frame', type=str, default='base_link')
    p.add_argument('--tip-link',   type=str, default='gripper_base')

    # Planning params
    p.add_argument('--speed', type=float, default=0.3, help='Velocity scaling (0..1).')
    p.add_argument('--accel', type=float, default=None, help='Acceleration scaling (0..1). If unset, equals --speed.')
    p.add_argument('--planning-attempts', type=int, default=4)
    p.add_argument('--allowed-planning-time', type=float, default=30.0,
                   help='Maximum planning time in seconds (default: 30.0 for testing)')
    
    # Real robot options
    p.add_argument('--real-robot', action='store_true', 
                   help='Optimize settings for real robot (increases tolerances, slower speeds)')
    p.add_argument('--safety-margin', type=float, default=0.02,
                   help='Safety margin for collision avoidance (meters)')

    # Planner selection
    p.add_argument('--ompl-planner', type=str, default='RRTstar',
                   choices=['RRTConnect', 'RRTstar', 'PRMstar', 'EST', 'KPIECE1'],
                   help='OMPL planner to use (RRTConnect recommended for orientation constraints)')

    return p


def main():
    args = build_argparser().parse_args()
    rclpy.init()
    node = SixDPlanner(args)
    try:
        node.run()
        sys.exit(0)
    except Exception as e:
        node.get_logger().error(str(e))
        sys.exit(1)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
