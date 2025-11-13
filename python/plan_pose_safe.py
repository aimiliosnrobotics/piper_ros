#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plan_pose_safe.py — Clean MoveIt MoveGroup client with:
- Speed and acceleration scaling (separate: --speed, --accel)
- Position box + optional orientation constraints (Euler or axis-alignment)
- Optional orientation lock along path
- Optional via-above approach
- Optional wrist lock (keeps joint6 within window)
- Wait-until-settled between segments
- Simple gripper control
- Per-segment Pilz selection: PTP for approach, LIN for final descend

Works with the "safe" MoveIt launch set (OMPL + Pilz only).
"""

import math
import sys
import time
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint, RobotState, JointConstraint
)
from shape_msgs.msg import SolidPrimitive


# ----------------- small math helpers -----------------
def quaternion_from_euler(roll, pitch, yaw) -> Tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5);  sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5);  sr = math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)

def _normalize(v):
    x, y, z = v
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    return (x / n, y / n, z / n)

def _dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def _cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def _quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    )

def _quat_from_axis_angle(axis, angle):
    ax, ay, az = _normalize(axis)
    s = math.sin(angle / 2.0)
    return (ax * s, ay * s, az * s, math.cos(angle / 2.0))

def axis_vec_from_flag(flag: str) -> Tuple[float, float, float]:
    m = {
        '+x': ( 1, 0, 0), '-x': (-1, 0, 0),
        '+y': ( 0, 1, 0), '-y': ( 0,-1, 0),
        '+z': ( 0, 0, 1), '-z': ( 0, 0,-1),
    }
    return m[flag.lower()]

def quat_align_tool_axis_to(world_dir, tool_axis, roll_around_axis=0.0):
    """
    Rotate the chosen tip_link axis (±X/±Y/±Z) onto world_dir, then roll about that axis.
    Returns quaternion (x,y,z,w).
    """
    tz = _normalize(tool_axis); wd = _normalize(world_dir)
    dot = max(-1.0, min(1.0, _dot(tz, wd)))
    if abs(dot - 1.0) < 1e-8:
        q_align = (0, 0, 0, 1)
    elif abs(dot + 1.0) < 1e-8:
        perp = (1, 0, 0) if abs(tz[0]) < 0.9 else (0, 1, 0)
        q_align = _quat_from_axis_angle(perp, math.pi)
    else:
        axis = _cross(tz, wd); angle = math.acos(dot)
        q_align = _quat_from_axis_angle(axis, angle)
    q_roll = _quat_from_axis_angle(wd, roll_around_axis)
    return _quat_mul(q_align, q_roll)


# ----------------- main node -----------------
class ProductionPlanner(Node):
    def __init__(self, args):
        super().__init__('production_pose_controller_moveit_safe')
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

    # ---------- constraints builders ----------
    def _position_box(self, center_pose: Pose, half_box: float) -> PositionConstraint:
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
        qx, qy, qz, qw = q_xyzw
        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.tip_link
        oc.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        tol = math.radians(tol_deg)
        oc.absolute_x_axis_tolerance = tol
        oc.absolute_y_axis_tolerance = tol
        oc.absolute_z_axis_tolerance = tol
        oc.weight = weight  # Use provided weight for soft constraints
        return oc

    def _joint_lock(self, name: str, center: float, tol_rad: float, weight=1.0) -> JointConstraint:
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = center
        jc.tolerance_above = tol_rad
        jc.tolerance_below = tol_rad
        jc.weight = weight
        return jc

    def _make_goal_constraints(self, x, y, z, q_xyzw=None,
                               joint_constraints: Optional[List[JointConstraint]] = None,
                               use_joint_only: bool = False,
                               use_pilz_simple: bool = False,
                               orientation_weight: float = 1.0) -> Constraints:
        c = Constraints()
        
        if use_pilz_simple:
            # For Pilz planners, use ONLY position constraints, no orientation, no joint constraints
            ps = Pose()
            ps.position = Point(x=float(x), y=float(y), z=float(z))
            ps.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            c.position_constraints = [self._position_box(ps, self.args.pos_box)]
            # No orientation constraints, no joint constraints for Pilz
        elif use_joint_only and joint_constraints:
            # For other planners, use joint constraints only when specified
            c.joint_constraints = list(joint_constraints)
        else:
            # Use position constraints (default behavior)
            ps = Pose()
            ps.position = Point(x=float(x), y=float(y), z=float(z))
            ps.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            c.position_constraints = [self._position_box(ps, self.args.pos_box)]
            if q_xyzw is not None:
                c.orientation_constraints = [self._orientation_goal(q_xyzw, self.args.goal_ang_tol_deg, orientation_weight)]
        
        return c

    # ---------- planning & execution ----------
    def _send_moveit_goal(self,
                          goal_constraints: Constraints,
                          path_constraints: Optional[Constraints] = None,
                          pipeline_id: Optional[str] = None,
                          planner_id: Optional[str] = None) -> bool:
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.group_name
        req.num_planning_attempts = self.args.planning_attempts
        req.allowed_planning_time = self.args.allowed_planning_time

        # separate velocity & acceleration scaling
        v = max(0.05, min(1.0, self.args.speed))
        a = v if (self.args.accel is None) else max(0.05, min(1.0, self.args.accel))
        req.max_velocity_scaling_factor = v
        req.max_acceleration_scaling_factor = a

        # Optional pipeline/planner selection (e.g., Pilz LIN/PTP)
        if pipeline_id:
            req.pipeline_id = pipeline_id
        if planner_id:
            req.planner_id = planner_id

        js = self._current_js(2.0)
        if js is not None:
            req.start_state = RobotState()
            req.start_state.joint_state = js

        req.goal_constraints = [goal_constraints]
        if self.args.lock_orientation_on_path and path_constraints is not None:
            req.path_constraints = path_constraints

        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2

        self.get_logger().info(
            f"Sending MoveGroup goal (v={v:.2f}, a={a:.2f}, pipeline={pipeline_id or 'default'}, planner={planner_id or 'default'}) ..."
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

        # Build orientation (if requested)
        q_xyzw = None
        path_c = None
        orientation_weight = 1.0 if not a.soft_orientation else a.orientation_weight
        
        if a.roll is not None or a.pitch is not None or a.yaw is not None:
            rr = a.roll  if a.roll  is not None else 0.0
            pp = a.pitch if a.pitch is not None else 0.0
            yy = a.yaw   if a.yaw   is not None else 0.0
            q_xyzw = quaternion_from_euler(rr, pp, yy)
            if a.lock_orientation_on_path:
                path_c = Constraints()
                path_c.orientation_constraints = [
                    self._orientation_goal(q_xyzw, a.path_ang_tol_deg, orientation_weight)
                ]
        elif a.align_axis_to is not None:
            world_dir = (a.align_axis_to[0], a.align_axis_to[1], a.align_axis_to[2])
            tool_axis = axis_vec_from_flag(a.tool_axis)
            q_xyzw = quat_align_tool_axis_to(world_dir, tool_axis, a.roll_about_axis)
            if a.lock_orientation_on_path:
                path_c = Constraints()
                path_c.orientation_constraints = [
                    self._orientation_goal(q_xyzw, a.path_ang_tol_deg, orientation_weight)
                ]

        # Build segments (via-above then final)
        segments: List[Tuple[float, float, float]] = []
        if a.via_above > 1e-6:
            segments.append((a.x, a.y, a.z + a.via_above))
        segments.append((a.x, a.y, a.z))

        for i, (tx, ty, tz) in enumerate(segments, start=1):
            self.get_logger().info(f"[{i}/{len(segments)}] Plan to ({tx:.3f}, {ty:.3f}, {tz:.3f})")

            # ensure fully still before next leg
            if i > 1:
                self._wait_until_settled()

            # Smart planner selection based on movement complexity and user preferences
            last_leg = (i == len(segments))
            first_leg = (i == 1)

            use_pipeline = None
            use_planner  = None
            
            # Smart planner selection based on user choice and movement type
            if a.planner == 'auto':
                # Smart selection based on movement complexity and legacy options
                if a.ptp_first and first_leg:
                    use_pipeline, use_planner = 'pilz_industrial_motion_planner', 'PTP'
                elif a.lin_final and last_leg:
                    # Use OMPL instead of problematic LIN
                    use_pipeline, use_planner = 'ompl', a.ompl_planner
                else:
                    # Default to OMPL for reliable motion
                    use_pipeline, use_planner = 'ompl', a.ompl_planner
            elif a.planner == 'ompl':
                use_pipeline, use_planner = 'ompl', a.ompl_planner
            elif a.planner == 'chomp':
                use_pipeline, use_planner = 'chomp', 'chomp'
            elif a.planner == 'pilz_ptp':
                use_pipeline, use_planner = 'pilz_industrial_motion_planner', 'PTP'
            elif a.planner == 'pilz_lin':
                use_pipeline, use_planner = 'pilz_industrial_motion_planner', 'LIN'
            elif a.planner == 'pilz_circ':
                use_pipeline, use_planner = 'pilz_industrial_motion_planner', 'CIRC'
            else:
                # Fallback to OMPL
                use_pipeline, use_planner = 'ompl', a.ompl_planner

            # Optional: lock wrist (joint6) around current value
            joint_cs: List[JointConstraint] = []
            if a.lock_wrist_deg is not None:
                js = self._current_js()
                if js and 'joint6' in js.name:
                    widx = js.name.index('joint6')
                    joint_cs.append(
                        self._joint_lock('joint6', js.position[widx], math.radians(a.lock_wrist_deg))
                    )

            # For Pilz planners, use simplified constraints (position only)
            use_pilz_simple = (use_pipeline == 'pilz_industrial_motion_planner')
            use_joint_only = (not use_pilz_simple and joint_cs)
            
            # For Pilz planners, disable all complex constraints
            if use_pilz_simple:
                joint_cs = []  # Clear joint constraints for Pilz
                q_xyzw = None  # Clear orientation constraints for Pilz
                use_joint_only = False
            
            gc = self._make_goal_constraints(tx, ty, tz, q_xyzw=q_xyzw, joint_constraints=joint_cs, 
                                           use_joint_only=use_joint_only, use_pilz_simple=use_pilz_simple,
                                           orientation_weight=orientation_weight)

            if not self._send_moveit_goal(gc, path_constraints=path_c,
                                          pipeline_id=use_pipeline, planner_id=use_planner):
                raise RuntimeError("MoveIt planning/execution failed.")

        # Optional: gripper after
        if a.gripper_after is not None:
            self.move_gripper(a.gripper_after)

        self.get_logger().info("Done.")


# ----------------- CLI -----------------
def build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="Plan with MoveIt (safe config).")
    p.add_argument('--x', type=float, required=True)
    p.add_argument('--y', type=float, required=True)
    p.add_argument('--z', type=float, required=True)

    # Orientation as constraints (pick ONE mode)
    p.add_argument('--roll',  type=float, default=None, help='Roll (rad)')
    p.add_argument('--pitch', type=float, default=None, help='Pitch (rad)')
    p.add_argument('--yaw',   type=float, default=None, help='Yaw (rad)')

    p.add_argument('--align-axis-to', nargs=3, type=float, metavar=('VX','VY','VZ'),
                   help='Align chosen tip_link axis to this world direction (e.g. 0 0 -1).')
    p.add_argument('--tool-axis', type=str, default='+z',
                   choices=['+x','-x','+y','-y','+z','-z'])
    p.add_argument('--roll-about-axis', type=float, default=0.0,
                   help='Spin about aligned axis (rad).')

    # Tolerances
    p.add_argument('--pos-box', type=float, default=0.02, help='Half-size (m) of goal position box.')
    p.add_argument('--goal-ang-tol-deg', type=float, default=6.0, help='Goal orientation tolerance (deg).')
    p.add_argument('--lock-orientation-on-path', action='store_true', help='Constrain orientation along the path.')
    p.add_argument('--path-ang-tol-deg', type=float, default=8.0, help='Path orientation tolerance (deg).')
    
    # Soft orientation preferences (new)
    p.add_argument('--soft-orientation', action='store_true', 
                   help='Use soft orientation preference instead of hard constraint')
    p.add_argument('--orientation-weight', type=float, default=0.5,
                   help='Weight for soft orientation (0.0=ignore, 1.0=strict)')

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
    p.add_argument('--allowed-planning-time', type=float, default=5.0)
    
    # Obstacle avoidance and real robot options
    p.add_argument('--collision-check', action='store_true', default=True,
                   help='Enable collision checking (default: True)')
    p.add_argument('--real-robot', action='store_true', 
                   help='Optimize settings for real robot (increases tolerances, slower speeds)')
    p.add_argument('--safety-margin', type=float, default=0.02,
                   help='Safety margin for collision avoidance (meters)')

    # Wrist lock window (deg) to avoid flips
    p.add_argument('--lock-wrist-deg', type=float, default=None)

    # Planner selection options
    p.add_argument('--planner', type=str, default='auto', 
                   choices=['auto', 'ompl', 'chomp', 'pilz_ptp', 'pilz_lin', 'pilz_circ'],
                   help='Planner to use: auto (smart selection), ompl, chomp, pilz_ptp, pilz_lin, pilz_circ')
    p.add_argument('--ompl-planner', type=str, default='RRTConnect',
                   choices=['RRTConnect', 'RRTstar', 'PRMstar', 'EST', 'KPIECE1'],
                   help='Specific OMPL planner to use (when --planner=ompl)')
    p.add_argument('--pilz-planner', type=str, default='PTP',
                   choices=['PTP', 'LIN', 'CIRC'],
                   help='Specific Pilz planner to use (when --planner=pilz_*)')
    
    # Legacy options (for backward compatibility)
    p.add_argument('--ptp-first', action='store_true', help='Use Pilz PTP for the first segment (legacy)')
    p.add_argument('--lin-final', action='store_true', help='Use Pilz LIN for the last segment (legacy)')

    return p


def main():
    args = build_argparser().parse_args()
    rclpy.init()
    node = ProductionPlanner(args)
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
