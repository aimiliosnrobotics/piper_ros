#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plan_orient_gripper_moveit.py — MoveIt planner (MoveGroup action) with:
- Speed scaling
- Position box + orientation constraints (Euler or axis-alignment)
- Optional keep-orientation along path
- Via-above approach
- Gripper control before/after

ROS 2 Humble / MoveIt 2
"""

import math
import sys
import time
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint, RobotState
)
from shape_msgs.msg import SolidPrimitive


# ----------------- small math helpers -----------------
def quaternion_from_euler(roll, pitch, yaw) -> Tuple[float,float,float,float]:
    cy = math.cos(yaw * 0.5);  sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5);  sr = math.sin(roll * 0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return (x, y, z, w)

def _normalize(v):
    x,y,z = v
    n = math.sqrt(x*x + y*y + z*z) or 1.0
    return (x/n, y/n, z/n)

def _dot(a,b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def _cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

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
    s = math.sin(angle/2.0)
    return (ax*s, ay*s, az*s, math.cos(angle/2.0))

def axis_vec_from_flag(flag: str) -> Tuple[float,float,float]:
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
        q_align = (0,0,0,1)
    elif abs(dot + 1.0) < 1e-8:
        # 180° around something perpendicular to tz; pick a stable one
        perp = (1,0,0) if abs(tz[0]) < 0.9 else (0,1,0)
        q_align = _quat_from_axis_angle(perp, math.pi)
    else:
        axis = _cross(tz, wd); angle = math.acos(dot)
        q_align = _quat_from_axis_angle(axis, angle)
    q_roll = _quat_from_axis_angle(wd, roll_around_axis)
    return _quat_mul(q_align, q_roll)


# ----------------- main node -----------------
class ProductionPlanner(Node):
    def __init__(self, args):
        super().__init__('production_pose_controller_moveit')
        self.args = args

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

    def _on_js(self, msg: JointState): self._last_js = msg

    def _current_js(self, wait_sec=2.0) -> Optional[JointState]:
        t0 = time.time()
        while self._last_js is None and (time.time() - t0) < wait_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._last_js

    # ---------- constraints builders ----------
    def _position_box(self, center_pose: Pose, half_box: float) -> PositionConstraint:
        """Create a small box around (x,y,z) as a PositionConstraint for tip_link."""
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        # BOX dimensions are full lengths; we use 2*half_box
        box.dimensions = [2*half_box, 2*half_box, 2*half_box]

        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.tip_link
        pc.constraint_region.primitives.append(box)
        pc.constraint_region.primitive_poses.append(center_pose)
        pc.weight = 1.0
        return pc

    def _orientation_goal(self, q_xyzw, tol_deg: float) -> OrientationConstraint:
        qx,qy,qz,qw = q_xyzw
        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.tip_link
        oc.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        tol = math.radians(tol_deg)
        oc.absolute_x_axis_tolerance = tol
        oc.absolute_y_axis_tolerance = tol
        oc.absolute_z_axis_tolerance = tol
        oc.weight = 1.0
        return oc

    def _make_goal_constraints(self, x,y,z, q_xyzw=None) -> Constraints:
        ps = Pose()
        ps.position = Point(x=float(x), y=float(y), z=float(z))
        # the orientation on PositionConstraint pose is irrelevant; it centers the box
        ps.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        c = Constraints()
        c.position_constraints = [ self._position_box(ps, self.args.pos_box) ]
        if q_xyzw is not None:
            c.orientation_constraints = [ self._orientation_goal(q_xyzw, self.args.goal_ang_tol_deg) ]
        return c

    # ---------- planning & execution ----------
    def _send_moveit_goal(self, goal_constraints: Constraints, path_constraints: Optional[Constraints]=None) -> bool:
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.group_name
        req.num_planning_attempts = self.args.planning_attempts
        req.allowed_planning_time = self.args.allowed_planning_time
        req.max_velocity_scaling_factor = max(0.05, min(1.0, self.args.speed))
        req.max_acceleration_scaling_factor = req.max_velocity_scaling_factor

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
        
        # Add trajectory smoothing for smoother motion
        req.workspace_parameters.header.frame_id = self.base_frame
        req.workspace_parameters.header.stamp = self.get_clock().now().to_msg()
        req.workspace_parameters.header.frame_id = self.base_frame

        self.get_logger().info("Sending MoveGroup goal ...")
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
        # simple timing: keep modestly slow and predictable
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
        if a.roll is not None or a.pitch is not None or a.yaw is not None:
            rr = a.roll  if a.roll  is not None else 0.0
            pp = a.pitch if a.pitch is not None else 0.0
            yy = a.yaw   if a.yaw   is not None else 0.0
            q_xyzw = quaternion_from_euler(rr, pp, yy)
            if a.lock_orientation_on_path:
                path_c = Constraints()
                path_c.orientation_constraints = [
                    self._orientation_goal(q_xyzw, a.path_ang_tol_deg)
                ]
        elif a.align_axis_to is not None:
            # Align chosen tip_link axis to a world direction (e.g. face down: --align-axis-to 0 0 -1)
            world_dir = (a.align_axis_to[0], a.align_axis_to[1], a.align_axis_to[2])
            tool_axis = axis_vec_from_flag(a.tool_axis)
            q_xyzw = quat_align_tool_axis_to(world_dir, tool_axis, a.roll_about_axis)
            if a.lock_orientation_on_path:
                path_c = Constraints()
                path_c.orientation_constraints = [
                    self._orientation_goal(q_xyzw, a.path_ang_tol_deg)
                ]
        else:
            # No orientation constraint: planner may choose any orientation
            pass

        # Build segments (via-above then final)
        segments: List[Tuple[float,float,float]] = []
        if a.via_above > 1e-6:
            segments.append((a.x, a.y, a.z + a.via_above))
        segments.append((a.x, a.y, a.z))

        for i,(tx,ty,tz) in enumerate(segments, start=1):
            self.get_logger().info(f"[{i}/{len(segments)}] Plan to ({tx:.3f}, {ty:.3f}, {tz:.3f})")
            gc = self._make_goal_constraints(tx,ty,tz, q_xyzw=q_xyzw)
            if not self._send_moveit_goal(gc, path_constraints=path_c):
                raise RuntimeError("MoveIt planning/execution failed.")

        # Optional: gripper after
        if a.gripper_after is not None:
            self.move_gripper(a.gripper_after)

        self.get_logger().info("Done.")


# ----------------- CLI -----------------
def build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="Plan with MoveIt to a 3D position (and optional orientation), with speed scaling and gripper control.")
    p.add_argument('--x', type=float, required=True)
    p.add_argument('--y', type=float, required=True)
    p.add_argument('--z', type=float, required=True)

    # Orientation as constraints (pick ONE mode)
    # 1) Euler mode
    p.add_argument('--roll',  type=float, default=None, help='Roll (rad) for goal orientation constraint')
    p.add_argument('--pitch', type=float, default=None, help='Pitch (rad) for goal orientation constraint')
    p.add_argument('--yaw',   type=float, default=None, help='Yaw (rad) for goal orientation constraint')

    # 2) Axis-alignment mode (preferred for "face down/up/side")
    p.add_argument('--align-axis-to', nargs=3, type=float, metavar=('VX','VY','VZ'),
                   help='Align chosen tip_link axis to this world direction (e.g. 0 0 -1 for face-down).')
    p.add_argument('--tool-axis', type=str, default='+z',
                   choices=['+x','-x','+y','-y','+z','-z'],
                   help='Which axis of tip_link is the tool/approach axis (default +z).')
    p.add_argument('--roll-about-axis', type=float, default=0.0,
                   help='Spin (rad) about that aligned axis after alignment.')

    # Tolerances
    p.add_argument('--pos-box', type=float, default=0.02, help='Half-size (m) of goal position box.')
    p.add_argument('--goal-ang-tol-deg', type=float, default=4.0, help='Goal orientation tolerance (deg).')
    p.add_argument('--lock-orientation-on-path', action='store_true', help='Constrain orientation along the whole path.')
    p.add_argument('--path-ang-tol-deg', type=float, default=6.0, help='Path orientation tolerance (deg) when locked.')

    # Path shaping
    p.add_argument('--via-above', type=float, default=0.0, help='Approach Z meters above the final target, then descend.')

    # Gripper
    p.add_argument('--gripper-before', type=float, default=None, help='Open/close gripper (m) before moving (0..0.035).')
    p.add_argument('--gripper-after',  type=float, default=None, help='Open/close gripper (m) after moving (0..0.035).')
    p.add_argument('--gripper-joint',  type=str, default='joint7', help='Gripper joint name (default joint7).')

    # Frames / links
    p.add_argument('--base-frame', type=str, default='base_link', help='Pose frame.')
    p.add_argument('--tip-link',   type=str, default='gripper_base', help='EEF link defined in SRDF.')

    # Planning params
    p.add_argument('--speed', type=float, default=0.4, help='MoveIt velocity/acceleration scaling (0..1).')
    p.add_argument('--planning-attempts', type=int, default=5)
    p.add_argument('--allowed-planning-time', type=float, default=5.0)

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
        rclpy.shutdown()
        sys.exit(1)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
