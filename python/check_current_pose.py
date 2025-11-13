#!/usr/bin/env python3
"""
Check Current End Effector Pose
================================

Queries the current end effector pose via TF and displays it.
Also calculates a small offset for testing minimal moves.
"""

import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PoseStamped, Point
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import math
import time


class PoseChecker(Node):
    def __init__(self):
        super().__init__('pose_checker')
        
        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # IK service for testing
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        
        # Get current joint state
        self._last_js = None
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(JointState, '/joint_states', self._on_js, qos)
        
        self.get_logger().info("Waiting for TF tree and joint states...")
        time.sleep(2.0)  # Give TF time to populate
        
        # Wait for joint state
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._last_js is not None:
                break
        
        # Check current pose
        self.check_current_pose()
    
    def _on_js(self, msg: JointState):
        """Store latest joint state"""
        self._last_js = msg
        
    def check_current_pose(self):
        """Check current end effector pose"""
        # Try multiple frame names
        frames_to_try = ['link8', 'link7', 'world']
        
        transform = None
        used_frame = None
        
        for frame in frames_to_try:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=2.0)
                )
                used_frame = frame
                break
            except TransformException:
                continue
        
        if transform is None:
            # Try without base_link (might be world or different frame)
            try:
                transform = self.tf_buffer.lookup_transform(
                    'world',
                    'link8',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=2.0)
                )
                used_frame = 'link8 (from world)'
            except TransformException:
                pass
        
        if transform:
            trans = transform.transform.translation
            rot = transform.transform.rotation
            
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"Current End Effector Pose ({used_frame}):")
            self.get_logger().info(f"  Position: x={trans.x:.4f}, y={trans.y:.4f}, z={trans.z:.4f} m")
            self.get_logger().info(f"  Orientation: qx={rot.x:.4f}, qy={rot.y:.4f}, qz={rot.z:.4f}, qw={rot.w:.4f}")
            self.get_logger().info("=" * 60)
            
            # Test IK with current pose to verify IK service works
            if self._last_js is not None and self.ik_client.service_is_ready():
                self.get_logger().info("")
                self.get_logger().info("Testing IK with current pose...")
                from geometry_msgs.msg import PoseStamped, Point, Quaternion
                pose = PoseStamped()
                pose.header.frame_id = "base_link"
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position = Point(x=trans.x, y=trans.y, z=trans.z)
                pose.pose.orientation = Quaternion(x=rot.x, y=rot.y, z=rot.z, w=rot.w)
                
                req = GetPositionIK.Request()
                req.ik_request.group_name = "arm"
                req.ik_request.ik_link_name = "link8"
                req.ik_request.pose_stamped = pose
                req.ik_request.timeout = Duration(sec=0, nanosec=int(0.5 * 1e9))
                req.ik_request.avoid_collisions = True
                req.ik_request.robot_state.joint_state = self._last_js
                
                future = self.ik_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                if future.result():
                    res = future.result()
                    if res.error_code.val == 1:
                        self.get_logger().info("  ✓ IK works with current pose")
                    else:
                        self.get_logger().warn(f"  ⚠ IK failed for current pose: {res.error_code.val}")
            
            # Calculate a small offset (2cm) - very minimal
            small_offset = 0.02
            small_x = trans.x + small_offset
            small_y = trans.y
            small_z = trans.z
            
            self.get_logger().info("")
            self.get_logger().info("Minimal test move (2cm forward, same orientation):")
            self.get_logger().info(f"python3 python/plan_6d_real_robot.py \\")
            self.get_logger().info(f"    --x {small_x:.3f} \\")
            self.get_logger().info(f"    --y {small_y:.3f} \\")
            self.get_logger().info(f"    --z {small_z:.3f} \\")
            self.get_logger().info(f"    --qx {rot.x:.4f} \\")
            self.get_logger().info(f"    --qy {rot.y:.4f} \\")
            self.get_logger().info(f"    --qz {rot.z:.4f} \\")
            self.get_logger().info(f"    --qw {rot.w:.4f} \\")
            self.get_logger().info(f"    --speed 0.1")
            
            # Also try position-only with minimal move
            self.get_logger().info("")
            self.get_logger().info("Minimal test move (2cm forward, position-only):")
            self.get_logger().info(f"python3 python/plan_6d_real_robot.py \\")
            self.get_logger().info(f"    --x {small_x:.3f} \\")
            self.get_logger().info(f"    --y {small_y:.3f} \\")
            self.get_logger().info(f"    --z {small_z:.3f} \\")
            self.get_logger().info(f"    --position-only --speed 0.1")
            
            return
        
        # If TF failed, try to get from joint states
        self.get_logger().warn("Could not get pose from TF - trying alternative methods...")
        
        try:
            from sensor_msgs.msg import JointState
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            
            # Subscribe to joint states temporarily
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            
            js_received = [None]
            
            def js_callback(msg):
                js_received[0] = msg
            
            sub = self.create_subscription(JointState, '/joint_states', js_callback, qos)
            
            # Wait for joint state
            for _ in range(20):  # 2 seconds
                rclpy.spin_once(self, timeout_sec=0.1)
                if js_received[0]:
                    break
            
            if js_received[0]:
                js = js_received[0]
                self.get_logger().info("Got joint states, but cannot calculate pose without FK")
                self.get_logger().info("Joint positions: " + str([f"{p:.3f}" for p in js.position[:6]]))
            
        except Exception as e:
            self.get_logger().error(f"Could not get joint states: {e}")
        
        # Final fallback - use the position from the user's TF output
        self.get_logger().error("Could not get transform from TF tree")
        self.get_logger().error("Make sure:")
        self.get_logger().error("  1. Robot hardware is running (Terminal 2)")
        self.get_logger().error("  2. Joint state relay is running (Terminal 3)")
        self.get_logger().error("  3. MoveIt is running (Terminal 5)")
        self.get_logger().error("  4. TF tree is being published")
            
            self.get_logger().info("=" * 60)
            self.get_logger().info("Current End Effector Pose (link8 in base_link frame):")
            self.get_logger().info(f"  Position: x={trans.x:.4f}, y={trans.y:.4f}, z={trans.z:.4f} m")
            self.get_logger().info(f"  Orientation: qx={rot.x:.4f}, qy={rot.y:.4f}, qz={rot.z:.4f}, qw={rot.w:.4f}")
            self.get_logger().info("=" * 60)
            
            # Calculate a small offset (5cm in each direction)
            offset = 0.05
            test_x = trans.x + offset
            test_y = trans.y + offset
            test_z = trans.z + offset
            
            self.get_logger().info("")
            self.get_logger().info("Suggested test command (small move from current position):")
            self.get_logger().info(f"python3 python/plan_6d_real_robot.py \\")
            self.get_logger().info(f"    --x {test_x:.3f} \\")
            self.get_logger().info(f"    --y {test_y:.3f} \\")
            self.get_logger().info(f"    --z {test_z:.3f} \\")
            self.get_logger().info(f"    --position-only --speed 0.1")
            
            # Also try a very small move (1cm)
            small_offset = 0.01
            small_x = trans.x + small_offset
            small_y = trans.y + small_offset
            small_z = trans.z + small_offset
            
            self.get_logger().info("")
            self.get_logger().info("Very small test move (1cm offset):")
            self.get_logger().info(f"python3 python/plan_6d_real_robot.py \\")
            self.get_logger().info(f"    --x {small_x:.3f} \\")
            self.get_logger().info(f"    --y {small_y:.3f} \\")
            self.get_logger().info(f"    --z {small_z:.3f} \\")
            self.get_logger().info(f"    --position-only --speed 0.1")
            
        except TransformException as ex:
            self.get_logger().error(f"Could not transform from base_link to link8: {ex}")
            self.get_logger().error("Make sure:")
            self.get_logger().error("  1. Robot hardware is running (Terminal 2)")
            self.get_logger().error("  2. Joint state relay is running (Terminal 3)")
            self.get_logger().error("  3. MoveIt is running (Terminal 5)")
            self.get_logger().error("  4. TF tree is being published")
            
            # Try to list available frames
            try:
                self.get_logger().info("Available frames in TF tree:")
                frames = self.tf_buffer.all_frames_as_string()
                self.get_logger().info(frames)
            except Exception as e:
                self.get_logger().error(f"Could not list frames: {e}")


def main():
    rclpy.init()
    node = PoseChecker()
    try:
        rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

