#!/usr/bin/env python3
"""
Calculate Link8 (End Effector) Pose from Link7
==============================================

Uses forward kinematics to calculate link8 position from link7 or joint states.
Based on your TF output: link7 is at [0.184, -0.001, 0.192]
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time


class Link8PoseCalculator(Node):
    def __init__(self):
        super().__init__('link8_pose_calculator')
        
        # FK service client
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        
        # Wait for FK service
        self.get_logger().info("Waiting for FK service...")
        if not self.fk_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("FK service not available - using approximate calculation")
            self.calculate_approximate()
            return
        
        # Get joint states
        self.get_logger().info("Getting joint states...")
        js_received = [None]
        
        def js_callback(msg):
            js_received[0] = msg
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        sub = self.create_subscription(JointState, '/joint_states', js_callback, qos)
        
        # Wait for joint state
        for _ in range(50):  # 5 seconds
            rclpy.spin_once(self, timeout_sec=0.1)
            if js_received[0]:
                break
        
        if js_received[0]:
            self.calculate_fk(js_received[0])
        else:
            self.get_logger().warn("No joint state received - using approximate calculation")
            self.calculate_approximate()
    
    def calculate_fk(self, joint_state):
        """Calculate link8 pose using MoveIt FK service"""
        request = GetPositionFK.Request()
        request.header.frame_id = "base_link"
        request.fk_link_names = ["link8"]
        request.robot_state.joint_state = joint_state
        
        self.get_logger().info("Calling FK service...")
        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.result():
            response = future.result()
            if response.error_code.val == 1:  # SUCCESS
                pose = response.pose_stamped[0].pose
                pos = pose.position
                
                self.get_logger().info("=" * 60)
                self.get_logger().info("Link8 (End Effector) Position (from FK):")
                self.get_logger().info(f"  x = {pos.x:.4f} m")
                self.get_logger().info(f"  y = {pos.y:.4f} m")
                self.get_logger().info(f"  z = {pos.z:.4f} m")
                self.get_logger().info("=" * 60)
                
                # Calculate minimal move (1cm offset)
                offset = 0.01
                self.get_logger().info("")
                self.get_logger().info("Minimal test move (1cm offset):")
                self.get_logger().info(f"python3 python/plan_6d_real_robot.py \\")
                self.get_logger().info(f"    --x {pos.x + offset:.3f} \\")
                self.get_logger().info(f"    --y {pos.y + offset:.3f} \\")
                self.get_logger().info(f"    --z {pos.z + offset:.3f} \\")
                self.get_logger().info(f"    --position-only --speed 0.1")
            else:
                self.get_logger().error(f"FK service error: {response.error_code.val}")
                self.calculate_approximate()
        else:
            self.get_logger().error("FK service call failed")
            self.calculate_approximate()
    
    def calculate_approximate(self):
        """Approximate link8 from link7 position"""
        # From your TF output: link7 at [0.184, -0.001, 0.192]
        # Link8 is typically 5-10cm forward along tool axis
        # For approximation, we'll use link7 position + small offset
        link7_x = 0.184
        link7_y = -0.001
        link7_z = 0.192
        
        # Approximate: link8 is ~5cm forward (in +x direction typically)
        link8_x = link7_x + 0.05
        link8_y = link7_y
        link8_z = link7_z
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Approximate Link8 Position (from link7):")
        self.get_logger().info(f"  link7 position: x={link7_x:.4f}, y={link7_y:.4f}, z={link7_z:.4f}")
        self.get_logger().info(f"  link8 position (approx): x={link8_x:.4f}, y={link8_y:.4f}, z={link8_z:.4f}")
        self.get_logger().info("=" * 60)
        
        # Calculate minimal move (1cm offset)
        offset = 0.01
        self.get_logger().info("")
        self.get_logger().info("Minimal test move (1cm offset from link8):")
        self.get_logger().info(f"python3 python/plan_6d_real_robot.py \\")
        self.get_logger().info(f"    --x {link8_x + offset:.3f} \\")
        self.get_logger().info(f"    --y {link8_y + offset:.3f} \\")
        self.get_logger().info(f"    --z {link8_z + offset:.3f} \\")
        self.get_logger().info(f"    --position-only --speed 0.1")


def main():
    rclpy.init()
    node = Link8PoseCalculator()
    try:
        rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()






