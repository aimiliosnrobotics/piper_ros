#!/usr/bin/env python3
"""
Test script to verify the safe configuration works properly.
This script tests both the launch file and the planning script.
"""

import subprocess
import time
import signal
import sys
import os

def test_launch_file():
    """Test that the launch file starts without errors."""
    print("Testing launch file...")
    
    # Start the launch file in background
    process = subprocess.Popen([
        "ros2", "launch", "piper_with_gripper_moveit", "piper_moveit_safe.launch.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait a bit for startup
    time.sleep(10)
    
    # Check if process is still running
    if process.poll() is None:
        print("✓ Launch file started successfully")
        # Kill the process
        process.terminate()
        process.wait(timeout=5)
        return True
    else:
        stdout, stderr = process.communicate()
        print("✗ Launch file failed to start")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        return False

def test_planning_script():
    """Test that the planning script works with simple commands."""
    print("Testing planning script...")
    
    # Test 1: Simple movement without Pilz
    print("  Testing simple movement...")
    result = subprocess.run([
        "python3", "python/plan_pose_safe.py", 
        "--x", "0.22", "--y", "0.15", "--z", "0.25", 
        "--speed", "0.30", "--gripper-before", "0.035", "--gripper-after", "0.000"
    ], capture_output=True, text=True, timeout=30)
    
    if result.returncode == 0:
        print("  ✓ Simple movement test passed")
    else:
        print("  ✗ Simple movement test failed")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    # Test 2: Pilz PTP movement
    print("  Testing Pilz PTP movement...")
    result = subprocess.run([
        "python3", "python/plan_pose_safe.py", 
        "--x", "0.22", "--y", "0.15", "--z", "0.25", 
        "--speed", "0.30", "--ptp-first", "--gripper-before", "0.035", "--gripper-after", "0.000"
    ], capture_output=True, text=True, timeout=30)
    
    if result.returncode == 0:
        print("  ✓ Pilz PTP movement test passed")
    else:
        print("  ✗ Pilz PTP movement test failed")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    return True

def main():
    print("=== Testing Safe Configuration ===")
    
    # Test launch file
    if not test_launch_file():
        print("Launch file test failed. Exiting.")
        sys.exit(1)
    
    # Test planning script
    if not test_planning_script():
        print("Planning script test failed. Exiting.")
        sys.exit(1)
    
    print("=== All tests passed! ===")

if __name__ == "__main__":
    main()
