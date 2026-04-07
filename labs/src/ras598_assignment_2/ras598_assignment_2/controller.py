#!/usr/bin/env python3

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class PathController(Node):
    def __init__(self) -> None:
        super().__init__('controller')

        # ===== Parameters for first working version =====
        self.declare_parameter('path_topic', '/pruned_path')
        self.declare_parameter('odom_topic', '/ground_truth')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.declare_parameter('control_rate', 10.0)
        self.declare_parameter('waypoint_tolerance', 0.15)
        self.declare_parameter('goal_tolerance', 0.12)

        self.declare_parameter('max_linear_speed', 0.35)
        self.declare_parameter('max_angular_speed', 1.2)

        self.declare_parameter('k_linear', 0.8)
        self.declare_parameter('k_angular', 1.8)

        self.declare_parameter('rotate_in_place_angle_deg', 25.0)

        path_topic = self.get_parameter('path_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.control_rate = float(self.get_parameter('control_rate').value)
        self.waypoint_tolerance = float(self.get_parameter('waypoint_tolerance').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)

        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)

        self.k_linear = float(self.get_parameter('k_linear').value)
        self.k_angular = float(self.get_parameter('k_angular').value)

        rotate_deg = float(self.get_parameter('rotate_in_place_angle_deg').value)
        self.rotate_in_place_angle = math.radians(rotate_deg)

        # ===== State =====
        self.current_pose = None  # (x, y, yaw)
        self.path_points: List[Tuple[float, float]] = []
        self.current_waypoint_idx = 0
        self.has_active_path = False
        self.last_path_stamp = None

        # ===== ROS interfaces =====
        self.path_sub = self.create_subscription(
            Path,
            path_topic,
            self.path_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)

        self.get_logger().info('Controller node started.')
        self.get_logger().info(f'  path topic: {path_topic}')
        self.get_logger().info(f'  odom topic: {odom_topic}')
        self.get_logger().info(f'  cmd_vel topic: {cmd_vel_topic}')

    def path_callback(self, msg: Path) -> None:
        if len(msg.poses) == 0:
            self.get_logger().warn('Received empty path. Ignoring.')
            return

        new_points: List[Tuple[float, float]] = []
        for pose_stamped in msg.poses:
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            new_points.append((px, py))

        self.path_points = new_points
        self.current_waypoint_idx = 0
        self.has_active_path = True
        self.last_path_stamp = msg.header.stamp

        self.get_logger().info(
            f'Received new pruned path with {len(self.path_points)} waypoint(s).'
        )

    def odom_callback(self, msg: Odometry) -> None:
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        self.current_pose = (px, py, yaw)

    def publish_stop(self) -> None:
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def control_loop(self) -> None:
        if self.current_pose is None:
            return

        if not self.has_active_path or len(self.path_points) == 0:
            self.publish_stop()
            return

        x, y, yaw = self.current_pose

        # Safety: if index already beyond end, stop
        if self.current_waypoint_idx >= len(self.path_points):
            self.publish_stop()
            self.has_active_path = False
            self.get_logger().info('Path completed.')
            return

        # Current target waypoint
        tx, ty = self.path_points[self.current_waypoint_idx]

        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)

        # Final waypoint uses tighter stopping logic
        is_final_waypoint = (self.current_waypoint_idx == len(self.path_points) - 1)
        tolerance = self.goal_tolerance if is_final_waypoint else self.waypoint_tolerance

        # Advance waypoint if reached
        if dist < tolerance:
            if is_final_waypoint:
                self.publish_stop()
                self.has_active_path = False
                self.get_logger().info('Final goal reached. Stopping.')
                return
            else:
                self.current_waypoint_idx += 1
                self.get_logger().info(
                    f'Advancing to waypoint {self.current_waypoint_idx + 1}/{len(self.path_points)}'
                )
                return

        desired_yaw = math.atan2(dy, dx)
        yaw_error = wrap_to_pi(desired_yaw - yaw)

        cmd = Twist()

        # If heading error is large, rotate first
        if abs(yaw_error) > self.rotate_in_place_angle:
            cmd.linear.x = 0.0
            cmd.angular.z = max(
                -self.max_angular_speed,
                min(self.max_angular_speed, self.k_angular * yaw_error)
            )
        else:
            linear_cmd = self.k_linear * dist
            angular_cmd = self.k_angular * yaw_error

            cmd.linear.x = min(self.max_linear_speed, linear_cmd)
            cmd.angular.z = max(
                -self.max_angular_speed,
                min(self.max_angular_speed, angular_cmd)
            )

        self.cmd_pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()