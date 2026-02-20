import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool
import random
import math

# CHOICE: X
class ReactiveNavigator(Node):
    def __init__(self):
        super().__init__('reactive_navigator')
        
        # 1. Setup Service
        self.srv = self.create_service(SetBool, 'toggle_navigation', self.toggle_callback)
        self.active = False

        # 2. Setup Pub/Sub
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # 3. Parameter
        self.declare_parameter('safety_distance', 0.5)

    def toggle_callback(self, request, response):
        # Service Callback
        self.active = request.data
        if not self.active:
            stop_msg = Twist()
            self.publisher_.publish(stop_msg)
        
        response.success = True
        response.message = f"Navigation set to {self.active}"
        return response

    def get_index_for_angle(self, msg, angle_deg):
        # Helper to convert degrees to msg index
        pass

    def scan_callback(self, msg):
        if not self.active:
            return

        # TODO: Calculate indices for your specific Member role
        pass

    def execute_rotation_maneuver(self):
        # TODO: Implement Member A, B, or C rotation logic here
        self.get_logger().info("Obstacle Detected! Executing rotation maneuver.")
        
        pass

    def drive_forward(self):
        msg = Twist()
        msg.linear.x = 0.05 
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveNavigator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
