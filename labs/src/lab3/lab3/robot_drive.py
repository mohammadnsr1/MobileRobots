import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import numpy as np

class RobotDrive(Node):
    def __init__(self):
        super().__init__('robot_drive')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        time.sleep(1) # Wait for publisher to register
        self.get_logger().info('Starting Motion Loop...')
        self.execute_shape()

    def move_forward(self, duration):
        msg = Twist()
        msg.linear.x = 0.2  # 0.2 m/s
        self.publisher_.publish(msg)
        time.sleep(duration)
        self.stop()

    def turn_robot(self, duration):
        msg = Twist()
        msg.angular.z = 0.5 # rad/s
        self.publisher_.publish(msg)
        time.sleep(duration) 
        self.stop()

    def stop(self):
        self.publisher_.publish(Twist())
        time.sleep(0.5)

    def execute_shape(self):
        # implementing the rectangle shape: move forward, turn, move forward, turn, move forward, turn, move forward, turn
        for _ in range(4):
            self.move_forward(2)  # Move forward for 2 seconds
            self.turn_robot(1.57)  # Turn 90 degrees (1.57 radians)
        pass
    

def main():
    rclpy.init()
    node = RobotDrive()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()