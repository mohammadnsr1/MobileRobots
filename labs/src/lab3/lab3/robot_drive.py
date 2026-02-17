import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped # Changed
import time

class RobotDrive(Node):
    def __init__(self):
        super().__init__('robot_drive')
        self.publisher_ = self.create_publisher(TwistStamped, '/robot_10/cmd_vel', 10)
        time.sleep(1) 
        self.get_logger().info('Starting Motion Loop...')
        self.execute_shape()

    def create_stamped_msg(self, linear_x=0.0, angular_z=0.0):
        """Helper to create a TwistStamped message with current time."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link' # Common frame for velocity
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        return msg

    def move_forward(self, duration):
        msg = self.create_stamped_msg(linear_x=1.0)
        self.publisher_.publish(msg)
        time.sleep(duration)
        self.stop()

    def turn_robot(self, duration):
        msg = self.create_stamped_msg(angular_z=1.57)
        self.publisher_.publish(msg)
        time.sleep(duration) 
        self.stop()

    def stop(self):
        msg = self.create_stamped_msg() # Defaults to zeros
        self.publisher_.publish(msg)
        time.sleep(0.5)

    def execute_shape(self):
        # implementing the rectangle shape: move forward, turn, move forward, turn, move forward, turn, move forward, turn
        for _ in range(4):
            self.move_forward(2)  # Move forward for 2 seconds
            self.stop()
            self.turn_robot(2)
        pass
    

def main():
    rclpy.init()
    node = RobotDrive()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()