import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import time

#option A : Rectangle
class RobotDrive(Node):
    def __init__(self):
        super().__init__('robot_drive')
        self.publisher_ = self.create_publisher(TwistStamped, '/robot_10/cmd_vel', 10)

        time.sleep(1.0)

        self.linear_x = 0.0
        self.angular_z = 0.0

        self.get_logger().info('Starting Motion Loop...')
        self.execute_shape()
        self.get_logger().info('Done.')

    def create_stamped_msg(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(self.linear_x)
        msg.twist.angular.z = float(self.angular_z)
        return msg

    def move_forward(self, duration, hz=100.0):
        self.linear_x = 1.0
        self.angular_z = 0.0
        dt = 1.0 / hz
        start = time.time()
        while time.time() - start < duration:
            self.publisher_.publish(self.create_stamped_msg())
            time.sleep(dt)
        self.stop()

    def turn_robot(self, duration, hz=100.0):
        self.linear_x = 0.0
        self.angular_z = 1.57
        dt = 1.0 / hz
        start = time.time()
        while time.time() - start < duration:
            self.publisher_.publish(self.create_stamped_msg())
            time.sleep(dt)
        self.stop()

    def stop(self, hz=100.0, hold=0.3):
        self.linear_x = 0.0
        self.angular_z = 0.0
        dt = 1.0 / hz
        start = time.time()
        while time.time() - start < hold:
            self.publisher_.publish(self.create_stamped_msg())
            time.sleep(dt)

    def execute_shape(self):
        big = 1.0
        small = 0.5
        turn_90 = 1.0
        for _ in range(2):
            self.move_forward(big)
            self.stop()
            self.turn_robot(turn_90)
            self.stop()
            self.move_forward(small)
            self.stop()
            self.turn_robot(turn_90)

def main():
    rclpy.init()
    node = RobotDrive()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
