import rclpy
from rclpy.node import Node

from std_msgs.msg import String 

class Lab1_Publisher(Node):

    def __init__(self):
        super().__init__('lab1_publisher')
        self.publisher = self.create_publisher(String, 'lab1_topic', 10)
        self.timer_ = self.create_timer(0.5,self.timer_callback)
        self.get_logger().info('Lab1Publisher has been started')
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello world! %d' % self.i
        self.i += 1
        self.publisher.publish(msg)
        self.get_logger().info('Published message: %s' % msg.data)



def main(args = None):
    rclpy.init(args = args)
    lab1_publisher = Lab1_Publisher()
    rclpy.spin(lab1_publisher)
    lab1_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

