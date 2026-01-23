import rclpy 
from rclpy.node import Node

from std_msgs.msg import String


class lab1_subscriber(Node):

    def __init__(self):
        super().__init__('lab1_subscriber')
        self.subscriber = self.create_subscription(String, 'lab1_topic', self.subscriber_callback, 10)
        self.get_logger().info('Lab1Subscriber has been started')

    def subscriber_callback(self, msg):
        self.get_logger().info('I heard: %s' % msg.data)


def main(args = None):
    rclpy.init(args = args)
    lab1_subscriber_node = lab1_subscriber()
    rclpy.spin(lab1_subscriber_node)
    lab1_subscriber_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()