import math

import rclpy
from rclpy.node import Node

from example_interfaces.srv import Trigger
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/planner_markers', 10)

        # Subscribers
        self.gt_sub = self.create_subscription(
            Odometry, '/ground_truth', self.ground_truth_callback, 10
        )
        self.energy_sub = self.create_subscription(
            Float32, '/energy_consumed', self.energy_callback, 10
        )

        # Service client
        self.task_client = self.create_client(Trigger, '/get_task')

        self.current_pose = None
        self.current_yaw = None
        self.energy = None

        self.start = None
        self.goal = None

        self.get_logger().info('Waiting for /get_task service...')
        while not self.task_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/get_task not available yet...')

        self.request_task()

    def request_task(self):
        req = Trigger.Request()
        future = self.task_client.call_async(req)
        future.add_done_callback(self.handle_task_response)

    def handle_task_response(self, future):
        try:
            response = future.result()
            if not response.success:
                self.get_logger().error('Task service returned success=False')
                return

            # Expected format: "start_x,start_y,goal_x,goal_y"
            values = [float(v.strip()) for v in response.message.split(',')]
            if len(values) != 4:
                self.get_logger().error(f'Unexpected task format: {response.message}')
                return

            sx, sy, gx, gy = values
            self.start = (sx, sy)
            self.goal = (gx, gy)

            self.get_logger().info(f'Start: {self.start}')
            self.get_logger().info(f'Goal:  {self.goal}')

        except Exception as e:
            self.get_logger().error(f'Failed to parse task response: {e}')

    def ground_truth_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # yaw from quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.current_pose = (x, y)
        self.current_yaw = yaw

    def energy_callback(self, msg: Float32):
        self.energy = msg.data


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()