import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
import os

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

        self.map_resolution = 0.032
        self.map_origin_x = -8.0
        self.map_origin_y = -8.0
        self.grid_resolution = 0.2
        self.map_width = 500   # replace with actual loaded image width
        self.map_height = 500  # replace with actual loaded image height
        self.grid_resolution = 0.2

        pkg_share = get_package_share_directory('ras598_assignment_2')

        image_path = os.path.join(pkg_share,'maps', 'cave_filled.png')

        map_img = self.load_map(image_path)

        self.get_logger().info('Waiting for /get_task service...')
        while not self.task_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/get_task not available yet...')

        self.request_task()


        inflated = self.inflate_map(map_img)
        self.occupancy_grid = self.downsample_to_grid(inflated)
        

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

            gsx, gsy = self.world_to_grid(sx, sy)
            ggx, ggy = self.world_to_grid(gx, gy)

            self.get_logger().info(f'Start cell: {gsx}, {gsy}')
            self.get_logger().info(f'Goal cell: {ggx}, {ggy}')

            self.get_logger().info(f"Start occupied? {self.occupancy_grid[gsx, gsy]}")
            self.get_logger().info(f"Goal occupied? {self.occupancy_grid[ggy, ggx]}")
            

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

    def world_to_pixel(self, x, y):
        u = int((x - self.map_origin_x) / self.map_resolution)
        v = int(self.map_height - (y - self.map_origin_y) / self.map_resolution)
        return u, v

    def pixel_to_world(self, u, v):
        x = self.map_origin_x + u * self.map_resolution
        y = self.map_origin_y + (self.map_height - v) * self.map_resolution
        return x, y

    def world_to_grid(self, x, y):
        gx = int((x - self.map_origin_x) / self.grid_resolution)
        gy = int((y - self.map_origin_y) / self.grid_resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = self.map_origin_x + (gx + 0.5) * self.grid_resolution
        y = self.map_origin_y + (gy + 0.5) * self.grid_resolution
        return x, y

    def load_map(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"Failed to load map image: {image_path}")

        self.map_height, self.map_width = img.shape
        self.get_logger().info(f"Map size: {self.map_width} x {self.map_height}")

        # obstacle = 1 , free = 0
        _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        occupancy = (binary > 0).astype(np.uint8)

        return occupancy

    def inflate_map(self, occupancy):
        inflation_radius_pixels = int(0.6 / self.map_resolution)

        kernel_size = 2 * inflation_radius_pixels + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        inflated = cv2.dilate(occupancy, kernel)

        return inflated

    def downsample_to_grid(self, inflated):
        world_width = self.map_width * self.map_resolution
        world_height = self.map_height * self.map_resolution

        grid_w = int(round(world_width / self.grid_resolution))
        grid_h = int(round(world_height / self.grid_resolution))

        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

        for gy in range(grid_h):
            for gx in range(grid_w):
                # World bounds of this planning cell
                wx_min = self.map_origin_x + gx * self.grid_resolution
                wx_max = wx_min + self.grid_resolution

                wy_min = self.map_origin_y + gy * self.grid_resolution
                wy_max = wy_min + self.grid_resolution

                # Convert world bounds to image pixel bounds
                u_min, v_max = self.world_to_pixel(wx_min, wy_min)
                u_max, v_min = self.world_to_pixel(wx_max, wy_max)

                # Sort because image y is flipped
                x0 = max(0, min(u_min, u_max))
                x1 = min(self.map_width - 1, max(u_min, u_max))

                y0 = max(0, min(v_min, v_max))
                y1 = min(self.map_height - 1, max(v_min, v_max))

                block = inflated[y0:y1 + 1, x0:x1 + 1]

                if block.size > 0 and np.any(block):
                    grid[gy, gx] = 1

        self.get_logger().info(f"Planning grid size: {grid_w} x {grid_h}")
        return grid


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()