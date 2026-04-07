import math
import cv2
import numpy as np
import heapq
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
import os

from example_interfaces.srv import Trigger
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/planner_markers', 10)
        self.raw_path_pub = self.create_publisher(Path, '/raw_path', 10)
        self.pruned_path_pub = self.create_publisher(Path, '/pruned_path', 10)  

        #timers
        self.marker_timer = self.create_timer(0.1, self.publish_path_markers)

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
        self.global_frame = "map"

        self.map_resolution = 0.032
        self.map_origin_x = -8.0
        self.map_origin_y = -8.0
        self.grid_resolution = 0.2
        self.map_width = 500   
        self.map_height = 500  
        self.grid_resolution = 0.2
        self.raw_world_path = None


        #prunning the path
        self.pruned_grid_path = None
        self.pruned_world_path = None



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

            start_cell = (gsx,gsy)
            goal_cell = (ggx, ggy)

            grid_path = self.astar(start_cell, goal_cell)
            if grid_path is None:
                return

            self.raw_grid_path = grid_path
            self.raw_world_path = self.grid_path_to_world(grid_path)

            self.get_logger().info(f"Raw path has {len(self.raw_grid_path)} cells")

            self.pruned_grid_path = self.prune_path(self.raw_grid_path)
            self.pruned_world_path = self.grid_path_to_world(self.pruned_grid_path)

            self.get_logger().info(f"Pruned path has {len(self.pruned_grid_path)} waypoints")
            

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


    def is_in_bounds(self, gx, gy):
        h, w = self.occupancy_grid.shape
        return 0 <= gx < w and 0 <= gy < h


    def is_free(self, gx, gy):
        return self.is_in_bounds(gx, gy) and self.occupancy_grid[gy, gx] == 0


    def get_neighbors(self, gx, gy):
        neighbors = []
        directions = [
            (-1,  0, 1.0),
            ( 1,  0, 1.0),
            ( 0, -1, 1.0),
            ( 0,  1, 1.0),
            (-1, -1, math.sqrt(2)),
            (-1,  1, math.sqrt(2)),
            ( 1, -1, math.sqrt(2)),
            ( 1,  1, math.sqrt(2)),
        ]

        for dx, dy, cost in directions:
            nx, ny = gx + dx, gy + dy
            if self.is_free(nx, ny):
                neighbors.append((nx, ny, cost))

        return neighbors

    def heuristic(self, a, b):
        ax, ay = a
        bx, by = b
        return math.hypot(bx - ax, by - ay)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def astar(self, start, goal):
        if not self.is_free(*start):
            self.get_logger().error(f"Start is occupied: {start}")
            return None

        if not self.is_free(*goal):
            self.get_logger().error(f"Goal is occupied: {goal}")
            return None

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal)}

        closed_set = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue

            if current == goal:
                return self.reconstruct_path(came_from, current)

            closed_set.add(current)

            for nx, ny, move_cost in self.get_neighbors(*current):
                neighbor = (nx, ny)

                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_heap, (f, neighbor))

        self.get_logger().error("A* failed to find a path")
        return None

    def grid_path_to_world(self, grid_path):
        world_path = []
        for gx, gy in grid_path:
            wx, wy = self.grid_to_world(gx, gy)
            world_path.append((wx, wy))
        return world_path
    
    def line_of_sight_grid(self, cell_a, cell_b):
        x0, y0 = cell_a
        x1, y1 = cell_b

        dx = x1 - x0
        dy = y1 - y0

        steps = max(abs(dx), abs(dy))

        if steps == 0:
            return self.is_free(x0, y0)

        for i in range(steps + 1):
            t = i / steps
            x = int(round(x0 + t * dx))
            y = int(round(y0 + t * dy))

            if not self.is_free(x, y):
                return False

        return True

    
    def publish_raw_path_marker(self):
        if self.raw_world_path is None:
            return

        marker_array = MarkerArray()

        raw_marker = Marker()
        raw_marker.header.frame_id = "map"
        raw_marker.header.stamp = self.get_clock().now().to_msg()
        raw_marker.ns = "planner"
        raw_marker.id = 0
        raw_marker.type = Marker.LINE_STRIP
        raw_marker.action = Marker.ADD
        raw_marker.scale.x = 0.05

        raw_marker.color.r = 0.0
        raw_marker.color.g = 1.0
        raw_marker.color.b = 0.0
        raw_marker.color.a = 1.0

        raw_marker.pose.orientation.w = 1.0

        for x, y in self.raw_world_path:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            raw_marker.points.append(p)

        marker_array.markers.append(raw_marker)
        self.marker_pub.publish(marker_array)

    def prune_path(self, grid_path):
        if grid_path is None or len(grid_path) <= 2:
            return grid_path

        pruned = [grid_path[0]]
        anchor_idx = 0

        while anchor_idx < len(grid_path) - 1:
            farthest_visible_idx = anchor_idx + 1

            for test_idx in range(anchor_idx + 1, len(grid_path)):
                if self.line_of_sight_grid(grid_path[anchor_idx], grid_path[test_idx]):
                    farthest_visible_idx = test_idx
                else:
                    break

            pruned.append(grid_path[farthest_visible_idx])
            anchor_idx = farthest_visible_idx

        return pruned
    def publish_pruned_path_marker(self, marker_array):
        if self.pruned_world_path is None:
            return

        pruned_marker = Marker()
        pruned_marker.header.frame_id = self.global_frame
        pruned_marker.header.stamp = self.get_clock().now().to_msg()
        pruned_marker.ns = "planner"
        pruned_marker.id = 1
        pruned_marker.type = Marker.LINE_STRIP
        pruned_marker.action = Marker.ADD
        pruned_marker.scale.x = 0.08

        pruned_marker.color.r = 0.0
        pruned_marker.color.g = 0.0
        pruned_marker.color.b = 1.0
        pruned_marker.color.a = 1.0

        pruned_marker.pose.orientation.w = 1.0

        for x, y in self.pruned_world_path:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.08
            pruned_marker.points.append(p)

        marker_array.markers.append(pruned_marker)
    
    
    def publish_path_markers(self):
        marker_array = MarkerArray()

        if self.raw_world_path is not None:
            raw_marker = Marker()
            raw_marker.header.frame_id = self.global_frame
            raw_marker.header.stamp = self.get_clock().now().to_msg()
            raw_marker.ns = "planner"
            raw_marker.id = 0
            raw_marker.type = Marker.LINE_STRIP
            raw_marker.action = Marker.ADD
            raw_marker.scale.x = 0.05

            raw_marker.color.r = 0.0
            raw_marker.color.g = 1.0
            raw_marker.color.b = 0.0
            raw_marker.color.a = 1.0

            raw_marker.pose.orientation.w = 1.0

            for x, y in self.raw_world_path:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.05
                raw_marker.points.append(p)

            marker_array.markers.append(raw_marker)

        self.publish_pruned_path_marker(marker_array)

        if len(marker_array.markers) > 0:
            self.marker_pub.publish(marker_array)
        
        if self.raw_world_path is not None:
            self.publish_path(self.raw_world_path, self.raw_path_pub, frame_id=self.global_frame)

        if self.pruned_world_path is not None:
            self.publish_path(self.pruned_world_path, self.pruned_path_pub, frame_id=self.global_frame)


    def publish_path(self, path_points, publisher, frame_id='map'):
        """
        path_points: list of (x, y) tuples in map frame
        """
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        for x, y in path_points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()