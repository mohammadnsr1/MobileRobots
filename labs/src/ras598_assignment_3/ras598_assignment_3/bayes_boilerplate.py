import os
import re

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from marker_msgs.msg import MarkerDetection
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from scipy.ndimage import gaussian_filter
from std_msgs.msg import ColorRGBA
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray


def normalize_angle_deg(angle):
    """Wrap angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def parse_world_file(path):
    """Parse Stage .world file for landmark positions."""
    found = {}
    if not os.path.exists(path):
        return found

    with open(path, 'r', encoding='utf-8') as world_file:
        content = world_file.read()

    block_pattern = re.compile(r'my_block\s*\((.*?)\)', re.DOTALL)
    pose_pattern = re.compile(r'pose\s*\[\s*([-\d.]+)\s+([-\d.]+)')
    id_pattern = re.compile(r'fiducial_return\s+(\d+)')

    for block_content in block_pattern.findall(content):
        pose_match = pose_pattern.search(block_content)
        id_match = id_pattern.search(block_content)
        if pose_match and id_match:
            found[int(id_match.group(1))] = (
                float(pose_match.group(1)),
                float(pose_match.group(2)),
            )

    return found


def yaw_deg_from_pose(pose):
    q = pose.orientation
    return np.degrees(euler_from_quaternion([q.x, q.y, q.z, q.w])[2])


def decompose_turn_go_turn(x0, y0, yaw0_deg, x1, y1, yaw1_deg):
    dx = x1 - x0
    dy = y1 - y0
    d_trans = float(np.hypot(dx, dy))

    if d_trans > 1e-6:
        heading_deg = np.degrees(np.arctan2(dy, dx))
        d_rot1_deg = normalize_angle_deg(heading_deg - yaw0_deg)
    else:
        d_rot1_deg = 0.0

    d_rot2_deg = normalize_angle_deg(yaw1_deg - yaw0_deg - d_rot1_deg)
    return d_rot1_deg, d_trans, d_rot2_deg

class BayesFilter3D(Node):
    def __init__(self, world_file_path):
        super().__init__('bayes_filter_3d_node')
        
        # --- CONFIGURATION ---
        self.world_size = 16.0
        self.resolution = 0.2
        self.theta_res = 10
        self.grid_dim = int(self.world_size / self.resolution)
        self.theta_dim = int(360 / self.theta_res)
        self.sigma_rot1_deg = 0.711
        self.sigma_trans_m = 0.050
        self.sigma_rot2_deg = 1.743
        self.sigma_r = 0.5
        self.sigma_b_deg = 15.0

        # --- ROS PUBLISHERS ---
        self.costmap_pub = self.create_publisher(OccupancyGrid, 'viz/belief_costmap', 10)
        self.landmark_pub = self.create_publisher(MarkerArray, 'viz/landmarks', 10)
        self.gt_path_pub = self.create_publisher(Path, 'viz/gt_path', 10)
        self.odom_path_pub = self.create_publisher(Path, 'viz/odom_path', 10)
        
        # Path messages initialization
        self.gt_path_msg = Path()
        self.gt_path_msg.header.frame_id = 'map'
        self.odom_path_msg = Path()
        self.odom_path_msg.header.frame_id = 'map'

        # --- FILTER STATE INITIALIZATION ---
        self.landmarks = parse_world_file(world_file_path)
        self.initial_pose = [-7.0, -7.0, 90.0]
        
        # Trajectory tracking for visualization
        self.odom_x, self.odom_y = self.initial_pose[0], self.initial_pose[1]
        self.odom_th = np.radians(self.initial_pose[2])
        
        # Initialize self.belief in initialize_belief()
        self.initialize_belief(pose=self.initial_pose)

        self.last_odom_pose = None

        # --- SUBSCRIPTIONS ---
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Odometry, '/ground_truth', self.gt_callback, 10)
        self.create_subscription(MarkerDetection, '/fiducials', self.fiducial_callback, 10)

        # Refresh landmarks in RViz
        self.create_timer(1.0, self._publish_landmarks)

        # Print Landmarks Locations
        self.get_logger().info("--- Landmark Locations ---")
        for tid, pos in self.landmarks.items():
            lx, ly = pos
            self.get_logger().info(f"ID {tid}: x={lx:.2f}, y={ly:.2f}")
        self.get_logger().info("---------------------------")
    
    # -------------------------------------------------------------------------
    # UTILITY & VISUALIZATION FUNCTIONS
    # You don't need to change the function code for these functions.
    # -------------------------------------------------------------------------

    def _publish_landmarks(self):
        """Publishes landmark locations as Markers for RViz."""
        ma = MarkerArray()
        for tid, (tx, ty) in self.landmarks.items():
            # Cylinder Marker
            c = Marker()
            c.header.frame_id = 'map'
            c.id = tid
            c.type = Marker.CYLINDER
            c.action = Marker.ADD
            c.pose.position.x, c.pose.position.y, c.pose.position.z = tx, ty, 0.5
            c.scale.x, c.scale.y, c.scale.z = 0.3, 0.3, 1.0
            c.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            ma.markers.append(c)
            # Text ID Marker
            t = Marker()
            t.header.frame_id = 'map'
            t.id = tid + 1000
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.text = f"ID: {tid}"
            t.pose.position.x, t.pose.position.y, t.pose.position.z = tx, ty + 0.5, 1.2
            t.scale.z = 0.4 
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            ma.markers.append(t)
        self.landmark_pub.publish(ma)

    def _publish_costmap(self):
        """Publishes the 2D projected belief as an OccupancyGrid."""
        if not hasattr(self, 'belief'): return
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution, grid.info.width, grid.info.height = self.resolution, self.grid_dim, self.grid_dim
        grid.info.origin.position.x, grid.info.origin.position.y = -8.0, -8.0

        belief_2d = np.sum(self.belief, axis=2)
        belief_flipped = np.flipud(belief_2d) # Match ROS bottom-up convention
        
        max_val = np.max(belief_flipped)
        if max_val > 0:
            data = (belief_flipped / max_val * 100).astype(np.int8)
        else:
            data = np.zeros_like(belief_flipped, dtype=np.int8)
        
        grid.data = data.flatten().tolist()
        self.costmap_pub.publish(grid)
    
    
    # -------------------------------------------------------------------------
    #  ASSIGNMENT TASKS
    # -------------------------------------------------------------------------

    def _normalize_belief(self):
        total = np.sum(self.belief)
        if total <= 0.0:
            self.belief.fill(1.0 / self.belief.size)
        else:
            self.belief /= total

    def zero_wrapped_edges(self, arr, row_shift, col_shift):
        """Clear wrapped probability after np.roll."""
        if row_shift > 0:
            arr[:row_shift, :] = 0.0
        elif row_shift < 0:
            arr[row_shift:, :] = 0.0

        if col_shift > 0:
            arr[:, :col_shift] = 0.0
        elif col_shift < 0:
            arr[:, col_shift:] = 0.0

        return arr

    def gt_callback(self, msg):
        """Updates Ground Truth path for visualization."""
        p = PoseStamped(header=msg.header)
        p.pose.position.x, p.pose.position.y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self.gt_path_msg.poses.append(p)
        self.gt_path_pub.publish(self.gt_path_msg)

    def initialize_belief(self, pose=None):
        """
        TASK 1: Initialize the 3D Probability Density Function (PDF).
        If 'pose' is provided, initialize the belief as a localized 
        distribution (e.g., a point or Gaussian). If 'pose' is None, 
        initialize a Uniform distribution across the entire state space.
        """
        self.belief = np.zeros((self.grid_dim, self.grid_dim, self.theta_dim), dtype=np.float64)

        if pose is None:
            self.belief.fill(1.0 / self.belief.size)
            return

        ix, iy, ith = self.real_to_grid(pose[0], pose[1], pose[2])
        self.belief[iy, ix, ith] = 1.0
        self.belief = gaussian_filter(self.belief, sigma=(1.0, 1.0, 0.6), mode='constant')
        self._normalize_belief()

    def real_to_grid(self, x, y, theta_deg):
        """
        TASK 2(a): Coordinate Transformation (Real -> Grid).
        Convert continuous world coordinates to discrete 3D grid indices.
        Returns: (ix, iy, ith)
        """
        ix = int((x + self.world_size / 2.0) / self.resolution)
        iy_world = int((y + self.world_size / 2.0) / self.resolution)
        iy = self.grid_dim - 1 - iy_world

        ix = int(np.clip(ix, 0, self.grid_dim - 1))
        iy = int(np.clip(iy, 0, self.grid_dim - 1))

        theta_deg = theta_deg % 360.0
        ith = int(np.round(theta_deg / self.theta_res)) % self.theta_dim
        return ix, iy, ith

    def grid_to_real(self):
        """
        TASK 2(b): Coordinate Transformation (Grid -> Real).
        Generate 3D numpy arrays containing the real-world (x, y, theta) 
        values for every cell in the belief grid.
        Returns: rx, ry, rth (all numpy arrays of shape self.belief.shape)
        """
        x_centers = -self.world_size / 2.0 + (np.arange(self.grid_dim) + 0.5) * self.resolution
        y_centers = self.world_size / 2.0 - (np.arange(self.grid_dim) + 0.5) * self.resolution
        theta_centers = np.arange(self.theta_dim, dtype=np.float64) * self.theta_res

        rx = x_centers[np.newaxis, :, np.newaxis] * np.ones(self.belief.shape, dtype=np.float64)
        ry = y_centers[:, np.newaxis, np.newaxis] * np.ones(self.belief.shape, dtype=np.float64)
        rth = theta_centers[np.newaxis, np.newaxis, :] * np.ones(self.belief.shape, dtype=np.float64)
        return rx, ry, rth

    def predict(self, curr_msg, last_msg):
        """
        TASK 3: Motion Model (Prediction).
        Implement the 'Turn-Go-Turn' model. Update self.belief to reflect 
        the robot's movement between last_msg and curr_msg.
        """
        curr_pose = curr_msg.pose.pose
        last_pose = last_msg.pose.pose

        curr_x = curr_pose.position.x
        curr_y = curr_pose.position.y
        last_x = last_pose.position.x
        last_y = last_pose.position.y

        curr_yaw_deg = yaw_deg_from_pose(curr_pose)
        last_yaw_deg = yaw_deg_from_pose(last_pose)
        d_rot1_deg, d_trans_m, d_rot2_deg = decompose_turn_go_turn(
            last_x, last_y, last_yaw_deg,
            curr_x, curr_y, curr_yaw_deg,
        )
        sigma_rot1_bins = self.sigma_rot1_deg / self.theta_res
        sigma_rot2_bins = self.sigma_rot2_deg / self.theta_res
        sigma_trans_cells = self.sigma_trans_m / self.resolution

        belief_stage = self.belief.copy()

        rot1_bins = int(np.round(d_rot1_deg / self.theta_res))
        belief_stage = np.roll(belief_stage, shift=rot1_bins, axis=2)
        if sigma_rot1_bins > 0.0:
            belief_stage = gaussian_filter(
                belief_stage,
                sigma=(0.0, 0.0, sigma_rot1_bins),
                mode='constant',
            )

        translated = np.zeros_like(belief_stage)
        for ith in range(self.theta_dim):
            theta_rad = np.radians(ith * self.theta_res)
            shift_x_m = d_trans_m * np.cos(theta_rad)
            shift_y_m = d_trans_m * np.sin(theta_rad)
            shift_cols = int(np.round(shift_x_m / self.resolution))
            shift_rows = int(np.round(-shift_y_m / self.resolution))
            shifted = np.roll(
                belief_stage[:, :, ith],
                shift=(shift_rows, shift_cols),
                axis=(0, 1),
            )
            translated[:, :, ith] = self.zero_wrapped_edges(shifted, shift_rows, shift_cols)

        belief_stage = translated
        if sigma_trans_cells > 0.0:
            belief_stage = gaussian_filter(
                belief_stage,
                sigma=(sigma_trans_cells, sigma_trans_cells, 0.0),
                mode='constant',
            )

        rot2_bins = int(np.round(d_rot2_deg / self.theta_res))
        belief_stage = np.roll(belief_stage, shift=rot2_bins, axis=2)
        if sigma_rot2_bins > 0.0:
            belief_stage = gaussian_filter(
                belief_stage,
                sigma=(0.0, 0.0, sigma_rot2_bins),
                mode='constant',
            )

        self.belief = belief_stage
        self._normalize_belief()

    def update_measurement(self, landmark_id, measured_range, measured_bearing_deg):
        """
        TASK 4: Measurement Model (Update).
        Correct the belief using a landmark sighting.
        """
        if landmark_id not in self.landmarks:
            return

        lx, ly = self.landmarks[landmark_id]
        rx, ry, rth = self.grid_to_real()

        dx = lx - rx
        dy = ly - ry
        expected_range = np.hypot(dx, dy)
        expected_bearing_deg = normalize_angle_deg(np.degrees(np.arctan2(dy, dx)) - rth)

        range_error = measured_range - expected_range
        bearing_error = normalize_angle_deg(measured_bearing_deg - expected_bearing_deg)

        p_r = np.exp(-0.5 * (range_error / self.sigma_r) ** 2)
        p_b = np.exp(-0.5 * (bearing_error / self.sigma_b_deg) ** 2)
        likelihood = p_r * p_b

        self.belief *= likelihood
        self.belief += 1e-12
        self._normalize_belief()


    def odom_callback(self, msg):
        """Handles robot motion, trajectory visualization and performas the Prediction update."""
        if self.last_odom_pose is None:
            self.last_odom_pose = msg
            return
        
        q = msg.pose.pose.orientation
        curr_yaw_deg = np.degrees(euler_from_quaternion([q.x, q.y, q.z, q.w])[2]) % 360
        
        q_old = self.last_odom_pose.pose.pose.orientation
        old_yaw_deg = np.degrees(euler_from_quaternion([q_old.x, q_old.y, q_old.z, q_old.w])[2]) % 360

        # Calculates the differential odometry
        dx = msg.pose.pose.position.x - self.last_odom_pose.pose.pose.position.x
        dy = msg.pose.pose.position.y - self.last_odom_pose.pose.pose.position.y
        dth = normalize_angle_deg(curr_yaw_deg - old_yaw_deg)

        # Update and Publish Odom Path
        self.odom_th += np.radians(dth)
        self.odom_x += (dx * np.cos(self.odom_th)) - (dy * np.sin(self.odom_th))
        self.odom_y += (dx * np.sin(self.odom_th)) + (dy * np.cos(self.odom_th))
        
        p = PoseStamped(header=msg.header)
        p.pose.position.x, p.pose.position.y = self.odom_x, self.odom_y
        self.odom_path_msg.poses.append(p)
        self.odom_path_pub.publish(self.odom_path_msg)

        # Run the prediction loop only when there is sufficient motion
        if np.sqrt(dx**2 + dy**2) > 0.001 or abs(dth) > 0.1:
            self.predict(msg, self.last_odom_pose)
            self._publish_costmap()
        self.last_odom_pose = msg
    
    def fiducial_callback(self, msg):
        """
        TASK 5: Fiducial Callback.
        Performs the Measurement update for every landmark (also called marker) seen by the robot
        """
        for marker in msg.markers:
            if not marker.ids:
                continue

            landmark_id = int(marker.ids[0])
            rel_x = marker.pose.position.x
            rel_y = marker.pose.position.y

            measured_range = float(np.hypot(rel_x, rel_y))
            measured_bearing_deg = float(np.degrees(np.arctan2(rel_y, rel_x)))
            self.update_measurement(landmark_id, measured_range, measured_bearing_deg)

        # Publishes the probability distribution costmap
        self._publish_costmap() # Dont remove this line
    

def main():
    rclpy.init()

    world_path = os.path.expanduser("/home/mohammadnsr1/coursework/MobileRobots/labs/src/stage_ros2/world/cave.world")
    if not os.path.exists(world_path):
        print("\n" + "="*50)
        print("ERROR: World file not found!")
        print(f"Path attempted: {world_path}")
        print("-" * 50)
        print("FIX: Please open your script and update the 'world_path' variable")
        print("to match your ROS 2 workspace name (e.g., ~/dev_ws/src/...)")
        print("="*50 + "\n")
        return # Exit the program gracefully


    node = BayesFilter3D(world_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
