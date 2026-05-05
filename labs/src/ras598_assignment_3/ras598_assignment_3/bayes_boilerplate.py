import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from marker_msgs.msg import MarkerDetection
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
from scipy.ndimage import gaussian_filter
from tf_transformations import euler_from_quaternion
import re
import os

class BayesFilter3D(Node):
    def __init__(self, world_file_path):
        super().__init__('bayes_filter_3d_node')
        
        # --- CONFIGURATION ---
        self.world_size = 16.0         # Total width/height of the environment (meters)
        self.resolution = 0.2          # Size of one grid cell (meters)
        self.theta_res = 10            # Angular resolution (degrees)
        self.grid_dim = int(self.world_size / self.resolution)
        self.theta_dim = int(360 / self.theta_res)

        # --- ROS PUBLISHERS ---
        self.costmap_pub = self.create_publisher(OccupancyGrid, 'viz/belief_costmap', 10)
        self.landmark_pub = self.create_publisher(MarkerArray, 'viz/landmarks', 10)
        self.gt_path_pub = self.create_publisher(Path, 'viz/gt_path', 10)
        self.odom_path_pub = self.create_publisher(Path, 'viz/odom_path', 10)
        
        # Path messages initialization
        self.gt_path_msg = Path(header={'frame_id': 'map'})
        self.odom_path_msg = Path(header={'frame_id': 'map'})

        # --- FILTER STATE INITIALIZATION ---
        self.landmarks = self._parse_world_file(world_file_path)
        self.initial_pose = [-7.0, -7.0, 90.0]  # [x, y, theta_degrees] DO NOT CHANGE THIS!
        
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

    def _parse_world_file(self, path):
        """Parses Stage .world file for landmark positions."""
        found = {}
        if not os.path.exists(path): return found
        with open(path, 'r') as f:
            content = f.read()
        block_pattern = re.compile(r'my_block\s*\((.*?)\)', re.DOTALL)
        pose_pattern = re.compile(r'pose\s*\[\s*([-\d.]+)\s+([-\d.]+)')
        id_pattern = re.compile(r'fiducial_return\s+(\d+)')
        for block_content in block_pattern.findall(content):
            p_match = pose_pattern.search(block_content)
            id_match = id_pattern.search(block_content)
            if p_match and id_match:
                found[int(id_match.group(1))] = (float(p_match.group(1)), float(p_match.group(2)))
        return found

    def _publish_landmarks(self):
        """Publishes landmark locations as Markers for RViz."""
        ma = MarkerArray()
        for tid, (tx, ty) in self.landmarks.items():
            # Cylinder Marker
            c = Marker(header={'frame_id': 'map'}, id=tid, type=Marker.CYLINDER, action=Marker.ADD)
            c.pose.position.x, c.pose.position.y, c.pose.position.z = tx, ty, 0.5
            c.scale.x, c.scale.y, c.scale.z = 0.3, 0.3, 1.0
            c.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            ma.markers.append(c)
            # Text ID Marker
            t = Marker(header={'frame_id': 'map'}, id=tid + 1000, type=Marker.TEXT_VIEW_FACING)
            t.text = f"ID: {tid}"
            t.pose.position.x, t.pose.position.y, t.pose.position.z = tx, ty + 0.5, 1.2
            t.scale.z = 0.4 
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            ma.markers.append(t)
        self.landmark_pub.publish(ma)

    def _publish_costmap(self):
        """Publishes the 2D projected belief as an OccupancyGrid."""
        if not hasattr(self, 'belief'): return
        grid = OccupancyGrid(header={'frame_id': 'map', 'stamp': self.get_clock().now().to_msg()})
        grid.info.resolution, grid.info.width, grid.info.height = self.resolution, self.grid_dim, self.grid_dim
        grid.info.origin.position.x, grid.info.origin.position.y = -8, -8

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

    def normalize_angle_deg(self, angle):
        """Wrap angle to [-180, 180)."""
        return (angle + 180.0) % 360.0 - 180.0

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

        curr_q = curr_pose.orientation
        last_q = last_pose.orientation
        curr_yaw_rad = euler_from_quaternion([curr_q.x, curr_q.y, curr_q.z, curr_q.w])[2]
        last_yaw_rad = euler_from_quaternion([last_q.x, last_q.y, last_q.z, last_q.w])[2]
        curr_yaw_deg = np.degrees(curr_yaw_rad)
        last_yaw_deg = np.degrees(last_yaw_rad)

        dx_world = curr_x - last_x
        dy_world = curr_y - last_y
        dtheta_deg = self.normalize_angle_deg(curr_yaw_deg - last_yaw_deg)

        d_trans = np.hypot(dx_world, dy_world)
        heading_deg = np.degrees(np.arctan2(dy_world, dx_world)) if d_trans > 1e-9 else last_yaw_deg
        d_rot1 = self.normalize_angle_deg(heading_deg - last_yaw_deg) if d_trans > 1e-9 else 0.0

        theta_shift_bins = int(np.round(dtheta_deg / self.theta_res))
        rotated_belief = np.roll(self.belief, shift=theta_shift_bins, axis=2)
        predicted = np.zeros_like(rotated_belief)

        for ith in range(self.theta_dim):
            theta_bin_deg = ith * self.theta_res
            theta_total_rad = np.radians(theta_bin_deg + d_rot1)
            shift_x_m = d_trans * np.cos(theta_total_rad)
            shift_y_m = d_trans * np.sin(theta_total_rad)
            shift_cols = int(np.round(shift_x_m / self.resolution))
            shift_rows = int(np.round(-shift_y_m / self.resolution))

            shifted = np.roll(rotated_belief[:, :, ith], shift=(shift_rows, shift_cols), axis=(0, 1))
            predicted[:, :, ith] = self.zero_wrapped_edges(shifted, shift_rows, shift_cols)

        self.belief = gaussian_filter(predicted, sigma=(1.0, 1.0, 0.6), mode='constant')
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
        expected_bearing_deg = self.normalize_angle_deg(np.degrees(np.arctan2(dy, dx)) - rth)

        range_error = measured_range - expected_range
        bearing_error = self.normalize_angle_deg(measured_bearing_deg - expected_bearing_deg)

        sigma_r = 0.5
        sigma_b = 15.0
        p_r = np.exp(-0.5 * (range_error / sigma_r) ** 2)
        p_b = np.exp(-0.5 * (bearing_error / sigma_b) ** 2)
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
        dth = (curr_yaw_deg - old_yaw_deg + 180) % 360 - 180 

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
            self.last_odom_pose = msg
            self._publish_costmap()
    
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
