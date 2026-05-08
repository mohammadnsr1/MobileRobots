import os

import numpy as np
import rclpy
from marker_msgs.msg import MarkerDetection
from nav_msgs.msg import Odometry
from rclpy.node import Node

from ras598_assignment_3.bayes_boilerplate import normalize_angle_deg, parse_world_file, yaw_deg_from_pose


class SensorModelTuner(Node):
    def __init__(self, world_file_path):
        super().__init__('sensor_model_tuner')

        self.landmarks = parse_world_file(world_file_path)
        self.gt_pose = None

        self.range_errors_m = []
        self.bearing_errors_deg = []
        self.measurement_count = 0
        self.last_report_count = 0

        self.create_subscription(Odometry, '/ground_truth', self.gt_callback, 10)
        self.create_subscription(MarkerDetection, '/fiducials', self.fiducial_callback, 10)
        self.create_timer(5.0, self.report_stats)

        self.get_logger().info(f'loaded {len(self.landmarks)} landmarks from world file')

    def gt_callback(self, msg):
        self.gt_pose = msg.pose.pose

    def fiducial_callback(self, msg):
        if self.gt_pose is None:
            return

        robot_x = self.gt_pose.position.x
        robot_y = self.gt_pose.position.y
        robot_yaw_deg = yaw_deg_from_pose(self.gt_pose)

        for marker in msg.markers:
            if not marker.ids:
                continue

            landmark_id = int(marker.ids[0])
            if landmark_id not in self.landmarks:
                continue

            rel_x = marker.pose.position.x
            rel_y = marker.pose.position.y
            measured_range = float(np.hypot(rel_x, rel_y))
            measured_bearing_deg = float(np.degrees(np.arctan2(rel_y, rel_x)))

            landmark_x, landmark_y = self.landmarks[landmark_id]
            dx = landmark_x - robot_x
            dy = landmark_y - robot_y
            expected_range = float(np.hypot(dx, dy))
            expected_bearing_deg = normalize_angle_deg(np.degrees(np.arctan2(dy, dx)) - robot_yaw_deg)

            e_r = measured_range - expected_range
            e_b = normalize_angle_deg(measured_bearing_deg - expected_bearing_deg)

            self.range_errors_m.append(e_r)
            self.bearing_errors_deg.append(e_b)
            self.measurement_count += 1

        if self.measurement_count - self.last_report_count >= 25:
            self.report_stats()

    def format_stats(self, values, units):
        if not values:
            return f'mean=0.000 {units}, std=0.000 {units}'
        arr = np.asarray(values, dtype=np.float64)
        return f'mean={np.mean(arr):.3f} {units}, std={np.std(arr):.3f} {units}'

    def report_stats(self):
        if self.measurement_count == 0 or self.measurement_count == self.last_report_count:
            return

        self.last_report_count = self.measurement_count
        self.get_logger().info(f'measurements: {self.measurement_count}')
        self.get_logger().info(f'range error: {self.format_stats(self.range_errors_m, "m")}')
        self.get_logger().info(f'bearing error: {self.format_stats(self.bearing_errors_deg, "deg")}')

    def print_final_summary(self):
        sigma_r = float(np.std(np.asarray(self.range_errors_m, dtype=np.float64))) if self.range_errors_m else 0.0
        sigma_b_deg = float(np.std(np.asarray(self.bearing_errors_deg, dtype=np.float64))) if self.bearing_errors_deg else 0.0

        self.get_logger().info('--- Final sensor tuning summary ---')
        self.get_logger().info(f'measurements = {self.measurement_count}')
        self.get_logger().info(f'sigma_r = {sigma_r:.3f}')
        self.get_logger().info(f'sigma_b_deg = {sigma_b_deg:.3f}')
        self.get_logger().info('Paste into BayesFilter3D:')
        self.get_logger().info(f'self.sigma_r = {sigma_r:.3f}')
        self.get_logger().info(f'self.sigma_b_deg = {sigma_b_deg:.3f}')


def main():
    rclpy.init()

    world_path = os.path.expanduser(
        '/home/mohammadnsr1/coursework/MobileRobots/labs/src/stage_ros2/world/cave.world'
    )
    if not os.path.exists(world_path):
        print('\n' + '=' * 50)
        print('ERROR: World file not found!')
        print(f'Path attempted: {world_path}')
        print('=' * 50 + '\n')
        return

    node = SensorModelTuner(world_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_final_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
