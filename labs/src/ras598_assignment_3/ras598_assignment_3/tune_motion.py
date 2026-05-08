import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from ras598_assignment_3.bayes_boilerplate import decompose_turn_go_turn, normalize_angle_deg, yaw_deg_from_pose


class MotionModelTuner(Node):
    def __init__(self):
        super().__init__('motion_model_tuner')

        self.prev_odom_pose = None
        self.curr_odom_pose = None
        self.prev_gt_pose = None
        self.curr_gt_pose = None

        self.rot1_errors_deg = []
        self.trans_errors_m = []
        self.rot2_errors_deg = []
        self.sample_count = 0

        self.last_report_count = 0
        self.min_trans_m = 0.01
        self.min_rot_deg = 1.0

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Odometry, '/ground_truth', self.gt_callback, 10)
        self.create_timer(5.0, self.report_stats)

    def odom_callback(self, msg):
        self.prev_odom_pose = self.curr_odom_pose
        self.curr_odom_pose = msg.pose.pose
        self.try_process_sample()

    def gt_callback(self, msg):
        self.prev_gt_pose = self.curr_gt_pose
        self.curr_gt_pose = msg.pose.pose
        self.try_process_sample()

    def try_process_sample(self):
        if any(
            item is None
            for item in (self.prev_odom_pose, self.curr_odom_pose, self.prev_gt_pose, self.curr_gt_pose)
        ):
            return

        odom_prev = self.prev_odom_pose
        odom_curr = self.curr_odom_pose
        gt_prev = self.prev_gt_pose
        gt_curr = self.curr_gt_pose

        odom_motion = decompose_turn_go_turn(
            odom_prev.position.x,
            odom_prev.position.y,
            yaw_deg_from_pose(odom_prev),
            odom_curr.position.x,
            odom_curr.position.y,
            yaw_deg_from_pose(odom_curr),
        )
        gt_motion = decompose_turn_go_turn(
            gt_prev.position.x,
            gt_prev.position.y,
            yaw_deg_from_pose(gt_prev),
            gt_curr.position.x,
            gt_curr.position.y,
            yaw_deg_from_pose(gt_curr),
        )

        max_rot = max(abs(odom_motion[0] + odom_motion[2]), abs(gt_motion[0] + gt_motion[2]))
        max_trans = max(odom_motion[1], gt_motion[1])
        if max_trans < self.min_trans_m and max_rot < self.min_rot_deg:
            return

        e_rot1 = normalize_angle_deg(odom_motion[0] - gt_motion[0])
        e_trans = odom_motion[1] - gt_motion[1]
        e_rot2 = normalize_angle_deg(odom_motion[2] - gt_motion[2])

        self.rot1_errors_deg.append(e_rot1)
        self.trans_errors_m.append(e_trans)
        self.rot2_errors_deg.append(e_rot2)
        self.sample_count += 1

        self.prev_odom_pose = self.curr_odom_pose
        self.prev_gt_pose = self.curr_gt_pose

        if self.sample_count - self.last_report_count >= 25:
            self.report_stats()

    def format_stats(self, values, units):
        if not values:
            return f'mean=0.000 {units}, std=0.000 {units}'
        arr = np.asarray(values, dtype=np.float64)
        return f'mean={np.mean(arr):.3f} {units}, std={np.std(arr):.3f} {units}'

    def report_stats(self):
        if self.sample_count == 0 or self.sample_count == self.last_report_count:
            return

        self.last_report_count = self.sample_count
        self.get_logger().info(f'samples: {self.sample_count}')
        self.get_logger().info(f'e_rot1: {self.format_stats(self.rot1_errors_deg, "deg")}')
        self.get_logger().info(f'e_trans: {self.format_stats(self.trans_errors_m, "m")}')
        self.get_logger().info(f'e_rot2: {self.format_stats(self.rot2_errors_deg, "deg")}')

    def print_final_summary(self):
        sigma_rot1_deg = float(np.std(np.asarray(self.rot1_errors_deg, dtype=np.float64))) if self.rot1_errors_deg else 0.0
        sigma_trans_m = float(np.std(np.asarray(self.trans_errors_m, dtype=np.float64))) if self.trans_errors_m else 0.0
        sigma_rot2_deg = float(np.std(np.asarray(self.rot2_errors_deg, dtype=np.float64))) if self.rot2_errors_deg else 0.0

        self.get_logger().info('--- Final motion tuning summary ---')
        self.get_logger().info(f'samples = {self.sample_count}')
        self.get_logger().info(f'sigma_rot1_deg = {sigma_rot1_deg:.3f}')
        self.get_logger().info(f'sigma_trans_m = {sigma_trans_m:.3f}')
        self.get_logger().info(f'sigma_rot2_deg = {sigma_rot2_deg:.3f}')
        self.get_logger().info('Paste into BayesFilter3D:')
        self.get_logger().info(f'self.sigma_rot1_deg = {sigma_rot1_deg:.3f}')
        self.get_logger().info(f'self.sigma_trans_m = {sigma_trans_m:.3f}')
        self.get_logger().info(f'self.sigma_rot2_deg = {sigma_rot2_deg:.3f}')


def main():
    rclpy.init()
    node = MotionModelTuner()
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
