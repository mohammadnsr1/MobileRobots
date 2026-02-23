import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool
import math
from rclpy.duration import Duration


class ReactiveNavigator(Node):
    def __init__(self):
        super().__init__('reactive_navigator')

        # 1. Setup Service
        self.srv = self.create_service(SetBool, 'toggle_navigation', self.toggle_callback)
        self.active = False

        # 2. Setup Pub/Sub
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # 3. Parameter
        self.declare_parameter('safety_distance', 0.5)

        # --- Task 2 state ---
        self.trigger_distance = self.get_parameter('safety_distance').value  # trigger condition based on safety_distance
        self.latest_front = None
        self.latest_left = None
        self.latest_right = None

        # turning state (non-blocking)
        self.turning = False
        self.turn_end_time = None
        self.turn_cmd = Twist()

        # control loop at 10 Hz
        self.control_timer = self.create_timer(0.1, self.control_loop)

    def toggle_callback(self, request, response):
        self.active = request.data

        # Strongly recommended: stop immediately when disabled
        if not self.active:
            self.publisher_.publish(Twist())
            self.turning = False
            self.turn_end_time = None

        response.success = True
        response.message = f"Navigation set to {self.active}"
        return response

    def get_index_for_angle(self, msg: LaserScan, angle_deg: float) -> int:
        """Convert an angle in degrees (robot frame) to the closest ranges[] index using LaserScan metadata."""
        angle_rad = math.radians(angle_deg)
        idx = int(round((angle_rad - msg.angle_min) / msg.angle_increment))
        # clamp to valid bounds
        return max(0, min(idx, len(msg.ranges) - 1))

    def _sector_min_distance(self, msg: LaserScan, start_deg: float, end_deg: float) -> float:
        """Return the minimum valid range in [start_deg, end_deg] (degrees)."""
        i0 = self.get_index_for_angle(msg, start_deg)
        i1 = self.get_index_for_angle(msg, end_deg)

        if i0 <= i1:
            # the inner region 
            indices = range(i0, i1 + 1)
        else:
            # the outer region region 
            indices = list(range(i0, len(msg.ranges))) + list(range(0, i1 + 1))

        vals = []
        for i in indices:
            r = msg.ranges[i]
            if r is None:
                continue
            if not math.isfinite(r):
                continue
            if r <= 0.0:
                continue
            vals.append(r)

        return min(vals) if vals else float('inf')

    def scan_callback(self, msg: LaserScan):
        if not self.active:
            return

        # Define sectors (degrees): adjust if your class defines different ones
        # Front: -15..+15, Left: +60..+120, Right: -120..-60
        self.latest_front = self._sector_min_distance(msg, -15.0, 15.0)
        self.latest_left  = self._sector_min_distance(msg, -135, -45)
        self.latest_right = self._sector_min_distance(msg, 45, 135)

    def control_loop(self):
        if not self.active:
            return

        now = self.get_clock().now()

        #If currently turning, keep turning until time is up
        if self.turning:
            if now >= self.turn_end_time:
                self.turning = False
                self.publisher_.publish(Twist())  # stop after the turn
            else:
                self.publisher_.publish(self.turn_cmd)
            return

        # If no scan received yet, do nothing
        if self.latest_front is None:
            return

        # Trigger condition: obstacle closer than 0.3m in front
        if self.latest_front < self.trigger_distance:
            self.execute_rotation_maneuver()
        else:
            self.drive_forward()

    def execute_rotation_maneuver(self):
        # Smart Pivot: turn toward the side with larger clearance
        left_clear = self.latest_left if self.latest_left is not None else 0.0
        right_clear = self.latest_right if self.latest_right is not None else 0.0

        direction = +1.0 if left_clear >= right_clear else -1.0  # + = CCW/left, - = CW/right

        angular_speed = 1.57  # rad/s (tune if needed)
        duration = (math.pi / 2.0) / angular_speed  # 90 degrees

        self.turn_cmd = Twist()
        self.turn_cmd.angular.z = direction * angular_speed
        self.turn_cmd.linear.x = 0.0

        self.turning = True
        self.turn_end_time = self.get_clock().now() + Duration(seconds=duration)

        self.get_logger().info(
            f"Obstacle < {self.trigger_distance:.2f}m. Pivoting {'LEFT' if direction>0 else 'RIGHT'} "
            f"(L={left_clear:.2f}, R={right_clear:.2f})"
        )

    def drive_forward(self):
        msg = Twist()
        msg.linear.x = 1.0
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveNavigator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()