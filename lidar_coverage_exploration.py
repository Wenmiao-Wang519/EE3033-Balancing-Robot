
import math
import numpy as np
import os
import yaml
import rclpy


from collections import deque
from rclpy.time import Time
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
)
from yolo_msgs.msg import DetectionArray
from std_srvs.srv import Trigger
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose, Spin
from rclpy.duration import Duration
from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, CameraInfo
from geometry_msgs.msg import (
    PoseStamped,
    PoseArray,
    PointStamped
)
from PIL import Image
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def bresenham(x0, y0, x1, y1):
    """Return grid cells along a line."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        yield x0, y0

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x0 += sx

        if e2 < dx:
            err += dx
            y0 += sy


class CoverageExplorer(Node):
    def camera_info_callback(self, msg):

        if (
            msg.width <= 0
            or msg.height <= 0
            or msg.k[0] <= 0
            or msg.k[4] <= 0
        ):
            self.get_logger().warn(
                'Invalid camera calibration'
            )
            return

        self.camera_fx = float(msg.k[0])
        self.camera_cx = float(msg.k[2])
        self.image_width = int(msg.width)


    def yolo_callback(self, msg):

        if not self.exploration_started:
            return

        if self.mission_state not in ('SEARCHING', 'SCANNING'):
            return

        if self.camera_fx is None:
            return

        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            return

        detection_time = Time.from_msg(
            msg.header.stamp,
            clock_type=self.get_clock().clock_type
        )

        age = (
            self.get_clock().now() - detection_time
        ).nanoseconds / 1e9

        if age < -0.2 or age > self.max_detection_age:
            return
        stamp_ns = (
            msg.header.stamp.sec * 1_000_000_000
            + msg.header.stamp.nanosec
        )

        if self.last_detection_stamp_ns == stamp_ns:
            return

        self.last_detection_stamp_ns = stamp_ns        
        # Only consider the specified target class
        matches = [
            d for d in msg.detections
            if d.class_name.strip().lower() == self.target_class
            and d.score >= self.target_confidence
        ]

        if not matches:
            self.target_confirmation_count = 0
            return

        # Use highest-confidence detection
        detection = max(matches, key=lambda d: d.score)

        center_x = detection.bbox.center.position.x

        estimate = self.estimate_target_from_lidar(
            center_x,
            msg.header.stamp
        )

        if estimate is None:
            self.target_confirmation_count = 0
            self.pending_target_position = None

            self.get_logger().info(
                f'{self.target_class} visible, '
                'but no reliable LiDAR range yet',
                throttle_duration_sec=3.0
            )
            return

        x, y, distance = estimate

        # Check spatial consistency of consecutive detections
        if self.pending_target_position is not None:
            old_x, old_y = self.pending_target_position

            if math.hypot(x - old_x, y - old_y) > 0.20:
                self.target_confirmation_count = 0

        self.pending_target_position = (x, y)
        self.target_confirmation_count += 1

        self.get_logger().info(
            f'Target: {self.target_class} | '
            f'Confidence: {detection.score:.2f} | '
            f'LiDAR distance: {distance:.2f} m | '
            f'Map position: ({x:.2f}, {y:.2f}) | '
            f'Confirmation: {self.target_confirmation_count}/'
            f'{self.required_target_confirmations}'
        )

        if (
            self.target_confirmation_count
            >= self.required_target_confirmations
        ):
            self.get_logger().warn(
                f'TARGET FOUND: {self.target_class}. '
                'Stopping exploration.'
            )

            self.mission_state = 'STOPPING'
            self.stop_exploration()

    def start_camera_scan(self):

        if self.mission_state != 'SEARCHING':
            return

        if self.navigation_active or self.spin_active:
            return

        if self.spin_goal_pending:
            return

        if not self.spin_client.server_is_ready():
            self.get_logger().warn(
                'Nav2 Spin server unavailable; skipping scan'
            )
            return

        self.mission_state = 'SCANNING'
        self.scan_step_index = 0

        # Observe the initial direction first
        self.scan_pause_until = (
            self.get_clock().now()
            + Duration(seconds=self.scan_pause_seconds)
        )

        self.get_logger().info(
            'Starting camera scan at frontier viewpoint'
        )


    def scan_control_callback(self):

        if self.mission_state != 'SCANNING':
            return

        if self.spin_active or self.spin_goal_pending:
            return

        if self.scan_pause_until is None:
            return

        if self.get_clock().now() < self.scan_pause_until:
            return

        self.scan_pause_until = None

        # Five rotations plus the initial view = six views.
        # The final orientation is not exactly the starting
        # orientation; the camera views still cover 360 degrees
        # approximately, with overlap.
        if self.scan_step_index >= self.scan_total_steps - 1:
            self.get_logger().info(
                'Camera scan completed; resuming exploration'
            )

            self.scan_pause_until = None
            self.mission_state = 'SEARCHING'
            return

        self.send_spin_goal()


    def send_spin_goal(self):

        if self.mission_state != 'SCANNING':
            return

        if self.spin_active or self.spin_goal_pending:
            return

        if not self.spin_client.server_is_ready():

            self.get_logger().warn(
                'Spin server unavailable. '
                'Skipping camera scan.'
            )

            self.scan_pause_until = None
            self.scan_step_index = 0
            self.mission_state = 'SEARCHING'
            return
        self.cancel_requested = False
        goal = Spin.Goal()
        goal.target_yaw = float(self.scan_step_angle)
        goal.time_allowance = Duration(
            seconds=15.0
        ).to_msg()

        self.spin_goal_pending = True

        future = self.spin_client.send_goal_async(goal)
        future.add_done_callback(
            self.spin_response_callback
        )

    def spin_response_callback(self, future):

        self.spin_goal_pending = False

        try:
            handle = future.result()
        except Exception as ex:
            self.get_logger().error(
                f'Spin goal request failed: {ex}'
            )
            if self.mission_state == 'STOPPING':
                self.mission_state = 'CANCEL_FAILED'
            else:
                self.mission_state = 'SCAN_FAILED'
            return

        if not handle.accepted:
            self.get_logger().warn('Nav2 rejected Spin goal')

            if self.mission_state == 'STOPPING':
                self.finish_target_detection()
            else:
                self.mission_state = 'SCAN_FAILED'
            return

        self.spin_goal_handle = handle
        self.spin_active = True

        result_future = handle.get_result_async()
        result_future.add_done_callback(
            self.spin_result_callback
        )

        if self.mission_state == 'STOPPING':
            self.stop_exploration()

    def spin_result_callback(self, future):

        try:
            status = future.result().status
        except Exception as ex:
            self.get_logger().error(
                f'Spin result error: {ex}'
            )
            status = None

        self.spin_active = False
        self.spin_goal_handle = None

        if self.mission_state == 'STOPPING':

            if status in (
                GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_SUCCEEDED
            ):
                self.finish_target_detection()
            else:
                self.mission_state = 'CANCEL_FAILED'
            return


        if status != GoalStatus.STATUS_SUCCEEDED:

            self.get_logger().warn(
                f'Camera scan rotation failed: {status}. '
                'Skipping remaining rotations and '
                'resuming frontier exploration.'
            )

            # Clear the scanning pause
            self.scan_pause_until = None

            # Reset scanning progress
            self.scan_step_index = 0

            # Return to normal frontier exploration
            self.mission_state = 'SEARCHING'

            return


        self.scan_step_index += 1

        self.get_logger().info(
            f'Camera scan orientation '
            f'{self.scan_step_index + 1}/'
            f'{self.scan_total_steps}'
        )

        self.scan_pause_until = (
            self.get_clock().now()
            + Duration(seconds=self.scan_pause_seconds)
        )


    def send_navigation_goal(self, pose):

        if (
            self.navigation_active
            or self.spin_active
            or self.spin_goal_pending
            or self.mission_state not in ('SEARCHING', 'RETURNING')
        ):
            return

        if not self.nav_client.server_is_ready():
            self.get_logger().warn('Nav2 not ready')
            return

        goal = NavigateToPose.Goal()
        goal.pose = pose

        self.cancel_requested = False
        self.navigation_active = True
        self.current_goal = pose

        if self.mission_state == 'RETURNING':
            self.goal_type = 'HOME'
        else:
            self.goal_type = 'FRONTIER'

        self.get_logger().info(
            f'Sending {self.goal_type} goal to Nav2: '
            f'({pose.pose.position.x:.2f}, '
            f'{pose.pose.position.y:.2f})'
        )

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(
            self.goal_response_callback
        )


    def goal_response_callback(self, future):
        try:
            handle = future.result()
        except Exception as ex:
            self.get_logger().error(
                f'Navigation goal request failed: {ex}'
            )
            if self.mission_state == 'RETURNING':
                self.finish_navigation()
                self.returning_home = False
                self.mission_state = 'RETURN_FAILED'
                return
            if self.mission_state == 'STOPPING':
                self.finish_navigation()
                self.mission_state = 'CANCEL_FAILED'
                return

            if self.mission_state == 'SEARCHING':
                self.mark_goal_failed()

            self.finish_navigation()
            return

        if not handle.accepted:
            self.get_logger().warn(
                'Nav2 rejected navigation goal'
            )
            if self.mission_state == 'RETURNING':
                self.finish_navigation()
                self.returning_home = False
                self.mission_state = 'RETURN_FAILED'
                return
            if self.mission_state == 'SEARCHING':
                self.mark_goal_failed()

            self.finish_navigation()

            if self.mission_state == 'STOPPING':
                self.finish_target_detection()

            return

        self.goal_handle = handle

        result_future = handle.get_result_async()
        result_future.add_done_callback(
            self.goal_result_callback
        )

        if self.mission_state == 'STOPPING':
            self.stop_exploration()
        else:
            self.get_logger().info(
                'Nav2 accepted frontier goal'
            )

            

    def goal_result_callback(self, future):

        try:
            status = future.result().status
        except Exception as ex:
            self.get_logger().error(
                f'Navigation result error: {ex}'
            )
            status = None

        # Handle return-home navigation result
        if self.mission_state == 'RETURNING':

            self.finish_navigation()

            if status == GoalStatus.STATUS_SUCCEEDED:

                self.returning_home = False
                self.mission_state = 'FINISHED'

                self.get_logger().info(
                    'RETURN COMPLETE: Robot reached starting pose'
                )

            else:
                self.returning_home = False
                self.mission_state = 'RETURN_FAILED'

                self.get_logger().error(
                    f'Return to start failed: {status}'
                )

            return

        if self.mission_state == 'STOPPING':
            self.finish_navigation()

            if status in (
                GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_SUCCEEDED
            ):
                self.finish_target_detection()
            else:
                self.get_logger().error(
                    f'Navigation termination uncertain: {status}'
                )
                self.mission_state = 'CANCEL_FAILED'
            return

        if self.mission_state != 'SEARCHING':
            self.finish_navigation()
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.frontiers_reached += 1

            self.get_logger().info(
                f'Frontier reached: {self.frontiers_reached}'
            )
        else:
            self.get_logger().warn(
                f'Frontier navigation failed: {status}'
            )
            self.mark_goal_failed()

        self.finish_navigation()

        if (
            status == GoalStatus.STATUS_SUCCEEDED
            and self.frontiers_reached
                % self.scan_every_n_frontiers == 0
        ):
            self.start_camera_scan()


    def mark_goal_failed(self):

        if self.current_goal is not None:
            self.failed_goals.append((
                self.current_goal.pose.position.x,
                self.current_goal.pose.position.y
            ))

    def finish_navigation(self):

        self.navigation_active = False
        self.current_goal = None
        self.goal_handle = None
        self.goal_type = None

    def start_exploration_callback(self, request, response):

        if self.exploration_started:
            response.success = False
            response.message = 'Exploration already started'
            return response

        try:
            x, y, yaw = self.lookup_pose('base_footprint')

        except TransformException as ex:
            response.success = False
            response.message = f'TF unavailable: {ex}'
            return response

        if not self.nav_client.server_is_ready():
            response.success = False
            response.message = 'Nav2 is not ready'
            return response

        self.start_pose = (x, y, yaw)
        if self.last_scan_time is None:
            response.success = False
            response.message = 'No valid LiDAR scan received'
            return response

        scan_age = (
            self.get_clock().now() - self.last_scan_time
        ).nanoseconds / 1e9

        if scan_age > 2.0:
            response.success = False
            response.message = 'LiDAR data is stale'
            return response
        self.exploration_started = True

        self.get_logger().info(
            f'Starting pose: ({x:.2f}, {y:.2f}), '
            f'yaw={math.degrees(yaw):.1f} degrees'
        )

        response.success = True
        response.message = 'Frontier exploration enabled'
        return response
  
    def __init__(self):
        super().__init__('lidar_coverage_explorer')
        self.max_detection_age = 1.0
        
        # Target object recognized by YOLO
        self.declare_parameter('target_class', 'bottle') ##BOTTLE
        self.target_class = (
            self.get_parameter('target_class').value
            .strip().lower()
        )
        self.target_confidence = 0.70
        self.required_target_confirmations = 3
        self.target_confirmation_count = 0
        self.last_detection_stamp_ns = None
 
        self.max_scan_detection_dt = 0.20
        self.lidar_angle_tolerance = math.radians(3.0)
        self.min_lidar_matches = 3
        self.map_msg = None
        self.free = None
        self.covered = None
        self.robot_x = None
        self.robot_y = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.origin_yaw = None
        self.height = 0
        self.width = 0
        self.scan_history = deque(maxlen=20)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self
        )

        map_qos = QoSProfile(depth=1)
        map_qos.durability = (
            DurabilityPolicy.TRANSIENT_LOCAL
        )
        map_qos.reliability = (
            ReliabilityPolicy.RELIABLE
        )

        scan_qos = QoSProfile(depth=10)
        scan_qos.reliability = (
            ReliabilityPolicy.BEST_EFFORT
        )

        self.declare_parameter(
        'map_file',
        '/home/ros2_ws/install/balance_mapping/share/balance_mapping/maps/balance_map_laab.yaml'
        )

        self.map_file = self.get_parameter(
            'map_file'
        ).value


        self.create_subscription(
            LaserScan, '/scan',
            self.scan_callback, scan_qos
        )

        self.coverage_pub = self.create_publisher(
            OccupancyGrid, '/coverage_map', map_qos
        )

        self.goal_pub = self.create_publisher(
            PoseStamped, '/coverage_goal', 10
        )

        self.frontier_pub = self.create_publisher(
            PoseArray, '/coverage_frontiers', 10
        )
        # Publisher for estimated object position
        self.target_pub = self.create_publisher(
            PointStamped,
            '/target_position',
            10
        )

        # RViz2 object visualization
        marker_qos = QoSProfile(depth=1)
        marker_qos.reliability = ReliabilityPolicy.RELIABLE
        marker_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.object_marker_pub = self.create_publisher(
            MarkerArray,
            '/detected_object_markers',
            marker_qos
        )

        # Camera calibration
        self.camera_fx = None
        self.camera_cx = None
        self.image_width = None


        # Horizontal pixel matching tolerance

        # Camera alignment relative to robot
        # Use calibrated values if camera is not
        # perfectly aligned with base_footprint.

        self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.camera_info_callback,
            10
        )

        self.create_subscription(
            DetectionArray,
            '/yolo/detections',
            self.yolo_callback,
            10
        )

        # Track LiDAR reception and successful processing
        self.last_scan_received_time = None
        self.last_scan_time = None

        # Queue scans until their timestamped TF is available
        self.pending_scans = deque(maxlen=10)

        # Maximum time to wait for a scan transform
        self.max_scan_wait = 0.5

        # Number of scans dropped because TF was unavailable
        self.dropped_scans = 0

        # Process queued scans at 50 Hz
        self.scan_processing_timer = self.create_timer(
            0.02,
            self.process_pending_scans
        )
        
        # Store estimated target coordinates
        self.target_x = None
        self.target_y = None
        self.target_found = False
        # Object confirmation
        # Mission state
        self.mission_state = 'SEARCHING'
        self.cancel_requested = False
        self.pending_target_position = None 

        # Positioning state
        self.goal_type = None

        # Camera verification timeout 

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )
        # Nav2 Spin action
        self.spin_client = ActionClient(
            self,
            Spin,
            'spin'
        )

        # Scan selected frontier viewpoints
        self.scan_every_n_frontiers = 1
        self.frontiers_reached = 0

        # Six camera views, separated by 60 degrees
        self.scan_step_angle = math.radians(60.0)
        self.scan_total_steps = 6
        self.scan_step_index = 0

        # Allow camera observations between rotations
        self.scan_pause_seconds = 1.0
        self.scan_pause_until = None

        # Spin action tracking
        self.spin_active = False
        self.spin_goal_pending = False
        self.spin_goal_handle = None

        # Periodic scan controller
        self.create_timer(
            0.1,
            self.scan_control_callback
        )

        self.exploration_started = False
        self.navigation_active = False
        self.current_goal = None
        self.goal_handle = None
        self.failed_goals = []
        self.start_pose = None
        # Return to starting position after detecting target
        self.returning_home = False
        self.home_goal_sent = False
        self.create_service(
            Trigger,
            '/start_frontier_exploration',
            self.start_exploration_callback
        )        
        self.load_map_from_file(self.map_file)
        self.create_timer(2.0, self.plan_callback)

        self.get_logger().info(
            'LiDAR coverage explorer started'
        )

    def stop_exploration(self):

        self.mission_state = 'STOPPING'

        # Cancel active Spin action first
        if self.spin_goal_handle is not None:

            if self.cancel_requested:
                return

            self.cancel_requested = True

            future = (
                self.spin_goal_handle.cancel_goal_async()
            )

            future.add_done_callback(
                self.cancel_response_callback
            )
            return

        # Wait for pending Spin goal response
        if self.spin_goal_pending or self.spin_active:
            return

        # Cancel active frontier navigation
        if self.goal_handle is not None:

            if self.cancel_requested:
                return

            self.cancel_requested = True

            future = self.goal_handle.cancel_goal_async()

            future.add_done_callback(
                self.cancel_response_callback
            )
            return

        # Wait for pending navigation goal response
        if self.navigation_active:
            return

        # No action is active
        self.finish_target_detection()

    def cancel_response_callback(self, future):

        if self.mission_state != 'STOPPING':
            return

        try:
            response = future.result()

            if not response.goals_canceling:
                self.get_logger().error(
                    'Cancellation not accepted'
                )
                self.mission_state = 'CANCEL_FAILED'
                return

            self.get_logger().info(
                'Cancellation accepted'
            )

        except Exception as ex:
            self.get_logger().error(
                f'Cancellation failed: {ex}'
            )
            self.mission_state = 'CANCEL_FAILED'

    def return_to_start(self):

        if self.start_pose is None:
            self.get_logger().error(
                'Cannot return home: starting pose unavailable'
            )
            self.mission_state = 'RETURN_FAILED'
            return

        if self.navigation_active or self.spin_active:
            return

        if self.home_goal_sent:
            return

        x, y, yaw = self.start_pose

        home_pose = self.make_pose(x, y, yaw)

        self.get_logger().info(
            f'Returning to start: '
            f'x={x:.2f}, y={y:.2f}, '
            f'yaw={math.degrees(yaw):.1f} degrees'
        )
        if not self.nav_client.server_is_ready():
            self.get_logger().error(
                'Nav2 unavailable: cannot return to start'
            )
            self.mission_state = 'RETURN_FAILED'
            return
        self.mission_state = 'RETURNING'
        self.returning_home = True
        self.home_goal_sent = True

        self.send_navigation_goal(home_pose)


    def finish_target_detection(self):

        if self.mission_state != 'STOPPING':
            return

        # Wait until all previous actions have terminated
        if (
            self.navigation_active
            or self.spin_active
            or self.spin_goal_pending
            or self.goal_handle is not None
            or self.spin_goal_handle is not None
        ):
            return

        if self.pending_target_position is None:
            self.mission_state = 'UNVERIFIED'
            return

        if self.start_pose is None:
            self.get_logger().error(
                'Starting pose unavailable'
            )
            self.mission_state = 'RETURN_FAILED'
            return

        x, y = self.pending_target_position

        # Publish bottle position before returning
        self.publish_target_position(x, y)

        self.get_logger().info(
            f'Bottle localized at ({x:.3f}, {y:.3f})'
        )

        # Reset cancellation state
        self.cancel_requested = False

        # Return to the original starting pose
        self.return_to_start()


    def estimate_target_from_lidar(self, pixel_x, detection_stamp):

        if self.camera_fx is None or not self.scan_history:
            return None

        image_time = Time.from_msg(
            detection_stamp,
            clock_type=self.get_clock().clock_type
        )

        scan = min(
            self.scan_history,
            key=lambda s: abs(
                (
                    Time.from_msg(
                        s.header.stamp,
                        clock_type=self.get_clock().clock_type
                    ) - image_time
                ).nanoseconds
            )
        )

        # Compare camera capture and LiDAR scan times
        scan_time = Time.from_msg(
            scan.header.stamp,
            clock_type=self.get_clock().clock_type
        )

        dt = abs(
            (scan_time - image_time).nanoseconds / 1e9
        )

        if dt > self.max_scan_detection_dt:
            return None

        # Horizontal camera bearing.
        # Assumes camera and LiDAR face the same direction.
        bearing = -math.atan2(
            pixel_x - self.camera_cx,
            self.camera_fx
        )

        # Find LiDAR beams around this bearing
        ranges = []
        for i, r in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment

            error = math.atan2(
                math.sin(angle - bearing),
                math.cos(angle - bearing)
            )

            if abs(error) > self.lidar_angle_tolerance:
                continue

            if not math.isfinite(r):
                continue

            if not (scan.range_min <= r <= scan.range_max):
                continue

            if r > 3.0:
                continue

            ranges.append((r, angle))

        if len(ranges) < self.min_lidar_matches:
            return None

        # Median is less sensitive to isolated range errors
        distance = float(np.median([r for r, _ in ranges]))

        # Reject a window with strongly inconsistent ranges
        consistent = [
            (r, a) for r, a in ranges
            if abs(r - distance) < 0.10
        ]

        if len(consistent) < self.min_lidar_matches:
            return None

        angle = float(np.median([a for _, a in consistent]))

        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                scan.header.frame_id,
                scan_time
            )
        except TransformException:
            return None

        p = transform.transform.translation
        yaw = yaw_from_quaternion(
            transform.transform.rotation
        )

        # Estimated object position in map frame
        target_x = p.x + distance * math.cos(yaw + angle)
        target_y = p.y + distance * math.sin(yaw + angle)

        return target_x, target_y, distance


    def load_map_from_file(self, yaml_path):

        # Step 1: Read the YAML file
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)

        # Step 2: Get map metadata
        self.resolution = float(config['resolution'])

        origin = config['origin']

        self.origin_x = float(origin[0])
        self.origin_y = float(origin[1])
        self.origin_yaw = float(origin[2])

        # Step 3: Find the corresponding PGM image
        image_name = config['image']

        if os.path.isabs(image_name):
            image_path = image_name
        else:
            image_path = os.path.join(
                os.path.dirname(yaml_path),
                image_name
            )

        # Step 4: Read the image
        image = Image.open(image_path).convert('L')

        pixels = np.array(image, dtype=np.uint8)

        # Convert image coordinates to ROS map coordinates
        pixels = np.flipud(pixels)

        self.height, self.width = pixels.shape

        # Step 5: Convert pixels to occupancy values
        negate = int(config.get('negate', 0))

        occupied_thresh = float(
            config.get('occupied_thresh', 0.65)
        )

        free_thresh = float(
            config.get('free_thresh', 0.196)
        )

        if negate == 0:
            probability = (
                255.0 - pixels.astype(np.float32)
            ) / 255.0
        else:
            probability = (
                pixels.astype(np.float32) / 255.0
            )

        grid = np.full(
            (self.height, self.width),
            -1,
            dtype=np.int8
        )

        grid[probability < free_thresh] = 0

        grid[probability > occupied_thresh] = 100

        # Step 6: Create OccupancyGrid message
        msg = OccupancyGrid()

        msg.header.frame_id = 'map'
        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height

        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0

        msg.info.origin.orientation.z = math.sin(
            self.origin_yaw / 2.0
        )

        msg.info.origin.orientation.w = math.cos(
            self.origin_yaw / 2.0
        )

        msg.data = grid.flatten().tolist()

        # Step 7: Initialize coverage map
        self.map_msg = msg

        self.free = (grid >= 0) & (grid < 20)

        self.covered = np.zeros(
            (self.height, self.width),
            dtype=bool
        )

        self.get_logger().info(
            f'Map loaded successfully: '
            f'{self.width} x {self.height}, '
            f'resolution = {self.resolution} m/cell'
        )

        self.get_logger().info(
            f'Free cells: {np.count_nonzero(self.free)}'
        )

    def find_safe_nearby_viewpoint(self, gx, gy):
        # Search nearby cells for a safe observation point.
        search_radius = math.ceil(0.6 / self.resolution)

        best = None
        best_distance = float('inf')

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):

                nx = gx + dx
                ny = gy + dy

                if not self.valid(nx, ny):
                    continue

                if not self.safe_viewpoint(nx, ny):
                    continue

                distance = math.hypot(dx, dy)

                if distance < best_distance:
                    best_distance = distance
                    best = (nx, ny)

        return best

    def world_to_grid(self, x, y):
        dx = x - self.origin_x
        dy = y - self.origin_y

        c = math.cos(self.origin_yaw)
        s = math.sin(self.origin_yaw)

        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy

        gx = math.floor(local_x / self.resolution)
        gy = math.floor(local_y / self.resolution)

        return gx, gy

    def grid_to_world(self, gx, gy):
        lx = (gx + 0.5) * self.resolution
        ly = (gy + 0.5) * self.resolution

        c = math.cos(self.origin_yaw)
        s = math.sin(self.origin_yaw)

        x = self.origin_x + c * lx - s * ly
        y = self.origin_y + s * lx + c * ly

        return x, y

    def valid(self, gx, gy):
        return (
            0 <= gx < self.width and
            0 <= gy < self.height
        )

    def lookup_pose(self, frame):
        transform = self.tf_buffer.lookup_transform(
            'map',
            frame,
            rclpy.time.Time()
        )

        p = transform.transform.translation
        q = transform.transform.rotation

        return p.x, p.y, yaw_from_quaternion(q)

    def scan_callback(self, msg):
        now = self.get_clock().now()
        self.last_scan_received_time = now
        self.pending_scans.append((msg, now))


    
    def process_pending_scans(self):

        if not self.pending_scans:
            return

        msg, received_time = self.pending_scans[0]

        wait_age = (
            self.get_clock().now() - received_time
        ).nanoseconds / 1e9

        if wait_age < 0 or wait_age > self.max_scan_wait:
            self.pending_scans.popleft()
            self.dropped_scans += 1
            return

        scan_time = Time.from_msg(
            msg.header.stamp,
            clock_type=self.get_clock().clock_type
        )

        if not self.tf_buffer.can_transform(
            'map',
            msg.header.frame_id,
            scan_time
        ):
            return

        self.pending_scans.popleft()
        self.process_scan(msg)

    def process_scan(self, msg):

        if self.free is None:
            return

        try:
            # Get LiDAR pose at scan timestamp
            transform = self.tf_buffer.lookup_transform(
                'map',
                msg.header.frame_id,
                rclpy.time.Time.from_msg(
                    msg.header.stamp
                )
            )

            p = transform.transform.translation
            sx = p.x
            sy = p.y

            q = transform.transform.rotation
            yaw = yaw_from_quaternion(q)

            rx, ry, _ = self.lookup_pose(
                'base_footprint'
            )

        except TransformException as ex:
            self.get_logger().warn(
                f'LiDAR TF unavailable: {ex}',
                throttle_duration_sec=3.0
            )
            return
        self.scan_history.append(msg)
        self.last_scan_time = self.get_clock().now()

        self.robot_x = rx
        self.robot_y = ry

        # Keep all your existing scan processing below.
        if (
            not self.exploration_started
            or self.mission_state != 'SEARCHING'
        ):
            return
        x0, y0 = self.world_to_grid(sx, sy)

        if not self.valid(x0, y0):
            return

        max_range = min(msg.range_max, 3.0)
        # Subsample beams to reduce computation.
        for i in range(0, len(msg.ranges), 4):
            distance = msg.ranges[i]

            if math.isnan(distance):
                continue

            if math.isinf(distance):
                distance = max_range
            elif distance < msg.range_min:
                continue
            else:
                distance = min(distance, max_range)

            angle = (
                yaw + msg.angle_min +
                i * msg.angle_increment
            )

            end_x = sx + distance * math.cos(angle)
            end_y = sy + distance * math.sin(angle)

            x1, y1 = self.world_to_grid(end_x, end_y)
            # Check only finite, genuine LiDAR returns

            for gx, gy in bresenham(
                x0, y0, x1, y1
            ):
                if not self.valid(gx, gy):
                    break

                # Do not mark obstacles or cells
                # behind obstacles as inspected.
                if not self.free[gy, gx]:
                    break

                self.covered[gy, gx] = True

        # Remove contributions from the oldest scan



    def find_frontiers(self):
        if self.free is None:
            return []

        unsearched = self.free & ~self.covered
        frontiers = []

        for gy in range(1, self.height - 1):
            for gx in range(1, self.width - 1):

                if not unsearched[gy, gx]:
                    continue

                neighbors = self.covered[
                    gy-1:gy+2,
                    gx-1:gx+2
                ]

                if np.any(neighbors):
                    frontiers.append((gx, gy))

        return frontiers

    def safe_viewpoint(self, gx, gy):
        """Require free-space clearance around goal."""
        radius = math.ceil(0.25 / self.resolution)

        if (
            gx - radius < 0 or
            gy - radius < 0 or
            gx + radius >= self.width or
            gy + radius >= self.height
        ):
            return False

        region = self.free[
            gy-radius:gy+radius+1,
            gx-radius:gx+radius+1
        ]

        return bool(np.all(region))

    def make_pose(self, x, y, yaw=0.0):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)

        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)

        return pose

    def publish_coverage(self):
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )
        msg.info = self.map_msg.info

        output = np.full(
            (self.height, self.width),
            -1, dtype=np.int8
        )

        output[self.free] = 0
        output[self.covered & self.free] = 100

        msg.data = output.flatten().tolist()
        self.coverage_pub.publish(msg)


    def publish_object_marker(self, x, y):

        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Green sphere at the estimated bottle position
        sphere = Marker()
        sphere.header.frame_id = 'map'
        sphere.header.stamp = stamp
        sphere.ns = 'detected_bottle'
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD

        sphere.pose.position.x = float(x)
        sphere.pose.position.y = float(y)
        sphere.pose.position.z = 0.10
        sphere.pose.orientation.w = 1.0

        sphere.scale.x = 0.15
        sphere.scale.y = 0.15
        sphere.scale.z = 0.15

        sphere.color.r = 0.0
        sphere.color.g = 1.0
        sphere.color.b = 0.0
        sphere.color.a = 1.0

        sphere.lifetime = Duration(
            seconds=0.0
        ).to_msg()

        markers.markers.append(sphere)

        # Floating text showing class and map coordinates
        label = Marker()
        label.header.frame_id = 'map'
        label.header.stamp = stamp
        label.ns = 'detected_bottle_label'
        label.id = 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD

        label.pose.position.x = float(x)
        label.pose.position.y = float(y)
        label.pose.position.z = 0.35
        label.pose.orientation.w = 1.0

        label.scale.z = 0.12

        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 1.0

        label.text = (
            f'{self.target_class}: '
            f'({x:.2f}, {y:.2f})'
        )

        label.lifetime = Duration(
            seconds=0.0
        ).to_msg()

        markers.markers.append(label)

        self.object_marker_pub.publish(markers)

        self.get_logger().info(
            'Published detected object marker to RViz2'
        )

    def publish_target_position(self, x, y):

        # Create PointStamped message
        target_msg = PointStamped()

        # Set coordinate frame
        target_msg.header.frame_id = 'map'

        target_msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        # Set estimated object coordinates
        target_msg.point.x = float(x)
        target_msg.point.y = float(y)
        target_msg.point.z = 0.0

        # Publish coordinates
        self.target_pub.publish(target_msg)

        # Store target position
        self.target_x = float(x)
        self.target_y = float(y)
        self.target_found = True

        # Publish the bottle marker in RViz2
        self.publish_object_marker(x, y)

        self.get_logger().info(
            f'OBJECT POSITION ESTIMATED: '
            f'x={x:.3f} m, '
            f'y={y:.3f} m'
        )

        
    def plan_callback(self):
        if self.mission_state != 'SEARCHING':
            return
        if self.last_scan_time is None:
            return

        scan_age = (
            self.get_clock().now()
            - self.last_scan_time
        ).nanoseconds / 1e9

        if scan_age > 1.0:
            self.get_logger().warn(
                'LiDAR data is stale; skipping frontier planning',
                throttle_duration_sec=3.0
            )
            return        
        if not self.exploration_started:
            return
        if self.navigation_active:
            return
        # Check whether the planning timer is running
        self.get_logger().info(
            'Planning callback running'
        )

        # Check whether the map has been loaded
        if self.free is None:
            self.get_logger().warn(
                'Map has not been loaded'
            )
            return

        # Check whether the robot position is available
        if self.robot_x is None:
            self.get_logger().warn(
                'Robot position not available - check LiDAR and TF'
            )
            return

        self.get_logger().info(
            f'Robot position: '
            f'x={self.robot_x:.2f}, '
            f'y={self.robot_y:.2f}'
        )

        # Publish the current coverage map
        self.publish_coverage()

        # Find frontier cells
        frontiers = self.find_frontiers()

        # Calculate coverage percentage
        total_free = np.count_nonzero(self.free)

        total_covered = np.count_nonzero(
            self.covered & self.free
        )

        coverage_percentage = (
            100.0 * total_covered / total_free
            if total_free > 0 else 0.0
        )

        self.get_logger().info(
            f'Coverage: {coverage_percentage:.1f}% | '
            f'Frontiers detected: {len(frontiers)}'
        )

        # Create PoseArray for visualization
        pose_array = PoseArray()
        pose_array.header.frame_id = 'map'
        pose_array.header.stamp = (
            self.get_clock().now().to_msg()
        )

        candidates = []

        rejected_clearance = 0
        rejected_distance = 0

        # Reduce number of candidate viewpoints
        for gx, gy in frontiers[::5]:

            viewpoint = self.find_safe_nearby_viewpoint(gx, gy)

            if viewpoint is None:
                rejected_clearance += 1
                continue

            gx, gy = viewpoint

            # Convert grid coordinates to world coordinates
            x, y = self.grid_to_world(gx, gy)
            if any(
                math.hypot(x - fx, y - fy) < 0.30
                for fx, fy in self.failed_goals
            ):
                continue
            # Calculate distance from robot
            distance = math.hypot(
                x - self.robot_x,
                y - self.robot_y
            )

            # Ignore waypoints too close to robot
            if distance < 0.4:
                rejected_distance += 1
                continue

            # Calculate robot orientation
            yaw = math.atan2(
                y - self.robot_y,
                x - self.robot_x
            )

            pose = self.make_pose(x, y, yaw)

            pose_array.poses.append(pose.pose)
            candidates.append((distance, pose))

        # Publish candidate frontier positions
        self.frontier_pub.publish(pose_array)

        # Print candidate statistics
        self.get_logger().info(
            f'Valid candidates: {len(candidates)} | '
            f'Rejected (clearance): {rejected_clearance} | '
            f'Rejected (distance): {rejected_distance}'
        )

        # Check if any safe candidates exist
        if not candidates:
            self.get_logger().warn(
                'No safe frontier candidates found'
            )
            return

        # Select nearest candidate
        candidates.sort(key=lambda item: item[0])

        next_goal = candidates[0][1]

        # Publish selected goal
        self.goal_pub.publish(next_goal)

        # Command Nav2 to navigate to the frontier
        self.send_navigation_goal(next_goal)

        self.get_logger().info(
            'Published candidate coverage goal: '
            f'({next_goal.pose.position.x:.2f}, '
            f'{next_goal.pose.position.y:.2f})'
        )



def main(args=None):
    rclpy.init(args=args)
    node = CoverageExplorer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
