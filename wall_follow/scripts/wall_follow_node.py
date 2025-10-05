import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        # -----------------------------
        # Topics
        # -----------------------------
        lidarscan_topic = '/scan'           # LiDAR scans
        drive_topic = '/drive'              # Car drive commands

        # -----------------------------
        # Create Subscriber for LiDAR scans
        # -----------------------------
        # When a LaserScan message is received, call scan_callback
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.scan_callback,
            10  # QoS depth
        )

        # -----------------------------
        # Create Publisher for car drive commands
        # -----------------------------
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10  # QoS depth
        )

        # -----------------------------
        # PID gains
        # -----------------------------
        # These values can be tuned later
        self.kp = 1.0
        self.ki = 0.0
        self.kd = 0.1

        # -----------------------------
        # Error history for PID
        # -----------------------------
        self.integral = 0.0        # For integral term
        self.prev_error = 0.0      # For derivative term
        self.error = 0.0           # Current error

        # -----------------------------
        # Other useful values
        # -----------------------------
        self.desired_distance = 1.0  # meters from the wall (set point)
        self.lookahead_distance = 1.0  # L in equations
        self.theta = np.deg2rad(45)   # Example LiDAR angle for a (can be tuned)


    def get_range(self, range_data, angle):
    """
    Returns the LiDAR measurement at a given angle.
    Handles NaNs and infs by returning a large value (10 m).

    Args:
        range_data: LiDAR ranges array
        angle: desired angle in radians (within angle_min and angle_max)

    Returns:
        range measurement in meters
    """
    # LiDAR info: assume uniform scan and full 360 degrees
    angle_min = -np.pi
    angle_max = np.pi
    num_ranges = len(range_data)

    # Clip angle to valid range
    angle = np.clip(angle, angle_min, angle_max)

    # Map angle to index
    index = int((angle - angle_min) / (angle_max - angle_min) * (num_ranges - 1))

    # Get distance
    r = range_data[index]

    # Handle invalid measurements
    if np.isnan(r) or np.isinf(r):
        r = 10.0

    return r


def get_error(self, range_data, desired_distance):
    """
    Compute the wall-following error using two LiDAR measurements.
    Follows left wall (counter-clockwise loop).

    Args:
        range_data: LiDAR ranges array
        desired_distance: distance to maintain from wall

    Returns:
        error (meters)
    """
    # LiDAR beam angles
    # b = left beam at 90 deg (pi/2), a = forward-left at theta (45 deg example)
    b = self.get_range(range_data, np.pi/2)
    a = self.get_range(range_data, np.pi/2 - self.theta)

    # Compute angle between car and wall
    alpha = np.arctan((a * np.cos(self.theta) - b) / (a * np.sin(self.theta)))

    # Current distance to wall
    D_t = b * np.cos(alpha)

    # Future distance with lookahead
    D_t1 = D_t + self.lookahead_distance * np.sin(alpha)

    # Error = desired distance - predicted distance
    error = desired_distance - D_t1
    return error


def pid_control(self, error, velocity):
    """
    Apply PID controller and publish Ackermann drive message.

    Args:
        error: computed wall-following error
        velocity: desired speed (m/s)
    """
    # Proportional term
    P = self.kp * error

    # Integral term
    self.integral += error
    I = self.ki * self.integral

    # Derivative term
    D = self.kd * (error - self.prev_error)

    # PID output → steering angle
    angle = P + I + D
    self.prev_error = error

    # Clamp steering angle to feasible range (e.g., -30° to 30°)
    max_angle = np.deg2rad(30)
    angle = np.clip(angle, -max_angle, max_angle)

    # Prepare drive message
    drive_msg = AckermannDriveStamped()
    drive_msg.drive.steering_angle = float(angle)
    drive_msg.drive.speed = float(velocity)

    # Publish
    self.drive_publisher.publish(drive_msg)


def scan_callback(self, msg):
    """
    Process LiDAR scan, compute error, adjust speed, and actuate car.

    Args:
        msg: sensor_msgs/LaserScan
    """
    # Compute error using LiDAR
    error = self.get_error(msg.ranges, self.desired_distance)

    # Determine speed based on PID steering angle (simplified)
    # Approximate steering angle using proportional term only for speed
    predicted_angle = self.kp * error
    predicted_angle_deg = np.rad2deg(abs(predicted_angle))

    if predicted_angle_deg <= 10:
        velocity = 1.5
    elif predicted_angle_deg <= 20:
        velocity = 1.0
    else:
        velocity = 0.5

    # Apply PID and publish drive message
    self.pid_control(error, velocity)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
