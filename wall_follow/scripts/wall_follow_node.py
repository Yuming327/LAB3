import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    def __init__(self):
        super().__init__('wall_follow_node')
        # ... [same __init__ code as yours] ...

    def get_range(self, range_data, angle):
        """
        Returns the LiDAR measurement at a given angle.
        Handles NaNs and infs by returning a large value (10 m).
        """
        angle_min = -np.pi
        angle_max = np.pi
        num_ranges = len(range_data)

        angle = np.clip(angle, angle_min, angle_max)
        index = int((angle - angle_min) / (angle_max - angle_min) * (num_ranges - 1))
        r = range_data[index]
        if np.isnan(r) or np.isinf(r):
            r = 10.0
        return r

    def get_error(self, range_data, desired_distance):
        b = self.get_range(range_data, np.pi/2)
        a = self.get_range(range_data, np.pi/2 - self.theta)
        alpha = np.arctan((a * np.cos(self.theta) - b) / (a * np.sin(self.theta)))
        D_t = b * np.cos(alpha)
        D_t1 = D_t + self.lookahead_distance * np.sin(alpha)
        error = desired_distance - D_t1
        return error

    def pid_control(self, error, velocity):
        P = self.kp * error
        self.integral += error
        I = self.ki * self.integral
        D = self.kd * (error - self.prev_error)
        angle = P + I + D
        self.prev_error = error
        max_angle = np.deg2rad(30)
        angle = np.clip(angle, -max_angle, max_angle)
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(angle)
        drive_msg.drive.speed = float(velocity)
        self.drive_publisher.publish(drive_msg)

    def scan_callback(self, msg):
        error = self.get_error(msg.ranges, self.desired_distance)
        predicted_angle = self.kp * error
        predicted_angle_deg = np.rad2deg(abs(predicted_angle))
        if predicted_angle_deg <= 10:
            velocity = 1.5
        elif predicted_angle_deg <= 20:
            velocity = 1.0
        else:
            velocity = 0.5
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
