import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    def __init__(self):
        super().__init__('wall_follow_node')
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.kp, self.ki, self.kd = 1.0, 0.0, 0.1
        self.integral, self.prev_error = 0.0, 0.0
        self.desired_distance, self.lookahead_distance = 1.0, 1.0
        self.theta = np.deg2rad(45)
        self.angle_min, self.angle_max = -np.pi, np.pi

    def get_range(self, range_data, angle):
        angle = np.clip(angle, self.angle_min, self.angle_max)
        index = int((angle - self.angle_min) / (self.angle_max - self.angle_min) * (len(range_data)-1))
        r = range_data[index]
        return 10.0 if np.isnan(r) or np.isinf(r) else r

    def get_error(self, range_data, desired_distance):
        b = self.get_range(range_data, np.pi/2)
        a = self.get_range(range_data, np.pi/2 - self.theta)
        alpha = np.arctan2(a*np.cos(self.theta)-b, a*np.sin(self.theta))
        D_t = b*np.cos(alpha)
        D_t1 = D_t + self.lookahead_distance*np.sin(alpha)
        return desired_distance - D_t1

    def pid_control(self, error, velocity):
        P = self.kp*error
        self.integral += error
        I = self.ki*self.integral
        D = self.kd*(error - self.prev_error)
        angle = np.clip(P + I + D, -np.deg2rad(30), np.deg2rad(30))
        self.prev_error = error
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(angle)
        msg.drive.speed = float(velocity)
        self.drive_publisher.publish(msg)

    def scan_callback(self, msg):
        self.angle_min, self.angle_max = msg.angle_min, msg.angle_max
        error = self.get_error(msg.ranges, self.desired_distance)
        angle_deg = np.rad2deg(abs(self.kp*error))
        velocity = 1.5 if angle_deg<=10 else 1.0 if angle_deg<=20 else 0.5
        self.pid_control(error, velocity)

def main(args=None):
    rclpy.init(args=args)
    node = WallFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
