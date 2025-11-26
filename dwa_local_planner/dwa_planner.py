def publish_goal_marker(self):
        """Publish goal position with text label in RViz"""
        if self.goal is None:
            return
        
        # Create sphere marker for goal position
        goal_sphere = Marker()
        goal_sphere.header.frame_id = "map"
        goal_sphere.type = Marker.SPHERE
        goal_sphere.action = Marker.ADD
        goal_sphere.id = 100
        goal_sphere.pose.position.x = float(self.goal[0])
        goal_sphere.pose.position.y = float(self.goal[1])
        goal_sphere.pose.position.z = 0.1
        goal_sphere.scale.x = 0.15
        goal_sphere.scale.y = 0.15
        goal_sphere.scale.z = 0.15
        goal_sphere.color.r = 0.0
        goal_sphere.color.g = 1.0
        goal_sphere.color.b = 0.0
        goal_sphere.color.a = 0.8
        
        # Create text marker with coordinates
        goal_text = Marker()
        goal_text.header.frame_id = "map"
        goal_text.type = Marker.TEXT_VIEW_FACING
        goal_text.action = Marker.ADD
        goal_text.id = 101
        goal_text.pose.position.x = float(self.goal[0])
        goal_text.pose.position.y = float(self.goal[1])
        goal_text.pose.position.z = 0.35
        goal_text.scale.z = 0.1
        goal_text.color.r = 1.0
        goal_text.color.g = 1.0
        goal_text.color.b = 0.0
        goal_text.color.a = 1.0
        goal_text.text = f"Goal\n({self.goal[0]:.2f}, {self.goal[1]:.2f})"
        
        self.pub_goal_marker.publish(goal_sphere)
        self.pub_goal_marker.publish(goal_text)

def main(args=None):    def goal_cb(self, msg):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.get_logger().info(f"New Goal: {self.goal}")
        self.publish_goal_marker()#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion
import math
import numpy as np
from scipy.interpolate import CubicSpline

class EnhancedDWAPlanner(Node):
    def __init__(self):
        super().__init__('enhanced_dwa_planner')
        
        # --- Declare Parameters ---
        self.declare_parameter('max_speed', 0.22)
        self.declare_parameter('max_yaw_rate', 1.0)
        self.declare_parameter('min_speed', -0.22)  # Backwards motion
        self.declare_parameter('predict_time', 4.0)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('v_reso', 0.02)
        self.declare_parameter('w_reso', 0.05)
        self.declare_parameter('robot_radius', 0.13)
        self.declare_parameter('goal_cost_gain', 2.0)
        self.declare_parameter('speed_cost_gain', 0.1)
        self.declare_parameter('obs_cost_gain', 1.0)
        self.declare_parameter('smoothness_cost_gain', 0.05)  # Penalize jerk
        self.declare_parameter('max_obstacle_range', 3.0)
        self.declare_parameter('goal_tolerance', 0.20)
        self.declare_parameter('enable_backwards', True)
        self.declare_parameter('enable_path_smoothing', True)
        
        # --- Load Parameters ---
        self.load_parameters()
        
        # --- State ---
        self.goal = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.w = 0.0
        self.scan_obs = np.empty((0, 2))
        self.last_v = 0.0
        self.last_w = 0.0

        # --- ROS Setup ---
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile=qos)
        self.sub_goal_1 = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_cb, 10)
        
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_vis = self.create_publisher(MarkerArray, '/dwa_vis', 10)
        self.pub_goal_marker = self.create_publisher(Marker, '/goal_marker', 10)
        
        # Parameter callback
        self.add_on_set_parameters_callback(self.on_parameters_changed)
        
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Enhanced DWA Ready. Waiting for Goal...")
        self.get_logger().info(f"Backwards motion: {self.enable_backwards}")
        self.get_logger().info(f"Path smoothing: {self.enable_path_smoothing}")

    def load_parameters(self):
        """Load all parameters from parameter server"""
        self.max_speed = self.get_parameter('max_speed').value
        self.max_yaw_rate = self.get_parameter('max_yaw_rate').value
        self.min_speed = self.get_parameter('min_speed').value
        self.predict_time = self.get_parameter('predict_time').value
        self.dt = self.get_parameter('dt').value
        self.v_reso = self.get_parameter('v_reso').value
        self.w_reso = self.get_parameter('w_reso').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.goal_cost_gain = self.get_parameter('goal_cost_gain').value
        self.speed_cost_gain = self.get_parameter('speed_cost_gain').value
        self.obs_cost_gain = self.get_parameter('obs_cost_gain').value
        self.smoothness_cost_gain = self.get_parameter('smoothness_cost_gain').value
        self.max_obstacle_range = self.get_parameter('max_obstacle_range').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.enable_backwards = self.get_parameter('enable_backwards').value
        self.enable_path_smoothing = self.get_parameter('enable_path_smoothing').value

    def on_parameters_changed(self, params):
        """Callback when parameters are updated"""
        for param in params:
            if param.name in ['max_speed', 'max_yaw_rate', 'goal_cost_gain', 
                             'speed_cost_gain', 'obs_cost_gain', 'smoothness_cost_gain',
                             'enable_backwards', 'enable_path_smoothing']:
                self.load_parameters()
                self.get_logger().info(f"Updated parameter: {param.name} = {param.value}")
        return rclpy.parameter_client.SetParametersResult(successful=True)

    def goal_cb(self, msg):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.get_logger().info(f"New Goal: {self.goal}")

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        o = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([o.x, o.y, o.z, o.w])
        self.v = msg.twist.twist.linear.x
        self.w = msg.twist.twist.angular.z

    def scan_cb(self, msg):
        ranges = np.array(msg.ranges)
        if len(ranges) == 0: 
            self.scan_obs = np.empty((0, 2))
            return
        
        # Localized view for performance
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        mask = (ranges > 0.05) & (ranges < self.max_obstacle_range)
        
        if not np.any(mask):
            self.scan_obs = np.empty((0, 2))
            return
            
        obs_x = ranges[mask] * np.cos(angles[mask])
        obs_y = ranges[mask] * np.sin(angles[mask])
        self.scan_obs = np.vstack((obs_x, obs_y)).T

    def motion_model(self, state, u):
        """state: [x, y, yaw, v, w]"""
        state[2] += u[1] * self.dt 
        state[0] += u[0] * math.cos(state[2]) * self.dt
        state[1] += u[0] * math.sin(state[2]) * self.dt
        state[3] = u[0]
        state[4] = u[1]
        return state

    def calc_trajectory(self, v, w):
        """Predict trajectory with motion model"""
        traj = []
        state = np.array([0.0, 0.0, 0.0, self.v, self.w])
        time = 0
        while time <= self.predict_time:
            state = self.motion_model(state, [v, w])
            traj.append(state[0:2].copy())
            time += self.dt
        return np.array(traj)

    def smooth_trajectory(self, traj):
        """Apply cubic spline smoothing to reduce jerky paths"""
        if len(traj) < 4:
            return traj
        
        try:
            x = traj[:, 0]
            y = traj[:, 1]
            
            # Create parameter for spline
            t = np.linspace(0, 1, len(traj))
            
            # Fit cubic splines
            cs_x = CubicSpline(t, x)
            cs_y = CubicSpline(t, y)
            
            # Evaluate at finer resolution
            t_smooth = np.linspace(0, 1, len(traj) * 2)
            x_smooth = cs_x(t_smooth)
            y_smooth = cs_y(t_smooth)
            
            return np.column_stack((x_smooth, y_smooth))
        except Exception as e:
            self.get_logger().warn(f"Spline smoothing failed: {e}")
            return traj

    def check_collision_strict(self, trajectory):
        """Check collision with margin"""
        if len(self.scan_obs) == 0: 
            return False
        
        for point in trajectory:
            diff = self.scan_obs - point 
            dist_sq = np.sum(diff**2, axis=1) 
            min_dist_sq = np.min(dist_sq)
            
            if min_dist_sq <= self.robot_radius**2:
                return True
        return False

    def calc_smoothness_cost(self, v, w):
        """Penalize sudden velocity changes (reduce jerk)"""
        dv = abs(v - self.last_v)
        dw = abs(w - self.last_w)
        return dv + dw

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def dwa_search(self):
        """DWA search with smooth cost and backwards support"""
        # Dynamic window constraints
        vs = [self.v - 1.0*self.dt, self.v + 1.0*self.dt, self.w - 2.0*self.dt, self.w + 2.0*self.dt]
        
        if self.enable_backwards:
            dw = [max(vs[0], self.min_speed), 
                  min(vs[1], self.max_speed), 
                  max(vs[2], -self.max_yaw_rate), 
                  min(vs[3], self.max_yaw_rate)]
        else:
            dw = [max(vs[0], 0.0), 
                  min(vs[1], self.max_speed), 
                  max(vs[2], -self.max_yaw_rate), 
                  min(vs[3], self.max_yaw_rate)]
        
        best_u = [0.0, 0.0]
        best_cost = float('inf')
        best_traj = []
        
        safe_trajs = []
        unsafe_trajs = []

        v_samples = np.arange(dw[0], dw[1], self.v_reso)
        w_samples = np.arange(dw[2], dw[3], self.w_reso)
        
        if self.goal is None: 
            return best_u, best_traj, safe_trajs, unsafe_trajs

        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        local_goal_x = dx * math.cos(-self.yaw) - dy * math.sin(-self.yaw)
        local_goal_y = dx * math.sin(-self.yaw) + dy * math.cos(-self.yaw)

        for v in v_samples:
            for w in w_samples:
                traj = self.calc_trajectory(v, w)
                
                if self.check_collision_strict(traj):
                    unsafe_trajs.append(traj)
                    continue
                
                safe_trajs.append(traj)

                # Smooth trajectory if enabled
                if self.enable_path_smoothing:
                    traj_eval = self.smooth_trajectory(traj)
                else:
                    traj_eval = traj

                # Cost evaluation
                to_goal_angle = math.atan2(local_goal_y - traj_eval[-1][1], 
                                          local_goal_x - traj_eval[-1][0])
                heading_cost = abs(self.normalize_angle(to_goal_angle - traj_eval[-1][2] 
                                   if len(traj_eval[-1]) > 2 else 0))
                dist_cost = math.sqrt((local_goal_x - traj_eval[-1][0])**2 + 
                                     (local_goal_y - traj_eval[-1][1])**2)
                speed_cost = self.max_speed - abs(v)
                smoothness_cost = self.calc_smoothness_cost(v, w)

                final_cost = (self.goal_cost_gain * heading_cost +
                             self.speed_cost_gain * speed_cost +
                             self.obs_cost_gain * dist_cost +
                             self.smoothness_cost_gain * smoothness_cost)

                if final_cost < best_cost:
                    best_cost = final_cost
                    best_u = [v, w]
                    best_traj = traj

        return best_u, best_traj, safe_trajs, unsafe_trajs

    def control_loop(self):
        """Main control loop"""
        if self.goal is None: 
            return
        
        dist_to_goal = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        if dist_to_goal < self.goal_tolerance:
            self.pub_cmd.publish(Twist())
            self.goal = None
            self.get_logger().info("GOAL REACHED")
            return

        u, best_traj, safe_trajs, unsafe_trajs = self.dwa_search()

        # Store current command for smoothness cost
        self.last_v = u[0]
        self.last_w = u[1]

        cmd = Twist()
        cmd.linear.x = float(u[0])
        cmd.angular.z = float(u[1])
        self.pub_cmd.publish(cmd)
        
        self.publish_vis(best_traj, safe_trajs, unsafe_trajs)

    def publish_vis(self, best_traj, safe_trajs, unsafe_trajs):
        """Publish visualization markers"""
        ma = MarkerArray()
        
        # 1. Unsafe Paths (RED)
        if unsafe_trajs:
            m_unsafe = Marker()
            m_unsafe.header.frame_id = "base_link"
            m_unsafe.type = Marker.LINE_LIST
            m_unsafe.action = Marker.ADD
            m_unsafe.id = 1
            m_unsafe.scale.x = 0.002
            m_unsafe.color.r = 1.0
            m_unsafe.color.a = 0.2
            for traj in unsafe_trajs[::4]:
                for i in range(len(traj)-1):
                    m_unsafe.points.append(Point(x=traj[i][0], y=traj[i][1]))
                    m_unsafe.points.append(Point(x=traj[i+1][0], y=traj[i+1][1]))
            ma.markers.append(m_unsafe)
        
        # 2. Safe Paths (BLUE)
        if safe_trajs:
            m_safe = Marker()
            m_safe.header.frame_id = "base_link"
            m_safe.type = Marker.LINE_LIST
            m_safe.action = Marker.ADD
            m_safe.id = 2
            m_safe.scale.x = 0.002
            m_safe.color.b = 1.0
            m_safe.color.a = 0.2
            for traj in safe_trajs[::2]:
                for i in range(len(traj)-1):
                    m_safe.points.append(Point(x=traj[i][0], y=traj[i][1]))
                    m_safe.points.append(Point(x=traj[i+1][0], y=traj[i+1][1]))
            ma.markers.append(m_safe)

        # 3. Best Path (GREEN)
        if len(best_traj) > 0:
            m_best = Marker()
            m_best.header.frame_id = "base_link"
            m_best.type = Marker.LINE_STRIP
            m_best.action = Marker.ADD
            m_best.id = 3
            m_best.scale.x = 0.03
            m_best.color.g = 1.0
            m_best.color.a = 1.0
            for p in best_traj:
                m_best.points.append(Point(x=p[0], y=p[1]))
            ma.markers.append(m_best)
            
        self.pub_vis.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = EnhancedDWAPlanner()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()