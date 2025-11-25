# ROS subscribers/publishers
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile=qos)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.sub_rviz_goal = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_vis = self.create_publisher(MarkerArray, '/dwa_vis', 10)
        
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("DWB Local Planner initialized (Nav2 Python Port)")#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, Point, PoseStamped, Pose
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion
import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import copy

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Trajectory2D:
    """Represents a 2D trajectory"""
    vx: float = 0.0
    vy: float = 0.0
    theta: float = 0.0
    
    def __eq__(self, other):
        return (self.vx == other.vx and 
                self.vy == other.vy and 
                self.theta == other.theta)

@dataclass
class CriticScore:
    """Score from a single critic"""
    name: str = ""
    raw_score: float = 0.0
    scale: float = 1.0

@dataclass
class TrajectoryScore:
    """Total score for a trajectory"""
    traj: Trajectory2D = None
    scores: List[CriticScore] = None
    total: float = 0.0
    
    def __post_init__(self):
        if self.traj is None:
            self.traj = Trajectory2D()
        if self.scores is None:
            self.scores = []

# ============================================================================
# TRAJECTORY GENERATOR (Equivalent to Nav2's)
# ============================================================================

class TrajectoryGenerator:
    """Generates candidate trajectories for DWA sampling"""
    
    def __init__(self, max_speed: float, max_yaw_rate: float,
                 accel_lim_x: float, accel_lim_theta: float,
                 predict_time: float, sim_granularity: float,
                 v_samples: int, theta_samples: int):
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.accel_lim_x = accel_lim_x
        self.accel_lim_theta = accel_lim_theta
        self.predict_time = predict_time
        self.sim_granularity = sim_granularity
        self.v_samples = v_samples
        self.theta_samples = theta_samples
        
        self.velocity_iterator = []
        self.yaw_iterator = []
        self.v_index = 0
        self.theta_index = 0
    
    def startNewIteration(self, current_velocity: Trajectory2D):
        """Reset iterators and prepare for new iteration"""
        # Dynamic Window - acceleration limits
        min_v = max(0.0, current_velocity.vx - self.accel_lim_x * self.sim_granularity)
        max_v = min(self.max_speed, current_velocity.vx + self.accel_lim_x * self.sim_granularity)
        
        min_theta = max(-self.max_yaw_rate, 
                       current_velocity.theta - self.accel_lim_theta * self.sim_granularity)
        max_theta = min(self.max_yaw_rate,
                       current_velocity.theta + self.accel_lim_theta * self.sim_granularity)
        
        # Generate samples
        self.velocity_iterator = np.linspace(min_v, max_v, self.v_samples).tolist()
        self.yaw_iterator = np.linspace(min_theta, max_theta, self.theta_samples).tolist()
        
        self.v_index = 0
        self.theta_index = 0
    
    def hasMoreTwists(self) -> bool:
        """Check if more trajectories to generate"""
        return self.v_index < len(self.velocity_iterator)
    
    def nextTwist(self) -> Trajectory2D:
        """Get next velocity command"""
        v = self.velocity_iterator[self.v_index]
        theta = self.yaw_iterator[self.theta_index]
        
        twist = Trajectory2D(vx=v, vy=0.0, theta=theta)
        
        self.theta_index += 1
        if self.theta_index >= len(self.yaw_iterator):
            self.theta_index = 0
            self.v_index += 1
        
        return twist
    
    def generateTrajectory(self, pose: Pose, velocity: Trajectory2D,
                          twist: Trajectory2D) -> List[Tuple[float, float]]:
        """Simulate trajectory given initial pose, current velocity, and desired twist"""
        trajectory = []
        x, y = 0.0, 0.0  # Local frame
        theta = 0.0
        
        vx = velocity.vx
        vtheta = velocity.theta
        
        time = 0.0
        while time <= self.predict_time:
            # Add point to trajectory
            trajectory.append((x, y))
            
            # Update velocity (kinematic model)
            vx = twist.vx
            vtheta = twist.theta
            
            # Update pose
            if abs(vtheta) < 1e-6:
                # Straight line
                x += vx * self.sim_granularity * math.cos(theta)
                y += vx * self.sim_granularity * math.sin(theta)
            else:
                # Arc motion
                radius = vx / vtheta
                theta_new = theta + vtheta * self.sim_granularity
                x += radius * (math.sin(theta_new) - math.sin(theta))
                y += radius * (-math.cos(theta_new) + math.cos(theta))
                theta = theta_new
            
            time += self.sim_granularity
        
        return trajectory

# ============================================================================
# TRAJECTORY CRITICS
# ============================================================================

class TrajectoryCritic(ABC):
    """Base class for trajectory critics"""
    
    def __init__(self, name: str):
        self.name = name
        self.scale = 1.0
    
    def getName(self) -> str:
        return self.name
    
    def getScale(self) -> float:
        return self.scale
    
    def setScale(self, scale: float):
        self.scale = scale
    
    def prepare(self, pose: Pose, velocity: Trajectory2D, 
                goal_pose: Pose, transformed_plan: Path) -> bool:
        """Prepare critic for scoring (called once per planning cycle)"""
        return True
    
    @abstractmethod
    def scoreTrajectory(self, trajectory: List[Tuple[float, float]]) -> float:
        """Score a trajectory. Return is added to total cost."""
        pass
    
    def debrief(self, cmd_vel: Trajectory2D):
        """Called after trajectory is selected"""
        pass

class ObstacleCritic(TrajectoryCritic):
    """Scores based on proximity to obstacles"""
    
    def __init__(self, name: str = "ObstacleCritic"):
        super().__init__(name)
        self.scale = 2.0
        self.scan_obs = np.empty((0, 2))
        self.robot_radius = 0.20
    
    def setScanData(self, obs: np.ndarray):
        self.scan_obs = obs
    
    def prepare(self, pose: Pose, velocity: Trajectory2D,
                goal_pose: Pose, transformed_plan: Path) -> bool:
        return True
    
    def scoreTrajectory(self, trajectory: List[Tuple[float, float]]) -> float:
        """Exponential cost for proximity to obstacles"""
        if len(self.scan_obs) == 0 or len(trajectory) == 0:
            return 0.0
        
        min_dist = float('inf')
        for point in trajectory[::2]:
            obs_array = self.scan_obs
            point_array = np.array(point)
            distances = np.linalg.norm(obs_array - point_array, axis=1)
            min_dist = min(min_dist, np.min(distances))
        
        if min_dist < self.robot_radius:
            return float('inf')  # Collision
        
        # Exponential cost
        return max(0.0, (self.robot_radius - min_dist) * 10.0)

class GoalDistanceCritic(TrajectoryCritic):
    """Scores based on distance to goal after trajectory"""
    
    def __init__(self, name: str = "GoalDistanceCritic"):
        super().__init__(name)
        self.scale = 2.0
        self.goal_pose = None
    
    def prepare(self, pose: Pose, velocity: Trajectory2D,
                goal_pose: Pose, transformed_plan: Path) -> bool:
        self.goal_pose = goal_pose
        return True
    
    def scoreTrajectory(self, trajectory: List[Tuple[float, float]]) -> float:
        """Distance from trajectory end to goal"""
        if self.goal_pose is None or len(trajectory) == 0:
            return float('inf')
        
        end_pos = np.array(trajectory[-1])
        goal_pos = np.array([self.goal_pose.position.x, self.goal_pose.position.y])
        
        distance = np.linalg.norm(goal_pos - end_pos)
        return distance

class SpeedCritic(TrajectoryCritic):
    """Encourages forward motion"""
    
    def __init__(self, name: str = "SpeedCritic"):
        super().__init__(name)
        self.scale = 0.1
        self.max_speed = 0.18
        self.velocity = None
    
    def prepare(self, pose: Pose, velocity: Trajectory2D,
                goal_pose: Pose, transformed_plan: Path) -> bool:
        self.velocity = velocity
        return True
    
    def scoreTrajectory(self, trajectory: List[Tuple[float, float]]) -> float:
        """Penalize low speed"""
        if self.velocity is None:
            return float('inf')
        
        return (self.max_speed - self.velocity.vx)

class PathCritic(TrajectoryCritic):
    """Scores based on alignment with global path"""
    
    def __init__(self, name: str = "PathCritic"):
        super().__init__(name)
        self.scale = 0.5
        self.transformed_plan = None
    
    def prepare(self, pose: Pose, velocity: Trajectory2D,
                goal_pose: Pose, transformed_plan: Path) -> bool:
        self.transformed_plan = transformed_plan
        return True
    
    def scoreTrajectory(self, trajectory: List[Tuple[float, float]]) -> float:
        """Distance from trajectory to path"""
        if self.transformed_plan is None or len(trajectory) == 0:
            return 0.0
        
        end_pos = np.array(trajectory[-1])
        min_path_dist = float('inf')
        
        for pose in self.transformed_plan.poses:
            path_pos = np.array([pose.pose.position.x, pose.pose.position.y])
            dist = np.linalg.norm(path_pos - end_pos)
            min_path_dist = min(min_path_dist, dist)
        
        return min_path_dist

# ============================================================================
# DWB LOCAL PLANNER (Exact Nav2 Python Port)
# ============================================================================

class DWBLocalPlanner(Node):
    """Equivalent to dwb_core::DWBLocalPlanner in C++"""
    
    def __init__(self):
        super().__init__('dwb_local_planner')
        
        # Parameters (equivalent to ROS parameter declarations)
        self.max_speed = 0.18
        self.max_yaw_rate = 1.0
        self.accel_lim_x = 1.0
        self.accel_lim_theta = 2.0
        self.predict_time = 2.0
        self.sim_granularity = 0.1
        self.robot_radius = 0.20
        self.transform_tolerance = 0.1
        self.short_circuit_trajectory_evaluation = True
        self.debug_trajectory_details = False
        
        # Trajectory generator (plugin-like in Nav2)
        self.traj_generator = TrajectoryGenerator(
            max_speed=self.max_speed,
            max_yaw_rate=self.max_yaw_rate,
            accel_lim_x=self.accel_lim_x,
            accel_lim_theta=self.accel_lim_theta,
            predict_time=self.predict_time,
            sim_granularity=self.sim_granularity,
            v_samples=15,
            theta_samples=15
        )
        
        # Initialize critics (equivalent to loadCritics())
        self.critics: List[TrajectoryCritic] = [
            ObstacleCritic("ObstacleCritic"),
            GoalDistanceCritic("GoalDistanceCritic"),
            SpeedCritic("SpeedCritic"),
            PathCritic("PathCritic"),
        ]
        
        # Robot state
        self.pose = Pose()
        self.velocity = Trajectory2D()
        self.goal_pose = Pose()
        self.global_plan = Path()  # Store global path
        self.transformed_plan = Path()
        self.scan_obs = np.empty((0, 2))
        
        # ROS subscribers/publishers
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile=qos)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_vis = self.create_publisher(MarkerArray, '/dwa_vis', 10)
    
    def odom_cb(self, msg: Odometry):
        """Update robot pose and velocity"""
        self.pose.position.x = msg.pose.pose.position.x
        self.pose.position.y = msg.pose.pose.position.y
        self.pose.position.z = msg.pose.pose.position.z
        self.pose.orientation = msg.pose.pose.orientation
        
        self.velocity.vx = msg.twist.twist.linear.x
        self.velocity.vy = msg.twist.twist.linear.y
        _, _, yaw = euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        self.velocity.theta = msg.twist.twist.angular.z
    
    def scan_cb(self, msg: LaserScan):
        """Process laser scan"""
        ranges = np.array(msg.ranges)
        if len(ranges) == 0:
            return
        
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        mask = (ranges > 0.05) & (ranges < 4.0)
        
        obs_x = ranges[mask] * np.cos(angles[mask])
        obs_y = ranges[mask] * np.sin(angles[mask])
        self.scan_obs = np.vstack((obs_x, obs_y)).T
        
        # Update obstacle critic
        for critic in self.critics:
            if isinstance(critic, ObstacleCritic):
                critic.setScanData(self.scan_obs)
    
    def goal_cb(self, msg: PoseStamped):
        """Set goal"""
        self.goal_pose = msg.pose
        self.get_logger().info(f"New goal: ({msg.pose.position.x}, {msg.pose.position.y})")
    
    def plan_cb(self, msg: Path):
        """Receive global plan from path planner (e.g., Nav2 planner_server)"""
        self.global_plan = msg
        if len(msg.poses) > 0:
            self.goal_pose = msg.poses[-1].pose
            self.get_logger().info(f"Received plan with {len(msg.poses)} waypoints")
    
    def plan_cb(self, msg: Path):
        """Receive global plan from path planner (e.g., Nav2 planner_server)"""
        self.global_plan = msg
        if len(msg.poses) > 0:
            self.goal_pose = msg.poses[-1].pose
            self.get_logger().info(f"Received plan with {len(msg.poses)} waypoints")
    
    def transformGlobalPlan(self) -> Path:
        """Transform global plan to local frame (base_link) like Nav2 does"""
        if len(self.global_plan.poses) == 0:
            # Fallback to just goal
            transformed = Path()
            transformed.header.frame_id = "base_link"
            goal_pose_stamped = PoseStamped()
            goal_pose_stamped.pose = self.goal_pose
            transformed.poses.append(goal_pose_stamped)
            return transformed
        
        # Transform plan poses from global frame to local frame (base_link)
        transformed = Path()
        transformed.header.frame_id = "base_link"
        
        for global_pose in self.global_plan.poses:
            # Convert global frame pose to local frame (simple 2D rotation/translation)
            gx = global_pose.pose.position.x
            gy = global_pose.pose.position.y
            
            # Translate to robot position
            dx = gx - self.pose.position.x
            dy = gy - self.pose.position.y
            
            # Rotate by -robot_yaw (inverse rotation)
            _, _, robot_yaw = euler_from_quaternion([
                self.pose.orientation.x,
                self.pose.orientation.y,
                self.pose.orientation.z,
                self.pose.orientation.w
            ])
            
            local_x = dx * math.cos(-robot_yaw) - dy * math.sin(-robot_yaw)
            local_y = dx * math.sin(-robot_yaw) + dy * math.cos(-robot_yaw)
            
            # Add to transformed plan
            local_pose = PoseStamped()
            local_pose.pose.position.x = local_x
            local_pose.pose.position.y = local_y
            local_pose.pose.position.z = 0.0
            local_pose.pose.orientation = global_pose.pose.orientation
            transformed.poses.append(local_pose)
        
        return transformed
    
    def coreScoringAlgorithm(self) -> TrajectoryScore:
        """Equivalent to DWBLocalPlanner::coreScoringAlgorithm()"""
        best: TrajectoryScore = None
        worst: TrajectoryScore = None
        legal_trajectory_found = False
        
        # Start iteration
        self.traj_generator.startNewIteration(self.velocity)
        
        # Iterate through all candidate trajectories
        while self.traj_generator.hasMoreTwists():
            twist = self.traj_generator.nextTwist()
            trajectory = self.traj_generator.generateTrajectory(
                self.pose, self.velocity, twist
            )
            
            # Score trajectory
            try:
                score = self.scoreTrajectory(trajectory, twist)
                legal_trajectory_found = True
                
                if best is None or score.total < best.total:
                    best = score
                
                if worst is None or score.total > worst.total:
                    worst = score
            
            except Exception as e:
                # Illegal trajectory (collision, etc.)
                if self.debug_trajectory_details:
                    self.get_logger().warn(f"Trajectory rejected: {str(e)}")
                continue
        
        if not legal_trajectory_found:
            # Rotate in place to find opening
            self.get_logger().warn("No legal trajectories found! Rotating...")
            twist = Trajectory2D(vx=0.0, vy=0.0, theta=1.0)
            trajectory = self.traj_generator.generateTrajectory(
                self.pose, self.velocity, twist
            )
            best = TrajectoryScore(traj=twist)
            best.total = 0.0
        
        return best
    
    def scoreTrajectory(self, trajectory: List[Tuple[float, float]],
                       twist: Trajectory2D) -> TrajectoryScore:
        """Equivalent to DWBLocalPlanner::scoreTrajectory()"""
        score = TrajectoryScore(traj=twist)
        
        # Call prepare on all critics
        for critic in self.critics:
            if not critic.prepare(self.pose, self.velocity, self.goal_pose, 
                                 self.transformed_plan):
                self.get_logger().warn(f"Critic {critic.getName()} failed to prepare")
        
        # Score with each critic
        for critic in self.critics:
            cs = CriticScore()
            cs.name = critic.getName()
            cs.scale = critic.getScale()
            
            if cs.scale == 0.0:
                score.scores.append(cs)
                continue
            
            critic_score = critic.scoreTrajectory(trajectory)
            cs.raw_score = critic_score
            score.scores.append(cs)
            
            # Check for collision or invalid
            if math.isinf(critic_score):
                raise Exception(f"Invalid trajectory: {critic.getName()}")
            
            score.total += critic_score * cs.scale
            
            # Short circuit evaluation
            if self.short_circuit_trajectory_evaluation:
                # Stop if this trajectory is already worse than best
                # (optimization - requires passing best score)
                pass
        
        return score
    
    def computeVelocityCommands(self) -> Twist:
        """Equivalent to DWBLocalPlanner::computeVelocityCommands()"""
        self.prepareGlobalPlan()
        
        best_score = self.coreScoringAlgorithm()
        
        # Debrief critics
        for critic in self.critics:
            critic.debrief(best_score.traj)
        
        cmd_vel = Twist()
        cmd_vel.linear.x = best_score.traj.vx
        cmd_vel.linear.y = best_score.traj.vy
        cmd_vel.angular.z = best_score.traj.theta
        
        return cmd_vel
    
    def control_loop(self):
        """Main control loop"""
        if self.goal_pose.position.x == 0.0 and self.goal_pose.position.y == 0.0:
            return
        
        # Check if goal reached
        dist_to_goal = math.hypot(
            self.goal_pose.position.x - self.pose.position.x,
            self.goal_pose.position.y - self.pose.position.y
        )
        
        if dist_to_goal < 0.20:
            self.pub_cmd.publish(Twist())
            self.goal_pose = Pose()
            self.get_logger().info("GOAL REACHED")
            return
        
        cmd_vel = self.computeVelocityCommands()
        self.pub_cmd.publish(cmd_vel)
        self.publish_visualizations()
    
    def publish_visualizations(self):
        """Publish debug visualizations"""
        ma = MarkerArray()
        
        if len(self.scan_obs) > 0:
            m_obs = Marker()
            m_obs.header.frame_id = "base_link"
            m_obs.type = Marker.POINTS
            m_obs.action = Marker.ADD
            m_obs.id = 0
            m_obs.scale.x = 0.05
            m_obs.scale.y = 0.05
            m_obs.color.r = 1.0
            m_obs.color.a = 1.0
            for p in self.scan_obs[::5]:
                m_obs.points.append(Point(x=p[0], y=p[1]))
            ma.markers.append(m_obs)
        
        self.pub_vis.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = DWBLocalPlanner()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()