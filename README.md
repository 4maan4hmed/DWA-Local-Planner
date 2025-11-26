# Clean DWA Planner

A lightweight, well-structured implementation of the Dynamic Window Approach (DWA) algorithm for real-time robot navigation and obstacle avoidance in ROS 2.

## Overview

The Clean DWA Planner is a local path planner that generates collision-free trajectories by evaluating a dynamic window of velocity commands. It continuously searches for the best linear and angular velocity combination that balances goal-seeking, speed, and obstacle avoidance.

**Key Features:**
- Real-time trajectory evaluation and visualization
- Collision detection with safety margins
- Dynamic window velocity constraints
- Multi-objective cost function (goal alignment, speed, obstacle distance)
- ROS 2 native implementation with Humble compatibility
- Debug visualization of safe/unsafe trajectories

## Algorithm

DWA operates in three main steps:

1. **Dynamic Window Calculation**: Restricts velocity search space based on robot acceleration limits and current state
2. **Trajectory Sampling**: Generates predicted paths for candidate velocity pairs over a prediction horizon
3. **Cost Evaluation**: Scores each trajectory using a weighted cost function and selects the best one

This enables smooth, predictable robot motion while maintaining real-time performance.

## Requirements

### Dependencies
- ROS 2 (Humble or later)
- Python 3.8+
- NumPy
- `tf_transformations`

### Install Dependencies
```bash
sudo apt install python3-numpy python3-transforms3d
```

## Installation

1. Clone into your ROS 2 workspace:
```bash
cd ~/ros2_ws/src
git clone https://github.com/yourrepo/clean_dwa_planner
cd ~/ros2_ws
```

2. Build the package:
```bash
colcon build --packages-select clean_dwa_planner
source install/setup.bash
```

## Usage

### Launch the Planner
```bash
ros2 run clean_dwa_planner clean_dwa_planner
```

The node will wait for a goal pose before starting navigation.

### Set a Goal

**Option 1: Using ROS 2 CLI**
```bash
ros2 topic pub /goal_pose geometry_msgs/PoseStamped "{header: {frame_id: 'map'}, pose: {position: {x: 5.0, y: 5.0, z: 0.0}, orientation: {x: 0, y: 0, z: 0, w: 1}}}"
```

**Option 2: Using RViz**
Use the "2D Goal Pose" tool to click a target location on the map.

**Option 3: Programmatically**
Publish to `/goal_pose` or `/move_base_simple/goal` topics with a `PoseStamped` message.

### Visualization

Monitor the planner's behavior in RViz:
```bash
rviz2
```

Add a MarkerArray display and subscribe to `/dwa_vis`. You'll see:
- **Green lines**: Selected best trajectory
- **Blue lines**: Safe candidate trajectories
- **Red lines**: Unsafe/collision trajectories
- **Red points**: Detected obstacles

## Parameters

Edit these values in the `__init__` method to tune performance:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_speed` | 0.18 m/s | Maximum forward velocity |
| `max_yaw_rate` | 1.0 rad/s | Maximum rotation rate |
| `predict_time` | 2.0 s | Prediction horizon |
| `dt` | 0.1 s | Time step for trajectory generation |
| `v_reso` | 0.02 m/s | Linear velocity resolution |
| `w_reso` | 0.05 rad/s | Angular velocity resolution |
| `robot_radius` | 0.20 m | Safety bubble radius |
| `goal_cost_gain` | 2.0 | Weight for heading error |
| `speed_cost_gain` | 0.1 | Weight for speed preference |
| `obs_cost_gain` | 2.0 | Weight for distance to obstacles |

**Tuning Tips:**
- Increase `obs_cost_gain` for more conservative obstacle avoidance
- Increase `goal_cost_gain` for more direct paths
- Adjust `max_speed` and `max_yaw_rate` for your robot's capabilities
- Higher resolution values = better quality but more CPU usage

## Topics

### Subscriptions
| Topic | Type | Description |
|-------|------|-------------|
| `/odom` | `nav_msgs/Odometry` | Robot odometry (pose & velocity) |
| `/scan` | `sensor_msgs/LaserScan` | Laser scanner data |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Goal location (primary) |
| `/move_base_simple/goal` | `geometry_msgs/PoseStamped` | Goal location (rviz compatible) |

### Publications
| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands to robot |
| `/dwa_vis` | `visualization_msgs/MarkerArray` | Debug visualization |

## How It Works

### Motion Model
The planner uses a simple kinematic model to predict robot trajectories:
```
x(t+1) = x(t) + v·cos(yaw)·dt
y(t+1) = y(t) + v·sin(yaw)·dt
yaw(t+1) = yaw(t) + w·dt
```

### Collision Detection
For each sampled trajectory, the planner checks if any predicted position comes within `robot_radius` of detected obstacles. Trajectories violating this constraint are discarded.

### Cost Function
The best trajectory minimizes:
```
cost = goal_gain·|heading_error| + speed_gain·(max_speed - v) + obs_gain·distance_to_goal
```

## Performance Considerations

- **Control Loop Rate**: 10 Hz (adjustable via `self.create_timer`)
- **CPU Usage**: Optimized to skip every 2nd trajectory point in collision checks
- **Trajectory Sampling**: 5–8 velocity samples typically sufficient for smooth motion
- **Laserscan QoS**: Uses BEST_EFFORT for compatibility with unreliable connections

## Troubleshooting

**Robot not moving?**
- Ensure odometry is publishing on `/odom`
- Verify laser data is available on `/scan`
- Check that a goal has been set

**Jerky/oscillating motion?**
- Reduce `goal_cost_gain` or increase `obs_cost_gain`
- Increase `predict_time` for longer lookahead
- Lower velocity resolution (`v_reso`) for smoother decisions

**Getting stuck near obstacles?**
- Increase `robot_radius` for more margin
- Decrease `obs_cost_gain` to explore riskier paths
- Ensure obstacle data is accurate

**High CPU usage?**
- Increase `v_reso` and `w_reso` (coarser sampling)
- Reduce `predict_time`
- Skip more trajectory points in `check_collision_strict()`

## License

MIT License - See LICENSE file for details

## References

- Fox, D., Burgard, W., & Thrun, S. (1997). "The Dynamic Window Approach to Collision Avoidance"
- ROS 2 Navigation Framework: https://docs.ros.org/en/humble/index.html