# maze_env.py
import gym
import numpy as np
from gym import spaces
from controller import Supervisor, Robot, Motor
import time

TIME_STEP = 16
GOAL_THRESHOLD = 1.0  # Threshold for goal proximity
MAX_SPEED = 10
ACTION_DURATION = 0.4  # Duration of each action in seconds

class RobotMazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, maze):
        super(RobotMazeEnv, self).__init__()
        self.supervisor = Supervisor()

        # Initialize motors
        self.setup_devices()
        self.leftMotor = self.supervisor.getDevice('motor_1')
        self.rightMotor = self.supervisor.getDevice('motor_2')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)

        self.leftSpeed = 1.0 * MAX_SPEED
        self.rightSpeed = 1.0 * MAX_SPEED

        self.maze = maze
        self.agent_position = np.array([0.0, 0.0, 0.0])
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # Action space: 4 actions (forward, left, right, brake)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: GPS (x,y) + compass (yaw)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(3,),  # [x, y, yaw]
            dtype=np.float32
        )
        
        # Goal position
        self.goal_position = np.array([-5.0, 10.0, 0.0])
        self.GOAL_THRESHOLD = GOAL_THRESHOLD
        
        # Initialize Webots robot and supervisor
        self.robot_node = self.supervisor.getFromDef("RbtWhse")
        self.robot_name = self.robot_node.getField('name').getSFString()
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        # Store initial positions
        self.original_position = np.array(self.translation_field.getSFVec3f())
        self.original_rotation = np.array(self.rotation_field.getSFRotation())
        self.wheel1_node = self.supervisor.getFromDef("wheel1_")
        self.wheel4_node = self.supervisor.getFromDef("wheel4_")
        self.original_wheel1_position = np.array(self.wheel1_node.getField("translation").getSFVec3f())
        self.original_wheel4_position = np.array(self.wheel4_node.getField("translation").getSFVec3f())
        
        # Store original rotations for wheels
        self.original_wheel1_rotation = self.wheel1_node.getField("rotation").getSFRotation()
        self.original_wheel4_rotation = self.wheel4_node.getField("rotation").getSFRotation()

    def setup_devices(self):
        # imu for compass
        self.imu = self.supervisor.getDevice('imu')
        self.imu.enable(TIME_STEP)

        # gps
        self.gps = self.supervisor.getDevice('gps')
        self.gps.enable(TIME_STEP)
        self.supervisor.step(TIME_STEP)  # For GPS to take first sample

    def reset(self):
        # Reset speeds
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)

        # Reset positions
        self.translation_field.setSFVec3f(self.original_position.tolist())
        self.rotation_field.setSFRotation(self.original_rotation.tolist())
        self.wheel1_node.getField("translation").setSFVec3f(self.original_wheel1_position.tolist())
        self.wheel4_node.getField("translation").setSFVec3f(self.original_wheel4_position.tolist())
        self.wheel1_node.getField("rotation").setSFRotation(self.original_wheel1_rotation)
        self.wheel4_node.getField("rotation").setSFRotation(self.original_wheel4_rotation)

        # Reset properties
        self.agent_position = self.original_position.copy()
        self.leftSpeed = 0
        self.rightSpeed = 0

        # step supervisor for GPS to update
        self.supervisor.step(TIME_STEP) 
        
        # Return initial state (GPS + compass)
        return self._get_state()

    def step(self, action):
        # Apply the action
        self._apply_action(action)
        
        # Get new state
        state = self._get_state()
        self.agent_position = np.array(self.gps.getValues())
        
        # Calculate distance and reward
        distance = self._get_distance_to_target(self.goal_position)
        reward = self._calculate_reward(distance)
        done = distance < GOAL_THRESHOLD
        
        info = {
            "distance": distance,
            "position": self.agent_position
        }
        
        return state, reward, done, info

    def _get_state(self):
        # Get compass (yaw only) and GPS
        yaw = self.imu.getRollPitchYaw()[2]
        gps_coords = np.array(self.gps.getValues())[:2]  # Only x,y coordinates
        return np.concatenate([gps_coords, [yaw]])

    def _get_distance_to_target(self, goal_position):
        return np.linalg.norm(self.agent_position[:2] - goal_position[:2])
    
    def _calculate_reward(self, distance):
        if distance < GOAL_THRESHOLD:
            print(f"{self.robot_name} : Goal reached!")
            return 10
        return -distance

    def _apply_action(self, action):
        if action == 0:  # Forward
            self.rightSpeed = MAX_SPEED
            self.leftSpeed = MAX_SPEED
        elif action == 1:  # Left
            self.leftSpeed = 0
            self.rightSpeed = MAX_SPEED
        elif action == 2:  # Right
            self.leftSpeed = MAX_SPEED
            self.rightSpeed = 0
        elif action == 3:  # Brake
            self.leftSpeed = 0
            self.rightSpeed = 0

        # Apply action for duration
        start_time = self.supervisor.getTime()
        while self.supervisor.getTime() - start_time < ACTION_DURATION:
            self.leftMotor.setVelocity(self.leftSpeed)
            self.rightMotor.setVelocity(self.rightSpeed)
            self.supervisor.step(TIME_STEP)
