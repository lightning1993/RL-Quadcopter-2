import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 300
        self.action_high = 500
        self.action_size = 4

        # Goal
        if target_pos is None :
            print("Setting default init pose")
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
#         reward = -min(abs(self.target_pos[2] - self.sim.pose[2]), 20.0)  # reward = zero for matching target z, -ve as you go farther, upto -20
#         z_reward = np.tanh(1 - 0.003*(abs(self.sim.pose[2] - self.target_pos[2]))).sum()
#         xy_reward = np.tanh(1 - 0.009*(abs(self.sim.pose[:2] - self.target_pos[:2]))).sum()
#         reward = z_reward + xy_reward
        reward = 0.0
        sim = self.sim
        reward += 0.01*sim.pose[2]
        reward -= 0.02*(abs(sim.pose[:2])).sum()
        
        reward -= 0.025*(abs(sim.pose[5]))

        reward += 0.005*sim.v[2]
        reward -= 0.01*(abs(sim.v[:2])).sum()

        reward -= 0.025*(abs(self.sim.angular_v[:3])).sum()
        if self.sim.angular_v[2] == 0 or self.sim.pose[5] == 0:
            reward += 100

        return reward / 100

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state