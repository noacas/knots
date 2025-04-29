import gym
from gym import spaces
import numpy as np

from braid_env import BraidEnvironment


class CustomEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go to the right.
    """

    def __init__(self):
        super(CustomEnv, self).__init__()

        self._env = BraidEnvironment(
            n_braids_max=5,
            n_letters_max=10,
            max_steps=20,
            max_steps_in_generation=2,
            potential_based_reward=False,
        )

        # Define action and observation space
        # They must be gym.spaces objects

        self.action_space = self._env.get_action_space()

        # Example: Box observation space with 4 dimensions
        # Each dimension has values between -10 and 10
        self.observation_space = self._env.get_model_dim()

        # Initialize state
        self.state = None
        self.steps = 0
        self.max_steps = 200  # Maximum episode length

    def reset(self):
        """
        Reset the environment to an initial state and returns the initial observation.

        Returns:
            observation (object): the initial observation.
        """
        # Initialize state to random values within observation space
        #self.state = self.observation_space.sample()
        #self.steps = 0
        self.env.reset()
        return self.env.get_state()

    def step(self, action):
        """
        Take one step in the environment.

        Args:
            action (int): The action to take

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information
        """
        # Implement your environment dynamics here
        # This is a simple example where:
        # - Action 0: move left (decrease first state dimension)
        # - Action 1: stay
        # - Action 2: move right (increase first state dimension)

        return self._env.step(action)
        self.steps += 1

        if action == 0:
            self.state[0] -= 1
        elif action == 2:
            self.state[0] += 1

        # Clip state to be within bounds
        self.state = np.clip(self.state,
                             self.observation_space.low,
                             self.observation_space.high)

        # Calculate reward (example: reward increases as agent moves right)
        reward = self.state[0]  # Reward based on first dimension value

        # Check if episode is done
        done = False
        if self.state[0] >= 8:  # Success condition
            done = True
            reward += 10  # Bonus reward for success
        elif self.steps >= self.max_steps:  # Episode length limit
            done = True

        # Additional info
        info = {}

        return self.state, reward, done, info

    def render(self, mode='human'):
        """
        Render the environment to the screen.
        """
        if mode == 'human':
            print(f"Current state: {self.state}, Steps: {self.steps}")

    def close(self):
        """
        Clean up resources
        """
        pass