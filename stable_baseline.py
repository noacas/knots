import logging

import gymnasium as gym
import numpy as np
from gym.wrappers import RecordEpisodeStatistics

from sb3_contrib import TRPO
from braid_env import BraidEnvironment
from reformer_networks import ReformerFeatureExtractor


if __name__ == '__main__':
    env = BraidEnvironment(
            n_braids_max=5,
            n_letters_max=10,
            max_steps=20,
            max_steps_in_generation=5,
            potential_based_reward=False,
            device="mps",
        )

    policy_kwargs = {
        "features_extractor_class": ReformerFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "depth": 6,
            "heads": 8,
        },
        "net_arch": dict(pi=[64, 32], vf=[64, 32]),
    }

    model = TRPO("MlpPolicy",
                 env,
                 verbose=1,
                 device="mps",
                 policy_kwargs=policy_kwargs,
                 )
    model.learn(total_timesteps=10_000, log_interval=4)

    # obs, _ = env.reset()
    # action, _states = model.predict(obs, deterministic=True)
    # obs, reward, terminated, truncated, info = env.step(action)
    # env.render()
    # if terminated or truncated:
    #   obs, _ = env.reset()