import logging

import gymnasium as gym
import numpy as np
from gym.wrappers import RecordEpisodeStatistics

from sb3_contrib import TRPO
from sb3_contrib.trpo import MultiInputPolicy

import torch

from typing import Tuple, Dict, Any, Optional, Union

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from braid_relation import shift_left, shift_right, braid_relation1, braid_relation2
from markov_move import conjugation_markov_move
from random_knot import two_random_equivalent_knots
from reformer_networks import ReformerKnots
from reward_reshaper import RewardShaper
from smart_collapse import smart_collapse
from subsequence_similarity import subsequence_similarity


class BraidEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_braids_max=20, n_letters_max=40, max_steps=100,
                 max_steps_in_generation=30, potential_based_reward=False,
                 should_randomize_cur_and_target=True, render_mode=None,
                 device="cpu"
                 ):
        self.max_steps = max_steps
        self.max_steps_in_generation = max_steps_in_generation
        self.n_braids_max = n_braids_max  # in action space
        self.n_letters_max = n_letters_max  # Maximum length of a braid
        self.punishment_for_illegal_action = -400
        self.render_mode = render_mode

        self.current_braid = torch.tensor([], dtype=torch.float)
        self.start_braid = torch.tensor([], dtype=torch.float)
        self.target_braid = torch.tensor([], dtype=torch.float)
        self.steps_taken = 0
        self.success = False
        self.done = False

        self.reward_shaper = None
        if potential_based_reward:
            self.reward_shaper = RewardShaper()

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(-(n_braids_max-1), n_braids_max-1, shape=(n_letters_max,), dtype=np.int32),
                "target": gym.spaces.Box(-(n_braids_max-1), n_braids_max-1, shape=(n_letters_max,), dtype=np.int32),
            }
        )
        self.action_space = gym.spaces.Discrete(n_braids_max + 4)

        self.device = device

        if should_randomize_cur_and_target:
            self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        Dict, Dict[str, Any]]:
        # Initialize random start and target braids
        super().reset(seed=seed)

        # Get tensors from the knot generator
        self.current_braid, self.target_braid = two_random_equivalent_knots(
            n_max=self.n_braids_max, n_max_second=self.n_letters_max, n_moves=self.max_steps_in_generation
        )

        self.steps_taken = 0
        self.success = False
        self.done = False

        state = self.get_state()

        if self.reward_shaper is not None:
            # Reset the reward shaper for a new episode
            self.reward_shaper.reset(self.max_steps)

        info = {}
        return state, info

    def render(self):
        if self.render_mode == "human":
            # Simple rendering - just print the current and target braids
            print(f"Current braid: {self.current_braid}")
            print(f"Target braid: {self.target_braid}")
            print(f"Steps taken: {self.steps_taken}")
        return None

    def close(self):
        # Nothing to clean up
        pass

    def get_padded_braid(self, braid: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(braid, (0, self.n_letters_max - len(braid)))

    def get_padded_braids(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_padded_braid(self.current_braid), self.get_padded_braid(self.target_braid)

    def get_state(self) -> Dict:
        # Combine current braid and target braid as state, each is padded to max length with zeros
        cur, tar = self.get_padded_braids()
        return {
                "agent": cur.numpy(),
                "target": tar.numpy(),
            }

    def get_model_dim(self) -> int:
        return self.n_letters_max * 2  # length of the state (2 padded braids)

    def get_action_space(self) -> int:
        # Actions:
        # 0: SmartCollapse
        # 1-self.n_braids_max-1: Markov moves of type 1 (σᵢ)
        # self.n_braids_max: ShiftLeft
        # self.n_braids_max+1: ShiftRight
        # self.n_braids_max+2: BraidRelation1 and ShiftRight
        # self.n_braids_max+3: BraidRelation2 and ShiftRight
        return self.n_braids_max + 4

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict[str, Any]]:
        # Apply the selected Markov move to the current braid
        # Actions:
        # 0: SmartCollapse
        # 1-self.n_braids_max-1: Markov moves of type 1 (σᵢ)
        # self.n_braids_max: ShiftLeft
        # self.n_braids_max+1: ShiftRight
        # self.n_braids_max+2: BraidRelation1 and ShiftRight
        # self.n_braids_max+3: BraidRelation2 and ShiftRight
        self.steps_taken += 1
        should_punish = False

        previous_braid = self.current_braid.clone()

        # Apply the selected action
        if action == 0:
            next_braid = smart_collapse(self.current_braid)
        elif action < self.n_braids_max:
            # Apply positive crossing σᵢ
            if len(self.current_braid) < self.n_letters_max - 1:
                # this action is only valid if the braid will not surpass its maximum length after the move
                next_braid = conjugation_markov_move(self.current_braid, action, 0)
            else:
                # if the braid is at its maximum length, the action is invalid
                should_punish = True
                next_braid = self.current_braid.clone()
        elif action == self.n_braids_max:
            next_braid = shift_left(self.current_braid)
        elif action == self.n_braids_max + 1:
            next_braid = shift_right(self.current_braid)
        elif action == self.n_braids_max + 2:
            # Apply BraidRelation1 and ShiftRight
            next_braid = braid_relation1(self.current_braid)
        elif action == self.n_braids_max + 3:
            # Apply BraidRelation2 and ShiftRight
            next_braid = braid_relation2(self.current_braid)
        else:
            raise ValueError(f"Invalid action: {action}")

        if len(next_braid) >= self.n_letters_max:
            # if the braid is at its maximum length, the episode is over
            pass

        # Check if done
        self.success = torch.equal(next_braid, self.target_braid)
        if self.success:
            logging.info("Found transformation! after %d steps", self.steps_taken)

        # In Gymnasium, we need to return both terminated and truncated
        terminated = self.success  # Episode ends successfully when braids match
        truncated = self.steps_taken >= self.max_steps  # Episode is truncated when max steps reached

        info = {"success": self.success}

        # Calculate reward based on similarity to target
        if should_punish:
            # if the action was invalid, apply a punishment
            reward = self.punishment_for_illegal_action
        else:
            reward = self.calculate_reward(
                current_braid=previous_braid,
                next_braid=next_braid,
                target_braid=self.target_braid
            )

        # Add success reward
        if self.success:
            reward += 1000  # Large positive reward for success

        # Update the current braid
        self.current_braid = next_braid

        # Render if needed
        if self.render_mode == "human":
            self.render()

        state = self.get_state()

        return state, reward, terminated, truncated, info

    def calculate_reward(self, current_braid, next_braid, target_braid) -> float:

        if self.reward_shaper is None:
            # if no reward shaper is used, the reward is based on the similarity to the target braid
            # would return 0 if the braids are equal
            # and -100 if they are completely different
            return subsequence_similarity(next_braid, target_braid) * -100.0
        shaped_reward = self.reward_shaper.potential_based_reward(
            next_braid=next_braid,
            current_braid=current_braid,
            target_braid=target_braid
        )
        return shaped_reward


class ReformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(ReformerFeatureExtractor, self).__init__(observation_space, features_dim)

        # Determine the input dimensions from the observation space
        agent_dim = observation_space["agent"].shape[0]
        target_dim = observation_space["target"].shape[0]

        # Initialize the ReformerKnots network
        self.reformer = ReformerKnots(
            dim=features_dim,
            depth=4,
            heads=4,
            output_dim=features_dim,
            max_seq_len=agent_dim+target_dim,
            bucket_size=min(64, agent_dim // 2 if agent_dim > 4 else agent_dim),
            lsh_dropout=0.1,
            causal=False
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Extract agent and target observations
        agent_obs = observations["agent"]
        target_obs = observations["target"]

        input_obs = torch.cat([agent_obs, target_obs], dim=1)
        return self.reformer(input_obs)


if __name__ == '__main__':
    env = BraidEnvironment(
            n_braids_max=5,
            n_letters_max=10,
            max_steps=20,
            max_steps_in_generation=5,
            potential_based_reward=False,
            device="mps",
        )

    env = RecordEpisodeStatistics(env)

    policy_kwargs = {
        "features_extractor_class": ReformerFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[64, 32], vf=[64, 32]),
    }

    model = TRPO(MultiInputPolicy, env, verbose=1, device="mps", policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=10_000, log_interval=4)

    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
          obs, _ = env.reset()