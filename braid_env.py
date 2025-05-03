from typing import Tuple, Dict, Any, Optional, Union

import logging

import numpy as np
import torch

import gymnasium as gym
from numpy import ndarray

from braid_relation import shift_left, shift_right, braid_relation1, braid_relation2
from markov_move import conjugation_markov_move
from random_knot import two_random_equivalent_knots
from reward_reshaper import RewardShaper
from smart_collapse import smart_collapse
from subsequence_similarity import subsequence_similarity


class BraidEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_braids_max=5, n_letters_max=5, max_steps=100,
                 max_steps_in_generation=30, device="mps",
                 render_mode=None, potential_based_reward=False):
        self.max_steps = max_steps
        self.max_steps_in_generation = max_steps_in_generation
        self.n_braids_max = n_braids_max  # in action space
        self.n_letters_max = n_letters_max  # Maximum length of a braid
        self.punishment_for_illegal_action = -400

        self.current_braid = torch.tensor([], dtype=torch.float)
        self.start_braid = torch.tensor([], dtype=torch.float)
        self.target_braid = torch.tensor([], dtype=torch.float)
        self.steps_taken = 0
        self.success = False
        self.done = False

        self.reward_shaper = None
        if potential_based_reward:
            self.reward_shaper = RewardShaper()

        self.observation_space = gym.spaces.Box(-(n_braids_max - 1), n_braids_max - 1, shape=(2 * n_letters_max,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(n_braids_max + 4)

        self.render_mode = render_mode

        self.device = device

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> tuple[
        ndarray, dict[Any, Any]]:
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

    def get_state(self) -> np.ndarray:
        # Combine current braid and target braid as state, each is padded to max length with zeros
        cur, tar = self.get_padded_braids()
        return torch.cat([cur, tar]).numpy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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

        previous_braid = self.current_braid.clone().to(device=self.current_braid.device)

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
                next_braid = self.current_braid
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
        self.done = self.steps_taken >= self.max_steps

        terminated = self.success  # Episode ends successfully when braids match
        truncated = self.done  # Episode is truncated when max steps reached

        info = {"is_success": self.success}

        # Calculate reward based on similarity to target
        if should_punish:
            # if the action was invalid, apply a punishment
            reward = self.punishment_for_illegal_action
        else:
            reward = self.calculate_reward(current_braid=previous_braid, next_braid=next_braid, target_braid=self.target_braid)

        # Add success reward
        if self.success:
            reward += 1000  # Large positive reward for success

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
        shaped_reward = self.reward_shaper.potential_based_reward(next_braid=next_braid,
                                                        current_braid=current_braid,
                                                        target_braid=target_braid)
        return shaped_reward