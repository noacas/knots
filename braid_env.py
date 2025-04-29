from typing import Tuple

import logging
import torch

from braid_relation import shift_left, shift_right, braid_relation1, braid_relation2
from markov_move import conjugation_markov_move
from random_knot import two_random_equivalent_knots
from reward_reshaper import RewardShaper
from smart_collapse import smart_collapse
from subsequence_similarity import subsequence_similarity


class BraidEnvironment:
    def __init__(self, n_braids_max=20, n_letters_max=40, max_steps=100,
                 max_steps_in_generation=30, potential_based_reward=False,
                 should_randomize_cur_and_target=True):
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

        if should_randomize_cur_and_target:
            self.reset()

    def reset(self) -> torch.Tensor:
        # Initialize random start and target braids
        self.current_braid, self.target_braid = two_random_equivalent_knots(
            n_max=self.n_braids_max, n_max_second=self.n_letters_max, n_moves=self.max_steps_in_generation
        )
        self.start_braid = self.current_braid.clone()
        self.steps_taken = 0
        self.success = False
        self.done = False
        state = self.get_state()
        if self.reward_shaper is not None:
            # Reset the reward shaper for a new episode
            self.reward_shaper.reset(self.max_steps)
        return state

    def get_padded_braid(self, braid: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(braid, (0, self.n_letters_max - len(braid)))

    def get_padded_braids(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_padded_braid(self.current_braid), self.get_padded_braid(self.target_braid)

    def get_state(self) -> torch.Tensor:
        # Combine current braid and target braid as state, each is padded to max length with zeros
        cur, tar = self.get_padded_braids()
        return torch.cat([cur, tar])

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

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]: #, float]:
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
            self.current_braid = smart_collapse(self.current_braid)
        elif action < self.n_braids_max:
            # Apply positive crossing σᵢ
            if len(self.current_braid) < self.n_letters_max - 1:
                # this action is only valid if the braid will not surpass its maximum length after the move
                self.current_braid = conjugation_markov_move(self.current_braid, action, 0)
            else:
                # if the braid is at its maximum length, the action is invalid
                should_punish = True
        elif action == self.n_braids_max:
            self.current_braid = shift_left(self.current_braid)
        elif action == self.n_braids_max + 1:
            self.current_braid = shift_right(self.current_braid)
        elif action == self.n_braids_max + 2:
            # Apply BraidRelation1 and ShiftRight
            self.current_braid = braid_relation1(self.current_braid)
        elif action == self.n_braids_max + 3:
            # Apply BraidRelation2 and ShiftRight
            self.current_braid = braid_relation2(self.current_braid)
        else:
            raise ValueError(f"Invalid action: {action}")


        if len(self.current_braid) >= self.n_letters_max:
            # if the braid is at its maximum length, the episode is over
            pass

        # Check if done
        self.success = torch.equal(self.current_braid, self.target_braid)
        if self.success:
            logging.info("Found transformation! after %d steps", self.steps_taken)
        self.done = self.steps_taken >= self.max_steps
        info = {"needs_reset": self.steps_taken >= self.max_steps}

        # Calculate reward based on similarity to target
        if should_punish:
            # if the action was invalid, apply a punishment
            reward = self.punishment_for_illegal_action
        else:
            reward = self.calculate_reward(current_braid=previous_braid, next_braid=self.current_braid, target_braid=self.target_braid)

        return self.get_state(), reward, self.success, info

    def calculate_reward(self, current_braid, next_braid, target_braid) -> float:
        if self.reward_shaper is None:
            return subsequence_similarity(next_braid, target_braid) * 100.0
        shaped_reward = self.reward_shaper.potential_based_reward(next_braid=next_braid,
                                                        current_braid=current_braid,
                                                        target_braid=target_braid)
        return shaped_reward