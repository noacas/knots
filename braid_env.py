from typing import Tuple

import torch

from braid_relation import shift_left, shift_right, braid_relation1, braid_relation2
from markov_move import conjugation_markov_move
from random_knot import two_random_equivalent_knots
from smart_collapse import smart_collapse


class BraidEnvironment:
    def __init__(self, n_braids_max=20, n_letters_max=40, max_steps=100,
                 normalize=True, temperature=1.0,
                 should_randomize_cur_and_target=True):
        self.max_steps = max_steps
        self.n_braids_max = n_braids_max  # in action space
        self.n_letters_max = n_letters_max  # Maximum length of a braid
        self.punishment_for_illegal_action = -4 * self.n_braids_max

        self.current_braid = torch.tensor([], dtype=torch.float16)
        self.target_braid = torch.tensor([], dtype=torch.float16)
        self.steps_taken = 0

        self.normalize = normalize
        self.temperature = max(0.1, temperature)

        if should_randomize_cur_and_target:
            self.reset()

    def reset(self) -> torch.Tensor:
        # Initialize random start and target braids
        self.target_braid, self.current_braid = two_random_equivalent_knots(
            n_max=self.n_braids_max, n_max_second=self.n_letters_max, n_moves=self.max_steps
        )
        # set up else
        self.steps_taken = 0
        state = self.get_state()
        return state

    def get_padded_braid(self, braid: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(braid, (0, self.n_letters_max - len(braid)))

    def get_padded_braids(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_padded_braid(self.current_braid), self.get_padded_braid(self.target_braid)

    def get_state(self) -> torch.Tensor:
        # Combine current braid and target braid as state, each is padded to max length with zeros
        cur, tar = self.get_padded_braids()
        return torch.cat([cur, tar])

    def get_env_from_state(self, env_state: torch.Tensor) -> "BraidEnvironment":
        new_env = BraidEnvironment(
            n_braids_max=self.n_braids_max,
            n_letters_max=self.n_letters_max,
            max_steps=self.max_steps,
            normalize=self.normalize,
            temperature=self.temperature,
            should_randomize_cur_and_target=False,
        )
        new_env.steps_taken = self.steps_taken

        def trim_zeros_tensor(tensor):
            nonzero_indices = torch.nonzero(tensor, as_tuple=True)[0]
            if nonzero_indices.numel() == 0:
                return torch.tensor([], dtype=tensor.dtype, device=tensor.device)  # Return empty tensor if all zeros
            return tensor[nonzero_indices[0]: nonzero_indices[-1] + 1]

        new_env.current_braid = trim_zeros_tensor(env_state[:self.n_letters_max])
        new_env.target_braid = trim_zeros_tensor(env_state[self.n_letters_max:])
        return new_env

    def get_model_dim(self) -> int:
        return self.n_letters_max  # max length of a braid

    def get_action_space(self) -> int:
        # Actions:
        # 0: SmartCollapse
        # 1-self.n_braids_max-1: Markov moves of type 1 (σᵢ)
        # self.n_braids_max: ShiftLeft
        # self.n_braids_max+1: ShiftRight
        # self.n_braids_max+2: BraidRelation1 and ShiftRight
        # self.n_braids_max+3: BraidRelation2 and ShiftRight
        return self.n_braids_max + 4

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        # Apply the selected Markov move to the current braid
        # Actions:
        # 0: SmartCollapse
        # 1-self.n_braids_max-1: Markov moves of type 1 (σᵢ)
        # self.n_braids_max: ShiftLeft
        # self.n_braids_max+1: ShiftRight
        # self.n_braids_max+2: BraidRelation1 and ShiftRight
        # self.n_braids_max+3: BraidRelation2 and ShiftRight
        self.steps_taken += 1

        if action == 0:
            self.current_braid = smart_collapse(self.current_braid)
        elif action < self.n_braids_max:
            # Apply positive crossing σᵢ
            if len(self.current_braid) < self.n_letters_max - 1:
                # this action is only valid if the braid will not surpass its maximum length after the move
                self.current_braid = conjugation_markov_move(self.current_braid, action, 0)
            else:
                # if the braid is at its maximum length, the action is invalid
                return self.get_state(), self.punishment_for_illegal_action, False, {}
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

        # Calculate reward based on similarity to target
        reward = self.calculate_reward()

        if len(self.current_braid) >= self.n_letters_max:
            # if the braid is at its maximum length, the episode is over
            pass

        # Check if done
        done = (torch.equal(self.current_braid, self.target_braid) or self.steps_taken >= self.max_steps)

        return self.get_state(), reward, done, {}

    def braid_word_difference(self) -> torch.Tensor:
        cur, tar = self.get_padded_braids()
        different_elements = (cur != tar).float()
        severity = (cur.abs() - tar.abs()).abs().float()
        same_abs = (cur.abs() == tar.abs()).float()
        diff_sign = (torch.sign(cur) != torch.sign(tar)).float()
        orientation_mismatch = same_abs * diff_sign * 0.5
        combined_diff = (different_elements * severity) + orientation_mismatch
        return combined_diff

    def calculate_reward(self) -> float:
        combined_diff = self.braid_word_difference()
        tempered_diff = combined_diff / self.temperature
        total_diff = torch.sum(tempered_diff).item()
        if self.normalize:
            total_diff = total_diff / self.n_letters_max
        return -total_diff