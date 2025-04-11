from typing import Tuple

import logging
import torch

from braid_relation import shift_left, shift_right, braid_relation1, braid_relation2
from markov_move import conjugation_markov_move
from random_knot import two_random_equivalent_knots
from smart_collapse import smart_collapse
from subsequence_similarity import subsequence_similarity


class BraidEnvironment:
    def __init__(self, n_braids_max=20, n_letters_max=40, max_steps=100,
                 max_steps_in_generation=30, normalize=True, temperature=1.0,
                 should_randomize_cur_and_target=True):
        self.max_steps = max_steps
        self.max_steps_in_generation = max_steps_in_generation
        self.n_braids_max = n_braids_max  # in action space
        self.n_letters_max = n_letters_max  # Maximum length of a braid
        self.punishment_for_illegal_action = -4000 * self.n_braids_max

        self.current_braid = torch.tensor([], dtype=torch.float)
        self.start_braid = torch.tensor([], dtype=torch.float)
        self.target_braid = torch.tensor([], dtype=torch.float)
        #self.chosen_moves = torch.tensor([], dtype=torch.float)
        #self.intermediate_braids = torch.zeros(self.max_steps, self.n_letters_max, dtype=torch.float)
        self.steps_taken = 0
        self.success = False

        #self.normalize = normalize
        #self.temperature = max(0.1, temperature)

        if should_randomize_cur_and_target:
            self.reset()

    def reset(self) -> torch.Tensor:
        # Initialize random start and target braids
        self.current_braid, self.target_braid = two_random_equivalent_knots(
            n_max=self.n_braids_max, n_max_second=self.n_letters_max, n_moves=self.max_steps_in_generation
        )
        self.start_braid = self.current_braid.clone()
        #self.chosen_moves = torch.tensor([], dtype=torch.float)
        #self.intermediate_braids = torch.zeros(self.max_steps, self.n_letters_max, dtype=torch.float)
        self.steps_taken = 0
        self.success = False
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
            max_steps_in_generation=self.max_steps_in_generation,
            #normalize=self.normalize,
            #temperature=self.temperature,
            should_randomize_cur_and_target=False,
        )

        def trim_zeros_tensor(tensor):
            nonzero_indices = torch.nonzero(tensor, as_tuple=True)[0]
            if nonzero_indices.numel() == 0:
                return torch.tensor([], dtype=tensor.dtype)  # Return empty tensor if all zeros # device=tensor.device?
            return tensor[nonzero_indices[0]: nonzero_indices[-1] + 1]

        new_env.current_braid = trim_zeros_tensor(env_state[:self.n_letters_max])
        new_env.target_braid = trim_zeros_tensor(env_state[self.n_letters_max:])
        new_env.start_braid = self.current_braid.clone()
        #new_env.chosen_moves = torch.tensor([], dtype=torch.float)
        #new_env.intermediate_braids = torch.zeros(self.max_steps, self.n_letters_max, dtype=torch.float)

        return new_env

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

        #self.chosen_moves = torch.cat([self.chosen_moves, torch.tensor([action], dtype=torch.float)])

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

        # Calculate reward based on similarity to target
        if should_punish:
            # if the action was invalid, apply a punishment
            reward = self.punishment_for_illegal_action
            similarity = -1
        else:
            reward, similarity = self.calculate_reward()

        #self.intermediate_braids[self.steps_taken-1] = self.get_padded_braid(self.current_braid)

        if len(self.current_braid) >= self.n_letters_max:
            # if the braid is at its maximum length, the episode is over
            pass

        # Check if done
        self.success = torch.equal(self.current_braid, self.target_braid)
        if self.success:
            logging.info("Found transformation! after %d steps", self.steps_taken)
        info = {"needs_reset": self.steps_taken >= self.max_steps}

        return self.get_state(), reward, self.success, info

    def braid_word_difference(self) -> torch.Tensor:

        cur, tar = self.get_padded_braids()
        different_elements = (cur != tar).float()
        severity = (cur.abs() - tar.abs()).abs().float()
        same_abs = (cur.abs() == tar.abs()).float()
        diff_sign = (torch.sign(cur) != torch.sign(tar)).float()
        orientation_mismatch = same_abs * diff_sign * 0.5
        combined_diff = (different_elements * severity) + orientation_mismatch
        return combined_diff

    def calculate_reward(self) -> Tuple[float, float]:
        # subsequence_similarity returns 0 if sequences are identical, 1 if no common subsequences
        similarity =  subsequence_similarity(self.current_braid, self.target_braid)
        return similarity * -1000 - self.steps_taken, similarity