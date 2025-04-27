from typing import Any, Dict

import torch

from subsequence_similarity import subsequence_similarity


class RewardShaper:
    """
    Enhances rewards from the braid environment to provide better learning signals.
    """

    def __init__(
            self,
            discount_factor: float = 0.99,
            early_success_bonus: float = 2.0,
            progress_scale: float = 1.5,
            consistency_bonus: float = 0.1,
            step_penalty: float = -0.01
    ):
        """
        Initialize the reward shaper.

        Args:
            discount_factor: Discount factor for future rewards
            early_success_bonus: Bonus for succeeding early in the episode
            progress_scale: Scaling factor for progress-based rewards
            consistency_bonus: Bonus for consistent improvement
            step_penalty: Small penalty for each step to encourage efficiency
        """
        self.discount_factor = discount_factor
        self.early_success_bonus = early_success_bonus
        self.progress_scale = progress_scale
        self.consistency_bonus = consistency_bonus
        self.step_penalty = step_penalty

        # Tracking variables
        self.episode_step = 0
        self.previous_similarities = []
        self.best_similarity_so_far = 1.0  # Lower is better (0 = identical)
        self.max_steps = 100  # Default max steps, can be overridden in the environment

    def reset(self, max_steps: int = 100) -> None:
        """Reset tracking variables for a new episode."""
        self.episode_step = 0
        self.previous_similarities = []
        self.best_similarity_so_far = 1.0
        self.max_steps = max_steps

    def shape_reward(
            self,
            reward: float,
            current_braid: torch.Tensor,
            target_braid: torch.Tensor,
            done: bool,
            success: bool,
    ) -> float:
        """
        Shape the reward to provide better learning signals.

        Args:
            reward: Original reward from environment
            current_braid: Current state of the braid
            target_braid: Target state of the braid
            done: Whether the episode is done
            success: Whether the episode was successful
            max_steps: Additional information from the environment

        Returns:
            Shaped reward
        """
        self.episode_step += 1

        # Calculate similarity to track progress
        current_similarity = subsequence_similarity(current_braid, target_braid)
        self.previous_similarities.append(float(current_similarity))

        # Initial shaped reward is the original reward
        shaped_reward = reward

        # Add step penalty to encourage efficiency
        shaped_reward += self.step_penalty

        # Add progress-based reward component
        if len(self.previous_similarities) > 1:
            # Improved similarity compared to last step
            if current_similarity < self.previous_similarities[-2]:
                improvement = self.previous_similarities[-2] - current_similarity
                shaped_reward += improvement * self.progress_scale

            # Improved best similarity so far
            if current_similarity < self.best_similarity_so_far:
                best_improvement = self.best_similarity_so_far - current_similarity
                shaped_reward += best_improvement * 2.0  # Larger bonus for beating previous best
                self.best_similarity_so_far = float(current_similarity)

        # Add consistency bonus if there's been consistent improvement
        if len(self.previous_similarities) >= 3:
            if self.previous_similarities[-1] < self.previous_similarities[-2] < self.previous_similarities[-3]:
                shaped_reward += self.consistency_bonus

        # Success bonus that rewards early successes more
        if done and success:
            time_bonus = self.early_success_bonus * (1.0 - self.episode_step / self.max_steps)
            shaped_reward += max(1.0, time_bonus)  # Minimum bonus of 1.0

        return shaped_reward

    def get_potential(
            self,
            current_braid: torch.Tensor,
            target_braid: torch.Tensor
    ) -> float:
        """
        Calculate a potential function for potential-based shaping.
        Lower similarity means higher potential (0 = identical)

        Args:
            current_braid: Current state of the braid
            target_braid: Target state of the braid

        Returns:
            Potential value (higher is better)
        """
        # Calculate similarity (lower is better, 0 means identical)
        similarity = subsequence_similarity(current_braid, target_braid)

        # Convert to potential (higher is better)
        potential = 100.0 - (float(similarity) * 100.0)  # Scale to 0-100 range

        return potential

    def potential_based_reward(
            self,
            current_braid: torch.Tensor,
            next_braid: torch.Tensor,
            target_braid: torch.Tensor
    ) -> float:
        """
        Calculate potential-based shaped reward (Ng, Harada, Russell approach).
        This ensures that the shaped rewards don't change the optimal policy.

        Args:
            current_braid: Current state of the braid
            next_braid: Next state of the braid after action
            target_braid: Target state of the braid

        Returns:
            Potential-based shaped reward
        """
        current_potential = self.get_potential(current_braid, target_braid)
        next_potential = self.get_potential(next_braid, target_braid)

        # Potential-based shaping formula: γΦ(s') - Φ(s)
        shaped_reward = (self.discount_factor * next_potential) - current_potential

        return shaped_reward