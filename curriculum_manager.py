import csv
import logging
import os

import numpy as np
from typing import Dict, List, Optional, Tuple

from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

from braid_env import BraidEnvironment


class CurriculumManager:
    """Manages curriculum learning for braid manipulation tasks.

    Dynamically adjusts task difficulty based on agent performance.
    """

    def __init__(
            self,
            initial_steps_in_generation: int = 2,
            max_steps_in_generation: int = 10,
            success_threshold: float = 0.5,
            evaluation_window: int = 100,
            increase_step_size: int = 1,
            min_evaluations_before_increase: int = 50,
            save_dir: str = 'curriculum_data',
    ):
        """Initialize the curriculum manager.

        Args:
            initial_steps_in_generation: Initial number of steps allowed in generation
            max_steps_in_generation: Maximum number of steps in generation
            success_threshold: Success rate required to increase difficulty
            evaluation_window: Number of episodes to consider for success rate calculation
            increase_step_size: How much to increase parameters when advancing curriculum
            min_evaluations_before_increase: Minimum evaluations before allowing difficulty increase
        """
        self.current_steps_in_generation = initial_steps_in_generation
        self.max_steps_in_generation = max_steps_in_generation
        self.success_threshold = success_threshold
        self.evaluation_window = evaluation_window
        self.increase_step_size = increase_step_size
        self.min_evaluations_before_increase = min_evaluations_before_increase

        # Track success/failure history
        self.success_history: List[bool] = []
        self.curriculum_history: List[Dict] = []
        self.evaluation_counter = 0
        self.last_update = 0 # in which evaluation the curriculum was updated

        # Save curriculum state
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Save initial state to history
        self._record_curriculum_state()

    def _record_curriculum_state(self):
        """Record the current curriculum state to history."""
        self.curriculum_history.append({
            'steps_in_generation': self.current_steps_in_generation,
            'success_rate': self.get_current_success_rate(),
            'evaluation_counter': self.evaluation_counter
        })

    def get_current_success_rate(self) -> float:
        """Calculate the current success rate based on recent history."""
        if not self.success_history:
            return 0.0

        # Consider only the most recent window of episodes
        if self.evaluation_counter - self.last_update < self.evaluation_window:
            recent_history = self.success_history[self.last_update:]
        else:
            recent_history = self.success_history[-self.evaluation_window:]

        if len(recent_history) == 0:
            return 0.0

        success_rate = sum(recent_history) / len(recent_history)
        return success_rate

    def record_episode_result(self, success: bool) -> bool:
        """Record the result of an episode and check if curriculum should advance.

        Args:
            success: Whether the episode was successful

        Returns:
            bool: Whether the curriculum difficulty was increased
        """
        self.success_history.append(success)
        self.evaluation_counter += 1

        # Check if we should increase difficulty
        difficulty_increased = False
        if self.evaluation_counter - self.last_update >= self.min_evaluations_before_increase:
            difficulty_increased = self._maybe_increase_difficulty()

        return difficulty_increased

    def _maybe_increase_difficulty(self) -> bool:
        """Check success rate and increase difficulty if threshold met.

        Returns:
            bool: Whether difficulty was increased
        """
        current_success_rate = self.get_current_success_rate()

        # Only consider increasing difficulty if we have enough episodes
        if len(self.success_history) < self.evaluation_window:
            return False

        # Check if success rate meets threshold
        if current_success_rate >= self.success_threshold:
            # Determine what to increase first - prioritize steps in generation initially
            # then balance between both parameters

            # If we haven't reached max steps in generation, increase that first
            if self.current_steps_in_generation < self.max_steps_in_generation:
                # Increase steps in generation
                new_steps = min(
                    self.current_steps_in_generation + self.increase_step_size,
                    self.max_steps_in_generation
                )
                self.current_steps_in_generation = new_steps
                self._record_curriculum_state()
                self.last_update = self.evaluation_counter

                # Plot curriculum history
                self.plot_curriculum_history()

                return True

        return False

    def reset_environment_for_curriculum(self, env) -> None:
        """Update environment parameters based on current curriculum stage.

        Args:
            env: The environment to update
        """
        env.max_steps_in_generation = self.current_steps_in_generation

    def save_curriculum_data(self):
        """
        Save curriculum history data to a CSV file.
        """
        with open(os.path.join(self.save_dir, 'curriculum_data.csv'), 'w', newline='') as csvfile:
            fieldnames = ['evaluation_counter', 'steps_in_generation', 'success_rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for entry in self.curriculum_history:
                writer.writerow({
                    'evaluation_counter': entry['evaluation_counter'],
                    'steps_in_generation': entry['steps_in_generation'],
                    'success_rate': entry['success_rate']
                })

    def plot_curriculum_history(self):
        self.save_curriculum_data()
        # Extract data from curriculum history
        eval_counters = [entry['evaluation_counter'] for entry in self.curriculum_history]
        steps_in_generation = [entry['steps_in_generation'] for entry in self.curriculum_history]
        success_rates = [entry['success_rate'] for entry in self.curriculum_history]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot Steps in Generation
        ax1.plot(eval_counters, steps_in_generation, 'r-o', label='Steps in Generation')
        ax1.set_ylabel('Steps in Generation')
        ax1.set_title('Curriculum Learning Progress')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot Success Rate
        ax2.plot(eval_counters, success_rates, 'b-o', label='Success Rate')
        ax2.set_xlabel('Evaluation Counter')
        ax2.set_ylabel('Success Rate')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.legend()

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'curriculum_progress.png'), dpi=200)

        # Create combined figure
        plt.figure(figsize=(10, 6))
        plt.plot(eval_counters, steps_in_generation, 'r-', label='Steps in Generation')
        plt.plot(eval_counters, success_rates, 'b-', label='Success Rate')
        plt.xlabel('Evaluation Counter')
        plt.title('Curriculum Learning Progress (Combined)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add second y-axis
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot([], [], 'r-')  # Dummy plot for consistent legend
        ax2.set_ylabel('Steps in Generation')

        # Save combined figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'curriculum_progress_combined.png'), dpi=200)
        plt.close('all')


class EpisodeSuccessHook(BaseCallback):
    """Hook to track episode success for curriculum learning."""

    def __init__(self, curriculum_manager: CurriculumManager, env: BraidEnvironment, exp_buffer=None):
        super(EpisodeSuccessHook, self).__init__()
        self.curriculum_manager = curriculum_manager
        self.env = env

    def _on_step(self):
        # Check if an episode just ended
        if self.env.done or self.env.success:
            # Record episode result for curriculum learning
            if self.curriculum_manager:
                difficulty_increased = self.curriculum_manager.record_episode_result(self.env.success)
                if difficulty_increased:
                    # Reset the environment with new parameters
                    self.curriculum_manager.reset_environment_for_curriculum(self.env)
                    logging.info(
                        f"steps_in_generation={self.curriculum_manager.current_steps_in_generation}")

        return True

    def _on_training_end(self) -> None:
        self.curriculum_manager.plot_curriculum_history()