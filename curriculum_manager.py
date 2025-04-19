import logging

import numpy as np
from typing import Dict, List, Optional, Tuple

import pfrl


class CurriculumManager:
    """Manages curriculum learning for braid manipulation tasks.

    Dynamically adjusts task difficulty based on agent performance.
    """

    def __init__(
            self,
            initial_braid_length: int = 10,
            max_braid_length: int = 40,
            initial_steps_in_generation: int = 10,
            max_steps_in_generation: int = 30,
            success_threshold: float = 0.25,
            evaluation_window: int = 100,
            increase_step_size: int = 2,
            min_evaluations_before_increase: int = 5
    ):
        """Initialize the curriculum manager.

        Args:
            initial_braid_length: Starting braid length
            max_braid_length: Maximum braid length to reach
            initial_steps_in_generation: Initial number of steps allowed in generation
            max_steps_in_generation: Maximum number of steps in generation
            success_threshold: Success rate required to increase difficulty
            evaluation_window: Number of episodes to consider for success rate calculation
            increase_step_size: How much to increase parameters when advancing curriculum
            min_evaluations_before_increase: Minimum evaluations before allowing difficulty increase
        """
        self.current_braid_length = initial_braid_length
        self.max_braid_length = max_braid_length
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

        # Save initial state to history
        self._record_curriculum_state()

    def _record_curriculum_state(self):
        """Record the current curriculum state to history."""
        self.curriculum_history.append({
            'braid_length': self.current_braid_length,
            'steps_in_generation': self.current_steps_in_generation,
            'success_rate': self.get_current_success_rate(),
            'evaluation_counter': self.evaluation_counter
        })

    def get_current_success_rate(self) -> float:
        """Calculate the current success rate based on recent history."""
        if not self.success_history:
            return 0.0

        # Consider only the most recent window of episodes
        recent_history = self.success_history[-self.evaluation_window:]
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
        if self.evaluation_counter >= self.min_evaluations_before_increase:
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
                return True

            # If we've maxed out steps but can still increase braid length
            elif self.current_braid_length < self.max_braid_length:
                # Increase braid length
                new_length = min(
                    self.current_braid_length + self.increase_step_size,
                    self.max_braid_length
                )
                self.current_braid_length = new_length
                self._record_curriculum_state()
                return True

        return False

    def get_current_parameters(self) -> Dict[str, int]:
        """Get the current curriculum parameters.

        Returns:
            Dict with current braid length and steps in generation
        """
        return {
            'current_braid_length': self.current_braid_length,
            'max_steps_in_generation': self.current_steps_in_generation
        }

    def reset_environment_for_curriculum(self, env) -> None:
        """Update environment parameters based on current curriculum stage.

        Args:
            env: The environment to update
        """
        env.n_braids_max = self.current_braid_length
        env.max_steps_in_generation = self.current_steps_in_generation

    def get_curriculum_progress(self) -> float:
        """Calculate overall curriculum progress as percentage.

        Returns:
            Float between 0.0 and 1.0 representing progress through curriculum
        """
        # Calculate progress as average of both dimensions
        braid_progress = (self.current_braid_length - 10) / (self.max_braid_length - 10)
        steps_progress = (self.current_steps_in_generation - 10) / (self.max_steps_in_generation - 10)

        # Clip between 0 and 1
        braid_progress = max(0.0, min(1.0, braid_progress))
        steps_progress = max(0.0, min(1.0, steps_progress))

        return (braid_progress + steps_progress) / 2.0


class EpisodeSuccessHook(pfrl.experiments.StepHook):
    """Hook to track episode success for curriculum learning."""

    def __init__(self, curriculum_manager, exp_buffer=None):
        self.curriculum_manager = curriculum_manager
        self.exp_buffer = exp_buffer
        self.current_episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        self.success_count = 0
        self.episode_count = 0

    def __call__(self, env, agent, step):
        # Get environment episode information
        info = env.get_episode_info() if hasattr(env, 'get_episode_info') else {}

        # Check if an episode just ended
        if hasattr(env, 'episode_ended') and env.episode_ended:
            self.episode_count += 1
            success = info.get('success', False)

            if success:
                self.success_count += 1

            # Record episode result for curriculum learning
            if self.curriculum_manager:
                difficulty_increased = self.curriculum_manager.record_episode_result(success)
                if difficulty_increased:
                    # Reset the environment with new parameters
                    self.curriculum_manager.reset_environment_for_curriculum(env)
                    logging.info(
                        f"Curriculum difficulty increased: braid_length={self.curriculum_manager.current_braid_length}, "
                        f"steps_in_generation={self.curriculum_manager.current_steps_in_generation}")

            # Store episode in experience buffer if available
            if self.exp_buffer and hasattr(self, 'current_episode_data'):
                # Finalize current episode data
                episode_data = {
                    'states': np.array(self.current_episode_data['states']),
                    'actions': np.array(self.current_episode_data['actions']),
                    'rewards': np.array(self.current_episode_data['rewards']),
                    'next_states': np.array(self.current_episode_data['next_states']),
                    'dones': np.array(self.current_episode_data['dones'])
                }

                # Add to experience buffer
                if len(episode_data['states']) > 0:
                    self.exp_buffer.add_episode(episode_data, success)

                # Reset for next episode
                self.current_episode_data = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'next_states': [],
                    'dones': []
                }

        # Record current step data for experience buffer
        if self.exp_buffer and hasattr(agent, 'last_state') and hasattr(agent, 'last_action'):
            if agent.last_state is not None and agent.last_action is not None:
                # Get current state, action, reward
                last_state = agent.last_state
                last_action = agent.last_action

                # Get reward and next state
                reward = env.last_reward if hasattr(env, 'last_reward') else 0
                next_state = env.last_obs if hasattr(env, 'last_obs') else last_state
                done = env.episode_ended if hasattr(env, 'episode_ended') else False

                # Add to current episode data
                self.current_episode_data['states'].append(last_state)
                self.current_episode_data['actions'].append(last_action)
                self.current_episode_data['rewards'].append(reward)
                self.current_episode_data['next_states'].append(next_state)
                self.current_episode_data['dones'].append(done)