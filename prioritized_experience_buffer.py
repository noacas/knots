import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
import torch


class PrioritizedExperienceBuffer:
    """Experience replay buffer with prioritization of successful episodes.

    Stores episodes and allows sampling with higher probability for successful ones.
    Can be used alongside TRPO even though TRPO is normally on-policy.
    """

    def __init__(
            self,
            capacity: int = 100,
            success_priority_factor: float = 3.0,
            success_retention_ratio: float = 0.3,
    ):
        """Initialize the experience buffer.

        Args:
            capacity: Maximum number of episodes to store
            success_priority_factor: How much to prioritize successful episodes (multiplier)
            success_retention_ratio: Minimum ratio of successful episodes to maintain
        """
        self.capacity = capacity
        self.success_priority_factor = success_priority_factor
        self.success_retention_ratio = success_retention_ratio

        # Separate buffers for successful and unsuccessful episodes
        self.successful_episodes = deque(maxlen=int(capacity * success_retention_ratio))
        self.unsuccessful_episodes = deque(maxlen=capacity - len(self.successful_episodes))

        # Tracking stats
        self.total_episodes_seen = 0
        self.total_successful_episodes = 0

    def add_episode(self, episode_data: Dict[str, Any], success: bool) -> None:
        """Add an episode to the buffer.

        Args:
            episode_data: Dictionary containing episode data (states, actions, rewards, etc.)
            success: Whether the episode was successful
        """
        self.total_episodes_seen += 1

        if success:
            self.total_successful_episodes += 1
            self.successful_episodes.append(episode_data)
        else:
            self.unsuccessful_episodes.append(episode_data)

    def sample_episode(self) -> Optional[Dict[str, Any]]:
        """Sample an episode from the buffer with prioritization.

        Returns:
            A sampled episode or None if the buffer is empty
        """
        if not self.successful_episodes and not self.unsuccessful_episodes:
            return None

        # Calculate probabilities for sampling from each buffer
        p_successful = self.success_priority_factor / (self.success_priority_factor + 1)
        p_unsuccessful = 1.0 - p_successful

        # Adjust if one buffer is empty
        if not self.successful_episodes:
            p_successful = 0.0
            p_unsuccessful = 1.0
        elif not self.unsuccessful_episodes:
            p_successful = 1.0
            p_unsuccessful = 0.0

        # Sample which buffer to use
        use_successful = random.random() < p_successful

        # Sample from the chosen buffer
        if use_successful and self.successful_episodes:
            return random.choice(self.successful_episodes)
        elif self.unsuccessful_episodes:
            return random.choice(self.unsuccessful_episodes)
        else:
            return None

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of episodes.

        Args:
            batch_size: Number of episodes to sample

        Returns:
            List of sampled episodes
        """
        batch = []
        for _ in range(batch_size):
            episode = self.sample_episode()
            if episode is not None:
                batch.append(episode)

        return batch

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the experience buffer.

        Returns:
            Dictionary with buffer statistics
        """
        return {
            'total_episodes_seen': self.total_episodes_seen,
            'total_successful_episodes': self.total_successful_episodes,
            'successful_episodes_in_buffer': len(self.successful_episodes),
            'unsuccessful_episodes_in_buffer': len(self.unsuccessful_episodes),
            'buffer_utilization': (len(self.successful_episodes) + len(self.unsuccessful_episodes)) / self.capacity
        }

    def __len__(self) -> int:
        """Get the current size of the buffer.

        Returns:
            Total number of episodes currently stored
        """
        return len(self.successful_episodes) + len(self.unsuccessful_episodes)


class ExperienceReplayTRPO:
    """Adapter to integrate experience replay with TRPO algorithm.

    TRPO is normally on-policy, but this class allows limited off-policy learning
    by occasionally feeding in experiences from successful episodes.
    """

    def __init__(
            self,
            agent,
            experience_buffer: PrioritizedExperienceBuffer,
            replay_probability: float = 0.2,
            replay_batch_size: int = 4
    ):
        """Initialize the experience replay adapter.

        Args:
            agent: The TRPO agent to adapt
            experience_buffer: Buffer containing stored episodes
            replay_probability: Probability of doing experience replay on each step
            replay_batch_size: Number of episodes to replay when triggered
        """
        self.agent = agent
        self.experience_buffer = experience_buffer
        self.replay_probability = replay_probability
        self.replay_batch_size = replay_batch_size

    def step(self, state, action, reward, next_state, done, reset_if_done=True):
        """Step the agent and possibly perform experience replay.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            reset_if_done: Whether to reset on done

        Returns:
            Same as agent.step()
        """
        # Normal agent step
        result = self.agent.step(state, action, reward, next_state, done, reset_if_done)

        # Occasionally perform experience replay
        if random.random() < self.replay_probability:
            self._perform_experience_replay()

        return result

    def _perform_experience_replay(self):
        """Sample episodes from buffer and have agent learn from them."""
        episodes = self.experience_buffer.sample_batch(self.replay_batch_size)

        # Skip if no episodes were sampled
        if not episodes:
            return

        # For each episode, have the agent process the transitions
        for episode in episodes:
            states = episode['states']
            actions = episode['actions']
            rewards = episode['rewards']
            next_states = episode['next_states']
            dones = episode['dones']

            # Since TRPO is on-policy, we don't want to update its policy directly from
            # these off-policy samples. Instead, we'll use them just to update the value function.
            for i in range(len(states) - 1):
                # Update only the value function, not the policy
                if hasattr(self.agent, 'update_value_function'):
                    self.agent.update_value_function(
                        states[i], actions[i], rewards[i], next_states[i], dones[i]
                    )
                # If no dedicated method exists, we'll use a custom implementation
                else:
                    # This is a simplified value function update - you may need to modify
                    # this based on your TRPO implementation's internals
                    if hasattr(self.agent, 'vf') and hasattr(self.agent, 'vf_optimizer'):
                        state_value = self.agent.vf(torch.from_numpy(states[i]).float())
                        next_state_value = self.agent.vf(torch.from_numpy(next_states[i]).float())
                        target_value = rewards[i] + (1 - dones[i]) * self.agent.gamma * next_state_value

                        value_loss = ((state_value - target_value) ** 2).mean()

                        self.agent.vf_optimizer.zero_grad()
                        value_loss.backward()
                        self.agent.vf_optimizer.step()