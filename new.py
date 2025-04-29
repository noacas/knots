import argparse
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, Any, Optional

import gymnasium as gym
from gymnasium import spaces
import logging

# Import Tianshou components
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import TRPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic

# Import BraidEnvironment components - you'll need to adjust these imports based on your project structure
from braid_relation import shift_left, shift_right, braid_relation1, braid_relation2
from markov_move import conjugation_markov_move
from random_knot import two_random_equivalent_knots
from reward_reshaper import RewardShaper
from smart_collapse import smart_collapse
from subsequence_similarity import subsequence_similarity


class BraidEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_braids_max=20, n_letters_max=40, max_steps=100,
                 max_steps_in_generation=30, potential_based_reward=False,
                 device="cpu",
                 should_randomize_cur_and_target=True, render_mode="human"):
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

        # Define observation and action spaces for Gymnasium compatibility
        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(n_letters_max * 2,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_braids_max + 4)

        self.device = device

        if should_randomize_cur_and_target:
            self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        # Initialize random start and target braids
        super().reset(seed=seed)

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

        # Make sure we return a numpy array, not a torch tensor
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

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

    def get_state(self) -> torch.Tensor:
        # Combine current braid and target braid as state, each is padded to max length with zeros
        cur, tar = self.get_padded_braids()
        return torch.cat([cur, tar]).to(self.device)

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

        previous_braid = self.current_braid.clone()
        # Ensure the device is CPU
        if previous_braid.device.type != 'cpu':
            previous_braid = previous_braid.cpu()

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
            reward = self.calculate_reward(current_braid=previous_braid, next_braid=next_braid,
                                           target_braid=self.target_braid)

        # Add success reward
        if self.success:
            reward += 1000  # Large positive reward for success

        self.current_braid = next_braid

        # Render if needed
        if self.render_mode == "human":
            self.render()

        # Make sure we return a numpy array, not a torch tensor
        state = self.get_state()
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

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


def get_args():
    parser = argparse.ArgumentParser()
    # Environment settings
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--n-braids-max', type=int, default=20)
    parser.add_argument('--n-letters-max', type=int, default=40)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--max-steps-in-generation', type=int, default=30)
    parser.add_argument('--potential-based-reward', action='store_true')
    # TRPO specific arguments
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--advantage-normalization', action='store_true', default=True,
                        help='Normalize advantage if true')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="Value function coefficient")
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help="Max gradient norm for clipping")
    # Training parameters
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=30000)
    parser.add_argument('--step-per-collect', type=int, default=2000)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    # Logger related
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--render', action='store_true', help='Render during training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def make_env(n_braids_max, n_letters_max, max_steps, max_steps_in_generation,
             potential_based_reward, device, render_mode=None):
    """Function to create environment instances for vectorized environments"""
    return lambda: BraidEnvironment(
        n_braids_max=n_braids_max,
        n_letters_max=n_letters_max,
        max_steps=max_steps,
        max_steps_in_generation=max_steps_in_generation,
        potential_based_reward=potential_based_reward,
        device=device,
        render_mode=render_mode
    )


def train_trpo(args=get_args()):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print(f"Using device: {args.device}")

    # Create a sample environment to get dimensions
    env = BraidEnvironment(
        n_braids_max=args.n_braids_max,
        n_letters_max=args.n_letters_max,
        max_steps=args.max_steps,
        max_steps_in_generation=args.max_steps_in_generation,
        potential_based_reward=args.potential_based_reward,
        device=args.device,
        render_mode="human" if args.render else None
    )

    # Seed everything for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create training and test environments
    render_mode_train = "human" if args.render else None
    train_envs = DummyVectorEnv(
        [make_env(
            args.n_braids_max,
            args.n_letters_max,
            args.max_steps,
            args.max_steps_in_generation,
            args.potential_based_reward,
            args.device,
            render_mode_train if i == 0 else None  # Only render the first env if render is enabled
        ) for i in range(args.training_num)]
    )
    test_envs = DummyVectorEnv(
        [make_env(
            args.n_braids_max,
            args.n_letters_max,
            args.max_steps,
            args.max_steps_in_generation,
            args.potential_based_reward,
            args.device,
            None  # Don't render test environments
        ) for _ in range(args.test_num)]
    )

    # Seed training and test environments
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Get state and action information from environment
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    # Build networks for discrete action space - updated for newer Tianshou API
    net = Net(
        state_shape=state_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device
    )
    actor = Actor(
        preprocess_net=net,
        action_shape=action_shape,
        hidden_sizes=[],  # No additional hidden layers after net
        device=args.device
    )
    critic = Critic(
        preprocess_net=net,
        hidden_sizes=[],  # No additional hidden layers after net
        device=args.device
    )

    # For discrete action space in TRPO
    dist_fn = torch.distributions.Categorical

    # Create TRPO policy - updated for latest API
    policy = TRPOPolicy(
        actor=actor,
        critic=critic,
        optim=torch.optim.Adam(critic.parameters(), lr=args.lr),
        dist_fn=dist_fn,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        reward_normalization=True,
        advantage_normalization=args.advantage_normalization,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm
    )

    # Create collectors for training and testing
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.step_per_collect, args.training_num)
    )
    test_collector = Collector(policy, test_envs)

    # Create logger
    log_path = os.path.join(args.logdir, 'BraidEnv', 'trpo')
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # Training
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        save_best_fn=lambda policy: torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth')),
        save_checkpoint_fn=lambda epoch, env_step, gradient_step: torch.save({
            'epoch': epoch + 1,
            'env_step': env_step,
            'gradient_step': gradient_step,
            'policy': policy.state_dict(),
        }, os.path.join(log_path, f'checkpoint_{epoch}.pth')),
        logger=logger,
    )

    # Clean up
    train_envs.close()
    test_envs.close()
    env.close()

    return result, policy


if __name__ == '__main__':
    # Run training
    result, policy = train_trpo()
    print(f'Finished training! Use policy.forward() to utilize the trained policy.')