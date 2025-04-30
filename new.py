import argparse
import os
import numpy as np
import torch
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, Any, Optional, Union

import gymnasium as gym
from gymnasium import spaces
import logging

import datetime
import pprint

from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.optim.lr_scheduler import LambdaLR

from tianshou.data import ReplayBuffer, Collector, VectorReplayBuffer
from tianshou.policy import TRPOPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.env import DummyVectorEnv

# Import BraidEnvironment components - you'll need to adjust these imports based on your project structure
from braid_relation import shift_left, shift_right, braid_relation1, braid_relation2
from markov_move import conjugation_markov_move
from random_knot import two_random_equivalent_knots
from reformer_networks import ReformerKnots
from reward_reshaper import RewardShaper
from smart_collapse import smart_collapse
from subsequence_similarity import subsequence_similarity

# Monkey patch to allow tensors in Tianshou
# This will modify Tianshou's behavior to support direct tensor operations
from tianshou.data.batch import _parse_value, Batch, _alloc_by_keys_diff, _create_value

# Store the original function to call it later
original_parse_value = _parse_value


def tensor_parse_value(v: Any) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(v, torch.Tensor):
        return v  # Return tensor directly without conversion
    return original_parse_value(v)  # Fallback to original for non-tensors


# Replace the function in tianshou.data.batch
import tianshou.data.batch

tianshou.data.batch._parse_value = tensor_parse_value


class BraidEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_braids_max=20, n_letters_max=40, max_steps=100,
                 max_steps_in_generation=30, potential_based_reward=False,
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
        # (Using float32 to match default PyTorch dtype)
        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(n_letters_max * 2,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_braids_max + 4)

        if should_randomize_cur_and_target:
            self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        torch.Tensor, Dict[str, Any]]:
        # Initialize random start and target braids
        super().reset(seed=seed)

        # Get tensors from the knot generator
        self.current_braid, self.target_braid = two_random_equivalent_knots(
            n_max=self.n_braids_max, n_max_second=self.n_letters_max, n_moves=self.max_steps_in_generation
        )

        self.steps_taken = 0
        self.success = False
        self.done = False

        state_tensor = self.get_state()

        if self.reward_shaper is not None:
            # Reset the reward shaper for a new episode
            self.reward_shaper.reset(self.max_steps)

        info = {}
        return state_tensor, info

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
        return torch.cat([cur, tar]).cpu().numpy()

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

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
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

        state_tensor = self.get_state()

        return state_tensor, reward, terminated, truncated, info

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
    parser.add_argument('--norm-adv', action='store_true', default=True,
                        help='Normalize advantage if true')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="Value function coefficient")
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help="Max gradient norm for clipping")
    # TRPO additional arguments
    parser.add_argument('--rew-norm', action='store_true', default=False,
                        help="Normalize reward if true")
    parser.add_argument('--bound-action-method', type=str, default="clip",
                        help="Method to bound actions: clip or tanh")
    parser.add_argument('--optim-critic-iters', type=int, default=5,
                        help="Number of iterations for optimizing critic")
    parser.add_argument('--max-kl', type=float, default=0.01,
                        help="Max KL divergence between old and new policy")
    parser.add_argument('--backtrack-coeff', type=float, default=0.8,
                        help="Backtracking coefficient for line search")
    parser.add_argument('--max-backtracks', type=int, default=10,
                        help="Maximum number of backtracks for line search")
    parser.add_argument('--lr-decay', action='store_true', default=False,
                        help="Decay learning rate linearly if true")
    # Buffer and training parameters
    parser.add_argument('--buffer-size', type=int, default=20000)
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
    parser.add_argument('--resume-path', type=str, default=None,
                        help='Path to resume training from a saved model')
    parser.add_argument('--resume-id', type=str, default=None,
                        help='ID to resume training')
    parser.add_argument('--logger', type=str, default='tensorboard',
                        choices=['tensorboard', 'wandb'])
    parser.add_argument('--wandb-project', type=str, default='braid-rl',
                        help='Wandb project name')
    parser.add_argument('--task', type=str, default='braid',
                        help='Task name for logging')
    parser.add_argument('--watch', action='store_true',
                        help='Only watch the trained agent without training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def make_env(n_braids_max, n_letters_max, max_steps, max_steps_in_generation,
             potential_based_reward, render_mode=None):
    """Function to create environment instances for vectorized environments"""
    return lambda: BraidEnvironment(
        n_braids_max=n_braids_max,
        n_letters_max=n_letters_max,
        max_steps=max_steps,
        max_steps_in_generation=max_steps_in_generation,
        potential_based_reward=potential_based_reward,
        render_mode=render_mode,
    )


def train_trpo(args: argparse.Namespace = get_args()) -> None:
    env = BraidEnvironment(
        n_braids_max=args.n_braids_max,
        n_letters_max=args.n_letters_max,
        max_steps=args.max_steps,
        max_steps_in_generation=args.max_steps_in_generation,
        potential_based_reward=args.potential_based_reward,
        render_mode="human" if args.render else None,
    )
    train_envs = DummyVectorEnv(
        [make_env(
            args.n_braids_max,
            args.n_letters_max,
            args.max_steps,
            args.max_steps_in_generation,
            args.potential_based_reward,
            None,  # Only render the first env if render is enabled
        ) for i in range(args.training_num)]
    )
    test_envs = DummyVectorEnv(
        [make_env(
            args.n_braids_max,
            args.n_letters_max,
            args.max_steps,
            args.max_steps_in_generation,
            args.potential_based_reward,
            None,  # Don't render test environments
        ) for _ in range(args.test_num)]
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    if args.use_reformer:
        net_a = ReformerKnots(
            dim=args.reformer_dim,
            depth=args.reformer_depth,
            max_seq_len=args.state_shape,
            heads=args.reformer_heads,
            bucket_size=min(64, args.state_shape // 2 if args.state_shape > 4 else args.state_shape),
            n_hashes=4,
            ff_chunks=10,
            lsh_dropout=0.1,
            causal=False,  # Non-causal for braid self-attention
            n_local_attn_heads=2,
            use_full_attn=(args.state_shape <= 64),  # Use full attention for small braids
            output_dim=args.hidden_sizes,
        )
        actor = Actor(
            net_a,
            action_shape=args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)
        critic = Critic(
            net_a,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        )
    else:
        actor = Net(
            args.state_shape,
            action_shape=args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device
        )
        net_a = Net(
            args.state_shape,
            hidden_sizes=args.hidden_sizes,
            activation=nn.Tanh,
            device=args.device,
        )
        critic = Critic(net_a, device=args.device).to(args.device)


    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    optim = torch.optim.Adam(critic.parameters(), lr=args.lr)
    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch

        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    policy: TRPOPolicy = TRPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        advantage_normalization=args.norm_adv,
        optim_critic_iters=args.optim_critic_iters,
        max_kl=args.max_kl,
        backtrack_coeff=args.backtrack_coeff,
        max_backtracks=args.max_backtracks,
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt["model"])
        train_envs.set_obs_rms(ckpt["obs_rms"])
        test_envs.set_obs_rms(ckpt["obs_rms"])
        print("Loaded agent from: ", args.resume_path)

    # collector
    buffer: Union[VectorReplayBuffer, ReplayBuffer]
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    # Create logger
    log_path = os.path.join(args.logdir, 'BraidEnv', 'trpo')
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        state = {"model": policy.state_dict(), "obs_rms": train_envs.get_obs_rms()}
        torch.save(state, os.path.join(log_path, "policy.pth"))

    if not args.watch:
        # trainer
        result = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            step_per_collect=args.step_per_collect,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False,
        ).run()
        pprint.pprint(result)

    # Let's watch its performance!
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(result)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    train_trpo()