"""Training inspired by the training presented at 'Learning to Unknot'"""
import torch
from torch import nn

import argparse
import logging

import pfrl

from braid_env import BraidEnvironment

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU device ID. Set to -1 to use CPUs only."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--steps", type=int, default=5 * 10 ** 6, help="Total time steps for training."
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100000,
        help="Interval between evaluation phases in steps.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=100,
        help="Number of episodes ran in an evaluation phase",
    )
    parser.add_argument(
        "--render", action="store_true", default=False, help="Render the env"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run demo episodes, not training",
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help=(
            "Directory path to load a saved agent data from"
            " if it is a non-empty string."
        ),
    )
    parser.add_argument(
        "--trpo-update-interval",
        type=int,
        default=5000,
        help="Interval steps of TRPO iterations.",
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help=(
            "Monitor the env by gym.wrappers.Monitor."
            " Videos and additional log will be saved."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    return args


def main():
    args = parse_args()

    # Set random seed
    pfrl.utils.set_random_seed(args.seed)

    args.outdir = pfrl.experiments.prepare_output_dir(args, args.outdir)

    env = BraidEnvironment(n_braids_max=5, n_letters_max=10, max_steps=10, max_steps_in_generation=1)
    timestep_limit = env.max_steps
    obs_size = env.get_model_dim()
    action_size = env.get_action_space()
    print("Observation space:", obs_size)
    print("Action space:", action_size)

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = pfrl.nn.EmpiricalNormalization(
       obs_size, clip_threshold=5
    )

    policy = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_size),
        pfrl.policies.SoftmaxCategoricalHead(),
    )

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1e-2)

    # TRPO's policy is optimized via CG and line search, so it doesn't require
    # an Optimizer. Only the value function needs it.
    vf_opt = torch.optim.Adam(vf.parameters())

    # Hyperparameters in http://arxiv.org/abs/1709.06560
    agent = pfrl.agents.TRPO(
        policy=policy,
        vf=vf,
        vf_optimizer=vf_opt,
        obs_normalizer=obs_normalizer,
        gpu=args.gpu,
        update_interval=args.trpo_update_interval,
        max_kl=0.01,
        conjugate_gradient_max_iter=20,
        conjugate_gradient_damping=1e-1,
        gamma=0.995,
        lambd=0.97,
        vf_epochs=5,
        entropy_coef=0.01,
    )

    if args.load or args.load_pretrained:
        # either load or load_pretrained must be false
        assert not args.load or not args.load_pretrained
        if args.load:
            agent.load(args.load)
        else:
            agent.load(
                pfrl.utils.download_model(
                    "TRPO", args.env, model_type=args.pretrained_type
                )[0]
            )

    if args.demo:
        eval_stats = pfrl.experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
        import json
        import os

        with open(os.path.join(args.outdir, "demo_scores.json"), "w") as f:
            json.dump(eval_stats, f)
    else:
        pfrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            eval_env=env,
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            train_max_episode_len=timestep_limit,
        )


if __name__ == "__main__":
    main()