import torch
import argparse
import logging
import os.path as os_pth
import json

import pfrl

from braid_env import BraidEnvironment
from metrics import MetricsTracker, MetricsEvaluationHook, MetricsStepHook
from reformer_networks import create_reformer_policy, create_reformer_vf
from feed_forward_networks import create_ffn_policy, create_ffn_vf, initialize_ffn
from trpo import MyTRPO as TRPO

# Configure PyTorch for memory efficiency
torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 precision
torch.backends.cudnn.benchmark = True  # Use cuDNN autotuner
torch.cuda.empty_cache()  # Clear any existing cached memory

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU device ID, 0 for GPU, -1 to use CPUs only."
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
        "--demo",
        action="store_true",
        default=False,
        help="Run demo episodes, not training",
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
        "--current-braid-length",
        type=int,
        default=20,
        help="Current braid length.",
    )
    parser.add_argument(
        "--target-braid-length",
        type=int,
        default=40,
        help="Target braid length.",
    )
    parser.add_argument(
        "--max-steps-for-braid",
        type=int,
        default=100,
        help="Maximum steps for braid.",
    )
    parser.add_argument(
        "--max-steps-in-generation",
        type=int,
        default=20,
        help="Maximum steps in generation.",
    )
    parser.add_argument(
        "--use-reformer",
        action="store_true",
        default=True,
        help="Use Reformer networks instead of regular FFN",
    )
    parser.add_argument(
        "--reformer-depth",
        type=int,
        default=2,
        help="Depth of Reformer networks",
    )
    parser.add_argument(
        "--reformer-heads",
        type=int,
        default=4,
        help="Number of attention heads in Reformer",
    )
    parser.add_argument(
        "--reformer-dim",
        type=int,
        default=64,
        help="Hidden dimension for Reformer",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    return args


def run(seed=0, gpu=-1, outdir="results", steps=5 * 10 ** 6, eval_interval=100000,
                      eval_n_runs=100, demo=False, load="", load_pretrained=False,
                      trpo_update_interval=5000, log_level=logging.INFO,
                      current_braid_length=20, target_braid_length=40,
                      max_steps_for_braid=100, max_steps_in_generation=20,
                      use_reformer=True, reformer_depth=2, reformer_heads=4, reformer_dim=64
                      ):
    """Run the training or demo process.

    Args:
        seed (int): Random seed
        gpu (int): GPU device ID, 0 for GPU, -1 to use CPUs only
        outdir (str): Directory path to save output files
        steps (int): Total time steps for training
        eval_interval (int): Interval between evaluation phases in steps
        eval_n_runs (int): Number of episodes ran in an evaluation phase
        demo (bool): Run demo episodes, not training
        load (str): Directory path to load a saved agent data from
        load_pretrained (bool): Whether to load a pretrained model
        trpo_update_interval (int): Interval steps of TRPO iterations
        log_level (int): Level of the root logger
        current_braid_length (int): Current braid length
        target_braid_length (int): Target braid length
        max_steps_for_braid (int): Maximum steps for braid
        max_steps_in_generation (int): Maximum steps in generation
        use_reformer (bool): Whether to use Reformer networks
        reformer_depth (int): Depth of Reformer networks
        reformer_heads (int): Number of attention heads in Reformer
        reformer_dim (int): Hidden dimension for Reformer
    """
    # Create a dictionary of arguments for compatibility with existing code
    args = {
        "seed": seed,
        "gpu": gpu,
        "outdir": outdir,
        "steps": steps,
        "eval_interval": eval_interval,
        "eval_n_runs": eval_n_runs,
        "demo": demo,
        "load": load,
        "load_pretrained": load_pretrained,
        "trpo_update_interval": trpo_update_interval,
        "log_level": log_level,
        "current_braid_length": current_braid_length,
        "target_braid_length": target_braid_length,
        "max_steps_for_braid": max_steps_for_braid,
        "max_steps_in_generation": max_steps_in_generation,
        "use_reformer": use_reformer,
        "reformer_depth": reformer_depth,
        "reformer_heads": reformer_heads,
        "reformer_dim": reformer_dim,
    }

    assert args["current_braid_length"] <= args[
        "target_braid_length"], "Current braid length should be less than or equal to target braid length."

    # Set up logging
    logging.basicConfig(level=args["log_level"])

    pfrl.utils.set_random_seed(args["seed"])
    args["outdir"] = pfrl.experiments.prepare_output_dir(args, args["outdir"])
    metrics_tracker = MetricsTracker(os_pth.join(args["outdir"], 'metrics'))
    evaluation_hook = MetricsEvaluationHook(metrics_tracker, args["outdir"])
    step_hook = MetricsStepHook(metrics_tracker)

    # curriculum_manager = CurriculumManager(
    #     initial_braid_length=args["initial_braid_length"],
    #     max_braid_length=args["max_braid_length"],
    #     initial_steps_in_generation=args["initial_steps_in_generation"],
    #     max_steps_in_generation=args["max_steps_in_generation"],
    #     success_threshold=args["success_threshold"],
    # )
    # current_params = curriculum_manager.get_current_parameters()
    # current_braid_length = current_params["current_braid_length"]
    # current_steps_in_generation = current_params["max_steps_in_generation"]

    env = BraidEnvironment(
        n_braids_max=args["current_braid_length"],
        n_letters_max=args["target_braid_length"],
        max_steps=args["max_steps_for_braid"],
        max_steps_in_generation=args["max_steps_in_generation"],
    )
    timestep_limit = env.max_steps
    obs_size = env.get_model_dim()
    action_size = env.get_action_space()
    print("Observation space:", obs_size)
    print("Action space:", action_size)

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = pfrl.nn.EmpiricalNormalization(
        obs_size, clip_threshold=5
    )

    # Create networks based on configuration
    if args["use_reformer"]:
        print("Using Reformer networks")
        policy = create_reformer_policy(obs_size, action_size, reformer_depth, reformer_heads, reformer_dim)
        vf = create_reformer_vf(obs_size, reformer_depth, reformer_heads, reformer_dim)

        # # Initialize the networks
        # initialize_reformer_network(policy.policy_net)  # Initialize the Reformer part of the policy
        # initialize_reformer_network(vf)
    else:
        print("Using regular feed-forward networks")
        # Original FFN implementation
        policy = create_ffn_policy(obs_size, action_size)
        vf = create_ffn_vf(obs_size)
        initialize_ffn(policy, vf)

    # TRPO's policy is optimized via CG and line search, so it doesn't require
    # an Optimizer. Only the value function needs it.
    vf_opt = torch.optim.Adam(vf.parameters())

    # Hyperparameters in http://arxiv.org/abs/1709.06560
    agent = TRPO(
        policy=policy,
        vf=vf,
        vf_optimizer=vf_opt,
        obs_normalizer=obs_normalizer,
        gpu=args["gpu"],
        update_interval=args["trpo_update_interval"],
        max_kl=0.01,
        conjugate_gradient_max_iter=20,
        conjugate_gradient_damping=1e-1,
        gamma=0.95,
        lambd=0.97,
        vf_epochs=5,
        entropy_coef=0.01,
    )

    if args["load"] or args.get("load_pretrained", False):
        agent.load(args["load"])

    agent = torch.compile(agent, mode="max-autotune")

    if args["demo"]:
        eval_stats = pfrl.experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args["eval_n_runs"],
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args["eval_n_runs"],
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
        with open(os_pth.join(args["outdir"], "demo_scores.json"), "w") as f:
            json.dump(eval_stats, f)
    else:
        pfrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            eval_env=env,
            outdir=args["outdir"],
            steps=args["steps"],
            eval_n_steps=None,
            eval_n_episodes=args["eval_n_runs"],
            eval_interval=args["eval_interval"],
            train_max_episode_len=timestep_limit,
            step_hooks=[step_hook],
            evaluation_hooks=[evaluation_hook],
        )

    metrics_tracker.plot_learning_curves()


def main():
    """Parse command line arguments and run the code."""
    args = parse_args()
    # Convert the argparse Namespace to a dictionary
    args_dict = vars(args)
    # Call run with unpacked arguments
    run(**args_dict)


if __name__ == "__main__":
    main()