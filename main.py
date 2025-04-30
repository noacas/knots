import torch
import argparse
import logging
import os.path as os_pth
import json
from curriculum_manager import CurriculumManager, EpisodeSuccessHook
from metrics import MetricsTracker, MetricsEvaluationHook, MetricsStepHook


from tqdm.rich import tqdm

tqdm.pandas()

from sb3_contrib import TRPO

from braid_env import BraidEnvironment
from reformer_networks import ReformerFeatureExtractor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="mps", help="mps, cuda, or cpu"
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
        default=10000,
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
        default=5,
        help="Current braid length.",
    )
    parser.add_argument(
        "--target-braid-length",
        type=int,
        default=10,
        help="Target braid length.",
    )
    parser.add_argument(
        "--max-steps-for-braid",
        type=int,
        default=30,
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
        default=False,
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
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        default=False,
        help="Use curriculum learning",
    )
    parser.add_argument(
        "--initial-steps-in-generation",
        type=int,
        default=2,
        help="Initial steps in generation",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.5,
        help="Success threshold for curriculum learning",
    )
    parser.add_argument(
        "--potential-based-reward",
        action="store_true",
        default=False,
        help="Use potential-based reward shaping",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    return args


def run(seed=0, device="mps", outdir="results", steps=5 * 10 ** 6, eval_interval=10000,
        eval_n_runs=100, demo=False, load="", load_pretrained=False,
        trpo_update_interval=5000, log_level=logging.INFO,
        current_braid_length=20, target_braid_length=40,
        max_steps_for_braid=100, max_steps_in_generation=20,
        use_reformer=True, reformer_depth=2, reformer_heads=4, reformer_dim=64,
        use_curriculum=True, initial_steps_in_generation=2, success_threshold=0.5,
        potential_based_reward=False
        ):
    """Run the training or demo process.

    Args:
        seed (int): Random seed
        device (str): The device to use (mps, cuda, or cpu)
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
        use_curriculum (bool): Whether to use curriculum learning
        initial_steps_in_generation (int): Initial steps in generation
        success_threshold (float): Success threshold for curriculum learning
        potential_based_reward (bool): Whether to use potential-based reward shaping
    """
    # Create a dictionary of arguments for compatibility with existing code
    args = {
        "seed": seed,
        "device": device,
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
        "use_curriculum": use_curriculum,
        "initial_steps_in_generation": initial_steps_in_generation,
        "success_threshold": success_threshold,
        "potential_based_reward": potential_based_reward,
    }

    assert args["current_braid_length"] <= args[
        "target_braid_length"], "Current braid length should be less than or equal to target braid length."

    # Set up logging
    logging.basicConfig(level=args["log_level"])

    # args["outdir"] = pfrl.experiments.prepare_output_dir(args, args["outdir"])
    # metrics_tracker = MetricsTracker(os_pth.join(args["outdir"], 'metrics'))
    # evaluation_hook = MetricsEvaluationHook(metrics_tracker, args["outdir"])
    # step_hook = MetricsStepHook(metrics_tracker)
    # step_hooks = [step_hook]

    curriculum_manager = None
    max_steps_in_generation = args["max_steps_in_generation"]
    if args["use_curriculum"]:
        curriculum_manager = CurriculumManager(
            initial_steps_in_generation=args["initial_steps_in_generation"],
            max_steps_in_generation=args["max_steps_in_generation"],
            success_threshold=args["success_threshold"],
            save_dir=args["outdir"],
        )
        max_steps_in_generation = args["initial_steps_in_generation"]
        #step_hooks.append(EpisodeSuccessHook(curriculum_manager))

    env = BraidEnvironment(
        n_braids_max=args["current_braid_length"],
        n_letters_max=args["target_braid_length"],
        max_steps=args["max_steps_for_braid"],
        max_steps_in_generation=max_steps_in_generation,
        potential_based_reward=args["potential_based_reward"],
        device=args["device"],
    )
    timestep_limit = env.max_steps

    policy_kwargs = None
    if args["use_reformer"]:
        policy_kwargs = {
            "features_extractor_class": ReformerFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "depth": args["reformer_depth"],
                "heads": args["reformer_heads"],
            },
            "net_arch": dict(pi=[64, 32], vf=[64, 32]),
        }

    model = TRPO("MlpPolicy",
                 env,
                 verbose=1,
                 device="mps",
                 policy_kwargs=policy_kwargs,
                 learning_rate=1e-3,
                 stats_window_size=100,
                 tensorboard_log=os_pth.join(args["outdir"], "tensorboard"),
                 gae_lambda=0.97,
                 gamma=0.95,
                 n_steps=args["steps"]
                 )
    model.learn(total_timesteps=10_000, log_interval=4, progress_bar=True)
    model.save(outdir)

    # metrics_tracker.plot_learning_curves()
    # if curriculum_manager is not None:
    #     curriculum_manager.plot_curriculum_history()


def main():
    """Parse command line arguments and run the code."""
    args = parse_args()
    # Convert the argparse Namespace to a dictionary
    args_dict = vars(args)
    # Call run with unpacked arguments
    run(**args_dict)


if __name__ == "__main__":
    main()