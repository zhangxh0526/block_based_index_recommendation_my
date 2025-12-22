from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy
import importlib
from pettingzoo.classic import rps_v2
import sys
sys.path.append("..")
import rl_index_selection.PettingZoo.custom_environment.env.index_environment as eie
import copy
import importlib
import logging
import pickle
import sys

import gym_db  # noqa: F401
from gym_db.common import EnvironmentType

from .experiment import Experiment

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # sys.path.append("/data2/fray/index/rl_index_selection")
    CONFIGURATION_FILE = "experiments/tpch.json"
    experiment = Experiment(CONFIGURATION_FILE)
    if experiment.config["rl_algorithm"]["stable_baselines_version"] == 2:
        from stable_baselines.common.callbacks import EvalCallbackWithTBRunningAverage
        from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

        algorithm_class = getattr(
            importlib.import_module("stable_baselines"), experiment.config["rl_algorithm"]["algorithm"]
        )
    elif experiment.config["rl_algorithm"]["stable_baselines_version"] == 3:
        from stable_baselines3.common.callbacks import EvalCallbackWithTBRunningAverage
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

        algorithm_class = getattr(
            importlib.import_module("stable_baselines3"), experiment.config["rl_algorithm"]["algorithm"]
        )
    else:
        raise ValueError

    experiment.prepare()

    ParallelEnv = SubprocVecEnv if experiment.config["parallel_environments"] > 1 else DummyVecEnv

    # training_env = ParallelEnv(
    #     [experiment.make_mult_env(env_id) for env_id in range(experiment.config["parallel_environments"])]
    # )
    # training_env = VecNormalize(
    #     training_env, norm_obs=True, norm_reward=True, gamma=experiment.config["rl_algorithm"]["gamma"], training=True
    # )
    env = experiment.make_mult_env(0)
    experiment.model_type = algorithm_class
    # Step 1: Load the PettingZoo environment
    env = eie.env(render_mode="human")

    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)

    # Step 3: Define policies for each agent
    policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], env)

    # Step 4: Convert the env to vector format
    env = DummyVectorEnv([lambda: env])

    # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    collector = Collector(policies, env)

    # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    result = collector.collect(n_episode=1, render=0.1)
