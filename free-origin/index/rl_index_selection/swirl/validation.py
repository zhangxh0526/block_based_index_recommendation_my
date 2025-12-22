import sys
import time

from pettingzoo import ParallelEnv
from stable_baselines3.common.env_util import make_vec_env

sys.path.append('..')
import copy
import importlib
import logging
import pickle
import sys
#import gym_db  # noqa: F401
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize, sync_envs_normalization
from sb3_contrib import MaskablePPO
from gym_db.common import EnvironmentType
from swirl.custom_callback import EvalCallbackWithTBRunningAverage
#from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from swirl.experiment import Experiment
from swirl.block_based_feature_extractor import CustomCombinedExtractor

if __name__ == "__main__":
    # 配置日志记录，设置日志级别为 INFO
    logging.basicConfig(level=logging.INFO)

    experiment_folder_path = "../experiment_results/validation"

    with open(f"{experiment_folder_path}/experiment_object.pickle", "rb") as handle:
        experiment = pickle.load(handle)
    experiment.experiment_folder_path = experiment_folder_path

    with open(f"{experiment_folder_path}/validation_workloads.pickle", "rb") as handle:
        wl_validation = pickle.load(handle)
    experiment.workload_generator.wl_validation = wl_validation

    # 设置实验使用的模型类型为 MaskablePPO
    experiment.model_type = MaskablePPO
    model = MaskablePPO.load(f"{experiment_folder_path}/final_model.zip")
    ParallelEnv = DummyVecEnv

    training_env = ParallelEnv(
        [experiment.make_env(env_id) for env_id in range(experiment.config["parallel_environments"])]
    )

    training_env = VecNormalize.load(f"{experiment_folder_path}/vec_normalize.pkl", training_env)

    # 要让模型使用加载的 VecNormalize 环境，需要重新设置环境
    model.set_env(training_env)

    experiment.set_model(model)

    if "slalom" in experiment.config["comparison_algorithms"]:
        experiment.config["comparison_algorithms"].remove("slalom")

    if "extend_partition" in experiment.config["comparison_algorithms"]:
        experiment.config["comparison_algorithms"].remove("extend_partition")

    experiment.compare()

    experiment.finish()


