import sys
import time
import argparse

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

    # 允许通过命令行指定配置文件路径，未指定则用默认
    parser = argparse.ArgumentParser(description="Run swirl experiment", add_help=True)
    parser.add_argument("config", nargs="?", default="../experiments/tpchskew.json",
                        help="Path to experiment configuration JSON")
    args = parser.parse_args()
    CONFIGURATION_FILE = args.config

    # 创建 Experiment 类的实例，传入配置文件路径
    experiment = Experiment(CONFIGURATION_FILE)

    # 调用 Experiment 实例的 prepare 方法，进行实验的准备工作
    experiment.prepare()

    experiment.compare()

    # # file_path = f"../experiment_results/TPCH/ID_TPCH_Test_Experiment_timetamps_1742544729/experiment_object.pickle"
    # # with open(file_path, "rb") as handle:
    # #     loaded_experiment = pickle.load(handle)
    # # experiment = loaded_experiment
    # # 根据实验配置中的并行环境数量选择并行环境类型
    # #ParallelEnv = SubprocVecEnv if experiment.config["parallel_environments"] > 1 else DummyVecEnv
    # # 使用 DummyVecEnv 作为并行环境类型
    # ParallelEnv = DummyVecEnv
    #
    # # 创建训练环境，使用 ParallelEnv 包装多个环境实例
    # training_env = ParallelEnv(
    #     [experiment.make_env(env_id) for env_id in range(experiment.config["parallel_environments"])]
    # )
    # # 对训练环境进行归一化处理，包括观测值和奖励值
    # training_env = VecNormalize(
    #     training_env, norm_obs=True, norm_reward=True, gamma=experiment.config["rl_algorithm"]["gamma"], training=True
    # )
    #
    # # 设置实验使用的模型类型为 MaskablePPO
    # experiment.model_type = MaskablePPO
    #
    # # 将实验对象保存为 pickle 文件，以便后续使用
    # with open(f"{experiment.experiment_folder_path}/experiment_object.pickle", "wb") as handle:
    #     pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # # 根据观测管理器类型选择不同的策略参数
    # if experiment.config["observation_manager"] == "BISLearnerObservationManager":
    #     # 定义策略的关键字参数，包括特征提取器类和网络架构
    #     policy_kwargs = dict(
    #         features_extractor_class=CustomCombinedExtractor,
    #         #net_arch=dict(pi=[512, 512], vf=[256, 256])
    #         net_arch=dict(pi=experiment.config["rl_algorithm"]["model_architecture"]["net_arch"][0]["vf"],
    #                       vf=experiment.config["rl_algorithm"]["model_architecture"]["net_arch"][0]["pi"])
    #     )
    #
    #     # 创建 MaskablePPO 模型实例
    #     model = MaskablePPO(
    #         policy="MultiInputPolicy",
    #         env=training_env,
    #         verbose=2,
    #         policy_kwargs=policy_kwargs,
    #         seed=experiment.config["random_seed"],
    #         gamma=experiment.config["rl_algorithm"]["gamma"],
    #         tensorboard_log="tensor_log",
    #         **experiment.config["rl_algorithm"]["args"],
    #     )
    # else:
    #     # 当观测管理器不是 BISLearnerObservationManager 时，创建 MaskablePPO 模型实例
    #     model = MaskablePPO(
    #         policy=experiment.config["rl_algorithm"]["policy"],
    #         env=training_env,
    #         verbose=2,
    #         seed=experiment.config["random_seed"],
    #         gamma=experiment.config["rl_algorithm"]["gamma"],
    #         tensorboard_log="tensor_log",
    #         # 复制模型架构配置，避免被修改
    #         policy_kwargs=copy.copy(
    #             experiment.config["rl_algorithm"]["model_architecture"]
    #         ),
    #         **experiment.config["rl_algorithm"]["args"],
    #     )
    #
    # # 记录警告信息，显示创建的模型使用的神经网络架构
    # logging.warning(f"Creating model with NN architecture: {experiment.config['rl_algorithm']['model_architecture']}")
    #
    # # 将创建的模型设置到实验对象中
    # experiment.set_model(model)
    #
    # # 调用实验对象的 compare 方法，进行比较操作
    # experiment.compare()

    # callback_test_env = VecNormalize(
    #     DummyVecEnv([experiment.make_env(0, EnvironmentType.TESTING)]),
    #     norm_obs=True,
    #     norm_reward=False,
    #     gamma=experiment.config["rl_algorithm"]["gamma"],
    #     training=False,
    # )
    # test_callback = EvalCallbackWithTBRunningAverage(
    #     n_eval_episodes=experiment.config["workload"]["validation_testing"]["number_of_workloads"],
    #     eval_freq=round(experiment.config["validation_frequency"] / experiment.config["parallel_environments"]),
    #     eval_env=callback_test_env,
    #     verbose=1,
    #     name="test",
    #     deterministic=True,
    #     comparison_performances=experiment.comparison_performances["test"],
    # )

    # # 创建验证环境，并进行归一化处理
    # callback_validation_env = VecNormalize(
    #     DummyVecEnv([experiment.make_env(0, EnvironmentType.VALIDATION)]),
    #     norm_obs=True,
    #     norm_reward=False,
    #     gamma=experiment.config["rl_algorithm"]["gamma"],
    #     training=False,
    # )
    # # 创建验证回调函数，用于在训练过程中进行验证
    # validation_callback = EvalCallbackWithTBRunningAverage(
    #     n_eval_episodes=len(experiment.config["budgets"]["validation"]),
    #     eval_freq=round(experiment.config["validation_frequency"] / experiment.config["parallel_environments"]),
    #     eval_env=callback_validation_env,
    #     best_model_save_path=experiment.experiment_folder_path,
    #     verbose=1,
    #     name="validation",
    #     deterministic=True,
    #     comparison_performances=experiment.comparison_performances["validation"],
    # )
    # # 定义回调函数列表，初始只包含验证回调函数
    # callbacks = [validation_callback]
    #
    # # 如果存在多验证工作负载，则创建多验证环境和回调函数
    # if len(experiment.multi_validation_wl) > 0:
    #     callback_multi_validation_env = VecNormalize(
    #         DummyVecEnv([experiment.make_env(0, EnvironmentType.VALIDATION, experiment.multi_validation_wl)]),
    #         norm_obs=True,
    #         norm_reward=False,
    #         gamma=experiment.config["rl_algorithm"]["gamma"],
    #         training=False,
    #     )
    #     multi_validation_callback = EvalCallbackWithTBRunningAverage(
    #         n_eval_episodes=len(experiment.multi_validation_wl),
    #         eval_freq=round(experiment.config["validation_frequency"] / experiment.config["parallel_environments"]),
    #         eval_env=callback_multi_validation_env,
    #         best_model_save_path=experiment.experiment_folder_path,
    #         verbose=1,
    #         name="multi_validation",
    #         deterministic=True,
    #         comparison_performances={},
    #     )
    #     # 将多验证回调函数添加到回调函数列表中
    #     callbacks.append(multi_validation_callback)
    # start_time = time.time()
    # print(f"调用实验对象的 start_learning的开始时间：{start_time}")
    # # 调用实验对象的 start_learning 方法，开始学习过程
    # experiment.start_learning()
    # # 调用模型的 learn 方法，开始训练模型
    # model.learn(
    #     total_timesteps=experiment.config["timesteps"],
    #     callback=callbacks,
    #     tb_log_name=experiment.id,
    # )
    # # 调用实验对象的 finish_learning 方法，结束学习过程
    # experiment.finish_learning(
    #     training_env,
    #     validation_callback.moving_average_step * experiment.config["parallel_environments"],
    #     validation_callback.best_model_step * experiment.config["parallel_environments"],
    # )
    # end_time = time.time()
    # print(f"调用实验对象的 finish_learning 方法后的时间 ：f{end_time}")
    # print(f"耗时：{end_time - start_time}")
    # # 再次调用实验对象的 compare 方法，进行比较操作
    # # experiment.compare()
    #
    # # 调用实验对象的 finish 方法，结束实验
    # experiment.finish()


