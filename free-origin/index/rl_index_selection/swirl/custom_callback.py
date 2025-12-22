import os
import warnings
from typing import Union, List, Dict, Any, Optional
import gym
import numpy as np
import collections
# TensorFlow is only needed for optional TensorBoard summaries. Make it optional
# so non-RL runs (e.g., Extend baselines) don't fail when TF is absent.
try:  # pragma: no cover - import guard
    import tensorflow as tf  # type: ignore
except ImportError:  # pragma: no cover
    tf = None

from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation



class EvalCallbackWithTBRunningAverage(EventCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the `mean_reward`
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 1,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1,
                 name="",
                 comparison_performances=None):
        # 调用父类构造函数初始化回调机制
        super(EvalCallbackWithTBRunningAverage, self).__init__(callback_on_new_best, verbose=verbose)

        # 初始化评估核心参数
        self.n_eval_episodes = n_eval_episodes  # 每次评估的episode数量
        self.eval_freq = eval_freq              # 评估频率（回调触发次数）
        self.deterministic = deterministic      # 是否使用确定性动作策略
        self.render = render                    # 评估时是否渲染环境
        self.name = name                        # 回调名称（用于多场景区分）

        # 初始化奖励跟踪变量（记录最佳性能指标）
        self.best_mean_ra_reward = -np.inf      # 最佳10步移动平均奖励
        self.best_mean_ra_reward_3 = -np.inf    # 最佳3步移动平均奖励
        self.best_mean_reward = -np.inf         # 最佳单轮平均奖励
        self.moving_average_step = -1           # 最佳移动平均模型的步数
        self.best_model_step = -1               # 最佳单轮奖励模型的步数
        self.last_mean_reward = -np.inf         # 上一次评估的平均奖励

        # 构建性能对比字符串（用于评估日志输出）
        self.comparison_performance_str = ""
        for key, value in comparison_performances.items():
            if len(value) < 1:
                continue
            # 拼接格式："指标名: 平均值 (原始值列表) - "
            self.comparison_performance_str += f"{key}: {np.mean(value):.2f} ({value}) - "

        # 模型保存文件名附录（多验证场景下使用）
        self.model_appendix = ""
        if name == "multi_validation":
            self.model_appendix = "_mv"

        # 统一环境类型为VecEnv（确保评估环境接口一致性）
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # 非向量环境转换为DummyVecEnv
        # 确保评估环境为单环境实例
        assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"
        self.eval_env = eval_env

        # 初始化模型保存和日志路径
        self.best_model_save_path = best_model_save_path  # 最佳模型保存目录
        # 日志文件路径（默认为evaluations.npz）
        if log_path is not None:
            log_path = os.path.join(log_path, 'evaluations')
        self.log_path = log_path

        # 初始化评估结果记录列表
        self.evaluations_results = []    # 存储每次评估的奖励数据
        self.evaluations_timesteps = []  # 存储每次评估的时间步
        self.evaluations_length = []     # 存储每次评估的episode长度

        # 初始化移动平均奖励队列（滑动窗口）
        self.MEAN_REWARDS_LENGTH = 10    # 10步移动平均窗口大小
        self.mean_rewards = collections.deque(maxlen=self.MEAN_REWARDS_LENGTH)  # 10步奖励队列
        self.mean_rewards_3 = collections.deque(maxlen=3)  # 3步奖励队列


    def _init_callback(self):
        """
        回调初始化方法：执行回调前置检查与环境准备工作
        - 验证训练环境与评估环境类型一致性
        - 创建模型保存和日志所需的目录
        """
        # 环境类型一致性检查：避免因环境包装器不同导致的评估偏差
        # 某些极端情况下（如训练/评估环境使用不同包装器）可能失效
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

        # 目录创建：确保模型保存和日志路径存在（若已存在则不报错）
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)  # 创建最佳模型保存目录
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)  # 创建日志文件所在目录


    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # print("Start evaluating ...")
            # print("env current indexes: {}".format(self.eval_env[0].current_indexes))

            episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                               n_eval_episodes=self.n_eval_episodes,
                                                               render=self.render,
                                                               deterministic=self.deterministic,
                                                               return_episode_rewards=True)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results, ep_lengths=self.evaluations_length)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            self.mean_rewards.append(mean_reward)
            self.mean_rewards_3.append(mean_reward)

            episode_performances = self.eval_env.get_attr("episode_performances")[0]
            assert len(episode_performances) == len(episode_rewards), f"{len(episode_performances)} vs {len(episode_rewards)} \n {episode_performances} vs {episode_rewards}"
            perfs = []
            for perf in episode_performances:
                perfs.append(round(perf["achieved_cost"], 2))

            mean_performance = np.mean(perfs)

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
                print(f"Mean performance: {mean_performance:.2f} ({perfs}) - {self.comparison_performance_str}")

            moving_average = None
            if len(self.mean_rewards) == self.MEAN_REWARDS_LENGTH:
                moving_average = sum(self.mean_rewards) / self.MEAN_REWARDS_LENGTH

            if moving_average is not None and moving_average > self.best_mean_ra_reward:
                if self.verbose > 0:
                    print(f"New best running average reward: {moving_average} over {self.best_mean_ra_reward}\n  {self.mean_rewards}")
                if self.best_model_save_path is not None:
                    self.moving_average_step = self.n_calls
                    self.model.save(os.path.join(self.best_model_save_path, f'moving_average_model{self.model_appendix}'))
                self.best_mean_ra_reward = moving_average
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

            moving_average_3 = None
            if len(self.mean_rewards_3) == 3:
                moving_average_3 = sum(self.mean_rewards_3) / 3

            if moving_average_3 is not None and moving_average_3 > self.best_mean_ra_reward_3:
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, f'moving_average_model_3{self.model_appendix}'))
                self.best_mean_ra_reward_3 = moving_average_3

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward} over {self.best_mean_reward}\n  {self.mean_rewards}")
                if self.best_model_save_path is not None:
                    self.best_model_step = self.n_calls
                    self.model.save(os.path.join(self.best_model_save_path, f'best_mean_reward_model{self.model_appendix}'))
                self.best_mean_reward = mean_reward

            # Log scalar value
            # summary = tf.Summary(value=[tf.Summary.Value(tag=f'episode_reward/validation_reward_{self.name}', simple_value=mean_reward)])
            # self.locals['writer'].add_summary(summary, self.num_timesteps)
            #
            # summary = tf.Summary(value=[tf.Summary.Value(tag=f'episode_reward/validation_length_{self.name}', simple_value=mean_ep_length)])
            # self.locals['writer'].add_summary(summary, self.num_timesteps)
            #
            # summary = tf.Summary(value=[tf.Summary.Value(tag=f'episode_reward/validation_achieved_percentage_{self.name}', simple_value=mean_performance)])
            # self.locals['writer'].add_summary(summary, self.num_timesteps)


        return True

