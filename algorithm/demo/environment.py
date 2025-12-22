import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C # DQN coming soon
from stable_baselines3.common.env_util import make_vec_env
from attention_based_feature_extractor import *
from param import *

class GoLeftEnv(gym.Env):
  """
  这是一个让智能体学习一直向右上走的 2D grid 环境
  """
  metadata = {'render.modes': ['console']}

  def __init__(self, grid_size=10):
    super(GoLeftEnv, self).__init__()

    # 2D-grid 的大小(正方形)
    self.grid_size = grid_size
    # agent 初始化在 grid 的左下角
    self.agent_x_pos = 0
    self.agent_y_pos = 0

    # fake input
    self.observation_space = spaces.Dict({"position": spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype=np.float32),
                                          "block_state_inf": spaces.Box(low=-np.inf, high=np.inf,
                                                                        shape=(args.block_type, args.block_feature_dim), dtype=np.float32),
                                          "workload_inf": spaces.Box(low=-np.inf, high=np.inf,
                                                                        shape=(args.workload_size, args.query_feature_dim), dtype=np.float32),
                                          "block_based_workload_inf": spaces.Box(low=-np.inf, high=np.inf,
                                                                        shape=(args.block_type, 2 * args.indexed_attributes), dtype=np.float32),
                                          })

    # 定义 action  observation
    # 离散行为空间: left、 right
    n_actions = 2
    self.action_space = spaces.MultiDiscrete([n_actions, n_actions])
    # 观测是智能体现在的位置
    # self.observation_space = spaces.Box(low=0, high=self.grid_size,
    #                                     shape=(2,), dtype=np.float32)

  def reset(self):
    """
    Important: 观测必须是一个 np.array
    :return: (np.array)
    """
    # Initialize the agent
    # self.agent_pos = [0, 0]
    # self.agent_x_pos = 0
    # self.agent_y_pos = 0
    # here we convert to float32 to make it more general (in case we want to use continuous actions)
    # return np.array(self.agent_pos).astype(np.float32)

    return {"position": np.array([0, 0]),
            "block_state_inf": np.array([[0, 0], [0, 0]]),
            "workload_inf": np.array([[0, 0], [0, 0]]),
            "block_based_workload_inf": np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])}

  def step(self, action):
    action_x = action[0]
    action_y = action[1]
    if action_x == 1:
      self.agent_x_pos += 1
    else:
      self.agent_x_pos -= 1

    if action_y == 1:
      self.agent_y_pos += 1
    else:
      self.agent_y_pos -= 1
    # 如果走到边缘就不能继续走了
    self.agent_pos = np.clip([self.agent_x_pos, self.agent_y_pos], 0, self.grid_size)
    # 如果走到最左边代表结束了
    done = bool(self.agent_pos[0] == self.grid_size and self.agent_pos[1] == self.grid_size)
    # 走到最左边就给一个正的 reward
    reward = 1 if self.agent_pos[0] == self.grid_size and self.agent_pos[1] == self.grid_size else 0
    # 目前没有需要额外输出的信息
    info = {}
    return {"position": np.array(self.agent_pos),
            "block_state_inf": np.array([[0, 0], [0, 0]]),
            "workload_inf": np.array([[0, 0], [0, 0]]),
            "block_based_workload_inf": np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])}, reward, done, info

  def render(self, mode='console'):
    # 在命令行中渲染
    if mode != 'console':
      raise NotImplementedError()
    # agent is represented as a cross, rest as a dot
    print("." * self.agent_pos, end="")
    print("x", end="")
    print("." * (self.grid_size - self.agent_pos))

  def close(self):
    pass


env = GoLeftEnv(grid_size=10)
env = make_vec_env(lambda: env, n_envs=1)
# model = A2C('MlpPolicy', env, verbose=1).learn(5000)
policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor
)
model = A2C("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1).learn(10000)

# Test the trained agent
obs = env.reset()
n_steps = 20
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  # env.render(mode='console')
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break