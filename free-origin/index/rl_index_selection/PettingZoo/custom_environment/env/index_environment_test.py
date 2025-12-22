import collections
import copy
import logging
import random

import gym

import sys 
sys.path.append("..") 
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.index import Index
from index_selection_evaluation.selection.utils import b_to_mb
from gym_db.common import EnvironmentType
from swirl.experiment import Experiment
import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

NONE = -1

def env(render_mode=None,environment_type=EnvironmentType.TRAINING, config=None):
    ''''''
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    默认情况下，env函数通常将环境包装在包装器中。
    您可以找到这些方法的完整文档
    在开发人员文档的其他地方。
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    # 若render_mode为"ansi"，则internal_render_mode为"human"
    # 若render_mode为"human"，则internal_render_mode还是为"human"
    # 若render_mode不为以上2个，则internal_render_mode = render_mode

    env = raw_env(render_mode=internal_render_mode,environment_type=EnvironmentType.TRAINING, config=None)
    # This wrapper is only for environments which print results to the terminal
    # 此包装器仅适用于将结果打印到终端的环境

    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    # 该包装器有助于离散行动空间的错误处理

    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # 提供各种有用的用户错误
    # Strongly recommended
    # 强烈推荐

    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    ''''''
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    元数据包含环境常量。从体育馆，我们继承了“render_modes”，
    元数据，指定哪些模式可以放入render（）方法中。
    至少应该支持人工模式。
    “名称”元数据允许对环境进行漂亮的打印。
    """

    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None,environment_type=EnvironmentType.TRAINING, config=None):
        ''''''
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        init方法接受环境参数和
        应定义以下属性：
        -可能的智能体
        -行动空间
        -观测空间
        初始化后不应更改这些属性。
        """
        self.possible_agents = ['block_agent', 'index_agent']
        # possible_agents指的应该是允许的智能体的名字。

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        # zip()函数可以将列表组装为字典。
        # ------------对数据库信息进行初始化
        self.rnd = random.Random()
        self.rnd.seed(config["random_seed"])
        self.env_id = config["env_id"]
        self.environment_type = environment_type
        self.config = config

        self.number_of_resets = 0
        self.total_number_of_steps = 0

        self.connector = PostgresDatabaseConnector(config["database_name"], autocommit=True)
        self.connector.drop_indexes()
        self.cost_evaluation = CostEvaluation(self.connector)

        self.globally_indexable_columns = config["globally_indexable_columns"]
        # In certain cases, workloads are consumed: therefore, we need copy
        self.workloads = copy.copy(config["workloads"])
        self.current_workload_idx = 0
        self.similar_workloads = config["similar_workloads"]
        self.max_steps_per_episode = config["max_steps_per_episode"]
        self.partition_num = config["partition_num"]

        self.action_manager = config["action_manager"]
        self.action_manager.test_variable = self.env_id
        # -----------
        self._action_spaces = {
            'block_agent': Discrete(self.partition_num),
            'index_agent':Discrete(self.action_manager.number_of_index_per_partition)
            }
        self.observation_manager = config["observation_manager"]
        self._observation_spaces = {
            'block_agent': Discrete(len(self.observation_manager.number_of_features)),
            'index_agent':Discrete(len(self.observation_manager.number_of_features)+1)
            # 1 represents the block_agent's action
        }
        self.render_mode = render_mode
        self.reward_calculator = config["reward_calculator"]

        self._init_modifiable_state()

        if self.environment_type != environment_type.TRAINING:
            self.episode_performances = collections.deque(maxlen=len(config["workloads"]))

    # this cache ensures that same space object is returned for the same agent
    # 该缓存确保为同一智能体返回相同的空间对象
    # allows action space seeding to work as expected
    # 允许操作空间种子按预期工作
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self._observation_spaces[agent]


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def render(self):
        ''''''
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        渲染环境。在人工模式下，它可以打印到终端，打开
        打开一个图形窗口，或者打开一些人类可以看到和理解的其他显示。
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:
            string = "Current state: Agent1: Block{} , Agent2: Index{}".format(
                self.state[self.agents[0]], self.state[self.agents[1]]
            )
        else:
            string = "Game over"
        print(string)

    def observe(self, agent):
        ''''''
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        观察应返回指定智能体的观察结果。此功能
        应该返回理智的观察结果（尽管不一定是最新的）
        在调用reset（）之后的任何时间。
        """

        # observation of one agent is the previous state of the other
        #对一个代理的观察是另一个代理先前的状态
        return np.array(self.observations[agent])

    def close(self):
        ''''''
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        关闭应释放任何图形显示、子流程、网络连接
        或任何其他环境数据，这种环境数据不应继续存在，
        当用户不再使用该环境之后。
        """
        pass

    def reset(self, seed=None, return_info=False, options=None):
        ''''''
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        Reset需要初始化以下属性
        -智能体
        -奖励
        -累计奖励
        -终止
        -截断
        -信息
        -智能体选择
        并且必须设置环境，以便render（）、step（）和observe（）
        可以毫无问题地调用。
        在这里，它设置了step（）使用的状态字典，和step（）和observe（）使用的观测字典
        """
        self.agents = self.possible_agents[:]
        # 后面带上[:]，则是拷贝的意思，而非指向原来的对象。好处是：
        # 修改agents，不会影响到原来的possible_agents
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: NONE for agent in self.agents}
        self.num_moves = 0  # 时间步计数
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        我们的agent_selector实用程序允许轻松地循环遍历代理列表。
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next() # 我认为这里的第一次next()等同于reset()
        # 这两个有啥区别？为何要用next()？
        # 我的理解是：这个next()的作用是循环迭代智能体列表。
        # selector是迭代选择器；self.agent_selection是“当前智能体”（我结合后半部分代码推导出来的）
        # 初始化时，self.agent_selection要用选择器的next()功能来指向第0号智能体。其实reset()也会指向第0号智能体。

        self.number_of_resets += 1
        self.total_number_of_steps += self.steps_taken
        initial_observation = self._init_modifiable_state()
        return initial_observation
    
    def _step_asserts(self, actions):
        assert self.action_space('block_agent').contains(actions[0]) and self.action_space('index_agent').contains(actions[1]), f"{actions} ({type(actions)}) invalid"
        assert (
            self.valid_actions[self.global_index] == self.action_manager.ALLOWED_ACTION
        ), f"Agent has chosen invalid action: {self.global_index}"
        assert (
            Index(self.globally_indexable_columns[self.global_index]) not in self.current_indexes
        ), f"{Index(self.globally_indexable_columns[self.global_index])} already in self.current_indexes"

    def step(self, action):
        ''''''
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        步骤（动作）为当前代理执行动作（由agent_selection指定），并且需要更新：
        -奖励
        -_cumulative_rewards（累积奖励）
        -终止
        -截断
        -信息
        -agent_selection（到下一个代理）
        以及observe（）或render（）使用的任何内部状态
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # 处理步进已经死亡的智能体
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            # 接受一个代理的None操作，并将agent_selection移动到
            # 下一个死亡智能体，或者如果不再有死亡智能体，则移动到下个存活智能体
            self._was_dead_step(action) # 哪里定义了这个函数？在父类里
            # 这个函数会把截断或终止的智能体从智能体列表中删除。对应的代码是：
            '''
            del self.terminations[agent]
            del self.truncations[agent]
            del self.rewards[agent]
            del self._cumulative_rewards[agent]
            del self.infos[agent]
            self.agents.remove(agent)
            '''

            # 我认为，既然死亡了，就直接在这里把action设为None好了。
            # 不过，考虑到这篇代码里没有明确游戏结束的标志，所以这里仍然保留原状。
            # 事实上，self._was_dead_step(action)这个函数可以视为控制游戏结束的函数。
            # 这篇代码从头到尾都没有把terminations改为True
            return


        agent = self.agent_selection # 这里是下一个代理吗？不是。而是当前智能体
        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        # 最后一步执行的智能体有其累计值（因为它是由last（）返回的），因此该智能体的累计值应该在0处重新开始
        # 啥意思？没明白。
        # 这篇代码没写last()。last()这个函数是父类的
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        # 存储当前智能体的动作
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        # 如果是最后一个行动的智能体，则收取奖励
        if self._agent_selector.is_last():  # 这里的意思应该是：当前智能体是最后一个，则更新该时间步的奖励
        # 可以理解为：目的是控制一个完整的时间步。
            # culculate rewards
            block_id = self.state[self.agents[0]]
            index_id = self.state[self.agents[1]]
            self.global_index = self.action_manager.get_global_action(block_id, index_id)
            self._step_asserts([block_id,index_id])
            self.steps_taken += 1
            old_index_size = 0
            new_index = Index(self.globally_indexable_columns[self.global_index])
            self.current_indexes.add(new_index)
            if not new_index.is_single_column():
                parent_index = Index(new_index.columns[:-1])

                for index in self.current_indexes:
                    if index == parent_index:
                        old_index_size = index.estimated_size

                self.current_indexes.remove(parent_index)

                assert old_index_size > 0, "Parent index size must have been found if not single column index."
            environment_state = self._update_return_env_state(
                init=False, new_index=new_index, old_index_size=old_index_size
            )
            current_observation = self.observation_manager.get_observation(environment_state)
            self.valid_actions, is_valid_action_left = self.action_manager.update_valid_actions(
                index_id, self.current_budget, self.current_storage_consumption, block_id
            )
            episode_done = self.steps_taken >= self.max_steps_per_episode or not is_valid_action_left
            reward = self.reward_calculator.calculate_reward(environment_state)

            # TODO:two agents may need different reward
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = reward,reward
            if episode_done and self.environment_type != EnvironmentType.TRAINING:
                self._report_episode_performance(environment_state)
                self.current_workload_idx += 1
            # print(f"Indexes: {len(self.current_indexes)}")
            self.num_moves += 1 # 这个变量是时间步计数
            # The truncations dictionary must be updated for all players.
            # 必须为所有玩家更新截断字典。
            self.truncations = {
                agent: episode_done for agent in self.agents
            }
            # 这里截断的判断标准是：智能体行动的次数是否大于最大时间步。
            # 可以理解为：截断的意义就在于控制最大时间步。

            # observe the current state
            # 观察当前状态
            for i in self.agents:
                shared_observation = copy.deepcopy(current_observation)
                other_agent_action = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]
                self.observations[i] = np.append(shared_observation, other_agent_action)
        else:  # 此时一个时间步还没结束。
            # necessary so that observe() returns a reasonable observation at all times.
            # 必要的，以便observe（）始终返回合理的观测值。
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # no rewards are allocated until both players give an action
            # 在两名玩家都做出动作之前，不会分配奖励
            self._clear_rewards()

        # selects the next agent.
        # 选择下一个智能体。
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        # 向_cumulative_rewards添加奖励
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def _init_modifiable_state(self):
        self.current_indexes = set()
        self.steps_taken = 0
        self.current_storage_consumption = 0
        self.reward_calculator.reset()

        if len(self.workloads) == 0:
            self.workloads = copy.copy(self.config["workloads"])

        if self.environment_type == EnvironmentType.TRAINING:
            if self.similar_workloads:
                # 200 is an arbitrary value
                self.current_workload = self.workloads.pop(0 + self.env_id * 200)
            else:
                self.current_workload = self.rnd.choice(self.workloads)
        else:
            self.current_workload = self.workloads[self.current_workload_idx % len(self.workloads)]

        self.current_budget = self.current_workload.budget
        self.previous_cost = None

        self.valid_actions = self.action_manager.get_initial_valid_actions(self.current_workload, self.current_budget)
        environment_state = self._update_return_env_state(init=True)

        state_fix_for_episode = {
            "budget": self.current_budget,
            "workload": self.current_workload,
            "initial_cost": self.initial_costs,
        }
        self.observation_manager.init_episode(state_fix_for_episode)

        initial_observation = self.observation_manager.get_observation(environment_state)

        return initial_observation
    
    def _update_return_env_state(self, init, new_index=None, old_index_size=None):
        total_costs, plans_per_query, costs_per_query = self.cost_evaluation.calculate_cost_and_plans(
            self.current_workload, self.current_indexes, store_size=True
        )

        if not init:
            self.previous_cost = self.current_costs
            self.previous_storage_consumption = self.current_storage_consumption

        self.current_costs = total_costs

        if init:
            self.initial_costs = total_costs

        new_index_size = None
        # TODO:需要考虑index大小在不同平台上的差异
        if new_index is not None:
            self.current_storage_consumption += new_index.estimated_size
            self.current_storage_consumption -= old_index_size

            # This assumes that old_index_size is not None if new_index is not None
            assert new_index.estimated_size >= old_index_size

            new_index_size = new_index.estimated_size - old_index_size
            if new_index_size == 0:
                new_index_size = 1

            if self.current_budget:
                assert b_to_mb(self.current_storage_consumption) <= self.current_budget, (
                    "Storage consumption exceeds budget: "
                    f"{b_to_mb(self.current_storage_consumption)} "
                    f" > {self.current_budget}"
                )

        environment_state = {
            "action_status": self.action_manager.current_action_status,
            "current_storage_consumption": self.current_storage_consumption,
            "current_cost": self.current_costs,
            "previous_cost": self.previous_cost,
            "initial_cost": self.initial_costs,
            "new_index_size": new_index_size,
            "plans_per_query": plans_per_query,
            "costs_per_query": costs_per_query,
        }

        return environment_state

    def get_cost_eval_cache_info(self):
        return self.cost_evaluation.cost_requests, self.cost_evaluation.cache_hits, self.cost_evaluation.costing_time

    def get_cost_eval_cache(self):
        return self.cost_evaluation.cache
    
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy
import importlib
from pettingzoo.classic import rps_v2

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
    env = env(render_mode="human")

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
