''''''
'''
This is a carefully commented version of the PettingZoo rock paper scissors environment.
这是一个经过仔细注释的PettingZoo石头剪刀布环境版本。
https://pettingzoo.farama.org/content/environment_creation/
'''
import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

ROCK = 0
PAPER = 1
SCISSORS = 2
NONE = 3
MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
NUM_ITERS = 5  # 这个常量是最大时间步。原为100。
REWARD_MAP = {
    (ROCK, ROCK): (0, 0),
    (ROCK, PAPER): (-1, 1),
    (ROCK, SCISSORS): (1, -1),
    (PAPER, ROCK): (1, -1),
    (PAPER, PAPER): (0, 0),
    (PAPER, SCISSORS): (-1, 1),
    (SCISSORS, ROCK): (-1, 1),
    (SCISSORS, PAPER): (1, -1),
    (SCISSORS, SCISSORS): (0, 0),
}


def env(render_mode=None):
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

    env = raw_env(render_mode=internal_render_mode)
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

    def __init__(self, render_mode=None):
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
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        # possible_agents指的应该是允许的智能体的名字。

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        # zip()函数可以将列表组装为字典。
        #self.agent_name_mapping为{'player_0': 0, 'player_1': 1}

        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # 这里定义并记录了健身房的空间：https://gymnasium.farama.org/api/spaces/
        self._action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
        # self._action_spaces为{'player_0': Discrete(3), 'player_1': Discrete(3)}
        self._observation_spaces = {
            agent: Discrete(4) for agent in self.possible_agents
        }
        self.render_mode = render_mode

    # this cache ensures that same space object is returned for the same agent
    # 该缓存确保为同一智能体返回相同的空间对象
    # allows action space seeding to work as expected
    # 允许操作空间种子按预期工作
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Discrete(4)


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

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
            string = "Current state: Agent1: {} , Agent2: {}".format(
                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
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
        # 其实这句话可以忽略。它只是简单替代了self.agent_selection

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
            # rewards for all agents are placed in the .rewards dictionary
            # 所有智能体的奖励都放在奖励字典中
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
                (self.state[self.agents[0]], self.state[self.agents[1]])
            ]

            self.num_moves += 1 # 这个变量是时间步计数
            # The truncations dictionary must be updated for all players.
            # 必须为所有玩家更新截断字典。
            self.truncations = {
                agent: self.num_moves >= NUM_ITERS for agent in self.agents
            }
            # 这里截断的判断标准是：智能体行动的次数是否大于最大时间步。
            # 可以理解为：截断的意义就在于控制最大时间步。

            # observe the current state
            # 观察当前状态
            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]
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

    
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy

from pettingzoo.classic import rps_v2

if __name__ == "__main__":
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
