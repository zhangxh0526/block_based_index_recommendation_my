import os

# os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
from Environment.fake_environment import fakeEnv
import tensorflow._api.v2.compat.v1 as tf
import multiprocessing as mp
from utils import *
from compute_baselines import *
from actor_agent import ActorAgent
from tf_logger import TFLogger
from param import *


def invoke_model(actor_agent, obs, exp):


    # invoking the learning model
    block_acts, index_acts, block_probs, index_probs, \
    workload_inputs, block_inputs, candidate_inputs, space_inputs, \
    block_valid_mask, index_valid_mask = actor_agent.invoke_model(obs)

    block_act_vec = np.zeros(block_probs.shape)
    block_act_vec[0, block_acts[0]] = 1

    # for storing job index
    index_act_vec = np.zeros(index_probs.shape)
    index_act_vec[0, index_acts[0]] = 1

    if sum(block_valid_mask[0, :]) == 0 or sum(index_valid_mask[0, :]) == 0:
        # no node is valid to assign
        return None, None


    # parse action
    block_idx = block_acts[0]
    index_idx = index_acts[0]


    # store experience
    exp['block_acts'].append(block_acts)
    exp['index_acts'].append(index_acts)
    exp['index_acts'].append(index_acts)
    exp['index_probs'].append(index_probs)

    exp['workload_inputs'].append(workload_inputs)
    exp['block_inputs'].append(block_inputs)
    exp['candidate_inputs'].append(candidate_inputs)
    exp['space_inputs'].append(space_inputs)

    exp['block_act_vec'].append(block_act_vec)
    exp['index_act_vec'].append(index_act_vec)

    exp['block_valid_mask'].append(block_valid_mask)
    exp['index_valid_mask'].append(index_valid_mask)


    return block_idx, index_idx


def train_agent(agent_id, param_queue, reward_queue, adv_queue, gradient_queue):
    # model evaluation seed
    tf.set_random_seed(agent_id)

    # set up environment
    env = fakeEnv()

    # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.worker_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.worker_gpu_fraction))

    sess = tf.Session(config=config)

    # set up actor agent
    actor_agent = ActorAgent(
        sess, args.workload_dim, args.block_num, args.block_dim, args.candidate_dim, args.space_dim,
        args.hid_dims, args.attention_dim, args.output_dim)

    # collect experiences
    while True:
        # get parameters from master
        (actor_params, seed, max_time, entropy_weight) = \
            param_queue.get()

        # synchronize model
        actor_agent.set_params(actor_params)


        # set up storage for experience
        exp = {'block_acts': [], 'index_acts': [], \
               'block_probs': [], 'index_probs': [], \
               'workload_inputs': [], 'block_inputs': [], \
               'candidate_inputs': [], \
               'space_inputs': [], 'block_act_vec': [], \
               'index_act_vec': [], 'block_valid_mask': [], \
               'index_valid_mask': [],
               'reward': [], 'wall_time': []}

        try:


            # run experiment
            obs = env.observe()
            done = False

            # initial time
            exp['wall_time'].append(env.wall_time.curr_time)

            while not done:

                block_idx, index_idx = invoke_model(actor_agent, obs, exp)

                obs, reward, done = env.step(block_idx, index_idx)

                if block_idx is not None:
                    # valid action, store reward and time
                    exp['reward'].append(reward)
                    exp['wall_time'].append(env.wall_time.curr_time)
                elif len(exp['reward']) > 0:
                    # Note: if we skip the reward when node is None
                    # (i.e., no available actions), the sneaky
                    # agent will learn to exhaustively pick all
                    # nodes in one scheduling round, in order to
                    # avoid the negative reward
                    exp['reward'][-1] += reward
                    exp['wall_time'][-1] = env.wall_time.curr_time

            # report reward signals to master
            reward_queue.put(
                [exp['reward'], exp['wall_time'],
                 len(env.finished_queries),
                 np.mean([j.completion_time - j.start_time \
                          for j in env.finished_queries]),
                 env.wall_time.curr_time >= env.max_time])

            # get advantage term from master
            batch_adv = adv_queue.get()

            if batch_adv is None:
                # some other agents panic for the try and the
                # main thread throw out the rollout, reset and
                # try again now
                continue

            # compute gradients
            actor_gradient, loss = compute_actor_gradients(
                actor_agent, exp, batch_adv, entropy_weight)

            # report gradient to master
            gradient_queue.put([actor_gradient, loss])

        except AssertionError:
            # ask the main to abort this rollout and
            # try again
            reward_queue.put(None)
            # need to still get from adv_queue to
            # prevent blocking
            adv_queue.get()

def compute_actor_gradients(actor_agent, exp, batch_adv, entropy_weight):
    batch_points = truncate_experiences(exp['job_state_change'])

    all_gradients = []
    all_loss = [[], [], 0]

    for b in range(len(batch_points) - 1):
        # need to do different batches because the
        # size of dags in state changes
        ba_start = batch_points[b]
        ba_end = batch_points[b + 1]

        # use a piece of experience
        workload_inputs = np.vstack(exp['workload_inputs'][ba_start: ba_end])
        block_inputs = np.vstack(exp['block_inputs'][ba_start: ba_end])
        candidate_inputs = np.vstack(exp['candidate_inputs'][ba_start: ba_end])
        space_inputs = np.vstack(exp['space_inputs'][ba_start: ba_end])
        block_valid_mask = np.vstack(exp['block_valid_mask'][ba_start: ba_end])
        index_valid_mask = np.vstack(exp['index_valid_mask'][ba_start: ba_end])
        block_act_vec = np.vstack(exp['block_act_vec'][ba_start: ba_end])
        index_act_vec = np.vstack(exp['index_act_vec'][ba_start: ba_end])
        adv = batch_adv[ba_start : ba_end, :]

        # compute gradient
        act_gradients, loss = actor_agent.get_gradients(
            workload_inputs, block_inputs, candidate_inputs, space_inputs,
            block_valid_mask, index_valid_mask,
            block_act_vec, index_act_vec,
            adv, entropy_weight)

        all_gradients.append(act_gradients)
        all_loss[0].append(loss[0])
        all_loss[1].append(loss[1])

    all_loss[0] = np.sum(all_loss[0])
    all_loss[1] = np.sum(all_loss[1])  # to get entropy
    all_loss[2] = np.sum(batch_adv ** 2) # time based baseline loss

    # aggregate all gradients from the batches
    gradients = aggregate_gradients(all_gradients)

    return gradients, all_loss

class AveragePerStepReward(object):
    def __init__(self, size):
        self.size = size
        self.count = 0
        self.reward_record = []
        self.time_record = []
        self.reward_sum = 0
        self.time_sum = 0

    def add(self, reward, time):
        if self.count >= self.size:
            stale_reward = self.reward_record.pop(0)
            stale_time = self.time_record.pop(0)
            self.reward_sum -= stale_reward
            self.time_sum -= stale_time
        else:
            self.count += 1

        self.reward_record.append(reward)
        self.time_record.append(time)
        self.reward_sum += reward
        self.time_sum += time

    def add_list(self, list_reward, list_time):
        assert len(list_reward) == len(list_time)
        for i in range(len(list_reward)):
            self.add(list_reward[i], list_time[i])

    def add_list_filter_zero(self, list_reward, list_time):
        assert len(list_reward) == len(list_time)
        for i in range(len(list_reward)):
            if list_time[i] != 0:
                self.add(list_reward[i], list_time[i])
            else:
                assert list_reward[i] == 0

    def get_avg_per_step_reward(self):
        return float(self.reward_sum) / float(self.time_sum)


def main():
    np.random.seed(1)
    tf.set_random_seed(1)

    # create result and model folder
    create_folder_if_not_exists("./result")
    create_folder_if_not_exists("./model")

    # initialize communication queues
    params_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    reward_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    adv_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    gradient_queues = [mp.Queue(1) for _ in range(args.num_agents)]

    # set up training agents
    agents = []
    for i in range(args.num_agents):
        agents.append(mp.Process(target=train_agent, args=(
            i, params_queues[i], reward_queues[i],
            adv_queues[i], gradient_queues[i])))

    # start training agents
    for i in range(args.num_agents):
        agents[i].start()

    # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.master_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.master_gpu_fraction))

    sess = tf.Session(config=config)

    # set up actor agent
    actor_agent = ActorAgent(
        sess, args.workload_dim, args.block_num, args.block_dim, args.candidate_dim, args.space_dim,
        args.hid_dims, args.attention_dim, args.output_dim)

    # tensorboard logging
    tf_logger = TFLogger(sess, [
        'actor_loss', 'entropy', 'value_loss', 'episode_length',
        'average_reward_per_second', 'sum_reward', 'reset_probability',
        'num_jobs', 'reset_hit', 'average_job_duration',
        'entropy_weight'])

    # store average reward for computing differential rewards
    avg_reward_calculator = AveragePerStepReward(
        args.average_reward_storage_size)

    # initialize entropy parameters
    entropy_weight = args.entropy_weight_init

    # initialize episode reset probability
    reset_prob = args.reset_prob

    # ---- start training process ----
    for ep in range(1, args.num_ep):
        print('training epoch', ep)

        # synchronize the model parameters for each training agent
        actor_params = actor_agent.get_params()

        # generate max time stochastically based on reset prob
        max_time = generate_coin_flips(reset_prob)

        # send out parameters to training agents
        for i in range(args.num_agents):
            params_queues[i].put([
                actor_params, args.seed + ep,
                max_time, entropy_weight])

        # storage for advantage computation
        all_rewards, all_diff_times, all_times, \
        all_num_finished_queries, all_avg_query_duration, \
        all_reset_hit, = [], [], [], [], [], []

        t1 = time.time()

        # get reward from agents
        any_agent_panic = False

        for i in range(args.num_agents):
            result = reward_queues[i].get()

            if result is None:
                any_agent_panic = True
                continue
            else:
                batch_reward, batch_time, \
                num_finished_queries, avg_query_duration, \
                reset_hit = result

            diff_time = np.array(batch_time[1:]) - \
                        np.array(batch_time[:-1])

            all_rewards.append(batch_reward)
            all_diff_times.append(diff_time)
            all_times.append(batch_time[1:])
            all_num_finished_queries.append(num_finished_queries)
            all_avg_query_duration.append(avg_query_duration)
            all_reset_hit.append(reset_hit)

            avg_reward_calculator.add_list_filter_zero(
                batch_reward, diff_time)

        t2 = time.time()
        print('got reward from workers', t2 - t1, 'seconds')

        if any_agent_panic:
            # The try condition breaks in some agent (should
            # happen rarely), throw out this rollout and try
            # again for next iteration (TODO: log this event)
            for i in range(args.num_agents):
                adv_queues[i].put(None)
            continue

        # compute differential reward
        all_cum_reward = []
        avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
        for i in range(args.num_agents):
            if args.diff_reward_enabled:
                # differential reward mode on
                rewards = np.array([r - avg_per_step_reward * t for \
                                    (r, t) in zip(all_rewards[i], all_diff_times[i])])
            else:
                # regular reward
                rewards = np.array([r for \
                                    (r, t) in zip(all_rewards[i], all_diff_times[i])])

            cum_reward = discount(rewards, args.gamma)

            all_cum_reward.append(cum_reward)

        # compute baseline
        baselines = get_piecewise_linear_fit_baseline(all_cum_reward, all_times)

        # give worker back the advantage
        for i in range(args.num_agents):
            batch_adv = all_cum_reward[i] - baselines[i]
            batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])
            adv_queues[i].put(batch_adv)

        t3 = time.time()
        print('advantage ready', t3 - t2, 'seconds')

        actor_gradients = []
        all_action_loss = []  # for tensorboard
        all_entropy = []  # for tensorboard
        all_value_loss = []  # for tensorboard

        for i in range(args.num_agents):
            (actor_gradient, loss) = gradient_queues[i].get()

            actor_gradients.append(actor_gradient)
            all_action_loss.append(loss[0])
            all_entropy.append(-loss[1] / \
                               float(all_cum_reward[i].shape[0]))
            all_value_loss.append(loss[2])

        t4 = time.time()
        print('worker send back gradients', t4 - t3, 'seconds')

        actor_agent.apply_gradients(
            aggregate_gradients(actor_gradients), args.lr)

        t5 = time.time()
        print('apply gradient', t5 - t4, 'seconds')

        tf_logger.log(ep, [
            np.mean(all_action_loss),
            np.mean(all_entropy),
            np.mean(all_value_loss),
            np.mean([len(b) for b in baselines]),
            avg_per_step_reward * args.reward_scale,
            np.mean([cr[0] for cr in all_cum_reward]),
            reset_prob,
            np.mean(all_num_finished_queries),
            np.mean(all_reset_hit),
            np.mean(all_avg_query_duration),
            entropy_weight])

        # decrease entropy weight
        entropy_weight = decrease_var(entropy_weight,
                                      args.entropy_weight_min, args.entropy_weight_decay)

        # decrease reset probability
        reset_prob = decrease_var(reset_prob,
                                  args.reset_prob_min, args.reset_prob_decay)

        if ep % args.model_save_interval == 0:
            actor_agent.save_model(args.model_folder + \
                                   'model_ep_' + str(ep))

    sess.close()


if __name__ == '__main__':
    main()