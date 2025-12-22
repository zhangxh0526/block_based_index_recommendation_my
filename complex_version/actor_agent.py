import keras.layers
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
import tensorflow.keras.layers as tl
import bisect
from agent import Agent
from tf_ops import *
from mlp import MLP
from mha import *
import tf_slim
from param import *

class ActorAgent(Agent):
    def __init__(self, sess,
                 workload_dim, block_num, block_dim, candidate_dim, space_dim,
                 hid_dims, attention_dim, output_dim,
                 eps=1e-6, act_fn=leaky_relu,
                 optimizer=tf.train.AdamOptimizer, scope='actor_agent'):

        Agent.__init__(self)

        self.sess = sess
        self.workload_dim = workload_dim
        self.block_num = block_num
        self.block_dim = block_dim
        self.candidate_dim = candidate_dim
        self.space_dim = space_dim
        self.hid_dims = hid_dims
        self.attention_dim = attention_dim
        self.output_dim = output_dim

        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.scope = scope


        # input workload feature: [batch_size, query_templates, num_features]
        self.workload_inputs = tf.placeholder(tf.float32, [None, args.query_template_max, args.query_dim])
        # block_based_workload_feature: [batch_size, block_num, block_based_query_dim]
        self.block_based_workload_inputs = tf.placeholder(tf.float32, [None, args.block_limit, args.block_based_query_dim])
        # input block feature: [batch_size, block_num, num_features]
        self.block_inputs = tf.placeholder(tf.float32, [None, args.block_limit, args.block_dim])
        # input index config feature: [batch_size, block_num, num_features]
        self.candidate_inputs = tf.placeholder(tf.float32, [None, args.block_num, args.candidate_dim])
        # input space feature: [batch_size, ]
        self.space_inputs = tf.placeholder(tf.float32, [None, self.space_dim])


        # self.workload_embedding = MLP(self.workload_inputs, self.workload_dim, self.hid_dims, self.attention_dim,
        #                               act_fn, scope='workload_mlp').outputs
        # self.block_embedding = MLP(self.block_inputs, self.block_dim, self.hid_dims, self.attention_dim,
        #                               act_fn, scope='block_mlp').outputs

        # attention
        attention = MultiHeadAttention(self.attention_dim, 1, 0.5)

        # self.workload_embedding = tf.reshape(self.workload_embedding, shape=(-1, 1, self.attention_dim))
        # self.block_embedding = tf.reshape(self.block_embedding, shape=(-1, self.block_num, self.attention_dim))

        self.block_aware_workload_embedding = attention.forward(self.workload_inputs, self.block_based_workload_inputs, self.block_based_workload_inputs)

        # valid mask for block action ([batch_size, total_num_nodes])
        self.block_valid_mask = tf.placeholder(tf.float32, [None, None])

        # valid mask for index action ([batch_size, num_jobs * num_exec_limits])
        self.index_valid_mask = tf.placeholder(tf.float32, [None, None])

        self.block_probs, self.index_probs = self.actor_network(
            self.workload_inputs, self.block_inputs,
            self.block_aware_workload_embedding,
            self.candidate_inputs, self.space_inputs,
            self.block_valid_mask, self.index_valid_mask,
            self.act_fn)

        # draw action based on the probability (from OpenAI baselines)
        # block_acts [batch_size, 1]
        logits = tf.log(self.block_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.block_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 1)

        # index_acts [batch_size, 1]
        logits = tf.log(self.index_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.index_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 1)

        # Selected action for block, 0-1 vector ([batch_size, total_num_blocks])
        self.block_act_vec = tf.placeholder(tf.float32, [None, None])
        # Selected action for index, 0-1 vector ([batch_size, total_num_blocks])
        self.index_act_vec = tf.placeholder(tf.float32, [None, None])

        # advantage term (from Monte Calro or critic) ([batch_size, 1])
        self.adv = tf.placeholder(tf.float32, [None, 1])

        # use entropy to promote exploration, this term decays over time
        self.entropy_weight = tf.placeholder(tf.float32, ())

        # select block action probability
        self.selected_block_prob = tf.reduce_sum(tf.multiply(
            self.block_probs, self.block_act_vec),
            reduction_indices=1, keep_dims=True)

        # select index action probability
        self.selected_index_prob = tf.reduce_sum(tf.multiply(
            self.index_probs, self.index_act_vec),
            reduction_indices=1, keep_dims=True)

        # actor loss due to advantge (negated)
        self.adv_loss = tf.reduce_sum(tf.multiply(
            tf.log(self.block_probs * self.index_probs + \
            self.eps), -self.adv))

        # block_entropy
        self.block_entropy = tf.reduce_sum(tf.multiply(
            self.block_probs, tf.log(self.block_probs + self.eps)))

        # index entropy
        self.index_entropy = tf.reduce_sum(tf.multiply(
            self.index_probs, tf.log(self.index_probs + self.eps)))

        # entropy loss
        self.entropy_loss = self.block_entropy + self.index_entropy

        # normalize entropy
        self.entropy_loss /= \
            tf.log(tf.cast(tf.shape(self.block_probs)[1], tf.float32))
            # normalize over batch size (note: adv_loss is sum)
            # * tf.cast(tf.shape(self.node_act_probs)[0], tf.float32)

        # define combined loss
        self.act_loss = self.adv_loss + self.entropy_weight * self.entropy_loss

        # get training parameters
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        # operations for setting network parameters
        self.input_params, self.set_params_op = \
            self.define_params_op()

        # actor gradients
        self.act_gradients = tf.gradients(self.act_loss, self.params)

        # adaptive learning rate
        self.lr_rate = tf.placeholder(tf.float32, shape=[])

        # actor optimizer
        self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)

        # apply gradient directly to update parameters
        self.apply_grads = self.optimizer(self.lr_rate).\
            apply_gradients(zip(self.act_gradients, self.params))

        # network paramter saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())

        # if args.saved_model is not None:
        #     self.saver.restore(self.sess, args.saved_model)

    def actor_network(self, workload_inputs, block_inputs,
            block_aware_workload_embedding,
            candidate_inputs, space_inputs,
            block_valid_mask, index_valid_mask,
            act_fn):

        # choose block
        batch_size = tf.shape(block_valid_mask)[0]

        workload_inputs = tf.reshape(workload_inputs, [batch_size, -1, self.workload_dim])

        block_inputs = tf.reshape(block_inputs, [batch_size, -1, self.block_num * self.block_dim])

        block_aware_workload_embedding = tf.reshape(block_aware_workload_embedding,
                                                     [batch_size, -1, self.attention_dim])

        #
        with tf.variable_scope(self.scope):
            # -- part A, the distribution over nodes --
            merge_query = tf.concat([
                workload_inputs,
                block_inputs,
                block_aware_workload_embedding,
                candidate_inputs,
                space_inputs], axis=2)

            block_hid_0 = tf_slim.fully_connected(merge_query, 32, activation_fn=act_fn)
            block_hid_1 = tf_slim.fully_connected(block_hid_0, 16, activation_fn=act_fn)
            block_hid_2 = tf_slim.fully_connected(block_hid_1, 8, activation_fn=act_fn)
            block_outputs = tf_slim.fully_connected(block_hid_2, 1, activation_fn=None)

            # reshape the output dimension (batch_size, total_num_blocks)
            block_outputs = tf.reshape(block_outputs, [batch_size, -1])

            # valid mask on block
            block_valid_mask = (block_valid_mask - 1) * 10000.0

            # apply mask
            block_outputs = block_outputs + block_valid_mask

            # do masked softmax over nodes on the graph
            block_outputs = tf.nn.softmax(block_outputs, dim=-1)

            # -- part B, the distribution over executor limits --
            merge_index = tf.concat([
                workload_inputs,
                block_inputs,
                attribute_aware_block_embedding,
                candidate_inputs,
                space_inputs], axis=2)

            index_hid_0 = tf_slim.fully_connected(merge_index, 32, activation_fn=act_fn)
            index_hid_1 = tf_slim.fully_connected(index_hid_0, 16, activation_fn=act_fn)
            index_hid_2 = tf_slim.fully_connected(index_hid_1, 8, activation_fn=act_fn)
            index_outputs = tf_slim.fully_connected(index_hid_2, 1, activation_fn=None)

            # reshape the output dimension (batch_size, total_num_nodes)
            index_outputs = tf.reshape(index_outputs, [batch_size, -1])

            # valid mask on block
            index_valid_mask = (index_valid_mask - 1) * 10000.0

            # apply mask
            index_outputs = index_outputs + index_valid_mask

            # do masked softmax over nodes on the graph
            index_outputs = tf.nn.softmax(index_outputs, dim=-1)


            return block_outputs, index_outputs

    def apply_gradients(self, gradients, lr_rate):
        self.sess.run(self.apply_grads, feed_dict={
            i: d for i, d in zip(
                self.act_gradients + [self.lr_rate],
                gradients + [lr_rate])
        })

    def define_params_op(self):
        # define operations for setting network parameters
        input_params = []
        for param in self.params:
            input_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op


    def get_params(self):
        return self.sess.run(self.params)

    def save_model(self, file_path):
        self.saver.save(self.sess, file_path)

    def get_gradients(self, workload_inputs, block_inputs, candidate_inputs, space_inputs,
                      block_valid_mask, index_valid_mask,
                      block_act_vec, index_act_vec, adv, entropy_weight):

        return self.sess.run([self.act_gradients,
            [self.adv_loss, self.entropy_loss]],
            feed_dict={i: d for i, d in zip(
                [self.workload_inputs] + [self.block_inputs] + \
                [self.candidate_inputs] + [self.space_inputs] +
                [self.block_valid_mask] + [self.index_valid_mask] +
                [self.block_act_vec] + [self.index_act_vec] +
                [self.adv] + [self.entropy_weight],\
                [workload_inputs] + [block_inputs] +\
                [candidate_inputs] + [space_inputs] +
                [block_valid_mask] + [index_valid_mask] +
                [block_act_vec] + [index_act_vec] +
                [adv] + [entropy_weight])
        })

    def predict(self, workload_inputs, block_inputs, candidate_inputs, space_inputs,
                      block_valid_mask, index_valid_mask):
        return self.sess.run([self.block_probs, self.index_probs,
            self.block_acts, self.index_acts],\
            feed_dict={i: d for i, d in zip(
                 [self.workload_inputs] + [self.block_inputs] +\
                 [self.candidate_inputs] + [self.space_inputs] +
                 [self.block_valid_mask] + [self.index_valid_mask] +
                 [workload_inputs] + [block_inputs] +\
                 [candidate_inputs] + [space_inputs] +
                 [block_valid_mask] + [index_valid_mask])
        })

    def set_params(self, input_params):
        self.sess.run(self.set_params_op, feed_dict={
            i: d for i, d in zip(self.input_params, input_params)
        })

    def translate_state(self, obs):
        """
        Translate the observation to matrix form
        """


        #return workload, block, candidate, space, block_mask, index_mask
        return


    def resolve_obs(self, obs):
        return

    def invoke_model(self, obs):
        # implement this module here for training
        # (to pick up state and action to record)
        workload, block, candidate, space, block_mask, index_mask = obs

        workload_inputs, block_inputs, candidate_inputs, space_inputs, \
        block_valid_mask, index_valid_mask = self.resolve_obs(obs)

        # invoke learning model
        block_probs, index_probs, block_acts, index_acts = \
            self.predict(workload_inputs, block_inputs, candidate_inputs, space_inputs,
                      block_valid_mask, index_valid_mask)

        return block_acts, index_acts, block_probs, index_probs, \
               workload_inputs, block_inputs, candidate_inputs, space_inputs, \
                block_valid_mask, index_valid_mask

    def get_action(self, obs):

        # parse observation
        workload, block, candidate, space, block_mask, index_mask = obs


        # invoking the learning model
        block_acts, index_acts, block_probs, index_probs, \
        workload_inputs, block_inputs, candidate_inputs, space_inputs, \
        block_valid_mask, index_valid_mask = self.invoke_model(obs)

        if sum(block_mask[0, :]) == 0 or sum(index_mask[0, :]) == 0:
            # no node is valid to assign
            return None, None

        # node_act should be valid
        assert block_mask[0, block_acts[0]] == 1

        # parse node action
        block_id = block_acts[0]

        index_id = index_acts[0]

        return block_id, index_id
