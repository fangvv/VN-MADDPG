"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.

Using:
tensorflow 1.14.0
gym 0.15.3
"""

import tensorflow as tf
import numpy as np
import Environment_marl

#####################  hyper parameters  ####################
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
# GAMMA = 0.001  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 50000
BATCH_SIZE = 32
# BATCH_SIZE = 4
OUTPUT_GRAPH = False


# Runs policy for X episodes and returns average reward
def get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """
    # include V2I/V2V fast_fading, V2V interference, V2I/V2V 信道信息（PL+shadow）,
    # 剩余时间, 剩余负载

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([i])))
    # 这里有所有感兴趣的物理量：V2V_fast V2I_fast V2V_interference V2I_abs V2V_abs、剩余时间、剩余负载


###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # memory里存放当前和下一个state，动作和奖励
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')  # 输入
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            self.a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, self.a_, scope='target', trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        # print(s[np.newaxis, :])
        temp = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        return temp[0]

    def learn(self):
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        loss = self.sess.run(self.td_error, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        return loss

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 64, activation=tf.nn.relu, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 16, activation=tf.nn.relu, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 4, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[1], name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            # n_l1 = 256
            n_l1 = 64
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 16, activation=tf.nn.relu, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 4, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################
if __name__ == '__main__':
    np.random.seed(1)
    tf.set_random_seed(1)

    n_veh = 4
    n_neighbor = 1
    n_RB = n_veh
    env = Environment_marl.Environ(n_veh, n_neighbor)
    env.new_random_game()  # initialize parameters in env

    n_episode = 2000
    n_step_per_episode = int(env.time_slow / env.time_fast)  # 0.1/0.001 = 100
    epsi_final = 0.01  # 探索最终值          ##13
    epsi_anneal_length = int(0.8 * n_episode)  # 探索退火长度

    a_dim = 1
    # s_dim = 33
    s_dim = 32
    a_bound = [-1, 1]
    ddpg = DDPG(a_dim, s_dim, a_bound)
    var = 0.8

    action_all_training = np.zeros([n_veh, n_neighbor, 2])
    time_step = 0

    for i_episode in range(n_episode):
        print("-------------------------")
        # 根据i确定epsi（递增->不变）
        if i_episode < epsi_anneal_length:
            var = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode    #3
        else:
            var = epsi_final

        if i_episode % 100 == 0:      # 每100次更新一次位置、邻居、快衰、信道。
            env.renew_positions()   # update vehicle position
            env.renew_neighbor()
            env.renew_channel()     # update channel slow fading
            env.renew_channels_fastfading()     # update channel fast fading

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))  # 初始化demand time_limit active_links(全1)
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        sum_reward = 0
        epsi_V2I_Rate = 0
        epsi_V2V_Rate = 0
        epsi_V2V_success = 0

        for i_step in range(n_step_per_episode):  # range内是0.1/0.001 = 100
            time_step = i_episode * n_step_per_episode + i_step
            remainder = time_step % (n_veh * n_neighbor)

            state_old_all = []
            action_all = []
            states = []
            for i in range(n_veh):  # 对每一个链路
                for j in range(n_neighbor):
                    state = get_state(env, [i, j], i_episode / (n_episode - 1), var)  # 获取该链路的state【对于单个链路】
                    states.append(state)
                    state_old_all.append(state)

                    if i == remainder:
                        action_t = ddpg.choose_action(state)
                        action_t = np.clip(action_t + np.random.randn(1) * var, 0, 1)  # 加个噪声
                    else:
                        action_t = np.random.rand(1) * 2 - 1
                    action_all.append(action_t.tolist())

                    action_all_training[i, j, 0] = i  # chosen RB
                    action_all_training[i, j, 1] = (action_all[i][j] + 1) * 0.1  # power level

            action_temp = action_all_training.copy()
            train_reward, V2I_Rate, V2V_Rate, V2V_success = env.act_for_training(action_temp)  # 通过action_for_training得到reward【这里是对于所有链路的】如果是sarl，则把计算reward的放到上面的for内，其他一样
            sum_reward += train_reward
            epsi_V2I_Rate += np.sum(V2I_Rate)
            epsi_V2V_success += V2V_success

            if i_episode < 10 or i_episode > n_episode - 100:
                print("Step: ", i_step, ", V2I rate: ", np.round(V2I_Rate, 2))
                print("V2V Rate             : ", np.round(V2V_Rate.reshape(4), 2), ", V2V success: ", V2V_success)
                print("Remaining V2V payload: ", np.round(env.demand.reshape(4), 2))

            env.renew_channels_fastfading()  # 更新快衰
            env.Compute_Interference(action_temp)  # 根据action计算干扰

            state_new_all = []
            for i in range(n_veh):  # 使用for循环对每个链路
                for j in range(n_neighbor):
                    state_old = state_old_all[n_neighbor * i + j]
                    action = action_all[n_neighbor * i + j]
                    state_new = get_state(env, [i, j], i_episode / (n_episode - 1), var)  # 计算新状态
                    state_new_all.append(state_new)
            ddpg.store_transition(states[remainder], action_all[remainder], train_reward, state_new_all[remainder])  # add entry to this agent's memory

            if ddpg.pointer > MEMORY_CAPACITY:
 
                loss = ddpg.learn()

        print(
            "Episode：" + str(i_episode) + ", Explore：" + str(round(var, 4)) + ", Reward: " + str(round(sum_reward, 4)))
        print("Avg V2I rate: ", round(epsi_V2I_Rate / 100, 4), ", Avg V2V success: ", round(epsi_V2V_success / 100, 4))

