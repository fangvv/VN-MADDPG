from __future__ import division, print_function
import random
import scipy
import scipy.io
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import Environment_marl
import os
from replay_memory import ReplayMemory
import sys

my_config = tf.ConfigProto()
# my_config.gpu_options.allow_growth=True
my_config.gpu_options.allow_growth=False

# 定义CLASS Agent：Agent(object)，无输入参数，内容是一些算法参数，注意memory的实现方法是ReplayMemory
class Agent(object):
    def __init__(self, memory_entry_size):
        # self.discount = 1       # gamma
        self.discount = 0.99       # gamma
        # self.double_q = True
        self.double_q = False
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)


# ################## SETTINGS ######################
# 参数初始化：这部分直接写在代码中，没有函数，大概包括：地图属性（路口坐标，整体地图尺寸）、#车、#邻居、#RB、#episode，一些算法参数
# 对于地图参数 up_lanes / down_lanes / left_lanes / right_lanes 的含义，首先要了解本次所用的系统模型由3GPP TR 36.885的城市案例
# 给出，每条街有四个车道（正反方向各两个车道） ，车道宽3.5m，模型网格（road grid）的尺寸以黄线之间的距离确定，为433m*250m，
# 区域面积为1299m*750m。仿真中等比例缩小为原来的1/2（这点可以由 width 和 height 参数是 / 2 的看出来），
# 反映在车道的参数上就是在 lanes 中的 i / 2.0 。
'''
下面以 up_lanes 为例进行说明。在上图中我们可以看到，车道宽3.5m，所以将车视作质点的话，应该是在3.5m的车道中间移动的，
因此在 up_lanes 中 in 后面的 中括号里 3.5 需要 /2，第二项的3.5就是通向双车道的第二条车道的中间；
第三项 +250 就是越过建筑物的第一条同向车道，以此类推
'''
up_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]]

width = 750/2
height = 1298/2

IS_TRAIN = 1
IS_TEST = 1-IS_TRAIN

label = 'marl_model'

n_veh = 4
n_neighbor = 1
n_RB = n_veh

env = Environment_marl.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game()  # initialize parameters in env

n_episode = 2000
n_step_per_episode = int(env.time_slow/env.time_fast)   # 0.1/0.001 = 100
epsi_final = 0.01       # 探索最终值
epsi_anneal_length = int(0.8*n_episode)     # 探索退火长度
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4

######################################################
def get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([idx[0]])))
    # 这里有所有感兴趣的物理量：V2V_fast V2I_fast V2V_interference V2I_abs V2V_abs

# -----------------------------------------------------------
n_hidden_1 = 256
n_hidden_2 = 64
n_hidden_3 = 16
n_input = len(get_state(env=env))
n_output = n_RB * len(env.V2V_power_dB_List)        # 4 * 功率level

g = tf.Graph()
with g.as_default():
    # ============== Training network ========================
    x = tf.placeholder(tf.float32, [None, n_input]) # 输入

    w_1 = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3 = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4 = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

    b_1 = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
    b_2 = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
    b_3 = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
    b_4 = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w_1), b_1))
    layer_1_b = tf.layers.batch_normalization(layer_1)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, w_2), b_2))
    layer_2_b = tf.layers.batch_normalization(layer_2)
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, w_3), b_3))
    layer_3_b = tf.layers.batch_normalization(layer_3)
    y = tf.nn.relu(tf.add(tf.matmul(layer_3, w_4), b_4))
    g_q_action = tf.argmax(y, axis=1)

    # compute loss
    g_target_q_t = tf.placeholder(tf.float32, None, name="target_value")
    g_action = tf.placeholder(tf.int32, None, name='g_action')
    action_one_hot = tf.one_hot(g_action, n_output, 1.0, 0.0, name='action_one_hot')
    q_acted = tf.reduce_sum(y * action_one_hot, reduction_indices=1, name='q_acted')

    g_loss = tf.reduce_mean(tf.square(g_target_q_t - q_acted), name='g_loss')       # 求误差
    optim = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.95, epsilon=0.01).minimize(g_loss)    # 梯度下降

    # ==================== Prediction network ========================
    x_p = tf.placeholder(tf.float32, [None, n_input])

    w_1_p = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2_p = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3_p = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4_p = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

    b_1_p = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
    b_2_p = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
    b_3_p = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
    b_4_p = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

    layer_1_p = tf.nn.relu(tf.add(tf.matmul(x_p, w_1_p), b_1_p))
    layer_1_p_b = tf.layers.batch_normalization(layer_1_p)

    layer_2_p = tf.nn.relu(tf.add(tf.matmul(layer_1_p_b, w_2_p), b_2_p))
    layer_2_p_b = tf.layers.batch_normalization(layer_2_p)

    layer_3_p = tf.nn.relu(tf.add(tf.matmul(layer_2_p_b, w_3_p), b_3_p))
    layer_3_p_b = tf.layers.batch_normalization(layer_3_p)

    y_p = tf.nn.relu(tf.add(tf.matmul(layer_3_p_b, w_4_p), b_4_p))

    g_target_q_idx = tf.placeholder('int32', [None, None], 'output_idx')        # 输入，这是一个（n, 2）的list
    target_q_with_idx = tf.gather_nd(y_p, g_target_q_idx)                       # 提取首参的某几行/列

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


# 预测：predict(sess, s_t, ep, test_ep = False)，此函数用于驱动NN，生成动作action
def predict(sess, s_t, ep, test_ep=False):

    n_power_levels = len(env.V2V_power_dB_List)
    if np.random.rand() < ep and not test_ep:
        pred_action = np.random.randint(n_RB*n_power_levels)
    else:
        pred_action = sess.run(g_q_action, feed_dict={x: [s_t]})[0]
    return pred_action      #动作取值集合：{0,1,2...,15}，这里的action是一个int，但内涵了RB和power_level的信息，在本代码后面Training和Testing中都有出现


def q_learning_mini_batch(current_agent, current_sess):
    """ Training a sampled mini-batch """

    batch_s_t, batch_s_t_plus_1, batch_action, batch_reward = current_agent.memory.sample()

    if current_agent.double_q:  # double q-learning
        pred_action = current_sess.run(g_q_action, feed_dict={x: batch_s_t_plus_1})
        q_t_plus_1 = current_sess.run(target_q_with_idx, {x_p: batch_s_t_plus_1, g_target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})
        batch_target_q_t = current_agent.discount * q_t_plus_1 + batch_reward
    else:
        q_t_plus_1 = current_sess.run(y_p, {x_p: batch_s_t_plus_1})
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        batch_target_q_t = current_agent.discount * max_q_t_plus_1 + batch_reward

    _, loss_val = current_sess.run([optim, g_loss], {g_target_q_t: batch_target_q_t, g_action: batch_action, x: batch_s_t})
    return loss_val


def update_target_q_network(sess):
    """ Update target q network once in a while """

    sess.run(w_1_p.assign(sess.run(w_1)))
    sess.run(w_2_p.assign(sess.run(w_2)))
    sess.run(w_3_p.assign(sess.run(w_3)))
    sess.run(w_4_p.assign(sess.run(w_4)))

    sess.run(b_1_p.assign(sess.run(b_1)))
    sess.run(b_2_p.assign(sess.run(b_2)))
    sess.run(b_3_p.assign(sess.run(b_3)))
    sess.run(b_4_p.assign(sess.run(b_4)))


# --------------------------------------------------------------
agents = []
sesses = []
for ind_agent in range(n_veh * n_neighbor):  # initialize agents
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)

    sess = tf.Session(graph=g,config=my_config)
    sess.run(init)
    sesses.append(sess)

# ------------------------- Training -----------------------------
record_reward = np.zeros([n_episode*n_step_per_episode, 1])
record_loss = []
if __name__ == '__main__':
    for i_episode in range(n_episode):
        print("-------------------------")

        if i_episode < epsi_anneal_length:
            epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
        else:
            epsi = epsi_final
        if i_episode%100 == 0:      # 每100次更新一次位置、邻居、快衰、信道。
            env.renew_positions()   # update vehicle position
            env.renew_neighbor()
            env.renew_channel() # update channel slow fading
            env.renew_channels_fastfading() # update channel fast fading

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))    # 初始化demand time_limit active_links(全1)
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        sum_reward = 0
        epsi_V2I_Rate = 0
        epsi_V2V_Rate = 0
        epsi_V2V_success = 0
        # sum_action = np.zeros(8)

        for i_step in range(n_step_per_episode):                # range内是0.1/0.001 = 100
            time_step = i_episode*n_step_per_episode + i_step   # time_step是整体的step
            state_old_all = []
            action_all = []
            action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            for i in range(n_veh):  # 对每一个链路
                for j in range(n_neighbor):
                    state = get_state(env, [i, j], i_episode/(n_episode-1), epsi)   # 获取该链路的state【对于单个链路】
                    state_old_all.append(state)
                    action = predict(sesses[i*n_neighbor+j], state, epsi)   # 通过predict得到action（包含RB和POWER的信息）【对于单个链路】
                    action_all.append(action)

                    action_all_training[i, j, 0] = i  # chosen RB
                    action_all_training[i, j, 1] = int(np.floor(action / n_RB)) # power level   # 根据action得到action_all_trainging = [车，邻居，RB/power]【讲单个链路的内容存储起来】

            action_all_training_temp = action_all_training.copy()

            action_temp = action_all_training.copy()
            train_reward, V2I_Rate, V2V_Rate, V2V_success = env.act_for_training(action_temp)    # 通过action_for_training得到reward【这里是对于所有链路的】如果是sarl，则把计算reward的放到上面的for内，其他一样
            sum_reward += train_reward
            epsi_V2I_Rate += np.sum(V2I_Rate)
            epsi_V2V_success += V2V_success

            if i_episode < 10 or i_episode > n_episode - 100:
                print("Step: ", i_step, ", V2I rate: ", np.round(V2I_Rate, 2))
                print("V2V Rate             : ", np.round(V2V_Rate.reshape(4), 2), ", V2V success: ", V2V_success)
                print("Remaining V2V payload: ", np.round(env.demand.reshape(4), 2))



            env.renew_channels_fastfading()         # 更新快衰
            env.Compute_Interference(action_temp)   # 根据action计算干扰

            for i in range(n_veh):              # 使用for循环对每个链路
                for j in range(n_neighbor):
                    state_old = state_old_all[n_neighbor * i + j]
                    action = action_all[n_neighbor * i + j]
                    state_new = get_state(env, [i, j], i_episode/(n_episode-1), epsi)   # 计算新状态
                    agents[i * n_neighbor + j].memory.add(state_old, state_new, train_reward, action)  # add entry to this agent's memory将（state_old,state_new,train_reward,action)加入agent的memory中【所以说这里的memory每一条是对于单个链路的】

                    if i_episode>500:
                        if time_step % mini_batch_step == mini_batch_step-1:
                            loss_val_batch = q_learning_mini_batch(agents[i*n_neighbor+j], sesses[i*n_neighbor+j])
                            record_loss.append(loss_val_batch)
                            if i == 0 and j == 0:
                                print('step:', time_step, 'agent',i*n_neighbor+j, 'loss', loss_val_batch)
                        if time_step % target_update_step == target_update_step-1:
                            update_target_q_network(sesses[i*n_neighbor+j])
                            if i == 0 and j == 0:
                                print('Update target Q network...')

        print(
            "Episode：" + str(i_episode) + ", Explore：" + str(round(epsi, 4)) + ", Reward: " + str(round(sum_reward, 4)))
        print("Avg V2I rate: ", round(epsi_V2I_Rate / 100, 4), ", Avg V2V success: ", round(epsi_V2V_success / 100, 4))

# close sessions
for sess in sesses:
    sess.close()


