import numpy as np
import tensorflow as tf

import Environment_marl
from model_agent_maddpg import MADDPG
from replay_buffer import ReplayBuffer


def create_init_update(oneline_name, target_name, tau=0.99):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]

    return target_init, target_update


agent1_ddpg = MADDPG('agent1')
agent1_ddpg_target = MADDPG('agent1_target')

agent2_ddpg = MADDPG('agent2')
agent2_ddpg_target = MADDPG('agent2_target')

agent3_ddpg = MADDPG('agent3')
agent3_ddpg_target = MADDPG('agent3_target')

agent4_ddpg = MADDPG('agent4')
agent4_ddpg_target = MADDPG('agent4_target')

saver = tf.train.Saver()

agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1_actor', 'agent1_target_actor')
agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic')

agent2_actor_target_init, agent2_actor_target_update = create_init_update('agent2_actor', 'agent2_target_actor')
agent2_critic_target_init, agent2_critic_target_update = create_init_update('agent2_critic', 'agent2_target_critic')

agent3_actor_target_init, agent3_actor_target_update = create_init_update('agent3_actor', 'agent3_target_actor')
agent3_critic_target_init, agent3_critic_target_update = create_init_update('agent3_critic', 'agent3_target_critic')

agent4_actor_target_init, agent4_actor_target_update = create_init_update('agent4_actor', 'agent4_target_actor')
agent4_critic_target_init, agent4_critic_target_update = create_init_update('agent4_critic', 'agent4_target_critic')


def get_agents_action(o_n, sess, noise_rate):
    agent1_action = agent1_ddpg.action(state=[o_n[0]], sess=sess) + np.random.randn(2) * noise_rate
    agent2_action = agent2_ddpg.action(state=[o_n[1]], sess=sess) + np.random.randn(2) * noise_rate
    agent3_action = agent3_ddpg.action(state=[o_n[2]], sess=sess) + np.random.randn(2) * noise_rate
    agent4_action = agent4_ddpg.action(state=[o_n[3]], sess=sess) + np.random.randn(2) * noise_rate

    return agent1_action, agent2_action, agent3_action, agent4_action

def train_agent(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess, other_actors):
    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch = agent_memory.sample(32)

    act_batch = total_act_batch[:, 0, :]
    other_act_batch = np.hstack([total_act_batch[:, 1, :], total_act_batch[:, 2, :], total_act_batch[:, 3, :]])

    obs_batch = total_obs_batch[:, 0, :]

    next_obs_batch = total_next_obs_batch[:, 0, :]
    next_other_actor1_o = total_next_obs_batch[:, 1, :]
    next_other_actor2_o = total_next_obs_batch[:, 2, :]
    next_other_actor3_o = total_next_obs_batch[:, 3, :]
    # 获取下一个情况下另外三个agent的行动
    next_other_action = np.hstack([other_actors[0].action(next_other_actor1_o, sess), other_actors[1].action(next_other_actor2_o, sess), other_actors[2].action(next_other_actor3_o, sess)])
    target = rew_batch.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=next_obs_batch, action=agent_ddpg.action(next_obs_batch, sess),
                                                                     other_action=next_other_action, sess=sess)
    agent_ddpg.train_actor(state=obs_batch, other_action=other_act_batch, sess=sess)
    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess)

    sess.run([agent_actor_target_update, agent_critic_target_update])

def get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """
    # include V2I/V2V fast_fading, V2V interference, V2I/V2V 信道信息（PL+shadow）,
    # 剩余时间, 剩余负载

    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    # 这里有所有感兴趣的物理量：V2V_fast V2I_fast V2V_interference V2I_abs V2V_abs、剩余时间、剩余负载


if __name__ == '__main__':
    n_veh = 4
    n_neighbor = 1
    n_RB = n_veh
    env = Environment_marl.Environ(n_veh, n_neighbor)
    env.new_random_game()  # initialize parameters in env

    # n_episode = 3000
    n_episode = 2000
    n_step_per_episode = int(env.time_slow / env.time_fast)  # 0.1/0.001 = 100
    epsi_final = 0.01  # 探索最终值          ##13
    epsi_anneal_length = int(0.8 * n_episode)  # 探索退火长度
    mini_batch_step = n_step_per_episode
    target_update_step = n_step_per_episode * 4

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run([agent1_actor_target_init, agent1_critic_target_init,
              agent2_actor_target_init, agent2_critic_target_init,
              agent3_actor_target_init, agent3_critic_target_init,
              agent4_actor_target_init, agent4_critic_target_init])

    memory_size = 50000
    agent1_memory = ReplayBuffer(memory_size)
    agent2_memory = ReplayBuffer(memory_size)
    agent3_memory = ReplayBuffer(memory_size)
    agent4_memory = ReplayBuffer(memory_size)

    for i_episode in range(n_episode):
        print("-------------------------")

        if i_episode < epsi_anneal_length:
            var = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
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
            # 初始化state_old_all,action_all action_all_training
            time_step = i_episode * n_step_per_episode + i_step  # time_step是整体的step
            state_old_all = []
            action_all = []
            states = []
            action_all_training = np.zeros([n_veh, n_neighbor, 2])   ####11
            for i in range(n_veh):  # 对每一个链路
                for j in range(n_neighbor):
                    state = get_state(env, [i, j], i_episode/(n_episode-1), var)   # 获取该链路的state【对于单个链路】
                    states.append(state)
                    state_old_all.append(state)

            agent1_action, agent2_action, agent3_action, agent4_action = np.clip(np.array(get_agents_action(states, sess, var)), -1, 1)  # 通过predict得到action（包含RB和POWER的信息）【对于单个链路】

            agent1_action[0][0] = 0
            agent2_action[0][0] = 1
            agent3_action[0][0] = 2
            agent4_action[0][0] = 3
            action = [agent1_action.tolist(), agent2_action.tolist(), agent3_action.tolist(), agent4_action.tolist()]

            action_all.append(agent1_action)
            action_all.append(agent2_action)
            action_all.append(agent3_action)
            action_all.append(agent4_action)

            for i in range(n_veh):  # 对每一个链路
                for j in range(n_neighbor):
                    action_all_training[i, j, 0] = action[i][j][0]  # chosen RB
                    action_all_training[i, j, 1] = (action[i][j][1] + 1) * 0.1  # power level

            # All agents take actions simultaneously, obtain shared reward, and update the environment.
            action_temp = action_all_training.copy()
            train_reward, V2I_Rate, V2V_Rate, V2V_success = env.act_for_training(action_temp)  # 通过action_for_training得到reward【这里是对于所有链路的】如果是sarl，则把计算reward的放到上面的for内，其他一样
            sum_reward += train_reward
            epsi_V2I_Rate += np.sum(V2I_Rate)
            # epsi_V2V_Rate += V2V_Rate
            epsi_V2V_success += V2V_success

            if i_episode < 10 or i_episode > n_episode - 100:
                print("Step: ", i_step, ", V2I rate: ", np.round(V2I_Rate, 2))
                print("V2V Rate             : ", np.round(V2V_Rate.reshape(4), 2), ", V2V success: ", V2V_success)
                print("Remaining V2V payload: ", np.round(env.demand.reshape(4), 2))

            env.renew_channels_fastfading()  # 更新快衰
            env.Compute_Interference(action_temp)  # 根据action计算干扰

            state_new_all = []
            for i in range(n_veh):              # 使用for循环对每个链路
                for j in range(n_neighbor):
                    state_old = state_old_all[n_neighbor * i + j]
                    action = action_all[n_neighbor * i + j]
                    state_new = get_state(env, [i, j], i_episode/(n_episode-1), var)   # 计算新状态
                    state_new_all.append(state_new)

            agent1_memory.add(np.vstack([state_old_all[0], state_old_all[1], state_old_all[2], state_old_all[3]]),
                              np.vstack([agent1_action[0], agent2_action[0], agent3_action[0], agent4_action[0]]),
                              train_reward,
                              np.vstack([state_new_all[0], state_new_all[1], state_new_all[2], state_new_all[3]]))  # add entry to this agent's memory将（state_old,state_new,train_reward,action)加入agent的memory中【所以说这里的memory每一条是对于单个链路的】

            agent2_memory.add(np.vstack([state_old_all[1], state_old_all[2], state_old_all[3], state_old_all[0]]),
                              np.vstack([agent2_action[0], agent3_action[0], agent4_action[0], agent1_action[0]]),
                              train_reward,
                              np.vstack([state_new_all[1], state_new_all[2], state_new_all[3], state_new_all[0]]))

            agent3_memory.add(np.vstack([state_old_all[2], state_old_all[3], state_old_all[0], state_old_all[1]]),
                              np.vstack([agent3_action[0], agent4_action[0], agent1_action[0], agent2_action[0]]),
                              train_reward,
                              np.vstack([state_new_all[2], state_new_all[3], state_new_all[0], state_new_all[1]]))

            agent4_memory.add(np.vstack([state_old_all[3], state_old_all[0], state_old_all[1], state_old_all[2]]),
                              np.vstack([agent4_action[0], agent1_action[0], agent2_action[0], agent3_action[0]]),
                              train_reward,
                              np.vstack([state_new_all[3], state_new_all[0], state_new_all[1], state_new_all[2]]))

            if i_episode > 500:
                train_agent(agent1_ddpg, agent1_ddpg_target, agent1_memory, agent1_actor_target_update,
                            agent1_critic_target_update, sess, [agent2_ddpg_target, agent3_ddpg_target, agent4_ddpg_target])

                train_agent(agent2_ddpg, agent2_ddpg_target, agent2_memory, agent2_actor_target_update,
                            agent2_critic_target_update, sess, [agent3_ddpg_target, agent4_ddpg_target, agent1_ddpg_target])

                train_agent(agent3_ddpg, agent3_ddpg_target, agent3_memory, agent3_actor_target_update,
                            agent3_critic_target_update, sess, [agent4_ddpg_target, agent1_ddpg_target, agent2_ddpg_target])

                train_agent(agent4_ddpg, agent4_ddpg_target, agent4_memory, agent4_actor_target_update,
                            agent4_critic_target_update, sess, [agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target])

        print("Episode：" + str(i_episode) + ", Explore：" + str(round(var, 4)) + ", Reward: " + str(round(sum_reward, 4)))
        print("Avg V2I rate: ", round(epsi_V2I_Rate/100, 4), ", Avg V2V success: ", round(epsi_V2V_success/100, 4))

    sess.close()
