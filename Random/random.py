"""
random
"""

import numpy as np
import Environment_marl

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


###############################  training  ####################################
file_name = "random_1MB.txt"
if __name__ == '__main__':
    np.random.seed(1)

    n_veh = 4
    n_neighbor = 1
    n_RB = n_veh
    env = Environment_marl.Environ(n_veh, n_neighbor)
    env.new_random_game()  # initialize parameters in env

    n_episode = 100
    n_step_per_episode = int(env.time_slow / env.time_fast)  # 0.1/0.001 = 100
    epsi_final = 0.01  # 探索最终值          ##13
    epsi_anneal_length = int(0.8 * n_episode)  # 探索退火长度

    action_all_training = np.zeros([n_veh, n_neighbor, 2])
    time_step = 0
    for i_episode in range(n_episode):
        print("-------------------------")
        with open(file_name, 'a') as file_obj:
            file_obj.write("\n-------------------------")
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
            i = int(np.floor(remainder / n_neighbor))
            j = remainder % n_neighbor

            state = get_state(env, [i, j], i_step/(n_step_per_episode-1), var)
            action = np.random.uniform(-1, 1, 1)
            action = action.tolist()
            action_all = action

            action_all_training[i, j, 0] = i  # chosen RB
            action_all_training[i, j, 1] = (action[0]+1) * 0.1  # power level   # 根据action得到action_all_trainging = [车，邻居，RB/power]【讲单个链路的内容存储起来】
            # All agents take actions simultaneously, obtain shared reward, and update the environment.
            # 所有代理同时采取行动，获得共享奖励，并更新环境。
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

                with open(file_name, 'a') as file_obj:
                    file_obj.write("\nStep: " + '{:d}'.format(i_step) + ", V2I rate: " + str(np.round(V2I_Rate, 2)))
                    file_obj.write("\nV2V Rate             : " + str(np.round(V2V_Rate.reshape(4), 2)) + ", V2V success: " + '{:.2f}'.format(V2V_success))
                    file_obj.write("\nRemaining V2V payload: " + str(np.round(env.demand.reshape(4), 2)))

            env.renew_channels_fastfading()  # 更新快衰
            env.Compute_Interference(action_temp)  # 根据action计算干扰


        print(
            "Episode：" + str(i_episode) + ", Explore：" + str(round(var, 4)) + ", Reward: " + str(round(sum_reward, 4)))
        print("Avg V2I rate: ", round(epsi_V2I_Rate / 100, 4), ", Avg V2V success: ", round(epsi_V2V_success / 100, 4))

        with open(file_name, 'a') as file_obj:
            file_obj.write("\nEpisode: " + '{:d}'.format(i_episode) + ", Reward: " +  str(round(sum_reward, 4)))
            file_obj.write("\nAvg V2I rate: " + str(round(epsi_V2I_Rate / 100, 4)) + ", Avg V2V success: " + str(round(epsi_V2V_success / 100, 4)))

