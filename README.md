# VN-MADDPG

This is the source code for our paper: **基于多智能体深度强化学习的车联网通信资源分配优化**. A brief introduction of this work is as follows:

> 无线网络的高速发展为车联网提供了更好的支持，但是如何为高速移动车辆提供更高质量的服务仍然是一个挑战.通过分析多个车对车（Vehicle-to-Vehicle， V2V）链路重用的车对基础设施（Vehicle-to-Infrastructure， V2I）链路占用的频谱，研究了基于连续动作空间的多智能体深度强化学习的车联网中的频谱共享问题.车辆高移动性带来的信道的快速变化为集中式管理网络资源带来了局限性，因此将资源共享建模为多智能体深度强化学习问题，提出一种基于分布式执行的多智能体深度确定性策略梯度（Multi-Agent Deep Deterministic Policy Gradient， MADDPG）算法.每个智能体与车联网环境进行交互并观察到自己的局部状态，均获得一个共同的奖励，通过汇总其他智能体的动作集中训练Critic网络，从而改善各个智能体选取的功率控制.通过设计奖励函数和训练机制，多智能体算法可以实现分布式资源分配，有效提高了V2I链路的总容量和V2V链路的传输速率.

> The rapid development of wireless networks has provided better support for the Internet of the Connected Vehicle, but ensuring higher quality services for high-speed moving vehicles remains a challenge. By analyzing the spectrum occupancy of Vehicle-to-Infrastructure (V2I) links reused by multiple Vehicle-to-Vehicle (V2V) links, this study investigates the spectrum sharing problem in the Internet of the Connected Vehicle using multi-agent deep reinforcement learning based on continuous action spaces. The rapid channel variations caused by the high mobility of vehicles pose limitations for centralized network resource management. Therefore, resource sharing is modeled as a multi-agent deep reinforcement learning problem, and a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm based on distributed execution is proposed. Each agent interacts with the Internet of the Connected Vehicle environment and observes its local state, receiving a shared reward. The Critic network is centrally trained by aggregating the actions of other agents, thereby improving the power control selections of individual agents. By designing reward functions and training mechanisms, the multi-agent algorithm achieves distributed resource allocation, effectively enhancing the total capacity of V2I links and the transmission rate of V2V links.

本文发表在北京交通大学学报，[链接](http://jdxb.bjtu.edu.cn/CN/abstract/abstract3830.shtml)

## Citation

	@article{方维维2022基于多智能体深度强化学习的车联网通信资源分配优化,
		title={基于多智能体深度强化学习的车联网通信资源分配优化},
		author={方维维 and 王云鹏 and 张昊 and 孟娜},
		journal={北京交通大学学报},
		volume={46},
		number={2},
		pages={64--72},
		year={2022}
	}
	
## Stargazers over time
[![Stargazers over time](https://starchart.cc/fangvv/VN-MADDPG.svg)](https://starchart.cc/fangvv/VN-MADDPG)

## For more

We have another work on [DDPG](https://github.com/fangvv/UAV-DDPG) for your reference, and you can simply use Ray for implementing [DRL algorithms](https://github.com/ray-project/ray/tree/master/rllib/algorithms) now.

## 不足甚多，时日已久，仅供参考，无法提供更多支持

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
