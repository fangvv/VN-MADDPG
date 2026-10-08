# VN-MADDPG

[![GitHub](https://img.shields.io/badge/Project_Homepage-181717?logo=github)](https://github.com/fangvv/VN-MADDPG) — [https://github.com/fangvv/VN-MADDPG](https://github.com/fangvv/VN-MADDPG)

This is the source code for our paper: **基于多智能体深度强化学习的车联网通信资源分配优化**. A brief introduction of this work is as follows:

> 无线网络的高速发展为车联网提供了更好的支持，但是如何为高速移动车辆提供更高质量的服务仍然是一个挑战.通过分析多个车对车（Vehicle-to-Vehicle， V2V）链路重用的车对基础设施（Vehicle-to-Infrastructure， V2I）链路占用的频谱，研究了基于连续动作空间的多智能体深度强化学习的车联网中的频谱共享问题.车辆高移动性带来的信道的快速变化为集中式管理网络资源带来了局限性，因此将资源共享建模为多智能体深度强化学习问题，提出一种基于分布式执行的多智能体深度确定性策略梯度（Multi-Agent Deep Deterministic Policy Gradient， MADDPG）算法.每个智能体与车联网环境进行交互并观察到自己的局部状态，均获得一个共同的奖励，通过汇总其他智能体的动作集中训练Critic网络，从而改善各个智能体选取的功率控制.通过设计奖励函数和训练机制，多智能体算法可以实现分布式资源分配，有效提高了V2I链路的总容量和V2V链路的传输速率.

> The rapid development of wireless networks has provided better support for the Internet of the Connected Vehicle, but ensuring higher quality services for high-speed moving vehicles remains a challenge. By analyzing the spectrum occupancy of Vehicle-to-Infrastructure (V2I) links reused by multiple Vehicle-to-Vehicle (V2V) links, this study investigates the spectrum sharing problem in the Internet of the Connected Vehicle using multi-agent deep reinforcement learning based on continuous action spaces. The rapid channel variations caused by the high mobility of vehicles pose limitations for centralized network resource management. Therefore, resource sharing is modeled as a multi-agent deep reinforcement learning problem, and a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm based on distributed execution is proposed. Each agent interacts with the Internet of the Connected Vehicle environment and observes its local state, receiving a shared reward. The Critic network is centrally trained by aggregating the actions of other agents, thereby improving the power control selections of individual agents. By designing reward functions and training mechanisms, the multi-agent algorithm achieves distributed resource allocation, effectively enhancing the total capacity of V2I links and the transmission rate of V2V links.

This work is published in Journal of Beijing Jiaotong University. Click [here](http://jdxb.bjtu.edu.cn/CN/abstract/abstract3830.shtml) for our paper online.

## Required software

- TensorFlow 1.X
- NumPy
- SciPy

## Project Structure

```
VN-MADDPG/
├── MADDPG/                          # Main algorithm: Multi-Agent DDPG
│   ├── Environment_marl.py          # Environment simulator: channel models, vehicle model, interference & reward
│   ├── model_agent_maddpg.py        # MADDPG Agent: Actor-Critic network definitions
│   ├── maddpg.py                    # Main training script: training loop, target network updates
│   ├── replay_buffer.py             # Experience replay buffer (uniform + prioritized)
│   ├── segment_tree.py              # Segment tree (Sum/Min) for prioritized replay
│   └── replay_memory.py             # Legacy replay memory (used by MADQN)
├── MADQN/                           # Baseline: Multi-Agent DQN (discrete action space)
│   ├── Environment_marl.py
│   ├── madqn.py                     # MADQN implementation: discrete power levels + DQN
│   └── replay_memory.py
├── SAMADDPG/                        # Baseline: Single-Agent DDPG (independent learning)
│   ├── Environment_marl.py
│   └── DDPG_method.py               # Single-agent DDPG: no Critic information sharing
├── Random/                          # Baseline: random policy
│   ├── Environment_marl.py
│   └── random.py
└── README.md
```

## Core Modules

### Environment (`Environment_marl.py`)

The environment is built on the 3GPP TR 36.885 urban grid model. The simulated area is 1299 m × 750 m (scaled down to 1/2) with a Manhattan grid layout. Each street has 4 lanes (2 per direction), with lane width of 3.5 m.

**Core classes:**

| Class | Description |
|---|---|
| `V2Vchannels` | V2V channel model: LOS/NLOS path loss + shadow fading (std = 3 dB) |
| `V2Ichannels` | V2I channel model: path loss + shadow fading (std = 8 dB), BS at grid center |
| `Vehicle` | Vehicle instance: position, direction, velocity, neighbor list, destination list |
| `Environ` | Main environment class: integrates channels, vehicles, interference, and reward computation |

**Key parameters:**

| Parameter | Value | Description |
|---|---|---|
| `n_Veh = 4` | – | Number of V2V transmitter vehicles |
| `n_neighbor = 1` | – | Communication neighbors per vehicle |
| `n_RB = 4` | – | Number of orthogonal resource blocks |
| `V2I_power_dB = 23` | 23 dBm | V2I link transmit power |
| `sig2_dB = -114` | -114 dBm | Noise power |
| `time_slow = 0.1 s` | 100 ms | Large-scale fading / position update period |
| `time_fast = 0.001 s` | 1 ms | Fast fading / decision period |
| `demand_size` | 8480 bits | V2V payload per 100 ms |
| `bandwidth = 1e6` | 1 MHz | Bandwidth per resource block |

**Key methods:**

- `new_random_game()` — Initialize vehicles (randomly placed in 4 directions), channels, and shadow fading.
- `renew_positions()` — Update all vehicle positions based on direction and velocity.
- `renew_neighbor()` — Update the neighbor list for each vehicle based on distance.
- `renew_channel()` / `renew_channels_fastfading()` — Update large-scale / small-scale channel fading.
- `Compute_Interference(action)` — Compute V2V inter-link interference based on all agents' actions.
- `act_for_training(action)` — Execute one MDP step: compute V2I rates, V2V rates, and transmission success, returning the shared reward.

### State Space & Action Space

**State space (per agent, 33 dimensions):**

| Index | Feature | Dimension |
|---|---|---|
| 0–3 | V2I fast fading across all RBs for this agent | `n_RB = 4` |
| 4–19 | V2V fast fading across all RBs for all V2V links | `n_Veh × n_RB = 16` |
| 20–23 | V2V interference across all RBs | `n_RB = 4` |
| 24 | V2I large-scale channel gain | 1 |
| 25–28 | V2V large-scale channel gain for all V2V links | `n_Veh = 4` |
| 29 | Remaining time ratio | 1 |
| 30 | Remaining payload ratio | 1 |
| 31–32 | Episode progress & exploration rate | 2 |

**Action space (per agent, 2 continuous dimensions):**

| Dimension | Meaning | Range |
|---|---|---|
| 0 | Selected resource block (RB) index | `[-1, 1]` mapped to `{0, 1, 2, 3}` |
| 1 | Transmit power level | `[-1, 1]` mapped to continuous power |

**Reward:** Shared reward = total V2I link capacity + V2V transmission success rate, balancing the performance of both link types.

### MADDPG Agent (`model_agent_maddpg.py`)

Based on the Centralized Training with Decentralized Execution (CTDE) framework. Each V2V link corresponds to one agent (4 agents in total), each with independent Actor and Critic networks.

**Network architecture:**

| Network | Layers | Output | Description |
|---|---|---|---|
| Actor | 256(ReLU+LN) → 64(ReLU+LN) → 16(ReLU+LN) → 2(Tanh) | `[-1, 1]` continuous action | Decentralized: only uses local observation |
| Critic | `[state, action, other_actions]` → 256(ReLU+LN) → 64(ReLU+LN) → 16(ReLU+LN) → 1 | Q-value | Centralized: aggregates all agents' actions |

> LN = Layer Normalization (`tf.contrib.layers.layer_norm`), used to stabilize training.

**Key hyperparameters:**

| Parameter | Value | Description |
|---|---|---|
| `LR_A = 0.001` | – | Actor learning rate (Adam optimizer) |
| `LR_C = 0.001` | – | Critic learning rate (Adam optimizer) |
| `γ = 0.999` | – | Discount factor |
| `τ = 0.99` | – | Soft target update: `θ_target = τ·θ_target + (1-τ)·θ_online` |
| `memory_size = 50000` | – | Experience replay buffer size (per agent) |
| `batch_size = 32` | – | Training batch size |
| `n_episode = 2000` | – | Number of training episodes |
| `epsi_final = 0.01` | – | Final exploration noise, linearly annealed from 1.0 |

### Training Loop (`maddpg.py`)

The main training loop in `maddpg.py` follows these steps:

1. **Initialization** — Create 4 agents (agent1–agent4), each with online and target Actor/Critic networks. Initialize target network parameters.
2. **Environment reset** — Every 100 steps, update vehicle positions, neighbor relationships, and large-scale channels. Fast fading is updated every step.
3. **State observation & action selection** — Each agent obtains its local observation, produces an action via the Actor network, and adds Gaussian exploration noise (annealed over episodes), clipped to `[-1, 1]`.
4. **Centralized training** — Training procedure per agent:
   - Sample a batch from its own replay buffer.
   - Compute next-step joint actions using target Actor + other agents' Actors.
   - Compute target Q-value via target Critic: `target = r + γ × Q_target(s', a', other_a')`.
   - Train Critic network (minimize TD error).
   - Train Actor network (maximize Q-value from Critic).
   - Soft-update target networks.
5. **Decentralized execution** — At test time, each agent decides using only its local observation through the Actor network, requiring no global information.

### Experience Replay (`replay_buffer.py`)

Two replay mechanisms are provided:

| Class | Description |
|---|---|
| `ReplayBuffer` | Uniform replay: circular buffer with random sampling |
| `PrioritizedReplayBuffer` | Prioritized experience replay: TD-error-based priority sampling with importance sampling weights |

`segment_tree.py` implements SumSegmentTree and MinSegmentTree, providing `O(log N)` sampling and update operations for prioritized replay.

### Baseline Methods

| Method | Directory | Description |
|---|---|---|
| **MADQN** | `MADQN/` | Discretizes continuous power control into discrete power levels, using DQN in a discrete action space. Dual-network architecture (eval/target) with hard updates. Same state space as MADDPG, output is Q-values over discrete actions. |
| **SAMADDPG** | `SAMADDPG/` | Single-Agent DDPG: each agent independently learns with DDPG, without Critic access to other agents' actions. Validates the necessity of Critic information sharing in CTDE. Actor: 64→16→4→2; Critic: 64→16→4→1. |
| **Random** | `Random/` | Randomly selects RB and power, serving as a lower-bound baseline. |

## Usage

```bash
# Install dependencies (TensorFlow 1.x required)
pip install tensorflow==1.14.0 numpy scipy

# Run MADDPG (main algorithm)
cd MADDPG
python maddpg.py

# Run MADQN baseline
cd MADQN
python madqn.py

# Run SAMADDPG baseline
cd SAMADDPG
python DDPG_method.py

# Run Random baseline
cd Random
python random.py
```

## ⭐ Star

**If you find this work useful for your research, please consider giving this repository a ⭐ star. Your support is greatly appreciated!**

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
