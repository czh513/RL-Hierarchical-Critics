# Hierarchical-Critics-Assignment-for-Multi-agent-Reinforcement-Learning
Multi-agent Reinforcement Learning

The Unity platform, a new open-source toolkit, has been used for creating and interacting with simulation environments. To be specific, the Unity Machine Learning Agents Toolkit (ML- Agents Toolkit) [Juliani et al., 2018] is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. 
It supports dynamic multi-agent interaction and agents can be trained using reinforcement learning through a simple-to-use Python API.

This code is modified by the baseline algorithm in the ML- Agents Toolkit, named Asynchronous Advantage Actor- Critic (A3C) with Proximal Policy Optimization (PPO) algorithm, A3C-PPO. We investigate the use of global information to speed up the learning process and increase the cumulative rewards of multiagent re- inforcement learning (MARL) tasks. Within the actor-critic MARL, we introduce multiple cooperative critics from two levels of the hierarchy and propose a hierarchical critic-based multi-agent reinforcement learning algorithm. 
In our approach, the agent is allowed to receive information from local and global critics in a competition task. The agent not only receives low-level details but also consider coordination from high levels that receiving global information to increase operation skills. Here, we define multiple cooperative critics in the top-bottom hierarchy, called the Hierarchical Critics Assignment (HCA) framework. It has been used in a tennis scenario and it can extend to complex environment.

Before using this code in Unity, please download the ML- Agents Toolkit first.
Then, please remember to:
1. replace 'HCA-A3C-PPO' folder to 'ppo' folder.
2. replace one of 'trainer_xxx.py' to 'Trainer.py' in ppo folder.  (x indicates a random name)
3. modify train_config.yaml for the Brain.
4. modify the number of observation in Brain.

Note: we used the tennis scenario for training process, so please look at the four types of observation spaces below as an example:
1. trainer_HRL: 8 vectors (t, t-1).
2. trainer_HRL_Ob: 10 vectors.
3. trainer_HRL_Obo: 10 vectors (t, t-1).
4. trainer_time_HRL_Obo: 10 vectors with 5-time steps interval.

Reference:
1. Z. Cao and C.T Lin (2019) Hierarchical Critics Assignment for Multi-agent Reinforcement Learning.
https://arxiv.org/pdf/1902.03079.pdf
