from __future__ import unicode_literals
import gym
import gym_gridworld

import optuna
import random
import openpyxl
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

#π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨
print(f'=========================================New Start=========================================')
# GPU μ λ³΄λ λ°μΌλ©΄μ GPU μ€μ νκΈ°
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
#π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨

env = gym.make('gridworld-v0')


env.seed(100)
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

#π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden1, hidden2): # stateκ° μΈν, κ·Έμ λν Qκ° κ³μ°μ΄ μμν
        super(QNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                                    nn.Linear(input_size, hidden1),
                                    nn.ReLU())
                                    
        self.layer2 = nn.Sequential(
                                    nn.Linear(hidden1, hidden2),
                                    nn.ReLU() )

        self.layer3 = nn.Sequential(    nn.Linear(hidden2, output_size)  )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) #            tensor([ 0.0757, -0.0513], grad_fn=) ννμ μΆλ ₯μ΄ λμ΄.
#                                        torch.float32
        return x


#π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨

class ReplayBuffer:

    # state, action, λ± λ³λ‘ μ μ₯ ν  λΉ array μμ±
    def __init__(self, obs_dim, act_dim, buff_size):        # obs_dim=4, act_dim=1, size=100_000
        '''
        self.state_buff = np.zeros([buff_size, obs_dim])    # μλμ np.zeros() μ€λͺ μ°Έμ‘° [64, 4]
        self.action_buff = np.zeros([buff_size, 1])
        self.reward_buff = np.zeros([buff_size, 1], dtype=np.float32)
        self.next_state_buff = np.zeros([buff_size, obs_dim])
        self.done_buff = np.zeros([buff_size, 1], dtype=np.float32)'''

        self.state_buff = torch.zeros([buff_size, obs_dim], dtype=torch.float32).to(device)    # μλμ np.zeros() μ€λͺ μ°Έμ‘° [64, 4]
        self.action_buff = torch.zeros([buff_size, 1], dtype=torch.int64).to(device)
        self.reward_buff = torch.zeros([buff_size, 1], dtype=torch.float32).to(device)
        self.next_state_buff = torch.zeros([buff_size, obs_dim], dtype=torch.float32).to(device)
        self.done_buff = torch.zeros([buff_size, 1], dtype=torch.float32).to(device)

        # self.stateμ κ²½μ°, [64, 4]μ ν¬κΈ°μΈλ°, state νλλ [0.3233, 2.3241, -0.3233, -2.3241]μ ννλ‘ μκ²Όμ.
        # μ  state νλλ₯Ό μ μ₯νλ©΄, (0,0)μ μ μ₯λλ κ² μλλΌ, component λ³λ‘ ν νμ λ€ λ€μ΄κ°.
        # κ·Έλμ 64 μνμ λν΄ 64κ° νμ΄ μκΉ.
        
        self.ptr = 0  # μ experienceκ° μ μ₯λ  μμΉλ₯Ό κ°λ¦¬ν΄.
        self.size = 0 
        self.max_size = buff_size

    def store(self, state, action, reward, next_state, done): # np.ndarray ννλ‘ μ μ₯

        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.int64).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)

        self.state_buff[self.ptr] = state   # μ μ₯λλ λ°©μμ μλ np.zeros() μ€λͺ μ°Έμ‘°
        self.action_buff[self.ptr] = action
        self.reward_buff[self.ptr] = reward
        self.next_state_buff[self.ptr] = next_state
        self.done_buff[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx= np.random.randint(0, self.size, size=batch_size) # λλ€νκ² experienceλ₯Ό λ½μμ΄
        '''
        return dict(state=torch.tensor(self.state_buff[idx], dtype=torch.float32), # NNμ λ£μ λ torch.float32 νμ΄μ΄μΌ ν¨.
                    action=torch.tensor(self.action_buff[idx], dtype=torch.long),
                    reward=torch.tensor(self.reward_buff[idx], dtype=torch.float32),
                    next_state=torch.tensor(self.next_state_buff[idx], dtype=torch.float32),
                    done=torch.tensor(self.done_buff[idx]))  # μ¬μ©ν  λλ keyλ‘ μΈλ±μ± ν΄μ λΆλ¬μ¬ κ±°μ.'''
        
        return dict(state=self.state_buff[idx],
                    action=self.action_buff[idx],
                    reward=self.reward_buff[idx],
                    next_state=self.next_state_buff[idx],
                    done=self.done_buff[idx]   )

#π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨

class Agent:
    def __init__(self, init_eps, min_eps, eps_decay, gamma, target_update_freq,
                qnet, target_qnet, optimizer, criterion):

        # epsλ μ¬μ€ attributeλ‘ μ λ§λ€κ³  'get_action'μμ λ°λ‘ μ¨λ λλλ°
        # Hyperparameter Tuningν  κ±°λΌμ μ΄λ κ² μ
        self.init_eps = init_eps #0.9
        self.min_eps = min_eps #0.05
        self.eps = self.init_eps
        self.eps_decay = eps_decay
        self.timer = 0

        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.qnet = qnet
        self.target_qnet = target_qnet
        self.optimizer = optimizer
        self.criterion = criterion
        
    # actionμ 0 λλ 1μ scalarλ‘ λ°ν ν  κ±°μ
    def get_action(self, state, step):
        self.timer += 1
        self.eps = self.min_eps + (self.init_eps - self.min_eps) * np.exp(-1. * self.timer / self.eps_decay)

        #state = torch.

        # μ²μμλ eps_thresholdκ° ν° κ°μ΄λΌ elseμμλ§ μλν¨
        # λ€μ΄μ¨ stateλ GPUμμ μλν¨.
        # μΆλ ₯ν  actionμ CPUμ λ¨Έλ¬Όλ¬μΌ ν¨.
        if random.random() > self.eps:
            action = self.qnet(state).cpu().detach().squeeze().numpy().argmax(axis=0)
            return int(action)

        else:
            action = env.action_space.sample()
            return int(action)



    def learn(self, batch, current_epi):
        state = batch['state'] # tensor([[-1.0468, -0.8232,  1.4239,  0.4460],
        #                                [-8.8516e-01,  4.3342e-02,  1.6168e+00, -7.7498e-01], .......


        action = batch['action'] # tensor([[1], 
        #                                  [0], 
        #                                  [0], 


        reward = batch['reward'] # tensor([[1.],
        #                                  [1.],
        #                                  [1.],

        next_state = batch['next_state'] # stateμ κ°μ

        done = batch['done'] #tensor([[0.],
        #                             [0.],
        #                             [0.],


        # Q(S,A)μ ν΄λΉνλ λΆλΆμ.
        # Qκ°μ ANNμΌλ‘ μμΈ‘ν κ±΄λ°, κ·Έ μ€μμ μ€μ λ‘ νλ actionμ gather νμ©ν΄ μ°Ύμ.
        # dim=1μ΄λκΉ 0μ΄κ³Ό 1μ΄ μ€μμ indexμ λ§λ κ²λ€μ κ³¨λΌμ¬ κ±°μ.
        current_q = self.qnet(state).gather(dim=1, index=action) # tensor([[0.3761],
        #                                                                  [0.3822],
        #                                                                  [0.3724],....... grad_fn=)

        # max Q(S',A')μ ν΄λΉνλ λΆλΆμ.
        # κ° νλ§λ€ 0κ³Ό 1μ λνμ¬ Qκ°μ΄ λμλλ°, dim=1μ΄λκΉ maxμ ν΄λΉνλ μ΄λ§ λ½μμ΄
        next_q = self.target_qnet(next_state).max(dim=1)[0].reshape(-1, 1)#tensor([[0.3761],
        #                                                                          [0.3822],
        #                                                                          [0.3724],....... grad_fn=)

        # R + gamma * max Q(S',A')
        td_target = reward + self.gamma * next_q * (1-done)

        # td_error = td_target - current_q
        loss = self.criterion(current_q, td_target)

        self.optimizer.zero_grad()
        loss.backward()

        # Q(S,A) <---- Q(S,A) + alpha*[R + gamma*max Q(S',A') - Q(S,A)]
        self.optimizer.step()


        if current_epi % self.target_update_freq == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())


#π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨

# κ°μ’ μ€μ 
max_epi = 600
reward_per_epi = []  # μνΌμλλ§λ€ λ³΄μκ° μ μ₯
step_per_epi_list = []  # μνΌμλλ§λ€ μ€ν μ μ μ₯

# Environment κ΄λ ¨
obs_dim = 9*16 #env.observation_space.shape[0]
act_dim = 5 #env.action_space.n

# Network κ΄λ ¨                                        π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯
qnet = QNetwork(input_size=obs_dim, output_size=act_dim, hidden1=256, hidden2=256).to(device)# state λ£μ΄μ, κ° actionμ λν Qκ° μΆλ ₯
target_qnet = QNetwork(input_size=obs_dim, output_size=act_dim, hidden1=256, hidden2=256).to(device)
target_qnet.load_state_dict(qnet.state_dict())

optimizer = optim.RAdam(qnet.parameters(), lr=1e-4) #π₯π₯π₯π₯π₯π₯π₯
criterion = nn.MSELoss().to(device)

# Replay Buffer κ΄λ ¨
memory = ReplayBuffer(obs_dim, act_dim, buff_size=100_000)

# Agent κ΄λ ¨                                               π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯
agent = Agent(init_eps=0.9, min_eps=0.05, eps_decay=200, gamma=0.95, target_update_freq=4,
                qnet=qnet, target_qnet=target_qnet, optimizer=optimizer, criterion=criterion)


#π₯π₯κ² κ΄λ ¨ μ€μ π₯π₯
max_epi = 3_000
reward_per_epi_list = []  # μνΌμλλ§λ€ λ³΄μκ° μ μ₯
step_per_epi_list = []  # μνΌμλλ§λ€ μ€ν μ μ μ₯


#π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨π¨
# μ μ₯ν  λλ np.arrayμ΄κ³  κΊΌλ΄μ¬ λλ torch.tensor

time_box = []

start_time = datetime.now().replace(microsecond=0) 


#==========GPU μΈ λ==========#
# μ΄ μλ forμμλ np.ndarrayκ° κ·Έλλ‘ μ μ§λκ³ 
# μ λ³΄μ΄λ κ³³μμ tensorλ‘ λ³ννκ³  GPUλ‘ λκΈ°λ κ±°μ.
# λ€μ μ¬κΈ°λ‘ λΆλ¬μ¬ λλ CPUλ‘ λκΈ°κ³  np.ndarrayλ‘ λΆλ¬μ΄.
for episode in range(1, max_epi+1): # μνΌμλ λ¨μ

    time_time = 0
    # λ°©κΈ λ§ λ°μ stateλ numpy.ndarray
    state = env.reset() 
    state = np.ravel(state)
   
    ''' np.ndarray
    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 5 0 0 4 0 0 0 0 0 0 4 0 0 1 1 2 2 2 2
    2 2 2 2 2 2 2 2 2 2 1 1 0 4 0 0 0 0 0 0 4 0 0 0 0 0 1 1 2 2 2 2 2 2 2 2 2
    2 2 2 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2
    1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
    '''

    done = False


    step_per_epi = 0   # μνΌμλλ§λ€ step μΌλ§λ λ²ν°λ μΆμ ν  κ±°μ
    reward_per_epi = 0 # μνΌμλλ§λ€ cumulative rewardλ₯Ό μΆμ ν  κ±°μ


    while not done: # step λ¨μ

        time_time += 1

        #env.render()

        # int μΆλ ₯ λμ΄
        action = agent.get_action(torch.tensor(state, dtype=torch.float32).to(device),  step=step_per_epi)

        env_info = env.step(action)

        next_state = env_info[0]
        next_state = np.ravel(next_state) # np.ndarray μΆλ ₯

        reward = env_info[1] - 1
        reward = reward # int μΆλ ₯

        done = env_info[2] # bool μΆλ ₯


        memory.store(state, action, reward, next_state, done)


        if memory.size >= 200: #π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯π₯
            batch = memory.sample(batch_size=32) # κΊΌλ΄μ¨ batchλ μ λΆ tensor ννλ‘ κ΅¬μ±λ¨
            agent.learn(batch, current_epi=episode)

        reward_per_epi += reward
        step_per_epi += 1

        state = next_state 

    reward_per_epi_list.append(reward_per_epi) # μνΌμλλ§λ€ λλκ³  μ΅μ’ μ»μ λμ  reward μ μ₯
    step_per_epi_list.append(step_per_epi) # μνΌμλλ§λ€ λλκ³  μ΅μ’ μ»μ step μ μ₯

    print(f'Elapsed_Time: {datetime.now().replace(microsecond=0) - start_time}')

    if episode % 20 == 0:
        print(f'Episode: {episode},  Avg.Rewards: {np.mean(reward_per_epi_list[-100:])},  Avg.Steps: {np.mean(step_per_epi_list[-100:])},\
            Epsilon: {agent.eps}')

    #if np.mean(reward_per_epi_list[-100:]) >= 2000:
    #    print(f'Environment solved in {episode} episodes!')
    #    break


env.close()

# plotting rewards/episode
fig = plt.figure(figsize=(12, 8)) 
plt.plot(reward_per_epi_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN_Rewards/Episode')
plt.grid()
plt.show()


end_time = datetime.now().replace(microsecond=0)

print(f'Start_Time: {start_time}')
print(f'End_Time: {end_time}')
print(f'Total_Training_Time: {end_time - start_time}')






