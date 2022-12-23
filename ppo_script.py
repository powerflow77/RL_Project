from __future__ import unicode_literals
import gym
import gym_gridworld


import os
import glob # 각종 파일 다룰 때 사용.
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

env = gym.make('gridworld-v0')

# 🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨

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

# 🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨

class RolloutBuffer:
    
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]   # 리스트 초기화(비우기)
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# 🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨

#🔥 Actor랑 Critic이 같은 클래스에 있지만, network는 각각 사용함.
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()# state_dim은 가속도 등 4, action_dim 왼쪽 오른쪽 2

        self.has_continuous_action_space = has_continuous_action_space # True if action space is continuous.
        
        # continuous action space의 경우, 초기 Variance 값 필요함.
        if has_continuous_action_space: 
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        #                                (이 크기 텐서를)     위 값으로 채워라.
        #                                대각선에 딱 끼워넣으려면 (col_dim, )의 차원 가져야함. & std 제곱해서 variance 만들었음.
        #                                [a, b, c, d] -> 이런 형식 가지고 내려가서 Covariance Matrix 만들 거임.

        #🔥Actor
        #🔥🔥Continuous Action Space
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                                    nn.Linear(state_dim, 64), 
                                    nn.Tanh(),
                                    nn.Linear(64, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, action_dim) )
                                    
        #🔥🔥Discrete Action Space
        else:
            self.actor = nn.Sequential(
                                    nn.Linear(state_dim, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, action_dim),
                                    nn.Softmax(dim=-1)  )     # 각 action을 할 확률이 출력.

        #🔥 Critic
        #🔥🔥 Critic은 Actor가 continuous인지 discrete인지 상관없음.
        self.critic = nn.Sequential(
                                nn.Linear(state_dim, 64),   # self.actor와 self.critic이 예측임. forward 함수 안 씀.
                                nn.Tanh(),
                                nn.Linear(64, 64),
                                nn.Tanh(),
                                nn.Linear(64, 1)  )
    
    # 위에서는 초기 action_std 넣어준 거고, 여기서는 학습 과정에서 들어오는 것들 활용하는 거임.
    def set_action_std(self, new_action_std): 
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError # 의도적 에러 발생.
    
    # Environment에 수행할 action 값을 뽑아낼 때 쓸 거임.
    def act(self, state):  # action 값이랑 그때의 logprob값을 반환.

        #🔥Continuous Action Space
        if self.has_continuous_action_space:
            action_mean = self.actor(state)                        # Mean🔥🔥이게 예측임 forward 함수 안 씀.🔥🔥
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0) # 초기값이든 현재값이든 받아서 Covariance Matrix 만듦.
            dist = MultivariateNormal(action_mean, cov_mat)        # Multivariate normal distribution 계산해줌.
        
        #🔥Discrete Action Space
        else:
            action_probs = self.actor(state)                       # 🔥🔥Mean 예측🔥🔥
            dist = Categorical(action_probs)                       # action에 대한 discrete distribution 계산해줌.

        action = dist.sample()                                     # 위가 무엇이든 간에, action 하나 뽑아옴.
        action_logprob = dist.log_prob(action)                     # 이때의 log(p(action|state)) 계산
        
        return action.detach(), action_logprob.detach()
    
    # act는 Environment에 할 action 뽑아내기
    # evaluate는 Buffer에서 꺼내온 데이터 가지고 계산해주는 거임.
    def evaluate(self, state, action):  # K_epoch Loop 안에서 호출됨.
        
        #🔥Continuous Action Space
        if self.has_continuous_action_space:
            action_mean = self.actor(state)                       # 🔥🔥Mean 예측🔥🔥
            
            action_var = self.action_var.expand_as(action_mean)   # 크기 맞게 사이즈 확장. & Variance
            cov_mat = torch.diag_embed(action_var).to(device)     # 초기값이든 현재값이든 받아서 Covariance Matrix 만듦.
            dist = MultivariateNormal(action_mean, cov_mat)       # Multivariate normal distribution 계산해줌.
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        
        #🔥Discrete Action Space
        else:
            action_probs = self.actor(state)     # 🔥예측
            dist = Categorical(action_probs)
            
        action_logprobs = dist.log_prob(action)  
        dist_entropy = dist.entropy()            # 🔥 분포에 대한 엔트로피(엔트로피 보너스에 사용.)
        state_values = self.critic(state)        # 🔥 Value Function 값 예측🔥🔥
        
        return action_logprobs, state_values, dist_entropy


# 🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma,\
                K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6): # 초기 action_std값 넣어줌.

        self.has_continuous_action_space = has_continuous_action_space # continuous action space냐 discrete냐

        #🔥🔥continuous action space이면 action_std_init 가져다 쓸거고, discrete 때는 None 넣을 거임.
        if has_continuous_action_space:
            self.action_std = action_std_init  

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # Replay Buffer 지정.
        self.buffer = RolloutBuffer() 

        #🔥🔥self.policy 선언🔥🔥
        # 업데이트 할 네트워크는 self.policy임.                                           초기 action_std 받음.
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},  
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}  ])# 클래스 내 다수의 NN에 대하여 쓰는 표현.

        #🔥🔥self.policy_old 선언🔥🔥
        # self.policy_old는 weight를 전달받기만 함.                                        이것도 초기 action_std 받음.
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)

        # 가지고 있던 NN weights를 self.policy_old에 이전해놓음.
        # 나중에 ratio 계산하려고 하나는 계속 업데이트, 나머지 하나는 weight 받고 가만히 있음.
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std): 
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)      # Covariance Matrix에 사용할 variance 값 계산
            self.policy_old.set_action_std(new_action_std)  # Covariance Matrix에 사용할 variance 값 계산.
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    
    def decay_action_std(self, action_std_decay_rate, min_action_std): 
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    # Environment에 대해 어떤 action할지 선택함.
    # Episode Loop에서 돌아감.
    def select_action(self, state):            # 에피소드의 과정에서, a 선택하고, 그에 대한 log_prob 구함.
        #                                        직후에, s, a, log_prob 저장함.     
        #🔥Continuous Action Space
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)  # model의 act가 예측해서 이런 거 뽑아내는 거임.
            #                                                          env.에 수행 할 좋은 action을 NN으로 뽑아내겠음.
            #                                                          
            self.buffer.states.append(state)             
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)  # 각종 값 저장.

            return action.detach().cpu().numpy().flatten()
        
        #🔥Discrete Action Space
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)  # model의 act가 예측해서 이런 거 뽑아내는 거임.
            
            self.buffer.states.append(state)               
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)    # 각종 값 저장.

            return action.item()


    #🔥🔥🔥 K_epoch Loop에서 실행되는 과정임.🔥🔥🔥
    def update(self):
        # return Gt를 만들기 위한 과정
        rewards = []
        discounted_reward = 0#          reversed는 iterabel한 객체에 대해 순서를 뒤집음.
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)  # returns Gt 만들어감.
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards(PPO2 반영)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # Replay Buffer에 저장된 것들을 가져다가 업데이트 할 거임.
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        #🔥PPO 네트워크 업데이트🔥---------------->> Training Loop 가장 바깥쪽에서 돌리는 건 buffer에 저장만 주구장창 하는 거임.
        #                                          특정 때마다 PPO 네트워크 업데이트. 학습 Loop는 K_epoch만큼 돌림. 
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            # 🔥학습 대상은 self.policy의 네트워크들🔥
            # 'old' 써있는 것들은 Buffer에서 가져다가 쓰는 거임.
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            #🔥🔥Loss Function(부호 매우 중요. pytorch는 gradient descent 제공.)🔥🔥
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # 학습 수행.
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # self.policy에 대한 학습이 끝나면 self.policy_old에 복사해줌.
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))



# 🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨


env_name = 'gridworld-v0'
has_continuous_action_space = False 

max_ep_len = 400                    # 한 개의 에피소드 당 최대 400 timestep까지만~
#💛💛💛💛💛💛💛💛💛💛💛💛💛
max_training_timesteps = int(1e5)   #💛전체 timestep은 1e5까지만 ---->> 이걸로 전체 학습 얼마나 빨리 끝낼지 조절할 수 있음.💛

print_freq = max_ep_len * 4         # 출력 시기
log_freq = max_ep_len * 2
save_model_freq = int(2e4)

action_std = None                   

update_timestep = max_ep_len * 10   # 업데이트 시기
K_epochs = 40                       # PPO 네트워크 업데이트 몇 번 하냐.
eps_clip = 0.2
gamma = 0.99

lr_actor = 3e-4
lr_critic = 1e-3

random_seed = 0

# 🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨

print("training environment name : " + env_name)

#env = gym.make('gridworld-v0')

# state space dimension🌸🌸🌸🌸🌸🌸🌸🌸🌸🌸🌸🌸🌸🌸🌸🌸🌸🌸🌸수정 사항
state_dim = 9*16 # 지훈이가 만든 Env.
#state_dim = 16*16 #내가 만든 Env.


# action space dimension
if has_continuous_action_space:
    action_dim = env.action_space.shape[0]  # continuous action space랑 
else:
    action_dim = 5
    #action_dim = env.action_space.n         # discrete action space랑 shape 표현이 서로 다름.


######################🔥🔥 logging(로깅) 🔥🔥######################

#### log files for multiple runs are NOT overwritten

log_dir = "PPO_logs"
os.makedirs(log_dir, exist_ok=True)         # PPO_logs라는 폴더 생성.

log_dir = log_dir + '/' + env_name + '/'
os.makedirs(log_dir, exist_ok=True)         # PPO_logs/CartPole-v1/라는 폴더 생성.


#### get number of log files in log directory
run_num = 0
current_num_files = next(os.walk(log_dir))[2] #❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓
run_num = len(current_num_files)


#### 매번 실행할 때마다 csv 형식의 log 파일 생성할 거임.
log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"   # csv 파일

print("current logging run number for " + env_name + " : ", run_num)
print("logging at : " + log_f_name)

#####################################################


################### checkpointing ###################

run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "PPO_preTrained"
os.makedirs(directory, exist_ok=True)   # PPO_preTrained라는 폴더 생성.

directory = directory + '/' + env_name + '/'
os.makedirs(directory, exist_ok=True)   # PPO_preTrained/CartPole-v1/라는 폴더 생성.

# 잘 학습된 PPO 모델을 저장할 거임.
checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

#####################################################


############# 각종 hyperparameters 출력 #############

print("--------------------------------------------------------------------------------------------")

print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)

print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

print("--------------------------------------------------------------------------------------------")

print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)

print("--------------------------------------------------------------------------------------------")

#🔥 Continuous Action Space  ---->> 카트폴이라 무시
if has_continuous_action_space:  
    print("Initializing a continuous action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("starting std of action distribution : ", action_std)
    print("decay rate of std of action distribution : ", action_std_decay_rate)
    print("minimum std of action distribution : ", min_action_std)
    print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")

#🔥 Discrete Action Space
else:
    print("Initializing a discrete action space policy") # 각종 설정들 출력.

print("--------------------------------------------------------------------------------------------")

print("PPO update frequency : " + str(update_timestep) + " timesteps") 
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)

print("--------------------------------------------------------------------------------------------")

print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)

if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

#####################################################

print("============================================================================================")

################# training procedure ################

# 🔥🔥🔥🔥🔥🔥🔥🔥initialize a PPO agent🔥🔥🔥🔥🔥🔥🔥🔥
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
#                                                                                           initial action_std인데 카트폴이라서 None임.

# track total training time
start_time = datetime.now().replace(microsecond=0) # microsecond 단위는 안 보겠음.
print("Started training at (GMT) : ", start_time)

print("============================================================================================")


# logging file은 csv 형식의 파일임.
log_f = open(log_f_name, "w+")              # log_f_name으로 된 csv 파일이 생성됨.
log_f.write('episode, timestep, reward\n')  # 이런 것들을 csv 파일에 기록.


# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0


# 🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨


# 🔥ppo_agent = PPO()

reward_box = []

while time_step <= max_training_timesteps:   # 설정된 최대 timestep 값보다 작으면 계속 돌림.
    
    state = env.reset()         # initial State 줌.
    state = np.ravel(state)

    current_ep_reward = 0

    for t in range(1, max_ep_len+1):  # 겜 횟수(에피소드)

        #env.render()
        
        #                                             Environment에 할 action 구하기.
        action = ppo_agent.select_action(state)     # Action  --->> 여기에 state, action, log_prob 저장 과정도 있음.
        #state, reward, done, _ = env.step(action)   # State_prime, Reward, Done
        #state = np.ravel(state)


        env_info = env.step(action)

        next_state = env_info[0]
        next_state = np.ravel(next_state)

        reward = env_info[1] - 1 # 1.0 //// 
        #print(reward)

        done = env_info[2]



        
        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)     # R 저장.
        ppo_agent.buffer.is_terminals.append(done)  # done 저장.
        
        time_step +=1
        current_ep_reward += reward     # 현재 에피소드의 총 보상

        #🔥특정 때마다 PPO 모델 업데이트🔥   -------->> 🔥위에서 계속 buffer에 쌓기만 하고, 업데이트는 특정 때만 함.🔥
        #                                               여기서 K_epoch Loop 돌아감.
        if time_step % update_timestep == 0:
            ppo_agent.update()

        # if continuous action space; then decay action std of ouput action distribution
        if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)  

        #🔥특정 때마다 값들 logging
        if time_step % log_freq == 0:

            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward)) # csv 파일에 기록❕❕
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0

        #🔥특정 때마다 average reward 출력
        if time_step % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

            print_running_reward = 0
            print_running_episodes = 0
            
        #🔥 특정 때마다 model weights 저장
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
            
        #🔥 겜 끝나면 에피소드 종료
        if done:
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    reward_box.append(current_ep_reward) # plotting rewards/episode 용도

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1


log_f.close()
env.close()


# 모든 과정 다 끝나고 나올 멘트
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)  # microsecond 단위는 안 보겠음.
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)  # 전체 과정 몇 분 걸렸냐
print("============================================================================================")


# plotting rewards/episode
fig = plt.figure(figsize=(12, 8)) 
plt.plot(reward_box)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('PPO_Rewards/Episode')
plt.grid()
plt.show()

