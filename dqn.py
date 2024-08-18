# DQN: Value-based

import numpy as np
import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
steps = 0

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # 리스트를 하나의 NumPy 배열로 변환한 후, 텐서로 변환
        s_lst = np.array(s_lst, dtype=np.float32)
        a_lst = np.array(a_lst, dtype=np.int64)
        r_lst = np.array(r_lst, dtype=np.float32)
        s_prime_lst = np.array(s_prime_lst, dtype=np.float32)
        done_mask_lst = np.array(done_mask_lst, dtype=np.float32)

        return torch.tensor(s_lst), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(5, 128) # 입력 5가지
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 31) # 출력 31가지 -45~45도까지 30등분

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,30) # 위와 같이 30까지 숫자를 뽑음
        else:
            return out.argmax().item()

def train(q, q_target, memory, optimizer, steps):
    for _ in range(steps):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name='Road_RL/Prototype 1', side_channels=[channel])
channel.set_configuration_parameters(time_scale=10)

env.reset()
behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0,:]

q = Qnet()
q_target = Qnet()
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()

print_interval = 1
score = 0.0
optimizer = optim.Adam(q.parameters(), lr=learning_rate)
goalchk=False


def rwd(t, st):
    global goalchk
    #rwd = abs(st)*10
    if goalchk: return 0

    dist = (t[0]-t[3])**2+(t[1]-t[4])**2+(t[2]-t[5])**2

    rwd = -dist

    if dist <= 100 :
        #rwd += (100000 - abs(st)+30)

        rwd += 1000000
        rwd += (100000 - dist)
        rwd -= st * 100
        print("goal")
        goalchk=True
    else:
        rwd -= 100

    obstacle_avoidance = cur_obs[-1] >= 15 and cur_obs[-3] >= 15 and cur_obs[-5] >= 15
    lane_position = cur_obs[2]  # 차선 위치를 나타내는 센서 값
    lane_center = 10 <= lane_position <= 20

    # 장애물 회피 및 차선 유지에 따른 보상
    if obstacle_avoidance and lane_center:
        rwd += 200  # 차선을 유지할 경우 보상
    else:
        rwd -= 100  # 차선을 벗어났을 경우 페널티

    return rwd


for n_epi in range(100000):
    epsilon = max(0.01, 0.4 - 0.01 * (n_epi/10 ))  # 초기 탐험률을 높게 설정하고 점진적으로 감소
    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    decision_steps, _ = env.get_steps(behavior_name)
    s = decision_steps.obs[0][0,6:]
    done = (decision_steps.reward[0] != 0.0)
    score = 0.0
    pre = 0.0
    steps = 0
    pre_scr = 0
    goalchk = False

    while not done:
        a = q.sample_action(torch.from_numpy(s).float(), epsilon)
        act = float(a - 15) / 15
        pre = pre * 0.8 + act * 0.2
        env.set_actions(behavior_name, np.array([[pre, 150, 150]]))
        env.step()
        behavior_name = list(env.behavior_specs)[0]
        decision_steps, _ = env.get_steps(behavior_name)
        s_prime = decision_steps.obs[0][0,6:]
        done = (decision_steps.reward[0] != 0)
        done_mask = 0.0 if done else 1.0
        cur_obs = decision_steps.obs[0][0,:]
        steps += 1
        r = rwd(cur_obs, steps)
        memory.put((s, a, (r - pre_scr) / 100, s_prime, done_mask))
        pre_scr = r
        s = s_prime

        score += r

        if done:
            break

    if memory.size() > 2000:
        train(q, q_target, memory, optimizer, steps)

    if n_epi % print_interval == 0 and n_epi != 0:
        q_target.load_state_dict(q.state_dict())
        print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, abs(score / print_interval), memory.size(), epsilon * 100))

        score = 0.0

env.close()
