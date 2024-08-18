# PPO - Policy based

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
learning_rate = 0.0003
gamma = 0.99
lambda_gae = 0.95
clip_epsilon = 0.2
buffer_limit = 50000
batch_size = 64
steps = 0
update_interval = 2048

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

        s_lst = np.array(s_lst, dtype=np.float32)
        a_lst = np.array(a_lst, dtype=np.float32)
        r_lst = np.array(r_lst, dtype=np.float32)
        s_prime_lst = np.array(s_prime_lst, dtype=np.float32)
        done_mask_lst = np.array(done_mask_lst, dtype=np.float32)

        return torch.tensor(s_lst), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor
        self.fc_pi = nn.Linear(128, 31)  # Policy network for 31 actions

        # Critic
        self.fc_v = nn.Linear(128, 1)    # Value network for state value

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc_pi(x), dim=0)
        return x

    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v

    def sample_action(self, obs, epsilon):
        prob = self.pi(obs)
        m = torch.distributions.Categorical(prob)
        return m.sample().item()

def compute_gae(rewards, values, dones, next_value):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lambda_gae * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
    return returns

def train(model, memory, optimizer, steps):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)

    td_target = r + gamma * model.v(s_prime) * done_mask
    delta = td_target - model.v(s)
    delta = delta.detach().numpy()

    advantage = compute_gae(r.numpy(), model.v(s).detach().numpy(), done_mask.numpy(), model.v(s_prime).detach().numpy())
    advantage = torch.tensor(advantage, dtype=torch.float32)

    pi = model.pi(s)
    pi_a = pi.gather(1, a)
    ratio = torch.exp(torch.log(pi_a) - torch.log(pi_a.detach()))

    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantage

    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(model.v(s), td_target.detach())

    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name='Road_RL/Prototype 1', side_channels=[channel])
channel.set_configuration_parameters(time_scale=10)

env.reset()
behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0,:]

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
memory = ReplayBuffer()

print_interval = 1
score = 0.0
goalchk=False

def rwd(t, st):
    global goalchk
    if goalchk: return 0

    dist = (t[0]-t[3])**2+(t[1]-t[4])**2+(t[2]-t[5])**2

    rwd = -dist

    if dist <= 100 :
        rwd += 1000000
        rwd += (100000 - dist)
        rwd -= st * 100
        print("goal")
        goalchk=True
    else:
        rwd -= 100

    obstacle_avoidance = cur_obs[-1] >= 15 and cur_obs[-3] >= 15 and cur_obs[-5] >= 15
    lane_position = cur_obs[2]
    lane_center = 10 <= lane_position <= 20

    if obstacle_avoidance and lane_center:
        rwd += 200
    else:
        rwd -= 100

    return rwd


for n_epi in range(100000):
    epsilon = max(0.01, 0.4 - 0.01 * (n_epi/10 ))
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
        a = model.sample_action(torch.from_numpy(s).float(), epsilon)
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

    if memory.size() > update_interval:
        train(model, memory, optimizer, steps)
        memory.buffer.clear()

    if n_epi % print_interval == 0 and n_epi != 0:
        print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, abs(score / print_interval), memory.size(), epsilon * 100))

        score = 0.0

env.close()
