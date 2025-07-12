import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Hiperparametros
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 1e-3
batch_size = 64
memory_size = 1000
target_update_freq = 10
num_episodes = 500

# Ambiente
env = gym.make('CartPole-v1')
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

# Replay Buffer
memory = deque(maxlen=memory_size)

# Rede Neural
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.fc(x)


# Intância das redes
policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# função para escolher ação (ε-greedy)
def select_action(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            return policy_net(state).argmax().item()


# Treinamento
for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(memory) >= batch_size:
            # Amostra mini-batch
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            # Q-alvo
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1, keepdim=True)[0]
                target_q = rewards + gamma * max_next_q * (1 - dones)


            # Q previsto
            current_q = policy_net(states).gather(1, actions)

            # Loss e otimização
            loss = criterion(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decaimento do epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode: {episode},Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")