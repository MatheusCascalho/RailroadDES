import numpy as np
import random

# Parâmetros do ambiente
n_states = 5  # estados 0 a 4
n_actions = 2  # ações: 0 = esquerda, 1 = direita
terminal_state = 4

# Hiperparâmetros do Q-Learning
alpha = 0.1      # taxa de aprendizado
gamma = 0.9      # fator de desconto
epsilon = 0.1    # exploração

# Inicializa a Q-table
Q = np.zeros((n_states, n_actions))

# Função para escolher ação com política ε-greedy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        return np.argmax(Q[state])

# Função de transição simples
def step(state, action):
    if action == 1:  # direita
        next_state = min(state + 1, n_states - 1)
    else:  # esquerda
        next_state = max(state - 1, 0)
    reward = 1 if next_state == terminal_state else 0
    return next_state, reward

# Treinamento
n_episodes = 1000
for episode in range(n_episodes):
    state = 0
    while state != terminal_state:
        action = choose_action(state)
        next_state, reward = step(state, action)
        best_next_q = np.max(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * best_next_q - Q[state, action])
        state = next_state

# Exibe a Q-table aprendida
print("Q-table final:")
print(Q)
