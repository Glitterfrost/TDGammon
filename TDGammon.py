import torch
import torch.nn as nn
import torch.optim as optim
import pyspiel
import numpy as np
from tqdm import tqdm

MAX_AGENT = 0
def encode_point(n):
    features = [0.0] * 4
    if n >= 1:
        features[0] = 1.0
    if n >= 2:
        features[1] = 1.0
    if n >= 3:
        features[2] = 1.0
    if n > 3:
        features[3] = (n - 3) / 2.0
    return features

def td_gammon_features(state):
    features = []


    for i in range(24):
        n0 = state.board(0, i)  
        n1 = state.board(1, i)  
        features.extend(encode_point(n0))
        features.extend(encode_point(n1))

  
    state_tensor = state.observation_tensor(MAX_AGENT)
    if state.current_player() == 0:
        bar_1 = state_tensor[192]
        bar_0 = state_tensor[195]

        off_1 = state_tensor[193]
        off_0 = state_tensor[196]
    else:
        bar_1 = state_tensor[195]
        bar_0 = state_tensor[192]

        off_1 = state_tensor[196]
        off_0 = state_tensor[193]

    features.append(bar_0 / 2.0)
    features.append(bar_1 / 2.0)
    features.append(off_0 / 15.0)
    features.append(off_1 / 15.0)

    if state.current_player() == 0:
        features.append(1.0)
        features.append(0.0)
    else:
        features.append(0.0)
        features.append(1.0)

    return np.array(features, dtype=np.float32)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.Sigmoid(),
            nn.Linear(40, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def resolve_chance_nodes(state):
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        sampled_action = np.random.choice(actions, p=probs)
        state.apply_action(sampled_action)
    return state

def train_td_lambda_gammon(episodes=50000, alpha=0.1, gamma=1, lambda_=0.8, epsilon_start=1.0, epsilon_min=0.1, device="cpu"):
    game = pyspiel.load_game("backgammon")
    input_dim = 198
    model = ValueNetwork(input_dim).to(device)
    epsilon = epsilon_start
    decay_rate = (epsilon_start - epsilon_min) / episodes

    for ep in tqdm(range(episodes)):
        state = resolve_chance_nodes(game.new_initial_state())

        
        eligibility_traces = [torch.zeros(weights.shape, requires_grad=False) for weights in list(model.parameters())]

        while not state.is_terminal():
            if state.is_chance_node():
                state = resolve_chance_nodes(state)
                continue

            current_player = state.current_player()
            legal_actions = state.legal_actions()

            
            if np.random.rand() < epsilon:
                action = np.random.choice(legal_actions)
            else:
                values = []
                for a in legal_actions:
                    s_prime = state.child(a)
                    x = torch.tensor(td_gammon_features(s_prime), dtype=torch.float32, device=device).unsqueeze(0)
                    v = model(x)
                    values.append(v.item())
                action = legal_actions[np.argmax(values)] if current_player == 0 else legal_actions[np.argmin(values)]

            next_state = state.child(action)

          
            x = torch.tensor(td_gammon_features(state), dtype=torch.float32, device=device).unsqueeze(0)
            v = model(x)

            if not next_state.is_terminal():
                reward = 0
                x_next = torch.tensor(td_gammon_features(next_state), dtype=torch.float32, device=device).unsqueeze(0)
                v_next = model(x_next)
            else:
                reward = 1.0 if current_player == 0 else 0
                v_next = torch.tensor([[reward]], dtype=torch.float32, device=device)

            delta = (gamma * v_next - v).item()
            model.zero_grad()
            v.backward()

            with torch.no_grad():
                parameters = list(model.parameters())
                for i, weights in enumerate(parameters):
                    eligibility_traces[i] = lambda_ * eligibility_traces[i] + weights.grad
                    weights.add_(alpha * delta * eligibility_traces[i])

            state = next_state

        epsilon = max(epsilon_min, epsilon - decay_rate)

    return model

#model = train_td_lambda_gammon(episodes=1000)
