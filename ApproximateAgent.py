import random 
from TDGammon import td_gammon_features, MAX_AGENT
import numpy as np
import torch

class ApproximateAgent():
    def __init__(self, player_name, player_id, learned_agent):
        self.player_name = player_name
        self.player_id = player_id
        self.learned_agent = learned_agent

    def step(self, state):
        best_action = None
        legal_actions = state.legal_actions()
        action_values = []
        for action in legal_actions:
            next_state = state.child(action)
            next_state = td_gammon_features(next_state)
            with torch.no_grad():
                value_next =  self.learned_agent(torch.tensor(next_state).unsqueeze(0))
                action_values.append(value_next.item())
        if self.player_id == MAX_AGENT:
            best_action = legal_actions[np.argmax(action_values)]
        else:
            best_action = legal_actions[np.argmin(action_values)]
        if best_action is None:
            best_action = random.choice(legal_actions)
        return best_action