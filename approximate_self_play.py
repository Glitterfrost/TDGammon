import TDGammon as TDG
import TDGammon_v2 as TDG_v2
from self_play import play_game, run_many_games
import ApproximateAgent as AA

import pyspiel
import matplotlib.pyplot as plt
import numpy as np

game = pyspiel.load_game("backgammon")
value1 =  TDG_v2.train_td_lambda_gammon(episodes=1)
value2 =  TDG_v2.train_td_lambda_gammon(episodes=1000)
# plt.plot(data['epsilon'])
# plt.savefig("test_epsilon_selfplay.png")

agent0 = AA.ApproximateAgent(player_id=0, player_name = "BLACK", learned_agent=value1)
agent1 = AA.ApproximateAgent(player_id=1, player_name = "WHITE", learned_agent=value2)
game = pyspiel.load_game("backgammon")

run_many_games(game, agent0, agent1,better=1 ,num_games=1000)


agent0 = AA.ApproximateAgent(player_id=0, player_name = "BLACK", learned_agent=value2)
agent1 = AA.ApproximateAgent(player_id=1, player_name = "WHITE", learned_agent=value1)
game = pyspiel.load_game("backgammon")

run_many_games(game, agent0, agent1,better = 0 ,num_games=1000)