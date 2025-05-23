import random
from tqdm import tqdm
def play_game(game, agent0, agent1, verbose=False):
    state = game.new_initial_state()
    agents = [agent0, agent1]
    count = 0
    while not state.is_terminal():

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = random.choices(actions, probs)[0]
            state.apply_action(action)
        else:
            count+=1
            current_player = state.current_player()
            action = agents[current_player].step(state)
            state.apply_action(action)

    returns = state.returns()
    if verbose:
        print(f"Game over. Returns: {returns}")
        print(state)
    return returns


def run_many_games(game, agent0, agent1,  better = -1, num_games=100,verbose = False):
    wins = {0: 0, 1: 0, "draws": 0}

    for i in tqdm(range(num_games)):
        returns = play_game(game, agent0, agent1, verbose=verbose)

        if returns[0] > returns[1]:
            wins[0] += 1
        elif returns[1] > returns[0]:
            wins[1] += 1
        else:
            wins["draws"] += 1
    if better == 0:
        print("Player 0 is better")
    else:
        print("Player 1 is better")
    print(f"\n--- Results after {num_games} games ---")
    print(f"Agent 0 wins: {wins[0]}")
    print(f"Agent 1 wins: {wins[1]}")
    print(f"Draws: {wins['draws']}")

    return wins
