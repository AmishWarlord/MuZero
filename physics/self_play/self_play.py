# want to play a game?

from physics.config import MuZeroConfig
from physics.game import AbstractGame
from physics.networks import AbstractNetwork
from physics.networks import SharedStorage
from physics.self_play.mcts import run_mcts, select_action, expand_node, add_exploration_noise
from physics.self_play import Node
from physics.training import ReplayBuffer

def run_selfplay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, train_episodes: int):
    """take the latest network, produces multiple games and save them in the shared replay buffer"""
    network = storage.latest_network()
    rewards = []
    for _ in range(train_episodes):
        game = play_game(config, network)
        replay_buffer.save_game(game)
        rewards.append(sum(game.rewards))

    return sum(rewards) / train_episodes

def run_eval(config: MuZeroConfig, storage: SharedStorage, eval_episodes: int):
    """evaluate MuZero without noise added to the prior of the root and without softmax action selection"""
    network = storage.latest_network()
    rewards = []
    for _ in range(eval_episodes):
        game = play_game(config, network, train=False)
        rewards.append(sum(game.rewards))
    return sum(rewards) / eval_episodes if eval_episodes else 0

def play_game(config: MuZeroConfig, network: AbstractNetwork, train: bool=True) -> AbstractGame:
    """
    each game is produced by starting at the initial board position, then repeatedly executing MCTS
    to generate moves until the game ends
    :param config:
    :param network:
    :param train:
    :return:
    """
    game = config.new_game()
    mode_action_select = 'softmax' if train else 'max'

    while not game.terminal() and len(game.history) < config.max_moves :
        # at the root of the tree, use the representation function to obtain a hidden state
        # given the current observation
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.legal_actions(), network.initial_inference(current_observation))
        if train:
            add_exploration_noise(config, root)

        # then run a monte carlo tree search using only action sequences and
        # the model learned by the networks
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network, mode=mode_action_select)
        game.apply(action)
        game.store_search_statistics(root)
    return game