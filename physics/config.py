import chess
from collections import namedtuple
from typing import Optional, Dict
from physics.game import AbstractGame
from physics.game import Chess
from physics.networks import BaseNetwork, UniformNetwork
from physics.networks import ChessNetwork
from tensorflow.keras import optimizers

KnownBounds = namedtuple('KnownBounds', ['min', 'max'])

class MuZeroConfig(object) :

    def __init__(self,
                 game,
                 num_training_loop: int,
                 num_episodes: int,
                 num_epochs: int,
                 network_args: Dict,
                 network,
                 action_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 td_steps: int,
                 num_actors: int,
                 batch_size: int,
                 lr_init: float,
                 visit_softmax_temperature_fn,
                 known_bounds: Optional[KnownBounds] = None,
                 board: Optional[chess.Board]=None,
                 is_white: Optional[bool]=None):
        """

        :param game:
        :param num_training_loop:
        :param num_episodes:
        :param num_epochs:
        :param network_args:
        :param network:
        :param action_space_size:
        :param max_moves:
        :param discount:
        :param dirichlet_alpha:
        :param num_simulations:
        :param td_steps:
        :param num_actors:
        :param batch_size:
        :param lr_init:
        :param visit_softmax_temperature_fn:
        :param known_bounds:
        """

        # env
        self.game = game

        ### Self-Play
        self.action_space_size = action_space_size
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.num_training_loop = num_training_loop
        self.num_episodes = num_episodes # num episodes per training loop
        self.num_epochs = num_epochs # num epochs per train loop

        self.training_steps = int(1e6)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps


        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.network_args = network_args
        self.network = network

        # Exponential learning rate schedule
        self.lr_init = lr_init
        # self.lr_decay_rate = 0.1

        # get chess-specific variables
        self.board = board
        self.is_white = is_white

    def new_game(self) -> AbstractGame:
        if type(self.game) == type(Chess) :
            return self.game(self.discount, self.board, self.is_white)
        return self.game(self.discount)

    def new_network(self) -> BaseNetwork:
        return self.network(**self.network_args)

    def uniform_network(self) -> UniformNetwork:
        return UniformNetwork(self.action_space_size)

    def new_optimizer(self) -> optimizers:
        return optimizers.SGD(learning_rate=self.lr_init, momentum=self.momentum)

    def new_adam(self) -> optimizers:
        return optimizers.Adam(learning_rate=self.lr_init)

def make_chess_config() -> MuZeroConfig :

    def visit_softmax_temperature(num_moves, training_steps, max_look=30):
        if num_moves < max_look :
            return 1.
        else :
            return 0.

    return MuZeroConfig(
        game=Chess,
        num_training_loop=100,
        num_episodes=20,
        num_epochs=20,
        network_args= {
            'action_size': 2,
            'state_size': 4,
            'representation_size': 4,
            'max_value': 500
        },
        network=ChessNetwork,
        max_moves=512,
        discount=.99,
        dirichlet_alpha=0.3,
        num_simulations=501,
        action_space_size=4672,
        batch_size=512,
        td_steps=10,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        lr_init=0.1,
        num_actors=2,
        known_bounds=KnownBounds(-1,1)
    )
