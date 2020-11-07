import collections
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, NamedTuple, Callable
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

### HELPERS ###
MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MinMaxStats(object) :
    """
    a class that holds the min max vals of the tree
    """

    def __init__(self,known_bounds: Optional[KnownBounds]):
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE

    def update(self,value:float):
        self.minimum = min(self.minimum,value)
        self.maximum = max(self.maximum,value)

    # if the value is unknown, set it to the default, the lowest possible value
    def normalize(self,value:float) -> float:

        if value is None : return 0.

        if self.maximum > self.minimum :
            # normalize only when we have set the max and min vals
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class MuZeroConfig(object) :
    def __init__(self,
                 action_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 visit_softmax_temperature_fn,
                 known_bounds: Optional[KnownBounds] = None):
        """

        :param action_space_size:
        :param discount:
        :param dirichlet_alpha:
        :param num_simulations:
        :param td_steps:
        :param lr_init:
        :param lr_decay_steps:
        :param visit_softmax_temperature_fn:
        :param known_bounds:
        """
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
        self.training_steps = int(1e6)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

    def new_game(self):
        return Game(self.action_space_size, self.discount)

def make_board_game_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float) -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if num_moves < 30 :
            return 1.
        else :
            return 0. # play according to the max

    return MuZeroConfig(
        action_space_size=action_space_size,
        max_moves=max_moves,
        discount=1.0,
        dirichlet_alpha=dirichlet_alpha,
        num_simulations=800,
        batch_size=2048,
        td_steps=max_moves,  # Always use Monte Carlo return.
        num_actors=3000,
        lr_init=lr_init,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=KnownBounds(-1, 1)
    )

def make_chess_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=4672, max_moves=512, dirichlet_alpha=0.3, lr_init=0.1)

def make_go_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=362, max_moves=722, dirichlet_alpha=0.03, lr_init=0.01)

def make_shogi_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=11259, max_moves=512, dirichlet_alpha=0.15, lr_init=0.1)

def make_atari_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        action_space_size=18,
        max_moves=27000,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=50,
        batch_size=1024,
        td_steps=10,
        num_actors=350,
        lr_init=0.05,
        lr_decay_steps=350e3,
        visit_softmax_temperature_fn=visit_softmax_temperature)

class Action(object) :

    def __init__(self, index:int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

class Player(object) :
    pass

class Node(object) :
    def __init__(self, prior:float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0 : return 0
        return self.value_sum / self.visit_count

class ActionHistory(object):
    """
    Simple history container used inside the search
    only used to keep track of actions executed
    """

    def __init__(self,history: List[Action], action_space_size:int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action:Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        return Player()

class Environment(object) :
    """the environment that RickZero is interacting with"""

    def step(self, action):
        pass

class Game(object):
    """A single episode of interaction with the environment"""

    def __init__(self, action_space_size: int, discount: float):
        self.environment = Environment() # game specific environment
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount

    def terminal(self) -> bool:
        # game specific termination rules
        pass

    def legal_actions(self) -> List[Action]:
        # game specific calculations of legal actions
        return []

    def apply(self, action:Action):
        reward = self.environment.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root:Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index:int):
        # Game-specific feature planes
        return []

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
        # the value target is the  discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1) :
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values) :
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else :
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i

            if current_index < len(self.root_values):
                targets.append((value, self.rewards[current_index],
                                self.child_visits[current_index]))
            else :
                # states past the end of games are treated as absorbing states
                targets.append((0,0,[]))
        return targets

    def to_play(self) -> Player:
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

class ReplayBuffer(object):

    def __init__(self, config:MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps:int, td_steps:int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i), g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                for (g, i) in game_pos]

    def sample_game(self) -> Game:
        # sample game from buffer either uniformly or according to some priority
        return self.buffer[0]

    def sample_position(self, game) -> int:
        # sample position from game either uniformly or according to some priority
        return -1

class NetworkOutput(NamedTuple):
    value:float
    reward:float
    policy_logits: Dict[Action, float]
    hidden_state: Optional[List[float]]

    @staticmethod
    def build_policy_logits(policy_logits):
        return {Action(i):logit for i, logit in enumerate(policy_logits[0])}

class AbstractNetwork(ABC):

    def __init__(self):
        self.training_steps = 0

    @abstractmethod
    def initial_inference(self, image) -> NetworkOutput:
        pass

    @abstractmethod
    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        pass

class UniformNetwork(AbstractNetwork):
    """policy -> uniform, value -> 0, reward -> 0"""

    def __init__(self, action_size: int):
        super().__init__()
        self.action_size = action_size

    def initial_inference(self, image) -> NetworkOutput:
        return NetworkOutput(0, 0, {Action(i): 1 / self.action_size for i in range(self.action_size)}, None)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        return NetworkOutput(0, 0, {Action(i): 1 / self.action_size for i in range(self.action_size)}, None)

class BaseNetwork(AbstractNetwork):

    def __init__(self, representation_network: Model, value_network: Model, policy_network: Model,
                 dynamic_network: Model, reward_network: Model):
        super().__init__()
        # Networks blocks
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        return NetworkOutput(0,0,{},[])

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        return NetworkOutput(0,0,{},[])

    def get_weights(self):
        # returns the weights of the network
        return []

    def training_steps(self) -> int:
        # how many steps / batches the network has been trained for
        return 0

    @abstractmethod
    def _value_transform(self, value: np.array) -> float:
        pass

    @abstractmethod
    def _reward_transform(self, reward: np.array) -> float:
        pass

    @abstractmethod
    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        pass

    def cb_get_variables(self) -> Callable:
        """Return a callback that return the trainable variables of the network."""

        def get_variables():
            networks = (self.representation_network, self.value_network, self.policy_network,
                        self.dynamic_network, self.reward_network)
            return [variables
                    for variables_list in map(lambda n: n.weights, networks)
                    for variables in variables_list]

        return get_variables


class SharedStorage(object):

    def __init__(self):
        self._networks = {}

    def latest_network(self) -> BaseNetwork:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else :
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network()

    def save_network(self, step:int, network:BaseNetwork):
        self._networks[step] = network

### END OF HELPERS ###
######################

# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    for _ in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer)

    train_network(config, storage, replay_buffer)

    return storage.latest_network()


#######################
## Part 1: Self-Play ##

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config:MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer) :
    while True :
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)

# each game is produced by starting at the initual board position, then
# repeatedly executing Monte Carlo Tree Search (MCTS) to generate moves until the end
# of the game is reached
def play_game(config: MuZeroConfig, network: BaseNetwork) -> Game :
    game = config.new_game()

    while not game.terminal() and len(game.history) < config.max_moves :
        # at the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.legal_actions(), network)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)
        return game

# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory, network: BaseNetwork):
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state, history.last_action())
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(search_path, network_output.value, history.to_play(), config.discount, min_max_stats)

def select_action(config: MuZeroConfig, num_moves: int, node: Node, network: BaseNetwork):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]

    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps()
    )

    _, action = softmax_sample(visit_counts, t)
    return action

def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
    # select child with highest USB score
    _, action, child = max(
        (ucb_score(config, node, child, min_max_stats), action, child) for action, child in node.children.items()
    )

    return action, child

# the score for a node is based on its value plut an exploration bonus based on the prior
def ucb_score(config: MuZeroConfig, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score

# we expand a node using the value, reward, and policy prediction obtained from the neural net
def expand_node(node: Node, to_play: Player, actions: List[Action], network_output: NetworkOutput) -> None:
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
    return None

# at the end of the simulation, propagate the evaluation all the way up the tree to the root
def backpropagate(search_path: List[Node], value:float, to_play: Player, discount: float, min_max_stats: MinMaxStats):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions
def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
    return

######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########

def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer) :
    network = BaseNetwork()
    learning_rate = config.lr_init * config.lr_decay_rate**\
                    (tf.compat.v1.train.get_global_step() / config.lr_decay_steps)
    # choose the right optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, momentum=config.momentum)

    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights(optimizer, network, batch, config.weight_decay)
    storage.save_network(config.training_steps, network)

# scale the gradient for backward pass
def scale_gradient(tensor, scale: float):
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)

def update_weights(optimizer: tf.keras.optimizers.Optimizer, network: BaseNetwork, batch, weight_decay: float):
    loss = 0
    for image, actions, targets in batch:
        # initial step, from the real observation
        value, reward, policy_logits, hidden_state = network.initial_inference(image)
        predictions = [(1., value, reward, policy_logits)]

        # recurrent steps, from action and previous hidden state
        for action in actions:
            value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
        predictions.append((1., len(actions), value, reward, policy_logits))

        hidden_state = scale_gradient(hidden_state, 0.5)

    for prediction, target in zip(predictions, targets):
        gradient_scale, value, reward, policy_logits = prediction
        target_value, target_reward, target_policy = target

        l = (
            scalar_loss(value, target_value) +
            scalar_loss(reward, target_reward) +
            tf.nn.softmax_cross_entropy_with_logits(
                logits=policy_logits, labels=target_policy)
        )

        loss += l

    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    optimizer.minimize(loss=loss, var_list=network.cb_get_variables())

def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari
    return -1


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################

# Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float):
    return 0, 0

def launch_job(f, *args):
    f(*args)

def make_uniform_network():
  return UniformNetwork(action_size=0)