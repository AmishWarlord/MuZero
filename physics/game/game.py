from abc import abstractmethod, ABC
from typing import List

from self_play.utils import Node


class Action(object):
    """ Class that represent an action of a game."""

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

class Player(object):
    """
    legacy player class
    """
    def __eq__(self,other):
        return True

class ActionHistory(object):
    """
    hisotry container used to record actionns executed
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        return Player()

class AbstractGame(ABC):
    """
    ABC implementation of a game
    """

    def __init__(self, discount: float):
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.discount = discount

    def apply(self, action: Action):
        # apply an action to an environment
        reward = self.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        """store the stats of each run"""

        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
        """make learning targets for training"""

        # target is the discounted root value of the search tree N steps into the future, plus all previous targets
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else : value = 0

            # add discounted values to value
            value += sum([reward * self.discount ** i
                          for i, reward in enumerate(self.rewards[current_index:bootstrap_index])])

            if current_index < len(self.root_values):
                targets.append((value, self.rewards[current_index], self.child_visits[current_index]))
            else:
                # all states past the game are absorbing states
                targets.append((0,0,[]))
        return targets

    def to_play(self) -> Player:
        """return current player"""
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

    # elements to be implemented by child classes
    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """
        return size of the action space
        for chess, take in the length of legal actions from the board"""
        pass

    @abstractmethod
    def step(self, action) -> float:
        """execute a single step of the game
        for chess, make 1 move"""
        pass

    @abstractmethod
    def terminal(self) -> bool:
        """check if the game is finished
        ask board if the game is over"""
        pass

    @abstractmethod
    def legal_actions(self) -> List[Action]:
        """return the legal actions available at this time
        for chess, use built-in board function"""
        pass

    def make_image(self, state_index):
        """compute state of the game"""
        pass