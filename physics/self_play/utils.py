# helper functions
import numpy as np
from typing import Optional
from collections import namedtuple

KnownBounds = namedtuple('KnownBounds', ['min', 'max'])
MAXIMUM_FLOAT_VALUE = float('inf')

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

class Node(object) :
    """the nodes in the Monte-Carlo Tree Search"""
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

def softmax_sample(visit_counts, actions, t):
    # choose an action randomly, weighted to its visit counts
    assert actions is not None, 'softmax_sample was passed `None` to actions'
    counts_exp = np.exp(visit_counts) * (1 / t)
    prob_vector = counts_exp / np.sum(counts_exp, axis=0)
    action_index = np.random.choice(len(actions), p=prob_vector)
    return actions[action_index]