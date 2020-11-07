from math import ceil, sqrt

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L2

from physics.game import Action
from physics.networks import BaseNetwork


class ChessNetwork(BaseNetwork):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 representation_size: int,
                 max_value: int,
                 hidden_neurons: int=64,
                 weight_decay: float = 1e-4,
                 representation_activation = 'tanh'):

        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = ceil(sqrt(max_value)) + 1

        l2 = L2(weight_decay)

        representation_network = Sequential([
            Dense(hidden_neurons, activation='relu', kernel_regularizer=l2),
            Dense(representation_size,activation=representation_activation,
                  kernel_regularizer=l2)
        ])

        value_network = Sequential([
            Dense(hidden_neurons, activation='relu', kernel_regularizer=l2),
            Dense(self.value_support_size, kernel_regularizer=l2)
        ])

        policy_network = Sequential([
            Dense(hidden_neurons, activation='relu', kernel_regularizer=l2),
            Dense(action_size, kernel_regularizer=l2)
        ])

        dynamic_network = Sequential([
            Dense(hidden_neurons, activation='relu', kernel_regularizer=l2),
            Dense(representation_size, activation=representation_activation,
                  kernel_regularizer=l2)
        ])

        reward_network = Sequential([
            Dense(16, activation='relu', kernel_regularizer=l2),
            Dense(1, kernel_regularizer=l2)
        ])

        super().__init__(representation_network, value_network, policy_network, dynamic_network, reward_network)

    def _value_transform(self, value_support: np.array) -> float:
        """
        first, computer expected value from the discrete support
        then, apply transformation
        :param value_support:
        :return:
        """

        value = self._softmax(value_support)
        value = np.dot(value, range(self.value_support_size))
        return np.asscalar(value) ** 2

    def _reward_transform(self, reward: np.array) -> float:
        return np.asscalar(reward)

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        conditioned_hidden = np.concatenate((hidden_state, np.eye(self.action_size)[action.index]))
        return np.expand_dims(conditioned_hidden, axis=0)

    @staticmethod
    def _softmax(values):
        values_exp = np.exp(values - np.max(values))
        return values_exp / np.sum(values_exp)

    def get_training_steps(self):
        pass