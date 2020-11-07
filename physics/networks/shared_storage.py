from tensorflow.keras import optimizers
from physics.networks import BaseNetwork, UniformNetwork, AbstractNetwork

class SharedStorage(object) :
    """save our network"""

    def __init__(self, network: BaseNetwork, uniform_network: UniformNetwork, optimizer: optimizers):
        self._networks = {}
        self.current_network = network
        self.uniform_network = uniform_network
        self.optimizer = optimizer

    def latest_network(self) -> AbstractNetwork:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        return self.uniform_network

    def save_network(self, step: int, network: BaseNetwork):
        self._networks[step] = network