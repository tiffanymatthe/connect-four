#!/usr/bin/env python
from Network import Network
from multiprocessing import Lock

class SharedStorage(object):
    def __init__(self):
        self.mutex = Lock()
        self._networks = {}

    def latest_network(self) -> Network:
        with self.mutex:
            if self._networks:
                return self._networks[max(self._networks.keys())]
            else:
                return make_uniform_network()

    def save_network(self, step: int, network: Network):
        with self.mutex:
            self._networks[step] = network.clone_network()

    def get_num_networks(self):
        return len(self._networks)


def make_uniform_network() -> Network:
    # TODO: should have weights which give uniform policy and value 0.5
    return Network()
