#!/usr/bin/env python
from Network import Network


class SharedStorage(object):
    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            return make_uniform_network()  # policy -> uniform, value -> 0.5

    def save_network(self, step: int, network: Network):
        self._networks[step] = network


def make_uniform_network() -> Network:
    # TODO: should have weights which give uniform policy and value 0.5
    return Network()
