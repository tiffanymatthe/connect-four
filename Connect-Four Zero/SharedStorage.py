#!/usr/bin/env python
import Network


class SharedStorage():
    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.iterkeys())]
        else:
            return make_uniform_network()  # policy -> uniform, value -> 0.5

    def save_network(self, step: int, network: Network):
        self._networks[step] = network


def make_uniform_network() -> Network:
    pass
