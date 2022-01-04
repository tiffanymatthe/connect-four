#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from C4Config import C4Config
from Network import Network
from SharedStorage import SharedStorage


def modify_weights(network: Network):
    weights = network.get_weights()
    print("Initial weight of network 1: {}".format(weights[0][0][0][0][0]))
    for layer in network.model.layers:
        layer.build(layer.input_shape)
    weights = network.get_weights()
    print("Modified weight of network 1: {}".format(weights[0][0][0][0][0]))

def check_saving():
    storage = SharedStorage()
    network = Network()
    storage.save_network(0, network)
    print("Network 0 address: {}".format(storage.latest_network()))
    modify_weights(network)
    storage.save_network(1, network)
    print("Network 1 address: {}".format(storage.latest_network()))
    first_one = storage._networks[0]
    weights = first_one.get_weights()
    print("Weight of network 0: {}".format(weights[0][0][0][0][0]))

if __name__ == "__main__":
    check_saving()
