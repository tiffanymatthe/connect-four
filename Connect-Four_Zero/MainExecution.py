#!/usr/bin/env python
from Network import Network
from NetworkTraining import NetworkTraining
from C4Config import C4Config
from SharedStorage import SharedStorage
from ReplayBuffer import ReplayBuffer

from timeit import default_timer as timer

def time_game():
    network = Network()
    config = C4Config()
    start = timer()
    NetworkTraining.play_game(config, network)
    end = timer()
    print(end - start)

def train_network():
    config = C4Config()
    NetworkTraining.alphazero(config)

if __name__ == "__main__":
    time_game()