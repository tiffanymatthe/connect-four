#!/usr/bin/env python
from Network import Network
from NetworkTraining import NetworkTraining
from C4Config import C4Config
from C4Game import C4Game
from SharedStorage import SharedStorage
from ReplayBuffer import ReplayBuffer

from timeit import default_timer as timer
import cProfile
import pstats

def time_game():
    network = Network()
    config = C4Config()
    start = timer()
    NetworkTraining.play_game(config, network)
    end = timer()
    print(end - start)

def profile_game():
    # https://stackoverflow.com/a/68805743
    with cProfile.Profile() as pr:
        time_game()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    # Now you have two options, either print the data or save it as a file
    # stats.print_stats() # Print The Stats
    stats.dump_stats("game_profile.prof") # Saves the data in a file, can me used to see the data visually
    # to view data, use pip install snakeviz, then in terminal type snakeviz game_profile.prof.

def profile_inference():
    network = Network()
    game = C4Game()
    image = game.make_image(-1)
    with cProfile.Profile() as pr:
        value, policy = network.inference(image)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats("inference_profile.prof")
    print(value)
    print(policy)

def train_network():
    config = C4Config()
    NetworkTraining.alphazero(config)

if __name__ == "__main__":
    # profile_inference()
    train_network()
    # profile_game()