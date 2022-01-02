#!/usr/bin/env python
import multiprocessing
from Network import Network
from NetworkTraining import NetworkTraining
from C4Config import C4Config
from C4Game import C4Game
from SharedStorage import SharedStorage
from ReplayBuffer import ReplayBuffer

from timeit import default_timer as timer
import cProfile
import pstats

import threading
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

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

def play_and_save(config, replay_buffer: ReplayBuffer, network):
    game = NetworkTraining.play_game(config, network)
    replay_buffer.save_game(game)

def test_shared_storage():
    config = C4Config()
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(config)    
    storage = SharedStorage()
    network = storage.latest_network()

    process_list = []
    for _ in range(5):
        p = Process(target=play_and_save, args=(config, replay_buffer, network))
        p.start()
        process_list.append(p)

    for process in process_list:
        process.join()

    print(len(replay_buffer.buffer))

def train_network():
    config = C4Config()
    NetworkTraining.alphazero(config)

if __name__ == "__main__":
    # profile_inference()
    # train_network()
    # profile_game()
    test_shared_storage()