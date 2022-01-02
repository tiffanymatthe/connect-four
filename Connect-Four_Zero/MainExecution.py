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

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

storage_global = SharedStorage()

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

def play_and_save(config, replay_buffer: ReplayBuffer, storage: SharedStorage):
    for _ in range(2):
        network = storage.latest_network()
        print("Num of networks: {}".format(storage.get_num_networks()))
        game = NetworkTraining.play_game(config, network)
        replay_buffer.save_game(game)

def play_and_save_global(config, replay_buffer: ReplayBuffer):
    for _ in range(2):
        network = storage_global.latest_network()
        print("Num of networks: {}".format(storage_global.get_num_networks()))
        game = NetworkTraining.play_game(config, network)
        replay_buffer.save_game(game)

def test_shared_storage():
    config = C4Config()
    BaseManager.register('ReplayBuffer', ReplayBuffer) # process writes to this
    BaseManager.register('SharedStorage', SharedStorage) # process reads from this, maybe inefficient
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(config)    
    storage = manager.SharedStorage()

    process_list = []
    for _ in range(5):
        p = Process(target=play_and_save, args=(config, replay_buffer, storage))
        p.start()
        process_list.append(p)

    for i in range(5):
        net = Network()
        storage.save_network(i, net)

    for process in process_list:
        process.join()

    print(replay_buffer.get_buffer_size())
    print(storage.get_num_networks())

def test_shared_storage_global_storage():
    config = C4Config()
    BaseManager.register('ReplayBuffer', ReplayBuffer) # process writes to this
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(config)

    process_list = []
    for _ in range(5):
        p = Process(target=play_and_save_global, args=(config, replay_buffer))
        p.start()
        process_list.append(p)

    for i in range(5):
        net = Network()
        storage_global.save_network(i, net)

    for process in process_list:
        process.join()

    print(replay_buffer.get_buffer_size())
    print(storage_global.get_num_networks())

def profile_multiprocessing():
    with cProfile.Profile() as pr:
        test_shared_storage_global_storage()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats("multiprocessing_global_profile.prof")

def train_network():
    config = C4Config()
    NetworkTraining.alphazero(config)

if __name__ == "__main__":
    # profile_inference()
    # train_network()
    # profile_game()
    # test_shared_storage()
    profile_multiprocessing()