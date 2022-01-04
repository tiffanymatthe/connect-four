#!/usr/bin/env python
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

from multiprocessing.managers import BaseManager
from multiprocessing import Process
import tensorflow as tf
import pstats
import cProfile
from timeit import default_timer as timer
from Node import Node
from ReplayBuffer import ReplayBuffer
from SharedStorage import SharedStorage
from C4Game import C4Game
from C4Config import C4Config
from NetworkTraining import NetworkTraining
from Network import Network
from DataGenerator import DataGenerator
import numpy as np

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
    # Saves the data in a file, can me used to see the data visually
    stats.dump_stats("profiles/game_profile.prof")
    # to view data, use pip install snakeviz, then in terminal type snakeviz game_profile.prof.


def profile_inference():
    network = Network()
    game = C4Game()
    image = game.make_image(-1)
    with cProfile.Profile() as pr:
        value, policy = network.inference(image)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats("profiles/inference_profile.prof")
    print(value)
    print(policy)


def play_and_save(config, replay_buffer: ReplayBuffer, storage: SharedStorage):
    for _ in range(2):
        network = storage.latest_network()
        print("Num of networks: {}".format(storage.get_num_networks()))
        game = NetworkTraining.play_game(config, network)
        replay_buffer.save_game(game)


def get_buffer_storage_from_base_manager(config):
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    BaseManager.register('SharedStorage', SharedStorage)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(config)
    storage = manager.SharedStorage()
    return replay_buffer, storage


def test_shared_storage():
    config = C4Config()
    replay_buffer, storage = get_buffer_storage_from_base_manager(config)
    process_list = []
    for _ in range(5):
        p = Process(target=play_and_save, args=(
            config, replay_buffer, storage))
        p.start()
        process_list.append(p)

    for i in range(5):
        net = Network()
        storage.save_network(i, net)

    for process in process_list:
        process.join()

    print(replay_buffer.get_buffer_size())
    print(storage.get_num_networks())


def profile_multiprocessing():
    with cProfile.Profile() as pr:
        test_shared_storage()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats("profiles/multiprocessing_profile.prof")


def train_network():
    config = C4Config()
    return NetworkTraining.alphazero(config)


def print_summary():
    network = Network()
    network.print_model_summary()


def get_int_input(message):
    user_input = input(message)
    while(not user_input.isnumeric()):
        user_input = input("Not an int, try again: ")
    return int(user_input)


def get_player_move(board):
    if (board.is_draw()):
        raise ValueError("DRAW")
    col = get_int_input("Please enter a column index: ")
    while(col < 0 or col >= board.num_col):
        col = get_int_input("Not in range, try again: ")
    success = False
    while (not success):
        try:
            board = board.move(col)
            success = True
        except ValueError:
            col = get_int_input("Col is full, try another one: ")
            board = board.move(col)

    return board


def play_against_model(model):
    board = Node()
    board.see_board()
    while True:
        board=get_player_move(board)
        board.see_board()
        if (board.is_terminal()):
            break
        image = DataGenerator.get_nn_input(board.current_state, board.turn)
        print(np.moveaxis(image, -1, 0))
        value, policy = model(image[None], training=False)
        print("value: {}, policies: {}".format(value, policy))
        board = board.move(np.argmax(policy))
        board.see_board()
        if (board.is_terminal()):
            break

    winner=board.get_winner()
    print("Winner is {}: {}".format(winner, board.colors[winner]))

if __name__ == "__main__":
    # profile_inference()
    final_network = train_network()
    # final_network.model.save("models/model_3")
    # print_summary()
    # profile_game()
    # test_shared_storage()
    # profile_multiprocessing()

    # model=tf.keras.models.load_model('Connect-Four_Zero/models/model_3')
    # # network = Network()
    # # network.model = model
    # # NetworkTraining.play_game(C4Config(), network)
    # play_against_model(model)
