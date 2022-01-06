#!/usr/bin/env python
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))
import numpy as np
from Network import Network
from DataGenerator import DataGenerator
from NetworkTraining import NetworkTraining
from C4Config import C4Config
from C4Game import C4Game
from Node import Node
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pstats
import cProfile


def train_network(model_name):
    config = C4Config(model_name)
    return NetworkTraining.alphazero(config)


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
        board = get_player_move(board)
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

    winner = board.get_winner()
    print("Winner is {}: {}".format(winner, board.colors[winner]))


if __name__ == "__main__":
    version = '1'
    final_network = train_network(version)
    # # model = tf.keras.models.load_model("models/model_3")
    # model = Network(C4Config()).cnn.model
    # play_against_model(model)
    # losses = Losses(f'losses_{version}')
    # losses.plot_losses()
