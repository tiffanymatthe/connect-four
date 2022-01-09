#!/usr/bin/env python
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))
import numpy as np
from MCTS import MCTS
from DataGenerator import DataGenerator
from NetworkTraining import NetworkTraining
from C4Config import C4Config
from Node import Node
import tensorflow as tf
from Network import Network
from SelfPlay import SelfPlay
from C4Game import C4Game
from Losses import Losses
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train_network(model_name, load=False):
    config = C4Config(model_name)
    return NetworkTraining.alphazero(config, load)


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


def play_against_model(network: Network):
    board = Node()
    board.see_board()
    while True:
        board = get_player_move(board)
        board.see_board()
        if (board.is_terminal()):
            break
        image = DataGenerator.get_nn_input(board.current_state, board.turn, True)
        value, policy = network.inference(image)
        print(value)
        print(policy)
        board = board.move(np.argmax(policy))
        # game = C4Game()
        # game.initialize_history(board.current_state, board.turn)
        # action, root = SelfPlay.run_mcts(C4Config(), game, network)
        # board = board.move(action)
        board.see_board()
        if (board.is_terminal()):
            break

    winner = board.get_winner()
    print("Winner is {}: {}".format(winner, board.colors[winner]))

def mcts_against_model(network: Network):
    board = Node()
    board.see_board()
    tree = MCTS()
    while True:
        for _ in range(C4Config().num_simulations):
            tree.do_rollout(board)
        board = tree.choose(board)
        board.see_board()
        if (board.is_terminal()):
            break
        game = C4Game()
        game.initialize_history(board.current_state, board.turn)
        action, root = SelfPlay.run_mcts(C4Config(), game, network)
        board = board.move(action)
        board.see_board()
        if (board.is_terminal()):
            break

    winner = board.get_winner()
    print("Winner is {}: {}".format(winner, board.colors[winner]))


if __name__ == "__main__":
    version = '12'
    # final_network = train_network(version, load=False)
    # # model = tf.keras.models.load_model("models/model_3")
    # network = Network(C4Config())
    # play_against_model(network)
    losses = Losses()
    losses.get_losses(version)
    losses.plot_losses()
