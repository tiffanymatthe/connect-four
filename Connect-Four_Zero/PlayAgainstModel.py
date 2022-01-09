#!/usr/bin/env python
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))
from Node import Node
from Network import Network
from DataGenerator import DataGenerator
import numpy as np
from C4Config import C4Config

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
        value, policy = network.inference(DataGenerator.get_nn_input(board.current_state, board.turn))
        board.move(np.argmax(policy))
        board.see_board()
        if (board.is_terminal()):
            break

    winner = board.get_winner()
    print("Winner is {}: {}".format(winner, board.colors[winner]))

if __name__ == "__main__":
    version = '33'
    network = Network(C4Config(), model_name=version)
    play_against_model()