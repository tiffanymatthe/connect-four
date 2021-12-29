import copy
import itertools
import random
from collections import namedtuple

import numpy as np

from utils import vector_add
import math
import Node

class Alpha_beta:

    #node is the current node in the game
    def alpha_beta_cutoff_search(node, d=4, cutoff_test=None, eval_fn=None):
        """Search game to determine best action; use alpha-beta pruning.
        This version cuts off search and uses an evaluation function."""

        player = 0

        # Functions used by alpha_beta
        def max_value(node, alpha, beta, depth):
            if cutoff_test(node, depth):
                return eval_fn()
            v = -np.inf
            for a in node.find_children():
                v = max(v, min_value(node.reward, alpha, beta, depth + 1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(node, alpha, beta, depth):
            if cutoff_test(node, depth):
                return eval_fn(node)
            v = np.inf
            for a in game.actions(node):
                v = min(v, max_value(node.reward, alpha, beta, depth + 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        # Body of alpha_beta_cutoff_search starts here:
        # The default test cuts off at depth d or at a terminal node
        cutoff_test = (cutoff_test or (lambda node, depth: depth > d or node.is_terminal()))
        eval_fn = eval_fn or (lambda node: node.reward())
        best_score = -np.inf
        beta = np.inf
        best_action = None
        for a in node.find_children():
            v = min_value(game.result(node, a), best_score, beta, 1)
            if v > best_score:
                best_score = v
                best_action = a
        return best_action


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

def play_game(iterations):
    tree = MCTS()
    board = Node.Node()
    board.see_board()
    while True:
        board = get_player_move(board)
        board.see_board()
        if (board.is_terminal()):
            break
        for _ in range(iterations):
            tree.do_rollout(board)
        board = tree.choose(board)
        board.see_board()
        if (board.is_terminal()):
            break

    winner = board.get_winner()
    print("Winner is {}: {}".format(winner, board.colors[winner]))

if __name__ == "__main__":
    play_game(200)
