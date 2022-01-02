#!/usr/bin/env python
import copy
import itertools
import random
from collections import namedtuple

import numpy as np
import math
from MCTS import MCTS
from Node import Node


class Alpha_beta:

    def __init__(self):
        self.evaluationTable = np.array([[3, 4, 5, 7, 5, 4, 3],
                                         [4, 6, 8, 10, 8, 6, 4],
                                         [5, 8, 11, 13, 11, 8, 5],
                                         [5, 8, 11, 13, 11, 8, 5],
                                         [4, 6, 8, 10, 8, 6, 4],
                                         [3, 4, 5, 7, 5, 4, 3]])
        self.rows = 6
        self.cols = 7

    # here is where the evaluation table is called
    def evaluateContent(self, node):
        """
        The numbers in the table above represent how many four conncted positions there
        can be including that space. For example, the first entry 3 represents the connected
        four along the diagonal, horizontal, and vertical. The higher the number, the more
        useful that space is. 

        This function will return value < 0 if the player with marker 'X' is likely to win, 
        value = 0 in case of a draw, and value > 0 if player with marker 'O' is likely to win.

        utility value below is 138 since sum of all entries in the table above is 276 = 2 x 138
        """

        utility = 138
        sum = 0
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                if (node.current_state[i][j] == 0):
                    sum += node.current_state[i][j]
                elif (node.current_state[i][j] == 1):
                    sum -= node.current_state[i][j]
        return utility + sum

    # node is the current node in the game

    def alpha_beta_pruning(self, state, game, d=4, cutoff_test=None, eval_fn=None):
        """Search game to determine best action; use alpha-beta pruning.
        This version cuts off search and uses an evaluation function."""

        """
        To determine which player's turn it is, we will look at the unfilled columns.
        If there are odd filled columns then it is player 2's turn otherwise 1.
        Player 1 is 1 and player 2 is 0 as defined in C4Game.py
        """
        player = game.to_play()

        # Functions used by alpha_beta

        def max_value(state, alpha, beta, depth):
            if cutoff_test(state, depth):
                return eval_fn(state)
            v = -np.inf
            for a in game.legal_actions():
                # Previously was the following line where the game.result returned
                # the utility or the reward values of the current game state
                # v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
                game.apply(a)
                toPlayResult = game.to_play()
                result = game.terminal_value(toPlayResult)
                v = max(v, min_value(result, alpha, beta, depth + 1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, alpha, beta, depth):
            if cutoff_test(state, depth):
                return eval_fn(state)
            v = np.inf

            for a in game.legal_actions():
                game.apply(a)
                toPlayResult = game.to_play()
                result = game.terminal_value(toPlayResult)
                v = min(v, max_value(result, alpha, beta, depth + 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        # Body of alpha_beta_cutoff_search starts here:
        # The default test cuts off at depth d or at a terminal state
        cutoff_test = (cutoff_test or (
            lambda state, depth: depth > d or game.terminal()))
        eval_fn = eval_fn or (lambda state: game.terminal_value(player))
        best_score = -np.inf
        beta = np.inf
        best_action = None
        for a in game.legal_actions():
            game.apply(a)
            toPlayResult = game.to_play()
            result = game.terminal_value(toPlayResult)
        v = min_value(result, best_score, beta, 1)
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


def play_against_MCTS(iterations):
    ab_tree = Alpha_beta()
    mcts_tree = MCTS()
    board = Node()
    board.see_board()
    while True:
        for _ in range(iterations):
            mcts_tree.do_rollout(board)
        board = mcts_tree.choose(board)
        board.see_board()
        if (board.is_terminal()):
            break
        # this needs to return a Node, same Node as what MCTS returns.
        board = ab_tree.alpha_beta_pruning(board, board)
        board.see_board()
        if (board.is_terminal()):
            break

    winner = board.get_winner()
    print("Winner is {}: {}".format(winner, board.colors[winner]))
    print("1 is MCTS, 0 is AlphaBeta")


if __name__ == "__main__":
    play_against_MCTS(100)
