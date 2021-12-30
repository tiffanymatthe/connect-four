#!/usr/bin/env python
import copy
import itertools
import random
from collections import namedtuple

import numpy as np
import math
import Node

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
    def evaluateContent(node):
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


    #node is the current node in the game
    def alpha_beta_pruning(state, node, d=4, cutoff_test=None, eval_fn=None):
        """Search game to determine best action; use alpha-beta pruning.
        This version cuts off search and uses an evaluation function."""

        player = 0 #how to get the current player's playing ID????

        def get_reward(self, node, child):
            """
            Need to modify this function but this returns the reward value 
            associated with a particular move. 
            """
            invert_reward = True
            while True:
                if node.is_terminal():
                    reward = node.reward()
                    return 1 - reward if invert_reward else reward
                node = child
                invert_reward = not invert_reward
            print("out of the while loop")

        # Functions used by alpha_beta
        def max_value(state, alpha, beta, depth):
            if cutoff_test(state, depth):
                return eval_fn(state)
            v = -np.inf
            for child in node.find_children():
                #result in this case represents the final outcome of choosing a specified move
                result = self.get_reward(node, child)
                v = max(v, min_value(result, alpha, beta, depth + 1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, alpha, beta, depth):
            if cutoff_test(state, depth):
                return eval_fn(state)
            v = np.inf
            for child in node.find_children():
                result = self.get_reward(node, child)
                v = min(v, max_value(result, alpha, beta, depth + 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        # Body of alpha_beta_cutoff_search starts here:
        # The default test cuts off at depth d or at a terminal state
        cutoff_test = (cutoff_test or (lambda state, depth: depth > d or node.is_terminal()))
        eval_fn = eval_fn or (lambda state: self.evaluateContent(node))
        best_score = -np.inf
        beta = np.inf
        best_action = None
        for child in node.find_children():
            result = self.get_reward(node, child)
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

def play_game():
    tree = Alpha_beta()
    board = Node.Node()
    board.see_board()
    while True:
        board = get_player_move(board)
        board.see_board()
        if (board.is_terminal()):
            break
        board = tree.alpha_beta_pruning(board, board)
        board.see_board()
        if (board.is_terminal()):
            break

    winner = board.get_winner()
    print("Winner is {}: {}".format(winner, board.colors[winner]))

if __name__ == "__main__":
    play_game()