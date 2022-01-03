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
        self.depth = 6

    # here is where the evaluation table is called
    def eval_fn(self, node):
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

    
    #Method used in alpha beta pruning search 
    def maxfunction(self, node, depth, alpha, beta):
        opponent = node.SwitchPlayer(node.to_play())
        node.turn = opponent
        legalmoves = node.unfilled_cols
        # print("list of legal moves in max ", legalmoves)
        if (depth==0) or len(node.unfilled_cols)==0:
            return self.eval_fn(node)
        value= -np.inf
        for col in legalmoves:
            # print("COOLUMNNNN IN MAXXX FUNCTION", col)
            if node.is_terminal():
                break
            if not node.is_col_full(col):
                newboard = node.move(col)
                value = max(value, self.minfunction(newboard, depth-1, alpha, beta))
                # print("LENGTHHH OF UNFILLED COLUMNS IN MAXX FUNCTION", len(node.unfilled_cols))
                newboard = node.unmove(col)
            # print("LENGTHHH OF UNFILLED COLUMNS IN new board MAXX FUNCTION", len(newboard.unfilled_cols))
                if value >= beta:
                    return value
            alpha = max(alpha, value)
        return value
    
    def minfunction(self, node, depth, alpha, beta):
        player = node.SwitchPlayer(node.to_play())
        node.turn = player
        legalmoves = node.unfilled_cols
        # print("List of legal moves ", legalmoves)
        if (depth==0) or len(node.unfilled_cols)==0:
            return self.eval_fn(node)
        value = np.inf
        for col in legalmoves:
            if node.is_terminal():
                break
            if not node.is_col_full(col):
                newboard = node.move(col)
                value = min(value, self.maxfunction(newboard, depth-1, alpha, beta))
                newboard = node.unmove(col)
                if value <= alpha:
                    return value
            beta = min(beta, value)
        return value
    

    def alphabetapruning(self, node, depth, alpha, beta): 
        #This is the alphabeta-function modified from: 
        #https://github.com/msaveski/connect-four
        values = []
        cols = []
        value = -np.inf
        for col in node.unfilled_cols:
            if node.is_terminal():
                break
            node = node.move(col)
            value = max(value, self.minfunction(node, depth-1, alpha, beta))
            values.append(value)
            cols.append(col)
            node = node.unmove(col)
        largestvalue= max(values)
        print(cols)
        print(values)
        for i in range(len(values)):
            if largestvalue==values[i]:
                position = cols[i]
                return largestvalue, position

    def searching_function(self, node, depth):
        #This function update turn to opponent and calls alphabeta (main algorithm) 
        #and after that update 
        #new board (add alphabeta position to old board) and returns new board.
        newboard = copy.deepcopy(node)
        value, position = self.alphabetapruning(newboard, depth, -np.inf, np.inf)
        print("TOOOOO MOVE ISSS COLUMNNNN" + str(position))
        board = node.move(position)
        return board


    #This is the older version of alpha_beta which never ran to its full extent
    #and had a lot of bugs. It is still here tho, just for reference.
    def alpha_beta_pruning(self, node, d=4, cutoff_test=None, eval_fn=None):
        """Search game to determine best action; use alpha-beta pruning.
        This version cuts off search and uses an evaluation function."""

        """
        To determine which player's turn it is, we will look at the unfilled columns.
        If there are odd filled columns then it is player 2's turn otherwise 1.
        Player 1 is 1 and player 2 is 0 as defined in C4Game.py
        """
        player = node.to_play()

        # Functions used by alpha_beta

        def max_value(node, alpha, beta, depth):
            if cutoff_test(node, depth):
                return eval_fn(node)
            v = -np.inf
            scratch_game = node.get_copy()
            for a in scratch_game.unfilled_cols:
                # Previously was the following line where the game.result returned
                # the utility or the reward values of the current game state
                # v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
                
                _, result = scratch_game.result(a)
                v = max(v, min_value(result, alpha, beta, depth + 1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(node, alpha, beta, depth):
            if cutoff_test(node, depth):
                return eval_fn(node)
            v = np.inf

            scratch_game = node.get_copy()
            for a in scratch_game.unfilled_cols:
                
                _, result = scratch_game.result(a)
                v = min(v, max_value(result, alpha, beta, depth + 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        # Body of alpha_beta_cutoff_search starts here:
        # The default test cuts off at depth d or at a terminal state
        cutoff_test = (cutoff_test or (
            lambda node, depth: depth > d or node.is_terminal()))
        eval_fn = eval_fn or (lambda node: node.utility(player))
        best_score = -np.inf
        beta = np.inf
        best_action = None

        scratch_game = node.get_copy()
        for a in scratch_game.unfilled_cols:
            _, result = scratch_game.result(a)
            v = min_value(result, best_score, beta, 1)
            if v > best_score:
                best_score = v
                best_action = a

        return node.move(best_action)


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
    print("initialized alpha beta")
    mcts_tree = MCTS()
    board = Node()
    board.see_board()
    while True:
        # this needs to return a Node, same Node as what MCTS returns.
        # board = ab_tree.alpha_beta_search(board)
        for _ in range(iterations):
            mcts_tree.do_rollout(board)
        board = mcts_tree.choose(board)
        board.see_board()
        print("mcts executed")
        if (board.is_terminal()):
            break
        board = ab_tree.searching_function(board, 8) #Here is AI's move. Takes as input current table (board), depth and opponents mark. Output should be new gameboard with AI's move.
        print("executed alpha beta")
        board.see_board()
        if (board.is_terminal()):
            break

    winner = board.get_winner()
    print("Winner is {}: {}".format(winner, board.colors[winner]))
    print("x is MCTS, o is AlphaBeta")


if __name__ == "__main__":
    play_against_MCTS(100)
