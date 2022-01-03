#!/usr/bin/env python

import random
import numpy as np


class Node():
    """
    https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
    A representation of a single board state for Connect 4.
    MCTS works by constructing a tree of these Nodes.
    """

    def __init__(self) -> None:
        """
        0 = marker for red player
        1 = marker for yellow player
        -1 = marker for empty space

        player 1 always goes first
        """
        self.num_col = 7
        self.col_height = 6
        # 0 is red, 1 is yellow, -1 is black
        self.current_state = - \
            np.ones((self.col_height, self.num_col)).astype(int)
        self.colors = {
            -1: "\033[40m - \033[0m",  # black
            0: "\033[41m o \033[0m",  # red
            1: "\033[43m x \033[0m"  # yellow
        }
        self.full_marker = -1
        self.turn = 1  # either 0 or 1, starts off with 1 always

        # full_marker if completely filled, else first unfilled row index
        self.unfilled_cols = np.ones(self.num_col).astype(
            int) * (self.col_height - 1)

    def updated_unfilled_cols(self):
        """
        Update row indices of unfilled columns in Connect 4.
        """
        cols = self.current_state.transpose().tolist()
        for i, col in enumerate(cols):
            try:
                self.unfilled_cols[i] = len(col) - 1 - col[::-1].index(-1)
            except ValueError:
                self.unfilled_cols[i] = -1

    def get_copy(self):
        """
        Gets copy of node.
        """
        new_node = Node()
        new_node.current_state = self.current_state.copy()
        new_node.unfilled_cols = self.unfilled_cols.copy()
        new_node.turn = self.turn
        return new_node

    def get_color_coded_background(self, i):
        """
        Get marker to print for a certain player or empty space.
        """
        return self.colors[i]

    def print_a_ndarray(self, map1, row_sep=" "):
        """
        Prints array.
        https://stackoverflow.com/questions/56496731/coloring-entries-in-an-matrix-2d-numpy-array/56497272
        """
        n, m = map1.shape
        vertical_padding = 2
        m = m + vertical_padding  # for vertical axis
        fmt_str = "\n".join([row_sep.join(["{}"]*m)]*n)
        column_labels = ['0', '1', '2', '3', '4', '5', '6']
        row_labels = ['a', 'b', 'c', 'd', 'e', 'f']

        map1 = np.pad(map1, [(0, 0), (vertical_padding, 0)])

        for i, label in enumerate(row_labels):
            map1[i, 1] = ' | '
            map1[i, 0] = label

        print(fmt_str.format(*map1.ravel()))
        col_axis = '  '.join(map(str, column_labels))
        print('    ' + '―' * 20)
        print('    ' + ' ' + col_axis)

    def see_board(self):
        """
        Display Board. 'o' represents player 0, 'x' represents player 1, '-' represents an empty space.
        """
        display_board = self.current_state
        coloured_board = np.vectorize(
            self.get_color_coded_background)(display_board)
        self.print_a_ndarray(coloured_board, row_sep="")
        print()

    def is_col_full(self, col_num: int) -> bool:
        """
        Parameters
        ----------
        col_num : int
            The column number to check if full. Needs to be valid.

        Returns True if the column is full, False otherwise.
        """
        return self.unfilled_cols[col_num] == self.full_marker

    def check_col_valid(self, col_num: int) -> None:
        """
        Parameters
        ----------
        col_num : int
            The column number to check if valid.

        Raises
        ------
        ValueError
            If column is out of bounds of Board columns.

        Returns None if ok.
        """
        if col_num < 0 or col_num >= self.num_col:
            raise ValueError("Column {} is out of bounds.".format(col_num))
        return None

    def is_draw(self):
        """
        Returns True if it is a draw, False otherwise.
        """
        for c in range(0, self.num_col):
            if not self.is_col_full(c):
                return False
        return True

    def find_longest_seq(self, arr, val):
        """
        Parameters
        ----------
        arr : np.ndarray
            array containing val and other values
        val
            value in array to find the longest unbroken sequence of

        Returns length of longest sequence of val in arr.
        Inspired by https://stackoverflow.com/a/38161867
        """
        idx_pairs = np.where(np.diff(np.hstack(([False], arr == val, [False]))))[
            0].reshape(-1, 2)
        seq_lengths = np.diff(idx_pairs, axis=1)
        if len(seq_lengths) > 0:
            return max(np.diff(idx_pairs, axis=1))
        else:
            return 0

    def is_win_state_in_list(self, arr_list, token):
        """
        Parameters
        ----------
        arr_list : list
            list of arrays
        token
            value to find its longest sequence in each array

        If there is a sequence of length >= 4 of value token in one of the arrays in
        list, then returns True. Else False.
        """
        for arr in arr_list:
            arr = np.array(arr)
            if (arr == token).sum() < 4:
                continue
            else:
                if self.find_longest_seq(arr, token) >= 4:
                    return True
        return False

    def is_winner(self, player_id) -> bool:
        """
        Parameters
        ----------
        player_id : int
            player id to check. Should match PlayerId

        Returns True if player associated to player_id has won. Else False.
        """
        if (self.current_state == player_id).sum() < 4:
            return False
        # add columns
        arr_list = self.current_state.transpose().tolist()
        # add rows
        arr_list.extend(self.current_state.tolist())
        # add diagonals
        diagonal_indices = range(-2, 4)
        for idx in diagonal_indices:
            diag1 = np.diagonal(self.current_state, offset=idx)
            diag2 = np.diagonal(np.fliplr(self.current_state), offset=idx)
            arr_list.append(diag1.tolist())
            arr_list.append(diag2.tolist())

        win = self.is_win_state_in_list(arr_list, player_id)
        return win

    def move(self, col: int):
        """
        Makes a move.
        ...
        Parameters
        ----------
        col : int
            The column number to play in, 0 <= col < self.col_num

        Raises
        ------
        ValueError
            If move is illegal.

        Returns new Node.
        """
        if (self.is_terminal()):
            print(self.current_state)
            raise RuntimeError("Move denied. Board is terminal.")
        self.check_col_valid(col)
        if self.is_col_full(col):
            raise ValueError("Invalid move. Column is full.")
        new_board = self.get_copy()
        new_board.current_state[new_board.unfilled_cols[col], col] = self.turn
        new_board.unfilled_cols[col] -= 1
        new_board.turn = not self.turn
        # self.current_state[self.unfilled_cols[col], col] = self.turn
        # self.unfilled_cols[col] -= 1
        # self.turn = not self.turn
        # self.updated_unfilled_cols()
        return new_board

    def unmove(self, col):
        new_board = self.get_copy()
        new_board.current_state[new_board.unfilled_cols[col], col] = -1 # This function make a move and increases count of moves
        # s.updated_unfilled_cols()
        return self


    def get_winner(self) -> int:
        """
        Returns -1 if there is no winner. Else returns 0 or 1.
        """
        if self.is_winner(0):
            return 0
        if self.is_winner(1):
            return 1
        else:
            return -1

    def find_children(self):
        """
        Returns all possible successors of this board state
        """
        if self.is_terminal():
            return {}
        cols = np.where(self.unfilled_cols != self.full_marker)[0].tolist()
        return set(self.move(c) for c in cols)

    def find_random_child(self):
        """
        Returns random successor of this board state (for more efficient simulation)
        """
        children = self.find_children()
        return random.choice(tuple(children))

    def is_terminal(self):
        """
        Returns True if the node has no children (reached end of game)
        """
        if self.is_draw():
            return True
        return self.get_winner() != -1

    def reward(self):
        if not self.is_terminal():
            self.see_board()
            raise RuntimeError("reward called on nonterminal board")
        if self.get_winner() == self.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board")
        if self.turn == (not self.get_winner()):
            return 0  # Your opponent has just won. Bad.
        if self.get_winner() == -1:
            return 0.5  # Board is a tie
        # The winner is neither True, False, nor -1
        self.see_board()
        raise RuntimeError(f"board has unknown winner type")

    def compute_utility(self):
        """
        This method computes the current utility of the board state. It returns 
        1 if the current player has won, 0 if the opponent won, and -1 if the board
        is a tie. The difference between this method and reward is that this will return 
        the utility even if the node is not terminal and not throw an error.
        """
        if self.get_winner() == 0:
            return 1
        if self.get_winner() == 1:
            return 0
        if self.get_winner() == -1:
            return 0.5

    #Returns the reward associated with the current player (1 if they win and -1 or 0 otherwise)
    def utility(self, player):
        if (self.is_winner(player)):
            return 1
        else:
            if (self.get_winner() == -1):
                return 0.5
            else:
                return 0

    def to_play(self):
        """
        This method returns the value representing which player's turn it is. It
        returns 1 if it's player 1 turn or 0 if it's player 2's turn
        """
        if (len(self.unfilled_cols) % 2):
            return 1
        else:
            return 0
    
    def SwitchPlayer(self, player):
        if player == 1:
            return 0
        elif player == 0:
            return 1
    
    def result(self, col:int):
        try:
            newNode = self.move(col)
        except:
            newNode = self
        return newNode, self.compute_utility()
    