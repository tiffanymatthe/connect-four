#!/usr/bin/env python
import numpy as np
from DataGenerator import DataGenerator


class C4Game(object):
    def __init__(self, history=None) -> None:
        self.history = history or []
        self.child_visits = []
        self.num_actions = 7
        self.num_col = 7
        self.col_height = 6

        self.__winner = -100

        self.player1 = 1
        self.player2 = 0
        self.empty = -1
        self.win_num = 4  # need 4 tokens in a line to win

        if history is None:
            self.history.append(
                self.empty * np.ones((self.col_height, self.num_col)).astype(int))

    def terminal(self) -> bool:
        """Returns True if the game has reached a terminal state (win, loss, draw). False otherwise."""
        last_state = self.history[-1]
        if (last_state == self.player1).sum() < self.win_num and (last_state == self.player2).sum() < self.win_num:
            return False

        full_board = not (last_state == self.empty).any()
        if full_board:
            self.__winner = 0
            return True

        arr_list = last_state.transpose().tolist()  # columns
        arr_list.extend(last_state.tolist())  # rows
        diagonal_indices = range(-2, 4)
        for idx in diagonal_indices:
            diag1 = np.diagonal(last_state, offset=idx)
            diag2 = np.diagonal(np.fliplr(last_state), offset=idx)
            arr_list.append(diag1.tolist())  # diagonals
            arr_list.append(diag2.tolist())  # diagonals

        p1_win = self.__is_win_state_in_list(arr_list, self.player1)
        p2_win = self.__is_win_state_in_list(arr_list, self.player2)

        if p1_win:
            self.__winner = self.player1
        if p2_win:
            self.__winner = self.player2

        return p1_win or p2_win


    def terminal_value(self, to_play):
        """Returns 1 if player to_play wins, 0 if tied, -1 if otherwise. Game needs to be terminal."""
        if self.__winner == -100:
            raise Exception("Run terminal() first to ensure game is terminal.")
        if self.__winner == to_play:
            return 1
        if self.__winner == 0:
            return 0
        else:
            return -1

    def legal_actions(self):
        """Returns legal actions of latest game state as a list of open column indices."""
        last_state = self.history[-1]
        unfilled_cols = self.__get_unfilled_cols(last_state)
        return np.where(unfilled_cols != -100)[0].tolist()

    def clone(self):
        return C4Game(list(self.history))

    def apply(self, action):
        """action: col number to add token"""
        last_state = self.history[-1]
        unfilled_cols = self.__get_unfilled_cols(last_state)
        if self.__is_col_full(unfilled_cols, action):
            raise ValueError(
                "Invalid action {}. Column is full.".format(action))
        new_state = last_state.copy()
        new_state[unfilled_cols[action], action] = self.to_play()
        self.history.append(new_state)

    def store_search_statistics(self, root):
        sum_visits = sum(
            child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count /
            sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index: int):
        return DataGenerator.get_nn_input(self.history[state_index], self.to_play())

    def make_target(self, state_index: int):
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self):
        """
        This method returns the player id of which player's turn it is 
        """
        return len(self.history) % 2

    def __find_longest_seq(self, arr, val) -> int:
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

    def __is_win_state_in_list(self, arr_list, token) -> bool:
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
            if (arr == token).sum() < self.win_num:
                continue
            else:
                if self.__find_longest_seq(arr, token) >= self.win_num:
                    return True
        return False

    def __get_unfilled_cols(self, state):
        """
        Returns numpy array of length num_col indicating first empty row (bottom up) in each col.
        If the ith column is full, array[i] = -100.
        """
        cols = state.transpose().tolist()
        unfilled_cols = np.zeros(self.num_col).astype(int)
        for i, col in enumerate(cols):
            try:
                unfilled_cols[i] = len(col) - 1 - col[::-1].index(-1)
            except ValueError:
                unfilled_cols[i] = -100
        return unfilled_cols

    def __is_col_full(self, unfilled_cols, col_num: int) -> bool:
        """Returns True if the column is full, False otherwise."""
        if col_num < 0 or col_num >= self.num_col:
            raise ValueError("Invalid column number {}".format(col_num))
        return unfilled_cols[col_num] == -100
