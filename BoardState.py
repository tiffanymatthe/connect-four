#!/usr/bin/env python
import numpy as np
import random
from PlayerId import PlayerId

class BoardState:
    def __init__(self) -> None:
        self.num_col = 7
        self.col_height = 6
        # 0 is red, 1 is yellow, -1 is black
        self.current_state = -np.ones((self.col_height, self.num_col)).astype(int)
        self.colors = {
            -1: "\033[40m - \033[0m", # black
            PlayerId.PLAYER_0: "\033[41m o \033[0m", # red
            PlayerId.PLAYER_1: "\033[43m x \033[0m" # yellow
        }
        self.full_marker = -1

        # full_marker if completely filled, else first unfilled row index
        self.unfilled_cols = np.ones(self.num_col).astype(int) * (self.col_height - 1)

    def get_copy(self):
        new_board = BoardState()
        new_board.current_state = self.current_state.copy()
        new_board.unfilled_cols = self.unfilled_cols.copy()
        return new_board

    def get_color_coded_background(self, i):
        """
        Get marker to print for a certain player or empty space.
        """
        return self.colors[i]
    
    def print_a_ndarray(self, map1, row_sep=" "):
        """
        Prints array.
        """
        # https://stackoverflow.com/questions/56496731/coloring-entries-in-an-matrix-2d-numpy-array/56497272
        n, m = map1.shape
        vertical_padding = 2
        m = m + vertical_padding # for vertical axis
        fmt_str = "\n".join([row_sep.join(["{}"]*m)]*n)
        column_labels = ['0','1','2','3','4','5','6']
        row_labels = ['a','b','c','d','e','f']

        map1 = np.pad(map1, [(0,0), (vertical_padding,0)])

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
        coloured_board = np.vectorize(self.get_color_coded_background)(display_board)
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
            The column number to check if full. Needs to be valid.

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
        print("It's a draw.")
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
        idx_pairs = np.where(np.diff(np.hstack(([False],arr==val,[False]))))[0].reshape(-1,2)
        seq_lengths = np.diff(idx_pairs,axis=1)
        if len(seq_lengths) > 0:
            return max(np.diff(idx_pairs,axis=1))
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

        win = self.is_win_state_in_list(arr_list, token)
        if win:
            print("Player {} won!".format(player_id))
        return win

    def move(self, player_id: int, col: int):
        """
        Makes a move for player_id.
        ...
        Parameters
        ----------
        player_id : int
            The player number. 0 or 1.
        col : int
            The column number to play in, 0 <= col < self.col_num
        
        Raises
        ------
        ValueError
            If move is illegal.

        Returns new BoardState.
        """
        self.check_col_valid(col)
        if self.is_col_full(col):
            raise ValueError("Invalid move. Column is full.")
        new_board = self.get_copy()
        if player_id == PlayerId.PLAYER_0:
            new_board.state[new_board.unfilled_cols[col], col] = PlayerId.PLAYER_0
        else:
            new_board.state[new_board.unfilled_cols[col], col] = PlayerId.PLAYER_1
        new_board.unfilled_cols[col] -= 1
        return new_board

    def get_winner(self) -> int:
        """
        Returns -1 if there is no winner.
        """
        if self.is_winner(PlayerId.PLAYER_0):
            return PlayerId.PLAYER_0
        if self.is_winner(PlayerId.PLAYER_1):
            return PlayerId.PLAYER_1
        else:
            return -1

    def is_game_over(self):
        return self.get_winner != -1

    def get_legal_actions(self) -> list:
        """
        Returns list of column indices that are not full.
        """
        return np.where(self.unfilled_cols != self.full_marker)[0].tolist()

    def game_result(self, player_id):
        """
        Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        tie or a loss.
        """
        winning_id = self.get_winner()
        if winning_id == -1 and self.is_draw():
                return 0
        if winning_id == player_id:
            return 1
        else:
            return -1