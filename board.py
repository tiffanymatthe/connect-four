#!/usr/bin/env python
import numpy as np

class board:
    def __init__(self) -> None:
        self.num_col = 7
        self.col_height = 6
        # 1 is red, -1 is yellow, 0 is black
        self.current_state = np.zeros((self.col_height, self.num_col)).astype(int)
        self.colors = {
            0: "\033[40m - \033[0m", # black
            1: "\033[41m o \033[0m", # red
            -1: "\033[43m x \033[0m" # yellow
        }

        # -1 if completely filled
        self.unfilled_states = np.ones(self.num_col).astype(int) * (self.col_height - 1)

    def get_color_coded_background(self, i):
        # https://stackoverflow.com/questions/56496731/coloring-entries-in-an-matrix-2d-numpy-array/56497272
        return self.colors[i]
    
    def print_a_ndarray(self, map, row_sep=" "):
        # https://stackoverflow.com/questions/56496731/coloring-entries-in-an-matrix-2d-numpy-array/56497272
        n, m = map.shape
        fmt_str = "\n".join([row_sep.join(["{}"]*m)]*n)
        print(fmt_str.format(*map.ravel()))

    def see_board(self):
        """
        Display board. 'o' represents white, 'x' represents black, '-' represents an empty space.
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
        return self.unfilled_states[col_num] == -1

    def check_col_valid(self, col_num: int) -> None:
        """
        Parameters
        ----------
        col_num : int
            The column number to check if full. Needs to be valid.

        Raises
        ------
        ValueError
            If column is out of bounds of board columns.

        Returns None if ok.
        """
        if col_num < 0 or col_num >= self.num_col:
            raise ValueError("Column {} is out of bounds.".format(col_num))
        return None

    def move(self, player_num: int, col: int) -> bool:
        """
        Makes a move for player_num. Modifies current_state of board.
        ...
        Parameters
        ----------
        player_num : int
            The player number. 0 or 1.
        col : int
            The column number to play in, 0 <= col < self.col_num

        Returns False if move is unplayable.
        """
        self.check_col_valid(col)
        if self.is_col_full(col):
            return False
        if player_num == 0:
            self.current_state[self.unfilled_states[col], col] = 1
        else:
            self.current_state[self.unfilled_states[col], col] = -1
        self.unfilled_states[col] -= 1
        return True
        
        

if __name__ == '__main__':
    board = board()
    board.see_board()
    success = board.move(0, 1)
    board.see_board()
    success = board.move(1, 1)
    board.see_board()
    success = board.move(0, 2)
    board.see_board()
    success = board.move(1, 4)
    board.see_board()