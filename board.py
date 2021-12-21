#!/usr/bin/env python
import numpy as np

class board:
    def __init__(self) -> None:
        self.num_col = 7
        self.col_height = 6
        # 1 is white, 2 is black, 0 is empty
        self.current_state = np.zeros((self.col_height, self.num_col)).astype(int)
        self.colors = {
            0: "\033[40m - \033[0m", # black
            1: "\033[41m o \033[0m", # red
            -1: "\033[43m x \033[0m" # yellow
        }

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

if __name__ == '__main__':
    board = board()
    board.see_board()