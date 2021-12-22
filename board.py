#!/usr/bin/env python
import numpy as np

class board:
    def __init__(self) -> None:
        self.num_col = 7
        self.col_height = 6
        # 1 is red, -1 is yellow, 0 is black
        self.current_state = np.zeros((self.col_height, self.num_col)).astype(int)
        self.p0_token = 1
        self.p1_token = -1
        self.colors = {
            0: "\033[40m - \033[0m", # black
            self.p0_token: "\033[41m o \033[0m", # red
            self.p1_token: "\033[43m x \033[0m" # yellow
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

    def find_longest_seq(self, arr, val):
        # https://stackoverflow.com/a/38161867
        idx_pairs = np.where(np.diff(np.hstack(([False],arr==val,[False]))))[0].reshape(-1,2)
        seq_lengths = np.diff(idx_pairs,axis=1)
        if len(seq_lengths) > 0:
            return max(np.diff(idx_pairs,axis=1))
        else:
            return 0

    def check_win_state_in_list(self, arr_list, token):
        for arr in arr_list:
            arr = np.array(arr)
            if (arr == token).sum() < 4:
                continue
            else:
                if self.find_longest_seq(arr, token) >= 4:
                    return True
        return False

    def check_winning_state(self, player_num):
        token = self.p0_token if player_num == 0 else self.p1_token
        if (self.current_state == token).sum() < 4:
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

        return self.check_win_state_in_list(arr_list, token)

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
            self.current_state[self.unfilled_states[col], col] = self.p0_token
        else:
            self.current_state[self.unfilled_states[col], col] = self.p1_token
        self.unfilled_states[col] -= 1
        return True
        
if __name__ == '__main__':
    board = board()
    for i in range(3):
        success = board.move(1, i)
    success = board.move(1, 1)
    success = board.move(0, 1)
    success = board.move(0, 2)
    success = board.move(1, 2)
    success = board.move(0, 3)
    success = board.move(0, 3)
    success = board.move(0, 3)
    success = board.move(1, 3)
    board.see_board()
    # print(board.check_winning_state(0))
    print(board.check_winning_state(1))