#!/usr/bin/env python
import numpy as np
import random

class board:
    def __init__(self) -> None:
        self.num_col = 7
        self.col_height = 6
        # 1 is red, -1 is yellow, 0 is black
        self.current_state = -np.ones((self.col_height, self.num_col)).astype(int)
        self.p0_token = 0
        self.p1_token = 1
        self.colors = {
            -1: "\033[40m - \033[0m", # black
            self.p0_token: "\033[41m o \033[0m", # red
            self.p1_token: "\033[43m x \033[0m" # yellow
        }

        # -1 if completely filled
        self.unfilled_states = np.ones(self.num_col).astype(int) * (self.col_height - 1)

    def get_color_coded_background(self, i):
        # https://stackoverflow.com/questions/56496731/coloring-entries-in-an-matrix-2d-numpy-array/56497272
        return self.colors[i]
    
    def print_a_ndarray(self, map1, row_sep=" "):
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
        Display board. 'o' represents player 0, 'x' represents player 1, '-' represents an empty space.
        """
        display_board = self.current_state
        coloured_board = np.vectorize(self.get_color_coded_background)(display_board)
        # print(coloured_board)
        # coloured_board.append(['0','1','2','3','4','5','6'])
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

    def check_draw(self):
        for c in range(0, self.num_col):
            if not self.is_col_full(c):
                return False
        print("It's a draw.")
        return True

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

        win = self.check_win_state_in_list(arr_list, token)
        if win:
            print("Player {} won!".format(player_num))
        return win

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

    def computer_move(self) -> bool:
        """
        Computer (player_num = 0) makes a move. If not possible (draw), returns False.
        """
        col = random.randint(0, self.num_col - 1)
        success = self.move(0, col)
        while (not success):
            if self.check_draw():
                return False
            col = random.randint(0, self.num_col - 1)
            success = self.move(0, col)

        return True

    def check_for_winner(self) -> int:
        if self.check_winning_state(0):
            return 0
        if self.check_winning_state(1):
            return 1
        else:
            return -1

    def get_player_move(self, player_id):
        if (self.check_draw()):
            return False
        col = int(input("Please enter a column index: "))
        while(col < 0 or col >= self.num_col):
            col = int(input("Not in range, try again: "))
        success = board.move(player_id, col)
        while(not success):
            col = int(input("Col is full, try another one: "))
            success = board.move(player_id, col)

        return True
        
if __name__ == '__main__':
    board = board()
    player_id = 1
    first_player = random.randint(0, 1)
    if first_player == 0:
        print("Computer (player 0) gets first move.")
        board.computer_move()
        board.see_board()
        board.get_player_move(player_id)
        board.see_board()
        board.computer_move()
    else:
        print("Player 1 gets first move.")
        board.get_player_move(player_id)
        board.see_board()
        board.computer_move()
    board.see_board()

    winner = board.check_for_winner()
    while (winner == -1):
        if (board.check_draw()):
            break
        board.get_player_move(player_id)
        board.see_board()
        board.computer_move()
        board.see_board()
        winner = board.check_for_winner()