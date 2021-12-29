#!/usr/bin/env python
import math
import Node
import numpy as np

class DataGenerator:
    def __init__(self):
        pass
    
    def validate_label(move_probabilities, win_probability):
        """
        Validates labels.

        move_probabilities: numpy array of length 7 with floats from 0 to 1, inclusive. Represents the probability of placing a token in the ith column of the connect-4 board.
        win_probability: probability that current player (one that needs to make a move) wins from the current game state. Value is in range [-1,1], end points inclusive. 1 if current player wins, -1 if opponent wins.

        Returns True if label is ok, False otherwise.
        """
        if type(move_probabilities) is not np.ndarray:
            return False
        if move_probabilities.size != 7:
            return False
        if win_probability < -1 or win_probability > 1:
            return False
        if np.any((move_probabilities < 0) | (move_probabilities > 1)):
            return False
        return True

    def validate_state(state):
        """
        state: 6 x 7 numpy array representing the connect-4 game state. Comes from MCTS node.

        Returns True if array only contains -1, 0, and 1, has the same number of 0 and 1 (or different by 1 count), and has no floating tokens. A floating token is a 0 or 1 that has one or multiple -1 under it in the array.
        Returns False otherwise.
        """
        if type(state) is not np.ndarray:
            return False
        if state.shape != (6, 7):
            return False
        unique_values = np.unique(state)
        if unique_values.size > 3:
            return False
        only_contains_expected = unique_values == -1 or unique_values == 0 or unique_values == 1
        if np.any(not only_contains_expected):
            return False
        if np.abs(np.count_nonzero(state == 0) - np.count_nonzero(state == 1)) > 1:
            return False
        for col in state.transpose():
            if contains_floating_token(col):
                return False

        return True

    def contains_floating_token(col):
        """
        col: numpy array of length 6 with values of -1, 0 or 1.

        Returns true if there is a floating token.
        TODO:
        - optimize code, for loops seems inefficient
        """
        empty_count = 0
        for i in range(6):
            if col[i] == -1:
                empty_count += 1
            else:
                break

        if empty_count == 6:
            return False
        
        return np.count_nonzero(col[empty_count:-1] == -1) > 0

    def get_nn_input(state, current_player_colour):
        """
        state: representation of connect-4 board, size 6 x 7. Only contains values -1, 0, and 1. -1 represents empty space, 0 and 1 represent player tokens.
        current_player_colour: 0 or 1, representing token of current player.

        Converts game state (as found in Node) to a 6 x 7 x 3 image stack with 3 binary feature planes.
        1st plane = current player's tokens.
        2nd plane = opponent player's tokens.
        3rd plane = constant plane representing colour to play.

        Returns 6 x 7 x 3 numpy array.
        """
        if not validate_state(state):
            raise ValueError("Invalid game state.")
        plane_1 = state == current_player_colour
        plane_2 = state == not current_player_colour
        plane_3 = np.ones((6,7)) * current_player_colour
        return np.stack((plane_1, plane_2, plane_3))