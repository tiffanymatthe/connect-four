#!/usr/bin/env python
import math
import Node
import numpy as np
import itertools

class DataGenerator:
    def __init__(self):
        pass
    
    @staticmethod
    def validate_label(move_probabilities, win_probability):
        """
        Validates labels.

        move_probabilities: numpy array of length 7 with floats from 0 to 1, inclusive.
        Represents the probability of placing a token in the ith column of the connect-4 board.
        win_probability: probability that current player (one that needs to make a move) wins from the current game state.
        Value is in range [-1,1], end points inclusive. 1 if current player wins, -1 if opponent wins.

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

    @staticmethod
    def validate_state(state):
        """
        state: 6 x 7 numpy array representing the connect-4 game state. Comes from MCTS node.

        Returns True if array only contains -1, 0, and 1,
        has the same number of 0 and 1 (or one more 1 value since 1 tokens start the game first), 
        and has no floating tokens.
        A floating token is a 0 or 1 that has one or multiple -1 under it in the array.
        TODO: exclude states with multiple winning patterns found

        Returns False otherwise.
        """
        if type(state) is not np.ndarray:
            return False
        if state.shape != (6, 7):
            return False
        unique_values = np.unique(state)
        if unique_values.size > 3:
            return False
        only_contains_expected = np.isin(unique_values, [-1,0,1])
        if np.any(~only_contains_expected):
            return False
        diff = np.count_nonzero(state == 1) - np.count_nonzero(state == 0)
        if diff > 1 or diff < 0:
            return False
        for col in state.transpose():
            if DataGenerator.contains_floating_token(col):
                return False

        return True

    @staticmethod
    def contains_floating_token(col):
        """
        col: numpy array of length 6 with values of -1, 0 or 1.

        Returns true if there is a floating token.
        """
        token_groups = np.array([token for token, group in itertools.groupby(col)])
        empty_count = np.count_nonzero(token_groups == -1)
        if empty_count > 0:
            if col[0] != -1:
                return True

        return empty_count > 1

    @staticmethod
    def get_nn_input(state, current_player_colour):
        """
        state: representation of connect-4 board, size 6 x 7. Only contains values -1, 0, and 1.
        -1 represents empty space, 0 and 1 represent player tokens.
        current_player_colour: 0 or 1, representing token of current player. assume correct.

        Converts game state (as found in Node) to a 3 x 6 x 7 image stack with 3 binary feature planes.
        1st plane = current player's tokens.
        2nd plane = opponent player's tokens.
        3rd plane = constant plane representing colour to play.

        Returns 3 x 6 x 7 numpy array.

        Raises ValueError if state is an invalid game state or if current_player_colour is wrong.
        TODO: figure out if current_player_colour input is needed
        """
        if not DataGenerator.validate_state(state):
            raise ValueError("Invalid game state.")
        if DataGenerator.get_current_player(state) != current_player_colour:
            raise ValueError("Invalid player colour {}".format(current_player_colour))
        plane_1 = np.isin(state, current_player_colour)
        plane_2 = np.isin(state, not current_player_colour)
        plane_3 = np.ones((6,7)) * current_player_colour
        return np.stack((plane_1, plane_2, plane_3))

    @staticmethod
    def get_current_player(state) -> int:
        """
        state: valid connect-4 game state

        Returns current player (the one who needs to make a move next). Assumes 1 moves first in game.

        Raises ValueError if state is an invalid game state.
        """
        if not DataGenerator.validate_state(state):
            raise ValueError("Invalid game state.")
        
        return int((np.count_nonzero(state == 1) - np.count_nonzero(state == 0)) != 1)