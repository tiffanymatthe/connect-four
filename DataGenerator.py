#!/usr/bin/env python
import math
import Node
import numpy as np

class DataGenerator:
    def __init__(self):
        pass
    
    def validate_label(move_probabilities, win_probability):
        """
        move_probabilities: numpy array of length 7 with floats from 0 to 1, inclusive. Represents the probability of placing a token in the ith column of the connect-4 board.
        win_probability: probability (integer) that current player (one that needs to make a move) wins from the current game state. -1 if loss, 0 if tied, 1 if won.

        Returns True if label is ok, False otherwise.
        """
        if type(move_probabilities) is not np.ndarray:
            return False
        if move_probabilities.size != 7:
            return False
        if type(win_probability) is not int:
            return False
        if win_probability != 0 and win_probability != -1 and win_probability != 1:
            return False
        if np.any((move_probabilities < 0) | (move_probabilities > 1)):
            return False
        return True

    