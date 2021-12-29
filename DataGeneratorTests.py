#!/usr/bin/env python
import unittest
from DataGenerator import DataGenerator
import numpy as np

class DataGeneratorTests(unittest.TestCase):
    # -------------------- LABEL TESTS ---------------------------------------------------
    def test_validate_label(self):
        move_probabilities = np.array([0.1, 0.2, 0.9, 0.9, 0, 1, 0.14])
        win_probability = 0.5
        is_valid_label = DataGenerator.validate_label(move_probabilities, win_probability)
        self.assertTrue(is_valid_label)

    def test_validate_label_empty(self):
        move_probabilities = np.array([])
        win_probability = 0.3
        is_valid_label = DataGenerator.validate_label(move_probabilities, win_probability)
        self.assertTrue(not is_valid_label)

    def test_validate_label_too_short(self):
        move_probabilities = np.array([0.2, 0.1, 0.5])
        win_probability = 0.2
        is_valid_label = DataGenerator.validate_label(move_probabilities, win_probability)
        self.assertTrue(not is_valid_label)

    def test_validate_label_too_long(self):
        move_probabilities = np.array([0.2, 0.1, 0.5, 0.1, 0.1, 0.2, 0.4, 0.5, 0.5, 0.9])
        win_probability = 0.999
        is_valid_label = DataGenerator.validate_label(move_probabilities, win_probability)
        self.assertTrue(not is_valid_label)

    def test_validate_label_move_values(self):
        move_probabilities = np.array([-1, -0.1, 0, 1, 0.2, 0.3, 0.7])
        win_probability = 0.134
        is_valid_label = DataGenerator.validate_label(move_probabilities, win_probability)
        self.assertTrue(not is_valid_label)
    
    def test_validate_label_win_value_max(self):
        move_probabilities = np.array([1, 0.1, 0, 1, 0.2, 0.3, 0.7])
        win_probability = 1
        is_valid_label = DataGenerator.validate_label(move_probabilities, win_probability)
        self.assertTrue(is_valid_label)

    def test_validate_label_win_value_min(self):
        move_probabilities = np.array([1, 0.1, 0, 1, 0.2, 0.3, 0.7])
        win_probability = -1
        is_valid_label = DataGenerator.validate_label(move_probabilities, win_probability)
        self.assertTrue(is_valid_label)

    def test_validate_label_win_value_not_in_range(self):
        move_probabilities = np.array([1, 0.1, 0, 1, 0.2, 0.3, 0.7])
        win_probability = -1.1
        is_valid_label = DataGenerator.validate_label(move_probabilities, win_probability)
        self.assertTrue(not is_valid_label)

    def test_validate_label_win_value_zero(self):
        move_probabilities = np.array([1, 0.1, 0, 1, 0.2, 0.3, 0.7])
        win_probability = 0
        is_valid_label = DataGenerator.validate_label(move_probabilities, win_probability)
        self.assertTrue(is_valid_label)

    # -------------------- STATE TESTS ---------------------------------------------------
    def test_validate_state(self):
        state = np.array([[-1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1,  0, -1, -1, -1],
                          [-1, -1, -1,  1, -1, -1, -1],
                          [-1,  0, -1,  1, -1,  0, -1],
                          [ 0,  1, -1,  1, -1,  1, -1]])
        self.assertTrue(DataGenerator.validate_state(state))

    def test_validate_state_empty(self):
        state = np.array([])
        self.assertTrue(not DataGenerator.validate_state(state))

    def test_validate_state_square(self):
        state = -np.ones((6,6))
        self.assertTrue(not DataGenerator.validate_state(state))

    def test_validate_state_null(self):
        state = None
        self.assertTrue(not DataGenerator.validate_state(state))

    def test_validate_state_wrong_player_first(self):
        state = np.array([[-1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1,  0, -1, -1, -1],
                          [-1, -1, -1,  1, -1, -1, -1],
                          [-1,  0, -1,  1,  0,  0, -1],
                          [ 0,  1, -1,  1,  0,  1, -1]])
        self.assertTrue(not DataGenerator.validate_state(state))

    def test_validate_state_full_board(self):
        # TODO: change once winning patterns are excluded
        row1 = np.ones(7)
        row2 = np.zeros(7)
        state = np.stack((row1,row2,row2,row1,row1,row2))
        self.assertTrue(DataGenerator.validate_state(state))

    def test_validate_state_floating_token(self):
        state = np.array([[-1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1,  0,  1, -1, -1],
                          [-1, -1, -1,  1, -1, -1, -1],
                          [-1,  0, -1,  1, -1,  0, -1],
                          [ 0,  1, -1,  1,  0,  1, -1]])
        self.assertTrue(not DataGenerator.validate_state(state))

    def test_validate_state_empty_board(self):
        state = -np.ones((6,7))
        self.assertTrue(DataGenerator.validate_state(state))

    def test_validate_state_invalid_tokens(self):
        state = -np.ones((6,7))
        state[0,3] = 2
        state[1,2] = 0.2
        state[5,5] = -15
        self.assertTrue(not DataGenerator.validate_state(state))

    # -------------------- FLOATING TOKEN TESTS ------------------------------------------
    def test_floating_token(self):
        col = np.array([-1,-1,0,1,1,-1])
        self.assertTrue(DataGenerator.contains_floating_token(col))

    def test_floating_token_full_col_false(self):
        col1 = np.zeros(6)
        col2 = np.ones(6)
        self.assertTrue(not DataGenerator.contains_floating_token(col1))
        self.assertTrue(not DataGenerator.contains_floating_token(col2))

    def test_floating_token_empty_false(self):
        col = -np.ones(6)
        self.assertTrue(not DataGenerator.contains_floating_token(col))

    def test_floating_token_full_with_holes_true(self):
        col = np.array([1,0,-1,1,-1,0])
        self.assertTrue(DataGenerator.contains_floating_token(col))

    def test_floating_token_at_end_true(self):
        col = np.array([1,0,0,1,1,-1])
        self.assertTrue(DataGenerator.contains_floating_token(col))

    # -------------------- GET NN INPUT TESTS --------------------------------------------
    def test_nn_input_game_empty_board(self):
        state = -np.ones((6,7))
        current_player_colour = 1
        stack = DataGenerator.get_nn_input(state, current_player_colour)
        self.assertEqual(stack.shape, (3,6,7))

    def test_nn_input_game_invalid_state(self):
        state = -np.ones((6,7))
        state[0, 2] = 1
        state[3,3] = 0
        current_player_colour = 1
        with self.assertRaises(ValueError):
            DataGenerator.get_nn_input(state, current_player_colour)

    def test_nn_input(self):
        state = np.array([[-1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1,  0, -1, -1, -1],
                          [-1, -1, -1,  1, -1, -1, -1],
                          [-1,  0, -1,  1, -1,  0,  1],
                          [ 0,  1, -1,  1, -1,  1,  0]])
        current_player_colour = 0

        current_player_layer = np.zeros((6,7))
        current_player_layer[5,0] = 1
        current_player_layer[4,1] = 1
        current_player_layer[2,3] = 1
        current_player_layer[4,5] = 1
        current_player_layer[-1,-1] = 1

        opponent_player_layer = np.zeros((6,7))
        opponent_player_layer[5,1] = 1
        opponent_player_layer[3,3] = 1
        opponent_player_layer[4,3] = 1
        opponent_player_layer[5,3] = 1
        opponent_player_layer[5,5] = 1
        opponent_player_layer[4,-1] = 1

        colour_layer = np.zeros((6,7))

        stack = DataGenerator.get_nn_input(state, current_player_colour)

        self.assertTrue((stack[0] == current_player_layer).all())
        self.assertTrue((stack[1] == opponent_player_layer).all())
        self.assertTrue((stack[2] == colour_layer).all())
        


if __name__ == '__main__':
    unittest.main()