#!/usr/bin/env python
import numpy as np
import random
import BoardState
from PlayerId import PlayerId

class SearchNode:
    # https://github.com/int8/monte-carlo-tree-search/blob/master/mctspy/tree/nodes.py
    # https://www.baeldung.com/java-monte-carlo-tree-search
    def __init__(self, board: BoardState, parent=None) -> None:
        self.board = board
        self.parent = parent # none for the root node
        self.children = [] # immediate possible actions from current node
        self._num_visits = 0
        self._results = {
            1: 0, # wins
            0: 0, # ties
            -1: 0 # losses
        }
        self._untried_actions = self.board.get_legal_actions()
        self.own_id = PlayerId.PLAYER_0
        self.enemy_id = PlayerId.PLAYER_1

    def q(self):
        wins = self._results[1]
        losses = self._results[-1]
        return wins - losses

    def n(self):
        return self._num_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.board.move(action)
        child_node = SearchNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.board.is_game_over()
    
    def rollout(self):
        current_rollout_state = self.board
        
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
            # current_rollout_state.see_board()
        return current_rollout_state.game_result(self.own_id.value)

    def backpropagate(self, result):
        self._num_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)
    
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 1000
        for i in range(simulation_no):
            v = self._tree_policy()
            # v.board.see_board()
            reward = v.rollout()
            v.backpropagate(reward)
        
        return self.best_child(c_param=0.)

# if __name__ == '__main__':
#     initial_state = BoardState.BoardState()
#     initial_state.current_state[5,0] = 0
#     initial_state.current_state[5,1] = 1
#     # initial_state.current_state[4,6] = 0
#     # initial_state.current_state[5,2] = 1
#     # initial_state.current_state[4,2] = 0
#     # initial_state.current_state[5,6] = 1
#     # initial_state.current_state[4,0] = 0
#     # initial_state.current_state[5,3] = 1
#     initial_state.see_board()
#     initial_state.updated_unfilled_cols()

#     root = SearchNode(initial_state)
#     selected_node = root.best_action() # for player 0
#     selected_node.board.see_board()

def AI_move(initial_state):
    root = SearchNode(initial_state)
    selected_node = root.best_action() # for player 0
    return selected_node.board

def get_int_input(message):
    user_input = input(message)
    while(not user_input.isnumeric()):
        user_input = input("Not an int, try again: ")
    return int(user_input)

def get_player_move(player_id, board):
    if (board.is_draw()):
        raise ValueError("DRAW")
    col = get_int_input("Please enter a column index: ")
    while(col < 0 or col >= board.num_col):
        col = get_int_input("Not in range, try again: ")
    success = False
    while (not success):
        try:
            board = board.move(col, player_id)
            success = True
        except ValueError:
            col = get_int_input("Col is full, try another one: ")
            board = board.move(col, player_id)

    return board

if __name__ == '__main__':
    board = BoardState.BoardState()
    manual_id = PlayerId.PLAYER_1
    print("Computer (player 0) gets first move.")
    board = AI_move(board)
    board.see_board()
    board = get_player_move(manual_id, board)
    board.see_board()
    board = AI_move(board)
    board.see_board()

    winner = board.get_winner()
    while (winner == -1):
        if (board.is_draw()):
            break
        board = get_player_move(manual_id, board)
        board.see_board()
        winner = board.get_winner()
        if (winner != -1):
            break
        board = AI_move(board)
        board.see_board()
        winner = board.get_winner()

    print("Winner is {}".format(winner))