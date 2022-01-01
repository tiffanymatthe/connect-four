#!/usr/bin/env python
from MCTS import MCTS
from Node import Node


class NeuralNetworkTrainer():
    def __init__(self) -> None:
        # should store data from self-play games as training data
        # probably only keep those from last iteration
        # could also store it in files too avoid losing too much data
        pass

    def __play_self_play_game(MCTS_iterations):
        # how to do self-play? two MCTS trees?
        # need to somehow save (s, pi, z) at intervals
        # how to get z from MCTS? is it just the result of the actual game? so need to populate at end?
        # TODO: implement
        tree = MCTS()
        board = Node()

        while (not board.is_terminal()):
            for _ in range(MCTS_iterations):
                tree.do_rollout(board)
            board = tree.choose(board)
