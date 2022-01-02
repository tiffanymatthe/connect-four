#!/usr/bin/env python

class C4Node(object):
    def __init__(self, prior_probability: float, root=False) -> None:
        # https://ai.stackexchange.com/questions/25451/how-does-alphazeros-mcts-work-when-starting-from-the-root-node
        self.visit_count = 1 if root else 0 # (N)
        self.to_play = -1 # player to make a move, 0 or 1. if -1, not yet initialized
        self.prior = prior_probability
        self.value_sum = 0 # total reward (W)
        self.children = {} # dictionary of key = action, value = child
    
    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
