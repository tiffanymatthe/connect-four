#!/usr/bin/env python

class C4Node():
    def __init__(self, prior_probability: float) -> None:
        self.visit_count = 0 # (N)
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