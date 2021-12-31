#!/usr/bin/env python

class C4Node():
    def __init__(self, prior_probability: float) -> None:
        self.visit_count = 0 # (N)
        self.to_play = -1 # what is this?
        self.prior = prior_probability
        self.value_sum = 0 # total reward (W)
        self.children = {}
    
    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count