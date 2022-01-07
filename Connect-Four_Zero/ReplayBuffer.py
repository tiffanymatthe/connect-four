#!/usr/bin/env python

class ReplayBuffer(object):
    def __init__(self) -> None:
        self.buffer = []
        self.iteration_size = 0 

    def save_game(self, game):
        self.buffer.append(game)
        self.iteration_size += 1

    def is_empty(self):
        return len(self.buffer) == 0

    def get_iteration_size(self):
        return self.iteration_size
    
    def reset_iteration(self):
        self.iteration_size = 0

    def get_buffer_size(self):
        return len(self.buffer)

    def get_batch(self, clear):
        batch = []
        if clear:
            self.buffer = self.buffer[-self.iteration_size:-1]
        for game in self.buffer:
            for i in range(len(game.history)):
                if i == 0:
                    continue
                batch.append((game.make_image(i), game.make_target(i)))
        return batch
