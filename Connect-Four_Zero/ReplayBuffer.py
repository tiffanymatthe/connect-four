#!/usr/bin/env python

class ReplayBuffer(object):
    def __init__(self) -> None:
        self.buffer = []    

    def save_game(self, game):
        self.buffer.append(game)

    def is_empty(self):
        return len(self.buffer) == 0

    def get_buffer_size(self):
        return len(self.buffer)
    
    def clear_buffer(self):
        self.buffer = []

    def get_batch(self):
        batch = []
        for game in self.buffer:
            for i in range(len(game.history)):
                if i == 0:
                    continue
                batch.append((game.make_image(i), game.make_target(i)))
        return batch
