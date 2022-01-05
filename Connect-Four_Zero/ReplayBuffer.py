#!/usr/bin/env python
import C4Config
import numpy as np
import time

class ReplayBuffer(object):
    def __init__(self, config: C4Config) -> None:
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []
        self.ready = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def is_empty(self):
        return len(self.buffer) == 0

    def get_buffer_size(self):
        return len(self.buffer)

    def make_buffer_ready(self):
        self.ready = self.buffer
        self.buffer = []

    def sample_batch(self):
        # Sample uniformly across positions.
        move_sum = float(sum(len(g.history) for g in self.ready))
        games = np.random.choice(
            self.ready,
            size=min(self.batch_size, len(self.ready) * 15),
            p=[len(g.history) / move_sum for g in self.ready])
        game_pos = [(g, np.random.randint(len(g.history))) for g in games]
        return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]
