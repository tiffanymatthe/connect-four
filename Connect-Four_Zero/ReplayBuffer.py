#!/usr/bin/env python
import numpy as np
from C4Config import C4Config

class ReplayBuffer(object):
    def __init__(self, config: C4Config) -> None:
        self.buffer = []
        self.buffer_size = config.sample_size
        self.window_size = config.window_size

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def is_empty(self):
        return len(self.buffer) == 0

    def get_buffer_size(self):
        return len(self.buffer)

    def sample_batch(self):
        # Sample uniformly across positions.
        move_sum = float(sum(len(g.history) for g in self.buffer))
        games = np.random.choice(
            self.buffer,
            size=min(self.sample_size, self.get_buffer_size() * 20),
            p=[len(g.history) / move_sum for g in self.buffer])
        game_pos = [(g, np.random.randint(len(g.history))) for g in games]
        return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]
