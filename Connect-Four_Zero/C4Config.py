#!/usr/bin/env python

class C4Config(object):

    def __init__(self, model_name=None):
        # Self-Play
        self.num_actors = 3 # 5000
        self.num_sampling_moves = 10 # 30
        self.max_moves = 42 + 1  # 512 for chess and shogi, 722 for Go.
        self.num_simulations = 14 # 800

        # Root prior exploration noise.
        # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25 # epsilon?

        # UCB formula
        self.pb_c_base = 19652 # puct?
        self.pb_c_init = 1.25

        # Training
        self.training_steps = int(50)
        self.checkpoint_interval = int(10)
        self.window_size = int(100) # int(1e6)
        self.batch_size = 256 # 4096

        self.weight_decay = 1e-4
        self.momentum = 0.9
        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        self.learning_rate_schedule = {
            0: 2e-1,
            20: 2e-2,
            40: 2e-3,
            45: 2e-4
        }

        self.min_initial_window = 2
        self.min_new_window = 1

        self.input_shape = (6,7,2)
        self.output_policy_shape = 7

        self.hidden_layers = [
            {'filters':75, 'kernel_size': (4,4)}
            , {'filters':75, 'kernel_size': (4,4)}
            , {'filters':75, 'kernel_size': (4,4)}
            , {'filters':75, 'kernel_size': (4,4)}
            , {'filters':75, 'kernel_size': (4,4)}
            , {'filters':75, 'kernel_size': (4,4)}
        ]

        self.model_name = model_name if model_name else 'no_name'
