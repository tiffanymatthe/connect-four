#!/usr/bin/env python

class C4Config(object):

    def __init__(self, model_name=None):
        # Self-Play
        self.num_actors = 4 # 5000
        self.num_sampling_moves = 12 # 30
        self.max_moves = 42 + 1  # 512 for chess and shogi, 722 for Go.
        self.num_simulations = 200 # 800
        self.num_games = 4 # per training step

        # Root prior exploration noise.
        # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_dirichlet_alpha = 0.4 # min noise up to 1
        self.root_exploration_fraction = 0.2 # higher means more random noise

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 2.5

        # Training
        self.epochs = 2
        self.batch_size = 32
        self.sample_size = 256
        self.window_size = 16

        self.iterations = int(2e3)
        self.random_iterations = int(0) # part of iterations count
        self.checkpoint_interval = int(100) # not currently in use

        self.weight_decay = 1e-4
        self.momentum = 0.9
        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        self.learning_rate_schedule = {
            0: 2e-1,
            20: 2e-2,
            30: 2e-3,
            35: 2e-4
        }

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

        # Validation
        self.val_games = 50
        self.win_factor = 1.3

        self.model_name = model_name if model_name else 'no_name'
