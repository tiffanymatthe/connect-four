#!/usr/bin/env python

class C4Config(object):

    def __init__(self):
        # Self-Play
        self.num_actors =  10 # 5000 #100
        self.num_sampling_moves = 30
        self.max_moves = 42 + 1  # 512 for chess and shogi, 722 for Go.
        self.num_simulations = 600 # 800

        # Root prior exploration noise.
        # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Training
        self.training_steps = int(500) # int(700e3) #int(500)
        self.checkpoint_interval = int(10) # int(1e3)
        self.window_size = int(100) # int(1e6)
        self.batch_size = 70 # 4096
        self.list_of_losses = [] #list of tensors containing the loss values
        self.final_loss_list = [] #list of float objects representing the loss values resulting from the training

        self.weight_decay = 1e-4
        self.momentum = 0.9
        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        self.learning_rate_schedule = {
            0: 2e-1,
            100e3: 2e-2,
            300e3: 2e-3,
            500e3: 2e-4
        }
