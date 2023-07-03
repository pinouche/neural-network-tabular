import argparse

from neural_network.trainers.trainer_neural_network import NNTrainer

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    trainer = NNTrainer(options)
    trainer.train()
