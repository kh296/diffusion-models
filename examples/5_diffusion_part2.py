import argparse
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import importlib
import os
import sys
import numpy as np

# Initialise environment variables that may be used at run time.
# Define network interface.
os.environ["GLOO_SOCKET_IFNAME"] = "en0"

# Add parent directory to path so we can grab the code that we need in the diffusion_models directory
sys.path.append("..")
from diffusion_models.utils import *
from diffusion_models.unet import UNetSmol
from diffusion_models.ddpm import DDPM


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--ntasks-per-node', default=-1, type=int,
                        help='number of tasks per node for distributed training'
                        '; -1 for number of tasks equal to number of GPUs')
    parser.add_argument('--dist-url', default='127.0.0.1', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-port', default='55100', type=str,
                        help='url port used to set up distributed training')
    parser.add_argument('--cpus-per-task', default=1, type=int,
                        help='number of CPUs per task')
    parser.add_argument('--time-steps', type=int, default=500,
                        help='total number of time steps (default: 500)')
    parser.add_argument('--start-variance', type=float, default=0.0001,
                        help='smallest variance (default: 0.0001)')
    parser.add_argument('--end-variance', type=float, default=0.02,
                        help='largest variance (default: 0.02)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 24)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--checkpoint-in', default='', type=str,
                        help='path from which to load checkpoint')
    parser.add_argument('--checkpoint-out', default='', type=str,
                        help='path to which to save checkpoint')
    parser.add_argument('--data-path', default='../data/', type=str,
            help='path to data')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
            help='number of epochs to wait before saving checkpoint data')
    parser.add_argument('--log-interval', type=int, default=1,
            help='number of epochs to wait before logging training status')
    parser.add_argument('--log-level', type=str, default="info",
            help='level for message logging')
    parser.add_argument('--log-ranks', nargs='*', type=int, default=[],
            help='rank(s) for which logging is to be performed')
    parser.add_argument('--use-tqdm', action='store_true',
            help='indicate whether to use tqdm to show progress')
    parser.add_argument('--cancel-on-exception', action='store_true',
            help='indicate whether to check for Slurm job id and cancel in case of exception')
    args = parser.parse_args()

    # Get the data.
    train_dataset, test_dataset = get_mnist(args.data_path)

    # Get the model and optimizer
    unet = UNetSmol(input_channels=1,
                    output_channels=1,
                    block_channels=[32,64,128])
    optimizer = optim.AdamW(unet.parameters(), lr=args.lr)

    # Define the scheduler
    scheduler = DDPM(
        model=unet,
        optimizer=optimizer,
        T=args.time_steps,
        start=args.start_variance,
        end=args.end_variance,
        device=get_device(),
        ntasks_per_node=args.ntasks_per_node,
        cpus_per_task=args.cpus_per_task,
        dist_url=args.dist_url,
        dist_port=args.dist_port,
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        log_level=args.log_level,
        log_ranks=args.log_ranks,
        use_tqdm=args.use_tqdm,
        cancel_on_exception=args.cancel_on_exception,
    )

    # Train for a few epochs

    losses, t_values = scheduler.train(train_dataset, epochs=args.epochs)

    scheduler.teardown()

if __name__ == '__main__':
    main()
