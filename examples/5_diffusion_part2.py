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

device = get_device()
print(f"Using device: {device}")

# Get the data.
DATA_PATH = '../data/'
train_dataset, test_dataset = get_mnist(DATA_PATH)

# Get the model and optimizer
unet = UNetSmol(input_channels=1,
                output_channels=1,
                block_channels=[32,64,128])
optimizer = optim.AdamW(unet.parameters(), lr=0.001)

# Define the scheduler
scheduler = DDPM(
    model = unet,
    optimizer = optimizer,
    T = 500,
    start = 0.0001,
    end = 0.02,
    device=device,
)

# Train for a few epochs

losses, t_values = scheduler.train(train_dataset, epochs=2)

scheduler.teardown()
