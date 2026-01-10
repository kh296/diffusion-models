import importlib
import os
from socket import gethostname

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from tqdm import tqdm

from diffusion_models.utils import get_device_type, get_backend

class DDPM:
    def __init__(self, model, optimizer, T: int, start: float, end: float,
                 device: torch.device = torch.device('cpu'),
                 ntasks_per_node: int = -1,
                 cpus_per_task: int = 1,
                 dist_url: str = "127.0.0.1",
                 dist_port: str = "55100"):
        """DDPM Scheduler

        Args:
            model (UNet.UNetSmol): The U-Net model backbone
            T (int): Total number of timesteps
            start (float): Smallest variance
            end (float): Largest variance
            device (torch.device, optional): Device. Defaults to torch.device('cpu')
            ntasks_per_node (int, optional): Number of tasks per node for distributed processing; -1 for number of tasks equal to number of devices on node
            cpus_per_task (int, optional): Number of CPUs per task
            dist_url (str, optional): URL used to set up distributed training
            dist_port (str, optional): Port used to set up distributed training
        """
        self.optimizer = optimizer
        self.dist_url = dist_url
        self.dist_port = dist_port

        device_type = get_device_type(device)
        try:
            device_module = importlib.import_module(f"torch.{device_type}")
        except ModuleNotFoundError:
            print(f"Torch module not found for device type '{device_type}'"
                  " - falling back to 'cpu'")
            device_module = None
        if device_module is not None:
            if not (getattr(device_module, "is_available", lambda: False)()):
                print(f"Device type '{device_type}' not available"
                      " - falling back to 'cpu'")
                device_module = None
        if device_module is None:
            device_type = "cpu"
            device_module = importlib.import_module("torch.cpu")

        if -1 == ntasks_per_node:
            ntasks_per_node = device_module.device_count()

        self.world_size = int(os.environ.get("PMI_SIZE", 1))
        self.rank = int(os.environ.get("PMI_RANK", 0))
        local_rank = (
                self.rank - ntasks_per_node * (self.rank // ntasks_per_node))
        self.device = f"{device_type}:{local_rank}"
        self.device_type = device_type
        print(f"host+device: {gethostname()}+{self.device}, "
                f"rank: {self.rank}, local_rank: {local_rank}, ", flush=True)

        self.backend = get_backend(device_type)
        if self.backend:
            self.setup()
            device_module.set_device(self.device)

        model_on_device = model.to(self.device)
        self.model = DDP(model_on_device) if self.backend else model_on_device

        self.T = T
        self.beta = torch.linspace(start, end, T).to(device)
        alpha = 1. - self.beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        self.noise_coefficient = (1 - alpha) / self.sqrt_one_minus_alpha_bar
        self.sqrt_alpha_inv = torch.sqrt(1 / alpha)


    def setup(self):
        # initialize the process group
        torch.distributed.init_process_group(
            backend=self.backend,
            init_method=f"tcp://{self.dist_url}:{self.dist_port}",
            rank=self.rank,
            world_size=self.world_size)
        print(f"Added to process group: host: {gethostname()}, "
                f"rank: {self.rank}, world_size: {self.world_size}", flush=True)


    def teardown(self):
        if self.backend:
            torch.distributed.destroy_process_group()


    def forward(self, x_0: torch.Tensor, t: float) -> torch.Tensor | torch.Tensor:
        """The forward diffusion process

        Args:
            x_0 (torch.Tensor): Initial input
            t (float): Current timestep

        Returns:
            torch.Tensor | torch.Tensor: The output of the forward diffusion process, the noise and
                the output of the model.
        """
        t = t.int()
        noise = torch.randn_like(x_0)
        xt = self.sqrt_alpha_bar[t, None, None, None] * x_0 + self.sqrt_one_minus_alpha_bar[t, None, None, None] * noise
        return xt, noise

    
    @torch.no_grad()
    def reverse(self, x_t: torch.Tensor, t: torch.Tensor, epsilon_t: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion process

        Args:
            x_t (torch.Tensor): _description_
            t (torch.Tensor): _description_
            epsilon_t (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        t = torch.squeeze(t[0].int())
        u_t = self.sqrt_alpha_inv[t] * (x_t - self.noise_coefficient[t] * epsilon_t)

        if t == 0:
            return u_t
        else:
            noise = torch.randn_like(x_t)
            return u_t + torch.sqrt(self.beta[t-1]) * noise


    @torch.no_grad()
    def sample(self, label: int, batch_size: int) -> list[torch.Tensor]:
        """Sample some images from the model with a given label

        Args:
            label (int): The digit you want to generate
            batch_size (int): How many examples you want to generate

        Returns:
            list[torch.Tensor]: A list containing batch_size number of images for each step of the
                diffusion process.
        """
        x_t = torch.randn(batch_size, 1, 28, 28).to(self.device)
        labels = torch.zeros(batch_size, 10).to(self.device)
        labels[torch.arange(batch_size), label] = 1

        x_ts = []
        for i in tqdm(range(0, self.T)[::-1]):
            t = torch.full((batch_size,), i).to(self.device)
            # t = t.float()
            epsilon_t = self.model(x_t, t.float(), labels)
            x_t = self.reverse(x_t, t, epsilon_t)
            x_ts.append(x_t.cpu().detach().numpy())

        return x_ts
    


    def loss_function(self, x_0: torch.Tensor, t: torch.Tensor, label: torch.Tensor, return_per_sample: bool = False) -> torch.Tensor:
        """Calculates the MSE loss between the true noise and the predicted noise.

        Args:
            x_0 (torch.Tensor): _description_
            t (torch.Tensor): _description_
            label (torch.Tensor): _description_
            return_per_sample (bool): If True, returns unreduced per-sample losses

        Returns:
            torch.Tensor: Either mean loss or per-sample losses depending on return_per_sample
        """
        x_noise, noise = self.forward(x_0, t)
        noise_prediction = self.model(x_noise, t.float(), label)
        
        # Calculate MSE loss without reduction
        per_sample_loss = F.mse_loss(noise, noise_prediction, reduction='none')
        
        # Take mean over all dimensions except batch dimension
        per_sample_loss = per_sample_loss.mean(dim=tuple(range(1, per_sample_loss.dim())))
        
        if return_per_sample:
            return per_sample_loss
        return per_sample_loss.mean()


    def train(self, train_dataset, epochs: int = 10, batch_size: int = 32) -> tuple[list[float], list[float]]:
        """Train the model

        Args:
            train_dataset (DataLoader): Dataset for training.
            epochs (int, optional): Defaults to 10.
            batch_size (int, optional): Defaults to 32.

        Returns:
            tuple[list[float], list[float]]: Returns (losses, t_values)
        """
        losses = []
        t_values = []

        train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank)

        train_kwargs = {"batch_size": batch_size}
        if self.device_type in ["cuda", "xpu"]:
            train_kwargs["num_workers"] = cpus_per_task
            train_kwargs["pin_memory"] = True

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=train_sampler,
                                                   **train_kwargs)
        
        for epoch in range(epochs):
            running_loss = 0
            for images, targets, labels in tqdm(train_loader, desc='Training', total=len(train_loader)):
                images, targets, labels = images.to(self.device), targets.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                t = torch.randint(0, self.T, (images.shape[0],), device=self.device)
                t = t.float()
                
                # Get per-sample losses instead of batch loss
                loss_per_sample = self.loss_function(images, t, labels, return_per_sample=True)  # You'll need to modify loss_function
                batch_loss = loss_per_sample.mean()  # For backward pass
                
                batch_loss.backward()
                self.optimizer.step()

                running_loss += batch_loss.item()
                # Extend both lists with per-sample values
                losses.extend(loss_per_sample.detach().cpu().numpy().tolist())
                t_values.extend(t.cpu().numpy().tolist())

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

        return losses, t_values
