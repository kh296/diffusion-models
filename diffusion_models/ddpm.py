from datetime import timedelta
import importlib
import logging
import math
import os
from socket import gethostname
import subprocess
import time
import traceback
from typing import List, Union

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from tqdm import tqdm

from diffusion_models.utils import(
        get_device_type,
        get_backend,
        get_int_from_env,
        )

class DDPM:
    def __init__(self, model, optimizer, T: int, start: float, end: float,
                 device: torch.device = torch.device('cpu'),
                 ntasks_per_node: int = -1,
                 cpus_per_task: int = 1,
                 dist_url: str = "127.0.0.1",
                 dist_port: str = "55100",
                 checkpoint_in: str = "",
                 checkpoint_out: str = "",
                 checkpoint_interval: int = 1,
                 log_level: str = "info",
                 log_interval: int = 1,
                 log_ranks: List[int] = [],
                 use_tqdm: bool = True,
                 cancel_on_exception = False,
                 timeout: Union[int, float, timedelta] = timedelta(minutes=5),
                 ):
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
            checkpoint_in (str, optional): Path to file from which to load checkpoint data
            checkpoint_out (str, optional): Path to file to which to save checkpoint data
            checkpoint_interval (int, optional): Number of epochs to wait before saving checkpoint data
            log_level (str, optional): Level for message logging
            log_interval (int, optional): Number of epochs to wait before logging training status
            log_ranks (List[int], optional): Ranks for which logging is to be performed
            use_tqdm (bool, optional): Indicate whether to use tqdm to show progress during training and sampling
            cancel_on_exception (bool, optional): Indicate whether to check for Slurm job id and cancel in case of exception
            timeout (Union[int, float, datetime.timedelta], optional): Timeout for distributed communication - in seconds if int or float  
        """
        logging.basicConfig(format="[{name}_{levelname}] {message}", style="{")
        self.logger = logging.getLogger(name=type(self).__name__)
        self.logger.setLevel(log_level.upper())
        self.optimizer = optimizer
        self.cpus_per_task = cpus_per_task
        self.dist_url = dist_url
        self.dist_port = dist_port
        self.checkpoint_in = checkpoint_in
        self.checkpoint_out = checkpoint_out
        self.checkpoint_interval = checkpoint_interval
        self.losses = []
        self.t_values = []
        self.log_interval = log_interval
        self.log_ranks = log_ranks
        self.use_tqdm = use_tqdm
        self.cancel_on_exception = cancel_on_exception
        self.job_id = os.getenv("SLURM_JOB_ID")
        self.timeout = (timeout if isinstance(timeout, timedelta)
                else timedelta(seconds=timeout))

        device_type = get_device_type(device)
        try:
            device_module = importlib.import_module(f"torch.{device_type}")
        except ModuleNotFoundError:
            self.logger.warning(f"Torch module not found for device type '{device_type}'"
                  " - falling back to 'cpu'")
            device_module = None
        if device_module is not None:
            if not (getattr(device_module, "is_available", lambda: False)()):
                self.logger.warning(f"Device type '{device_type}' not available"
                      " - falling back to 'cpu'")
                device_module = None
        if device_module is None:
            device_type = "cpu"
            device_module = importlib.import_module("torch.cpu")

        if -1 == ntasks_per_node:
            ntasks_per_node = device_module.device_count()

        self.world_size = get_int_from_env(
            ["PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "WORLD_SIZE", "SLURM_NTASKS"],
            1)

        self.rank = get_int_from_env(
            ["PMI_RANK", "OMPI_COMM_WORLD_RANK", "RANK", "SLURM_PROCID"], 0)

        self.local_world_size = ntasks_per_node
        self.local_rank = (
                self.rank - ntasks_per_node * (self.rank // ntasks_per_node))
        self.device = (device_type if "cpu" == device_type
                else f"{device_type}:{self.local_rank}")
        self.device_type = device_type

        self.backend = get_backend(device_type) if self.world_size > 1 else None
        info = (f"host+device: {gethostname()}+{self.device}, "
                f"backend: {self.backend}, "
                f"world_size: {self.world_size}, "
                f"rank: {self.rank}, local_rank: {self.local_rank}")
        if self.backend:
            self.setup()
            info = f"{info} - initialised process group"
            device_module.set_device(self.device)
        if self.rank in self.log_ranks or not self.log_ranks:
            self.logger.info(info)

        model_on_device = model.to(self.device)
        if self.checkpoint_in and os.path.isfile(self.checkpoint_in):
            checkpoint = torch.load(self.checkpoint_in, map_location=self.device)
            model_on_device.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = 1 + checkpoint['epoch']
            self.losses = checkpoint['losses']
            self.t_values = checkpoint['t_values']
            if 0 == self.rank:
                self.logger.info(
                        f"Loaded epoch {self.start_epoch} checkpoint data "
                        f"from: {self.checkpoint_in}")
        else:
            self.start_epoch = 0
            
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
        # Initialize the process group.
        torch.distributed.init_process_group(
            backend=self.backend,
            init_method=f"tcp://{self.dist_url}:{self.dist_port}",
            rank=self.rank,
            world_size=self.world_size,
            device_id=self.local_rank,
            timeout=self.timeout,
            )
        return


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
        source = range(0, self.T)[::-1]
        for i in (tqdm(source) if self.use_tqdm else source):
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
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                )

        train_kwargs = {"batch_size": batch_size}
        if self.device_type in ["cuda", "xpu"]:
            train_kwargs["num_workers"] = self.cpus_per_task
            train_kwargs["pin_memory"] = True

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=train_sampler,
                                                   **train_kwargs)
        
        end_epoch = self.start_epoch + epochs
        for epoch in range(self.start_epoch, end_epoch):
            epoch_plus_one = epoch + 1
            if self.backend:
                torch.distributed.barrier()
            t0 = time.time()
            running_loss = 0
            source = (tqdm(
                train_loader, desc='Training', total=len(train_loader))
                if self.use_tqdm else train_loader)
            for images, targets, labels in source:
                images, targets, labels = images.to(self.device), targets.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                t = torch.randint(0, self.T, (images.shape[0],), device=self.device)
                t = t.float()
                
                # Get per-sample losses instead of batch loss
                loss_per_sample = self.loss_function(images, t, labels, return_per_sample=True)  # You'll need to modify loss_function
                batch_loss = loss_per_sample.mean()  # For backward pass
                
                try:
                    batch_loss.backward()
                except RuntimeError as e:
                    if self.cancel_on_exception and self.job_id:
                        self.cancel(traceback.format_exc())
                    else:
                        raise RuntimeError(traceback.format_exc())
                self.optimizer.step()

                running_loss += batch_loss.item()
                if math.isnan(running_loss):
                    msg = (f"Epoch {epoch_plus_one}, "
                            f"rank {self.rank} - running_loss is nan")
                    if self.cancel_on_exception and self.job_id:
                        self.cancel(msg)
                    else:
                        raise RuntimeError(msg)

                # Extend both lists with per-sample values
                self.losses.extend(
                        loss_per_sample.detach().cpu().numpy().tolist())
                self.t_values.extend(t.cpu().numpy().tolist())

            t1 = time.time()
            if self.backend:
                torch.distributed.barrier()
            if (epoch_plus_one % self.log_interval == 0
                    or epoch_plus_one == end_epoch):
                if self.rank in self.log_ranks or not self.log_ranks:
                    self.logger.info(f'{gethostname()}+{self.device} - '
                            f'Epoch: {epoch_plus_one}/{end_epoch}, '
                            f'Time: {t1 - t0:.3f} s, '
                            f'Images: {len(train_loader)}, '
                            f'Loss: {running_loss/len(train_loader):.4f}')

                if self.checkpoint_out:
                    if self.backend:
                        torch.distributed.barrier()
                    if 0 == self.rank:
                        checkpoint = {
                                'epoch': epoch,
                                'model_state_dict': (
                                    self.model.module.state_dict()
                                    if self.backend
                                    else self.model.state_dict()),
                                'optimizer_state_dict': (
                                    self.optimizer.state_dict()),
                                'losses': self.losses,
                                't_values': self.t_values,
                                }
                        torch.save(checkpoint, self.checkpoint_out)
                        self.logger.info(
                                f'Saved epoch {epoch_plus_one} '
                                f'checkpoint data to: {self.checkpoint_out}')
                    if self.backend:
                        torch.distributed.barrier()

        return self.losses, self.t_values


    def cancel(self, msg=""):
        if self.cancel_on_exception and self.job_id:
            self.logger.error(msg)
            self.logger.error(f"Cancelling current job - id {self.job_id}")
            subprocess.run(["scancel", self.job_id])
        else:
            raise RuntimeError(msg)
