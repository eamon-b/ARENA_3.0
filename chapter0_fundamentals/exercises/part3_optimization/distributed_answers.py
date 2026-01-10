# %%
import importlib
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch as t
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from IPython.core.display import HTML
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor, optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part3_optimization.tests as tests
from part2_cnns.solutions import get_resnet_for_feature_extraction, Linear, ResNet34
from part3_optimization.utils import plot_fn, plot_fn_with_points
from plotly_utils import bar, imshow, line

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# %%
WORLD_SIZE = t.cuda.device_count()

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"

# %%
def send_receive(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # Send tensor to rank 1
        sending_tensor = t.zeros(1)
        print(f"{rank=}, sending {sending_tensor=}")
        dist.send(tensor=sending_tensor, dst=1)
    elif rank == 1:
        # Receive tensor from rank 0
        received_tensor = t.ones(1)
        print(f"{rank=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, received {received_tensor=}")

    dist.destroy_process_group()


if MAIN:
    world_size = 2  # simulate 2 processes
    mp.spawn(send_receive, args=(world_size,), nprocs=world_size, join=True)

# %%
assert t.cuda.is_available()
assert t.cuda.device_count() > 1, "This example requires at least 2 GPUs per machine"

# %%
def send_receive_nccl(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")

    if rank == 0:
        # Create a tensor, send it to rank 1
        sending_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, sending {sending_tensor=}")
        dist.send(sending_tensor, dst=1)  # Send tensor to CPU before sending
    elif rank == 1:
        # Receive tensor from rank 0 (it needs to be on the CPU before receiving)
        received_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, {device=}, received {received_tensor=}")

    dist.destroy_process_group()


if MAIN:
    world_size = 2  # simulate 2 processes
    mp.spawn(send_receive_nccl, args=(world_size,), nprocs=world_size, join=True)

# %%
def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank 0 to all other ranks.
    """
    if rank == src:
        # send the tensor
        for i in range(world_size):
            if i == src:
                continue
            dist.send(tensor, i)
    else:
        # recieve the tensor
        dist.recv(tensor, src=src)


if MAIN:
    tests.test_broadcast(broadcast, WORLD_SIZE)
# %%
def reduce(tensor: Tensor, rank, world_size, dst=0, op: Literal["sum", "mean"] = "sum"):
    """
    Reduces gradients to rank `dst`, so this process contains the sum or mean of all tensors across processes.
    """
    if rank == dst:
        # receive all tensor
        for i in range(world_size):
            if i == dst:
                continue
            received_tensor = torch.zeros_like(tensor)
            dist.recv(received_tensor, i)
            tensor.add_(received_tensor)
        if op == "mean":
            tensor.divide_(world_size)
    else:
        dist.send(tensor, dst)



def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    Allreduce the tensor across all ranks, using 0 as the initial gathering rank.
    """
    reduce(tensor, rank, world_size, 0, op)
    broadcast(tensor, rank, world_size, 0)


if MAIN:
    tests.test_reduce(reduce, WORLD_SIZE)
    tests.test_all_reduce(all_reduce, WORLD_SIZE)
# %%
class SimpleModel(t.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.param = t.nn.Parameter(t.tensor([2.0]))

    def forward(self, x: t.Tensor):
        return x - self.param


def run_simple_model(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")
    model = SimpleModel().to(device)  # Move the model to the device corresponding to this process
    optimizer = t.optim.SGD(model.parameters(), lr=0.1)

    input = t.tensor([rank], dtype=t.float32, device=device)
    output = model(input)
    loss = output.pow(2).sum()
    loss.backward()  # Each rank has separate gradients at this point

    print(f"Rank {rank}, before all_reduce, grads: {model.param.grad=}")
    all_reduce(model.param.grad, rank, world_size)  # Synchronize gradients
    print(f"Rank {rank}, after all_reduce, synced grads (summed over processes): {model.param.grad=}")

    optimizer.step()  # Step with the optimizer (this will update all models the same way)
    print(f"Rank {rank}, new param: {model.param.data}")

    dist.destroy_process_group()


if MAIN:
    world_size = 2
    mp.spawn(run_simple_model, args=(world_size,), nprocs=world_size, join=True)
# %%
from solutions import WandbResNetFinetuningArgs, get_cifar, AdamW
# %%
def get_untrained_resnet(n_classes: int) -> ResNet34:
    """Gets untrained resnet using code from part2_cnns.answers"""
    resnet = ResNet34()
    resnet.out_layers[-1] = Linear(resnet.out_features_per_group[-1], n_classes)
    return resnet


@dataclass
class DistResNetTrainingArgs(WandbResNetFinetuningArgs):
    world_size: int = 1
    wandb_project: str | None = "day3-resnet-dist-training"


class DistResNetTrainer:
    args: DistResNetTrainingArgs

    def __init__(self, args: DistResNetTrainingArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device(f"cuda:{rank}")


    def pre_training_setup(self):
        self.model = get_untrained_resnet(self.args.n_classes).to(self.device)
        if self.rank == 0:
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.wandb_name,
                config=self.args,
            )
            wandb.watch(self.model, log="all", log_freq=500)
        if self.args.world_size > 1:
            # broadcast this model to other nodes
            for param in self.model.parameters():
                broadcast(param.data, self.rank, self.args.world_size, src=0)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.trainset, self.testset = get_cifar()
        self.train_sampler = t.utils.data.DistributedSampler(
            self.trainset,
            num_replicas=self.args.world_size, # we'll divide each batch up into this many random sub-batches
            rank=self.rank, # this determines which sub-batch this process gets
        )
        self.train_loader = t.utils.data.DataLoader(
            self.trainset,
            self.args.batch_size, # this is the sub-batch size, i.e. the batch size that each GPU gets
            sampler=self.train_sampler, 
            num_workers=2,  # setting this low so as not to risk bottlenecking CPU resources
            pin_memory=True,  # this can improve data transfer speed between CPU and GPU
        )
        self.test_sampler = t.utils.data.DistributedSampler(
            self.testset,
            shuffle=False,
            num_replicas=self.args.world_size, # we'll divide each batch up into this many random sub-batches
            rank=self.rank, # this determines which sub-batch this process gets
        )
        self.test_loader = t.utils.data.DataLoader(
            self.testset,
            self.args.batch_size, # this is the sub-batch size, i.e. the batch size that each GPU gets
            sampler=self.test_sampler, 
            num_workers=2,  # setting this low so as not to risk bottlenecking CPU resources
            pin_memory=True,  # this can improve data transfer speed between CPU and GPU
        )
        self.examples_seen = 0

    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        t0 = time.time()

        imgs, labels = imgs.to(self.device), labels.to(self.device)
        logits = self.model(imgs)

        t1 = time.time()

        loss = F.cross_entropy(logits, labels)
        loss.backward()

        t2 = time.time()

        for param in self.model.parameters():
            all_reduce(param.grad, self.rank, self.args.world_size, "mean")
            # dist.all_reduce(param.grad, op=dist.ReduceOp.SUM); param.grad /= self.args.world_size

        t3 = time.time()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.examples_seen += len(imgs) * self.args.world_size # FIXME: need to do synchronization here - just times by world size :)
        # all_reduce(self.examples_seen, self.rank, self.args.world_size)
        if self.rank == 0:
            wandb.log(
                data={
                    "loss": loss,
                    "fwd_time": t1 - t0,
                    "bkwd_time": t2 - t1,
                    "dist_time": t3 - t2,
                },
                step=self.examples_seen)
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        pbar = tqdm(self.test_loader, desc="Evaluating", disable=self.rank != 0)
        num_correct = 0
        total = 0
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            pred = logits.argmax(dim=1)
            correct = (pred == labels).sum(dim=0).item()
            num_correct += correct
            total += labels.size(0)
        correct_tensor = torch.tensor([num_correct, total], device=self.device)
        all_reduce(correct_tensor, self.rank, self.args.world_size)
        num_correct, total = correct_tensor.tolist()
        accuracy = num_correct / total
        if self.rank == 0:
            wandb.log(data={"accuracy": accuracy}, step=self.examples_seen)
        return accuracy

    def train(self):
        self.pre_training_setup()
        
        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
            t0 = time.time()

            self.model.train()
            self.train_sampler.set_epoch(epoch)
            pbar = tqdm(self.train_loader, desc=f"Training epoch {epoch}", disable=self.rank != 0)
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(loss=f"{loss:.3f}", ex_seen=f"{self.examples_seen:06}")
            t1 = time.time()
            if self.rank == 0:
                wandb.log(data={"epoch_time": t1 - t0}, step=self.examples_seen)
            accuracy = self.evaluate()
            pbar.set_postfix(loss=f"{loss:.3f}", accuracy=f"{accuracy:.2f}", ex_seen=f"{self.examples_seen:06}")
        
        if self.rank == 0:
            wandb.finish()
            torch.save(self.model.state_dict(), f"resnet_{self.rank}.pth")


def dist_train_resnet_from_scratch(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingArgs(world_size=world_size)
    trainer = DistResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()


if MAIN:
    world_size = t.cuda.device_count()
    mp.spawn(dist_train_resnet_from_scratch, args=(world_size,), nprocs=world_size, join=True)
# %%
# the same except using DDP rather than my own broadcast and all_reduce
from torch.nn.parallel import DistributedDataParallel as DDP

class DDPResNetTrainer:
    args: DistResNetTrainingArgs

    def __init__(self, args: DistResNetTrainingArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device(f"cuda:{rank}")


    def pre_training_setup(self):
        self.model = DDP(
            get_untrained_resnet(self.args.n_classes).to(self.device),
            device_ids=[self.rank]
        )
        if self.rank == 0:
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.wandb_name,
                config=self.args,
            )
            wandb.watch(self.model, log="all", log_freq=500)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.trainset, self.testset = get_cifar()
        self.train_sampler = t.utils.data.DistributedSampler(
            self.trainset,
            num_replicas=self.args.world_size, # we'll divide each batch up into this many random sub-batches
            rank=self.rank, # this determines which sub-batch this process gets
        )
        self.train_loader = t.utils.data.DataLoader(
            self.trainset,
            self.args.batch_size, # this is the sub-batch size, i.e. the batch size that each GPU gets
            sampler=self.train_sampler, 
            num_workers=2,  # setting this low so as not to risk bottlenecking CPU resources
            pin_memory=True,  # this can improve data transfer speed between CPU and GPU
        )
        self.test_sampler = t.utils.data.DistributedSampler(
            self.testset,
            shuffle=False,
            num_replicas=self.args.world_size, # we'll divide each batch up into this many random sub-batches
            rank=self.rank, # this determines which sub-batch this process gets
        )
        self.test_loader = t.utils.data.DataLoader(
            self.testset,
            self.args.batch_size, # this is the sub-batch size, i.e. the batch size that each GPU gets
            sampler=self.test_sampler, 
            num_workers=2,  # setting this low so as not to risk bottlenecking CPU resources
            pin_memory=True,  # this can improve data transfer speed between CPU and GPU
        )
        self.examples_seen = 0

    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        t0 = time.time()

        imgs, labels = imgs.to(self.device), labels.to(self.device)
        logits = self.model(imgs)

        t1 = time.time()

        loss = F.cross_entropy(logits, labels)
        loss.backward()

        t2 = time.time()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.examples_seen += len(imgs) * self.args.world_size
        if self.rank == 0:
            wandb.log(
                data={
                    "loss": loss,
                    "fwd_time": t1 - t0,
                    "bkwd_dist_time": t2 - t1,
                },
                step=self.examples_seen)
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        pbar = tqdm(self.test_loader, desc="Evaluating", disable=self.rank != 0)
        num_correct = 0
        total = 0
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            pred = logits.argmax(dim=1)
            correct = (pred == labels).sum(dim=0).item()
            num_correct += correct
            total += labels.size(0)
        correct_tensor = torch.tensor([num_correct, total], device=self.device)
        all_reduce(correct_tensor, self.rank, self.args.world_size)
        num_correct, total = correct_tensor.tolist()
        accuracy = num_correct / total
        if self.rank == 0:
            wandb.log(data={"accuracy": accuracy}, step=self.examples_seen)
        return accuracy

    def train(self):
        self.pre_training_setup()
        
        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
            t0 = time.time()

            self.model.train()
            self.train_sampler.set_epoch(epoch)
            pbar = tqdm(self.train_loader, desc=f"Training epoch {epoch}", disable=self.rank != 0)
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(loss=f"{loss:.3f}", ex_seen=f"{self.examples_seen:06}")
            t1 = time.time()
            if self.rank == 0:
                wandb.log(data={"epoch_time": t1 - t0}, step=self.examples_seen)
            accuracy = self.evaluate()
            pbar.set_postfix(loss=f"{loss:.3f}", accuracy=f"{accuracy:.2f}", ex_seen=f"{self.examples_seen:06}")
        
        if self.rank == 0:
            wandb.finish()
            torch.save(self.model.state_dict(), f"resnet_DDP_{self.rank}.pth")


def ddp_train_resnet_from_scratch(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingArgs(world_size=world_size)
    trainer = DDPResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()


if MAIN:
    world_size = t.cuda.device_count()
    mp.spawn(ddp_train_resnet_from_scratch, args=(world_size,), nprocs=world_size, join=True)
# %%
