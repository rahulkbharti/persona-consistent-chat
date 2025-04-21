import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# ==== Model Banana ====
def build_model():
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )


# ==== Data Loader Setup ====
def build_dataloader(rank, world_size):
    x = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    return dataloader, sampler


# ==== Training Loop ====
def train_loop(rank, model, dataloader, sampler, optimizer, criterion, epochs):
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for inputs, labels in dataloader:
            inputs = inputs.cuda(rank)
            labels = labels.cuda(rank)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Rank {rank}, Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")


# ==== Process Group Initialization ====
def setup_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


# ==== Cleanup Process Group ====
def cleanup():
    dist.destroy_process_group()


# ==== Main Training Function ====
def train(rank, world_size):
    if rank >= torch.cuda.device_count():
        print(f"Invalid rank {rank} for available GPUs.")
        return

    setup_process_group(rank, world_size)

    model = build_model().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataloader, sampler = build_dataloader(rank, world_size)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train_loop(rank, ddp_model, dataloader, sampler, optimizer, criterion, epochs=5)

    cleanup()


# ==== Multi-Process Spawn ====
def main():
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("DDP needs at least 2 GPUs.")
        return

    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
