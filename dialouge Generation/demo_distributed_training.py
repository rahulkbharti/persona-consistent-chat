# ddp_kaggle_basic.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import torch.multiprocessing as mp


# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# Training function per process
def train(rank, world_size):
    # Initialize process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    # Data
    x = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Model
    model = SimpleModel().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):
        sampler.set_epoch(epoch)
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.cuda(rank), batch_y.cuda(rank)
            outputs = ddp_model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Rank {rank}, Epoch [{epoch+1}/5], Loss: {loss.item()}")

    dist.destroy_process_group()


def main():
    world_size = 1  # for Kaggle (2 GPUs)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
