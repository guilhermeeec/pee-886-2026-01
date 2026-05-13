from typing import Tuple
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

def get_mnist(
    partition_id: int, num_partitions: int, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% validation
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Padroniza o nome da coluna para evitar o erro de chave no apply_transforms
    partition_train_test = partition_train_test.rename_column("image", "img")

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((16, 16), antialias=True),
        v2.Lambda(lambda x: F.normalize(torch.flatten(x), p=2, dim=0))
    ])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = torch.stack([transform(img) for img in batch["img"]])
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valloader = DataLoader(
        partition_train_test["test"], batch_size=batch_size, num_workers=0
    )  # validation split
    return trainloader, valloader