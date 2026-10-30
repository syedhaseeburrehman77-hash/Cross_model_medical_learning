"""pytorchexample: A Flower / PyTorch app."""
from datasets_loaders import create_dataset
from torch.utils.data import DataLoader
from utils.partitioner_helper import get_partitioner
from collections import Counter
from torchvision import transforms

def get_transforms(train=True, image_size=224):
    size = (image_size, image_size)
    if train:
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(size),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            # transforms.ColorJitter(
            # brightness=0.2,
            #   contrast=0.2
            #),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def apply_transforms(batch, train=True, image_size=224):
    """Apply transforms to the partition from FederatedDataset."""
    image_transforms = get_transforms(train=train, image_size=image_size)
    batch["img"] = [image_transforms(img) for img in batch["img"]]
    return batch

dataset = None

def get_dataset():
    global dataset
    DATASET_PATH = "data/brain-tumor-multimodal-image"
    if dataset is None:
        # loader = create_dataset('cifar10', None)
        loader = create_dataset("brain_tumor", DATASET_PATH)
        dataset = loader.load()

    return dataset

def load_data(partition_id: int, num_partitions: int, batch_size: int, image_size: int = 224):

    dataset = get_dataset()
    partitioner = get_partitioner(num_partitions)
    
    partitioner.dataset = dataset["train"]
    partition = partitioner.load_partition(partition_id)

    print(f"Client {partition_id}: Partition size = {len(partition)}")

    # print(Counter(partition["modality"]))
    print(Counter(partition["label"]))
 
    partition = partition.train_test_split(test_size=0.15, seed=42,)

    print(f"Client {partition_id}: train={len(partition['train'])}, test={len(partition['test'])}")

    # partition = partition.with_transform(apply_transforms)
    partition["train"] = partition["train"].with_transform(
        lambda batch: apply_transforms(batch, train=True, image_size=image_size)
    )
    partition["test"] = partition["test"].with_transform(
        lambda batch: apply_transforms(batch, train=False, image_size=image_size)
    )
    trainloader = DataLoader( partition["train"], batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(partition["test"], batch_size=batch_size, shuffle=False,)

    return trainloader, testloader

def load_centralized_dataset(image_size: int = 224):
    """Load test set and return dataloader."""
    # Load entire test set
    dataset = get_dataset()
    test_dataset = dataset["test"]
    # dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    dataset = test_dataset.with_transform(
    lambda batch: apply_transforms(batch, train=False, image_size=image_size)
    )
    return DataLoader(dataset, batch_size=16)