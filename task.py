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
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
            brightness=0.2,
               contrast=0.2
            ),
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

dataset_cache = {}

def get_dataset(dataset_name: str, dataset_path: str, scenario: str = None):

    cache_key = (dataset_name, dataset_path, scenario)

    if cache_key not in dataset_cache:
        loader = create_dataset(dataset_name, dataset_path, scenario)
        dataset_cache[cache_key] = loader.load()

    return dataset_cache[cache_key]

def load_data(partition_id: int, num_partitions: int, batch_size: int, image_size: int = 224, dataset_name: str = None, dataset_path: str = None, scenario: str = None):

    dataset = get_dataset(dataset_name=dataset_name, dataset_path=dataset_path, scenario=scenario)
    # partitioner = get_partitioner(num_partitions)
    
    # partitioner.dataset = dataset["train"]
    # partition = partitioner.load_partition(partition_id)


    # Step 1: Assign modalities to clients
    modalities = sorted(list(set(dataset["train"]["modality"])), reverse=True)
    if len(modalities) != 2:
        raise ValueError(f"Expected 2 modalities in dataset, found: {modalities}")

    mod_0, mod_1 = modalities[0], modalities[1]
    if num_partitions == 2:
            modality_mapping = {
                0: mod_0,
                1: mod_1,
            }
    elif num_partitions == 4:
            modality_mapping = {
                0: mod_0,
                1: mod_0,
                2: mod_1,
                3: mod_1,
            }
    elif num_partitions == 6:
        modality_mapping = {
            0: mod_0,
            1: mod_0,
            2: mod_0,
            3: mod_1,
            4: mod_1,
            5: mod_1,
        }
    else:
        raise ValueError("Expected 2 or 4 or 6 clients")
        
    if partition_id not in modality_mapping:
        raise ValueError(f"Invalid partition ID: {partition_id}")
    
    modality = modality_mapping[partition_id]
    
     # Step 2: Filter dataset by modality (index selection avoids decoding all images)
    modality_indices = [
        i for i, m in enumerate(dataset["train"]["modality"]) if m == modality
    ]
    modality_dataset = dataset["train"].select(modality_indices)

    if num_partitions== 2:
        # One client for this modality: use full modality dataset
        partition = modality_dataset

    elif num_partitions == 4:
        partitioner = get_partitioner(2)
        partitioner.dataset = modality_dataset
        # Step 4: Map global client ID to local partition ID
        local_partition_id = partition_id % 2
        # Step 5: Load client partition
        partition = partitioner.load_partition(local_partition_id)

    else:
        # Step 3: Create partitioner with 2 partitions
        partitioner = get_partitioner(3)
        partitioner.dataset = modality_dataset
    
        # Step 4: Map global client ID to local partition ID
        local_partition_id = partition_id % 3
    
        # Step 5: Load client partition
        partition = partitioner.load_partition(local_partition_id)

    actual_modalities = Counter(partition["modality"])

    print(
        f"[Flower partition check] "
        f"client_id={partition_id}, "
        f"num_partitions={num_partitions}, "
        f"expected={modality}, "
        f"actual={actual_modalities}, "
        f"labels={Counter(partition['label'])}, "
        f"size={len(partition)}",
        flush=True,
    )

    if set(actual_modalities.keys()) != {modality}:
        raise ValueError(
            f"Client {partition_id} expected only {modality}, "
            f"but got {actual_modalities}"
        )
    
    # Step 6: Verify distribution
    print(f"\nClient ID: {partition_id}")
    print(f"Assigned modality: {modality}")
    print(f"Partition size: {len(partition)}")
    print("Modality distribution:", Counter(partition["modality"]))
    print("Label distribution:", Counter(partition["label"]))
    
    print(f"Client {partition_id}: Partition size = {len(partition)}")


    # print(Counter(partition["modality"]))
    #print(Counter(partition["label"]))
 
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

def load_centralized_dataset(image_size: int = 224, dataset_name: str = None, dataset_path: str = None, scenario: str = None,):
    """Load test set and return dataloader."""
    # Load entire test set
    dataset = get_dataset(dataset_name=dataset_name, dataset_path=dataset_path, scenario=scenario)
    test_dataset = dataset["test"]
    # dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    dataset = test_dataset.with_transform(
    lambda batch: apply_transforms(batch, train=False, image_size=image_size)
    )
    return DataLoader(dataset, batch_size=16)