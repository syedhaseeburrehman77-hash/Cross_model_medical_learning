from .brain_tumor_dataset import BrainTumorLoader
from .cifar10_dataset import CIFAR10Loader
from .milk10k_dataset import Milk10kLoader

DATASET_REGISTRY = {
    "brain_tumor": BrainTumorLoader,
    "cifar10": CIFAR10Loader,
    "milk10k": Milk10kLoader,
}