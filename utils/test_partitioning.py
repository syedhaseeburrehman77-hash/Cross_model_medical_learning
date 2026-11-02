from datasets_loaders import create_dataset
from utils.partitioner_helper import get_partitioner
from collections import Counter

DATASET_PATH = "data/brain-tumor-multimodal-image"
num_partitions = 4  # Change to 4 for the four-client test

# 1. Load dataset
loader = create_dataset("brain_tumor", DATASET_PATH)
dataset = loader.load()
print("Original dataset size:", 9618)
print("Loaded training samples:", len(dataset["train"]))
print("Loaded test samples:", len(dataset["test"]))
print(
    "Total loaded:",
    len(dataset["train"]) + len(dataset["test"])
)

train_dataset = dataset["train"]

# Add sample IDs BEFORE partitioning
train_dataset = train_dataset.add_column(
    "sample_id",
    list(range(len(train_dataset)))
)

print("\n========== ORIGINAL DATASET ==========")
print("Total samples:", len(train_dataset))
print("Modality distribution:", Counter(train_dataset["modality"]))
print("Label distribution:", Counter(train_dataset["label"]))

# 2. Assign modalities
if num_partitions == 2:
    modality_mapping = {
        0: "MRI",
        1: "CT",
    }
else:
    modality_mapping = {
        0: "MRI",
        1: "MRI",
        2: "CT",
        3: "CT",
    }

client_partitions = {}

for modality in ["MRI", "CT"]:

    # Find clients assigned to this modality
    modality_client_ids = [
        client_id
        for client_id, assigned_modality in modality_mapping.items()
        if assigned_modality == modality
    ]

    if not modality_client_ids:
        continue

    print(f"\n========== {modality} PARTITIONING ==========")

    modality_dataset = train_dataset.filter(
        lambda x: x["modality"] == modality
    )

    print("Total modality samples:", len(modality_dataset))
    print("Labels:", Counter(modality_dataset["label"]))


    if len(modality_client_ids) == 1:
        # 2-client test: one client gets the whole modality
        global_id = modality_client_ids[0]
        client_partitions[global_id] = modality_dataset

    else:
        # 4-client test: split modality between two clients
        partitioner = get_partitioner(len(modality_client_ids))
        partitioner.dataset = modality_dataset

        for local_id, global_id in enumerate(modality_client_ids):
            partition = partitioner.load_partition(local_id)
            client_partitions[global_id] = partition

# 5. Verify modality assignments
print("\n========== VERIFICATION ==========")

for client_id, partition in client_partitions.items():

    expected_modality = modality_mapping[client_id]

    actual_modalities = set(partition["modality"])

    assert actual_modalities == {expected_modality}, (
        f"Client {client_id} received incorrect modality!"
    )

    print(
        f"Client {client_id}: "
        f"PASS - Only {expected_modality} data"
    )

# 6. Verify no samples are lost or duplicated
for modality in ["MRI", "CT"]:

    expected = train_dataset.filter(
        lambda x: x["modality"] == modality
    )

    global_ids = [
    client_id
    for client_id, assigned_modality in modality_mapping.items()
    if assigned_modality == modality
]

    expected_ids = set(expected["sample_id"])

    assigned_ids = []

    for client_id in global_ids:
        assigned_ids.extend(
            client_partitions[client_id]["sample_id"]
        )

    assert len(assigned_ids) == len(set(assigned_ids)), (
        f"{modality}: duplicate samples found!"
    )

    assert expected_ids == set(assigned_ids), (
        f"{modality}: samples lost or duplicated!"
    )

    print(f"{modality}: PASS - No samples lost or duplicated")

print("\n========== GLOBAL TEST SET ==========")

test_dataset = dataset["test"]

print("Global test samples:", len(test_dataset))
print("Test modality distribution:", Counter(test_dataset["modality"]))
print("Test label distribution:", Counter(test_dataset["label"]))

print("\n========== ALL CHECKS PASSED ==========")