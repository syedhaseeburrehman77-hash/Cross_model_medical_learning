import csv
from pathlib import Path
from collections import Counter

from utils.partitioner_helper import get_partitioner


OUTPUT_DIR = Path("results")

FIELDNAMES = [
    "client_id",
    "split",
    "class_label",
    "modality",
    "num_samples",
]


def count_distribution(data, client_id, split):
    """Count samples by class and modality."""
    counts = Counter(zip(data["label"], data["modality"]))

    modalities = sorted(list(set(data["modality"])), reverse=True)
    labels = sorted(list(set(data["label"])))

    # Determine class label names dynamically based on modality
    if "MRI" in modalities or "CT" in modalities:
        label_names = {0: "Healthy", 1: "Tumor"}
    elif "Dermoscopy" in modalities or "Clinical" in modalities:
        label_names = {0: "Benign", 1: "Malignant"}
    else:
        label_names = {0: "Class 0", 1: "Class 1"}

    rows = []

    for label in labels:
        for modality in modalities:
            rows.append({
                "client_id": client_id,
                "split": split,
                "class_label": label_names.get(label, f"Class {label}"),
                "modality": modality,
                "num_samples": counts.get((label, modality), 0),
            })

    return rows


def write_csv(rows, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def export_data_distribution(dataset, num_partitions):
    """Export client train/validation and global test counts."""
    client_rows = []

    modalities = sorted(list(set(dataset["train"]["modality"])), reverse=True)
    if len(modalities) != 2:
        raise ValueError(f"Expected 2 modalities in dataset, found: {modalities}")

    mod_0, mod_1 = modalities[0], modalities[1]


    if num_partitions == 2:
        modality_mapping = {
            0: "mod_0",
            1: "mod_1",
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

    for client_id in range(num_partitions):
        modality = modality_mapping[client_id]

        # Filter the dataset to this client's modality (index selection avoids decoding all images)
        modality_indices = [
            i for i, m in enumerate(dataset["train"]["modality"]) if m == modality
        ]
        modality_dataset = dataset["train"].select(modality_indices)

        if num_partitions == 2:
            # Each client receives its full modality
            partition = modality_dataset
        elif num_partitions == 4:
            # Split each modality between its two clients
            partitioner = get_partitioner(2)
            partitioner.dataset = modality_dataset

            local_partition_id = client_id % 2
            partition = partitioner.load_partition(local_partition_id)
        
        else:
            # Split each modality between its three clients
            partitioner = get_partitioner(3)
            partitioner.dataset = modality_dataset

            local_partition_id = client_id % 3
            partition = partitioner.load_partition(local_partition_id)

        local_split = partition.train_test_split(
            test_size=0.15,
            seed=42,
        )

        client_rows.extend(
            count_distribution(
                local_split["train"], client_id, "train"
            )
        )

        client_rows.extend(
            count_distribution(
                local_split["test"], client_id, "validation"
            )
        )

    write_csv(
        client_rows,
        OUTPUT_DIR / "client_data_distribution.csv",
    )

    global_rows = count_distribution(
        dataset["test"], "global", "test"
    )

    write_csv(
        global_rows,
        OUTPUT_DIR / "global_test_distribution.csv",
    )

    # Print totals for verification.
    print("\nClient data distribution:")
    for client_id in range(num_partitions):
        for split_name in ("train", "validation"):
            total = sum(
                row["num_samples"]
                for row in client_rows
                if row["client_id"] == client_id
                and row["split"] == split_name
            )
            print(
                f"Client {client_id} | {split_name}: {total} samples"
            )

    global_total = sum(row["num_samples"] for row in global_rows)
    print(f"Global test: {global_total} samples")

    print(f"\nSaved: {OUTPUT_DIR / 'client_data_distribution.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'global_test_distribution.csv'}")