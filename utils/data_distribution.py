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

    label_names = {
        0: "Healthy",
        1: "Tumor",
    }

    rows = []

    for label in (0, 1):
        for modality in ("CT", "MRI"):
            rows.append({
                "client_id": client_id,
                "split": split,
                "class_label": label_names[label],
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

    half = max(num_partitions // 2, 1)
    if num_partitions == 2:
        modality_mapping = {
            0: "MRI",
            1: "CT",
        }
    elif num_partitions == 4:
        modality_mapping = {
            0: "MRI",
            1: "MRI",
            2: "CT",
            3: "CT",
        }
    else:
        modality_mapping = {i: ("MRI" if i < half else "CT") for i in range(num_partitions)}

    for client_id in range(num_partitions):
        modality = modality_mapping[client_id]

        # Filter the dataset to this client's modality
        modality_dataset = dataset["train"].filter(
            lambda x: x["modality"] == modality
        )

        if client_id < half:
            modality_clients = half
            local_partition_id = client_id
        else:
            modality_clients = num_partitions - half
            local_partition_id = client_id - half

        if modality_clients <= 1:
            partition = modality_dataset
        else:
            partitioner = get_partitioner(modality_clients)
            partitioner.dataset = modality_dataset
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