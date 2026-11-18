from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, Image
from .base import BaseDatasetLoader

class Milk10kLoader(BaseDatasetLoader):
 
    def __init__(
        self,
        root_dir: str,
        scenario: str = "binary",
        one_modality_per_lesion: bool = True,
        seed: int = 42,
    ):
        super().__init__(root_dir)
        self.scenario = scenario.lower()
        self.one_modality_per_lesion = one_modality_per_lesion
        self.seed = seed

    def load(self):
        dataset_root = Path(self.root_dir)
        csv_path = dataset_root / "metadata.csv"
        images_dir = dataset_root / "images"

        if not csv_path.exists():
            raise FileNotFoundError(f"metadata.csv not found at {csv_path}")
        if not images_dir.exists():
            raise FileNotFoundError(f"images directory not found at {images_dir}")

        df = pd.read_csv(csv_path)

        modalities = {
            "dermoscopic": "Dermoscopy",
            "clinical: close-up": "Clinical",
        }

        # 1. Define scenario target column and label mapping
        if self.scenario == "binary":
            target_col = "diagnosis_1"
            labels = {"Benign": 0, "Malignant": 1}
            valid_df = df[
                df["diagnosis_1"].isin(labels.keys()) & df["image_type"].isin(modalities.keys())
            ].copy()
        elif self.scenario in ("diagnosis_1", "diagnosis_1_multiclass"):
            target_col = "diagnosis_1"
            labels = {"Benign": 0, "Malignant": 1, "Indeterminate": 2}
            valid_df = df[
                df["diagnosis_1"].isin(labels.keys()) & df["image_type"].isin(modalities.keys())
            ].copy()
        elif self.scenario == "diagnosis_2":
            target_col = "diagnosis_2"
            valid_df = df[
                df["diagnosis_2"].notna() & df["image_type"].isin(modalities.keys())
            ].copy()
            unique_classes = sorted(valid_df["diagnosis_2"].unique())
            labels = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
        elif self.scenario == "diagnosis_3":
            target_col = "diagnosis_3"
            valid_df = df[
                df["diagnosis_3"].notna() & df["image_type"].isin(modalities.keys())
            ].copy()
            unique_classes = sorted(valid_df["diagnosis_3"].unique())
            labels = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
        elif self.scenario in ("ground_truth_diagnosis", "ground_truth", "diagnosis_gt"):
            gt_csv_path = dataset_root / "supplements" / "training_gt.csv"
            if not gt_csv_path.exists():
                candidate_paths = [
                    dataset_root / "training_gt.csv",
                    Path(r"D:\milk10k\supplements\training_gt.csv"),
                ]
                for cand in candidate_paths:
                    if cand.exists():
                        gt_csv_path = cand
                        break

            if not gt_csv_path.exists():
                raise FileNotFoundError(f"training_gt.csv not found at {gt_csv_path}")

            gt_df = pd.read_csv(gt_csv_path)

            if len(gt_df.columns) == 2 and "lesion_id" in gt_df.columns:
                target_col = [c for c in gt_df.columns if c != "lesion_id"][0]
                unique_classes = sorted(gt_df[target_col].dropna().unique())
            else:
                class_cols = [c for c in gt_df.columns if c != "lesion_id"]
                unique_classes = sorted(class_cols)
                target_col = "ground_truth_diagnosis"
                gt_df[target_col] = gt_df[class_cols].idxmax(axis=1)

            labels = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}

            merged_df = df.merge(gt_df[["lesion_id", target_col]], on="lesion_id", how="inner")
            valid_df = merged_df[
                merged_df[target_col].notna() & merged_df["image_type"].isin(modalities.keys())
            ].copy()
        else:
            raise ValueError(
                f"Unknown scenario: {self.scenario}. Choose from 'binary', 'diagnosis_1', 'diagnosis_2', 'diagnosis_3', 'ground_truth_diagnosis'."
            )

        # 2. Modality separation per lesion (stratify by diagnosis_1 to maintain balanced modality representation)
        if self.one_modality_per_lesion:
            lesions_df = valid_df[["lesion_id", "diagnosis_1"]].drop_duplicates(subset=["lesion_id"])
            lesions_dermo, lesions_clin = train_test_split(
                lesions_df,
                test_size=0.5,
                random_state=self.seed,
                stratify=lesions_df["diagnosis_1"],
            )

            dermo_lesion_set = set(lesions_dermo["lesion_id"])
            clin_lesion_set = set(lesions_clin["lesion_id"])

            is_dermo = valid_df["lesion_id"].isin(dermo_lesion_set) & (valid_df["image_type"] == "dermoscopic")
            is_clin = valid_df["lesion_id"].isin(clin_lesion_set) & (valid_df["image_type"] == "clinical: close-up")
            filtered_df = valid_df[is_dermo | is_clin]
        else:
            filtered_df = valid_df

        # 3. Build HuggingFace Dataset
        records = []
        for _, row in filtered_df.iterrows():
            diag = str(row[target_col]).strip()
            img_type = str(row["image_type"]).strip()
            img_path = images_dir / f"{row['isic_id']}.jpg"

            if img_path.exists() and diag in labels:
                records.append(
                    {
                        "img": str(img_path),
                        "label": labels[diag],
                        "modality": modalities[img_type],
                        "lesion_id": str(row["lesion_id"]).strip(),
                    }
                )

        dataset = Dataset.from_list(records)
        dataset = dataset.cast_column("img", Image())

        milk_dataset = dataset.train_test_split(
            test_size=0.15,
            seed=self.seed,
        )

        return milk_dataset