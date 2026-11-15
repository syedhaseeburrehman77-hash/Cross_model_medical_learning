from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, Image
from .base import BaseDatasetLoader

class Milk10kLoader(BaseDatasetLoader):
 
    def __init__(self, root_dir: str, one_modality_per_lesion: bool = True, seed: int = 42):
        super().__init__(root_dir)
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

        labels = {
            "Benign": 0,
            "Malignant": 1,
        }

        valid_df = df[
            df["diagnosis_1"].isin(labels.keys()) & df["image_type"].isin(modalities.keys())
        ].copy()

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

        records = []
        for _, row in filtered_df.iterrows():
            diag = str(row["diagnosis_1"]).strip()
            img_type = str(row["image_type"]).strip()
            img_path = images_dir / f"{row['isic_id']}.jpg"

            if img_path.exists():
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