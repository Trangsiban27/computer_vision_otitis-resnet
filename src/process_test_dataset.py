import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def create_dataframe(data_dir, nums_samples=119):
    data = []

    for class_dir in Path(data_dir).iterdir():

        if class_dir.is_dir():
            all_files = list(class_dir.glob("*"))

            selected_files = random.sample(all_files, min(nums_samples, len(all_files)))

            for img_path in selected_files:
                data.append({
                    "path": str(img_path),
                    "label": class_dir.name
                })

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_dataframe("test_data/processed")
    df.to_csv("test_data/test_data.csv", index=False)
    print("Dataset CSV created!")