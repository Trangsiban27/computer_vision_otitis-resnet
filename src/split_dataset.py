from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def create_dataframe(data_dir):
    data = []

    for class_dir in Path(data_dir).iterdir():

        if class_dir.is_dir():
            for img_path in class_dir.glob("*"):
                data.append({
                    "path": str(img_path),
                    "label": class_dir.name
                })

    return pd.DataFrame(data)

def split_dataset(df):

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df['label'],
        random_state=42,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df['label'],
        random_state=42
    )

    return train_df, val_df, test_df

if __name__ == "__main__":
    df = create_dataframe("data/raw")

    train_df, val_df, test_df = split_dataset(df)

    train_df.to_csv(
        "data/splits/train/train.csv",
        index=False
    )

    val_df.to_csv(
        "data/splits/val/val.csv",
        index=False
    )

    test_df.to_csv(
        "data/splits/test/test.csv",
        index=False
    )

    print("Splits done!")
    
    train = pd.read_csv("data/splits/train/train.csv")

    print(train_df['label'].value_counts())
    print(val_df['label'].value_counts())
    print(test_df['label'].value_counts())