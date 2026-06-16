
import pandas as pd
import cv2
from torch.utils.data import Dataset

LABEL_MAP = {
    "Acute Otitis Media": 0,
    "Chronic Otitis Media": 1,
    "Cerumen Impaction": 2,
    "Myringosclerosis": 3,
    "Normal": 4,
}

class OtitisDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]

        image_path = row['path']

        label = LABEL_MAP[row['label']]

        image = cv2.imread(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        if self.transform:
            image = self.transform(image)

        return image, label
