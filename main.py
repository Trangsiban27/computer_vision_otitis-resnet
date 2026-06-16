from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path('data/raw')

classes = []
sized = []

for folder in DATA_DIR.iterdir():
    if folder.is_dir():
        classes.append(folder.name)
        count = len(list(folder.glob('*')))

        print(f'{folder.name}: {count}')

for folder in DATA_DIR.iterdir():

    for img_path in folder.glob('*'):
        img = cv2.imread(str(img_path))

        if img is not None:
            h, w = img.shape[:2]

            sized.append((w, h))

df = pd.DataFrame(
    sized,
    columns=['Width', 'Height']
)

df.describe()

print(df)