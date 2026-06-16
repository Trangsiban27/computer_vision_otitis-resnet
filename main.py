from pathlib import Path

DATA_DIR = Path('data/raw')

for folder in DATA_DIR.iterdir():
    if folder.is_dir():
        print(folder.name)