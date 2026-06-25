import matplotlib.pyplot as plt
import json
from pathlib import Path

HISTORY_PATH = 'checkpoints/history.json'

def plot_training_history(history_file):
    with open(history_file, mode='r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    #--- Bieu dien Loss ---
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', marker='s')
    plt.title('Đồ thị Độ lỗi (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    #--- Bieu dien Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', marker='o')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', marker='s')
    plt.title('Đồ thị Độ chính xác (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. Hiển thị đồ thị
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_history(HISTORY_PATH)