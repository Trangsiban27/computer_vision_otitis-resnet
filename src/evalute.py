import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from model import build_model
from train import BEST_MODEL_PATH, get_device
from dataloader import test_loader
from label_mapping import CLASS_NAMES, BINARY_CLASS_NAME, map_5class_to_binary

def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel('Nhãn Thực Tế (True Label)', fontsize=12)
    plt.xlabel('Nhãn Dự Đoán (Predicted Label)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # plt.show()

def evaluate_model(checkpoint_path=BEST_MODEL_PATH):
    device = get_device()

    model = build_model(num_classes=5, resnet_version="resnet50", freeze_backbone=True, unfreeze_last_layers=1, se_reduction=16)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds_5class = []
    all_labels_5class = []

    with torch.no_grad():
        for images, lables in test_loader:
            images = images.to(device)
            lables = lables.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds_5class.extend(preds.cpu().numpy())
            all_labels_5class.extend(lables.cpu().numpy())

    all_preds_binary = map_5class_to_binary(all_preds_5class)
    all_labels_binary = map_5class_to_binary(all_labels_5class)

    print("\n" + "="*50)
    print("📊 BÁO CÁO PHÂN LOẠI ĐA LỚP (5 CLASSES)")
    print("="*50)
    print(classification_report(all_labels_5class, all_preds_5class, target_names=CLASS_NAMES))

    print("\n" + "="*50)
    print("🔬 BÁO CÁO PHÂN LOẠI NHỊ PHÂN (BÌNH THƯỜNG vs BỆNH LÝ)")
    print("="*50)
    print(classification_report(all_labels_binary, all_preds_binary, target_names=BINARY_CLASS_NAME))

    plot_confusion_matrix(
        y_true=all_labels_5class, 
        y_pred=all_preds_5class, 
        classes=CLASS_NAMES, 
        title="Ma trận nhầm lẫn - Phân loại 5 bệnh lý tai"
    )
    
    # Biểu đồ 2: Tổng quát Nhị phân (Gom nhóm Bệnh / Không bệnh)
    plot_confusion_matrix(
        y_true=all_labels_binary, 
        y_pred=all_preds_binary, 
        classes=BINARY_CLASS_NAME, 
        title="Ma trận nhầm lẫn - Sàng lọc Viêm tai giữa (Disease vs Normal)"
    )

    plt.show()

if __name__ == "__main__":
    evaluate_model()