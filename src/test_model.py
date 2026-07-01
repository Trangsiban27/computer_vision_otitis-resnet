import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from model import build_model
from dataloader import test_loader
from label_mapping import (
    CLASS_NAMES,
    BINARY_CLASS_NAME,
    map_5class_to_binary
)
from train import BEST_MODEL_PATH

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
RESULTS_PATH = Path("checkpoints/test_results.json")

EXPECTED_ORDER = [
    "Acute Otitis Media",
    "Chronic Otitis Media",
    "Cerumen Impaction",
    "Myringosclerosis",
    "Normal",
]

assert CLASS_NAMES == EXPECTED_ORDER, (
    "CLASS_NAMES trong label_mapping.py không khớp với LABEL_MAP trong dataset.py. "
    "Kiểm tra lại thứ tự index ở cả 2 file trước khi tin kết quả bên dưới."
)

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device



def _evaluate_one_level(y_true, y_pred, class_names, level_name, verbose=True):
    """In và trả về dict các metric chính cho 1 mức đánh giá (5-class hoặc binary)."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
 
    if verbose:
        print(f"\n{'='*20} {level_name} {'='*20}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall    (macro): {recall:.4f}")
        print(f"F1        (macro): {f1:.4f}")
        print("\nConfusion matrix (rows=true, cols=pred):")
        print(class_names)
        print(cm)
        print("\nClassification report:")
        print(report)
 
    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
    }


def evaluate(checkpoint_path=None, verbose=True):

    if checkpoint_path is None:
        checkpoint_path = BEST_MODEL_PATH
    checkpoint_path = Path(checkpoint_path)
 
    device = get_device()
    if verbose:
        print(f"Device: {device}")
 
    # ---- Load model + checkpoint ----
    model = build_model(num_classes=5, resnet_version="resnet50", freeze_backbone=True, unfreeze_last_layers=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
 
    if verbose:
        print(f"Loaded checkpoint: {checkpoint_path}")
 
    # ---- Inference trên toàn bộ test set ----
    all_preds = []
    all_labels = []
 
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
 
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
 
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
 
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
 
    # ===== Đánh giá mức 5-class (gốc) =====
    results_5class = _evaluate_one_level(
        all_labels, all_preds, CLASS_NAMES, "5-CLASS EVALUATION", verbose=verbose
    )
 
    # ===== Đánh giá mức binary (map ngược về Disease/Normal) =====
    binary_labels = map_5class_to_binary(all_labels)
    binary_preds = map_5class_to_binary(all_preds)
 
    results_binary = _evaluate_one_level(
        binary_labels, binary_preds, BINARY_CLASS_NAME,
        "BINARY EVALUATION (Disease vs Normal)", verbose=verbose
    )
 
    final_results = {
        "5class": results_5class,
        "binary": results_binary,
    }
 
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
 
    if verbose:
        print(f"\nĐã lưu kết quả chi tiết tại: {RESULTS_PATH}")
 
    return final_results

if __name__ == "__main__":
    evaluate()