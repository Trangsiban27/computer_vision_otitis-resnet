from pathlib import Path
import torch
import cv2
from train import BEST_MODEL_PATH, get_device, build_model
from transform import val_transform
from label_mapping import CLASS_NAMES, map_5class_to_binary, BINARY_CLASS_NAME

#cache model
_model_cache = {"model": None, "device": None}

def load_model(checkpoint_path=None):
    
    if checkpoint_path is None:
        checkpoint_path = BEST_MODEL_PATH
    checkpoint_path = Path(checkpoint_path)

    if _model_cache["model"] is not None:
        return _model_cache['model'], _model_cache['device']
    
    device = get_device()
    model = build_model(num_classes=5, freeze_backbone=True, unfreeze_last_block=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    _model_cache['model'] = model
    _model_cache['device'] = device
    
    return model, device

def predict(image_path, checkpoint_path=None):

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError('Image not found!')
    
    model, device = load_model(checkpoint_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError('Cannot read image!')
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = val_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    predicted_idx = int(probabilities.argmax())
    predicted_class_5 = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    binary_idx = int(map_5class_to_binary([predicted_idx])[0])
    predicted_binary = BINARY_CLASS_NAME[binary_idx]

    all_probs = {
        CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class_5": predicted_class_5,
        "predicted_class_5_confidence": confidence,
        "predicted_binary": predicted_binary,
        "all_class_probabilities": all_probs,
    }

if __name__ == "__main__":
    import sys
 
    # if len(sys.argv) < 2:
    #     print("Cách dùng: python predict.py <đường_dẫn_ảnh>")
    #     sys.exit(1)
 
    result = predict("normal5.jpg")
 
    print(f"\n--- Kết quả dự đoán ---")
    print(f"Lớp dự đoán (5-class): {result['predicted_class_5']} "
          f"(confidence: {result['predicted_class_5_confidence']:.4f})")
    print(f"Kết luận (binary):     {result['predicted_binary']}")
    print("\nXác suất từng lớp:")
    for class_name, prob in sorted(
        result["all_class_probabilities"].items(), key=lambda x: -x[1]
    ):
        print(f"  {class_name:25s}: {prob:.4f}")