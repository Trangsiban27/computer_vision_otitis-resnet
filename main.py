import argparse
 
from train import train
from test_model import evaluate
from predict import predict

def main():
    parser = argparse.ArgumentParser(
        description="ResNet otoscopic image classification - train/test/predict"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
 
    # ---- train ----
    train_parser = subparsers.add_parser("train", help="Huấn luyện model")
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Số epoch huấn luyện (mặc định: 10)"
    )
    train_parser.add_argument(
        "--no-freeze", action="store_true",
        help="Không freeze backbone (train toàn bộ ResNet18 từ đầu)"
    )
 
    # ---- test ----
    test_parser = subparsers.add_parser("test", help="Đánh giá model trên test set")
    test_parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Đường dẫn checkpoint .pth (mặc định: checkpoints/best_model.pth)"
    )
 
    # ---- predict ----
    predict_parser = subparsers.add_parser("predict", help="Dự đoán cho 1 ảnh mới")
    predict_parser.add_argument(
        "--image", type=str, required=True, help="Đường dẫn ảnh cần dự đoán"
    )
    predict_parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Đường dẫn checkpoint .pth (mặc định: checkpoints/best_model.pth)"
    )
 
    args = parser.parse_args()
 
    if args.mode == "train":
        train(
            num_epochs=args.epochs,
            freeze_backbone=not args.no_freeze,
            unfreeze_last_block=not args.no_freeze,
        )
 
    elif args.mode == "test":
        evaluate(checkpoint_path=args.checkpoint)
 
    elif args.mode == "predict":
        result = predict(args.image, checkpoint_path=args.checkpoint)
        print(f"\n--- Kết quả dự đoán cho: {args.image} ---")
        print(f"Lớp dự đoán (5-class): {result['predicted_class_5']} "
              f"(confidence: {result['predicted_class_5_confidence']:.4f})")
        print(f"Kết luận (binary):     {result['predicted_binary']}")
        print("\nXác suất từng lớp:")
        for class_name, prob in sorted(
            result["all_class_probabilities"].items(), key=lambda x: -x[1]
        ):
            print(f"  {class_name:25s}: {prob:.4f}")
 
 
if __name__ == "__main__":
    main()