import torch
import json
from model import (build_model, count_trainable_params)
from dataloader import (
    train_loader,
    val_loader
)
from pathlib import Path

NUM_EPOCHS = 10
CHECkPOINT_DIR = Path('checkpoints')
CHECkPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECkPOINT_DIR / 'best_model.pth'
HISTORY_PATH = CHECkPOINT_DIR / 'history.json'

#--- set devices ---
if torch.backends.mps.is_available():
        device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Model
model = build_model(num_classes=5, freeze_backbone=True, unfreeze_last_block=True)
model.to(device)

trainable, total = count_trainable_params(model)
print(f"Device: {next(model.parameters()).device}")
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# Loss
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), 'lr': 1e-5},
    {"params": model.fc.parameters(), 'lr': 1e-4}
])

#-- training history ---
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

best_val_acc = 0

for epoch in range(NUM_EPOCHS):

    # TRAIN
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss_avg = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # VALIDATION]
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
             
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
    val_loss_avg = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    #log ket qua
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"  Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss_avg:.4f} | Val   Acc: {val_acc:.4f}")
 
    history["train_loss"].append(train_loss_avg)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss_avg)
    history["val_acc"].append(val_acc)

    #Luu checkpoints
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"-> Saved new best model (val_acc={val_acc:.4f})")

#Luu lai lich su da training
with open(HISTORY_PATH, 'w') as f:
    json.dump(history, f, indent=2)

print(f"\nTraining done. Best val_acc: {best_val_acc:.4f}")
print(f"Best model saved at: {BEST_MODEL_PATH}")
print(f"History saved at: {HISTORY_PATH}")