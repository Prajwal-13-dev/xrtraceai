import os
import sys
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from collections import Counter
import torch.nn.functional as F

from dataset import XRTraceDataset
from model_base import XRAnomalyDetector

# 1. Absolute Path (Guarantees it goes to the right folder)
SAVE_DIR = r"C:\Users\Student3\Documents\xrtraceai\models\train_base"
os.makedirs(SAVE_DIR, exist_ok=True)

# 2. Dual Logger (Force UTF-8 and auto-flush)
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        # "a" appends text so it doesn't overwrite itself if you restart
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# This intercepts all print() statements and sends them to the file
sys.stdout = DualLogger(os.path.join(SAVE_DIR, "train.log"))

# 3. Your Hyperparameters
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4
BATCH_SIZE     = 64
EPOCHS         = 100
MAX_GRAD_NORM  = 1.0
EARLY_STOP_PAT = 15

TRAIN_DIR = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\train"
VAL_DIR   = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\val"
CLASS_NAMES = ["idle", "locomotion", "object_interaction", "anomalous"]

def compute_class_weights(train_dir, num_classes=4):
    import json
    from pathlib import Path
    
    counts = np.zeros(num_classes, dtype=np.float64)
    for mf in Path(train_dir).glob("*_meta.json"):
        with open(mf) as f:
            meta = json.load(f)
        dist = meta.get("class_distribution", {})
        counts[0] += dist.get("idle", 0)
        counts[1] += dist.get("locomotion", 0)
        counts[2] += dist.get("object_interaction", 0)
        counts[3] += dist.get("anomalous", 0)

    total = counts.sum()
    print("\nTrain class distribution:")
    for i, (name, c) in enumerate(zip(CLASS_NAMES, counts)):
        print(f"  [{i}] {name:<22}: {int(c):>10,}  ({100*c/total:.1f}%)")

    # Step 1: inverse frequency
    weights = 1.0 / (counts + 1e-8)
    
    # Step 2: square root dampening — prevents extreme ratios

    weights = np.sqrt(weights)
    
    # Step 3: normalise to sum to num_classes
    weights = weights / weights.sum() * num_classes
    weights = weights.astype(np.float32)

    print("\nSoftened class weights (sqrt dampened):")
    for i, (name, w) in enumerate(zip(CLASS_NAMES, weights)):
        print(f"  [{i}] {name:<22}: {w:.4f}")

    return torch.from_numpy(weights)


# Stratified WeightedRandomSampler


def build_weighted_sampler(dataset):
    labels = np.array(dataset.window_labels)
    class_counts = np.bincount(labels, minlength=4).astype(np.float64)
    
    # 1. Calculate how often classes naturally occur
    natural_rates = class_counts / class_counts.sum()
    
    # 2. Define the "Equal" distribution (25% for all 4 classes)
    equal_rates = np.array([0.25, 0.25, 0.25, 0.25])
    
    blended_rates = equal_rates
    blended_rates = blended_rates / blended_rates.sum()
    
    # Map the blended rates back to the individual windows
    sample_weights = blended_rates[labels]
    sample_weights = torch.from_numpy(sample_weights.astype(np.float32))
    
    print("\nEffective sampling rates (blended 80/20):")
    for i, (name, r) in enumerate(zip(CLASS_NAMES, blended_rates)):
        print(f"  [{i}] {name:<22}: {100*r:.1f}% of batches")
        
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )



# MAIN TRAINING FUNCTION


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    config_dict = {
        "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, 
        "epochs": EPOCHS, "max_grad_norm": MAX_GRAD_NORM
    }
    with open(os.path.join(SAVE_DIR, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=4)
        
    csv_path = os.path.join(SAVE_DIR, "metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'F1_Macro', 'F1_Anom', 'Loco_Prec', 'Loco_Rec', 'Loco_F1', 'ROC_AUC', 'LR'])
    # Datasets 
    print("\nLoading datasets...")
    train_dataset = XRTraceDataset(TRAIN_DIR, seq_length=60, stride=15)
    val_dataset   = XRTraceDataset(VAL_DIR,   seq_length=60, stride=60)

    # Class weights from actual distribution
    class_weights = compute_class_weights(TRAIN_DIR).to(device)

    # Stratified sampler
    sampler      = build_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,       
        drop_last=True,
        num_workers=4,          
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"\nTrain batches/epoch : {len(train_loader)}")
    print(f"Val   batches/epoch : {len(val_loader)}")

    # Model 
    model     = XRAnomalyDetector().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # LR scheduler — halve LR when val macro-F1 stops improving
    # mode='max' because we track F1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )

    # Training state 
    best_val_f1    = 0.0
    epochs_no_improve = 0

    print(f"\n{'Ep':>4} | {'TrLoss':>8} | {'VaLoss':>8} | "
          f"{'F1-macro':>8} | {'F1-anom':>8} | {'ROC-AUC':>7} | LR")
    print("-" * 72)


    for epoch in range(1, EPOCHS + 1):

        # TRAIN 
        model.train()
        train_loss_sum = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # VALIDATE
        model.eval()
        val_loss_sum = 0.0
        all_preds, all_trues, all_probs = [], [], []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                logits, _ = model(x_batch)
                val_loss_sum += criterion(logits, y_batch).item()

                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss_sum / len(val_loader)
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)

        # METRICS
# ── METRICS ──
        f1_macro = f1_score(all_trues, all_preds, average='macro', zero_division=0)
        f1_per_class = f1_score(all_trues, all_preds, average=None, zero_division=0)
        
        # Explicitly extract Precision and Recall for the diagnostic
        prec_per_class = precision_score(all_trues, all_preds, average=None, zero_division=0)
        rec_per_class = recall_score(all_trues, all_preds, average=None, zero_division=0)

        # Class 1 is Locomotion, Class 3 is Anomalous
        loco_prec = prec_per_class[1] if len(prec_per_class) > 1 else 0.0
        loco_rec  = rec_per_class[1] if len(rec_per_class) > 1 else 0.0
        loco_f1   = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
        f1_anomalous = f1_per_class[3] if len(f1_per_class) > 3 else 0.0

        try:
            roc_auc = roc_auc_score(all_trues, np.array(all_probs), multi_class='ovr', average='macro')
        except ValueError:
            roc_auc = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        
        # Updated Print Statement
        print(f"{epoch:>4} | {avg_train_loss:>8.4f} | {avg_val_loss:>8.4f} | "
              f"{f1_macro:>8.4f} | {f1_anomalous:>8.4f} | "
              f"Loco(P:{loco_prec:.2f} R:{loco_rec:.2f}) | {roc_auc:>7.4f} | {current_lr:.2e}")

        # ── 3. Append to metrics.csv (You will need to update your CSV header to match!) ──
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, avg_val_loss, f1_macro, f1_anomalous, 
                             loco_prec, loco_rec, loco_f1, roc_auc, current_lr])

        # Confusion matrix every 10 epochs 
        if epoch % 10 == 0 or epoch == 1:
            cm = confusion_matrix(all_trues, all_preds, labels=[0,1,2,3])
            print(f"\n  Confusion matrix (rows=true, cols=pred):")
            print(f"  {'':>4}  " + "  ".join(f"{n[:4]:>6}" for n in CLASS_NAMES))
            for i, row in enumerate(cm):
                print(f"  {CLASS_NAMES[i][:4]:>4}  " + 
                      "  ".join(f"{v:>6}" for v in row))
            # Prediction distribution
            pred_dist = Counter(all_preds.tolist())
            true_dist = Counter(all_trues.tolist())
            print(f"\n  Pred dist: { {CLASS_NAMES[k]: v for k,v in sorted(pred_dist.items())} }")
            print(f"  True dist: { {CLASS_NAMES[k]: v for k,v in sorted(true_dist.items())} }\n")

        # LR scheduler tracks F1
        scheduler.step(f1_macro)

        # Save best 
        if f1_macro > best_val_f1:
            best_val_f1 = f1_macro
            epochs_no_improve = 0
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'val_f1':     best_val_f1,
                'class_weights': class_weights.cpu().numpy(),
            }, os.path.join(SAVE_DIR, "best_model.pth"))
            print(f" New best saved (F1={best_val_f1:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PAT:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PAT} epochs)")
            break

    print(f"\nDone. Best val macro-F1: {best_val_f1:.4f}")
    print(f"Checkpoint: {SAVE_DIR}/best_model.pth")


if __name__ == "__main__":
    train_model()