import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F


from dataset import XRTraceDataset
from model import XRAnomalyDetector

# ============================================================================
# STEP 3: CONSERVATIVE HYPERPARAMETERS
# ============================================================================
# These are deliberately safe to establish a solid baseline without exploding gradients
LEARNING_RATE = 1e-4          # Conservative LR for BiLSTMs
WEIGHT_DECAY = 1e-4           # Light L2 regularization to prevent overfitting
BATCH_SIZE = 32
EPOCHS = 50
MAX_GRAD_NORM = 1.0           # CRITICAL: Prevents LSTM gradient explosions

# Saving intervals
SAVE_DIR = "checkpoints"
SAVE_INTERVAL = 20           # Save model every N epochs, regardless of performance

# Paths
TRAIN_DIR = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\train"
VAL_DIR = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\val"

os.makedirs(SAVE_DIR, exist_ok=True)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}\n")

    # 1. Load Datasets
    print("Loading Training Data...")
    train_dataset = XRTraceDataset(TRAIN_DIR, seq_length=60, stride=15)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print("Loading Validation Data...")
    val_dataset = XRTraceDataset(VAL_DIR, seq_length=60, stride=60) # Stride=60 prevents overlap leak in val
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model, Loss, and Optimizer
    model = XRAnomalyDetector().to(device)
    
    # CrossEntropyLoss automatically applies Softmax internally, so we feed it raw logits
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_f1 = 0.0

    # Table Header
    print(f"\n{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<8} | {'Precision':<9} | {'Recall':<6} | {'F1':<6} | {'ROC-AUC':<7}")
    print("-" * 70)

    # ============================================================================
    # STEP 4: TRACKING AND METRICS LOOP
    # ============================================================================
    for epoch in range(1, EPOCHS + 1):
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss_accum = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (unpack the tuple because of the attention mechanism)
            logits, _ = model(x_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Conservative Fix: Clip gradients before the optimizer steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            optimizer.step()
            train_loss_accum += loss.item()
            
        avg_train_loss = train_loss_accum / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss_accum = 0.0
        
        all_preds = []
        all_trues = []
        all_probs = [] # Needed for ROC-AUC
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                logits, _ = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss_accum += loss.item()
                
                # Get probabilities for ROC-AUC (Softmax across classes)
                probs = F.softmax(logits, dim=1)
                
                # Get hard predictions for Precision/Recall/F1 (Argmax)
                preds = torch.argmax(logits, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss_accum / len(val_loader)
        
        # --- CALCULATE METRICS ---
        # Using 'macro' to treat the anomaly class equally, even if it's a minority
        val_precision = precision_score(all_trues, all_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_trues, all_preds, average='macro', zero_division=0)
        val_f1 = f1_score(all_trues, all_preds, average='macro', zero_division=0)
        
        # ROC-AUC requires One-vs-Rest configuration for multi-class
        try:
            val_roc_auc = roc_auc_score(all_trues, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            # Handles edge case if val set doesn't contain all classes in an early epoch
            val_roc_auc = 0.0

        # Print formatted row
        print(f"{epoch:<6} | {avg_train_loss:<10.4f} | {avg_val_loss:<8.4f} | {val_precision:<9.4f} | {val_recall:<6.4f} | {val_f1:<6.4f} | {val_roc_auc:<7.4f}")

        # --- SAVING INTERVALS ---
        
        # 1. Save the absolute best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            
        # 2. Save at the specified interval (e.g., every 10 epochs)
        if epoch % SAVE_INTERVAL == 0:
            interval_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), interval_path)

    print(f"\nTraining Complete. Best Validation F1: {best_val_f1:.4f}")
    print(f"Weights saved to {SAVE_DIR}/")

if __name__ == "__main__":
    train_model()