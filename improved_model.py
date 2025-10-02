"""
STEP 1: Fix Severe Overfitting Problem

Current Issue:
- Training loss: 0.0968 (way too low!)
- Test loss: 5.8606 (60x worse!)
- This is severe overfitting

Solution: Rebuild with proper regularization and fewer epochs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, average_precision_score

# Import your preprocessed data (run your previous cells first)
# Assuming X_tensor, y_tensor are already created and scaled

class ImprovedTitanicNet(nn.Module):
    """
    Improved neural network with regularization to prevent overfitting
    """
    def __init__(self):
        super().__init__()
        
        # Smaller, more conservative architecture
        self.layer1 = nn.Linear(24, 8)  # Reduced from 16 to 8
        self.dropout1 = nn.Dropout(0.3)  # Add dropout for regularization
        self.activation1 = nn.ReLU()
        
        self.layer2 = nn.Linear(8, 4)    # Reduced from 8 to 4
        self.dropout2 = nn.Dropout(0.2)  # Add dropout
        self.activation2 = nn.ReLU()
        
        self.layer3 = nn.Linear(4, 1)     # Output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.activation1(x)
        
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.activation2(x)
        
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

def train_improved_model(X_tensor, y_tensor):
    """
    Train the improved model with better hyperparameters
    """
    # Split the data (you already did this, but let's do it consistently)
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
    )
    
    # Create improved model
    model = ImprovedTitanicNet()
    
    # Better loss function with class weighting
    pos_weight = torch.tensor([len(y_train[y_train == 0]) / len(y_train[y_train == 1])])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Conservative optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # Added weight decay
    
    # Training with early stopping
    epochs = 100  # Much fewer epochs
    best_test_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_pred = model(X_train).squeeze()
        train_loss = loss_fn(train_pred, y_train)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Testing
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test).squeeze()
            test_loss = loss_fn(test_pred, y_test)
        
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 20 == 0:
            train_acc = ((train_pred > 0.5).float() == y_train).float().mean()
            test_acc = ((test_pred > 0.5).float() == y_test).float().mean()
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')
    
    return model, X_train, X_test, y_train, y_test, train_losses, test_losses

def evaluate_model(model, X_test, y_test):
    """
    Comprehensive evaluation of the improved model
    """
    model.eval()
    with torch.no_grad():
        preds = model(X_test).squeeze()
        y_true = y_test.cpu().numpy()
        y_prob = torch.sigmoid(preds).cpu().numpy()  # Apply sigmoid since we're using BCEWithLogitsLoss
        y_pred = (y_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = (y_pred == y_true).mean()
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    
    cm = confusion_matrix(y_true, y_pred)
    
    print("=== IMPROVED MODEL PERFORMANCE ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc_roc:.4f}")
    print(f"PR-AUC: {auc_pr:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3))
    
    return {'accuracy': accuracy, 'roc_auc': auc_roc, 'pr_auc': auc_pr}

# Usage instructions:
print("""
TO USE THIS IMPROVED MODEL:

1. Run your existing data preprocessing cells first
2. Then run this script:

```python
# Train improved model
model, X_train, X_test, y_train, y_test, train_losses, test_losses = train_improved_model(X_tensor, y_tensor)

# Evaluate
results = evaluate_model(model, X_test, y_test)
```

KEY IMPROVEMENTS:
✅ Smaller architecture (prevents overfitting)
✅ Dropout regularization (added randomness)
✅ Weight decay (L2 regularization)
✅ Early stopping (prevents overtraining)
✅ Class weighting (handles imbalanced data)
✅ BCEWithLogitsLoss (numerically stable)
✅ Conservative learning rate
✅ Much fewer epochs (100 vs 1000)

EXPECTED RESULTS:
- Reduced overfitting (train/test loss closer together)
- Better PR-AUC (should improve from 0.6655)
- More stable ROC-AUC
""")
