import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report

# --- CONFIGURATION ---
INPUT_DIR = 'processed_data'
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20

# 1. DEFINE THE DATASET
class LoanDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. DEFINE THE MODEL (MLP)
class LoanClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LoanClassifier, self).__init__()
        # Layer 1: Input -> 64 neurons
        self.layer1 = nn.Linear(input_dim, 64)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3) # Prevent overfitting
        
        # Layer 2: 64 -> 32 neurons
        self.layer2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        
        # Output Layer: 32 -> 1 neuron (Probability)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.drop1(x)
        x = self.act2(self.layer2(x))
        x = self.drop2(x)
        x = self.sigmoid(self.output(x))
        return x

def train():
    # Load Data
    print("Loading tensors...")
    train_dataset = LoanDataset(f'{INPUT_DIR}/X_train.npy', f'{INPUT_DIR}/y_train.npy')
    test_dataset = LoanDataset(f'{INPUT_DIR}/X_test.npy', f'{INPUT_DIR}/y_test.npy')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # We load test data in one big batch for evaluation ease
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    # Initialize Model
    input_dim = train_dataset.X.shape[1]
    print(f"Detected Input Features: {input_dim}")
    
    model = LoanClassifier(input_dim)
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, labels in train_loader:
            labels = labels.unsqueeze(1) # Reshape to [batch, 1]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation
    print("\nEvaluating on Test Set...")
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            # Convert probabilities to binary predictions (Threshold = 0.5)
            preds = (outputs > 0.5).float()
            
            # Metrics
            y_true = labels.numpy()
            y_probs = outputs.numpy()
            y_pred = preds.numpy()
            
            auc = roc_auc_score(y_true, y_probs)
            f1 = f1_score(y_true, y_pred)
            
            print(f"Test AUC Score: {auc:.4f}")
            print(f"Test F1 Score:  {f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=['Fully Paid', 'Default']))
    
    # Save Model
    torch.save(model.state_dict(), 'loan_model.pth')
    print("Model saved to loan_model.pth")

if __name__ == "__main__":
    train()