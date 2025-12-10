import torch
import d3rlpy
import numpy as np
import pandas as pd
import torch.nn as nn
import os
import joblib

# --- CONFIGURATION ---
INPUT_DIR = 'processed_data'
DL_MODEL_PATH = 'loan_model.pth'
RL_MODEL_PATH = 'rl_agent_cql.d3' 

# 1. DEFINE DL MODEL
class LoanClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LoanClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.drop1(x)
        x = self.act2(self.layer2(x))
        x = self.drop2(x)
        x = self.sigmoid(self.output(x))
        return x

def analyze_results():
    print("Loading Data...")
    X_test = np.load(f'{INPUT_DIR}/X_test.npy')
    y_test = np.load(f'{INPUT_DIR}/y_test.npy')
    
    # Load Feature Names
    try:
        feature_names = joblib.load(f'{INPUT_DIR}/feature_names.pkl')
    except:
        feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
    
    # --- LOAD DL MODEL ---
    print("Loading Deep Learning Model...")
    dl_model = LoanClassifier(X_test.shape[1])
    dl_model.load_state_dict(torch.load(DL_MODEL_PATH))
    dl_model.eval()
    
    # Get DL Predictions
    with torch.no_grad():
        dl_probs = dl_model(torch.FloatTensor(X_test)).numpy().flatten()
        # DL Decision: If Prob(Default) > 0.5, we DENY (0). Otherwise APPROVE (1).
        dl_decisions = np.where(dl_probs > 0.5, 0, 1)

    # --- LOAD RL AGENT ---
    print(f"Loading RL Agent from {RL_MODEL_PATH}...")
    
    if not os.path.exists(RL_MODEL_PATH):
        print(f"ERROR: Could not find {RL_MODEL_PATH} in your folder.")
        print("Please run 04_train_rl.py first.")
        return

    # Load CQL Agent
    print(f"Initializing Network Architecture...")
    cql = d3rlpy.algos.DiscreteCQLConfig().create(device="cpu")
    
    # --- NETWORK INITIALIZATION FIX ---
    dummy_actions = np.array([0, 1] + [0]*8, dtype=np.int32) 
    dummy_terminals = np.zeros(10, dtype=np.float32)
    dummy_terminals[-1] = 1.0 # TERMINAL FLAG to satisfy validator
    
    dummy_dataset = d3rlpy.dataset.MDPDataset(
        observations=X_test[:10].astype(np.float32), 
        actions=dummy_actions,                       
        rewards=np.zeros(10, dtype=np.float32),
        terminals=dummy_terminals, 
    )
    
    cql.build_with_dataset(dummy_dataset)
    # ----------------------------------
    
    # Load weights
    cql.load_model(RL_MODEL_PATH)
    
    # Get RL Predictions
    print("Generating RL predictions...")
    rl_actions = cql.predict(X_test)
    
    # --- COMPARISON ---
    print("\n" + "="*40)
    print("FINAL COMPARISON REPORT")
    print("="*40)
    
    # 1. Confusion Matrix
    agreement = pd.crosstab(dl_decisions, rl_actions, rownames=['DL Model (0=Deny)'], colnames=['RL Agent (1=Approve)'])
    print("\nDecision Matrix:")
    print(agreement)
    
    # 2. Find Disagreements
    conflict_indices = np.where((dl_decisions == 0) & (rl_actions == 1))[0]
    
    print(f"\nFound {len(conflict_indices)} borrowers where DL says DENY but RL says APPROVE.")
    
    # 3. SHOWCASE: Find a Failure (Risk) AND a Success (Hero)
    print("\n--- CASE STUDY ANALYSIS ---")
    
    # FIND A FAILURE (RL Approved, but Defaulted)
    print("\n[Case 1: The Risk of Aggression]")
    for idx in conflict_indices:
        if y_test[idx] == 1: # Defaulted
            print(f"Borrower Index: {idx}")
            print(f"Outcome: DEFAULTED (The RL Agent lost money)")
            print(f"DL Risk Score: {dl_probs[idx]:.4f}")
            
            # Show features
            print("Key Features:")
            for f_idx, val in enumerate(X_test[idx]):
                if abs(val) > 1.5: 
                    print(f"  {feature_names[f_idx]}: {val:.2f}")
            break # Stop after finding one

    # FIND A HERO (RL Approved, and PAID OFF!)
    print("\n[Case 2: Unrealised profits, realised]")
    found_hero = False
    for idx in conflict_indices:
        if y_test[idx] == 0: # Fully Paid
            print(f"Borrower Index: {idx}")
            print(f"Outcome: FULLY PAID (The RL Agent made profit!)")
            print(f"DL Risk Score: {dl_probs[idx]:.4f} (DL incorrectly rejected this)")
            
            # Show features
            print("Key Features:")
            for f_idx, val in enumerate(X_test[idx]):
                if abs(val) > 1.5: 
                    print(f"  {feature_names[f_idx]}: {val:.2f}")
            found_hero = True
            break # Stop after finding one
            
    if not found_hero:
        print("No specific 'Hidden Gem' found in this test batch, but the strategy remains valid.")

    print("\nAnalysis Complete.")

if __name__ == "__main__":
    analyze_results()