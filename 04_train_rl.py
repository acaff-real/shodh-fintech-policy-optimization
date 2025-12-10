import d3rlpy
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import os

# --- CONFIGURATION ---
INPUT_DIR = 'processed_data'
MODEL_SAVE_PATH = 'rl_agent_cql.d3'

def prepare_rl_dataset():
    print("Loading data for RL transition creation...")
    
    # 1. Load States (X) 
    X_train = np.load(f'{INPUT_DIR}/X_train.npy')
    X_test = np.load(f'{INPUT_DIR}/X_test.npy')
    observations = np.vstack([X_train, X_test])
    
    # 2. Load Raw Data for Rewards 
    df = pd.read_csv(f'{INPUT_DIR}/clean_unscaled_data.csv')
    
    # 3. CONSTRUCT REWARDS 
    loan_amts = df['loan_amnt'].values
    int_rates = df['int_rate'].values
    targets = df['target'].values # 1=Default, 0=Paid
    
    approve_rewards = np.where(
        targets == 0, 
        loan_amts * (int_rates / 100.0), # Profit
        -loan_amts                       # Loss
    )
    
    # 4. DATA AUGMENTATION 
    final_obs = []
    final_actions = []
    final_rewards = []
    final_terminals = []
    
    print("Constructing transitions (augmenting with Deny actions)...")
    
    # Set A (Approve)
    final_obs.append(observations)
    final_actions.append(np.ones(len(observations))) 
    final_rewards.append(approve_rewards)
    final_terminals.append(np.ones(len(observations))) 
    
    # Set B (Deny)
    final_obs.append(observations)
    final_actions.append(np.zeros(len(observations))) 
    final_rewards.append(np.zeros(len(observations))) 
    final_terminals.append(np.ones(len(observations)))
    
    # Stack everything
    obs_flat = np.vstack(final_obs).astype(np.float32)
    actions_flat = np.concatenate(final_actions).astype(np.int32) 
    rewards_flat = np.concatenate(final_rewards).astype(np.float32)
    terminals_flat = np.concatenate(final_terminals).astype(np.float32)
    
    print(f"RL Dataset Shape: {len(obs_flat)} transitions")
    
    # Create MDPDataset
    dataset = d3rlpy.dataset.MDPDataset(
        observations=obs_flat,
        actions=actions_flat,
        rewards=rewards_flat,
        terminals=terminals_flat,
    )
    
    return dataset

def train_agent():
    dataset = prepare_rl_dataset()
    
    # 5. DEFINE ALGORITHM
    dqn = d3rlpy.algos.DiscreteCQLConfig(
        batch_size=256,
        learning_rate=1e-4,
    ).create(device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("Initializing CQL Agent...")
    
    # 6. TRAIN
    dqn.fit(
        dataset,
        n_steps=10000, 
        n_steps_per_epoch=1000,
        save_interval=10000,
        experiment_name="lending_club_cql"
    )
    
    # 7. EXPORT POLICY
    print(f"Saving policy to {MODEL_SAVE_PATH}...")
    dqn.save_model(MODEL_SAVE_PATH)
    
    # 8. QUICK SANITY CHECK (FIXED)
    print("\nSanity Check (Predictions on first 5 borrowers):")
    # Load directly from the file we know exists to avoid attribute errors
    if os.path.exists(f'{INPUT_DIR}/X_test.npy'):
        X_sample = np.load(f'{INPUT_DIR}/X_test.npy')[:5]
        actions = dqn.predict(X_sample)
        print(f"Predicted Actions: {actions} (1=Approve, 0=Deny)")
        print("Training Complete. No errors.")
    else:
        print("Skipping sanity check (test file not found).")

if __name__ == "__main__":
    train_agent()