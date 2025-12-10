import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# --- CONFIGURATION ---
INPUT_FILE = 'lending_club_sampled.csv'
OUTPUT_DIR = 'processed_data'
import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_eda_pipeline():
    print("Loading sampled data...")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. FEATURE SELECTION
    # We select features that define "Capacity, Capital, and Character" (The 3 C's of credit)
    numeric_features = [
        'loan_amnt',       # Size of loan
        'int_rate',        # Interest rate (proxy for risk assigned by human graders)
        'installment',     # Monthly payment size
        'annual_inc',      # Borrower income
        'dti',             # Debt-to-Income ratio (CRITICAL)
        'fico_range_low',  # FICO Score (CRITICAL)
        'open_acc',        # Number of open credit lines
        'pub_rec',         # Number of derogatory public records
        'revol_bal',       # Total credit revolving balance
        'revol_util',      # Revolving line utilization rate
        'total_acc',       # Total number of credit lines
        'mort_acc',        # Number of mortgage accounts
    ]
    
    # Categorical features need One-Hot Encoding
    categorical_features = ['home_ownership', 'term'] 
    
    target = 'target' # 0 = Fully Paid, 1 = Charged Off
    
    print(f"Selecting {len(numeric_features) + len(categorical_features)} features...")
    
    # Keep only these columns + target
    df_clean = df[numeric_features + categorical_features + [target]].copy()
    
    # 2. DATA CLEANING & IMPUTATION
    print("Handling missing values...")
    # Fill numeric NaNs with the median (robust to outliers)
    for col in numeric_features:
        if df_clean[col].isna().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    # Drop rows where categorical data might be missing (usually very few)
    df_clean.dropna(inplace=True)
    
    # 3. VISUAL EDA (Saves plots to disk)
    print("Generating EDA plots...")
    
    # Plot A: Risk Distribution by Interest Rate
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='target', y='int_rate', data=df_clean)
    plt.title('Interest Rate vs Loan Status (0=Paid, 1=Default)')
    plt.savefig('eda_interest_rate_risk.png')
    plt.close()
    
    # Plot B: Correlation Matrix
    plt.figure(figsize=(12, 10))
    # Select only numeric columns for correlation
    corr = df_clean[numeric_features + [target]].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig('eda_correlation_matrix.png')
    plt.close()
    
    # 4. PREPROCESSING FOR DEEP LEARNING
    print("Encoding and Scaling...")
    
    # One-Hot Encode Categoricals
    df_processed = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True)
    
    # Split X and y
    X = df_processed.drop(columns=[target]).values
    y = df_processed[target].values
    
    # Scale Features (Critical for Neural Networks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into Train/Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. SAVE ARTIFACTS
    print("Saving processed tensors...")
    np.save(f'{OUTPUT_DIR}/X_train.npy', X_train)
    np.save(f'{OUTPUT_DIR}/X_test.npy', X_test)
    np.save(f'{OUTPUT_DIR}/y_train.npy', y_train)
    np.save(f'{OUTPUT_DIR}/y_test.npy', y_test)
    
    # Save feature names for later analysis (important for the report!)
    feature_names = df_processed.drop(columns=[target]).columns.tolist()
    joblib.dump(feature_names, f'{OUTPUT_DIR}/feature_names.pkl')
    
    # Save the raw dataframe for the RL step (we need unscaled values for Rewards later)
    # We align the index with our X_train/X_test split for safety, but for now, 
    # we just save the clean dataframe to use in the RL step.
    df_clean.to_csv(f'{OUTPUT_DIR}/clean_unscaled_data.csv', index=False)
    
    print("Preprocessing Complete.")
    print(f"Final Input Shape: {X_train.shape}")
    print(f"Artifacts saved in /{OUTPUT_DIR}")

if __name__ == "__main__":
    run_eda_pipeline()