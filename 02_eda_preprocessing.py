import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib


INPUT_FILE = 'lending_club_sampled.csv'
OUTPUT_DIR = 'processed_data'
import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_eda_pipeline():
    print("Loading sampled data...")
    df = pd.read_csv(INPUT_FILE)
    
    
    
    numeric_features = [
        'loan_amnt',       
        'int_rate',        
        'installment',     
        'annual_inc',      
        'dti',             
        'fico_range_low',  
        'open_acc',        
        'pub_rec',         
        'revol_bal',       
        'revol_util',      
        'total_acc',       
        'mort_acc',        
    ]
    
    
    categorical_features = ['home_ownership', 'term'] 
    
    target = 'target' 
    
    print(f"Selecting {len(numeric_features) + len(categorical_features)} features...")
    
    
    df_clean = df[numeric_features + categorical_features + [target]].copy()
    
    
    print("Handling missing values...")
    
    for col in numeric_features:
        if df_clean[col].isna().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    
    df_clean.dropna(inplace=True)
    
    
    print("Generating EDA plots...")
    
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='target', y='int_rate', data=df_clean)
    plt.title('Interest Rate vs Loan Status (0=Paid, 1=Default)')
    plt.savefig('eda_interest_rate_risk.png')
    plt.close()
    
    
    plt.figure(figsize=(12, 10))
    
    corr = df_clean[numeric_features + [target]].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig('eda_correlation_matrix.png')
    plt.close()
    
    
    print("Encoding and Scaling...")
    
    
    df_processed = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True)
    
    
    X = df_processed.drop(columns=[target]).values
    y = df_processed[target].values
    
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    
    print("Saving processed tensors...")
    np.save(f'{OUTPUT_DIR}/X_train.npy', X_train)
    np.save(f'{OUTPUT_DIR}/X_test.npy', X_test)
    np.save(f'{OUTPUT_DIR}/y_train.npy', y_train)
    np.save(f'{OUTPUT_DIR}/y_test.npy', y_test)
    
    
    feature_names = df_processed.drop(columns=[target]).columns.tolist()
    joblib.dump(feature_names, f'{OUTPUT_DIR}/feature_names.pkl')
    
    
    
    
    df_clean.to_csv(f'{OUTPUT_DIR}/clean_unscaled_data.csv', index=False)
    
    print("Preprocessing Complete.")
    print(f"Final Input Shape: {X_train.shape}")
    print(f"Artifacts saved in /{OUTPUT_DIR}")

if __name__ == "__main__":
    run_eda_pipeline()