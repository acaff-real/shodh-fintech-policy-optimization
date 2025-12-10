import pandas as pd
import numpy as np
import os


INPUT_FILE = 'lending-club-data/accepted_2007_to_2018Q4.csv.gz' 
OUTPUT_FILE = 'lending_club_sampled.csv'
SAMPLE_RATE = 0.10  
RANDOM_SEED = 42

def process_data():
    print(f"Reading {INPUT_FILE}... this might take a minute.")

    sampled_chunks = []
    
    
    chunk_size = 100000
    
    
    
    
    with pd.read_csv(INPUT_FILE, chunksize=chunk_size, compression='gzip', low_memory=False) as reader:
        for i, chunk in enumerate(reader):
            
            
            
            mask = chunk['loan_status'].isin(['Fully Paid', 'Charged Off'])
            filtered_chunk = chunk[mask].copy()
            
            
            
            filtered_chunk['target'] = filtered_chunk['loan_status'].apply(
                lambda x: 1 if x == 'Charged Off' else 0
            )
            
            
            
            if not filtered_chunk.empty:
                sampled_chunk = filtered_chunk.sample(frac=SAMPLE_RATE, random_state=RANDOM_SEED)
                sampled_chunks.append(sampled_chunk)
            
            print(f"Processed chunk {i+1}...", end='\r')

    print("\nConcatenating chunks...")
    final_df = pd.concat(sampled_chunks, axis=0)
    
    
    final_df = final_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"Saving {len(final_df)} rows to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("Done! Stats:")
    print(final_df['loan_status'].value_counts(normalize=True))
    print(f"File saved: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    
    if not os.path.exists(INPUT_FILE):
        
        INPUT_FILE = 'lending-club-data/accepted_2007_to_2018Q4.csv'
        
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Could not find dataset at {INPUT_FILE}")
        print("Make sure you unzipped the folder correctly.")
    else:
        process_data()