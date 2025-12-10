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
    
    # Iterate through the file in chunks of 100k rows to avoid OOM (Out of Memory) errors
    chunk_size = 100000
    
    # Columns we definitely need for filtering
    # We read all columns, but you can optimize by specifying usecols if needed later
    
    with pd.read_csv(INPUT_FILE, chunksize=chunk_size, compression='gzip', low_memory=False) as reader:
        for i, chunk in enumerate(reader):
            
            # 1. FILTER: Keep only 'Fully Paid' or 'Charged Off'
            # The prompt asks to treat "Charged Off" as Default (1) and "Fully Paid" as (0)
            mask = chunk['loan_status'].isin(['Fully Paid', 'Charged Off'])
            filtered_chunk = chunk[mask].copy()
            
            # 2. ENCODE TARGET: Create the binary target column immediately
            # 1 = Default (Charged Off), 0 = Fully Paid
            filtered_chunk['target'] = filtered_chunk['loan_status'].apply(
                lambda x: 1 if x == 'Charged Off' else 0
            )
            
            # 3. SAMPLE: Randomly select rows from this chunk
            # We use distinct seeds per chunk to ensure randomness but reproducibility
            if not filtered_chunk.empty:
                sampled_chunk = filtered_chunk.sample(frac=SAMPLE_RATE, random_state=RANDOM_SEED)
                sampled_chunks.append(sampled_chunk)
            
            print(f"Processed chunk {i+1}...", end='\r')

    print("\nConcatenating chunks...")
    final_df = pd.concat(sampled_chunks, axis=0)
    
    # Shuffle the final dataset just to be safe
    final_df = final_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"Saving {len(final_df)} rows to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("Done! Stats:")
    print(final_df['loan_status'].value_counts(normalize=True))
    print(f"File saved: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    # Check if input exists
    if not os.path.exists(INPUT_FILE):
        # Fallback: maybe the user extracted the inner CSV?
        INPUT_FILE = 'lending-club-data/accepted_2007_to_2018Q4.csv'
        
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Could not find dataset at {INPUT_FILE}")
        print("Make sure you unzipped the folder correctly.")
    else:
        process_data()