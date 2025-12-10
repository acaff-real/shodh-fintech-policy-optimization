# Policy Optimization for Financial Decision-Making

##  Overview
This project compares a **Supervised Deep Learning (DL)** model against an **Offline Reinforcement Learning (RL)** agent for loan approval decisions. Using the LendingClub dataset, we demonstrate that while the DL model minimizes risk (defaults), the RL agent (trained via Conservative Q-Learning) maximizes profit, successfully identifying high-yield borrowers that the traditional model rejected.

##  Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
3. **Set up Kaggle API:**
    Ensure your kaggle.json is in C:\Users\YourName\.kaggle\ (Windows) or ~/.kaggle/ (Linux/Mac) to download the dataset automatically.


## Usage Guide (Step-by-Step)
Run the scripts in the following order to reproduce the analysis:

1. **Data Ingestion:** Downloads and samples the raw LendingClub data (10% sample) to prevent memory crashes.
    ```bash
    python 01_data_loader.py

2. **EDA & Feature Engineering:** Cleans the data, performs feature selection, generates correlation plots, and scales features.
    ```bash 
    python 02_eda_preprocessing.py

3. **Train Supervised Model (Deep Learning):** Trains a PyTorch MLP to predict default probability.
    ```bash
    python 03_train_model.py

4. **Train Offline RL Agent**: Trains a CQL (Conservative Q-Learning) agent using d3rlpy to optimize for profit.

   *Note:* This agent uses a profit-maximizing reward function. Our analysis shows it learns an **aggressive policy**, frequently approving high-interest/high-risk borrowers to capture "Alpha" (potential profit) that
   the conservative DL model misses.
    ```bash
    python 04_train_RL

6. **Final Comparison & Analysis:** Evaluates both models on the test set and finds specific "Conflict Cases" where the RL agent disagrees with the DL model.
    ```bash
    python 05_analysis.py


## Key Results
    DL Model: Conservative, High Accuracy on "Safe" loans.
    RL Agent: Aggressive, Profit-Driven.
    Conflict Analysis: The RL agent approved 400+ applicants rejected by the DL model, capturing "Hidden Gems" (borrowers who paid off high-interest loans) but also tends to incur higher risk.



