Master thesis code

This repository contains code for my master thesis project.

The project uses machine learning to detect attacks in network traffic. The code works with a cleaned version of the CSE-CIC-IDS2018 dataset and compares different models.

What the code does:

- Loads the dataset from CSV files
- Prepares and cleans the data
- Changes the labels into two classes:
        - 0 = benign traffic
        - 1 = attack traffic
- Splits the data into training 70%, validation 15% and test sets 15 %, 
- Trains and evaluates three models:
        - Random Forest
        - Isolation Forest
        - Hybrid model


Dataset

The dataset uses a cleaned version of CSE-CIC-IDS2018. Linked to here: https://data.mendeley.com/datasets/29hdbdzx2r/1

To run the code, place the dataset folder in the same folder as main.py.

The folder should be named:

CSE-CIC-IDS2018

How to run:

Install the needed packages:

pip install numpy pandas scikit-learn

Evaluation:

The models are evaluated using accuracy, precision, recall, F1-score, false positive rate and false negative rate.


Author

Kjersti Marie Tofte
