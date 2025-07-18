# preprocess.py

import pandas as pd
import torch

# Map letter grades to numbers
grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}

def load_data(csv_path):
    """
    Loads data and treats each course (row) as a separate sample.
    Returns features (X) and labels (y).
    """
    df = pd.read_csv(csv_path)

    # Map letter grades to numbers for the target variable
    df['Grade_Received'] = df['Grade_Received'].map(grade_map)

    # Features are the CAPE GPA and Instructor Rating
    features = df[['CAPE_GPA', 'Instructor_Rating']].values
    
    # Labels are the grades received
    labels = df['Grade_Received'].values

    # Convert to PyTorch tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long) # CrossEntropyLoss expects long tensors for labels

    return X, y

# Example usage
if __name__ == "__main__":
    X, y = load_data("data/courses.csv")
    print("Features Tensor (X) shape:", X.shape)
    print("Labels Tensor (y) shape:", y.shape)
    print("\nFirst 5 features:\n", X[:5])
    print("\nFirst 5 labels:\n", y[:5])