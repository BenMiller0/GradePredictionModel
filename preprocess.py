# preprocess.py

import pandas as pd
import torch

# Map letter grades to numbers
grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}

def load_student_history(csv_path):
    df = pd.read_csv(csv_path)

    # Map letter grades to numbers
    df['Grade_Received'] = df['Grade_Received'].map(grade_map)

    # Flatten CAPE GPA, instructor rating, and grade into a single input tensor
    features = []
    for _, row in df.iterrows():
        features.extend([row['CAPE_GPA'], row['Instructor_Rating'], row['Grade_Received']])

    return torch.tensor(features, dtype=torch.float32)

def load_target_grade(letter):
    return torch.tensor(grade_map[letter], dtype=torch.long)

# Example usage
if __name__ == "__main__":
    input_tensor = load_student_history("data/courses.csv")
    target_tensor = load_target_grade("B")  # set this to the grade you want to predict
    print("Input Tensor:", input_tensor)
    print("Target Tensor:", target_tensor)
