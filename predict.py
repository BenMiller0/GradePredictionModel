# predict.py

import torch
from model import GradePredictor
from preprocess import load_student_history

# Load input tensor
X = load_student_history("data/courses.csv").unsqueeze(0)

# Reload trained model
input_size = X.shape[1]
model = GradePredictor(input_size=input_size)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Predict
with torch.no_grad():
    logits = model(X)
    predicted_class = torch.argmax(logits, dim=1).item()

# Map back to grade
reverse_grade_map = {0: 'F', 1: 'D', 2: 'C', 3: 'B', 4: 'A'}
print(f"Predicted grade: {reverse_grade_map[predicted_class]}")
