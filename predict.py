# predict.py

import torch
from model import GradePredictor

# --- Hypothetical New Course Data ---
# Let's predict the grade for a course with:
# - an average historical GPA (CAPE_GPA) of 3.5
# - an instructor rating of 4.2
new_course_features = torch.tensor([[3.5, 4.2]], dtype=torch.float32)

# 1. Reload trained model
# The input_size must match what it was trained with (2 features)
input_size = 2
model = GradePredictor(input_size=input_size)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 2. Predict
with torch.no_grad():
    logits = model(new_course_features)
    predicted_class = torch.argmax(logits, dim=1).item()

# 3. Map back to grade
# Note: Your original map was reversed. A=4, B=3, etc.
reverse_grade_map = {4: 'A', 3: 'B', 2: 'C', 1: 'D', 0: 'F'}
print(f"Predicted grade for the new course: {reverse_grade_map[predicted_class]}")