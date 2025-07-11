# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import load_student_history, load_target_grade
from model import GradePredictor

# Load input and label
X = load_student_history("data/courses.csv").unsqueeze(0)  # shape: (1, input_size)
y = load_target_grade("B").unsqueeze(0)  # shape: (1,)

# Model
input_size = X.shape[1]
model = GradePredictor(input_size=input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        pred = torch.argmax(output, dim=1).item()
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Predicted class: {pred}")

# Save the model
torch.save(model.state_dict(), "model.pth")
