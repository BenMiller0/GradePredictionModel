# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from preprocess import load_data
from model import GradePredictor

# 1. Load the entire dataset
X, y = load_data("data/courses.csv")

# 2. Create a dataset and dataloader to handle batching
# This will shuffle the data and create mini-batches of 8 samples
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 3. Model - The input size is now 2 (CAPE_GPA, Instructor_Rating)
input_size = X.shape[1] # This will be 2
model = GradePredictor(input_size=input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
for epoch in range(100):
    total_loss = 0
    # Loop over batches of data
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch:03d} | Average Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "model.pth")
print("\nTraining complete. Model saved to model.pth")