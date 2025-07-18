import torch.nn as nn
import torch.nn.functional as F

class GradePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=16, output_size=5):
        super(GradePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)  # 5 classes: A–F

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw scores (logits)


# Only runs if executed directly - for testing purposes
if __name__ == "__main__":
    import torch

    from preprocess import load_student_history

    input_tensor = load_student_history("data/courses.csv")
    model = GradePredictor(input_size=len(input_tensor))

    with torch.no_grad():
        output = model(input_tensor)
        print("Logits:", output)
        print("Predicted class:", torch.argmax(output).item())
