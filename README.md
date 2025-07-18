# 🎓 GradePredictionModel

A PyTorch-based machine learning model that predicts a student's final letter grade in a target course using their past academic performance and CAPE/SET course data.

This proof-of-concept takes a single student’s course history (including course CAPE GPA, instructor rating, and grade received), flattens it into a feature vector, and trains a feedforward neural network to classify the expected grade (A–F) in a future course.

---

## 🚀 Features

- Custom preprocessing of student course history
- Grade-to-numeric encoding and classification
- Feedforward neural network with cross-entropy loss

---

## 📦 Requirements

- Python 3.10  
- pip packages:
  - `torch`
  - `pandas`

---

## 🛠️ Setup Instructions

### 1. Clone the repo (or download files)

```bash
git clone https://github.com/yourusername/GradePredictionModel.git
cd GradePredictionModel
```

### 2. Create a virtual environment

```bash
py -3.10 -m venv venv
```

### 3. Activate the virtual environment

```bash
# On Windows (CMD or Git Bash)
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install torch pandas
```

---

## 🧪 Running the Project

1. **Preprocess student history into tensors:**

```bash
python preprocess.py
```

2. **Train the model on the student data:**

```bash
python train.py
```

3. **Run inference to predict a grade:**

```bash
python predict.py
```

---

## 📁 File Structure

```
GradePredictionModel/
├── data
    ├── student_history.csv      # Input course history for one student
├── preprocess.py            # Converts history into input tensor
├── model.py                 # Defines the PyTorch model
├── train.py                 # Trains the model
├── predict.py               # Loads model and predicts grade
├── model.pth                # Saved trained weights (after training)
└── README.md
```

