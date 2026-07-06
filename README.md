# MNIST Handwritten Digit Classifier using PyTorch

A simple Feedforward Neural Network (FNN) built using **PyTorch** to classify handwritten digits from the **MNIST dataset**. The model learns to recognize digits **0–9** using fully connected layers, the **ReLU activation function**, **CrossEntropy Loss**, and the **Adam optimizer**.

After training, the model is evaluated on the MNIST test dataset and reports its classification accuracy.

---

## Features

- Downloads the MNIST dataset automatically
- Converts images into PyTorch tensors
- Uses DataLoader for mini-batch training
- Implements a Feedforward Neural Network using `nn.Module`
- Uses ReLU activation function
- Uses CrossEntropy Loss for multi-class classification
- Uses Adam optimizer
- Trains the model for 5 epochs
- Evaluates the model on the test dataset
- Displays the final test accuracy

---

## Dataset

**MNIST Handwritten Digits**

- Training Images: **60,000**
- Test Images: **10,000**
- Image Size: **28 × 28 pixels**
- Color Format: **Grayscale (1 channel)**
- Number of Classes: **10 (Digits 0–9)**

Example Classes:

```
0 1 2 3 4 5 6 7 8 9
```

---

## Technologies Used

- Python 3
- PyTorch
- Torchvision

---

## Installation

Install the required packages:

```bash
pip install torch torchvision
```

---

## Run the Program

```bash
python MNIST_Digit_Classifier.py
```

The MNIST dataset will automatically download the first time the program is executed.

---

## Project Workflow

```
MNIST Images
      │
      ▼
Convert Images to Tensor
      │
      ▼
Flatten (28×28 → 784)
      │
      ▼
Fully Connected Layer (784 → 128)
      │
      ▼
ReLU Activation
      │
      ▼
Fully Connected Layer (128 → 10)
      │
      ▼
CrossEntropy Loss
      │
      ▼
Backpropagation
      │
      ▼
Adam Optimizer
      │
      ▼
Updated Weights
```

---

## Model Architecture

```
Input Image
(1 × 28 × 28)
        │
        ▼
Flatten
784 Features
        │
        ▼
Linear Layer
784 → 128
        │
        ▼
ReLU Activation
        │
        ▼
Linear Layer
128 → 10
        │
        ▼
Output Scores (Digits 0–9)
```

---

## Activation Function

The hidden layer uses:

```python
F.relu()
```

ReLU (Rectified Linear Unit) introduces non-linearity into the network.

Formula:

```
ReLU(x) = max(0, x)
```

Example:

| Input | Output |
|-------|--------|
| -5 | 0 |
| -1 | 0 |
| 0 | 0 |
| 3 | 3 |
| 8 | 8 |

---

## Loss Function

The project uses:

```python
nn.CrossEntropyLoss()
```

CrossEntropy Loss compares the predicted scores with the correct digit labels.

- Small loss → Correct prediction
- Large loss → Incorrect prediction

It is commonly used for **multi-class classification** problems.

---

## Optimizer

The optimizer used is:

```python
torch.optim.Adam(model.parameters(), lr=0.01)
```

Adam automatically adjusts the learning rate for each model parameter, allowing faster and more stable training.

---

## Training Process

For every batch:

1. Load a batch of images.
2. Perform a forward pass.
3. Compute the CrossEntropy Loss.
4. Clear previous gradients.
5. Perform backpropagation.
6. Update the model parameters using Adam.

This process repeats for all batches in one epoch.

---

## Testing Process

After training:

- Gradient calculation is disabled using:

```python
with torch.no_grad():
```

- The model predicts the digit for every test image.
- The predicted digit is compared with the actual label.
- The overall classification accuracy is calculated.

Accuracy Formula:

```
Accuracy = (Correct Predictions / Total Images) × 100
```

---

## Output Example

```
Loss: 0.18 for Epoch: 1
Loss: 0.10 for Epoch: 2
Loss: 0.07 for Epoch: 3
Loss: 0.05 for Epoch: 4
Loss: 0.03 for Epoch: 5

Test Accuracy: 97.80%
```

*(Actual values may vary slightly each run because the model starts with randomly initialized weights.)*

---

## Concepts Covered

- Tensors
- Neural Networks
- Feedforward Neural Networks (FNN)
- Fully Connected Layers
- Forward Pass
- ReLU Activation
- CrossEntropy Loss
- Backpropagation
- Gradient Descent
- Adam Optimizer
- Epochs
- Mini-Batch Training
- DataLoader
- MNIST Dataset
- Model Evaluation
- Prediction Accuracy

---

## Future Improvements

- Add more hidden layers
- Include Dropout for regularization
- Save and load trained models
- Train using GPU (CUDA)
- Plot training loss and accuracy graphs
- Replace the Feedforward Neural Network with a Convolutional Neural Network (CNN) for higher accuracy

---

## Author

Created as a beginner-friendly PyTorch project for learning handwritten digit classification, neural networks, and deep learning fundamentals.
