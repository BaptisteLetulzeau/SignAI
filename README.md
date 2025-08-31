# T-YEP-600_msc2027 END 3rd YEAR PROJECT EPITECH

# MNIST Digit Classification with TensorFlow

This project implements a neural network for handwritten digit recognition using the MNIST dataset.

## Description

The code creates a `MNIST` class that encapsulates all necessary steps to:
- Load and preprocess MNIST data
- Build a dense neural network
- Train the model
- Evaluate performance

## Requirements

```bash
cd /install/install.txt
```

## Model Architecture

The neural network includes:
- **Input layer**: 784 neurons (28×28 pixels flattened)
- **Hidden layer 1**: 128 neurons + ReLU + Dropout (0.2)
- **Hidden layer 2**: 64 neurons + ReLU + Dropout (0.2)
- **Output layer**: 10 neurons + Softmax final activation (for 10 digits)

## Features

### Data Preprocessing
- Pixel normalization (0-255 → 0-1)
- Transformation to 1D vectors
- One-hot encoding of labels

### Training
- Optimizer: Adam
- Loss function: categorical_crossentropy
- Callbacks: Early stopping and learning rate reduction
- Batch size: 128
- Default epochs: 15

### Evaluation
- Test accuracy calculation
- Predictions on test set

## Configurable Parameters

- `epochs`: Number of training epochs (default: 15)
- `batch_size`: Batch size (default: 128)
- Network architecture modifiable in `build_model()`

## Expected Results

The model should achieve approximately 97-98% accuracy on MNIST test data.

## Final Results

The model achieve approximately 98% accuracy, that is perfect for neuronal network performances.

## Technical Notes

- Random seeds fixed for reproducibility (`np.random.seed(42)`, `tf.random.set_seed(42)`)
- Dropout used to prevent overfitting
- Callbacks for training optimization
