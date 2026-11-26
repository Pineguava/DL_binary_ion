# Ionosphere Binary Classification with TensorFlow & Keras

A binary classification project in which the model predicts whether radar signals directed at Earth's ionosphere indicate a **"good"** return (structure detected) or a **"bad"** return (signal passed through).

---

## Overview

- **Task:** Binary Classification (`0` = `good`, `1` = `bad`)
- **Framework:** TensorFlow / Keras
- **Loss Function:** Binary Cross-Entropy (`binary_crossentropy`)
- **Evaluation Metric:** Binary Accuracy (`binary_accuracy`)

---

## Dataset

The project uses the **Ionosphere dataset (`ion.csv`)** provided in Kaggle's collection:

- **Input Features:** Continuous numeric measurements derived from high-frequency radar signals targeting the ionosphere.
- **Target Column (`Class`):** Categorical label (`good` or `bad`), mapped to binary values:
  ```python
  df['Class'] = df['Class'].map({'good': 0, 'bad': 1})
  ```
- **Train / Validation Split:** 70% training / 30% validation (e.g., `df.sample(frac=0.7, random_state=0)`), with features scaled to `[0, 1]` using min-max normalization.

---

## Key Concepts

1. **Cross-Entropy Loss:** Measures the distance between predicted probability distributions and actual binary labels, providing smooth gradients for classification unlike regression losses (e.g., MAE).
2. **Sigmoid Activation:** Applied in the final output layer (`Dense(1, activation='sigmoid')`) to squash logits into probabilities in `[0, 1]`.
3. **Binary Accuracy:** Evaluates predictions against a `0.5` probability threshold (`metrics=['binary_accuracy']`).
4. **Early Stopping:** Uses `keras.callbacks.EarlyStopping` (`restore_best_weights=True`) to stop training when validation loss stops improving and prevent overfitting.

---

## Model Architecture

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=input_shape),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
```

*(Update layer sizes, batch size, or epochs above if you customized the architecture.)*

---

## Results

| Metric | Value |
| :--- | :---: |
| **Best Validation Loss (`binary_crossentropy`)** | `[0.4867]` |
| **Best Validation Accuracy (`binary_accuracy`)** | `[0.8095]` |

---
