# EMG Gesture Classification with Deep Learning

## 📌 Overview
This project explores different deep learning architectures for **Electromyography (EMG)-based gesture recognition**.  
The goal is to benchmark various model families (CNN, LSTM, CNN–LSTM hybrids, Transformer, Conformer) to evaluate their effectiveness at classifying EMG signals into gesture classes.

The project is organized into Jupyter notebooks, each dedicated to a specific experiment.

---

## 📂 Dataset
We use **DB1 from the Ninapro dataset**, which contains surface EMG (sEMG) signals recorded from multiple channels while subjects perform different hand and wrist movements.

- **Input format**: `(N, T, C)` numpy arrays
  - `N`: number of windows
  - `T`: window length
  - `C`: number of EMG channels
- **Labels**: Integer-encoded gesture IDs
- **Preprocessing**:
  - Normalization per channel
  - Sliding-window segmentation
  - Train/test splits with balanced classes
- Stored as `.npz` files for efficient loading.

---

## 🧪 Experiments

### 1. **Data Preparation**
- Implemented in `04_db1_preprocessing.ipynb`
- Converts raw EMG to fixed-length segments
- Saves processed data for training/testing

---

### 2. **CNN Baseline** (`05_db1_cnn_baseline.ipynb`)
- Pure convolutional approach
- Architecture:
  - 2 Conv1D blocks (Conv → Norm → ReLU → Pool)
  - Temporal pooling
  - Dense classification head

---

### 3. **LSTM Baseline** (`06_db1_lstm_baseline.ipynb`)
- Sequential modeling with BiLSTMs
- Architecture:
  - 2 stacked BiLSTMs
  - Temporal pooling
  - Fully connected classifier

---

### 4. **CNN–LSTM Hybrid** (`07_db1_cnn_lstm_baseline.ipynb`)
- CNN feature extractor + LSTM sequence model
- Flexible design:
  - 1–2 Conv1D layers
  - 2–3 LSTM layers
- Captures both **local patterns** and **long-range dependencies**

---

### 5. **Transformer Baseline** (`08_db1_transformer_baseline.ipynb`)
- Self-attention based model
- Architecture:
  - Input projection
  - Transformer encoder layers
  - Positional encoding (learnable)
  - Mean pooling → Classifier

---

### 6. **Conformer** (`09_db1_conformer_baseline.ipynb`)
- Hybrid of **convolutions** and **self-attention**
- Architecture:
  - Conformer blocks (Feed-forward → MHA → Conv → Feed-forward)
  - Residual connections and normalization
  - Mean pooling → Dense classifier
- Expected to better capture both **local temporal dynamics** and **global dependencies**

---

## ⚙️ Training Setup
- **Optimizer**: AdamW with weight decay
- **Loss**: Weighted cross-entropy (to handle class imbalance)
- **Learning rate**: `1e-3` (default)
- **Batch size**: Tuned per experiment (32–128)
- **Metrics**: Accuracy, F1-score, confusion matrix

---

## 📊 Results (to be updated)
Each experiment logs:
- Training/validation accuracy curves
- Final test accuracy
- Class-wise performance (confusion matrix)

A comparison table will be added once all models are trained.

---

## 🚀 Future Work
- Hyperparameter optimization
- Data augmentation (e.g., jitter, scaling, channel dropout)
- Subject-independent evaluation
- Real-time inference experiments

---

## 🗂 Repository Structure
```
├── 04_db1_preprocessing.ipynb
├── 05_db1_cnn_baseline.ipynb
├── 06_db1_lstm_baseline.ipynb
├── 07_db1_cnn_lstm_baseline.ipynb
├── 08_db1_transformer_baseline.ipynb
├── 09_db1_conformer_baseline.ipynb
├── models/          # Model definitions
├── data/            # Processed .npz files
└── README.md
```

✍️ Maintained as part of ongoing research into **EMG-based gesture recognition**.