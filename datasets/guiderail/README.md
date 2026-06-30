# Guide Rail Dataset

This directory contains the preprocessed Guide Rail trajectory dataset used in the experiments reported in the main paper.

## Dataset Structure

```text
guiderail/
├── train_input.npy
├── train_target.npy
├── test_input.npy
├── test_target.npy
└── scaler_coord.pkl
```

## File Description

- **train_input.npy**
  - Training input trajectories.
  - Shape: **(N, 50, 32)**

- **train_target.npy**
  - Ground-truth position labels corresponding to the training inputs.
  - Shape: **(N, 8)**

- **test_input.npy**
  - Testing input trajectories.
  - Shape: **(N, 50, 32)**

- **test_target.npy**
  - Ground-truth position labels corresponding to the testing inputs.
  - Shape: **(N, 8)**

- **scaler_coord.pkl**
  - Coordinate normalization parameters used during data preprocessing.
  - This scaler should be used to perform the inverse transformation of the predicted coordinates and recover the original coordinate values.

---

## Data Format

Each input sample consists of **50 consecutive time steps**, where each time step contains **32 features**.

The feature layout is organized as follows:

| Feature Index | Description |
|---------------|-------------|
| 0–7 | Position features |
| 8–15 | Velocity features |
| 16–23 | Acceleration features |
| 24–29 | Reserved features (not used in this work) |
| 30–31 | Depth features |

The target labels contain the corresponding ground-truth positions:

| Label Index | Description |
|-------------|-------------|
| 0–7 | Position labels |

---

## Data Preprocessing

All coordinate values have been normalized before training.

The normalization parameters are stored in **`scaler_coord.pkl`**.

During inference, the predicted coordinates should be transformed back to the original coordinate space by applying the inverse transformation of the provided scaler.

---

## Notes

- The released dataset is identical to the one used in the experiments reported in the main paper.
- No additional preprocessing is required before training.
- The provided normalization parameters should be used to recover predictions in the original coordinate space.
