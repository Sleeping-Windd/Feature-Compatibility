
# Guide Rail Dataset

This directory contains the preprocessed guide rail trajectory dataset used in the experiments reported in the paper.

## Dataset Structure

```text
guiderail/
├── train_input.npy
├── train_target.npy
├── test_input.npy
├── test_target.npy
└── scaler_coord.pkl
```

### File Description

* **train_input.npy**
  Input trajectories for model training.

* **train_target.npy**
  Ground-truth target coordinates corresponding to the training inputs.

* **test_input.npy**
  Input trajectories for model evaluation.

* **test_target.npy**
  Ground-truth target coordinates corresponding to the testing inputs.

* **scaler_coord.pkl**
  Serialized coordinate normalization parameters used during data preprocessing. This scaler should also be used to transform the predicted coordinates back to the original coordinate space.

## Data Preprocessing

All coordinate values have been normalized before training using the provided coordinate scaler.

During inference, the predicted coordinates should be converted back to the original coordinate system by applying the inverse transformation of `scaler_coord.pkl`.

## Notes

* The released dataset is identical to the one used in the experiments reported in the main text.
* No additional preprocessing is required before training.
* Users should load `scaler_coord.pkl` when converting model predictions to the original coordinate space.
