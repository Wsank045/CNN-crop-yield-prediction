# CNN-crop-yield-prediction
This project develops a convolutional neural network (CNN) to predict crop yield from 7-band multispectral UAV orthomosaic tiles (253×253 pixels)

# Features
- Custom PyTorch CNN pipeline for regression
- Supports 7-band multispectral image tiles (253×253 px)
- Flexible dataset loader with band selection and normalization (min-max, z-score)
- Multiple training strategies:
  - Baseline → train/test on same field
  - Mixed → combined training with multiple fields
  - LOFO (Leave-One-Field-Out) → cross-field generalization
  - Cluster → intra-field generalization using clusters - Train on others clusters and test on one cluster
  - Reverse-Cluster → inverse cluster-based evaluation - Train on one cluster and test on others

# Evaluation metrics: MSE, MAE, MAPE, R², Pearson correlation


