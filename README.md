# Gender Classification from Face Images

A PyTorch-based deep learning project for binary gender classification using facial images with ResNet18 backbone.

## Features

- **Imbalanced Dataset Handling**: Weighted sampling, class-weighted loss, balanced accuracy metrics
- **Data Augmentation**: Strong augmentation for minority class to improve generalization
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Comprehensive Evaluation**: Balanced accuracy, F1-score, per-class metrics
- **Modular Design**: Clean separation of concerns across multiple files

## Project Structure

```
├── model/
    ├── best_gender_model.pth         Downlaod from drive Link - https://drive.google.com/file/d/13rTJIkq_diaEp28wCgE5jsdqeAuHnDkQ/view?usp=sharing
├── Task_A/
    ├── config.py              # Configuration settings
    ├── dataset.py             # Dataset class and data loading
    ├── model.py               # Neural network architecture
    ├── trainer.py             # Training logic
    ├── utils.py               # Utility functions (plotting, evaluation)
    ├── main.py                # Main training script
    ├── test.py                # Test script
    ├── requirements.txt       # Dependencies
    └── data/                  # Dataset directory
        ├── train/
        │   ├── male/
        │   └── female/
        └── val/
            ├── male/
            └── female/
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your dataset:**
   - Organize images in the structure shown above
   - Supported formats: PNG, JPG, JPEG
   - Images will be automatically resized to 224x224

3. **Configure paths:**
   - Edit `config.py` to set your data paths and hyperparameters
   - Default paths: `data/train` and `data/val`

## Usage

### Training

```bash
python main.py
```

Key features during training:
- Automatic class weight calculation for imbalanced data
- Weighted random sampling for balanced batches
- Multi-metric evaluation (accuracy, balanced accuracy, F1-score)
- Best model selection based on balanced accuracy
- Training history plots and confusion matrix

### Testing

```bash
python test.py
```

### Configuration

Edit `config.py` to customize:
- **Data paths**: `TRAIN_DIR`, `VAL_DIR`
- **Hyperparameters**: `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`
- **Model settings**: `DROPOUT_RATE`, `PRETRAINED`
- **Output paths**: `MODEL_SAVE_PATH`, `PLOTS_DIR`

## Handling Imbalanced Data

This project includes several techniques for imbalanced datasets:

1. **Training Level**:
   - Weighted random sampling ensures balanced batches
   - Class-weighted CrossEntropyLoss penalizes minority class errors more
   - Strong data augmentation for better minority class representation

2. **Evaluation Level**:
   - Balanced accuracy as primary metric
   - Per-class precision, recall, and F1-scores
   - Confusion matrix visualization
   - Bias detection warnings

3. **Model Selection**:
   - Best model chosen by balanced accuracy
   - Early stopping prevents overfitting
   - Learning rate scheduling based on validation performance

## Expected Results

For typical imbalanced face datasets:
- **Regular accuracy**: ~85-90% (can be misleading due to class imbalance)
- **Balanced accuracy**: ~75-85% (more reliable metric)
- **Training time**: ~10-20 minutes on GPU for 20 epochs

## Files Overview

- **config.py**: Central configuration management
- **dataset.py**: Custom dataset class with weighted sampling
- **model.py**: ResNet18-based architecture with custom classifier head
- **trainer.py**: Training loop with balanced metrics and early stopping
- **utils.py**: Visualization, evaluation, and prediction utilities
- **main.py**: Orchestrates the complete training pipeline
- **test.py**: Test script

## Key Improvements Over Monolithic Code

1. **Modularity**: Each file has a single responsibility
2. **Reusability**: Components can be easily reused or modified
3. **Maintainability**: Easier to debug and extend
4. **Configuration**: Centralized settings management
5. **Clean Interface**: Clear separation between training and inference

## GPU/CPU Support

The code automatically detects and uses GPU if available, with fallback to CPU. Check `config.py` for device configuration.
