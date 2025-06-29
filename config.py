"""Configuration settings for gender classification project."""

import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths
TRAIN_DIR = "data\\train"
VAL_DIR = "data\\val"
TEST_DIR = "data/test"

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
WEIGHT_DECAY = 1e-4

# Model configuration
NUM_CLASSES = 2
PRETRAINED = True
DROPOUT_RATE = 0.5

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# Image preprocessing
IMG_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Output paths
MODEL_SAVE_PATH = "models/best_gender_model.pth"
PLOTS_DIR = "plots"
LOGS_DIR = "logs"

# Class mapping
CLASS_NAMES = ['Male', 'Female']
CLASS_TO_IDX = {'male': 0, 'female': 1}