"""Utility functions for visualization, evaluation, and prediction."""

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from config import *

def create_directories():
    """Create necessary directories for outputs."""
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

def plot_training_history(history):
    """Plot training history metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Balanced Accuracy
    ax3.plot(epochs, history['val_balanced_acc'], 'g-', label='Balanced Accuracy')
    ax3.set_title('Validation Balanced Accuracy')
    ax3.legend()
    ax3.grid(True)
    
    # F1 Score
    ax4.plot(epochs, history['val_f1'], 'm-', label='F1 Score')
    ax4.set_title('Validation F1 Score')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{PLOTS_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_model(y_true, y_pred):
    """Generate detailed evaluation report."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Calculate metrics
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc*100:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    return balanced_acc

def predict_single_image(model, image_path, transform):
    """Predict gender for a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return CLASS_NAMES[predicted_class], confidence

def print_dataset_info(train_loader, val_loader):
    """Print dataset information and class distribution."""
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    # Training set stats
    train_male = sum(1 for label in train_dataset.labels if label == 0)
    train_female = sum(1 for label in train_dataset.labels if label == 1)
    
    # Validation set stats
    val_male = sum(1 for label in val_dataset.labels if label == 0)
    val_female = sum(1 for label in val_dataset.labels if label == 1)
    
    print(f"\nDataset Information:")
    print(f"Training: {train_male} male, {train_female} female (Total: {len(train_dataset)})")
    print(f"Validation: {val_male} male, {val_female} female (Total: {len(val_dataset)})")
    print(f"Imbalance ratio: {train_male/train_female:.2f}:1 (Male:Female)")
    
    return train_male, train_female, val_male, val_female