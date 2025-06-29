"""Main training script for gender classification."""

import os
from dataset import create_data_loaders
from model import create_model
from trainer import Trainer
from utils import create_directories, plot_training_history, evaluate_model, print_dataset_info
from config import *

def main():
    """Main training function."""
    print("=== Gender Classification Training ===")
    
    # Create necessary directories
    create_directories()
    
    # Check if data directories exist
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print(f"Error: Please ensure data directories exist:")
        print(f"  - {TRAIN_DIR}")
        print(f"  - {VAL_DIR}")
        return
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, class_weights = create_data_loaders(TRAIN_DIR, VAL_DIR)
    
    # Print dataset information
    print_dataset_info(train_loader, val_loader)
    
    # Create model
    print(f"\nCreating model on {DEVICE}...")
    model = create_model()
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, class_weights)
    
    # Train the model
    val_preds, val_labels = trainer.train(EPOCHS)
    
    # Plot training history
    plot_training_history(trainer.history)
    
    # Evaluate the model
    final_balanced_accuracy = evaluate_model(val_labels, val_preds)
    
    print(f"\nTraining completed!")
    print(f"Final balanced accuracy: {final_balanced_accuracy*100:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()