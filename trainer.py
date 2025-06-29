"""Training logic for gender classification model."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score
from config import *

class Trainer:
    """Handles model training and validation."""
    
    def __init__(self, model, train_loader, val_loader, class_weights):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss and optimizer with class weights
        weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_balanced_acc': [], 'val_f1': []
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return running_loss / len(self.train_loader), 100 * correct / total
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100 * correct / total
        val_balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100
        val_f1 = f1_score(all_labels, all_preds, average='macro') * 100
        
        return val_loss, val_acc, val_balanced_acc, val_f1, all_preds, all_labels
    
    def train(self, epochs=EPOCHS):
        """Train the model for specified epochs."""
        print(f"Starting training on {DEVICE}...")
        
        best_balanced_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_balanced_acc, val_f1, val_preds, val_labels = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_balanced_acc)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_balanced_acc'].append(val_balanced_acc)
            self.history['val_f1'].append(val_f1)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Val Balanced Acc: {val_balanced_acc:.2f}%, Val F1: {val_f1:.2f}%")
            
            # Save best model
            if val_balanced_acc > best_balanced_acc:
                best_balanced_acc = val_balanced_acc
                torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
                print("✓ Best model saved!")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining completed! Best balanced accuracy: {best_balanced_acc:.2f}%")
        return val_preds, val_labels