"""Gender classification model architecture."""

import torch
import torch.nn as nn
from torchvision import models
from config import *

class GenderClassifier(nn.Module):
    """CNN model for gender classification using ResNet18 backbone."""
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=PRETRAINED):
        super(GenderClassifier, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace final layer for binary classification
        self.backbone.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_model():
    """Create and return the model instance."""
    model = GenderClassifier().to(DEVICE)
    return model

def load_model(model_path):
    """Load a trained model from checkpoint."""
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model