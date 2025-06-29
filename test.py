"""Validation script for gender prediction on val dataset."""

import os
import argparse
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import get_transforms
from model import load_model
from utils import predict_single_image
from config import *

def load_images_from_folder(folder):
    """Return list of image file paths from the given folder."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def evaluate_folder(model, image_paths, label, transform):
    """Predict all images in a folder and return predictions and labels."""
    predictions, labels = [], []
    for img_path in tqdm(image_paths, desc=f"Evaluating {'male' if label == 0 else 'female'}"):
        try:
            pred_class, _ = predict_single_image(model, img_path, transform)
            pred_label = 0 if pred_class.lower() == "male" else 1
            predictions.append(pred_label)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return predictions, labels

def main():
    """Main evaluation function for validation set."""
    parser = argparse.ArgumentParser(description='Evaluate model on validation set')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH, help='Path to trained model')
    parser.add_argument('--val_dir', type=str, default='data/test', help='Path to test folder')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)

    # Get transform (no augmentation)
    _, val_transform = get_transforms()

    # Load image paths
    male_dir = os.path.join(args.val_dir, 'male')
    female_dir = os.path.join(args.val_dir, 'female')
    male_images = load_images_from_folder(male_dir)
    female_images = load_images_from_folder(female_dir)

    # Run evaluation
    male_preds, male_labels = evaluate_folder(model, male_images, label=0, transform=val_transform)
    female_preds, female_labels = evaluate_folder(model, female_images, label=1, transform=val_transform)

    # Combine results
    all_preds = male_preds + female_preds
    all_labels = male_labels + female_labels

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n📊 Validation Results:")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"F1-Score   : {f1:.4f}")

if __name__ == "__main__":
    main()
