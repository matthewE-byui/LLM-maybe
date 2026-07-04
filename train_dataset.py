#!/usr/bin/env python3
"""Dataset selector and trainer for continuous learning"""

import sys
import os
from pathlib import Path

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from utils import load_dataset_by_name, get_tokenizer, TextDataset, count_parameters, AverageMeter
from train import setup_device, train_epoch, evaluate
from model import TransformerLM
from config import MODEL_CONFIG

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm


def main():
    """Main function"""
    print("\n" + "="*60)
    print("🧠 Dataset Trainer - Continuous Learning")
    print("="*60 + "\n")
    
    # Show available datasets
    datasets = {
        "1": ("wikitext", 5000, "WikiText (default, ~5k samples)"),
        "2": ("openwebtext", 2000, "OpenWebText (web text, ~2k samples)"),
        "3": ("cc-news", 1000, "CC-News (news articles, ~1k samples)"),
    }
    
    print("Available datasets:")
    for key, (name, samples, desc) in datasets.items():
        print(f"  {key}. {desc}")
    print()
    
    choice = input("Select dataset (1-3): ").strip()
    
    if choice not in datasets:
        print("❌ Invalid choice!")
        return
    
    dataset_name, num_samples, desc = datasets[choice]
    print(f"\n📥 Loading {desc}...")
    
    # Load dataset
    texts = load_dataset_by_name(dataset_name, split="train", num_samples=num_samples)
    
    if not texts:
        print("❌ Failed to load dataset")
        return
    
    print(f"✓ Loaded {len(texts)} samples")
    
    # Get training parameters
    print("\n" + "="*60)
    print("Training Parameters:")
    print("="*60)
    
    epochs_input = input(f"Number of epochs (default 3): ").strip()
    epochs = int(epochs_input) if epochs_input else 3
    
    batch_size_input = input(f"Batch size (default 4, higher = faster but uses more memory): ").strip()
    batch_size = int(batch_size_input) if batch_size_input else 4
    
    lr_input = input(f"Learning rate (default 5e-5): ").strip()
    learning_rate = float(lr_input) if lr_input else 5e-5
    
    print("\n" + "="*60)
    
    # Setup
    device = setup_device()
    tokenizer = get_tokenizer()
    
    # Load or create model
    checkpoint_path = "checkpoints/best_model.pt"
    model = TransformerLM(MODEL_CONFIG)
    
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("\nNo checkpoint found, training from scratch...")
    
    model.to(device)
    
    print(f"Model: {count_parameters(model):,} parameters")
    print(f"Dataset: {len(texts)} texts from {dataset_name}")
    print(f"Training: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    
    # Create dataset and dataloader
    print("\nPreparing dataset...")
    dataset = TextDataset(texts, tokenizer, MODEL_CONFIG["max_seq_length"])
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Setup optimizer and scaler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    gradient_accumulation_steps = 1
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    best_val_loss = float('inf')
    best_checkpoint_path = checkpoint_path
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, gradient_accumulation_steps
        )
        
        # Validate
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}\n")
        
        # Save if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"✓ Saved best model to {best_checkpoint_path}\n")
    
    print("="*60)
    print("✓ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {best_checkpoint_path}")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Training cancelled!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
