"""Training script for the LLM"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from model import TransformerLM
from utils import (
    load_wikitext_dataset, 
    get_tokenizer, 
    TextDataset,
    save_checkpoint,
    count_parameters,
    AverageMeter
)
from config import MODEL_CONFIG


def setup_device():
    """Setup device (GPU or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def train_epoch(model, train_loader, optimizer, scaler, device, epoch, gradient_accumulation_steps):
    """Train for one epoch"""
    model.train()
    loss_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Forward pass
        with autocast(enabled=MODEL_CONFIG["use_mixed_precision"]):
            logits = model(input_ids, attention_mask)
            
            # Compute loss (shift for next token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, MODEL_CONFIG["vocab_size"]),
                shift_labels.view(-1)
            )
            
            # Gradient accumulation
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Step optimizer every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MODEL_CONFIG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        loss_meter.update(loss.item() * gradient_accumulation_steps, input_ids.size(0))
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
    
    return loss_meter.avg


def evaluate(model, val_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            logits = model(input_ids, attention_mask)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, MODEL_CONFIG["vocab_size"]),
                shift_labels.view(-1)
            )
            
            loss_meter.update(loss.item(), input_ids.size(0))
    
    perplexity = torch.exp(torch.tensor(loss_meter.avg)).item()
    print(f"Validation Loss: {loss_meter.avg:.4f}, Perplexity: {perplexity:.2f}")
    
    return loss_meter.avg


def train():
    """Main training loop"""
    device = setup_device()
    
    # Create model
    print(f"\nCreating model...")
    model = TransformerLM(MODEL_CONFIG)
    model.to(device)
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Load data
    print(f"\nLoading WikiText dataset...")
    train_texts = load_wikitext_dataset(split="train", num_samples=10000)
    val_texts = load_wikitext_dataset(split="validation", num_samples=2000)
    
    # Tokenizer
    tokenizer = get_tokenizer(MODEL_CONFIG["vocab_size"])
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, MODEL_CONFIG["max_seq_length"])
    val_dataset = TextDataset(val_texts, tokenizer, MODEL_CONFIG["max_seq_length"])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG["batch_size"],
        shuffle=True,
        num_workers=MODEL_CONFIG["num_workers"],
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=MODEL_CONFIG["batch_size"],
        shuffle=False,
        num_workers=MODEL_CONFIG["num_workers"],
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=MODEL_CONFIG["learning_rate"], weight_decay=0.01)
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float("inf")
    print(f"\nStarting training...")
    
    for epoch in range(MODEL_CONFIG["num_epochs"]):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            MODEL_CONFIG["gradient_accumulation_steps"]
        )
        
        val_loss = evaluate(model, val_loader, device)
        
        # Save checkpoint
        if (epoch + 1) % 1 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                MODEL_CONFIG["checkpoint_dir"]
            )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            save_path = os.path.join(MODEL_CONFIG["checkpoint_dir"], "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")
        
        print(f"\nEpoch {epoch + 1}/{MODEL_CONFIG['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
    
    print("Training complete!")


if __name__ == "__main__":
    train()
