"""Fine-tuning utilities for online learning from web data"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from model import TransformerLM
from config import MODEL_CONFIG


def cleanup_text(text):
    """Clean web-scraped text for training"""
    if not isinstance(text, str):
        return ""

    # Preserve line granularity first, then normalize each accepted line.
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip empty lines and common web junk
        if not line:
            continue
        if any(skip in line.lower() for skip in ['cookie', 'javascript', 'advertisement', 'click here', 'redirected']):
            continue
        normalized = " ".join(line.split())
        if len(normalized) > 10:  # Only keep substantial lines
            cleaned_lines.append(normalized)

    return " ".join(cleaned_lines)


class PackedTokenDataset(Dataset):
    """Pack one long text into contiguous token blocks once, avoiding repeated tokenization."""

    def __init__(self, text, tokenizer, max_length):
        self.max_length = int(max_length)
        encoded = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        if not encoded:
            self.input_ids = torch.empty((0, self.max_length), dtype=torch.long)
            self.attention_mask = torch.empty((0, self.max_length), dtype=torch.long)
            return

        blocks = []
        masks = []
        for start in range(0, len(encoded), self.max_length):
            chunk = encoded[start:start + self.max_length]
            if len(chunk) < 16:
                continue
            pad_len = self.max_length - len(chunk)
            if pad_len > 0:
                chunk = chunk + [tokenizer.pad_token_id] * pad_len
            mask = [1] * min(len(encoded) - start, self.max_length)
            if pad_len > 0:
                mask += [0] * pad_len
            blocks.append(chunk[:self.max_length])
            masks.append(mask[:self.max_length])

        if not blocks:
            self.input_ids = torch.empty((0, self.max_length), dtype=torch.long)
            self.attention_mask = torch.empty((0, self.max_length), dtype=torch.long)
            return

        self.input_ids = torch.tensor(blocks, dtype=torch.long)
        self.attention_mask = torch.tensor(masks, dtype=torch.long)

    def __len__(self):
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def _build_optimizer(model, learning_rate, device):
    kwargs = {"lr": learning_rate}
    if getattr(device, "type", "cpu") == "cuda":
        try:
            return optim.AdamW(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass
        except Exception:
            pass
    return optim.AdamW(model.parameters(), **kwargs)


def _autocast_context(device):
    if getattr(device, "type", "cpu") == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=MODEL_CONFIG.get("use_mixed_precision", True))
    return torch.amp.autocast(device_type="cpu", enabled=False)


def fine_tune_on_text(model, text, tokenizer, device, learning_rate=1e-5, num_steps=20, checkpoint_path=None, verbose=True):
    """
    Fine-tune model on new text data for quick online learning
    
    Args:
        model: TransformerLM model
        text: String of text to learn from
        tokenizer: Tokenizer
        device: torch device
        learning_rate: Learning rate for fine-tuning
        num_steps: Number of gradient steps to take
        checkpoint_path: Path to save checkpoint after fine-tuning
    
    Returns:
        avg_loss: Average loss during fine-tuning
    """
    # Clean and prepare text
    text = cleanup_text(text)
    if len(text) < 100:
        if verbose:
            print(f"[WARN] Text too short for fine-tuning ({len(text)} chars)")
        return None
    
    # Tokenize once into contiguous blocks for much faster repeated fine-tuning.
    dataset = PackedTokenDataset(text, tokenizer, MODEL_CONFIG["max_seq_length"])
    if len(dataset) == 0:
        if verbose:
            print("[WARN] No usable token blocks for fine-tuning")
        return None

    block_count = len(dataset)
    if block_count >= 24:
        batch_size = 8
    elif block_count >= 8:
        batch_size = 4
    else:
        batch_size = min(2, block_count)

    dataloader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=(getattr(device, "type", "cpu") == "cuda"),
    )
    
    # Setup optimizer
    optimizer = _build_optimizer(model, learning_rate, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(getattr(device, "type", "cpu") == "cuda" and MODEL_CONFIG.get("use_mixed_precision", True)))
    model.train()
    
    total_loss = 0
    steps_taken = 0
    
    if verbose:
        print(f"\n[TRAIN] Fine-tuning on new knowledge ({len(text)} characters)...")

    pbar = tqdm(total=num_steps, desc="Fine-tune", leave=False, disable=(not verbose))
    
    for step in range(num_steps):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            with _autocast_context(device):
                logits = model(input_ids, attention_mask)
            
                # Compute loss (shifted for next-token prediction)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
            
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, MODEL_CONFIG["vocab_size"]),
                    shift_labels.view(-1)
                )
            
            # Backward pass
            optimizer.zero_grad()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            steps_taken += 1
            pbar.update(1)
            
            if steps_taken >= num_steps:
                break
        
        if steps_taken >= num_steps:
            break
    
    pbar.close()
    
    avg_loss = total_loss / steps_taken if steps_taken > 0 else 0
    if verbose:
        print(f"[OK] Fine-tuning complete. Avg loss: {avg_loss:.4f}")
    
    # Save checkpoint
    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        if verbose:
            print(f"[OK] Model saved to {checkpoint_path}")
    
    model.eval()  # Return to eval mode
    return avg_loss


def evaluate_text_loss(model, text, tokenizer, device, max_chars=3000):
    """Evaluate average next-token loss on a text slice."""
    text = cleanup_text(text)
    if len(text) < 80:
        return None

    eval_text = text[:max_chars]
    dataset = PackedTokenDataset(eval_text, tokenizer, MODEL_CONFIG["max_seq_length"])
    if len(dataset) == 0:
        return None

    dataloader = DataLoader(
        dataset,
        batch_size=min(8, len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=(getattr(device, "type", "cpu") == "cuda"),
    )
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with _autocast_context(device):
                logits = model(input_ids, attention_mask)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()

                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, MODEL_CONFIG["vocab_size"]),
                    shift_labels.view(-1)
                )
            total_loss += loss.item()
            steps += 1

    if steps == 0:
        return None
    return total_loss / steps


def fine_tune_with_quality_gate(
    model,
    train_text,
    tokenizer,
    device,
    learning_rate=1e-5,
    num_steps=20,
    checkpoint_path=None,
    eval_text=None,
    min_improvement=0.005,
    verbose=True,
):
    """
    Fine-tune and only save checkpoint if eval loss improves.

    Returns:
        dict with train_loss, before_loss, after_loss, saved, reason
    """
    before_loss = evaluate_text_loss(model, eval_text or train_text, tokenizer, device)
    train_loss = fine_tune_on_text(
        model,
        train_text,
        tokenizer,
        device,
        learning_rate=learning_rate,
        num_steps=num_steps,
        checkpoint_path=None,
        verbose=verbose,
    )
    after_loss = evaluate_text_loss(model, eval_text or train_text, tokenizer, device)

    saved = False
    reason = "no_eval"
    if checkpoint_path:
        if before_loss is None or after_loss is None:
            torch.save(model.state_dict(), checkpoint_path)
            saved = True
            reason = "saved_without_eval"
        else:
            improvement = before_loss - after_loss
            if improvement >= min_improvement:
                torch.save(model.state_dict(), checkpoint_path)
                saved = True
                reason = f"improved_by_{improvement:.4f}"
            else:
                reason = f"not_improved_before={before_loss:.4f}_after={after_loss:.4f}"

    return {
        "train_loss": train_loss,
        "before_loss": before_loss,
        "after_loss": after_loss,
        "saved": saved,
        "reason": reason,
    }


def fine_tune_checkpoint(checkpoint_path, text, tokenizer, device, learning_rate=1e-5, num_steps=20):
    """
    Load a checkpoint, fine-tune it on text, and save it back
    
    Args:
        checkpoint_path: Path to model checkpoint
        text: Text to fine-tune on
        tokenizer: Tokenizer
        device: torch device
        learning_rate: Learning rate
        num_steps: Number of training steps
    
    Returns:
        avg_loss: Loss during fine-tuning
    """
    # Load model and checkpoint
    model = TransformerLM(MODEL_CONFIG)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    
    # Fine-tune
    avg_loss = fine_tune_on_text(
        model,
        text,
        tokenizer,
        device,
        learning_rate,
        num_steps,
        checkpoint_path,
        verbose=True,
    )
    
    return avg_loss


