"""Utility functions for training and inference"""

import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from config import MODEL_CONFIG


class TextDataset(Dataset):
    """Dataset for text tokenization"""
    
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None  # Don't return tensors, just lists
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        # Convert to tensors manually
        import torch
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def download_web_text(urls):
    """Fetch and return text content from a list of URLs."""
    import requests
    from bs4 import BeautifulSoup
    texts = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            # strip HTML tags if present
            if "<" in resp.text and ">" in resp.text:
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text(separator=" ")
            else:
                text = resp.text
            texts.append(text)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    return texts


def load_wikitext_dataset(split="train", num_samples=None):
    """Load WikiText dataset or optionally augment with internet URLs."""
    texts = []
    try:
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-v1",
            split=split,
            trust_remote_code=True
        )
        # Filter out empty texts
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
        texts = dataset["text"].to_list()
        print(f"✓ Loaded WikiText dataset: {len(texts)} samples")
    except Exception as e:
        print(f"Error loading from Hugging Face: {e}")
        print("Using fallback dataset...")
        texts = [
            "Artificial intelligence is the ability of machines to learn and make decisions. " * 5
            for _ in range(100)
        ]
    
    # Optionally fetch extra text from web
    urls = MODEL_CONFIG.get("internet_urls", [])
    if urls:
        print("Downloading additional text from the internet...")
        web_texts = download_web_text(urls)
        texts.extend(web_texts)
    
    if num_samples is not None and num_samples < len(texts):
        texts = texts[:num_samples]
    
    return texts


def load_openwebtext_dataset(split="train", num_samples=1000):
    """Load OpenWebText dataset (larger web text corpus)"""
    texts = []
    try:
        print("Loading OpenWebText dataset (this may take a moment)...")
        dataset = load_dataset("openwebtext", split=split, trust_remote_code=True)
        
        # Limit to num_samples for faster loading
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        texts = dataset["text"].to_list()
        print(f"✓ Loaded OpenWebText: {len(texts)} samples")
    except Exception as e:
        print(f"Error loading OpenWebText: {e}")
        print("Falling back to WikiText...")
        texts = load_wikitext_dataset(split, num_samples)
    
    return texts


def load_ccnews_dataset(split="train", num_samples=500):
    """Load CC-News dataset (news articles)"""
    texts = []
    try:
        print("Loading CC-News dataset (this may take a moment)...")
        dataset = load_dataset("cc_news", split=split, trust_remote_code=True)
        
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        texts = [item["text"] for item in dataset]
        print(f"✓ Loaded CC-News: {len(texts)} samples")
    except Exception as e:
        print(f"Error loading CC-News: {e}")
        print("Falling back to WikiText...")
        texts = load_wikitext_dataset(split, num_samples)
    
    return texts


def load_dataset_by_name(dataset_name, split="train", num_samples=None):
    """Load dataset by name"""
    dataset_loaders = {
        "wikitext": lambda: load_wikitext_dataset(split, num_samples),
        "openwebtext": lambda: load_openwebtext_dataset(split, num_samples or 1000),
        "cc-news": lambda: load_ccnews_dataset(split, num_samples or 500),
    }
    
    if dataset_name not in dataset_loaders:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {', '.join(dataset_loaders.keys())}")
        return load_wikitext_dataset(split, num_samples)
    
    return dataset_loaders[dataset_name]()


def get_tokenizer(vocab_size=None):
    """Get or create tokenizer"""
    # Use GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_attention_mask(seq_length, device):
    """Create attention mask for padding"""
    return torch.ones((1, seq_length), device=device)


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    return epoch, loss


def count_parameters(model):
    """Count total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
