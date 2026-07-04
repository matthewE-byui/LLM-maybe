"""Configuration for the LLM model"""

# Model configuration
MODEL_CONFIG = {
    # Model architecture
    "vocab_size": 50257,  # Similar to GPT-2
    "max_seq_length": 1024,
    "hidden_size": 512,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "intermediate_size": 2048,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    
    # Training
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 5e-4,
    "num_epochs": 3,
    "warmup_steps": 500,
    "max_grad_norm": 1.0,
    
    # Device and precision
    "use_mixed_precision": True,
    "device": "cuda",  # Will be set to cuda if available, else cpu
    
    # Data
    "dataset_name": "wikitext",
    "dataset_config": "wikitext-2-v1",
    "internet_urls": [
        # add URLs here (plain text or HTML will be scraped)
        # e.g. "https://en.wikipedia.org/wiki/Artificial_intelligence"
    ],
    "num_workers": 0,
    
    # Checkpointing
    "checkpoint_dir": "checkpoints",
    "save_steps": 500,
    "eval_steps": 100,
}
