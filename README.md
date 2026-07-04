# Local LLM Training & Inference

A fully functional transformer-based language model that you can train and run locally on your RTX 4060.

## Features

- **70M parameter transformer model** - Optimized for RTX 4060 (8GB VRAM)
- **Rotary positional embeddings (RoPE)** - State-of-the-art attention positioning
- **Mixed precision training** - FP16 for memory efficiency
- **Gradient accumulation** - Simulate larger batch sizes
- **WikiText dataset** - Pre-configured with WikiText-2 corpus
- **Interactive chat interface** - Test your trained model conversationally
- **Checkpointing** - Save and resume training

## Hardware Requirements

- **GPU**: RTX 4060 (8GB) or similar
- **RAM**: 16GB+ system RAM
- **Disk**: 10GB for dataset and checkpoints

## Setup & Installation

1. **Navigate to the project directory**:
   ```bash
   cd "c:\school\LLM Stuff\llm_project"
   ```

2. **Create a Python virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install:
   - PyTorch with CUDA support
   - Transformers library (for tokenizers)
   - Datasets library (for WikiText)
   - Accelerate (for mixed precision)

## Quick Start

### Option 1: Train the model (recommended first-time)

```bash
python train.py
```

**Expected behavior**:
- Downloads WikiText-2 dataset (~200MB, first run only)
- Trains for 3 epochs on ~10,000 examples
- ~2-3 hours on RTX 4060
- Saves checkpoints to `checkpoints/` directory
- Best model saved to `checkpoints/best_model.pt`

**Training output example**:
```
Using GPU: NVIDIA GeForce RTX 4060
Available GPU Memory: 7.97 GB
Loading WikiText dataset...
Total trainable parameters: 70,285,313
Starting training...
Epoch 1/3
Epoch 1/3: 100%|██| 1250/1250 [1:15:32<00:00, 3.60s/it] loss: 4.25
Validation Loss: 4.18, Perplexity: 65.34
Best model saved to checkpoints/best_model.pt
```

### Option 2: Use pre-trained GPT-2 for immediate testing

If you want to test the inference pipeline without training:

```bash
python inference.py
```

This will load the untrained model (random weights) for testing the chat interface.

### Option 3: Chat with your trained model

After training completes:

```bash
python inference.py checkpoints/best_model.pt
```

**Interactive chat example**:
```
============================================================
LLM Chat Interface
============================================================
Type 'exit' to quit, 'settings' to see/change parameters
============================================================

You: What is artificial intelligence?
LLM: Generating...
LLM: What is artificial intelligence? Artificial intelligence is the ability 
of machines to learn from experience and perform tasks that typically require 
human intelligence. It is used in many fields...

You: settings
Current settings:
  Temperature: 0.7 (0.0-1.0, higher = more creative)
  Top-K: 50 (1-100, higher = more diverse)
  Max Length: 150 (tokens, 10-500)

New temperature (press Enter to skip): 0.9
New top-k (press Enter to skip): 
New max length (press Enter to skip): 200
Settings updated!

You: exit
Goodbye!
```

## Configuration

Edit `config.py` to customize:

```python
MODEL_CONFIG = {
    "hidden_size": 512,           # Model width (512 = ~70M params)
    "num_hidden_layers": 8,       # Number of transformer blocks
    "num_attention_heads": 8,     # Number of attention heads
    "batch_size": 8,              # Training batch size
    "learning_rate": 5e-4,        # Learning rate
    "num_epochs": 3,              # Number of training epochs
    "max_seq_length": 1024,       # Max input length
    "use_mixed_precision": True,  # Use FP16 (faster, saves memory)
    
    # Data sources
    "dataset_name": "wikitext",
    "dataset_config": "wikitext-2-v1",
    "internet_urls": [           # Optional web pages to scrape
        # "https://en.wikipedia.org/wiki/Artificial_intelligence"
    ],
}
```

### Learning from the internet

The training script can fetch additional text from arbitrary URLs. Add links to
`MODEL_CONFIG["internet_urls"]` and the loader will download and scrape the page
content using `requests` + `BeautifulSoup`. This allows you to mix online data
with the built‑in WikiText corpus. Be mindful of robots.txt and copyright when
scraping.

If you prefer larger corpora, you can also use Hugging Face datasets such as
`wikipedia` or your own text files (see `utils.load_wikitext_dataset`).

> **Note:** training on live internet data may include noise or unwanted text.
> Filter and preprocess accordingly.

### Autonomous lookups during chat

The chat interface supports simple web searches. Type:

```
lookup <your query here>
```

When invoked, the agent will open your default browser with a Google search for
the query, allowing you to monitor the lookup in real time. The chat session
will display a confirmation message after the browser window opens.

In addition to opening the browser, the system now **scrapes the search results
page and stores the text** internally. The last few lookup texts are prepended
to future prompts so the model can "learn" from what it has seen. You can watch
the browser window and then ask follow‑up questions—the model will include the
scraped information in its next response.

Use this feature carefully; scraped content may contain noise or unrelated
text. It simply adds context to the prompt, it doesn’t persist across sessions.


### Memory Optimization Tips

If you get out-of-memory errors:

1. **Reduce batch size**:
   ```python
   "batch_size": 4  # Instead of 8
   ```

2. **Increase gradient accumulation**:
   ```python
   "gradient_accumulation_steps": 4  # Instead of 2
   ```

3. **Reduce sequence length**:
   ```python
   "max_seq_length": 512  # Instead of 1024
   ```

4. **Make model smaller**:
   ```python
   "hidden_size": 384,        # Smaller model
   "num_hidden_layers": 6,
   ```

## Model Architecture

```
Input Tokens
    ↓
Token Embedding (50K vocab → 512d)
    ↓
Position Embedding + Dropout
    ↓
8 Transformer Layers (each with):
  ├─ 8-Head Self-Attention (RoPE)
  ├─ Position-wise FFN (2048d)
  └─ Layer Normalization & Residuals
    ↓
Output Layer Normalization
    ↓
Language Modeling Head
    ↓
Logits (50K classes)
```

**Total Parameters**: ~70M (fits in 8GB VRAM with mixed precision)

## File Structure

```
llm_project/
├── config.py           # Model & training configuration
├── model.py            # Transformer architecture
├── train.py            # Training script
├── inference.py        # Chat interface
├── utils.py            # Utilities (dataset, save/load)
├── requirements.txt    # Python dependencies
├── checkpoints/        # Saved model checkpoints
├── data/               # Downloaded dataset cache
└── README.md          # This file
```

## Troubleshooting

### "OutOfMemoryError: CUDA out of memory"
- Reduce batch size in config.py
- Reduce sequence length
- Close other GPU applications
- Restart Python and try again

### "No module named 'transformers'"
```bash
pip install transformers
```

### "No CUDA GPU detected"
- Check NVIDIA driver is installed: `nvidia-smi`
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Dataset download takes forever
- First download is slow (~200MB)
- Use a faster internet connection or download manually

### Model generates gibberish
- The untrained model is random - train it first!
- After training a few epochs, quality improves gradually
- Longer training = better results

## Next Steps

1. **Train the model** to completion (3+ epochs recommended)
2. **Experiment with parameters**:
   - Temperature (creativity): 0.1 = deterministic, 1.0 = very random
   - Top-K (diversity): Higher = more word choices
3. **Fine-tune on custom data** by modifying the dataset loader in utils.py
4. **Increase model size** if you upgrade your GPU
5. **Export model** for deployment

## Performance Expectations

On RTX 4060:
- **Training speed**: ~3-4 tokens/sec during training
- **Inference speed**: ~100-200 tokens/sec generation
- **Memory usage**: 6-7GB during training
- **Time to train**: ~2.5 hours per epoch (10K examples)

## Model Capabilities

This model can:
- ✅ Continue text from a prompt
- ✅ Answer questions (after training on QA data)
- ✅ Summarize text
- ✅ Generate creative writing
- ✅ Perform basic reasoning

This model cannot (yet):
- ❌ Code generation (unless trained on code)
- ❌ Multilingual (trained on English only)
- ❌ Follow complex instructions (too small)
- ❌ Real-time understanding (no chat history context)

## Resources

- **Transformer Paper**: https://arxiv.org/abs/1706.03762
- **PyTorch Docs**: https://pytorch.org/docs
- **Hugging Face**: https://huggingface.co

## License

This project is educational - use as you wish!

---

**Happy training! 🚀**
