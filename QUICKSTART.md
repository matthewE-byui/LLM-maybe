# QUICKSTART - Get Running in 5 Minutes

## 1. Install Dependencies (5 min)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
> If you later modify `requirements.txt` (e.g. to add scraping libraries),
> run the `pip install` command again to update the environment.
**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Verify Installation (2 min)

```bash
python verify_setup.py
```

You should see all checkmarks (✓). If not, the error message tells you what to fix.

## 3. Train Your LLM (2-3 hours)

```bash
python train.py
```

This will:
- Download WikiText dataset (~200MB, first time only)
- Train for 3 epochs
- Save best model to `checkpoints/best_model.pt`

**First run output:**
```
Loading WikiText dataset...
Total trainable parameters: 70,285,313
Starting training...
Epoch 1/3: ... loss: 4.25
```

## 4. Chat With Your Model (instant)

After training finishes:

```bash
python inference.py checkpoints/best_model.pt
```

Then type prompts:
```
You: What is AI?
LLM: What is AI? Artificial intelligence refers to...

You: settings
(change temperature, top-k, max length)

You: exit
```

---

**That's it!** You now have a working local LLM.

### Common Issues

**"Out of Memory"**
→ Edit `config.py`, change `"batch_size": 8` to `4`

**"No module named X"**
→ Make sure you ran `pip install -r requirements.txt` in the activated venv

**"No CUDA devices found"**
→ Install PyTorch with CUDA: 
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### What's happening?

1. **Transformer LM** - 70M parameter model
2. **WikiText dataset** - Pre-trained on Wikipedia text
3. **Mixed precision training** - Uses FP16 to fit in RTX 4060
4. **Inference** - Generate new text from prompts

See README.md for full documentation.
