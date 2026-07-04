"""Transformer LLM model implementation"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)"""
    
    def __init__(self, dim, max_seq_length=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_length = max_seq_length
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb


def rotate_half(x):
    """Rotate half the hidden dims of the input"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, emb):
    """Apply rotary positional embeddings to q and k"""
    emb = emb.unsqueeze(0).unsqueeze(0)
    q_embed = q * torch.cos(emb) + rotate_half(q) * torch.sin(emb)
    k_embed = k * torch.cos(emb) + rotate_half(k) * torch.sin(emb)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE"""
    
    def __init__(self, hidden_size, num_attention_heads, attention_dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        assert hidden_size % num_attention_heads == 0
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.query(hidden_states).reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = self.key(hidden_states).reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        v = self.value(hidden_states).reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        
        # Apply rotary embeddings
        ropeemb = self.rotary_emb(seq_len, hidden_states.device)
        q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        q, k = apply_rotary_pos_emb(q, k, ropeemb)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_len, self.hidden_size)
        
        output = self.output(context)
        return output


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, hidden_size, intermediate_size, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """Single transformer layer with attention and feed-forward"""
    
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, 
                 dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, attention_dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, intermediate_size, dropout)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, hidden_states, attention_mask=None):
        # Attention with residual connection
        attn_output = self.attention(hidden_states, attention_mask)
        attn_output = self.dropout1(attn_output)
        hidden_states = self.norm1(hidden_states + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(hidden_states)
        ff_output = self.dropout2(ff_output)
        hidden_states = self.norm2(hidden_states + ff_output)
        
        return hidden_states


class TransformerLM(nn.Module):
    """Full transformer language model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.position_embedding = nn.Embedding(config["max_seq_length"], config["hidden_size"])
        self.embedding_dropout = nn.Dropout(config["hidden_dropout_prob"])
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                config["hidden_size"],
                config["num_attention_heads"],
                config["intermediate_size"],
                config["hidden_dropout_prob"],
                config["attention_probs_dropout_prob"]
            )
            for _ in range(config["num_hidden_layers"])
        ])
        
        self.norm = nn.LayerNorm(config["hidden_size"])
        
        # Output layer
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"])
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask (lower triangular)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1
        )
        
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
            causal_mask = causal_mask + attention_mask
        else:
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Token and position embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        
        hidden_states = token_emb + pos_emb
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask)
        
        hidden_states = self.norm(hidden_states)
        
        # Output logits
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_k=50):
        """Generate text given initial input"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for next token
                outputs = self.forward(input_ids[:, -self.config["max_seq_length"]:])
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop if we generate end token (50256 is typically EOS)
                if next_token.item() == 50256:
                    break
        
        return input_ids
