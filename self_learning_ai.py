"""Self-learning AI core - autonomous learning with minimal restrictions"""

import torch
import os
import json
from datetime import datetime
from pathlib import Path
from fine_tune_utils import fine_tune_on_text
from transformers import AutoTokenizer
from model import TransformerLM
from config import MODEL_CONFIG
from quality_utils import normalize_text, text_quality_score, lexical_overlap_score


class SelfLearningAI:
    """Self-improving AI that learns from conversations and web data"""
    
    def __init__(self, checkpoint_path=None, device=None, enable_autonomous_learning=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Initializing Self-Learning AI on {self.device}...")
        
        # Model setup
        self.model = TransformerLM(MODEL_CONFIG)
        self.checkpoint_path = checkpoint_path or "checkpoints/best_model.pt"
        
        if os.path.exists(self.checkpoint_path):
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Learning settings
        self.enable_autonomous_learning = enable_autonomous_learning
        self.learning_stats = {
            "web_searches": 0,
            "web_fine_tunes": 0,
            "conversation_fine_tunes": 0,
            "total_learned_samples": 0,
            "model_updates": 0
        }
        
        # Knowledge management
        self.knowledge_bank = []  # Store all learned text
        self.conversation_history = []  # Store conversations
        self.learn_topics = set()  # Topics to autonomously learn about
        
        # Paths
        self.data_dir = Path("data/self_learning")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.load_learning_stats()
        self._print_welcome()

    def _assistant_prefix(self):
        return (
            "You are a helpful, accurate assistant. "
            "Answer directly, stay concise, and do not repeat the prompt. "
            "If the prompt is ambiguous, ask one focused clarifying question instead of guessing.\n\n"
        )

    def _score_completion(self, prompt, completion):
        text = normalize_text(completion)
        if not text:
            return 0.0
        score = text_quality_score(text)
        score += 0.25 * min(1.0, len(text) / 220.0)
        score += 0.25 * lexical_overlap_score(prompt, text)
        if len(text.split()) < 6:
            score *= 0.7
        return max(0.0, min(1.0, score))
    
    def _print_welcome(self):
        """Print welcome message with learning status"""
        print("\n" + "="*60)
        print("🤖 SELF-LEARNING AI ACTIVATED")
        print("="*60)
        print(f"Model: {self.checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Auto-Learning: {'ENABLED ✓' if self.enable_autonomous_learning else 'DISABLED'}")
        print(f"\nLearning Stats:")
        print(f"  Web searches learned: {self.learning_stats['web_searches']}")
        print(f"  Auto fine-tunes: {self.learning_stats['web_fine_tunes'] + self.learning_stats['conversation_fine_tunes']}")
        print(f"  Model updates: {self.learning_stats['model_updates']}")
        print("="*60 + "\n")
    
    def save_learning_stats(self):
        """Save learning statistics"""
        stats_file = self.data_dir / "learning_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.learning_stats, f, indent=2)
    
    def load_learning_stats(self):
        """Load learning statistics"""
        stats_file = self.data_dir / "learning_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.learning_stats = json.load(f)
    
    def save_conversation(self, user_text, ai_response):
        """Save conversation for later learning"""
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "user": user_text,
            "ai": ai_response,
            "learned": False
        }
        self.conversation_history.append(entry)
        
        # Auto-save to file
        conv_file = self.data_dir / "conversations.jsonl"
        with open(conv_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    
    def learn_from_conversation_history(self, num_entries=10):
        """Fine-tune on recent conversations"""
        if len(self.conversation_history) < 2:
            return None
        
        # Get recent unlearned conversations
        recent = [c for c in self.conversation_history[-num_entries:] if not c.get("learned", False)]
        if not recent:
            return None
        
        # Combine into training text
        combined_text = "\n".join([
            f"User: {c['user']}\nAI: {c['ai']}"
            for c in recent
        ])
        
        print(f"\n📚 Learning from {len(recent)} conversations...")
        
        avg_loss = fine_tune_on_text(
            self.model,
            combined_text,
            self.tokenizer,
            self.device,
            learning_rate=3e-5,  # Slightly lower for conversation data
            num_steps=15,
            checkpoint_path=self.checkpoint_path
        )
        
        if avg_loss is not None:
            self.learning_stats["conversation_fine_tunes"] += 1
            self.learning_stats["model_updates"] += 1
            
            # Mark as learned
            for c in recent:
                c["learned"] = True
            
            self.save_learning_stats()
            print("✓ Model improved from conversations!\n")
        
        return avg_loss
    
    def learn_from_web(self, text):
        """Learn from web search results"""
        if len(text) < 100:
            return None
        
        print(f"\n🌐 Absorbing web knowledge ({len(text)} chars)...")
        
        avg_loss = fine_tune_on_text(
            self.model,
            text,
            self.tokenizer,
            self.device,
            learning_rate=5e-5,
            num_steps=12,
            checkpoint_path=self.checkpoint_path
        )
        
        if avg_loss is not None:
            self.learning_stats["web_searches"] += 1
            self.learning_stats["web_fine_tunes"] += 1
            self.learning_stats["model_updates"] += 1
            self.learning_stats["total_learned_samples"] += 1
            
            self.knowledge_bank.append(text[:1000])
            self.save_learning_stats()
            print("✓ Model learned from web!\n")
        
        self.model.eval()  # Back to eval mode
        return avg_loss
    
    def learn_topic(self, topic):
        """Mark a topic for autonomous learning"""
        self.learn_topics.add(topic)
        print(f"✓ Will autonomously learn about: {topic}")
    
    def get_learning_summary(self):
        """Get summary of learning progress"""
        return f"""
=== LEARNING SUMMARY ===
Web sources processed: {self.learning_stats['web_searches']}
Fine-tunes from web: {self.learning_stats['web_fine_tunes']}
Fine-tunes from chats: {self.learning_stats['conversation_fine_tunes']}
Total model updates: {self.learning_stats['model_updates']}
Topics to learn: {len(self.learn_topics)}
"""
    
    def generate(self, prompt, max_length=200, temperature=0.7, top_k=50):
        """Generate response with knowledge context"""
        # Add context from recent knowledge
        if self.knowledge_bank:
            context = " ".join(self.knowledge_bank[-2:])[:300]
            prompt = f"Context: {context}\n\nUser: {prompt}"

        prompt = self._assistant_prefix() + prompt
        
        encoding = self.tokenizer(prompt, return_tensors=None)
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        
        candidates = []
        for candidate_temperature in (max(0.2, temperature * 0.85), temperature, min(0.95, temperature * 1.1)):
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    temperature=candidate_temperature,
                    top_k=top_k
                )

            decoded = self.tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)
            decoded = decoded.replace(prompt, "").strip()
            decoded = normalize_text(decoded)
            if decoded:
                candidates.append(decoded)

        if not candidates:
            return "I need a bit more context to answer that well."

        ranked = sorted(candidates, key=lambda item: self._score_completion(prompt, item), reverse=True)
        return ranked[0]
