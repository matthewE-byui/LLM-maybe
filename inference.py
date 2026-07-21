"""Inference and chat interface for the trained LLM"""

import torch
import os
from transformers import AutoTokenizer
from model import TransformerLM
from config import MODEL_CONFIG
from fine_tune_utils import fine_tune_on_text
from model_paths import resolve_active_checkpoint
from quality_utils import normalize_text, text_quality_score, lexical_overlap_score


class LLMChat:
    """Chat interface for the LLM"""
    
    def __init__(self, checkpoint_path=None, device=None, auto_learn=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading model...")
        self.model = TransformerLM(MODEL_CONFIG)
        
        checkpoint_path = checkpoint_path or resolve_active_checkpoint()
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            except TypeError:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print("Using untrained model (random weights)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # memory of looked-up text (will be prepended to prompts)
        self.knowledge = []
        
        # Web learning settings
        self.checkpoint_path = checkpoint_path
        self.auto_learn = auto_learn
        self.learning_count = 0
        self.learning_threshold = 1  # Fine-tune every N web searches

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
    
    def generate(self, prompt, max_length=200, temperature=0.7, top_k=50):
        """Generate text completion from a prompt with knowledge memory"""
        # prepend knowledge to prompt (simple concatenation, truncating if long)
        if self.knowledge:
            mem_text = "\n".join(self.knowledge[-3:])  # keep last 3 lookups
            prompt = mem_text + "\n" + prompt

        prompt = self._assistant_prefix() + prompt
        
        # Tokenize prompt manually (don't use return_tensors)
        encoding = self.tokenizer(prompt, return_tensors=None)
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        # ensure batch dimension
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

            decoded_ids = output_ids[0].cpu().tolist()
            completion = self.tokenizer.decode(decoded_ids, skip_special_tokens=True)
            completion = completion.replace(prompt, "").strip()
            completion = normalize_text(completion)
            if completion:
                candidates.append(completion)

        if not candidates:
            return "I need a bit more context to answer that well."

        ranked = sorted(candidates, key=lambda item: self._score_completion(prompt, item), reverse=True)
        return ranked[0]
    
    def chat(self):
        """Interactive chat loop"""
        import webbrowser, urllib.parse
        
        def search_and_open(q):
            # open a browser window with a search for q (Google)
            browse_url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(q)
            webbrowser.open(browse_url)
            # fetch static HTML results from DuckDuckGo for scraping
            scrape_url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(q)
            # also download text from the scraped results (simple scrape)
            try:
                import requests
                from bs4 import BeautifulSoup
                # fetch google results page
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                # scrape using DuckDuckGo html interface
                resp = requests.get(scrape_url, timeout=10, headers=headers)
                soup = BeautifulSoup(resp.text, "html.parser")
                # attempt to extract first result link
                first_link = None
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("/url?q=") and "google.com" not in href:
                        first_link = href.split("/url?q=")[1].split("&")[0]
                        break
                if first_link:
                    try:
                        r2 = requests.get(first_link, timeout=10, headers=headers)
                        s2 = BeautifulSoup(r2.text, "html.parser")
                        text = s2.get_text(separator=" ")
                    except Exception:
                        text = soup.get_text(separator=" ")
                else:
                    text = soup.get_text(separator=" ")
                # simple cleanup: drop boilerplate lines
                lines = []
                for line in text.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    # skip common unwanted fragments
                    if "Google Search" in stripped or "click here" in stripped.lower() or "redirected" in stripped.lower():
                        continue
                    lines.append(stripped)
                cleaned = " ".join(lines)
                # show a bit of what we grabbed for debugging
                print("\n[lookup scraped text sample]:", cleaned[:500], "\n")
                # store for future context, truncating to avoid huge memory
                self.knowledge.append(cleaned[:2000])
                
                # Auto fine-tuning on web results
                self.learning_count += 1
                if self.auto_learn and self.learning_count >= self.learning_threshold and len(cleaned) > 100:
                    try:
                        print("🧠 Auto-learning from web results...")
                        fine_tune_on_text(
                            self.model,
                            cleaned,
                            self.tokenizer,
                            self.device,
                            learning_rate=5e-5,  # Smaller learning rate for stability
                            num_steps=10,  # Quick fine-tune
                            checkpoint_path=self.checkpoint_path
                        )
                        self.learning_count = 0  # Reset counter
                        print("✓ Model updated with new knowledge!\n")
                        self.model.eval()  # Make sure we're in eval mode for inference
                    except Exception as e:
                        print(f"⚠️  Fine-tuning failed: {e}\n")
                        
            except Exception:
                pass
            return f"Opened browser search for: {q}"        
        print("\n" + "="*60)
        print("LLM Chat Interface")
        print("="*60)
        print("Type 'exit' to quit, 'settings' to see/change parameters")
        print("  Enter 'lookup <query>' to perform a web search")
        print("  🧠 Auto-learning enabled: model improves from web searches")
        print("="*60 + "\n")
        
        temperature = 0.7
        top_k = 50
        max_length = 150
        
        while True:
            prompt = input("\nYou: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() == "exit":
                print("Goodbye!")
                break
            
            if prompt.lower() == "settings":
                print(f"\nCurrent settings:")
                print(f"  Temperature: {temperature} (0.0-1.0, higher = more creative)")
                print(f"  Top-K: {top_k} (1-100, higher = more diverse)")
                print(f"  Max Length: {max_length} (tokens, 10-500)")
                print(f"  Auto-Learning: {'ON' if self.auto_learn else 'OFF'} (learns from web searches)")
                
                try:
                    new_temp = input("New temperature (press Enter to skip): ").strip()
                    if new_temp:
                        temperature = float(new_temp)
                    
                    new_k = input("New top-k (press Enter to skip): ").strip()
                    if new_k:
                        top_k = int(new_k)
                    
                    new_len = input("New max length (press Enter to skip): ").strip()
                    if new_len:
                        max_length = int(new_len)
                    
                    learn_input = input("Toggle auto-learning? (y/n, press Enter to skip): ").strip().lower()
                    if learn_input == 'y':
                        self.auto_learn = not self.auto_learn
                        print(f"Auto-Learning: {'ON' if self.auto_learn else 'OFF'}")
                    
                    print("Settings updated!")
                except ValueError:
                    print("Invalid input, keeping settings unchanged")
                continue
            
            # handle lookup commands
            if prompt.lower().startswith("lookup "):
                query = prompt[7:].strip()
                result = search_and_open(query)
                print(result)
                continue
            
            print("LLM: Generating...", end=" ", flush=True)
            response = self.generate(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
            print("\r" + " "*30 + "\r", end="")  # Clear "Generating..."
            print(f"LLM: {response}")


def main():
    """Main function"""
    import sys
    
    # Check for checkpoint path argument
    checkpoint_path = None
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Look for best model in checkpoints directory
        checkpoints_dir = MODEL_CONFIG["checkpoint_dir"]
        best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
    
    chat = LLMChat(checkpoint_path)
    chat.chat()


if __name__ == "__main__":
    main()
