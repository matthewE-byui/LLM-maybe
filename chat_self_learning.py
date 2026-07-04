"""Self-learning chat interface - minimal restrictions, maximum learning"""

import sys
import os
from self_learning_ai import SelfLearningAI
import webbrowser
import urllib.parse
import requests
from bs4 import BeautifulSoup


def search_and_scrape(query):
    """Search and return scraped text"""
    try:
        scrape_url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(scrape_url, timeout=10, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ")
        
        # Minimal cleanup
        lines = [l.strip() for l in text.split('\n') if l.strip() and len(l) > 10]
        cleaned = " ".join(lines[:2000])
        
        return cleaned
    except Exception as e:
        print(f"⚠️  Search failed: {e}")
        return None


def main():
    """Main chat loop with autonomous learning"""
    
    # Initialize self-learning AI
    checkpoint_path = None
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        if os.path.exists("checkpoints/best_model.pt"):
            checkpoint_path = "checkpoints/best_model.pt"
    
    ai = SelfLearningAI(checkpoint_path, enable_autonomous_learning=True)
    
    print("\n" + "="*60)
    print("SELF-LEARNING CHAT - FULL AUTONOMY MODE")
    print("="*60)
    print("\nCommands:")
    print("  'exit' or 'quit' - Leave chat")
    print("  'learn <topic>' - Learn about a topic autonomously")
    print("  'lookup <query>' - Search and learn from web")
    print("  'status' - Show learning progress")
    print("  'history' - Show conversation history")
    print("  'settings' - Adjust parameters")
    print("\nYour AI learns from EVERY interaction!")
    print("="*60 + "\n")
    
    temperature = 0.75
    top_k = 50
    max_length = 150
    auto_learn_conversations = True
    conversation_count = 0
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            # Exit commands
            if user_input.lower() in ['exit', 'quit']:
                print("\n✓ Learning mode disabled. Goodbye!")
                # Save any final stats
                ai.save_learning_stats()
                break
            
            # Status
            if user_input.lower() == 'status':
                print(ai.get_learning_summary())
                continue
            
            # History
            if user_input.lower() == 'history':
                print(f"\n📖 Conversations saved: {len(ai.conversation_history)}")
                if ai.conversation_history:
                    for i, conv in enumerate(ai.conversation_history[-5:]):
                        print(f"\n[{i+1}] {conv['timestamp']}")
                        print(f"  You: {conv['user'][:50]}...")
                print()
                continue
            
            # Settings
            if user_input.lower() == 'settings':
                print(f"\nCurrent settings:")
                print(f"  Temperature: {temperature}")
                print(f"  Top-K: {top_k}")
                print(f"  Max Length: {max_length}")
                print(f"  Auto-learn chats: {auto_learn_conversations}")
                
                try:
                    t = input("Temperature (press Enter to skip): ").strip()
                    if t:
                        temperature = float(t)
                    
                    k = input("Top-K (press Enter to skip): ").strip()
                    if k:
                        top_k = int(k)
                    
                    m = input("Max length (press Enter to skip): ").strip()
                    if m:
                        max_length = int(m)
                    
                    a = input("Disable auto-learn? (y/n, press Enter to skip): ").strip().lower()
                    if a == 'y':
                        auto_learn_conversations = not auto_learn_conversations
                    
                    print("✓ Settings updated!\n")
                except:
                    pass
                continue
            
            # Learn topic
            if user_input.lower().startswith('learn '):
                topic = user_input[6:].strip()
                ai.learn_topic(topic)
                continue
            
            # Web lookup
            if user_input.lower().startswith('lookup '):
                query = user_input[7:].strip()
                print(f"\n🔍 Searching and learning about: {query}...")
                
                # Open browser
                webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote_plus(query))
                print("✓ Browser opened for visual search\n")
                
                # Scrape and learn
                text = search_and_scrape(query)
                if text:
                    ai.learn_from_web(text)
                    print(f"[Web knowledge added] {len(text)} chars\n")
                continue
            
            # Regular chat
            print("\n🤖 AI: Thinking...", end=" ", flush=True)
            response = ai.generate(
                user_input,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
            
            print("\r" + " "*30 + "\r", end="")
            print(f"🤖 AI: {response}\n")
            
            # Save conversation
            ai.save_conversation(user_input, response)
            conversation_count += 1
            
            # Periodic learning from conversations
            if auto_learn_conversations and conversation_count >= 3:
                print("📚 Learning from recent conversations...")
                ai.learn_from_conversation_history(num_entries=3)
                conversation_count = 0
        
        except KeyboardInterrupt:
            print("\n\n✓ Chat ended. Goodbye!")
            ai.save_learning_stats()
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            continue


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
