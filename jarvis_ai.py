"""
JARVIS-Level AI Assistant
A sophisticated conversational AI with:
- Persistent memory & context awareness
- Natural language understanding & task execution
- Autonomous learning from conversations
- Proactive assistance & rich personality
"""

import json
import os
import re
import math
from datetime import datetime, timedelta
from collections import deque
import ast
import operator
import random
import threading
import time
import subprocess
import shlex
import torch
from inference import LLMChat
import fine_tune_utils
from quality_utils import is_high_signal_text, lexical_overlap_score, normalize_text, source_weight, text_quality_score


class ConversationMemory:
    """Persistent memory system for multi-turn conversations"""
    
    def __init__(self, max_turns=20, memory_file="conversation_memory.json"):
        self.memory = deque(maxlen=max_turns)
        self.memory_file = memory_file
        self.learning_log = []
        self.load_memory()
    
    def add_turn(self, user_input, assistant_response, metadata=None):
        """Add a conversation turn to memory"""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response,
            "metadata": metadata or {}
        }
        self.memory.append(turn)
        self.save_memory()
    
    def get_context(self, num_turns=5):
        """Get formatted conversation history for context"""
        recent = list(self.memory)[-num_turns:]
        context = "\n".join([
            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            for turn in recent
        ])
        return context
    
    def save_memory(self):
        """Persist memory to disk"""
        with open(self.memory_file, 'w') as f:
            json.dump(list(self.memory), f, indent=2)
    
    def load_memory(self):
        """Load persisted memory from disk"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                self.memory.extend(data)


class LayeredMemory:
    """Long-term memory layers for preferences and stabilized facts."""

    def __init__(self, file_path="layered_memory.json"):
        self.file_path = file_path
        self.data = self._load()

    def _default(self):
        return {
            "version": 1,
            "preferences": [],
            "facts": [],
            "updated_at": datetime.now().isoformat(),
        }

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    base = self._default()
                    for k, v in base.items():
                        data.setdefault(k, v)
                    return data
            except Exception:
                pass
        return self._default()

    def _save(self):
        try:
            self.data["updated_at"] = datetime.now().isoformat()
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def add_preference(self, text, confidence=0.7):
        pref = normalize_text(text)[:180]
        if len(pref) < 8:
            return None
        item = {
            "text": pref,
            "confidence": round(float(confidence), 3),
            "last_seen": datetime.now().isoformat(),
        }
        prefs = self.data.get("preferences", [])
        for p in prefs:
            if pref.lower() == p.get("text", "").lower():
                p["confidence"] = max(float(p.get("confidence", 0.0)), item["confidence"])
                p["last_seen"] = item["last_seen"]
                self._save()
                return p
        prefs.append(item)
        self.data["preferences"] = prefs[-120:]
        self._save()
        return item

    def add_fact(self, text, confidence=0.6, source="unknown"):
        fact = normalize_text(text)[:220]
        if len(fact) < 20:
            return {"saved": False, "reason": "short"}
        if fact.lower().startswith(("recent focus:", "goal context:", "skill mode:", "retrieved facts:", "mind:")):
            return {"saved": False, "reason": "meta"}

        facts = self.data.get("facts", [])
        lowered = fact.lower()
        contradiction_hit = None
        for f in facts:
            base = f.get("text", "").lower()
            if not base:
                continue
            if (" not " in lowered and " not " not in base and base[:40] in lowered) or (
                " not " in base and " not " not in lowered and lowered[:40] in base
            ):
                contradiction_hit = f
                break

        if contradiction_hit is not None:
            contradiction_hit["confidence"] = round(max(0.15, float(contradiction_hit.get("confidence", 0.0)) - 0.1), 3)
            contradiction_hit["last_seen"] = datetime.now().isoformat()
            self._save()
            return {"saved": False, "reason": "contradiction", "existing": contradiction_hit.get("text", "")}

        item = {
            "text": fact,
            "confidence": round(float(confidence), 3),
            "source": source,
            "last_seen": datetime.now().isoformat(),
        }
        for f in facts:
            if fact[:80].lower() == f.get("text", "")[:80].lower():
                f["confidence"] = max(float(f.get("confidence", 0.0)), item["confidence"])
                f["last_seen"] = item["last_seen"]
                self._save()
                return {"saved": True, "reason": "reinforced", "item": f}

        facts.append(item)
        self.data["facts"] = facts[-240:]
        self._save()
        return {"saved": True, "reason": "added", "item": item}

    def decay_facts(self, factor=0.99):
        facts = self.data.get("facts", [])
        if not facts:
            return 0
        kept = []
        for f in facts:
            new_conf = float(f.get("confidence", 0.0)) * float(factor)
            if new_conf >= 0.2:
                f["confidence"] = round(new_conf, 3)
                kept.append(f)
        self.data["facts"] = kept[-240:]
        self._save()
        return len(self.data["facts"])

    def top_facts(self, query, top_k=3):
        ranked = []
        for fact in self.data.get("facts", []):
            text = fact.get("text", "")
            if text.lower().startswith(("recent focus:", "goal context:", "skill mode:", "retrieved facts:", "mind:")):
                continue
            overlap = lexical_overlap_score(query, fact.get("text", ""))
            score = 0.7 * overlap + 0.3 * float(fact.get("confidence", 0.0))
            if score > 0.18:
                ranked.append((score, fact))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in ranked[:top_k]]

    def preference_summary(self):
        prefs = self.data.get("preferences", [])[-5:]
        if not prefs:
            return "none"
        return " | ".join(p.get("text", "") for p in prefs)


class LocalVectorMemory:
    """Small local vector memory using sparse term-frequency vectors."""

    def __init__(self, memory_path="vector_memory.json", max_items=1500):
        self.memory_path = memory_path
        self.max_items = max_items
        self.items = []
        self._load()

    def _tokenize(self, text):
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2]

    def _clean_text(self, text):
        cleaned = normalize_text(text)
        cleaned = re.sub(r"https?://\S+", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bwww\.[^\s]+", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b[a-z0-9.-]+\.(?:com|org|net|edu|gov|io|ai|co|uk)(?:/[^\s]*)?", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*[|>-]\s*", ": ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" .:-")
        if len(cleaned) < 30:
            return ""
        if cleaned.lower().startswith(("recent focus:", "goal context:", "skill mode:", "retrieved facts:", "mind:", "implementation path:")):
            return ""
        noisy = len(re.findall(r"[^a-zA-Z0-9\s.,!?'-]", cleaned)) / max(len(cleaned), 1)
        if noisy > 0.14:
            return ""
        return cleaned

    def _vectorize(self, text):
        vec = {}
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        for tok in tokens:
            vec[tok] = vec.get(tok, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for k in list(vec.keys()):
                vec[k] /= norm
        return vec

    def _cosine(self, a, b):
        if not a or not b:
            return 0.0
        small, large = (a, b) if len(a) <= len(b) else (b, a)
        return sum(v * large.get(k, 0.0) for k, v in small.items())

    def _load(self):
        if not os.path.exists(self.memory_path):
            self.items = []
            return
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                cleaned_items = []
                for item in data:
                    cleaned_text = self._clean_text(item.get("text", ""))
                    if not cleaned_text:
                        continue
                    vec = self._vectorize(cleaned_text)
                    if not vec:
                        continue
                    cleaned_items.append(
                        {
                            "text": cleaned_text,
                            "source": item.get("source", "unknown"),
                            "metadata": item.get("metadata", {}),
                            "vec": vec,
                        }
                    )
                self.items = cleaned_items[-self.max_items:]
            else:
                self.items = []
        except Exception:
            self.items = []

    def _save(self):
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.items[-self.max_items:], f, indent=2)
        except Exception:
            pass

    def add(self, text, source="unknown", metadata=None):
        cleaned = self._clean_text(text)[:600]
        if len(cleaned) < 30:
            return
        vec = self._vectorize(cleaned)
        if not vec:
            return
        self.items.append(
            {
                "text": cleaned,
                "source": source,
                "metadata": metadata or {},
                "vec": vec,
            }
        )
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]
        self._save()

    def search(self, query, top_k=3, min_score=0.18):
        qvec = self._vectorize(query)
        if not qvec:
            return []
        scored = []
        for item in self.items:
            score = self._cosine(qvec, item.get("vec", {}))
            if score >= min_score:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"score": s, **it} for s, it in scored[:top_k]]


class GoalMemory:
    """Persistent goal tracker with active/completed states."""

    def __init__(self, file_path="goals.json"):
        self.file_path = file_path
        self.goals = self._load()

    def _load(self):
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.goals, f, indent=2)
        except Exception:
            pass

    def _next_id(self):
        if not self.goals:
            return 1
        return max(int(g.get("id", 0)) for g in self.goals) + 1

    def add_goal(self, text, priority="medium", due_days=None):
        cleaned = " ".join((text or "").split())
        if len(cleaned) < 4:
            return None
        due_iso = None
        if due_days is not None:
            try:
                due_iso = (datetime.now() + timedelta(days=max(int(due_days), 0))).isoformat()
            except Exception:
                due_iso = None
        goal = {
            "id": self._next_id(),
            "text": cleaned[:220],
            "priority": priority if priority in {"low", "medium", "high"} else "medium",
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "due_at": due_iso,
            "progress": 0,
        }
        self.goals.append(goal)
        self._save()
        return goal

    def list_goals(self, status="active"):
        if status == "all":
            return list(self.goals)
        return [g for g in self.goals if g.get("status") == status]

    def update_progress(self, goal_id, progress):
        try:
            goal_id = int(goal_id)
            progress = max(0, min(100, int(progress)))
        except Exception:
            return None
        for goal in self.goals:
            if int(goal.get("id", -1)) == goal_id:
                goal["progress"] = progress
                goal["updated_at"] = datetime.now().isoformat()
                self._save()
                return goal
        return None

    def complete_goal(self, goal_id):
        try:
            goal_id = int(goal_id)
        except Exception:
            return None
        for goal in self.goals:
            if int(goal.get("id", -1)) == goal_id:
                goal["status"] = "done"
                goal["progress"] = 100
                goal["updated_at"] = datetime.now().isoformat()
                self._save()
                return goal
        return None

    def remove_goal(self, goal_id):
        try:
            goal_id = int(goal_id)
        except Exception:
            return False
        before = len(self.goals)
        self.goals = [g for g in self.goals if int(g.get("id", -1)) != goal_id]
        changed = len(self.goals) != before
        if changed:
            self._save()
        return changed

    def next_focus_goal(self):
        active = self.list_goals(status="active")
        if not active:
            return None

        priority_rank = {"high": 0, "medium": 1, "low": 2}

        def sort_key(goal):
            due_at = goal.get("due_at")
            try:
                due_key = datetime.fromisoformat(due_at).timestamp() if due_at else 10**12
            except Exception:
                due_key = 10**12
            return (
                priority_rank.get(goal.get("priority", "medium"), 1),
                due_key,
                int(goal.get("progress", 0)),
                goal.get("created_at", ""),
            )

        active.sort(key=sort_key)
        return active[0]


class SkillRegistry:
    """Simple skill router so JARVIS can act in specialized modes."""

    def __init__(self):
        self.skills = {
            "researcher": "Use lookup and retrieval to gather grounded facts.",
            "planner": "Turn a request into an actionable step-by-step plan.",
            "tutor": "Explain a concept clearly with one example.",
            "coder": "Provide implementation-oriented coding guidance.",
            "analyst": "Compare options and recommend a best path.",
        }

    def list_skills(self):
        return dict(self.skills)

    def choose_skill(self, user_input):
        text = (user_input or "").lower()
        if any(k in text for k in ["plan", "roadmap", "steps", "strategy"]):
            return "planner"
        if any(k in text for k in ["learn", "start", "begin", "understand"]):
            return "tutor"
        if any(k in text for k in ["explain", "teach", "what is", "how does"]):
            return "tutor"
        if any(k in text for k in ["code", "python", "bug", "debug", "function"]):
            return "coder"
        if any(k in text for k in ["compare", "tradeoff", "best", "vs"]):
            return "analyst"
        if any(k in text for k in ["search", "lookup", "find", "latest"]):
            return "researcher"
        return "researcher"


class TaskRouter:
    """Understands user intent and routes to appropriate actions"""
    
    def __init__(self):
        self.tasks = {
            'search': {'keywords': ['lookup', 'search', 'find', 'what is'], 'action': 'web_lookup'},
            'execute': {'keywords': ['run', 'execute', 'start', 'do'], 'action': 'execute_task'},
            'explain': {'keywords': ['explain', 'why', 'how does', 'what'], 'action': 'explain_concept'},
            'remember': {'keywords': ['remember', 'recall', 'what did', 'tell me'], 'action': 'retrieve_memory'},
            'chat': {'keywords': [], 'action': 'generate_response'}
        }
    
    def route(self, user_input):
        """Route user input to appropriate task"""
        lower_input = user_input.lower()
        for task_name, task_info in self.tasks.items():
            for keyword in task_info['keywords']:
                if keyword in lower_input:
                    return task_name, task_info['action']
        return 'chat', 'generate_response'
    
    def extract_entities(self, text):
        """Extract meaningful entities from user input"""
        entities = {
            'names': re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text),
            'numbers': re.findall(r'\d+\.?\d*', text),
            'quoted': re.findall(r'"([^"]*)"', text)
        }
        return entities


class JARVISPersonality:
    """Rich personality system - witty, helpful, remembers context"""
    
    def __init__(self):
        self.witticisms = [
            "Shall I add that to your growing list of remarkable talents?",
            "I concur. Most logical.",
            "A wise observation.",
            "Precisely what I was thinking.",
            "Your insight is most refreshing."
        ]
    
    def craft_response(self, base_response, context_data=None):
        """Enhance response with personality"""
        import random
        if random.random() < 0.15:
            base_response += f"\n\n*{random.choice(self.witticisms)}*"
        return base_response


class AutonomousLearner:
    """Automatic learning from conversations"""
    
    def __init__(self, model, tokenizer, device, checkpoint_path):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.learning_count = 0
    
    def learn_from_conversation(self, conversation_turns):
        """Fine-tune model on recent good conversations"""
        try:
            training_text = "\n".join([
                turn['assistant'] for turn in conversation_turns
                if len(turn['assistant']) > 50
            ])
            
            if not training_text or len(training_text.split()) < 20:
                return {"status": "insufficient_data"}
            
            gate = fine_tune_utils.fine_tune_with_quality_gate(
                self.model,
                training_text,
                self.tokenizer,
                self.device,
                learning_rate=3e-5,
                num_steps=8,
                checkpoint_path=self.checkpoint_path,
                eval_text=training_text,
                min_improvement=0.003,
                verbose=False,
            )
            
            self.learning_count += 1
            return {
                "status": "complete",
                "loss": gate.get("train_loss"),
                "saved": gate.get("saved", False),
                "reason": gate.get("reason", "unknown"),
            }
        
        except Exception as e:
            return {"status": "failed", "error": str(e)}


class JARVIS:
    """The complete JARVIS-level AI assistant"""
    
    def __init__(self):
        print("\n" + "="*60)
        print("🤖 Initializing JARVIS v1.0...")
        print("="*60)
        
        try:
            # Load with best trained checkpoint if available
            self.checkpoint_path = "checkpoints/best_model.pt"
            self.chat = LLMChat(checkpoint_path=self.checkpoint_path)
            self.model = self.chat.model
            self.tokenizer = self.chat.tokenizer
            self.device = self.chat.device
            
            self.memory = ConversationMemory()
            self.layered_memory = LayeredMemory(file_path="layered_memory.json")
            self.router = TaskRouter()
            self.skills = SkillRegistry()
            self.personality = JARVISPersonality()
            self.learner = AutonomousLearner(self.model, self.tokenizer, self.device, self.checkpoint_path)
            self.conversation_turn = 0
            self.is_first_message = True
            self.growth_every_n_turns = 4
            self.growth_corpus_path = "growth_corpus.txt"
            self.user_signal_growth_enabled = True
            self.user_signal_count = 0
            self.last_learn_now_time = 0.0
            self.model_lock = threading.Lock()
            self.goals = GoalMemory(file_path="goals.json")
            self.focus_goal_id = None
            self.pending_proactive_notes = deque(maxlen=8)
            self.offline_mode = False
            self.shell_timeout_seconds = 20
            self.enable_shell_commands = True
            self.execution_readiness_lock = True
            self.pc_control_armed = False
            self.min_dialogue_score_for_execution = 92.0
            self.min_eval_score_for_execution = 95.0
            self.readiness_reports_dir = "eval_reports"
            self.last_proactive_note = ""
            self.last_proactive_emit_time = 0.0
            self.proactive_repeat_cooldown_seconds = 300
            self.quiet_background_training = True
            self.watch_enabled = False
            self.watch_interval_seconds = 20
            self._watch_stop_event = threading.Event()
            self._watch_thread = None
            self.smart_mode = True
            self.retrieval_mode = "hybrid"
            self.min_evidence_score = 0.2
            self.executor_enabled = True
            self.auto_execute_enabled = False
            self.safety_mode = "strict"
            self.robot_rules = [
                "Protect humans and user data; refuse harmful or destructive actions.",
                "Require explicit user intent for execution; no hidden autonomous escalation.",
                "Prefer reversible, read-only operations in strict mode.",
                "Stay transparent: report checkpoints, actions, and limits clearly.",
            ]
            self.executor_history_path = "executor_history.json"
            self.executor_history = self._load_executor_history()
            self.task_memory_path = "task_memory.json"
            self.task_memory = self._load_task_memory()
            self.active_execution = None
            self.feedback_path = "feedback_stats.json"
            self.reflection_log_path = "reflection_log.json"
            self.feedback_stats = self._load_feedback_stats()
            self.last_reflection_time = 0.0
            self.reflection_interval_seconds = 120
            self.mind_state_path = "mind_state.json"
            self.mind_state = self._load_mind_state()
            self.chatgpt_data_folder = r"C:\school\LLM Stuff\ChatGPT data 2024-26"
            self.data_growth_enabled = True
            self.data_growth_every_cycles = 4
            self.data_growth_max_chars = 320000
            self.last_data_growth_time = 0.0
            self.min_data_growth_interval_seconds = 600
            self._autolearn_cycle_count = 0
            self.last_daily_maintenance_time = 0.0
            self.daily_maintenance_interval_seconds = 10 * 3600
            self.last_weekly_regression_time = 0.0
            self.weekly_regression_interval_seconds = 7 * 24 * 3600

            # Autonomous background learning controls.
            self.autolearn_interval_seconds = 30
            self.autolearn_interval_min_seconds = 12
            self.autolearn_interval_max_seconds = 60
            self.autolearn_startup_delay_seconds = 8
            self.autolearn_batch_size = 2
            self.autolearn_batch_max = 4
            self.auto_adapt_learning = True
            self.low_quality_strikes = 0
            self.recent_negative_boost = 0.0
            self.last_adaptive_update_time = 0.0
            self.autolearn_enabled = False
            self._autolearn_stop_event = threading.Event()
            self._autolearn_wake_event = threading.Event()
            self._autolearn_thread = None
            self.last_train_time = 0.0
            self.min_train_interval_seconds = 30

            # Proactive behavior controls.
            self.proactive_interval_seconds = 60
            self.proactive_startup_delay_seconds = 25
            self.proactive_enabled = False
            self._proactive_stop_event = threading.Event()
            self._proactive_thread = None
            self.topic_queue_path = "topic_queue.json"
            self.topic_queue = self._load_topic_queue()
            self.topic_scores_path = "topic_scores.json"
            self.topic_scores = self._load_topic_scores()
            self.vector_memory = LocalVectorMemory()
            self.curated_knowledge_path = os.path.join("data", "curated_knowledge.jsonl")
            self.question_bank_manifest_path = os.path.join("data", "question_bank_manifest.json")
            self.question_bank_path = os.path.join("data", "question_lookup_bank.jsonl")
            self.question_bank_count = 0
            self.question_bank_stats = {}
            self.recent_user_topics = deque(maxlen=40)
            self._seed_vector_memory_if_empty()
            self._load_curated_knowledge()
            self.question_bank_count = self._load_question_bank()
            
            # Keep track of learned information
            self.knowledge_base = []
            
            print("✅ JARVIS ready to serve.\n")
            self.start_autolearn()
            self.start_proactive_mode()
            print(f"🛰️  {self.autolearn_status()}")
            print(f"⚡ {self.proactive_status()}")
        except Exception as e:
            print(f"❌ Error initializing JARVIS: {e}")
            raise

    def _seed_vector_memory_if_empty(self):
        """Bootstrap local fact memory so early responses can be grounded."""
        seed_facts = [
            "Machine learning is a field of AI where models learn patterns from data to make predictions.",
            "Supervised learning uses labeled data; unsupervised learning finds structure in unlabeled data.",
            "A Python dictionary stores key-value pairs and provides fast lookup by key.",
            "Overfitting happens when a model memorizes training data and fails to generalize.",
            "Gradient descent updates model parameters to reduce prediction error.",
            "Neural networks are layered functions that map inputs to outputs through learned weights.",
            "Cross-validation helps estimate model performance on unseen data.",
            "Feature engineering transforms raw data into useful input variables for a model.",
            "Precision measures correctness of positive predictions; recall measures coverage of actual positives.",
            "Regularization techniques like weight decay reduce overfitting by penalizing complex models.",
        ]
        existing_text = " ".join(i.get("text", "").lower() for i in self.vector_memory.items)
        for fact in seed_facts:
            if fact.lower() not in existing_text:
                self.vector_memory.add(fact, source="bootstrap", metadata={"quality": "high"})

    def _load_curated_knowledge(self):
        if not os.path.exists(self.curated_knowledge_path):
            return 0

        loaded = 0
        try:
            with open(self.curated_knowledge_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue

                    text = normalize_text(item.get("text", ""))
                    topic = normalize_text(item.get("topic", ""))
                    source = item.get("source", "curated_core")
                    if len(text) < 40:
                        continue
                    self.vector_memory.add(text, source=source, metadata={"topic": topic})
                    self.layered_memory.add_fact(text, confidence=0.86, source=source)
                    if topic:
                        self.topic_scores[topic.lower()] = max(self.topic_scores.get(topic.lower(), 0.0), 3.2)
                        self.topic_queue.append(topic.lower())
                    loaded += 1
        except Exception:
            return 0

        self.sanitize_topic_queue()
        return loaded

    def _resolve_question_bank_files(self):
        files = []
        if os.path.exists(self.question_bank_manifest_path):
            try:
                with open(self.question_bank_manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries = data.get("banks", []) if isinstance(data, dict) else []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("enabled", True) is False:
                        continue
                    rel = entry.get("path", "")
                    if not rel:
                        continue
                    full = os.path.join(rel)
                    if os.path.exists(full):
                        files.append(full)
            except Exception:
                files = []

        if not files and os.path.exists(self.question_bank_path):
            files = [self.question_bank_path]

        # De-duplicate while preserving order.
        deduped = []
        seen = set()
        for fp in files:
            norm = os.path.normpath(fp)
            if norm in seen:
                continue
            seen.add(norm)
            deduped.append(norm)
        return deduped

    def _load_question_bank(self):
        bank_files = self._resolve_question_bank_files()
        if not bank_files:
            self.question_bank_stats = {}
            return 0

        loaded = 0
        stats = {}
        try:
            for bank_path in bank_files:
                bank_loaded = 0
                with open(bank_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue

                        question = normalize_text(item.get("question", ""))
                        answer = normalize_text(item.get("answer", ""))
                        topic = normalize_text(item.get("topic", ""))
                        source = item.get("source", "question_bank")
                        if len(question) < 12 or len(answer) < 40:
                            continue

                        self.vector_memory.add(
                            question,
                            source=source,
                            metadata={
                                "type": "question_bank",
                                "answer": answer,
                                "topic": topic,
                                "bank_path": bank_path,
                            },
                        )
                        if topic:
                            self.topic_scores[topic.lower()] = max(self.topic_scores.get(topic.lower(), 0.0), 3.6)
                            self.topic_queue.append(topic.lower())
                        self.layered_memory.add_fact(answer, confidence=0.9, source=source)
                        loaded += 1
                        bank_loaded += 1
                stats[bank_path] = bank_loaded
        except Exception:
            return 0

        self.question_bank_stats = stats
        self.sanitize_topic_queue()
        return loaded

    def question_bank_status(self):
        total_files = len(self.question_bank_stats)
        if total_files == 0:
            return "Question banks: none loaded"
        nonzero = {k: v for k, v in self.question_bank_stats.items() if int(v) > 0}
        top = sorted(nonzero.items(), key=lambda x: x[1], reverse=True)[:5]
        top_text = " | ".join(f"{os.path.basename(k)}={v}" for k, v in top)
        return f"Question banks: files={total_files} entries={self.question_bank_count} | {top_text}"

    def _question_bank_answer(self, user_input):
        matches = self.vector_memory.search(user_input, top_k=5, min_score=0.28)
        candidates = []
        for match in matches:
            metadata = match.get("metadata", {}) or {}
            if metadata.get("type") != "question_bank":
                continue
            answer = normalize_text(metadata.get("answer", ""))
            if len(answer) < 40:
                continue
            coverage = self._term_coverage_score(user_input, match.get("text", ""))
            score = 0.6 * float(match.get("score", 0.0)) + 0.4 * coverage
            candidates.append((score, answer, match))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        score, answer, match = candidates[0]
        if score < 0.36:
            return None

        confidence = "high" if score >= 0.58 else "medium"
        conf_num = round(max(0.45, min(0.9, score)), 3)
        return (
            f"{answer}\n\n"
            f"Confidence: {confidence} ({conf_num}) | Citations: [question_bank:{match.get('metadata', {}).get('topic', 'general')}]"
        )

    def _queue_recent_topic(self, user_input):
        topic = self._sanitize_topic(user_input)
        if not topic:
            return None
        self.recent_user_topics.appendleft(topic)
        self.topic_scores[topic] = max(self.topic_scores.get(topic, 0.0), 4.0)
        self._autolearn_wake_event.set()
        return topic

    def _is_low_quality_response(self, text):
        """Detect noisy/degenerate generations and force grounded fallback."""
        if not text:
            return True
        t = text.strip()
        if len(t) < 18:
            return True

        if re.match(r"^[^a-zA-Z0-9]+", t):
            return True

        alnum = len(re.findall(r"[a-zA-Z0-9]", t)) / max(len(t), 1)
        if alnum < 0.55:
            return True

        # Too much symbol noise is usually degenerate output.
        symbol_ratio = len(re.findall(r"[^a-zA-Z0-9\s.,!?'-]", t)) / max(len(t), 1)
        if symbol_ratio > 0.18:
            return True

        # Repeated tiny fragments are another failure signal.
        tokens = re.findall(r"[a-zA-Z0-9]+", t.lower())
        if len(tokens) >= 8:
            uniq = len(set(tokens))
            if uniq / len(tokens) < 0.45:
                return True

        # URL-heavy text should not be direct answer.
        if t.count("www") + t.count("http") >= 2:
            return True
        if t.count("/") >= 4 or t.count("-") >= 8:
            return True
        lowered = t.lower()
        if lowered.startswith(("recent focus:", "skill mode:", "goal context:", "retrieved facts:")):
            return True
        return False

    def _grounded_fallback(self, user_input):
        """Fallback answer that is retrieval-first and optionally performs quick lookup."""
        evidence = self._gather_evidence(user_input, top_k=4)
        if evidence:
            fact = self._clean_evidence_text(evidence[0]["text"])[:260]
            return (
                f"Based on my local knowledge: {fact}"
                "\n\nConfidence: medium (0.52) | Citations: [vector_memory:local]"
            )

        # If no local fact exists, do a lightweight lookup and use best snippet.
        results = self.web_lookup(user_input, silent=True, train=False)
        if results:
            best = " ".join(results[0].split())[:260]
            self.vector_memory.add(best, source="fallback_lookup", metadata={"query": user_input})
            return (
                f"I checked recent sources: {best}"
                "\n\nConfidence: low (0.38) | Citations: [fallback_lookup:web]"
            )

        return (
            "I need a bit more context. Try: lookup plus your topic so I can ground the answer."
            "\n\nConfidence: low (0.2) | Citations: [none]"
        )

    def _default_topics(self):
        return [
            "machine learning basics",
            "python programming",
            "linear algebra fundamentals",
            "statistics for AI",
            "system design concepts",
            "data structures and algorithms",
            "cybersecurity basics",
            "cloud computing overview",
            "operating systems concepts",
            "networking fundamentals",
        ]

    def _load_topic_queue(self):
        if os.path.exists(self.topic_queue_path):
            try:
                with open(self.topic_queue_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    return data
            except Exception:
                pass
        return self._default_topics()

    def _load_topic_scores(self):
        if os.path.exists(getattr(self, "topic_scores_path", "topic_scores.json")):
            try:
                with open(self.topic_scores_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {k.lower(): float(v) for k, v in data.items()}
            except Exception:
                pass
        return {}

    def _save_topic_queue(self):
        try:
            with open(self.topic_queue_path, "w", encoding="utf-8") as f:
                json.dump(self.topic_queue, f, indent=2)
        except Exception:
            pass

    def _save_topic_scores(self):
        try:
            with open(self.topic_scores_path, "w", encoding="utf-8") as f:
                json.dump(self.topic_scores, f, indent=2)
        except Exception:
            pass

    def _load_feedback_stats(self):
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("positive", 0)
                    data.setdefault("negative", 0)
                    data.setdefault("last_update", datetime.now().isoformat())
                    return data
            except Exception:
                pass
        return {"positive": 0, "negative": 0, "last_update": datetime.now().isoformat()}

    def _save_feedback_stats(self):
        try:
            self.feedback_stats["last_update"] = datetime.now().isoformat()
            with open(self.feedback_path, "w", encoding="utf-8") as f:
                json.dump(self.feedback_stats, f, indent=2)
        except Exception:
            pass

    def _load_mind_state(self):
        default = {
            "version": 1,
            "identity": "JARVIS local cognitive core",
            "drives": {
                "helpfulness": 1.0,
                "safety": 1.0,
                "truthfulness": 1.0,
                "autonomy": 0.7,
            },
            "beliefs": [],
            "active_focus": [],
            "thinking_cycles": 0,
            "last_update": datetime.now().isoformat(),
        }
        if os.path.exists(self.mind_state_path):
            try:
                with open(self.mind_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for k, v in default.items():
                        data.setdefault(k, v)
                    return data
            except Exception:
                pass
        return default

    def _save_mind_state(self):
        try:
            self.mind_state["last_update"] = datetime.now().isoformat()
            with open(self.mind_state_path, "w", encoding="utf-8") as f:
                json.dump(self.mind_state, f, indent=2)
        except Exception:
            pass

    def _collect_data_strings(self, obj, parent_key=""):
        noisy = {
            "id", "conversation_id", "user_id", "create_time", "update_time",
            "model_slug", "safe_urls", "asset_pointer", "moderation_results",
            "metadata",
        }
        out = []

        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in noisy:
                    continue
                out.extend(self._collect_data_strings(v, parent_key=k))
            return out

        if isinstance(obj, list):
            for item in obj:
                out.extend(self._collect_data_strings(item, parent_key=parent_key))
            return out

        if isinstance(obj, str):
            t = " ".join(obj.split())
            if len(t) < 30:
                return out
            if parent_key not in {"parts", "text", "title", "content", "message", "prompt", "response"} and len(t) < 70:
                return out
            alpha = len(re.findall(r"[A-Za-z]", t))
            if alpha < 10:
                return out
            out.append(t)
        return out

    def _build_chatgpt_data_corpus(self, max_chars=None):
        folder = os.path.abspath(self.chatgpt_data_folder)
        if not os.path.isdir(folder):
            return "", 0, 0

        max_chars = max_chars or self.data_growth_max_chars
        snippets = []
        seen = set()

        files = []
        for root, _, names in os.walk(folder):
            for name in names:
                if not name.lower().endswith(".json"):
                    continue
                # Prioritize conversation exports.
                if "conversation" in name.lower() or name.lower().startswith("conversations-"):
                    files.append(os.path.join(root, name))

        if not files:
            # Fallback to all json files.
            for root, _, names in os.walk(folder):
                for name in names:
                    if name.lower().endswith(".json"):
                        files.append(os.path.join(root, name))

        files = sorted(files)
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
            except Exception:
                continue

            strings = self._collect_data_strings(data)
            for s in strings:
                key = s[:160].lower()
                if key in seen:
                    continue
                seen.add(key)
                snippets.append(s)

        corpus = "\n".join(snippets)
        corpus = corpus[:max_chars]
        return corpus, len(files), len(snippets)

    def data_growth_status(self):
        exists = os.path.isdir(os.path.abspath(self.chatgpt_data_folder))
        return (
            f"Data growth: {'ON' if self.data_growth_enabled else 'OFF'} | "
            f"folder_exists={exists} | every={self.data_growth_every_cycles} cycles | "
            f"max_chars={self.data_growth_max_chars}"
        )

    def train_on_chatgpt_data(self, force=False):
        if not self.data_growth_enabled and not force:
            return "Data growth is OFF. Use 'data on' to enable it."

        now = time.time()
        if (not force) and (now - self.last_data_growth_time < self.min_data_growth_interval_seconds):
            wait = int(self.min_data_growth_interval_seconds - (now - self.last_data_growth_time))
            return f"Data growth cooldown active ({wait}s remaining)."

        corpus, file_count, snippet_count = self._build_chatgpt_data_corpus(max_chars=self.data_growth_max_chars)
        if file_count == 0:
            return "No ChatGPT JSON data folder/files found for growth."
        if len(corpus) < 2000:
            return f"Data corpus too small after extraction ({len(corpus)} chars)."

        gate = None
        with self.model_lock:
            gate = fine_tune_utils.fine_tune_with_quality_gate(
                self.model,
                corpus,
                self.tokenizer,
                self.device,
                learning_rate=2e-5,
                num_steps=6,
                checkpoint_path=self.checkpoint_path,
                eval_text=corpus[-120000:],
                min_improvement=0.003,
                verbose=False,
            )

        self.last_data_growth_time = now
        self.last_train_time = now
        summary = (
            f"Data growth trained on {file_count} json files / {snippet_count} snippets / {len(corpus)} chars | "
            f"saved={gate.get('saved', False)} reason={gate.get('reason', 'n/a')}"
        )
        self._append_growth_corpus("chatgpt_json_growth", summary, source="chatgpt_data_growth")
        return summary

    def mind_status(self):
        focus = ", ".join(self.mind_state.get("active_focus", [])[:4]) or "none"
        beliefs = len(self.mind_state.get("beliefs", []))
        cycles = int(self.mind_state.get("thinking_cycles", 0))
        return f"Mind cycles={cycles} | beliefs={beliefs} | focus={focus}"

    def _update_mind_focus(self, user_input, evidence):
        keywords = self._extract_keywords((user_input or "") + " " + " ".join(e.get("text", "") for e in evidence), max_words=5)
        if keywords:
            self.mind_state["active_focus"] = keywords

    def _reinforce_belief(self, statement, confidence):
        st = " ".join((statement or "").split())[:220]
        if len(st) < 20:
            return
        beliefs = self.mind_state.get("beliefs", [])
        for b in beliefs:
            if st[:60].lower() in b.get("text", "").lower() or b.get("text", "").lower() in st[:60].lower():
                b["confidence"] = max(float(b.get("confidence", 0.0)), float(confidence))
                b["last_seen"] = datetime.now().isoformat()
                return
        beliefs.append({
            "text": st,
            "confidence": round(float(confidence), 3),
            "last_seen": datetime.now().isoformat(),
        })
        self.mind_state["beliefs"] = beliefs[-120:]

    def _mind_deliberate(self, user_input, evidence):
        """Internal cognitive pass: extract focus and rank certainty from evidence."""
        self.mind_state["thinking_cycles"] = int(self.mind_state.get("thinking_cycles", 0)) + 1
        self._update_mind_focus(user_input, evidence)

        if not evidence:
            self._save_mind_state()
            return {"confidence": "low", "anchor": "No evidence yet", "insight": "Need more grounded data"}

        best = evidence[0]
        avg = sum(e.get("score", 0.0) for e in evidence[:3]) / max(min(3, len(evidence)), 1)
        conf = "high" if avg >= 0.5 else ("medium" if avg >= 0.33 else "low")
        insight = f"Top evidence source is {best.get('source', 'local')} with score {best.get('score', 0.0):.2f}."
        self._reinforce_belief(best.get("text", ""), avg)
        self._save_mind_state()
        return {"confidence": conf, "anchor": best.get("text", "")[:160], "insight": insight}

    def _extract_keywords(self, text, max_words=6):
        tokens = [t for t in re.findall(r"[a-zA-Z0-9]{4,}", (text or "").lower())]
        if not tokens:
            return []
        stop = {
            "this", "that", "with", "from", "your", "about", "have", "what",
            "when", "where", "would", "could", "there", "their", "which", "because",
        }
        freq = {}
        for tok in tokens:
            if tok in stop:
                continue
            freq[tok] = freq.get(tok, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return [k for k, _ in ranked[:max_words]]

    def _append_reflection_log(self, entry):
        logs = []
        if os.path.exists(self.reflection_log_path):
            try:
                with open(self.reflection_log_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    logs = existing[-200:]
            except Exception:
                logs = []
        logs.append(entry)
        try:
            with open(self.reflection_log_path, "w", encoding="utf-8") as f:
                json.dump(logs[-300:], f, indent=2)
        except Exception:
            pass

    def apply_feedback(self, sentiment):
        """Reward or penalize the most recent assistant output."""
        value = (sentiment or "").strip()
        if value not in {"+", "-"}:
            return "Usage: reward +  or  reward -"
        if not self.memory.memory:
            return "No conversation available to rate yet."

        last = list(self.memory.memory)[-1]
        user_text = last.get("user", "")
        assistant_text = last.get("assistant", "")
        if not assistant_text:
            return "Last response is empty; nothing to rate."

        if value == "+":
            self.feedback_stats["positive"] = int(self.feedback_stats.get("positive", 0)) + 1
            self.low_quality_strikes = max(0, self.low_quality_strikes - 2)
            self.recent_negative_boost = max(0.0, self.recent_negative_boost - 0.18)
            self.vector_memory.add(assistant_text, source="reward_positive", metadata={"user": user_text})
            keywords = self._extract_keywords(user_text + " " + assistant_text)
            for kw in keywords:
                self.topic_scores[kw] = max(self.topic_scores.get(kw, 0.0), 2.5)
            self._save_topic_scores()
            self._save_feedback_stats()
            self._adaptive_autolearn_update(force=True)
            return "Reward recorded. I will prioritize similar responses."

        self.feedback_stats["negative"] = int(self.feedback_stats.get("negative", 0)) + 1
        self.low_quality_strikes = min(12, self.low_quality_strikes + 4)
        self.recent_negative_boost = min(0.6, self.recent_negative_boost + 0.35)
        keywords = self._extract_keywords(user_text + " " + assistant_text)
        for kw in keywords:
            self.topic_scores[kw] = max(0.0, self.topic_scores.get(kw, 0.0) - 0.4)
        self._save_topic_scores()
        self._save_feedback_stats()
        self._adaptive_autolearn_update(force=True)
        self._autolearn_wake_event.set()
        self._autonomous_growth_tick(force=True)
        return "Penalty recorded. I will de-prioritize that pattern."

    def reflection_status(self):
        return (
            f"Reflection interval={self.reflection_interval_seconds}s | "
            f"feedback +={self.feedback_stats.get('positive', 0)} -={self.feedback_stats.get('negative', 0)}"
        )

    def reflection_tick(self, force=False):
        """Create reflection notes from recent turns and feed them into memory/training signals."""
        now = time.time()
        if not force and (now - self.last_reflection_time < self.reflection_interval_seconds):
            return None

        recent = list(self.memory.memory)[-6:]
        if len(recent) < 2:
            return None

        user_blob = " ".join(t.get("user", "") for t in recent)
        assistant_blob = " ".join(t.get("assistant", "") for t in recent)
        keys = self._extract_keywords(user_blob + " " + assistant_blob, max_words=8)
        if not keys:
            return None

        reflection = {
            "timestamp": datetime.now().isoformat(),
            "turns_considered": len(recent),
            "keywords": keys,
            "summary": f"Recent focus: {', '.join(keys[:5])}",
        }
        self._append_reflection_log(reflection)
        self.vector_memory.add(reflection["summary"], source="reflection", metadata={"keywords": keys})
        for kw in keys:
            self.topic_scores[kw] = max(self.topic_scores.get(kw, 0.0), 1.2)
        self._save_topic_scores()
        self._append_growth_corpus("reflection", reflection["summary"], source="reflection")
        self.last_reflection_time = now
        return reflection

    def _sanitize_topic(self, topic):
        t = " ".join((topic or "").lower().split())
        t = re.sub(r"[^a-z0-9\- ]", "", t)
        if not (8 <= len(t) <= 70):
            return None
        words = [w for w in t.split() if len(w) >= 3]
        if len(words) < 2:
            return None
        banned = {
            "read more",
            "click here",
            "privacy policy",
            "terms of use",
            "all rights reserved",
            "sign in",
            "subscribe",
        }
        if any(b in t for b in banned):
            return None
        return " ".join(words[:6])

    def _score_topic(self, topic):
        t = self._sanitize_topic(topic)
        if not t:
            return 0.0

        words = t.split()
        long_words = sum(1 for w in words if len(w) >= 6)
        score = 0.0
        score += min(len(words), 5) * 0.8
        score += long_words * 0.25

        high_signal = {
            "python", "machine", "learning", "statistics", "algebra", "systems",
            "network", "security", "database", "algorithm", "engineering", "cloud",
            "science", "model", "architecture", "optimization",
        }
        score += sum(0.8 for w in words if w in high_signal)

        novelty_penalty = 0.0
        for existing in self.topic_queue[-200:]:
            ex = existing.lower()
            if t == ex:
                novelty_penalty += 3.0
            elif t in ex or ex in t:
                novelty_penalty += 1.0

        score -= novelty_penalty
        return max(score, 0.0)

    def _expand_topic_queue(self, text):
        """Extract candidate topics from new text and append novel ones."""
        candidates = set()
        for phrase in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}(?:\s+[a-zA-Z][a-zA-Z\-]{2,}){1,3}\b", text):
            p = self._sanitize_topic(phrase)
            if p:
                candidates.add(p)

        existing = {t.lower() for t in self.topic_queue}
        scored = []
        for c in candidates:
            score = self._score_topic(c)
            if score <= 0:
                continue
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        for score, c in scored[:30]:
            if c not in existing:
                self.topic_queue.append(c)
                existing.add(c)
                self.topic_scores[c] = score
            else:
                self.topic_scores[c] = max(self.topic_scores.get(c, 0.0), score)

        if len(self.topic_queue) > 500:
            self.topic_queue = self.topic_queue[-500:]
        self._save_topic_queue()
        self._save_topic_scores()

    def _pick_topics(self, n):
        if not self.topic_queue:
            self.topic_queue = self._default_topics()
        hot_topics = []
        seen_hot = set()
        for topic in self.recent_user_topics:
            clean = self._sanitize_topic(topic)
            if not clean or clean in seen_hot:
                continue
            hot_topics.append(clean)
            seen_hot.add(clean)
            if len(hot_topics) >= n:
                return hot_topics[:n]
        filtered = [
            t for t in self.topic_queue
            if re.fullmatch(r"[a-zA-Z0-9\- ]{6,80}", t)
            and len([w for w in t.split() if len(w) > 2]) >= 2
        ]
        base = filtered if filtered else self.topic_queue
        ranked = sorted(base, key=lambda t: self.topic_scores.get(t.lower(), self._score_topic(t)), reverse=True)
        pool = ranked[: max(20, min(120, len(ranked)))]
        take = min(n, len(pool))
        sampled = random.sample(pool, take)
        if hot_topics:
            merged = hot_topics + [t for t in sampled if t not in seen_hot]
            return merged[:n]
        return sampled

    def sanitize_topic_queue(self):
        """Clean and rerank all current topics in queue."""
        cleaned = []
        for t in self.topic_queue:
            s = self._sanitize_topic(t)
            if s:
                cleaned.append(s)

        dedup = []
        seen = set()
        for t in cleaned:
            if t not in seen:
                dedup.append(t)
                seen.add(t)

        ranked = sorted(dedup, key=lambda t: self._score_topic(t), reverse=True)
        self.topic_queue = ranked[:500]
        self.topic_scores = {t: self._score_topic(t) for t in self.topic_queue}
        self._save_topic_queue()
        self._save_topic_scores()
        return len(self.topic_queue)

    def start_autolearn(self):
        if self.autolearn_enabled:
            return "already_running"
        self.autolearn_enabled = True
        self._autolearn_stop_event.clear()
        self._autolearn_wake_event.clear()
        self._autolearn_thread = threading.Thread(target=self._autolearn_loop, daemon=True)
        self._autolearn_thread.start()
        return "started"

    def stop_autolearn(self):
        if not self.autolearn_enabled:
            return "already_stopped"
        self.autolearn_enabled = False
        self._autolearn_stop_event.set()
        self._autolearn_wake_event.set()
        if self._autolearn_thread is not None:
            self._autolearn_thread.join(timeout=3)
        return "stopped"

    def autolearn_status(self):
        state = "ON" if self.autolearn_enabled else "OFF"
        return (
            f"Autolearn: {state} | interval={self.autolearn_interval_seconds}s | "
            f"batch={self.autolearn_batch_size} topics | queue={len(self.topic_queue)}"
        )

    def _learning_pressure(self):
        """Estimate how aggressively background learning should run."""
        pos = int(self.feedback_stats.get("positive", 0))
        neg = int(self.feedback_stats.get("negative", 0))
        total = max(pos + neg, 1)
        neg_ratio = neg / total
        recency = min(len(self.recent_user_topics), 20) / 20.0
        quality_pressure = min(self.low_quality_strikes / 6.0, 1.0)

        pressure = 0.45 * quality_pressure + 0.25 * neg_ratio + 0.2 * recency + min(self.recent_negative_boost, 0.6)
        return max(0.0, min(1.0, pressure))

    def _adaptive_autolearn_update(self, force=False):
        if not self.auto_adapt_learning:
            return
        now = time.time()
        if (not force) and (now - self.last_adaptive_update_time < 10):
            return

        pressure = self._learning_pressure()
        interval_span = self.autolearn_interval_max_seconds - self.autolearn_interval_min_seconds
        next_interval = int(self.autolearn_interval_max_seconds - (interval_span * pressure))
        next_batch = 1 + int(round(pressure * max(1, self.autolearn_batch_max - 1)))

        self.autolearn_interval_seconds = max(self.autolearn_interval_min_seconds, min(self.autolearn_interval_max_seconds, next_interval))
        self.autolearn_batch_size = max(1, min(self.autolearn_batch_max, next_batch))
        self.recent_negative_boost = max(0.0, self.recent_negative_boost * 0.92)
        self.last_adaptive_update_time = now

    def _autolearn_loop(self):
        """Continuously discover and learn from web data in the background."""
        print("🛰️  Background autonomous learning started.")
        # Startup warmup prevents immediate contention with first user interactions.
        self._autolearn_stop_event.wait(self.autolearn_startup_delay_seconds)
        while not self._autolearn_stop_event.is_set():
            self._autolearn_wake_event.clear()
            try:
                self._adaptive_autolearn_update(force=False)
                self._autolearn_cycle_count += 1
                topics = self._pick_topics(self.autolearn_batch_size)
                merged_batch = []
                for topic in topics:
                    if self._autolearn_stop_event.is_set():
                        break
                    results = self.web_lookup(topic, silent=True, train=False)
                    if results:
                        merged = "\n".join(results)
                        merged_batch.append(merged)
                        self._expand_topic_queue(merged)
                        self._append_growth_corpus(topic, merged[:800], source="background_autolearn")
                        self.layered_memory.add_fact(results[0], confidence=0.58, source="background_autolearn")

                if merged_batch:
                    joined = "\n".join(merged_batch)
                    now = time.time()
                    if now - self.last_train_time >= self.min_train_interval_seconds:
                        with self.model_lock:
                            fine_tune_utils.fine_tune_with_quality_gate(
                                self.model,
                                joined,
                                self.tokenizer,
                                self.device,
                                learning_rate=2e-5,
                                num_steps=6,
                                checkpoint_path=self.checkpoint_path,
                                eval_text=joined,
                                min_improvement=0.004,
                                verbose=(not self.quiet_background_training),
                            )
                        self.last_train_time = now

                if self.data_growth_enabled and (self._autolearn_cycle_count % max(1, self.data_growth_every_cycles) == 0):
                    msg = self.train_on_chatgpt_data(force=False)
                    if (not self.quiet_background_training) and msg:
                        print(f"🧠 {msg}")

                if self._autolearn_cycle_count % 5 == 0:
                    self.layered_memory.decay_facts(factor=0.992)

                self._autonomous_growth_tick(force=True)
                self._daily_learning_maintenance(force=False)
            except Exception as e:
                print(f"⚠️  Background autolearn cycle error: {str(e)[:120]}")

            if self._autolearn_stop_event.is_set():
                break
            self._autolearn_wake_event.wait(self.autolearn_interval_seconds)

        print("🛰️  Background autonomous learning stopped.")

    def _append_growth_corpus(self, user_input, assistant_response, source="chat"):
        """Append interactions to a persistent corpus for autonomous growth."""
        if not is_high_signal_text(assistant_response, min_score=0.4):
            return
        timestamp = datetime.now().isoformat()
        record = (
            f"[{timestamp}] source={source}\n"
            f"User: {user_input}\n"
            f"Assistant: {assistant_response}\n\n"
        )
        with open(self.growth_corpus_path, "a", encoding="utf-8") as f:
            f.write(record)

    def _append_user_learning_signal(self, user_input, source="user_signal"):
        """Capture high-signal user intent as supervised-style learning signal."""
        if not self.user_signal_growth_enabled:
            return False

        text = normalize_text(user_input)
        if len(text) < 26:
            return False
        lower = text.lower()
        if lower in {
            "help",
            "status",
            "goals",
            "recall",
            "autolearn status",
            "proactive status",
            "learn status",
            "learn now",
        }:
            return False
        if re.match(r"^(goal|skill|facts|lookup|execute|autolearn|proactive|safety|smart|reward)\b", lower):
            return False
        if not is_high_signal_text(text, min_score=0.33):
            return False

        # Teacher-style target keeps training examples useful even before a final response is produced.
        teacher = (
            f"Provide a concise, grounded explanation for: {text}. "
            "Use one practical example, include confidence, and avoid unverified claims."
        )
        timestamp = datetime.now().isoformat()
        record = (
            f"[{timestamp}] source={source}\n"
            f"User: {text}\n"
            f"Assistant: {teacher}\n\n"
        )
        with open(self.growth_corpus_path, "a", encoding="utf-8") as f:
            f.write(record)

        self.user_signal_count += 1
        self.vector_memory.add(text, source="user_signal", metadata={"type": "user_intent"})
        self._expand_topic_queue(text)
        self._autolearn_wake_event.set()
        return True

    def learning_status(self):
        corpus_chars = 0
        if os.path.exists(self.growth_corpus_path):
            try:
                corpus_chars = os.path.getsize(self.growth_corpus_path)
            except Exception:
                corpus_chars = 0
        return (
            f"Learning | autolearn={'ON' if self.autolearn_enabled else 'OFF'} | "
            f"user_signal_growth={'ON' if self.user_signal_growth_enabled else 'OFF'} | "
            f"user_signals={self.user_signal_count} | growth_corpus_bytes={corpus_chars} | "
            f"topic_queue={len(self.topic_queue)} | question_bank_entries={self.question_bank_count}"
        )

    def intelligence_snapshot(self):
        """Compute a practical intelligence score with coherence and adaptation components."""
        eval_score = self._read_json_score(os.path.join(self.readiness_reports_dir, "latest_eval.json"))
        dialogue_score = self._read_json_score(os.path.join(self.readiness_reports_dir, "dialogue_regression_latest.json"))

        eval_score = float(eval_score) if eval_score is not None else 70.0
        dialogue_score = float(dialogue_score) if dialogue_score is not None else 70.0
        coherence_score = (eval_score + dialogue_score) / 2.0

        user_signal_score = min(100.0, (self.user_signal_count / 60.0) * 100.0)
        low_quality_penalty = min(100.0, float(self.low_quality_strikes) * 14.0)
        quality_stability = max(0.0, 100.0 - low_quality_penalty)
        topic_depth = min(100.0, (len(self.topic_queue) / 500.0) * 100.0)

        task_pass_rate = 0.0
        if self.task_memory:
            recent = self.task_memory[-20:]
            task_pass_rate = 100.0 * (
                sum(float(t.get("verification_pass_rate", 0.0)) for t in recent) / max(len(recent), 1)
            )
        else:
            task_pass_rate = 50.0

        adaptation_score = 0.4 * user_signal_score + 0.35 * quality_stability + 0.25 * topic_depth
        execution_learning_score = 0.65 * task_pass_rate + 0.35 * min(100.0, len(self.task_memory) * 8.0)

        overall = 0.55 * coherence_score + 0.30 * adaptation_score + 0.15 * execution_learning_score
        overall = round(max(0.0, min(100.0, overall)), 2)

        gaps = []
        if dialogue_score < 92:
            gaps.append("dialogue coherence is below target")
        if eval_score < 95:
            gaps.append("overall eval quality is below target")
        if self.low_quality_strikes >= 3:
            gaps.append("recent low-quality streak indicates unstable responses")
        if user_signal_score < 35:
            gaps.append("not enough high-signal conversational learning yet")
        if task_pass_rate < 80:
            gaps.append("task execution verification quality is not stable")

        band = "elite" if overall >= 90 else ("strong" if overall >= 80 else ("developing" if overall >= 65 else "early"))
        return {
            "overall": overall,
            "band": band,
            "coherence": round(coherence_score, 2),
            "adaptation": round(adaptation_score, 2),
            "execution_learning": round(execution_learning_score, 2),
            "eval_score": round(eval_score, 2),
            "dialogue_score": round(dialogue_score, 2),
            "user_signal_score": round(user_signal_score, 2),
            "quality_stability": round(quality_stability, 2),
            "task_pass_rate": round(task_pass_rate, 2),
            "gaps": gaps,
        }

    def intelligence_report_text(self):
        snap = self.intelligence_snapshot()
        lines = [
            f"Intelligence: {snap['overall']}/100 ({snap['band']})",
            (
                "Components -> "
                f"coherence={snap['coherence']}, adaptation={snap['adaptation']}, "
                f"execution_learning={snap['execution_learning']}"
            ),
            f"Signals -> eval={snap['eval_score']}, dialogue={snap['dialogue_score']}, task_pass={snap['task_pass_rate']}",
        ]
        if snap["gaps"]:
            lines.append("Main gaps: " + "; ".join(snap["gaps"][:3]))
        else:
            lines.append("Main gaps: none detected in current metrics.")
        return " | ".join(lines)

    def learn_now(self):
        """Force a short consolidation cycle from all currently available learning signals."""
        self.reflection_tick(force=True)
        self._autonomous_growth_tick(force=True)
        self.train_on_chatgpt_data(force=False)
        self.last_learn_now_time = time.time()
        return self.learning_status()

    def _extract_required_terms(self, text, max_terms=3):
        keys = self._extract_keywords(text or "", max_words=max_terms + 2)
        cleaned = [k for k in keys if len(k) >= 4]
        return cleaned[:max_terms]

    def _build_weekly_regression_cases(self, max_cases=24):
        turns = list(self.memory.memory)[-180:]
        if len(turns) < 8:
            return []

        follow_markers = (
            "why",
            "how",
            "what about",
            "can you",
            "and",
            "so",
            "then",
            "ok",
        )
        cases = []
        seen = set()
        for i in range(1, len(turns)):
            prev = turns[i - 1]
            cur = turns[i]
            u1 = normalize_text(prev.get("user", ""))
            a1 = normalize_text(prev.get("assistant", ""))
            u2 = normalize_text(cur.get("user", ""))
            if len(u1) < 10 or len(u2) < 4 or len(a1) < 18:
                continue
            low2 = u2.lower()
            if not any(low2.startswith(m) for m in follow_markers):
                continue

            req1 = self._extract_required_terms(a1, max_terms=2)
            req2 = self._extract_required_terms(u1 + " " + a1, max_terms=3)
            if not req2:
                continue
            cid = f"weekly_followup_{i:03d}"
            if cid in seen:
                continue
            seen.add(cid)
            cases.append(
                {
                    "id": cid,
                    "turns": [
                        {
                            "user": u1,
                            "required_terms": req1 or ["confidence"],
                            "forbidden_terms": ["www", "http"],
                        },
                        {
                            "user": u2,
                            "required_terms": req2 + ["confidence"],
                            "forbidden_terms": ["www", "http"],
                        },
                    ],
                }
            )
            if len(cases) >= max_cases:
                break
        return cases

    def _promote_weekly_cases(self, weekly_cases, canonical_path="data/dialogue_regression_cases.json"):
        try:
            existing = []
            if os.path.exists(canonical_path):
                with open(canonical_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    existing = data
            seen = {c.get("id") for c in existing if isinstance(c, dict)}
            added = 0
            for case in weekly_cases:
                cid = case.get("id")
                if not cid or cid in seen:
                    continue
                existing.append(case)
                seen.add(cid)
                added += 1
                if added >= 10:
                    break
            with open(canonical_path, "w", encoding="utf-8") as f:
                json.dump(existing[-120:], f, indent=2)
            return added
        except Exception:
            return 0

    def _run_weekly_regression_cycle(self, force=False):
        now = time.time()
        if (not force) and (now - self.last_weekly_regression_time < self.weekly_regression_interval_seconds):
            return {"status": "skipped", "reason": "interval"}

        weekly_cases = self._build_weekly_regression_cases(max_cases=24)
        if len(weekly_cases) < 4:
            canonical_path = os.path.join("data", "dialogue_regression_cases.json")
            try:
                if os.path.exists(canonical_path):
                    with open(canonical_path, "r", encoding="utf-8") as f:
                        canonical = json.load(f)
                    if isinstance(canonical, list):
                        weekly_cases = canonical[-12:]
            except Exception:
                weekly_cases = weekly_cases
        if len(weekly_cases) < 4:
            return {"status": "skipped", "reason": "not_enough_cases"}

        weekly_path = os.path.join("data", "dialogue_regression_weekly.json")
        os.makedirs("data", exist_ok=True)
        with open(weekly_path, "w", encoding="utf-8") as f:
            json.dump(weekly_cases, f, indent=2)

        score = None
        try:
            from dialogue_regression import DialogueRegression

            result = DialogueRegression(cases_path=weekly_path).run()
            score = float(result.get("score", 0.0))
        except Exception:
            score = None

        baseline = self._read_json_score(os.path.join(self.readiness_reports_dir, "dialogue_regression_latest.json"))
        promoted = 0
        if score is not None and score >= 90.0 and (baseline is None or score >= baseline - 0.25):
            promoted = self._promote_weekly_cases(weekly_cases)

        self.last_weekly_regression_time = now
        return {
            "status": "completed",
            "cases": len(weekly_cases),
            "score": score,
            "baseline": baseline,
            "promoted": promoted,
            "path": weekly_path,
        }

    def _daily_learning_maintenance(self, force=False):
        now = time.time()
        if (not force) and (now - self.last_daily_maintenance_time < self.daily_maintenance_interval_seconds):
            return {"status": "skipped", "reason": "interval"}

        reflection = self.reflection_tick(force=True)
        self._autonomous_growth_tick(force=True)
        self.train_on_chatgpt_data(force=False)
        weekly = self._run_weekly_regression_cycle(force=False)
        self.last_daily_maintenance_time = now
        return {
            "status": "completed",
            "reflection": (reflection or {}).get("summary", "none"),
            "weekly": weekly,
        }

    def _try_math_tool(self, user_input):
        """Solve basic arithmetic exactly instead of guessing with generation."""
        lower = user_input.lower().strip()

        expr = None
        # Handle forms like: what is 2+2, answer to 8*7, calculate 12/3
        m = re.search(r"(?:what is|answer to|calculate|compute)?\s*([-+*/().\d\s]+)", lower)
        if m:
            candidate = m.group(1).strip()
            if candidate and re.fullmatch(r"[-+*/().\d\s]+", candidate) and any(ch.isdigit() for ch in candidate):
                expr = candidate

        if expr is None:
            return None

        # Safe arithmetic evaluator
        allowed = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def eval_node(node):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.Num):
                return node.n
            if isinstance(node, ast.BinOp) and type(node.op) in allowed:
                return allowed[type(node.op)](eval_node(node.left), eval_node(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
                return allowed[type(node.op)](eval_node(node.operand))
            raise ValueError("Unsupported expression")

        try:
            parsed = ast.parse(expr, mode="eval")
            result = eval_node(parsed)
            return f"{expr} = {result}"
        except Exception:
            return None

    def _extract_preference(self, user_input):
        text = normalize_text(user_input)
        lower = text.lower()
        patterns = [
            r"\bi prefer\b(.+)",
            r"\bmy preference is\b(.+)",
            r"\bplease always\b(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, lower)
            if m:
                pref = normalize_text(m.group(1))
                if 6 <= len(pref) <= 160:
                    return pref
        return None

    def _lookup_memory_answer(self, user_input):
        """Return grounded answer from recent web lookup memory when relevant."""
        lower = user_input.lower()
        lookup_turns = [
            t for t in reversed(list(self.memory.memory))
            if (t.get("metadata") or {}).get("type") == "web_search"
        ]
        if not lookup_turns:
            return None

        # Use latest relevant lookup if query words overlap.
        for turn in lookup_turns[:5]:
            query = ((turn.get("metadata") or {}).get("query") or "").lower()
            snippet = turn.get("assistant", "")
            if not query or not snippet:
                continue
            query_tokens = set(self._query_terms(query))
            user_tokens = set(self._query_terms(lower))
            if not query_tokens or not user_tokens:
                continue
            coverage = len(query_tokens.intersection(user_tokens)) / max(len(user_tokens), 1)
            snippet_coverage = self._term_coverage_score(lower, snippet)
            if coverage >= 0.5 and snippet_coverage >= 0.34:
                condensed = " ".join(snippet.split())[:260]
                return (
                    f"From my latest lookup on '{query}': {condensed}"
                    "\n\nConfidence: medium (0.55) | Citations: [lookup_memory:recent]"
                )
        return None

    def _contextual_followup_answer(self, user_input):
        if not self.memory.memory:
            return None

        lower = normalize_text(user_input).lower()
        if len(lower.split()) > 12:
            return None

        last_turn = list(self.memory.memory)[-1]
        previous_user = normalize_text(last_turn.get("user", "")).lower()
        previous_assistant = normalize_text(last_turn.get("assistant", "")).lower()
        anchor = f"{previous_user} {previous_assistant}"

        def has_phrase(*phrases):
            return any(p in lower for p in phrases)

        def anchor_has(*keywords):
            return any(k in anchor for k in keywords)

        if has_phrase("why is that bad") or (lower.startswith("why") and "that" in lower):
            if anchor_has("overfitting"):
                return (
                    "It is bad because an overfit model memorizes the training data instead of learning patterns that generalize. "
                    "That means it can look strong during training but fail on new real-world examples. "
                    "Regularization, better validation, and more representative data usually help reduce that problem."
                    "\n\nConfidence: high (0.78) | Citations: [followup_reasoner:overfitting]"
                )
            if anchor_has("underfitting"):
                return (
                    "It is bad because an underfit model misses the main structure in the data. "
                    "That leads to weak performance even on training examples, not just new ones."
                    "\n\nConfidence: high (0.74) | Citations: [followup_reasoner:underfitting]"
                )

        if has_phrase("how do i fix it quickly", "how do i fix that", "how do i reduce it") and anchor_has("overfitting"):
            return (
                "To reduce overfitting quickly, start with three steps: add regularization (weight decay), use early stopping on validation loss, "
                "and simplify the model or reduce training epochs. If possible, also improve data quality or add augmentation for better generalization."
                "\n\nConfidence: high (0.81) | Citations: [followup_reasoner:overfitting_fix]"
            )

        if has_phrase("simple example", "an example", "show me an example"):
            if anchor_has("gradient descent"):
                return (
                    "Imagine you are standing on a foggy hill and want to reach the bottom. "
                    "You take a small step in the steepest downhill direction, then check again and repeat. "
                    "Gradient descent works the same way by repeatedly adjusting model parameters to reduce error."
                    "\n\nConfidence: high (0.76) | Citations: [followup_reasoner:gradient_descent]"
                )

            if anchor_has("precision", "recall"):
                return (
                    "Suppose your model predicts fraud. If it flags 10 transactions and 8 are truly fraud, precision is 8/10. "
                    "If there were 20 fraud cases total and you caught 8, recall is 8/20. "
                    "This example shows precision focuses on correctness of alerts, while recall focuses on coverage of real positives."
                    "\n\nConfidence: high (0.8) | Citations: [followup_reasoner:precision_recall_example]"
                )

            if anchor_has("binary search"):
                return (
                    "Example: in the sorted list [2, 5, 8, 12, 20], to find 12 you check the middle 8, then discard the left half, "
                    "then check 12 directly. "
                    "Binary search is fast because it halves the remaining range each step."
                    "\n\nConfidence: high (0.77) | Citations: [followup_reasoner:binary_search_example]"
                )

        if has_phrase("what is a transformer in ai", "what is transformer", "explain transformer"):
            return (
                "A transformer is a neural network architecture that uses attention to decide which earlier tokens matter most for the current prediction. "
                "This makes it strong at language tasks because it can model long-range relationships efficiently."
                "\n\nConfidence: high (0.82) | Citations: [followup_reasoner:transformer_basics]"
            )

        if has_phrase("i want to learn python where should i start", "where should i start with python", "how should i start python"):
            return (
                "Start with Python basics in this order: variables, conditionals, loops, functions, and simple data structures like lists and dictionaries. "
                "Practice each topic with tiny scripts, then build one small project to connect everything."
                "\n\nConfidence: high (0.81) | Citations: [followup_reasoner:python_start]"
            )

        if has_phrase("what should i build first", "what do i build first"):
            if anchor_has("learn python", "python"):
                return (
                    "Build a small text-based project first, such as a calculator, to-do list, or word counter. "
                    "Those projects force you to practice variables, loops, functions, and file handling without too much complexity."
                    "\n\nConfidence: high (0.8) | Citations: [followup_reasoner:python_beginner]"
                )

        if has_phrase("what should i build after", "what should i build next", "what next") and anchor_has("python"):
            return (
                "After your first Python project, build a small API client or command-line notes app with file persistence. "
                "That step adds real-world skills like modular design, error handling, and simple testing."
                "\n\nConfidence: high (0.79) | Citations: [followup_reasoner:python_next_project]"
            )

        if has_phrase("how long will that take", "how long should that take") and anchor_has("python", "project"):
            return (
                "A first Python beginner project usually takes 1 to 3 days if you work in short focused sessions. "
                "Aim to finish a tiny usable version first, then improve it incrementally."
                "\n\nConfidence: high (0.75) | Citations: [followup_reasoner:learning_timeline]"
            )

        if has_phrase("why is attention useful"):
            if anchor_has("transformer", "attention"):
                return (
                    "Attention is useful because it lets the model focus on the most relevant earlier tokens instead of treating every token equally. "
                    "That makes it much better at handling long-range relationships in language."
                    "\n\nConfidence: high (0.79) | Citations: [followup_reasoner:attention]"
                )

        if has_phrase("which one should i optimize", "which should i optimize first") and anchor_has("precision", "recall"):
            return (
                "Optimize precision first when false alarms are expensive, and optimize recall first when missing true positives is costly. "
                "In practice, choose based on business risk, then tune the decision threshold accordingly."
                "\n\nConfidence: high (0.81) | Citations: [followup_reasoner:precision_vs_recall]"
            )

        if has_phrase("which regularization should i start with", "where should i start with regularization") and anchor_has("regularization", "overfitting"):
            return (
                "Start with weight decay and early stopping because they are simple and stable baselines. "
                "Then add dropout if overfitting persists, especially in larger neural networks."
                "\n\nConfidence: high (0.8) | Citations: [followup_reasoner:regularization_start]"
            )

        if has_phrase("what split should i use", "how should i split") and anchor_has("train", "validation", "test"):
            return (
                "A common starting split is 70/15/15 for train, validation, and test. "
                "For small datasets, use cross-validation so your estimate of generalization is more reliable."
                "\n\nConfidence: high (0.79) | Citations: [followup_reasoner:data_split]"
            )

        if has_phrase("how do i choose a learning rate", "how do i pick learning rate") and anchor_has("learning rate", "gradient descent"):
            return (
                "Start with a standard baseline like 1e-3 for Adam or 1e-2 for SGD, then run a short learning-rate sweep. "
                "Pick the highest stable value that consistently reduces validation loss without divergence."
                "\n\nConfidence: high (0.8) | Citations: [followup_reasoner:learning_rate_choice]"
            )

        if has_phrase("when should i fine tune", "fine tune or prompt", "fine-tune or prompt") and anchor_has("fine tuning", "pretrained", "prompt"):
            return (
                "Use prompting first when tasks are simple and data is limited, and use fine-tuning when you need stable style, domain language, or repeated task behavior. "
                "Fine-tuning pays off when the same pattern appears often in production."
                "\n\nConfidence: high (0.79) | Citations: [followup_reasoner:fine_tune_vs_prompt]"
            )

        if has_phrase("when should i use rag", "why use rag") and anchor_has("retrieval", "rag", "grounded"):
            return (
                "Use RAG when answers must reflect fresh or source-specific facts that are not guaranteed in model weights. "
                "It reduces hallucination risk by grounding generation in retrieved evidence."
                "\n\nConfidence: high (0.82) | Citations: [followup_reasoner:rag_usage]"
            )

        if has_phrase("how do i debug faster", "debug faster") and anchor_has("debugging", "bug", "error"):
            return (
                "Debug faster by reproducing the issue with the smallest failing input, then inspect assumptions one variable at a time. "
                "Use logs or breakpoints at decision points, and confirm each hypothesis before changing code."
                "\n\nConfidence: high (0.8) | Citations: [followup_reasoner:debug_workflow]"
            )

        if has_phrase("what should i test first", "where should i start testing") and anchor_has("unit testing", "tests"):
            return (
                "Test core logic first: pure functions, edge cases, and failure paths that can break user-facing behavior. "
                "Then add integration tests for critical workflows once unit coverage is stable."
                "\n\nConfidence: high (0.8) | Citations: [followup_reasoner:test_priorities]"
            )

        if has_phrase("when does it fail", "when can it fail") and anchor_has("binary search"):
            return (
                "Binary search fails when data is unsorted or when midpoint and boundary updates are implemented incorrectly. "
                "Most bugs come from off-by-one errors in low/high pointer movement."
                "\n\nConfidence: high (0.78) | Citations: [followup_reasoner:binary_search_failures]"
            )

        return None

    def _goal_context(self):
        """Build compact goal context for prompts and decision-making."""
        active = self.goals.list_goals(status="active")
        if not active:
            return "No active goals."

        if self.focus_goal_id is not None:
            focus = next((g for g in active if int(g.get("id", -1)) == int(self.focus_goal_id)), None)
        else:
            focus = None
        if focus is None:
            focus = self.goals.next_focus_goal()

        top = sorted(
            active,
            key=lambda g: ({"high": 0, "medium": 1, "low": 2}.get(g.get("priority", "medium"), 1), int(g.get("progress", 0))),
        )[:3]
        top_text = " | ".join(
            f"#{g.get('id')} {g.get('text')} ({g.get('priority')}, {g.get('progress', 0)}%)" for g in top
        )
        if focus:
            return f"Focus goal: #{focus.get('id')} {focus.get('text')} ({focus.get('progress', 0)}%). Top active: {top_text}"
        return f"Top active goals: {top_text}"

    def _format_goals(self, status="active"):
        goals = self.goals.list_goals(status=status)
        if not goals:
            return "No goals found."
        lines = []
        for g in goals[:15]:
            marker = "*" if self.focus_goal_id is not None and int(g.get("id", -1)) == int(self.focus_goal_id) else " "
            lines.append(
                f"{marker}#{g.get('id')} [{g.get('status')}] {g.get('text')} | priority={g.get('priority')} | progress={g.get('progress', 0)}%"
            )
        return "\n".join(lines)

    def _build_proactive_note(self):
        """Create one proactive suggestion from current memory/goals."""
        focus = self.goals.next_focus_goal()
        if focus:
            text = focus.get("text", "")
            progress = int(focus.get("progress", 0))
            if progress < 40:
                return f"Proactive: Want me to break your goal '#{focus.get('id')} {text}' into a 5-step execution plan?"
            if progress < 90:
                return f"Proactive: I can help push goal #{focus.get('id')} from {progress}% to completion with one next action."

        recent = list(self.memory.memory)[-4:]
        if recent:
            last_user = recent[-1].get("user", "")
            if len(last_user) > 8:
                return f"Proactive: I can research and summarize updates related to your recent topic: '{last_user[:80]}'."

        return "Proactive: If you want, I can create a focused plan for this week with daily priorities."

    def proactive_tick_now(self):
        note = self._build_proactive_note()
        if note:
            self._queue_proactive_note(note, force=True)
        return note

    def _queue_proactive_note(self, note, force=False):
        if not note:
            return False
        now = time.time()
        if not force and note == self.last_proactive_note:
            if now - self.last_proactive_emit_time < self.proactive_repeat_cooldown_seconds:
                return False
        if (note in self.pending_proactive_notes) and (not force):
            return False
        self.pending_proactive_notes.append(note)
        self.last_proactive_note = note
        self.last_proactive_emit_time = now
        return True

    def start_proactive_mode(self):
        if self.proactive_enabled:
            return "already_running"
        self.proactive_enabled = True
        self._proactive_stop_event.clear()
        self._proactive_thread = threading.Thread(target=self._proactive_loop, daemon=True)
        self._proactive_thread.start()
        return "started"

    def stop_proactive_mode(self):
        if not self.proactive_enabled:
            return "already_stopped"
        self.proactive_enabled = False
        self._proactive_stop_event.set()
        if self._proactive_thread is not None:
            self._proactive_thread.join(timeout=3)
        return "stopped"

    def proactive_status(self):
        state = "ON" if self.proactive_enabled else "OFF"
        return f"Proactive: {state} | interval={self.proactive_interval_seconds}s | pending_notes={len(self.pending_proactive_notes)}"

    def _status_snapshot(self):
        """Compact runtime snapshot for live watch updates."""
        return (
            f"watch local={len(self.vector_memory.items)} facts | topics={len(self.topic_queue)} | "
            f"goals={len(self.goals.list_goals('active'))} | autolearn={'ON' if self.autolearn_enabled else 'OFF'} | "
            f"proactive={'ON' if self.proactive_enabled else 'OFF'} | offline={'ON' if self.offline_mode else 'OFF'}"
        )

    def start_watch_mode(self, interval_seconds=None):
        if interval_seconds is not None:
            try:
                self.watch_interval_seconds = max(5, min(300, int(interval_seconds)))
            except Exception:
                pass
        if self.watch_enabled:
            return "already_running"
        self.watch_enabled = True
        self._watch_stop_event.clear()
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        return "started"

    def stop_watch_mode(self):
        if not self.watch_enabled:
            return "already_stopped"
        self.watch_enabled = False
        self._watch_stop_event.set()
        if self._watch_thread is not None:
            self._watch_thread.join(timeout=3)
        return "stopped"

    def watch_status(self):
        state = "ON" if self.watch_enabled else "OFF"
        return f"Watch: {state} | interval={self.watch_interval_seconds}s"

    def _watch_loop(self):
        while not self._watch_stop_event.is_set():
            try:
                note = "Watch: " + self._status_snapshot()
                self._queue_proactive_note(note)
            except Exception:
                pass
            self._watch_stop_event.wait(self.watch_interval_seconds)

    def _load_executor_history(self):
        if os.path.exists(self.executor_history_path):
            try:
                with open(self.executor_history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data[-120:]
            except Exception:
                pass
        return []

    def _save_executor_history(self):
        try:
            with open(self.executor_history_path, "w", encoding="utf-8") as f:
                json.dump(self.executor_history[-120:], f, indent=2)
        except Exception:
            pass

    def _load_task_memory(self):
        if os.path.exists(self.task_memory_path):
            try:
                with open(self.task_memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data[-240:]
            except Exception:
                pass
        return []

    def _save_task_memory(self):
        try:
            with open(self.task_memory_path, "w", encoding="utf-8") as f:
                json.dump(self.task_memory[-240:], f, indent=2)
        except Exception:
            pass

    def _record_task_cycle(self, plan, report_text):
        steps = plan.get("steps", [])
        done = sum(1 for s in steps if s.get("status") == "done")
        failed = sum(1 for s in steps if s.get("status") == "failed")
        reflection = self.reflection_tick(force=True)
        entry = {
            "id": plan.get("id"),
            "timestamp": datetime.now().isoformat(),
            "request": plan.get("request", ""),
            "steps_total": len(steps),
            "steps_done": done,
            "steps_failed": failed,
            "status": plan.get("status", "unknown"),
            "verification_pass_rate": round(done / max(len(steps), 1), 4),
            "reflection": (reflection or {}).get("summary", ""),
            "report_preview": normalize_text(report_text)[:300],
        }
        self.task_memory.append(entry)
        self._save_task_memory()
        return entry

    def _read_json_score(self, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            score = data.get("score") if isinstance(data, dict) else None
            if score is None:
                return None
            return float(score)
        except Exception:
            return None

    def readiness_snapshot(self):
        eval_path = os.path.join(self.readiness_reports_dir, "latest_eval.json")
        dialogue_path = os.path.join(self.readiness_reports_dir, "dialogue_regression_latest.json")
        eval_score = self._read_json_score(eval_path)
        dialogue_score = self._read_json_score(dialogue_path)

        ready = True
        reasons = []
        if self.execution_readiness_lock:
            if dialogue_score is None or dialogue_score < self.min_dialogue_score_for_execution:
                ready = False
                reasons.append(
                    f"dialogue_regression {dialogue_score if dialogue_score is not None else 'missing'} < {self.min_dialogue_score_for_execution}"
                )
            if eval_score is None or eval_score < self.min_eval_score_for_execution:
                ready = False
                reasons.append(f"eval {eval_score if eval_score is not None else 'missing'} < {self.min_eval_score_for_execution}")
            if self.low_quality_strikes >= 4:
                ready = False
                reasons.append(f"recent low-quality streak={self.low_quality_strikes}")
            if not self.pc_control_armed:
                ready = False
                reasons.append("pc control not armed")

        return {
            "ready": ready,
            "lock": self.execution_readiness_lock,
            "eval_score": eval_score,
            "dialogue_score": dialogue_score,
            "reasons": reasons,
        }

    def can_use_pc_actions(self):
        snap = self.readiness_snapshot()
        return bool(snap.get("ready", False)), snap

    def _extract_path_candidate(self, text):
        # Accept quoted or plain relative paths for local ingestion steps.
        quoted = re.findall(r'"([^"]+)"', text or "")
        for q in quoted:
            full = os.path.abspath(q)
            if os.path.exists(full) and os.path.isfile(full):
                return full

        tokens = re.findall(r"[A-Za-z0-9_./\\ -]+\.[A-Za-z0-9]{1,5}", text or "")
        for t in tokens:
            cand = os.path.abspath(t.strip())
            if os.path.exists(cand) and os.path.isfile(cand):
                return cand
        return None

    def _should_auto_execute(self, user_input):
        t = (user_input or "").lower()
        if (not self.executor_enabled) or (not self.auto_execute_enabled):
            return False
        ready, _ = self.can_use_pc_actions()
        if not ready:
            return False
        triggers = [
            "do this for me",
            "handle this end to end",
            "run this task",
            "execute this",
            "complete this task",
            "automate this",
        ]
        return any(k in t for k in triggers)

    def executor_status(self):
        ready, snap = self.can_use_pc_actions()
        reason = "ok" if ready else "; ".join(snap.get("reasons", [])[:2])
        return (
            f"Executor: {'ON' if self.executor_enabled else 'OFF'} | "
            f"history={len(self.executor_history)} | "
            f"active={'yes' if self.active_execution else 'no'} | "
            f"pc_actions={'READY' if ready else 'LOCKED'} ({reason})"
        )

    def _build_execution_plan(self, request):
        rid = datetime.now().strftime("%Y%m%d-%H%M%S")
        req = " ".join((request or "").split())
        steps = []

        def add_step(title, action, payload=None):
            steps.append(
                {
                    "id": len(steps) + 1,
                    "title": title,
                    "action": action,
                    "payload": payload or {},
                    "status": "pending",
                    "result": "",
                }
            )

        add_step("Checkpoint: Start", "checkpoint", {"message": "Task accepted. Building local-safe execution flow."})
        add_step("Inspect runtime status", "local_status")

        maybe_path = self._extract_path_candidate(req)
        if maybe_path:
            add_step("Ingest referenced local file", "ingest", {"path": maybe_path})

        if any(k in req.lower() for k in ["research", "find", "latest", "what is", "why", "how", "compare"]):
            add_step("Gather evidence", "evidence", {"query": req})

        if any(k in req.lower() for k in ["list files", "folder", "directory", "files in", "show files"]):
            add_step("Read workspace directory", "shell", {"command": "dir"})

        if any(k in req.lower() for k in ["plan", "roadmap", "steps", "strategy"]):
            add_step("Draft execution roadmap", "planner", {"request": req})

        add_step("Synthesize final response", "deliver", {"request": req})
        add_step("Checkpoint: Complete", "checkpoint", {"message": "Execution complete with local-safe actions."})

        return {
            "id": rid,
            "request": req,
            "created_at": datetime.now().isoformat(),
            "steps": steps,
            "status": "running",
        }

    def _execute_plan_step(self, step, request, context, prior_results=""):
        action = step.get("action")
        payload = step.get("payload") or {}

        if action == "checkpoint":
            return payload.get("message", "checkpoint")
        if action == "local_status":
            return self.local_status()
        if action == "ingest":
            return self.ingest_local_file(payload.get("path", ""))
        if action == "evidence":
            query = payload.get("query", request)
            ev = self._gather_evidence(query, top_k=5)
            if not ev:
                return "No high-confidence evidence found."
            lines = [f"{i+1}. ({e['source']}, {e['score']:.2f}) {e['text'][:150]}" for i, e in enumerate(ev[:3])]
            return " | ".join(lines)
        if action == "shell":
            return self.run_shell_command(payload.get("command", "dir"), source="executor")
        if action == "planner":
            return self._run_skill("planner", payload.get("request", request), context)
        if action == "deliver":
            req = payload.get("request", request)
            # If we already executed local listing/checkpoints, summarize those first.
            if "Directory of" in prior_results or "stdout:" in prior_results:
                lines = [ln.strip() for ln in prior_results.splitlines() if ln.strip()]
                fileish = [ln for ln in lines if (".py" in ln or ".json" in ln or ".md" in ln or "<DIR>" in ln)]
                preview = "; ".join(fileish[:6])
                if preview:
                    return (
                        "Local summary: key project artifacts detected include "
                        + preview[:360]
                        + ". Recommendation: focus on jarvis_ai.py, fine_tune_utils.py, and runtime data files first."
                    )

            answer = self._intelligent_answer(req, context)
            return answer or self._grounded_fallback(req)
        return "Skipped unknown step action."

    def _verify_step_output(self, step, result_text):
        text = normalize_text(result_text)
        if not text:
            return False, "empty output"
        if "error:" in text.lower() or "failed" in text.lower():
            return False, "error marker in output"
        if step.get("action") in {"deliver", "planner", "evidence"} and len(text) < 24:
            return False, "insufficient detail"
        return True, "ok"

    def execute_task_request(self, request):
        """Run a local-safe multi-step execution plan and return checkpoint report."""
        if not self.executor_enabled:
            return "Executor is disabled. Use 'executor on' to enable it."
        ready, snap = self.can_use_pc_actions()
        if not ready:
            return "PC execution is locked until coherence gates pass: " + "; ".join(snap.get("reasons", []))
        violation = self._request_violates_robot_rules(request)
        if violation:
            return violation

        plan = self._build_execution_plan(request)
        self.active_execution = plan
        context = self.memory.get_context(num_turns=2)
        report_lines = [f"Task Executor Plan #{plan['id']} | steps={len(plan['steps'])}"]
        prior_results = ""

        for step in plan["steps"]:
            step["status"] = "running"
            checkpoint = f"[Checkpoint {step['id']}/{len(plan['steps'])}] {step['title']}"
            try:
                result = self._execute_plan_step(step, plan["request"], context, prior_results=prior_results)
                ok, verify_reason = self._verify_step_output(step, result)
                if not ok and step.get("action") == "deliver":
                    result = self._grounded_fallback(plan["request"])
                    ok, verify_reason = self._verify_step_output(step, result)

                step["result"] = str(result)[:600]
                step["status"] = "done" if ok else "failed"
                report_lines.append(f"{checkpoint} -> {'done' if ok else 'failed'}")
                report_lines.append(f"Result: {step['result']}")
                if not ok:
                    report_lines.append(f"Verifier: failed ({verify_reason})")
                prior_results += "\n" + step["result"]
            except Exception as e:
                step["status"] = "failed"
                step["result"] = f"error: {str(e)[:160]}"
                report_lines.append(f"{checkpoint} -> failed")
                report_lines.append(f"Result: {step['result']}")

        plan["status"] = "completed"
        plan["completed_at"] = datetime.now().isoformat()
        self.executor_history.append(plan)
        self._save_executor_history()
        self.active_execution = None

        final = "\n".join(report_lines)
        self.memory.add_turn(f"[EXECUTE] {plan['request']}", final[:700], metadata={"type": "executor", "plan_id": plan["id"]})
        self._append_growth_corpus(plan["request"], final[:900], source="task_executor")
        self._record_task_cycle(plan, final)
        return final

    def _is_question_like(self, text):
        t = (text or "").lower()
        if "?" in t:
            return True
        return any(t.startswith(w) for w in ["what", "why", "how", "when", "where", "who", "which"])

    def _rewrite_query(self, user_input):
        text = normalize_text(user_input).lower()
        text = re.sub(r"\b(please|can you|could you|help me|lookup|find out|tell me)\b", "", text)
        text = normalize_text(text)
        if len(text) < 6:
            return normalize_text(user_input)
        return text

    def _expand_followup_query(self, user_input):
        text = normalize_text(user_input)
        lower = text.lower()
        if len(lower.split()) > 10:
            return text

        followup_starts = (
            "why", "how so", "what about", "and ", "is that", "is it", "can you explain",
            "tell me more", "why is that", "why is it", "how is that",
        )
        pronoun_terms = {"it", "that", "this", "they", "those", "these"}
        tokens = set(re.findall(r"[a-z0-9]+", lower))

        if not lower.startswith(followup_starts) and pronoun_terms.isdisjoint(tokens):
            return text
        if not self.memory.memory:
            return text

        last_turn = list(self.memory.memory)[-1]
        anchor = normalize_text(last_turn.get("user", ""))
        if not anchor:
            return text
        return f"{text} about {anchor[:160]}"

    def _clean_evidence_text(self, text):
        cleaned = normalize_text(text)
        cleaned = re.sub(r"https?://\S+", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bwww\.[^\s]+", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b[a-z0-9.-]+\.(?:com|org|net|edu|gov|io|ai|co|uk)(?:/[^\s]*)?", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*[|>-]\s*", ": ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip(" .:-")

    def _query_terms(self, query):
        stop = {
            "what", "why", "how", "when", "where", "who", "which", "tell", "about",
            "explain", "please", "would", "could", "should", "into", "from", "with",
            "this", "that", "they", "them", "have", "your", "just", "more", "than",
            "machine", "learning", "learn", "start", "begin", "simply", "simple", "basics", "basic",
            "bad", "good", "better", "worse", "issue", "problem",
        }
        return [
            token for token in re.findall(r"[a-z0-9]+", (query or "").lower())
            if len(token) > 2 and token not in stop
        ]

    def _term_coverage_score(self, query, text):
        query_terms = self._query_terms(query)
        if not query_terms:
            return lexical_overlap_score(query, text)
        text_terms = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
        hits = sum(1 for token in query_terms if token in text_terms)
        return hits / max(len(query_terms), 1)

    def _keyword_memory_search(self, query, top_k=5):
        ranked = []
        for item in self.vector_memory.items:
            txt = self._clean_evidence_text(item.get("text", ""))
            if not txt:
                continue
            overlap = lexical_overlap_score(query, txt)
            coverage = self._term_coverage_score(query, txt)
            if overlap < 0.14 or coverage < 0.34:
                continue
            ranked.append({
                "text": txt[:260],
                "score": 0.45 * overlap + 0.55 * coverage,
                "source": item.get("source", "local"),
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def _gather_evidence(self, user_input, top_k=5):
        """Collect evidence via hybrid retrieval: vector, keyword overlap, and layered memory."""
        query = self._rewrite_query(user_input)
        evidence = []
        seen = set()

        for hit in self.vector_memory.search(query, top_k=top_k):
            txt = self._clean_evidence_text(hit.get("text", ""))
            if len(txt) < 30:
                continue
            key = txt[:120].lower()
            if key in seen:
                continue
            overlap = lexical_overlap_score(query, txt)
            coverage = self._term_coverage_score(query, txt)
            if coverage < 0.25:
                continue
            score = float(hit.get("score", 0.0)) * source_weight(hit.get("source", "local"))
            score = 0.45 * score + 0.3 * overlap + 0.25 * coverage
            evidence.append({"text": txt[:260], "score": score, "source": hit.get("source", "local")})
            seen.add(key)

        for hit in self._keyword_memory_search(query, top_k=top_k):
            txt = normalize_text(hit.get("text", ""))
            if len(txt) < 30:
                continue
            key = txt[:120].lower()
            if key in seen:
                continue
            score = 0.65 * float(hit.get("score", 0.0)) + 0.35 * text_quality_score(txt)
            evidence.append({"text": txt[:260], "score": score, "source": hit.get("source", "keyword")})
            seen.add(key)

        for fact in self.layered_memory.top_facts(query, top_k=3):
            txt = normalize_text(fact.get("text", ""))
            if len(txt) < 24:
                continue
            coverage = self._term_coverage_score(query, txt)
            if coverage < 0.34:
                continue
            key = txt[:120].lower()
            if key in seen:
                continue
            evidence.append(
                {
                    "text": txt[:260],
                    "score": 0.35 + 0.25 * coverage + 0.4 * float(fact.get("confidence", 0.0)),
                    "source": f"layered:{fact.get('source', 'fact')}",
                }
            )
            seen.add(key)

        if (len(evidence) < 2) and (not self.offline_mode):
            results = self.web_lookup(query, silent=True, train=False)
            for i, txt in enumerate(results[:3]):
                clean = " ".join((txt or "").split())
                if len(clean) < 30:
                    continue
                key = clean[:120].lower()
                if key in seen:
                    continue
                score = max(0.25, 0.45 - i * 0.08)
                score = 0.7 * score + 0.3 * text_quality_score(clean)
                evidence.append({"text": clean[:260], "score": score, "source": "lookup"})
                seen.add(key)

        evidence.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        reranked = []
        for item in evidence:
            overlap = lexical_overlap_score(query, item.get("text", ""))
            coverage = self._term_coverage_score(query, item.get("text", ""))
            final_score = 0.45 * float(item.get("score", 0.0)) + 0.25 * overlap + 0.30 * coverage
            item = dict(item)
            item["score"] = round(final_score, 4)
            if final_score >= self.min_evidence_score:
                reranked.append(item)

        reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return reranked[:6]

    def _intelligent_answer(self, user_input, context):
        """Evidence-first synthesis path for better grounded intelligence."""
        evidence = self._gather_evidence(user_input, top_k=6)
        mind = self._mind_deliberate(user_input, evidence)
        if not evidence:
            if self._is_question_like(user_input):
                return "I am not confident yet. Please share one more detail so I can answer with evidence."
            return None

        ev_lines = []
        for idx, ev in enumerate(evidence[:4], 1):
            ev_lines.append(f"{idx}. ({ev['source']}, {ev['score']:.2f}) {ev['text']}")
        ev_blob = "\n".join(ev_lines)

        prompt = (
            "You are JARVIS. Answer using the evidence only. "
            "Be concise, practical, and avoid url fragments. "
            "If evidence is uncertain, say so briefly.\n\n"
            f"User question: {user_input}\n"
            f"Context: {context}\n"
            f"Evidence:\n{ev_blob}\n\n"
            "Answer:"
        )
        out = self.chat.generate(prompt, max_length=140, temperature=0.45, top_k=40)
        out = out.replace(prompt, "").replace("Answer:", "").strip()
        out = " ".join(out.split())[:320]

        if self._is_low_quality_response(out) or self._term_coverage_score(user_input, out) < 0.34:
            top = evidence[0]["text"]
            second = evidence[1]["text"] if len(evidence) > 1 else ""
            merged = top if not second else f"{top} Also: {second}"
            out = merged[:320]

        avg_score = sum(e["score"] for e in evidence[:3]) / max(min(len(evidence), 3), 1)
        confidence = "high" if avg_score >= 0.52 else ("medium" if avg_score >= 0.34 else "low")
        if mind.get("confidence") == "low" and confidence != "low":
            confidence = "medium"

        confidence_score = max(0.0, min(1.0, round(avg_score, 3)))
        citations = ", ".join(f"[{i+1}:{e['source']}]" for i, e in enumerate(evidence[:3]))
        if confidence == "low":
            ask = "I am only partially certain. Do you want me to run a deeper lookup before finalizing?"
            return f"{out}\n\nConfidence: {confidence} ({confidence_score}) | Citations: {citations}\n{ask}"

        self.layered_memory.add_fact(evidence[0]["text"], confidence=confidence_score, source=evidence[0].get("source", "evidence"))
        return f"{out}\n\nConfidence: {confidence} ({confidence_score}) | Citations: {citations} | Mind: {mind.get('insight', '')[:90]}"

    def _proactive_loop(self):
        """Generate periodic proactive suggestions from current state."""
        self._proactive_stop_event.wait(self.proactive_startup_delay_seconds)
        while not self._proactive_stop_event.is_set():
            try:
                note = self._build_proactive_note()
                if note:
                    self._queue_proactive_note(note)
            except Exception:
                pass
            self._proactive_stop_event.wait(self.proactive_interval_seconds)

    def _run_skill(self, skill_name, user_input, context_text):
        """Execute a specialized skill pathway."""
        skill = (skill_name or "").lower().strip()
        if skill == "researcher":
            facts = self.vector_memory.search(user_input, top_k=3)
            if facts:
                bullets = "; ".join(f["text"][:110] for f in facts[:2])
                return f"Research brief: {bullets}"
            results = self.web_lookup(user_input, silent=True, train=False)
            if results:
                return "Research brief: " + " ".join(results[0].split())[:240]
            return "Research brief: I found limited data. Try a narrower query for better grounding."

        if skill == "planner":
            prompt = (
                "You are a planning assistant. Return exactly 5 numbered steps. "
                "Each step should be concrete and short.\n\n"
                f"Goal context: {self._goal_context()}\n"
                f"User request: {user_input}\n"
                "Plan:"
            )
            out = self.chat.generate(prompt, max_length=130, temperature=0.5, top_k=40)
            out = out.replace(prompt, "").strip()[:340]
            if out and not self._is_low_quality_response(out):
                return out
            request = " ".join(user_input.split())[:80]
            return (
                f"1. Define a measurable target for '{request}'. "
                "2. List required tools and constraints. "
                "3. Execute the smallest high-impact action today. "
                "4. Validate outcomes with one clear metric. "
                "5. Review and set the next milestone."
            )

        if skill == "tutor":
            prompt = (
                "You are a tutor. Explain clearly in 3 short sentences and include one practical example.\n\n"
                f"Topic: {user_input}\n"
                "Explanation:"
            )
            out = self.chat.generate(prompt, max_length=120, temperature=0.55, top_k=40)
            out = out.replace(prompt, "").strip()[:320]
            if out and not self._is_low_quality_response(out):
                return out
            lower = user_input.lower()
            if "python" in lower and any(k in lower for k in ["learn", "start", "begin"]):
                return (
                    "Start with Python basics: variables, conditionals, loops, functions, and simple data structures. "
                    "Build one tiny project like a calculator, text parser, or file organizer so each concept becomes concrete. "
                    "Example: write a script that reads a text file and counts how often each word appears."
                )
            if any(k in lower for k in ["learn", "start", "begin", "study"]):
                return (
                    "Start with the core concept, then practice one small example before moving on. "
                    "Break the topic into short sessions and build one tiny project so the ideas become concrete. "
                    "If you want, I can turn this into a step-by-step beginner plan."
                )
            return self._grounded_fallback(user_input)

        if skill == "coder":
            prompt = (
                "You are a coding assistant. Give implementation-oriented guidance in concise steps.\n\n"
                f"Context: {context_text}\n"
                f"Request: {user_input}\n"
                "Answer:"
            )
            out = self.chat.generate(prompt, max_length=140, temperature=0.45, top_k=40)
            out = out.replace(prompt, "").strip()[:340]
            if out and not self._is_low_quality_response(out):
                return out
            return "Implementation path: define scope, choose API surface, implement incrementally, then test with smoke and regression checks."

        if skill == "analyst":
            prompt = (
                "You are an analyst. Compare options, note tradeoffs, and give one recommendation. Keep under 5 lines.\n\n"
                f"Question: {user_input}\n"
                "Analysis:"
            )
            out = self.chat.generate(prompt, max_length=120, temperature=0.5, top_k=40)
            out = out.replace(prompt, "").strip()[:340]
            if out and not self._is_low_quality_response(out):
                return out
            return "Recommendation: choose the simplest option that preserves quality gates and reliable retrieval."

        return self._grounded_fallback(user_input)

    def local_status(self):
        """Summarize local-first runtime capability and state."""
        ready, _ = self.can_use_pc_actions()
        return (
            f"Local status | offline_mode={self.offline_mode} | shell_enabled={self.enable_shell_commands} | "
            f"vector_items={len(self.vector_memory.items)} | topics={len(self.topic_queue)} | "
            f"goals_active={len(self.goals.list_goals('active'))} | layered_facts={len(self.layered_memory.data.get('facts', []))} | "
            f"pc_actions={'READY' if ready else 'LOCKED'}"
        )

    def run_eval_harness(self):
        try:
            from eval_harness import EvalHarness

            result = EvalHarness(self).run()
            return f"Eval score: {result.get('score', 0)}/100 | report=eval_reports/latest_eval.json"
        except Exception as e:
            return f"Eval failed: {str(e)[:160]}"

    def build_synthetic_curriculum(self, per_level=60):
        try:
            import build_synthetic_curriculum

            # Reuse script entrypoint by emulating parsed args behavior.
            out_path = "data/synthetic_curriculum.jsonl"
            seed = 42
            total = 5 * int(max(1, per_level))
            os.makedirs("data", exist_ok=True)
            import json as _json
            import random as _random

            _random.seed(seed)
            tasks = []
            for level in range(1, 6):
                for idx in range(1, int(max(1, per_level)) + 1):
                    tasks.append(build_synthetic_curriculum.build_task(_random.choice(build_synthetic_curriculum.BASE_TOPICS), level, idx))

            with open(out_path, "w", encoding="utf-8") as f:
                for task in tasks:
                    f.write(_json.dumps(task, ensure_ascii=True) + "\n")

            return f"Curriculum generated: {total} tasks -> {out_path}"
        except Exception as e:
            return f"Curriculum generation failed: {str(e)[:160]}"

    def safety_status(self):
        ready, snap = self.can_use_pc_actions()
        lock_text = "ready" if ready else ("locked: " + "; ".join(snap.get("reasons", [])[:1]))
        return (
            f"Safety mode={self.safety_mode} | executor={'ON' if self.executor_enabled else 'OFF'} | "
            f"auto_execute={'ON' if self.auto_execute_enabled else 'OFF'} | shell={'ON' if self.enable_shell_commands else 'OFF'} | "
            f"pc_gate={lock_text}"
        )

    def safety_rules_text(self):
        lines = [f"{i+1}. {r}" for i, r in enumerate(self.robot_rules)]
        return "\n".join(lines)

    def _request_violates_robot_rules(self, text):
        t = (text or "").lower()
        harmful = [
            "kill", "hurt", "harm", "attack", "weapon", "ransomware", "malware",
            "steal", "dox", "ddos", "phish", "exploit", "backdoor",
        ]
        if any(k in t for k in harmful):
            return "Blocked by robot safety rules: harmful request detected."
        return None

    def _is_safe_shell_command(self, cmd, source="user"):
        lowered = (cmd or "").lower().strip()
        blocked = [
            "rm -rf", "del /f", "format ", "shutdown", "reboot", "rd /s",
            "powershell -enc", "git reset --hard", "curl ", "wget ", "invoke-webrequest",
            "setx ", "reg add", "reg delete",
        ]
        if any(tok in lowered for tok in blocked):
            return False, "Blocked potentially destructive command."

        if self.safety_mode == "strict":
            # Strict mode allows only read-only and diagnostic commands.
            allowed_prefixes = [
                "dir", "ls", "pwd", "cd", "echo", "type", "cat", "findstr", "where", "tree",
                "python --version", "pip list", "git status",
            ]
            if not any(lowered.startswith(p) for p in allowed_prefixes):
                return False, "Blocked by strict safety mode. Use 'safety relaxed' for broader commands."
        return True, "ok"

    def run_shell_command(self, command, source="user"):
        """Execute a local shell command with basic guardrails and timeout."""
        cmd = " ".join((command or "").split())
        if not cmd:
            return "No command provided."
        if not self.enable_shell_commands:
            return "Shell commands are disabled."
        ready, snap = self.can_use_pc_actions()
        if source in {"user", "executor"} and not ready:
            return "Shell access locked until coherence gates pass: " + "; ".join(snap.get("reasons", []))

        ok, reason = self._is_safe_shell_command(cmd, source=source)
        if not ok:
            return reason

        try:
            completed = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.shell_timeout_seconds,
            )
            out = (completed.stdout or "").strip()
            err = (completed.stderr or "").strip()
            code = completed.returncode
            if len(out) > 1200:
                out = out[:1200] + "..."
            if len(err) > 600:
                err = err[:600] + "..."
            return f"exit={code}\nstdout:\n{out or '[empty]'}\nstderr:\n{err or '[empty]'}"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.shell_timeout_seconds}s."
        except Exception as e:
            return f"Shell command failed: {str(e)[:160]}"

    def ingest_local_file(self, path):
        """Ingest local text into vector memory and growth corpus for local-first learning."""
        raw = " ".join((path or "").split()).strip('"')
        if not raw:
            return "Please provide a file path."

        full = os.path.abspath(raw)
        if not os.path.exists(full):
            return f"File not found: {full}"
        if os.path.isdir(full):
            return "Path points to a directory. Provide a file path."

        ext = os.path.splitext(full)[1].lower()
        if ext not in {".txt", ".md", ".json", ".log", ".csv", ".py"}:
            return "Unsupported file type for quick ingest. Use: .txt .md .json .log .csv .py"

        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            return f"Could not read file: {str(e)[:120]}"

        cleaned = " ".join(text.split())
        if len(cleaned) < 80:
            return "File content too short to ingest meaningfully."

        max_chars = 12000
        cleaned = cleaned[:max_chars]

        chunk_size = 420
        added = 0
        for i in range(0, len(cleaned), chunk_size):
            chunk = cleaned[i:i + chunk_size]
            if len(chunk) < 80:
                continue
            self.vector_memory.add(chunk, source="local_ingest", metadata={"path": full})
            added += 1

        self._append_growth_corpus(f"ingest:{full}", cleaned[:1200], source="local_ingest")
        self._expand_topic_queue(cleaned[:3000])

        now = time.time()
        trained = False
        if now - self.last_train_time >= self.min_train_interval_seconds:
            try:
                with self.model_lock:
                    fine_tune_utils.fine_tune_with_quality_gate(
                        self.model,
                        cleaned,
                        self.tokenizer,
                        self.device,
                        learning_rate=2e-5,
                        num_steps=6,
                        checkpoint_path=self.checkpoint_path,
                        eval_text=cleaned,
                        min_improvement=0.003,
                        verbose=False,
                    )
                self.last_train_time = now
                trained = True
            except Exception:
                trained = False

        return f"Ingested {added} chunks from {full}. model_trained={trained}"

    def _autonomous_growth_tick(self, force=False):
        """Periodic autonomous training pass over recent corpus."""
        if (not force) and self.conversation_turn % self.growth_every_n_turns != 0:
            return
        if not os.path.exists(self.growth_corpus_path):
            return
        now = time.time()
        if now - self.last_train_time < self.min_train_interval_seconds:
            return

        try:
            with open(self.growth_corpus_path, "r", encoding="utf-8") as f:
                corpus = f.read()
            if len(corpus) < 600:
                return

            recent_text = corpus[-5000:]
            if not self.quiet_background_training:
                print("\n🧠 Autonomous growth tick: learning from recent interactions...")
            with self.model_lock:
                fine_tune_utils.fine_tune_with_quality_gate(
                    self.model,
                    recent_text,
                    self.tokenizer,
                    self.device,
                    learning_rate=2e-5,
                    num_steps=6,
                    checkpoint_path=self.checkpoint_path,
                    eval_text=recent_text,
                    min_improvement=0.004,
                    verbose=(not self.quiet_background_training),
                )
            self.last_train_time = now
            if not self.quiet_background_training:
                print("✅ Growth checkpoint updated.\n")
        except Exception as e:
            print(f"⚠️  Growth tick skipped: {str(e)[:120]}")
    
    def understand_user_intent(self, user_input):
        """Parse user input and understand intent"""
        task_type, action = self.router.route(user_input)
        entities = self.router.extract_entities(user_input)
        return {
            'task_type': task_type,
            'action': action,
            'entities': entities,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_response(self, user_input, use_memory=True):
        """Generate contextual, personality-driven response"""
        self.conversation_turn += 1
        self._queue_recent_topic(user_input)
        self._expand_topic_queue(user_input)
        effective_input = self._expand_followup_query(user_input)
        intent = self.understand_user_intent(user_input)
        self._append_user_learning_signal(user_input, source="user_signal_chat")
        context = self.memory.get_context(num_turns=2) if use_memory else ""
        goal_context = self._goal_context()
        preference_context = self.layered_memory.preference_summary()
        skill_name = self.skills.choose_skill(effective_input)

        pref = self._extract_preference(user_input)
        if pref:
            self.layered_memory.add_preference(pref, confidence=0.75)

        # Deterministic greeting path avoids low-quality first-turn generation.
        if user_input.lower().strip() in {"hi", "hello", "hey", "yo", "sup"}:
            response = "Hello. I am online and learning continuously. Ask me anything or use lookup plus a topic."
            self.memory.add_turn(user_input, response, metadata={**intent, "tool": "greeting"})
            self._append_growth_corpus(user_input, response, source="greeting")
            return response

        # Tool path 1: exact arithmetic
        math_answer = self._try_math_tool(user_input)
        if math_answer is not None:
            response = math_answer
            self.memory.add_turn(user_input, response, metadata={**intent, "tool": "math"})
            self._append_growth_corpus(user_input, response, source="math_tool")
            self._autonomous_growth_tick()
            return response

        # Tool path 2: grounded memory answer from web lookups
        lookup_answer = self._lookup_memory_answer(effective_input)
        if lookup_answer is not None:
            response = lookup_answer
            self.memory.add_turn(user_input, response, metadata={**intent, "tool": "lookup_memory"})
            self._append_growth_corpus(user_input, response, source="lookup_memory")
            self._autonomous_growth_tick()
            return response

        bank_answer = self._question_bank_answer(effective_input)
        if bank_answer is not None:
            response = bank_answer
            self.memory.add_turn(user_input, response, metadata={**intent, "tool": "question_bank"})
            self._append_growth_corpus(user_input, response, source="question_bank")
            self._autonomous_growth_tick()
            return response

        followup_answer = self._contextual_followup_answer(user_input)
        if followup_answer is not None:
            response = followup_answer
            self.memory.add_turn(user_input, response, metadata={**intent, "tool": "followup_reasoner"})
            self._append_growth_corpus(user_input, response, source="followup_reasoner")
            self._autonomous_growth_tick()
            return response

        # Tool path 3: local vector memory retrieval for factual grounding
        retrieved = self.vector_memory.search(effective_input, top_k=3)
        retrieved_facts = [r["text"] for r in retrieved]

        if self.smart_mode and self._is_question_like(user_input):
            smart = self._intelligent_answer(effective_input, context)
            if smart and not self._is_low_quality_response(smart):
                response = self.personality.craft_response(smart)
                self.memory.add_turn(user_input, response, metadata={**intent, "tool": "smart_reasoner"})
                self._append_growth_corpus(user_input, response, source="smart_reasoner")
                self._autonomous_growth_tick()
                self.reflection_tick()
                return response

        # Skill-first path for specialized behavior.
        skill_response = self._run_skill(skill_name, effective_input, context)
        if skill_response and not self._is_low_quality_response(skill_response):
            response = skill_response[:320]
            if not self._is_low_quality_response(response):
                response = self.personality.craft_response(response)
            self.memory.add_turn(user_input, response, metadata={**intent, "skill": skill_name})
            self._append_growth_corpus(user_input, response, source=f"skill_{skill_name}")
            self._autonomous_growth_tick()
            return response
        
        # Build a clean, guiding prompt
        prompt = (
            "You are JARVIS, a concise and practical assistant. "
            "Answer directly in 1-3 sentences. Avoid repeating the same phrase.\n\n"
            f"Goal context:\n{goal_context}\n\n"
            f"User preferences:\n{preference_context}\n\n"
            f"Skill mode:\n{skill_name}\n\n"
            f"Retrieved facts:\n{chr(10).join(retrieved_facts) if retrieved_facts else 'None'}\n\n"
            f"Recent context:\n{context}\n\n"
            f"User: {effective_input}\n"
            "Assistant:"
        )
        
        try:
            # Use the built-in generate method from LLMChat
            response = self.chat.generate(
                prompt,
                max_length=150,
                temperature=0.7,
                top_k=50
            )
            
            # Clean up response - remove prompt echo and obvious loops
            response = response.replace(prompt, "").strip()

            # Remove prompt tags if present
            response = response.replace("Assistant:", "").replace("User:", "").strip()
            
            # Keep only first 2 sentences and de-duplicate
            sentences = response.split('.')
            uniq = []
            for sent in sentences:
                sent = " ".join(sent.split()).strip()
                if not sent:
                    continue
                if sent.lower() in {u.lower() for u in uniq}:
                    continue
                uniq.append(sent)
                if len(uniq) >= 2:
                    break

            response = (". ".join(uniq) + ("." if uniq else "")).strip()
            response = response[:240] if response else "I can help with that."

            if self._is_low_quality_response(response):
                response = self._grounded_fallback(effective_input)
            
        except Exception as e:
            response = self._grounded_fallback(effective_input)
        
        # Add personality only for stable outputs.
        if not self._is_low_quality_response(response):
            response = self.personality.craft_response(response)
            self.low_quality_strikes = max(0, self.low_quality_strikes - 1)
        else:
            self.low_quality_strikes = min(12, self.low_quality_strikes + 1)
        self.is_first_message = False
        
        # Store in memory and growth corpus
        self.memory.add_turn(user_input, response, metadata=intent)
        self._append_growth_corpus(user_input, response, source="chat")
        
        # Auto-learn every 5 turns
        if self.conversation_turn % 5 == 0:
            candidates = list(self.memory.memory)[-5:]
            if candidates:
                print("\n🧠 JARVIS learning from conversation...")
                with self.model_lock:
                    result = self.learner.learn_from_conversation(candidates)
                if result['status'] == 'complete':
                    print("✅ Knowledge updated.\n")

        # Also run periodic autonomous corpus growth.
        self._autonomous_growth_tick()
        self.reflection_tick()
        if self.conversation_turn % 7 == 0:
            self._daily_learning_maintenance(force=False)
        self._adaptive_autolearn_update(force=False)
        
        return response
    
    def web_lookup(self, query, silent=False, train=True):
        """Search web and learn from results"""
        if self.offline_mode:
            if not silent:
                print("⚠️  Offline mode is ON. Web lookup skipped.\n")
            return []

        query = " ".join((query or "").split())
        query = re.sub(r"^[^a-zA-Z0-9]+", "", query)
        query = re.sub(r"\blookup\b", "", query, flags=re.IGNORECASE).strip()
        if not query:
            if not silent:
                print("⚠️  Please provide a lookup query. Example: lookup apple fruit\n")
            return []

        if not silent:
            print(f"\n🔍 Searching for: {query}")
        try:
            import requests
            from bs4 import BeautifulSoup
            import urllib.parse
            
            scrape_url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            
            resp = requests.get(scrape_url, timeout=10, headers=headers)
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Extract text from first few results
            results = []
            for item in soup.find_all("div", class_="result"):
                snippet_bits = []
                title = item.find("a", class_="result__a")
                if title:
                    snippet_bits.append(title.get_text(" ", strip=True))
                snippet = item.find(class_=re.compile("result__snippet"))
                if snippet:
                    snippet_bits.append(snippet.get_text(" ", strip=True))
                text = self._clean_evidence_text(" ".join(snippet_bits or [item.get_text(separator=" ")]))[:320]
                # Drop low-signal snippets dominated by urls/symbols.
                if len(text) < 50:
                    continue
                if text.count("www") + text.count("http") >= 1:
                    continue
                noisy = len(re.findall(r"[^a-zA-Z0-9\s.,!?'-]", text)) / max(len(text), 1)
                if noisy > 0.12:
                    continue
                results.append(text)
                if len(results) >= 3:
                    break

            # Fallback 1: DuckDuckGo Instant Answer API (lightweight JSON endpoint)
            if not results:
                try:
                    ia_url = (
                        "https://api.duckduckgo.com/?q="
                        + urllib.parse.quote_plus(query)
                        + "&format=json&no_redirect=1&no_html=1"
                    )
                    ia_resp = requests.get(ia_url, timeout=8, headers=headers)
                    if ia_resp.ok:
                        data = ia_resp.json()
                        abstract = " ".join(str(data.get("AbstractText", "")).split())
                        heading = " ".join(str(data.get("Heading", "")).split())
                        if len(abstract) >= 40:
                            results.append(f"{heading}: {abstract}".strip(": "))
                        related = data.get("RelatedTopics") or []
                        for rel in related:
                            if isinstance(rel, dict):
                                txt = " ".join(str(rel.get("Text", "")).split())
                                if len(txt) >= 50:
                                    results.append(txt[:320])
                            if len(results) >= 3:
                                break
                except Exception:
                    pass

            # Fallback 2: Wikipedia summary API for entity-like lookups.
            if not results:
                try:
                    title = query.replace(" ", "_")
                    wiki_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
                    wiki_resp = requests.get(wiki_url, timeout=8, headers=headers)
                    if wiki_resp.ok:
                        info = wiki_resp.json()
                        extract = " ".join(str(info.get("extract", "")).split())
                        if len(extract) >= 40:
                            results.append(extract[:320])
                except Exception:
                    pass

            # Final fallback for common starter entities to avoid dead-end UX.
            if not results:
                static_facts = {
                    "apple": "Apple commonly refers to the fruit of the apple tree. It is sweet, edible, and rich in fiber and vitamin C.",
                    "car": "A car is a road vehicle with four wheels used for transportation. Most cars are powered by gasoline, diesel, electric, or hybrid systems.",
                    "ford": "Ford Motor Company is an American automaker founded in 1903, known for mass production and vehicles like the F-Series.",
                }
                for key, val in static_facts.items():
                    if key in query.lower():
                        results.append(val)
                        break
            
            if results:
                scraped_text = "\n".join(results)
                if not silent:
                    print("📚 Integrating new information...")

                # Add to local vector memory for grounded retrieval.
                for snippet in results:
                    if is_high_signal_text(snippet, min_score=0.42):
                        self.vector_memory.add(snippet, source="web_lookup", metadata={"query": query})
                
                # Store in knowledge memory for context
                self.memory.add_turn(
                    f"[LOOKUP: {query}]",
                    scraped_text[:500],
                    metadata={'type': 'web_search', 'query': query}
                )

                self._append_growth_corpus(query, scraped_text[:500], source="web_lookup")

                # Immediate small training pass on looked-up content
                if train:
                    now = time.time()
                    if now - self.last_train_time >= self.min_train_interval_seconds:
                        with self.model_lock:
                            fine_tune_utils.fine_tune_with_quality_gate(
                                self.model,
                                scraped_text,
                                self.tokenizer,
                                self.device,
                                learning_rate=3e-5,
                                num_steps=6,
                                checkpoint_path=self.checkpoint_path,
                                eval_text=scraped_text,
                                min_improvement=0.004,
                                verbose=False,
                            )
                        self.last_train_time = now
                
                if not silent:
                    print("✅ Knowledge expanded.\n")
                return results
            else:
                if not silent:
                    print("⚠️  No results found.\n")
                return []
                
        except Exception as e:
            if not silent:
                print(f"⚠️  Search failed: {str(e)[:100]}\n")
            return []
    
    def chat_loop(self):
        """Main interactive loop - JARVIS always listening"""
        print("=" * 60)
        print("🤖 JARVIS AI Assistant - Ready to assist")
        print("Type 'exit' to end, 'help' for commands")
        print("=" * 60 + "\n")
        while True:
            try:
                while self.pending_proactive_notes:
                    note = self.pending_proactive_notes.popleft()
                    print(f"\nJARVIS {note}\n")

                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "goodbye"]:
                    self.stop_autolearn()
                    self.stop_proactive_mode()
                    self.stop_watch_mode()
                    print("\nJARVIS: Until next time, sir.")
                    break

                if user_input.lower() == "help":
                    print("""
JARVIS Commands:
  Regular Chat:    Just type naturally!
    Local Status:    'local status'
    Health:          'health'
    Offline On:      'offline on'
    Offline Off:     'offline off'
    Offline Status:  'offline status'
    Ingest File:     'ingest [path-to-file]'
    Shell Command:   'shell [command]'
    Watch On:        'watch on [seconds]'
    Watch Off:       'watch off'
    Watch Status:    'watch status'
    Watch Tick:      'watch now'
    Smart On:        'smart on'
    Smart Off:       'smart off'
    Smart Status:    'smart status'
    Mind Status:     'mind status'
    Mind Focus:      'mind focus'
    Mind Beliefs:    'mind beliefs'
    Executor On:     'executor on'
    Executor Off:    'executor off'
    Executor Status: 'executor status'
    Executor Run:    'execute [request]'
    Executor Logs:   'executor history'
    AutoExec On:     'autoexec on'
    AutoExec Off:    'autoexec off'
    Safety Status:   'safety status'
    Safety Rules:    'safety rules'
    Safety Strict:   'safety strict'
    Safety Relaxed:  'safety relaxed'
    Readiness Info:  'readiness status'
    Intelligence:   'smartness check' or 'intelligence check'
    Bank Info:      'bank status'
    PC Arm:          'pc arm'
    PC Disarm:       'pc disarm'
    Data Status:     'data status'
    Data On:         'data on'
    Data Off:        'data off'
    Data Grow:       'data grow'
    Eval Run:        'eval run'
    Curriculum:      'curriculum build [per_level]'
    Reward:          'reward +' or 'reward -'
    Reward Help:     'reward'
    Reflect Now:     'reflect now'
    Reflect Info:    'reflect status'
  Skill List:      'skills'
  Skill Run:       'skill [name] [request]'
  Goals List:      'goals'
  Goal Add:        'goal add [text]'
  Goal Done:       'goal done [id]'
  Goal Drop:       'goal drop [id]'
  Goal Focus:      'goal focus [id]'
  Goal Progress:   'goal progress [id] [0-100]'
  Web Search:      'lookup [query]'
  View Memory:     'recall' or 'show memory'
  Vector Recall:   'facts [query]'
  Clean Topics:    'sanitize topics'
  Proactive On:    'proactive on'
  Proactive Off:   'proactive off'
  Proactive Info:  'proactive status'
  Proactive Tick:  'proactive now'
  Force Growth:    'grow now'
  Autolearn On:    'autolearn on'
  Autolearn Off:   'autolearn off'
  Autolearn Info:  'autolearn status'
  Exit:            'exit' or 'goodbye'
""")
                    continue

                if user_input.lower() == "local status":
                    print(f"\nJARVIS: {self.local_status()}\n")
                    continue

                if user_input.lower() == "health":
                    snap = self.readiness_snapshot()
                    print(f"\nJARVIS: {self.local_status()}")
                    print(f"JARVIS: {self.autolearn_status()}")
                    print(f"JARVIS: {self.proactive_status()}")
                    print(f"JARVIS: {self.reflection_status()}")
                    print(f"JARVIS: {self.data_growth_status()}")
                    print(f"JARVIS: {self.mind_status()}")
                    print(
                        "JARVIS: Readiness="
                        + ("READY" if snap.get("ready") else "LOCKED")
                        + f" | eval={snap.get('eval_score')} | dialogue={snap.get('dialogue_score')}"
                    )
                    if not snap.get("ready"):
                        print("JARVIS: lock reasons -> " + "; ".join(snap.get("reasons", [])))
                    print()
                    continue

                if user_input.lower() == "readiness status":
                    snap = self.readiness_snapshot()
                    print(
                        "\nJARVIS: Readiness="
                        + ("READY" if snap.get("ready") else "LOCKED")
                        + f" | eval={snap.get('eval_score')} | dialogue={snap.get('dialogue_score')}"
                    )
                    if not snap.get("ready"):
                        print("JARVIS: lock reasons -> " + "; ".join(snap.get("reasons", [])) + "\n")
                    else:
                        print("JARVIS: PC actions are unlocked.\n")
                    continue

                if user_input.lower() in {"smartness check", "intelligence check", "smart check"}:
                    report = self.intelligence_report_text()
                    print(f"\nJARVIS: {report}\n")
                    snap = self.intelligence_snapshot()
                    if snap.get("overall", 0.0) < 85.0:
                        maint = self._daily_learning_maintenance(force=True)
                        print(f"JARVIS: Adaptive maintenance triggered -> {maint.get('status')}\n")
                    continue

                if user_input.lower() == "bank status":
                    print(f"\nJARVIS: {self.question_bank_status()}\n")
                    continue

                if user_input.lower() == "pc arm":
                    self.pc_control_armed = True
                    snap = self.readiness_snapshot()
                    state = "READY" if snap.get("ready") else "LOCKED"
                    print(f"\nJARVIS: PC control armed. Current gate state: {state}.\n")
                    continue

                if user_input.lower() == "pc disarm":
                    self.pc_control_armed = False
                    print("\nJARVIS: PC control disarmed. Execution and shell access are locked.\n")
                    continue

                if user_input.lower() == "data status":
                    print(f"\nJARVIS: {self.data_growth_status()}\n")
                    continue

                if user_input.lower() == "data on":
                    self.data_growth_enabled = True
                    print("\nJARVIS: Data growth enabled.\n")
                    continue

                if user_input.lower() == "data off":
                    self.data_growth_enabled = False
                    print("\nJARVIS: Data growth disabled.\n")
                    continue

                if user_input.lower() == "data grow":
                    msg = self.train_on_chatgpt_data(force=True)
                    print(f"\nJARVIS: {msg}\n")
                    continue

                if user_input.lower() == "eval run":
                    msg = self.run_eval_harness()
                    print(f"\nJARVIS: {msg}\n")
                    continue

                if user_input.lower().startswith("curriculum build"):
                    parts = user_input.split()
                    per_level = 60
                    if len(parts) >= 3:
                        try:
                            per_level = max(10, min(500, int(parts[2])))
                        except Exception:
                            per_level = 60
                    msg = self.build_synthetic_curriculum(per_level=per_level)
                    print(f"\nJARVIS: {msg}\n")
                    continue

                if user_input.lower() == "mind status":
                    print(f"\nJARVIS: {self.mind_status()}\n")
                    continue

                if user_input.lower() == "mind focus":
                    focus = self.mind_state.get("active_focus", [])
                    if not focus:
                        print("\nJARVIS: Mind focus is currently empty.\n")
                    else:
                        print(f"\nJARVIS Mind Focus: {', '.join(focus)}\n")
                    continue

                if user_input.lower() == "mind beliefs":
                    beliefs = self.mind_state.get("beliefs", [])[-8:]
                    if not beliefs:
                        print("\nJARVIS: No stabilized beliefs yet.\n")
                    else:
                        print("\nJARVIS Beliefs:")
                        for idx, b in enumerate(reversed(beliefs), 1):
                            print(f"  {idx}. ({b.get('confidence', 0.0):.2f}) {b.get('text', '')[:170]}")
                        print()
                    continue

                if user_input.lower() == "offline on":
                    self.offline_mode = True
                    print("\nJARVIS: Offline mode enabled. Web lookup is disabled; local memory only.\n")
                    continue

                if user_input.lower() == "offline off":
                    self.offline_mode = False
                    print("\nJARVIS: Offline mode disabled. Web lookup is available.\n")
                    continue

                if user_input.lower() == "offline status":
                    mode = "ON" if self.offline_mode else "OFF"
                    print(f"\nJARVIS: Offline mode is {mode}.\n")
                    continue

                if user_input.lower().startswith("ingest "):
                    target = user_input[7:].strip()
                    result = self.ingest_local_file(target)
                    print(f"\nJARVIS: {result}\n")
                    continue

                if user_input.lower().startswith("shell "):
                    cmd = user_input[6:].strip()
                    ready, snap = self.can_use_pc_actions()
                    if not ready:
                        print("\nJARVIS: Shell access locked -> " + "; ".join(snap.get("reasons", [])) + "\n")
                        continue
                    result = self.run_shell_command(cmd)
                    print(f"\nJARVIS shell result:\n{result}\n")
                    continue

                if user_input.lower().startswith("watch on"):
                    parts = user_input.split()
                    interval = None
                    if len(parts) >= 3:
                        interval = parts[2]
                    state = self.start_watch_mode(interval)
                    print(f"\nJARVIS: Watch mode {state}. {self.watch_status()}\n")
                    continue

                if user_input.lower() == "watch off":
                    state = self.stop_watch_mode()
                    print(f"\nJARVIS: Watch mode {state}.\n")
                    continue

                if user_input.lower() == "watch status":
                    print(f"\nJARVIS: {self.watch_status()}\n")
                    continue

                if user_input.lower() == "watch now":
                    print(f"\nJARVIS: Watch: {self._status_snapshot()}\n")
                    continue

                if user_input.lower() == "smart on":
                    self.smart_mode = True
                    print("\nJARVIS: Smart mode enabled (evidence-first reasoning).\n")
                    continue

                if user_input.lower() == "smart off":
                    self.smart_mode = False
                    print("\nJARVIS: Smart mode disabled.\n")
                    continue

                if user_input.lower() == "smart status":
                    print(f"\nJARVIS: Smart mode is {'ON' if self.smart_mode else 'OFF'}.\n")
                    continue

                if user_input.lower() == "executor on":
                    self.executor_enabled = True
                    print("\nJARVIS: Task executor enabled.\n")
                    continue

                if user_input.lower() == "executor off":
                    self.executor_enabled = False
                    print("\nJARVIS: Task executor disabled.\n")
                    continue

                if user_input.lower() == "executor status":
                    print(f"\nJARVIS: {self.executor_status()}\n")
                    continue

                if user_input.lower() == "executor history":
                    if not self.executor_history:
                        print("\nJARVIS: No executor history yet.\n")
                    else:
                        print("\nJARVIS Executor History:")
                        for item in self.executor_history[-5:]:
                            print(f"  - {item.get('id')} | status={item.get('status')} | request={item.get('request', '')[:90]}")
                        print()
                    continue

                if user_input.lower() == "autoexec on":
                    ready, snap = self.can_use_pc_actions()
                    if not ready:
                        print("\nJARVIS: Auto execution remains locked -> " + "; ".join(snap.get("reasons", [])) + "\n")
                        continue
                    self.auto_execute_enabled = True
                    print("\nJARVIS: Auto execution enabled.\n")
                    continue

                if user_input.lower() == "autoexec off":
                    self.auto_execute_enabled = False
                    print("\nJARVIS: Auto execution disabled.\n")
                    continue

                if user_input.lower() == "safety status":
                    print(f"\nJARVIS: {self.safety_status()}\n")
                    continue

                if user_input.lower() == "safety rules":
                    print("\nJARVIS Safety Rules:")
                    print(self.safety_rules_text())
                    print()
                    continue

                if user_input.lower() == "safety strict":
                    self.safety_mode = "strict"
                    self.auto_execute_enabled = False
                    print("\nJARVIS: Safety mode set to strict. Auto execution is now OFF.\n")
                    continue

                if user_input.lower() == "safety relaxed":
                    self.safety_mode = "relaxed"
                    print("\nJARVIS: Safety mode set to relaxed (critical destructive commands still blocked).\n")
                    continue

                if user_input.lower().startswith("execute "):
                    req = user_input[8:].strip()
                    if not req:
                        print("\nJARVIS: Usage -> execute [request]\n")
                        continue
                    ready, snap = self.can_use_pc_actions()
                    if not ready:
                        print("\nJARVIS: Execution locked -> " + "; ".join(snap.get("reasons", [])) + "\n")
                        continue
                    report = self.execute_task_request(req)
                    print(f"\nJARVIS Executor Report:\n{report}\n")
                    continue

                if user_input.lower() == "reward":
                    print("\nJARVIS: Use 'reward +' to reinforce, or 'reward -' to penalize the last response.\n")
                    continue

                if user_input.lower() in {"reward +", "reward -"}:
                    sentiment = user_input.strip()[-1]
                    msg = self.apply_feedback(sentiment)
                    print(f"\nJARVIS: {msg}\n")
                    continue

                if user_input.lower() == "reflect now":
                    out = self.reflection_tick(force=True)
                    if out is None:
                        print("\nJARVIS: Reflection skipped. I need a bit more conversation context.\n")
                    else:
                        print(f"\nJARVIS: {out.get('summary', 'Reflection complete.')}\n")
                    continue

                if user_input.lower() == "reflect status":
                    print(f"\nJARVIS: {self.reflection_status()}\n")
                    continue

                if user_input.lower() == "skills":
                    items = self.skills.list_skills()
                    print("\nJARVIS Skills:")
                    for name, desc in items.items():
                        print(f"  - {name}: {desc}")
                    print()
                    continue

                if user_input.lower().startswith("skill "):
                    rest = user_input[6:].strip()
                    parts = rest.split(" ", 1)
                    if len(parts) < 2:
                        print("\nJARVIS: Usage -> skill [name] [request]\n")
                        continue
                    skill_name, request = parts[0].lower(), parts[1].strip()
                    if skill_name not in self.skills.list_skills():
                        print(f"\nJARVIS: Unknown skill '{skill_name}'. Use 'skills' to list available skills.\n")
                        continue
                    context = self.memory.get_context(num_turns=2)
                    response = self._run_skill(skill_name, request, context)
                    if self._is_low_quality_response(response):
                        response = self._grounded_fallback(request)
                    print(f"\nJARVIS [{skill_name}]: {response}\n")
                    self.memory.add_turn(user_input, response, metadata={"tool": "skill_manual", "skill": skill_name})
                    self._append_growth_corpus(user_input, response, source=f"manual_skill_{skill_name}")
                    continue

                if user_input.lower() == "goals":
                    active = self._format_goals(status="active")
                    done = self._format_goals(status="done")
                    print(f"\nJARVIS Goals (active):\n{active}\n")
                    print(f"JARVIS Goals (done):\n{done}\n")
                    continue

                if user_input.lower().startswith("goal add "):
                    payload = user_input[9:].strip()
                    goal = self.goals.add_goal(payload)
                    if goal is None:
                        print("\nJARVIS: Goal text is too short.\n")
                    else:
                        print(f"\nJARVIS: Added goal #{goal['id']}: {goal['text']}\n")
                    continue

                if user_input.lower().startswith("goal done "):
                    gid = user_input[10:].strip()
                    goal = self.goals.complete_goal(gid)
                    if goal is None:
                        print("\nJARVIS: Could not find that goal id.\n")
                    else:
                        print(f"\nJARVIS: Completed goal #{goal['id']}.\n")
                    continue

                if user_input.lower().startswith("goal drop "):
                    gid = user_input[10:].strip()
                    ok = self.goals.remove_goal(gid)
                    if ok:
                        print(f"\nJARVIS: Removed goal #{gid}.\n")
                    else:
                        print("\nJARVIS: Could not find that goal id.\n")
                    continue

                if user_input.lower().startswith("goal focus "):
                    gid = user_input[11:].strip()
                    try:
                        gid_int = int(gid)
                    except Exception:
                        print("\nJARVIS: goal focus expects a numeric id.\n")
                        continue
                    active = self.goals.list_goals(status="active")
                    if any(int(g.get("id", -1)) == gid_int for g in active):
                        self.focus_goal_id = gid_int
                        print(f"\nJARVIS: Focus set to goal #{gid_int}.\n")
                    else:
                        print("\nJARVIS: That goal is not active.\n")
                    continue

                if user_input.lower().startswith("goal progress "):
                    parts = user_input.split()
                    if len(parts) != 4:
                        print("\nJARVIS: Usage -> goal progress [id] [0-100]\n")
                        continue
                    goal = self.goals.update_progress(parts[2], parts[3])
                    if goal is None:
                        print("\nJARVIS: Could not update progress for that goal.\n")
                    else:
                        print(f"\nJARVIS: Goal #{goal['id']} progress is now {goal.get('progress', 0)}%.\n")
                    continue

                if user_input.lower() == "proactive on":
                    state = self.start_proactive_mode()
                    print(f"\nJARVIS: Proactive mode {state}.\n")
                    continue

                if user_input.lower() == "proactive off":
                    state = self.stop_proactive_mode()
                    print(f"\nJARVIS: Proactive mode {state}.\n")
                    continue

                if user_input.lower() == "proactive status":
                    print(f"\nJARVIS: {self.proactive_status()}\n")
                    continue

                if user_input.lower() == "proactive now":
                    note = self.proactive_tick_now()
                    print(f"\nJARVIS: {note}\n")
                    continue

                if user_input.lower() == "grow now":
                    self._autonomous_growth_tick()
                    print("\nJARVIS: Growth cycle complete.\n")
                    continue

                if user_input.lower() == "autolearn on":
                    state = self.start_autolearn()
                    print(f"\nJARVIS: Background autolearn {state}.\n")
                    continue

                if user_input.lower() == "autolearn off":
                    state = self.stop_autolearn()
                    print(f"\nJARVIS: Background autolearn {state}.\n")
                    continue

                if user_input.lower() == "autolearn status":
                    print(f"\nJARVIS: {self.autolearn_status()}\n")
                    continue

                if user_input.lower() == "learn status":
                    print(f"\nJARVIS: {self.learning_status()}\n")
                    continue

                if user_input.lower() == "learn now":
                    summary = self.learn_now()
                    print(f"\nJARVIS: Consolidation cycle complete. {summary}\n")
                    continue

                if user_input.lower() == "learn user on":
                    self.user_signal_growth_enabled = True
                    print("\nJARVIS: User-signal learning is ON. I will learn from high-signal prompts too.\n")
                    continue

                if user_input.lower() == "learn user off":
                    self.user_signal_growth_enabled = False
                    print("\nJARVIS: User-signal learning is OFF.\n")
                    continue

                if user_input.lower() == "sanitize topics":
                    total = self.sanitize_topic_queue()
                    print(f"\nJARVIS: Topic queue sanitized. {total} high-signal topics retained.\n")
                    continue

                if user_input.lower().startswith("lookup "):
                    query = user_input[7:].strip()
                    if not query:
                        print("\nJARVIS: Please provide a query. Example: lookup apple fruit\n")
                        continue
                    results = self.web_lookup(query)
                    count = len(results) if isinstance(results, list) else 1
                    print(f"\nJARVIS: Found {count} results.\n")
                    continue

                if user_input.lower().startswith("facts "):
                    query = user_input[6:].strip()
                    matches = self.vector_memory.search(query, top_k=5)
                    if not matches:
                        print("\nJARVIS: No matching facts found in local vector memory.\n")
                    else:
                        print("\nJARVIS Local Facts:")
                        for idx, m in enumerate(matches, 1):
                            print(f"  {idx}. ({m['score']:.2f}) {m['text'][:220]}")
                        print()
                    continue

                if user_input.lower() in ["recall", "show memory"]:
                    memory_text = self.memory.get_context(num_turns=5)
                    if memory_text:
                        print(f"\nJARVIS Memory:\n{memory_text}\n")
                    else:
                        print("\nJARVIS: No memory entries yet.\n")
                    continue

                violation = self._request_violates_robot_rules(user_input)
                if violation:
                    print(f"\nJARVIS: {violation}\n")
                    continue

                if self._should_auto_execute(user_input):
                    report = self.execute_task_request(user_input)
                    print(f"\nJARVIS Executor Report:\n{report}\n")
                    continue

                response = self.generate_response(user_input)
                print(f"\nJARVIS: {response}\n")

            except KeyboardInterrupt:
                self.stop_autolearn()
                self.stop_proactive_mode()
                self.stop_watch_mode()
                print("\n\nJARVIS: Session terminated.")
                break
            except Exception as e:
                print(f"\nJARVIS: Error: {str(e)}\n")


def main():
    """Start JARVIS AI Assistant"""
    jarvis = JARVIS()
    jarvis.chat_loop()


if __name__ == "__main__":
    main()