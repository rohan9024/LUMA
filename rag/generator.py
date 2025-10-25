# luma/rag/generator.py
import os
from typing import List, Dict
import time

# You can integrate any LLM; here a stub that concatenates evidence.
class Generator:
    def __init__(self, model="openai:gpt-4o-mini"):
        self.model = model
        self.use_openai = "openai:" in model

    def answer(self, question: str, evidences: List[Dict]):
        # Compose grounded answer; enforce citations
        lines = [f"Q: {question}", "Answer (with citations):"]
        # Very simple heuristic: extract key facts from evidence text fields
        for i, e in enumerate(evidences, 1):
            src = e.get("source", "unknown")
            snip = e.get("text", "")[:400]
            lines.append(f"[{i}] {src}: {snip}")
        lines.append("Claimed facts are supported by the cited segments above.")
        return "\n".join(lines)