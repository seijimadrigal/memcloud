"""MemChip core — the 3-line API."""

from __future__ import annotations

import sqlite3
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from memchip.extraction.pipeline import ExtractionPipeline
from memchip.storage.sqlite_store import SQLiteStore
from memchip.retrieval.engine import RetrievalEngine


class MemChip:
    """
    Memory chip for AI agents.
    
    Usage:
        chip = MemChip()
        chip.add("User said they love hiking and live in Tokyo")
        result = chip.recall("Where does the user live?")
    """

    def __init__(
        self,
        db_path: str = "memchip.db",
        user_id: str = "default",
        llm_provider: str = "openrouter",
        llm_model: str = "openai/gpt-4.1-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
    ):
        self.user_id = user_id
        self.store = SQLiteStore(db_path=db_path)
        self.extractor = ExtractionPipeline(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key,
        )
        self.retrieval = RetrievalEngine(
            store=self.store,
            embedding_model=embedding_model,
            llm_provider=llm_provider,
            llm_model=llm_model,
            api_key=api_key,
        )

    def add(
        self,
        text: str,
        session_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a conversation or text to memory.
        Extracts structured memories (triples, summaries, entities, profiles).
        
        Returns dict with counts of extracted memories.
        """
        session_id = session_id or f"session_{int(time.time())}"
        timestamp = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")

        # Extract structured memories
        extraction = self.extractor.extract(
            text=text,
            user_id=self.user_id,
            session_id=session_id,
            timestamp=timestamp,
        )

        # Store raw conversation text for fallback search
        self.store.store_raw(
            user_id=self.user_id,
            session_id=session_id,
            text=text,
            timestamp=timestamp,
        )

        # Store all extracted memories
        stored = self.store.store_extraction(
            extraction=extraction,
            user_id=self.user_id,
            session_id=session_id,
            timestamp=timestamp,
        )

        return stored

    def recall(
        self,
        query: str,
        top_k: int = 10,
        max_tokens: int = 1500,
        agentic: bool = True,
    ) -> Dict[str, Any]:
        """
        Recall relevant memories for a query.
        
        Uses multi-stage retrieval:
        1. Hybrid search (BM25 + vector + graph)
        2. Agentic multi-round (if enabled and first pass insufficient)
        3. Reranking
        4. Context assembly
        
        Returns dict with 'memories' list and 'context' string.
        """
        return self.retrieval.recall(
            query=query,
            user_id=self.user_id,
            top_k=top_k,
            max_tokens=max_tokens,
            agentic=agentic,
        )

    def answer(self, query: str, agentic: bool = True) -> str:
        """
        Recall memories and generate an answer.
        Uses chain-of-thought reasoning over retrieved memories.
        """
        recall_result = self.recall(query=query, agentic=agentic)
        return self.retrieval.answer(
            query=query,
            context=recall_result["context"],
            memories=recall_result["memories"],
        )

    def clear(self, user_id: Optional[str] = None):
        """Clear all memories for a user."""
        self.store.clear(user_id=user_id or self.user_id)
