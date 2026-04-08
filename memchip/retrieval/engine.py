"""
Multi-stage retrieval engine.

Stage 1: Hybrid search (BM25 + vector + graph + raw text), fused with RRF
Stage 2: Agentic multi-round (if confidence < threshold, rephrase and re-search)
Stage 3: Context assembly (pack into token budget)
"""

from __future__ import annotations

import json
import re
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from memchip.llm import call_llm
from memchip.retrieval.prompts import (
    SUFFICIENCY_CHECK_PROMPT,
    MULTI_QUERY_PROMPT,
    ANSWER_PROMPT,
)

ENTITY_EXTRACTION_PROMPT = """Extract ALL important search terms from this question. Include:
- Person names (e.g., "Caroline", "Melanie")
- Place names (e.g., "Sweden", "Tokyo")
- Organization names
- Specific things mentioned (e.g., "book", "charity race", "adoption agency")
- Key nouns and verbs that would help find the answer in a database

Question: {query}

Return a JSON object: {{"entities": ["term1", "term2"], "key_phrases": ["phrase1", "phrase2"]}}
Return ONLY JSON."""


class RetrievalEngine:
    def __init__(
        self,
        store,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_provider: str = "openrouter",
        llm_model: str = "openai/gpt-4.1-mini",
        api_key: Optional[str] = None,
    ):
        self.store = store
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.api_key = api_key
        self._embedder = None
        self._embedding_model_name = embedding_model

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self._embedding_model_name)
            except ImportError:
                self._embedder = None
        return self._embedder

    def recall(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        max_tokens: int = 3000,
        agentic: bool = True,
    ) -> Dict[str, Any]:
        """Multi-stage retrieval pipeline."""

        # Stage 1: Hybrid search
        candidates = self._hybrid_search(query, user_id, top_k=top_k * 3)

        # Stage 2: Agentic multi-round (if enabled)
        if agentic and candidates:
            candidates = self._agentic_retrieval(query, candidates, user_id, top_k)

        # Stage 3: Score-adaptive truncation (keep all within 70% of top score)
        if candidates:
            top_score = candidates[0]["rrf_score"]
            threshold = top_score * 0.7
            ranked = [c for c in candidates if c["rrf_score"] >= threshold]
            ranked = ranked[:20]  # cap to avoid token explosion
        else:
            ranked = []

        # Get session dates for temporal context
        session_dates = self.store.get_session_dates(user_id)

        # Stage 4: Assemble context with session timeline
        context = self._assemble_context(ranked, max_tokens=max_tokens, session_dates=session_dates)

        return {
            "memories": ranked,
            "context": context,
            "num_candidates": len(candidates),
            "num_returned": len(ranked),
        }

    def _hybrid_search(self, query: str, user_id: str, top_k: int = 30) -> List[Dict]:
        """
        Hybrid retrieval combining:
        1. BM25 full-text search (via FTS5)
        2. Entity-based graph walk
        3. Profile attribute search
        4. Temporal event search
        5. Raw conversation text search
        """
        results = {}  # content -> {score, source, ...}

        # 1. BM25 search
        fts_results = self.store.search_fts(query, user_id, limit=top_k)
        for i, r in enumerate(fts_results):
            key = r["content"]
            if key not in results:
                # Look up timestamp for this memory
                ts = self.store.get_memory_timestamp(r["memory_type"], r.get("memory_id", ""))
                results[key] = {
                    "content": key,
                    "type": r["memory_type"],
                    "bm25_rank": i + 1,
                    "sources": ["bm25"],
                    "timestamp": ts,
                }
            else:
                results[key]["sources"].append("bm25")
                if "bm25_rank" not in results[key]:
                    results[key]["bm25_rank"] = i + 1

        # 2. Extract entities using LLM for better coverage
        entities = self._extract_query_entities(query)
        self._current_query_entities = entities  # cache for NER-weighted scoring
        
        for entity in entities:
            # Direct triple lookup (searches subject, object, and predicate)
            triples = self.store.get_triples(user_id, subject=entity)
            for t in triples:
                ts_info = f" [session: {t['timestamp']}]" if t.get('timestamp') else ""
                content = f"{t['subject']} {t['predicate']} {t['object']}{ts_info}"
                if content not in results:
                    results[content] = {"content": content, "type": "triple", "sources": []}
                results[content]["sources"].append("graph")
                results[content]["triple"] = t

            # Graph walk (2-hop)
            graph_results = self.store.graph_walk(user_id, entity, hops=2)
            for g in graph_results:
                ts_g = f" [session: {g['timestamp']}]" if g.get('timestamp') else ""
                content = f"{g['source_entity']} {g['relation']} {g['target_entity']}{ts_g}"
                if content not in results:
                    results[content] = {"content": content, "type": "relation", "sources": [], "timestamp": g.get("timestamp")}
                results[content]["sources"].append(f"graph_hop{g['hop']}")

            # 3. Profile search
            profiles = self.store.get_profile(user_id, person=entity)
            for p in profiles:
                content = f"{p['person']}: {p['attribute']} = {p['value']}"
                if content not in results:
                    results[content] = {"content": content, "type": "profile", "sources": []}
                results[content]["sources"].append("profile")

        # 4. Temporal search (for time-related queries)
        if self._is_temporal_query(query):
            events = self.store.get_temporal_events(user_id)
            for e in events:
                content = f"{e['event']} ({e.get('absolute_date', e.get('timestamp_raw', 'unknown date'))})"
                if content not in results:
                    results[content] = {"content": content, "type": "temporal", "sources": []}
                results[content]["sources"].append("temporal")

        # 5. Summaries (always include for context)
        summaries = self.store.get_summaries(user_id)
        for s in summaries:
            content = s["summary"]
            if content not in results:
                results[content] = {"content": content, "type": "summary", "sources": []}
            results[content]["sources"].append("summary")

        # 6. Raw conversation text search
        raw_results = self.store.search_raw(query, user_id, limit=5)
        for r in raw_results:
            # Truncate long raw text to most relevant snippet
            text = r["text"]
            if len(text) > 500:
                # Find the most relevant part by searching for query terms
                query_terms = [t.lower() for t in query.split() if len(t) > 3]
                best_pos = 0
                best_score = 0
                for i in range(0, len(text) - 200, 50):
                    chunk = text[i:i+400].lower()
                    score = sum(1 for t in query_terms if t in chunk)
                    if score > best_score:
                        best_score = score
                        best_pos = i
                text = text[best_pos:best_pos+500]
            
            ts = r.get("timestamp", "")
            content = f"[Raw conversation, session date: {ts}] {text}"
            if content not in results:
                results[content] = {"content": content, "type": "raw", "sources": []}
            results[content]["sources"].append("raw_text")

        # RRF fusion scoring
        candidates = list(results.values())
        for c in candidates:
            c["rrf_score"] = self._compute_rrf_score(c)

        # Sort by RRF score
        candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        return candidates[:top_k]

    def _agentic_retrieval(
        self, query: str, candidates: List[Dict], user_id: str, top_k: int
    ) -> List[Dict]:
        """
        Agentic multi-round retrieval.
        Check sufficiency, then do up to 2 rounds of re-search.
        """
        # Format current results for sufficiency check
        docs_text = "\n".join(
            f"[{i+1}] ({c['type']}) {c['content']}" for i, c in enumerate(candidates[:15])
        )

        # Check sufficiency
        check_response = call_llm(
            prompt=SUFFICIENCY_CHECK_PROMPT.format(query=query, retrieved_docs=docs_text),
            provider=self.llm_provider,
            model=self.llm_model,
            api_key=self.api_key,
        )

        try:
            check = json.loads(self._extract_json(check_response))
            if check.get("is_sufficient", True):
                return candidates
        except (json.JSONDecodeError, TypeError):
            return candidates

        # Generate complementary queries
        missing_info = check.get("missing_information", [])
        key_info = check.get("key_information_found", [])

        query_response = call_llm(
            prompt=MULTI_QUERY_PROMPT.format(
                original_query=query,
                key_info=json.dumps(key_info),
                missing_info=json.dumps(missing_info),
                retrieved_docs=docs_text,
            ),
            provider=self.llm_provider,
            model=self.llm_model,
            api_key=self.api_key,
        )

        try:
            multi_q = json.loads(self._extract_json(query_response))
            new_queries = multi_q.get("queries", [])
        except (json.JSONDecodeError, TypeError):
            return candidates

        # Round 1: Search with new queries
        all_results = {c["content"]: c for c in candidates}
        for new_q in new_queries[:3]:
            new_results = self._hybrid_search(new_q, user_id, top_k=top_k)
            for r in new_results:
                if r["content"] not in all_results:
                    r["sources"].append("agentic_requery")
                    all_results[r["content"]] = r

        # Round 2: Check again and do one more round if needed
        merged = list(all_results.values())
        for c in merged:
            c["rrf_score"] = self._compute_rrf_score(c)
        merged.sort(key=lambda x: x["rrf_score"], reverse=True)

        docs_text2 = "\n".join(
            f"[{i+1}] ({c['type']}) {c['content']}" for i, c in enumerate(merged[:15])
        )
        check2_response = call_llm(
            prompt=SUFFICIENCY_CHECK_PROMPT.format(query=query, retrieved_docs=docs_text2),
            provider=self.llm_provider,
            model=self.llm_model,
            api_key=self.api_key,
        )
        try:
            check2 = json.loads(self._extract_json(check2_response))
            if not check2.get("is_sufficient", True):
                missing2 = check2.get("missing_information", [])
                key2 = check2.get("key_information_found", [])
                q2_response = call_llm(
                    prompt=MULTI_QUERY_PROMPT.format(
                        original_query=query,
                        key_info=json.dumps(key2),
                        missing_info=json.dumps(missing2),
                        retrieved_docs=docs_text2,
                    ),
                    provider=self.llm_provider,
                    model=self.llm_model,
                    api_key=self.api_key,
                )
                multi_q2 = json.loads(self._extract_json(q2_response))
                for new_q in multi_q2.get("queries", [])[:3]:
                    new_results = self._hybrid_search(new_q, user_id, top_k=top_k)
                    for r in new_results:
                        if r["content"] not in all_results:
                            r["sources"].append("agentic_round2")
                            all_results[r["content"]] = r
        except (json.JSONDecodeError, TypeError):
            pass

        merged = list(all_results.values())
        for c in merged:
            c["rrf_score"] = self._compute_rrf_score(c)
        merged.sort(key=lambda x: x["rrf_score"], reverse=True)

        return merged[:top_k * 2]

    def _assemble_context(self, memories: List[Dict], max_tokens: int = 3000, session_dates: Dict[str, str] = None) -> str:
        """Assemble memories into a context string within token budget."""
        lines = []
        char_budget = max_tokens * 5  # generous ~5 chars per token to include more context

        # Add session timeline for temporal date resolution
        if session_dates:
            timeline = "SESSION TIMELINE (use to convert 'yesterday', 'last Saturday', etc. to absolute dates):\n"
            for sid, ts in sorted(session_dates.items(), key=lambda x: x[1]):
                timeline += f"  {sid}: {ts}\n"
            lines.append(timeline)

        for m in memories:
            # Include timestamp/session info for temporal resolution
            timestamp = ""
            # Try triple timestamp first
            triple = m.get("triple")
            if triple and triple.get("timestamp"):
                timestamp = f" [session date: {triple['timestamp']}]"
            # Fall back to memory-level timestamp (from FTS lookup or graph walk)
            elif m.get("timestamp"):
                timestamp = f" [session date: {m['timestamp']}]"
            line = f"[{m['type'].upper()}]{timestamp} {m['content']}"
            if sum(len(l) for l in lines) + len(line) > char_budget:
                break
            lines.append(line)

        return "\n".join(lines)

    def answer(self, query: str, context: str, memories: List[Dict]) -> str:
        """Generate an answer using retrieved memories."""
        response = call_llm(
            prompt=ANSWER_PROMPT.format(context=context, question=query),
            provider=self.llm_provider,
            model=self.llm_model,
            api_key=self.api_key,
        )

        # Extract final answer — be robust about format
        if "FINAL ANSWER:" in response:
            answer = response.split("FINAL ANSWER:")[-1].strip()
            # Remove any trailing reasoning that leaked
            # Take only first paragraph if multiple
            lines = [l.strip() for l in answer.split("\n") if l.strip()]
            if lines:
                return " ".join(lines[:3])  # Max 3 lines
            return answer
        # If no FINAL ANSWER marker, take the last substantial line
        lines = [l.strip() for l in response.strip().split("\n") if l.strip() and len(l.strip()) > 10]
        if lines:
            return lines[-1]
        return response.strip()

    def _compute_rrf_score(self, candidate: Dict, k: int = 60) -> float:
        """Reciprocal Rank Fusion score."""
        score = 0.0
        # BM25 rank contribution
        if "bm25_rank" in candidate:
            score += 1.0 / (k + candidate["bm25_rank"])
        # Source diversity bonus
        num_sources = len(set(candidate.get("sources", [])))
        score += num_sources * 0.1
        # Graph hop penalty (further hops = less relevant)
        for s in candidate.get("sources", []):
            if s.startswith("graph_hop"):
                hop = int(s[-1])
                score += 1.0 / (k + hop * 10)
            elif s == "graph":
                score += 1.0 / (k + 1)  # Direct match
            elif s == "profile":
                score += 1.0 / (k + 2)
            elif s == "temporal":
                score += 1.0 / (k + 3)
            elif s == "summary":
                score += 1.0 / (k + 5)
            elif s == "raw_text":
                score += 1.0 / (k + 4)
        # NER-weighted boost: entities from query matching candidate content
        query_entities = getattr(self, "_current_query_entities", [])
        if query_entities:
            content_lower = candidate.get("content", "").lower()
            entity_matches = sum(1 for e in query_entities if e.lower() in content_lower)
            if entity_matches > 0:
                score *= (1 + entity_matches * 2.0)  # 3x for 1 match, 5x for 2

        return score

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities and key terms from query using LLM."""
        try:
            response = call_llm(
                prompt=ENTITY_EXTRACTION_PROMPT.format(query=query),
                provider=self.llm_provider,
                model=self.llm_model,
                api_key=self.api_key,
                max_tokens=256,
            )
            data = json.loads(self._extract_json(response))
            entities = data.get("entities", [])
            phrases = data.get("key_phrases", [])
            # Combine and deduplicate
            all_terms = list(dict.fromkeys(entities + phrases))
            return [t for t in all_terms if len(t) > 1]
        except (json.JSONDecodeError, TypeError, Exception):
            # Fallback to simple heuristic
            return self._extract_query_entities_simple(query)

    def _extract_query_entities_simple(self, query: str) -> List[str]:
        """Fallback: extract entities using simple heuristics."""
        words = query.split()
        entities = []
        current = []
        skip = {"what", "when", "where", "who", "how", "why", "did", "does",
                "is", "are", "was", "were", "the", "a", "an", "in", "on",
                "at", "to", "for", "of", "with", "and", "or", "not", "has",
                "have", "had", "do", "will", "would", "could", "should",
                "about", "from", "by", "that", "this", "it", "they", "he",
                "she", "his", "her", "their", "its", "my", "your"}
        for w in words:
            if w.lower() in skip:
                if current:
                    entities.append(" ".join(current))
                    current = []
                continue
            if w[0].isupper() or w.replace("'s", "").replace("'", "").isalpha():
                current.append(w.rstrip("?.,!"))
            else:
                if current:
                    entities.append(" ".join(current))
                    current = []
        if current:
            entities.append(" ".join(current))
        return [e for e in entities if len(e) > 1]

    def _is_temporal_query(self, query: str) -> bool:
        """Check if query involves temporal reasoning."""
        temporal_keywords = {
            "when", "before", "after", "during", "since", "until", "ago",
            "last", "next", "first", "latest", "recent", "earlier", "later",
            "how long", "how often", "date", "time", "year", "month", "week",
            "day", "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        }
        query_lower = query.lower()
        return any(kw in query_lower for kw in temporal_keywords)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        for pattern in [r"\{.*\}", r"\[.*\]"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group()
        return text
