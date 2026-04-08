"""
Memory extraction pipeline.
Converts raw conversation text into structured memories:
- Semantic triples (subject-predicate-object)
- Conversation summaries
- Entity extraction + linking
- Profile attributes
- Temporal events with before/after relations
"""

from __future__ import annotations

import json
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from memchip.extraction.prompts import (
    TRIPLE_EXTRACTION_PROMPT,
    SUMMARY_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    TEMPORAL_EXTRACTION_PROMPT,
    PROFILE_EXTRACTION_PROMPT,
)
from memchip.llm import call_llm


@dataclass
class Extraction:
    """Result of extraction pipeline."""
    triples: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""
    entities: List[Dict[str, str]] = field(default_factory=list)
    temporal_events: List[Dict[str, Any]] = field(default_factory=list)
    profile_attributes: List[Dict[str, str]] = field(default_factory=list)
    raw_text: str = ""
    importance: int = 3  # 0-5 score


class ExtractionPipeline:
    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "openai/gpt-4.1-mini",
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def classify(self, text: str) -> Dict[str, Any]:
        """Rate importance 0-5. Single LLM call."""
        prompt = f"""Rate the importance of this text for long-term memory. Return ONLY a single integer 0-5.

0 = NOISE — heartbeats, status checks, "HEARTBEAT_OK", gateway connect/disconnect, empty exchanges
1 = OPERATIONAL — deployment logs, plugin updates, build outputs, routine confirmations
2 = ROUTINE — task progress updates, standard work, feature implementation steps
3 = SIGNIFICANT — decisions, preferences, useful findings, research results
4 = CRITICAL — architecture decisions, API keys/passwords, business strategy, key relationships
5 = FOUNDATIONAL — core identity, long-term strategy, permanent reference material

Return ONLY the number. Nothing else.

Text:
{text[:1500]}"""
        try:
            response = call_llm(
                prompt=prompt,
                provider=self.provider,
                model=self.model,
                api_key=self.api_key,
            )
            score = int(re.search(r'\d+', response.strip()).group())
            return {"importance": min(max(score, 0), 5)}
        except Exception:
            return {"importance": 3}

    def extract(
        self,
        text: str,
        user_id: str = "default",
        session_id: str = "",
        timestamp: str = "",
        score_threshold: int = 1,
    ) -> Extraction:
        """Run full extraction pipeline on text. Skips if importance < threshold."""
        import concurrent.futures
        result = Extraction(raw_text=text)

        # Classify: importance only (1 cheap LLM call)
        classification = self.classify(text)
        result.importance = classification["importance"]

        if result.importance < score_threshold:
            # Not worth remembering — return empty extraction
            return result

        # Run extractions in parallel (5 independent LLM calls)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            f_triples = executor.submit(self._extract_triples, text, timestamp)
            f_summary = executor.submit(self._extract_summary, text, timestamp)
            f_entities = executor.submit(self._extract_entities, text)
            f_temporal = executor.submit(self._extract_temporal, text, timestamp)
            f_profile = executor.submit(self._extract_profile, text)

            result.triples = f_triples.result()
            result.summary = f_summary.result()
            result.entities = f_entities.result()
            result.temporal_events = f_temporal.result()
            result.profile_attributes = f_profile.result()

        return result

    def _extract_triples(self, text: str, timestamp: str) -> List[Dict[str, str]]:
        """Extract semantic triples (subject-predicate-object)."""
        response = call_llm(
            prompt=TRIPLE_EXTRACTION_PROMPT.format(text=text, timestamp=timestamp),
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
        )
        return _parse_json_list(response)

    def _extract_summary(self, text: str, timestamp: str) -> str:
        """Extract conversation summary."""
        response = call_llm(
            prompt=SUMMARY_EXTRACTION_PROMPT.format(text=text, timestamp=timestamp),
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
        )
        return response.strip()

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract entities with types."""
        response = call_llm(
            prompt=ENTITY_EXTRACTION_PROMPT.format(text=text),
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
        )
        return _parse_json_list(response)

    def _extract_temporal(self, text: str, timestamp: str) -> List[Dict[str, Any]]:
        """Extract temporal events with ordering relations."""
        response = call_llm(
            prompt=TEMPORAL_EXTRACTION_PROMPT.format(text=text, timestamp=timestamp),
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
        )
        return _parse_json_list(response)

    def _extract_profile(self, text: str) -> List[Dict[str, str]]:
        """Extract profile attributes (preferences, habits, facts about people)."""
        response = call_llm(
            prompt=PROFILE_EXTRACTION_PROMPT.format(text=text),
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
        )
        return _parse_json_list(response)


def _parse_json_list(text: str) -> List[Dict]:
    """Parse JSON array from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Remove markdown code block wrapper
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        # Try to find JSON array in text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []
