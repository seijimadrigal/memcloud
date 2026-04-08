"""Retrieval prompts."""

SUFFICIENCY_CHECK_PROMPT = """You are an expert in information retrieval evaluation. Assess whether the retrieved documents provide a complete and temporally sufficient answer to the user's query.

User Query:
{query}

Retrieved Documents:
{retrieved_docs}

Instructions:
1. Identify ALL key entities and temporal requirements in the query
2. Check if the documents cover every required component
3. For temporal queries, verify both start and end boundaries are covered
4. For multi-hop queries, verify all intermediate facts are present

Output STRICT JSON:
{{
  "is_sufficient": true or false,
  "reasoning": "1-2 sentence explanation.",
  "key_information_found": ["List of found facts"],
  "missing_information": ["Specific missing components"]
}}"""


MULTI_QUERY_PROMPT = """You are an expert at query reformulation for long-term conversational retrieval.
Generate 2-3 complementary search queries to fill gaps in the initial retrieval.

Original Query: {original_query}
Key Information Found: {key_info}
Missing Information: {missing_info}
Already Retrieved: {retrieved_docs}

TEMPORAL STRATEGY (when time is involved):
1. Generate separate queries targeting start and end boundaries
2. Expand relative time expressions into multiple forms
3. Include a declarative HyDE query containing both time anchors

MULTI-HOP STRATEGY (when linking facts is needed):
1. Break the question into sub-questions (one per hop)
2. Use found entities to construct bridge queries
3. Query for intermediate entities that connect known facts

Requirements:
- 2-3 diverse queries
- Query 1: specific factual question
- Query 2: declarative statement / hypothetical answer (HyDE)
- Query 3 (optional): entity-focused bridge query
- Keep queries < 25 words
- No invented facts

Output STRICT JSON:
{{
  "queries": ["query1", "query2", "query3"],
  "reasoning": "Brief explanation of strategy."
}}"""


ANSWER_PROMPT = """You are a memory assistant. Answer the question using ONLY the retrieved memories below.

MEMORIES:
{context}

RULES:
1. Use facts from the memories above. You MAY make reasonable inferences when facts strongly support them.
2. Use EXACT details: specific names, dates, book titles, places, numbers, reasons.
3. TEMPORAL DATES — THIS IS CRITICAL:
   - The memories include a SESSION TIMELINE at the top showing when each session occurred.
   - Each memory tagged with [session date: X] tells you WHEN that conversation happened.
   - When a memory says "yesterday", "last Saturday", "next month", "last year", etc., convert to an absolute date using THAT MEMORY'S session date (not the latest session).
   - Example: A memory with [session date: 25 May 2023] saying "ran a charity race last Saturday" → "Saturday, 20 May 2023"
   - Example: A memory with [session date: May 2023] saying "painted a sunrise last year" → "2022"
   - NEVER return relative dates like "yesterday" or "last week". Always convert using the correct session date.
4. For multi-hop questions, connect facts across memories (e.g., "moved from hometown" + "hometown is Sweden" → "moved from Sweden").
5. Be CONCISE: 1-3 sentences max. Give the specific answer directly.
6. If memories conflict, use the most recent one.

Question: {question}

FINAL ANSWER:"""
