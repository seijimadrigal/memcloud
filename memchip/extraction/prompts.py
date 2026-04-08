"""
Extraction prompts — the heart of the system.
"""

TRIPLE_EXTRACTION_PROMPT = """You are a precise memory extraction system. Extract semantic triples from the conversation below.

Each triple is an atomic unit of knowledge: (subject, predicate, object).

RULES:
1. Extract EVERY factual statement — preferences, plans, events, relationships, opinions
2. Use EXACT names — "Alice's colleague Rob", not "a colleague"  
3. Include ALL numbers, dates, prices, percentages exactly as stated
4. Preserve frequencies — "every Tuesday and Thursday", not "twice a week"
5. One fact per triple — split compound facts into multiple triples
6. For temporal facts, include when in the predicate: "started working at (in March 2023)"
7. For opinions/preferences, use predicates like "likes", "prefers", "dislikes", "wants"
8. For relationships, be specific: "is married to", "is colleague of", "is sister of"
9. ALWAYS extract: book titles, movie names, song names, places of origin, reasons for decisions
10. For places of origin: "moved from X" → extract (Person, moved from, X) AND (Person, previously lived in, X)
11. For reasons/motivations: "chose X because Y" → extract (Person, chose X because, Y)

EXAMPLES:
Input: "Melanie read 'Nothing is Impossible' and 'Charlotte's Web' last month"
Output: [
  {{"subject": "Melanie", "predicate": "read book", "object": "Nothing is Impossible", "confidence": 0.95}},
  {{"subject": "Melanie", "predicate": "read book", "object": "Charlotte's Web", "confidence": 0.95}}
]

Input: "Caroline moved from Sweden to start a new life"
Output: [
  {{"subject": "Caroline", "predicate": "moved from", "object": "Sweden", "confidence": 0.95}},
  {{"subject": "Caroline", "predicate": "previously lived in", "object": "Sweden", "confidence": 0.95}}
]

Input: "Caroline chose the adoption agency because of their inclusivity and support for LGBTQ+ couples"
Output: [
  {{"subject": "Caroline", "predicate": "chose adoption agency because of", "object": "inclusivity and support for LGBTQ+ couples", "confidence": 0.95}}
]

Conversation timestamp: {timestamp}

Conversation:
{text}

Return a JSON array of triples. Each triple has: subject, predicate, object, confidence (0-1).
Extract ALL facts. Do not summarize or omit details. Return ONLY the JSON array."""


SUMMARY_EXTRACTION_PROMPT = """Summarize this conversation in 2-4 sentences. You MUST capture:
1. The main topics discussed
2. Key decisions or plans made (with specific reasons)
3. Any changes in circumstances or preferences
4. ALL specific details: book titles, place names, reasons for choices, names of organizations

IMPORTANT: Never generalize specific details. Write "read 'Nothing is Impossible'" not "read some books". Write "moved from Sweden" not "moved from another country". Write "chose agency for LGBTQ+ inclusivity" not "chose for personal reasons".

Conversation timestamp: {timestamp}

Conversation:
{text}

Write a concise, factual summary:"""


ENTITY_EXTRACTION_PROMPT = """Extract all named entities from this conversation.

For each entity provide:
- name: The entity's name (use full name when available)
- type: One of: PERSON, PLACE, ORGANIZATION, EVENT, PRODUCT, DATE, OTHER
- description: Brief description based on context (1 sentence)
- aliases: Other names/references to the same entity in the text

Conversation:
{text}

Return a JSON array. Example:
[{{"name": "Alice Chen", "type": "PERSON", "description": "Software engineer who recently moved to Tokyo", "aliases": ["Alice", "she"]}}]

Return ONLY the JSON array."""


TEMPORAL_EXTRACTION_PROMPT = """Extract temporal events and their ordering from this conversation.

For each event provide:
- event: What happened (specific, with names and details)
- timestamp: The original time reference from the text
- absolute_date: REQUIRED — Convert to actual date in YYYY-MM-DD format using the conversation timestamp
- duration: How long it lasted (if mentioned)
- recurring: Whether it's a recurring event (true/false)
- frequency: If recurring, how often

CRITICAL: You MUST convert ALL relative time references to absolute dates.

CONVERSION EXAMPLES (if conversation timestamp is "25 May 2023"):
- "last Saturday" → "2023-05-20" (the Saturday before May 25)
- "last Sunday" → "2023-05-21" (the Sunday before May 25) 
- "two weeks ago" → "2023-05-11"
- "last month" → "2023-04-25" (approximately)
- "yesterday" → "2023-05-24"
- "next Friday" → "2023-06-02"
- "the Sunday before" → calculate the actual Sunday before the conversation date

Conversation timestamp: {timestamp}

Conversation:
{text}

IMPORTANT: 
- EVERY event MUST have an absolute_date in YYYY-MM-DD format. Use the conversation timestamp to calculate it.
- If the exact date is ambiguous, give your best estimate and note it.
- "last Saturday" when conversation is on May 25 (Thursday) = May 20 (Saturday)

Return a JSON array. Return ONLY the JSON array."""


PROFILE_EXTRACTION_PROMPT = """Extract profile attributes about each person mentioned in this conversation.

Categories:
- PREFERENCE: Things they like/dislike/prefer
- HABIT: Regular behaviors or routines  
- FACT: Biographical facts (job, location, education, family, place of origin)
- GOAL: Things they want to achieve or are working toward
- OPINION: Their views or beliefs on topics
- SKILL: Things they're good at or experienced in
- RELATIONSHIP: Their connections to other people

For each attribute provide:
- person: Who this is about
- category: One of the categories above
- attribute: The specific attribute (e.g., "favorite food", "moved from", "book read")
- value: The EXACT value (e.g., "sushi", "Sweden", "Nothing is Impossible")
- confidence: How certain (0-1)

IMPORTANT: Extract specific values, not vague descriptions. Use exact book titles, place names, organization names, reasons.

Conversation:
{text}

Return a JSON array. Return ONLY the JSON array."""
