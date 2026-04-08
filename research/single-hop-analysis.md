# Single-Hop Failure Analysis (v21.4, conv-42)

## Results: 16/37 correct (43.2%)

## Failure Pattern Breakdown (21 failures)
- **Counting errors**: 5 (24%) — "how many X?" gets wrong number
- **Incomplete lists**: 7 (33%) — gets some items but misses others  
- **Over-generation**: 4 (19%) — adds plausible but wrong items
- **Wrong/vague**: 5 (24%) — gives vague or incorrect answer

## Root Cause Analysis

### The REAL Problem: Scattered Facts + Limited Context Window

Single-hop questions like "What things has Nate recommended to Joanna?" require aggregating information from **26 different chunks** spread across 29 conversation sessions.

Our v10 pipeline sends only **top 4 chunks** to the LLM. Even with perfect retrieval, 4 chunks can't cover 8+ recommendations scattered across the full conversation.

### Why EverMemOS Succeeds: Episode Narratives

EverMemOS doesn't retrieve raw chunks. They:
1. Extract **episode narratives** — rich 3rd-person summaries of each session
2. Each episode preserves ALL specific details (names, numbers, frequencies)
3. Retrieve **top 10 episodes** (covering more sessions = more facts)
4. Use a **7-step chain-of-thought answer prompt** that forces cross-memory linking

Their answer prompt has explicit steps for:
- RELEVANT MEMORIES EXTRACTION (list all relevant memories)
- KEY INFORMATION IDENTIFICATION (extract ALL specifics)
- CROSS-MEMORY LINKING (combine facts from different memories)
- DETAIL VERIFICATION CHECKLIST (verify nothing was missed)

### What We're Doing Wrong

1. **Raw chunks are too narrow** — each chunk is ~200 words from one conversation turn. To answer "What has Nate recommended?", you'd need chunks from 8+ different sessions.

2. **Profile text loses specifics** — our entity profiles summarize everything into a paragraph, losing specific items like "coconut oil" or "dairy-free margarine".

3. **Top 4 chunks is too few** — even with perfect reranking, 4 chunks can't cover facts scattered across 29 sessions. EverMemOS uses top 10 episodes.

4. **No chain-of-thought** — our answer prompt says "be concise, answer like trivia quiz" which causes the model to skip items it's unsure about.

5. **KG triples are too atomic** — "Nate → recommended_to_Joanna → coconut oil" is stored but never reaches the answer because the answerer uses chunks, not KG.

## Proposed Fix: Episode-Based Single-Hop (v22)

### Key Insight
We already HAVE episode summaries (29 episodes for conv-42). We're just not using them for single-hop.

### Architecture
1. For single-hop questions, retrieve **top 10 episodes** (not chunks) via FTS5 on episodes table
2. Also include **all KG triples** for the target entity (already extracted)
3. Use a **multi-step answer prompt** (adapted from EverMemOS):
   - Step 1: List all relevant episodes
   - Step 2: Extract ALL specific items/names/numbers from each
   - Step 3: Cross-reference between episodes
   - Step 4: Compile complete answer
4. Use gpt-4.1 (not mini) for the answer — it's better at following complex prompts

### Why This Should Work
- Episodes cover full sessions (~500 words each), so 10 episodes = ~5000 words of context
- KG triples catch specific items that episodes might abbreviate
- Multi-step prompt forces comprehensive aggregation
- This is essentially what EverMemOS does, adapted to our storage format

### Expected Impact
- Single-hop: 43% → 60-70% (episodes cover more ground + better prompt)
- Multi-hop: Should remain ~80% (unchanged)  
- Temporal: Should remain ~81% (unchanged)
- Overall: 70% → 75-80%

### Cost
- No extra infrastructure
- No reingestion needed (episodes already extracted)
- Just routing + prompt change
- May need gpt-4.1 for answer generation (~2-3x cost per question vs mini)
