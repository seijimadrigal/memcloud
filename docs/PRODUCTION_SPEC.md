# MemChip Production Architecture Spec

## Product Vision
Memory-as-a-Service for AI agents. Drop-in API that gives any LLM agent persistent, accurate long-term memory with cross-agent sharing.

**Tagline:** "The memory layer your AI agents are missing."

**Key differentiators:**
1. 85%+ LoCoMo accuracy (vs Mem0's 67%, Zep's 75%)
2. Multi-agent shared memory (nobody does this well)
3. Two-pass verified answers (reduces hallucination)
4. Open-source core + hosted cloud

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    API Gateway                       │
│            (FastAPI + rate limiting)                  │
│         POST /retain  POST /recall  POST /reflect    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ Ingestion│  │ Retrieval│  │  Answer Generation │  │
│  │ Pipeline │  │  Engine  │  │   (Two-Pass)       │  │
│  └────┬─────┘  └────┬─────┘  └────────┬──────────┘  │
│       │              │                  │             │
│  ┌────┴──────────────┴──────────────────┴──────────┐ │
│  │              Storage Layer                       │ │
│  │  PostgreSQL + pgvector + Redis Cache             │ │
│  └──────────────────────────────────────────────────┘ │
│                                                      │
│  ┌──────────────────────────────────────────────────┐ │
│  │           ML Models (Self-Hosted)                │ │
│  │  CrossEncoder (reranking) | LLM (answering)      │ │
│  │  Embedding Model (optional) | Entity Resolution  │ │
│  └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## API Design

### Core Endpoints

#### `POST /v1/retain` — Store memories
```json
{
  "bank_id": "user_123",
  "agent_id": "support_agent",
  "messages": [
    {"role": "user", "content": "I moved from Sweden last year", "timestamp": "2024-03-15T10:00:00Z"},
    {"role": "assistant", "content": "Welcome! How are you settling in?", "timestamp": "2024-03-15T10:00:05Z"}
  ],
  "metadata": {
    "session_id": "sess_abc",
    "speakers": {"user": "Caroline", "assistant": "Support Bot"}
  }
}
```

Response:
```json
{
  "status": "ok",
  "facts_extracted": 3,
  "entities_resolved": ["Caroline"],
  "processing_time_ms": 1200
}
```

#### `POST /v1/recall` — Query memories
```json
{
  "bank_id": "user_123",
  "query": "Where did Caroline move from?",
  "agent_id": "support_agent",
  "options": {
    "max_results": 10,
    "include_sources": true,
    "verified": true
  }
}
```

Response:
```json
{
  "answer": "Sweden",
  "confidence": 0.95,
  "sources": [
    {"fact": "Caroline moved from Sweden", "date": "2024-03-15", "session": "sess_abc"}
  ],
  "strategy": "v18_single_hop",
  "latency_ms": 850
}
```

#### `POST /v1/reflect` — Agentic reasoning over memories
```json
{
  "bank_id": "user_123",
  "query": "What would Caroline likely enjoy as a gift?",
  "agent_id": "personal_agent",
  "context": "Her birthday is coming up"
}
```

Response:
```json
{
  "answer": "Based on her interests in pottery and painting, art supplies or a pottery workshop voucher would be a good fit.",
  "reasoning": ["Caroline enjoys pottery workshops", "Caroline paints sunsets and nature scenes", "She values creative activities with family"],
  "confidence": 0.8
}
```

### Multi-Agent Endpoints

#### `POST /v1/banks` — Create memory bank
```json
{
  "bank_id": "user_123",
  "name": "Caroline's Memory",
  "sharing": {
    "mode": "shared_read",
    "agents": ["support_agent", "sales_agent", "personal_agent"],
    "write_agents": ["support_agent", "personal_agent"]
  }
}
```

#### `GET /v1/banks/{bank_id}/agents` — List agents with access
#### `POST /v1/banks/{bank_id}/permissions` — Update sharing rules

### Memory Management

#### `GET /v1/banks/{bank_id}/facts` — List stored facts
#### `DELETE /v1/banks/{bank_id}/facts/{fact_id}` — Delete a fact (GDPR)
#### `POST /v1/banks/{bank_id}/consolidate` — Force profile rebuild

---

## Database Schema (PostgreSQL)

```sql
-- Memory banks (one per user/entity)
CREATE TABLE banks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255) UNIQUE NOT NULL,  -- customer's bank_id
    org_id UUID NOT NULL REFERENCES orgs(id),
    name VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    settings JSONB DEFAULT '{}'
);

-- Agents with access to banks
CREATE TABLE bank_agents (
    bank_id UUID REFERENCES banks(id),
    agent_id VARCHAR(255) NOT NULL,
    permissions VARCHAR(20) DEFAULT 'read',  -- read, write, admin
    PRIMARY KEY (bank_id, agent_id)
);

-- Raw conversation chunks (for reranking retrieval)
CREATE TABLE chunks (
    id BIGSERIAL PRIMARY KEY,
    bank_id UUID REFERENCES banks(id),
    session_id VARCHAR(255),
    date DATE,
    text TEXT NOT NULL,
    chunk_index INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_chunks_bank ON chunks(bank_id);
-- FTS index
ALTER TABLE chunks ADD COLUMN tsv tsvector 
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX idx_chunks_fts ON chunks USING GIN(tsv);

-- Atomic facts (structured extraction)
CREATE TABLE facts (
    id BIGSERIAL PRIMARY KEY,
    bank_id UUID REFERENCES banks(id),
    entity VARCHAR(255),           -- resolved entity name
    fact_text TEXT NOT NULL,
    subject VARCHAR(255),
    predicate VARCHAR(255),        -- optional structured triple
    object TEXT,                    -- optional structured triple
    source_session VARCHAR(255),
    source_date DATE,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(255)        -- which agent stored this
);
CREATE INDEX idx_facts_bank_entity ON facts(bank_id, entity);
-- FTS on facts
ALTER TABLE facts ADD COLUMN tsv tsvector 
    GENERATED ALWAYS AS (to_tsvector('english', fact_text)) STORED;
CREATE INDEX idx_facts_fts ON facts USING GIN(tsv);

-- Entity profiles (consolidated summaries)
CREATE TABLE profiles (
    id BIGSERIAL PRIMARY KEY,
    bank_id UUID REFERENCES banks(id),
    entity VARCHAR(255) NOT NULL,
    profile_text TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(bank_id, entity)
);

-- Episode summaries
CREATE TABLE episodes (
    id BIGSERIAL PRIMARY KEY,
    bank_id UUID REFERENCES banks(id),
    session_id VARCHAR(255),
    date DATE,
    date_iso VARCHAR(10),
    summary TEXT,
    key_entities TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_episodes_bank ON episodes(bank_id);

-- Temporal events
CREATE TABLE temporal_events (
    id BIGSERIAL PRIMARY KEY,
    bank_id UUID REFERENCES banks(id),
    session_id VARCHAR(255),
    entity VARCHAR(255),
    event_text TEXT,
    event_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_temporal_bank_date ON temporal_events(bank_id, event_date);

-- Entity resolution mapping
CREATE TABLE entity_aliases (
    id BIGSERIAL PRIMARY KEY,
    bank_id UUID REFERENCES banks(id),
    alias VARCHAR(255) NOT NULL,        -- "Alice C.", "Alice Chen"
    canonical VARCHAR(255) NOT NULL,     -- "Alice Chen"
    confidence FLOAT DEFAULT 1.0,
    UNIQUE(bank_id, alias)
);

-- Audit log (for GDPR compliance)
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    bank_id UUID,
    agent_id VARCHAR(255),
    action VARCHAR(50),  -- retain, recall, delete, share
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Ingestion Pipeline (Retain)

```
Input: conversation messages
    │
    ▼
1. Chunk raw text (250 words, 50 overlap)
    → store in `chunks` table
    │
    ▼
2. Extract atomic facts (LLM call)
    → subject/predicate/object triples
    → store in `facts` table
    │
    ▼
3. Entity resolution
    → match "Alice", "Alice Chen" → canonical "Alice Chen"
    → store/update `entity_aliases`
    │
    ▼
4. Build episode summary (LLM call)
    → store in `episodes` table
    │
    ▼
5. Extract temporal events (LLM call)
    → store in `temporal_events` table
    │
    ▼
6. Update entity profiles (async, background)
    → append new facts to existing profile
    → store in `profiles` table
```

**Async processing:** Steps 2-6 run in a Celery/Redis task queue. `retain()` returns immediately after step 1 (chunking). Client can poll for completion or use webhook.

---

## Retrieval Engine (Recall)

```
Input: query + bank_id
    │
    ▼
1. Entity extraction from query
    → identify target entity via alias table
    │
    ▼
2. Parallel retrieval (3 strategies)
    ├─ FTS5/tsvector keyword search (chunks + facts)
    ├─ Entity-specific facts (all facts for target entity)
    └─ Temporal events (if time-related query detected)
    │
    ▼
3. CrossEncoder reranking
    → score all candidates against query
    → score-adaptive truncation
    │
    ▼
4. Answer generation (Pass 1)
    → LLM answers from top chunks + entity facts + profile
    │
    ▼
5. Verification (Pass 2, for list-type questions)
    → LLM filters answer to only items matching question scope
    │
    ▼
Output: verified answer + sources + confidence
```

---

## Multi-Agent Memory Sharing

### Sharing Modes

1. **Private** — Only the creating agent can read/write
2. **Shared Read** — Multiple agents read, one writes
3. **Shared Read/Write** — Multiple agents read and write
4. **Federated** — Each agent has private namespace + shared pool

### How It Works

```
Bank: user_123
├── Shared Pool (all agents can read)
│   ├── fact: "Caroline moved from Sweden"
│   ├── fact: "Caroline likes pottery"
│   └── profile: "Caroline — identity, interests..."
│
├── Agent: support_agent (private)
│   ├── fact: "Ticket #1234 resolved by restart"
│   └── fact: "Caroline prefers email contact"
│
└── Agent: sales_agent (private)
    ├── fact: "Caroline viewed pricing page 3x"
    └── fact: "Offered 20% discount on March 15"
```

When `sales_agent` calls `recall("What do we know about Caroline?")`:
- Gets shared pool facts + own private facts
- Does NOT see support_agent's private facts

### Cross-Agent Queries

```json
POST /v1/recall
{
  "bank_id": "user_123",
  "query": "Has Caroline had any support issues?",
  "agent_id": "sales_agent",
  "cross_agent": true,
  "request_agents": ["support_agent"]
}
```

This triggers a permission check → if support_agent allows sharing with sales_agent → returns relevant facts.

---

## Infrastructure

### Minimum Production Deploy

| Component | Service | Cost/mo |
|---|---|---|
| API + Workers | 2x c6g.large (ARM) | $120 |
| PostgreSQL | RDS db.r6g.large | $200 |
| Redis | ElastiCache t4g.medium | $50 |
| GPU (reranker) | 1x g5.xlarge (A10G) | $500 |
| LLM inference | OpenRouter or self-hosted | $100-500 |
| **Total** | | **$970-1,370/mo** |

### Handles
- ~500 concurrent users
- ~50K queries/day
- ~5K retain operations/day

### Scaling Path
- PostgreSQL → read replicas for recall, write primary for retain
- GPU → add instances behind load balancer
- LLM → batch API or self-hosted vLLM on A100 for 10x throughput
- Redis → cluster mode for >10K concurrent

---

## SDK (Python)

```python
from memchip import MemChip

mc = MemChip(api_key="mc_...", base_url="https://api.memchip.ai")

# Create a memory bank
bank = mc.banks.create(bank_id="user_123", name="Caroline")

# Store conversation
bank.retain(
    messages=[
        {"role": "user", "content": "I moved from Sweden last year"},
        {"role": "assistant", "content": "Welcome!"}
    ],
    agent_id="support_agent",
    session_id="sess_abc"
)

# Query memories
result = bank.recall("Where did Caroline move from?", agent_id="support_agent")
print(result.answer)      # "Sweden"
print(result.confidence)   # 0.95
print(result.sources)      # [{"fact": "Caroline moved from Sweden", ...}]

# Agentic reasoning
result = bank.reflect(
    "What gift would Caroline like?",
    agent_id="personal_agent"
)
print(result.answer)       # "Art supplies or pottery workshop voucher"
print(result.reasoning)    # ["Enjoys pottery", "Paints sunsets", ...]

# Multi-agent sharing
bank.permissions.grant("sales_agent", "read")
bank.permissions.grant("support_agent", "read_write")
```

---

## Pricing Model

| Tier | Price | Queries/mo | Banks | Agents | Retain ops |
|---|---|---|---|---|---|
| Free | $0 | 1,000 | 5 | 2 | 100 |
| Pro | $29/mo | 50,000 | 100 | 10 | 5,000 |
| Business | $99/mo | 500,000 | 1,000 | 50 | 50,000 |
| Enterprise | Custom | Unlimited | Unlimited | Unlimited | Unlimited |

**Usage-based overage:** $0.001/query, $0.01/retain after tier limit.

---

## Competitive Positioning

| Feature | MemChip | Mem0 | Zep | Backboard | Hindsight |
|---|---|---|---|---|---|
| LoCoMo Score | 85%+ | 67% | 75% | 90% | 89.6% |
| Multi-agent memory | ✅ Native | ❌ | ❌ | ❌ | ❌ |
| Verified answers | ✅ Two-pass | ❌ | ❌ | ❌ | ❌ |
| Open source core | ✅ | ✅ | Partial | ❌ | ✅ |
| Self-hostable | ✅ | ✅ | ✅ | ❌ | ✅ |
| Entity resolution | ✅ | ❌ | ✅ | Unknown | ✅ |
| Temporal reasoning | ✅ 95% | 55% | 80% | 92% | 91% |
| GDPR delete | ✅ | ✅ | ✅ | Unknown | ❌ |

---

## Roadmap

### Phase 1: MVP (2 weeks)
- [ ] FastAPI server with /retain, /recall endpoints
- [ ] PostgreSQL storage (migrate from SQLite)
- [ ] Python SDK
- [ ] Docker Compose deploy
- [ ] Basic auth (API keys)

### Phase 2: Multi-Agent (2 weeks)
- [ ] Memory bank sharing
- [ ] Agent permissions
- [ ] Cross-agent recall
- [ ] Entity resolution across agents

### Phase 3: Production Hardening (2 weeks)
- [ ] Rate limiting, usage metering
- [ ] Webhook for async retain completion
- [ ] Dashboard (bank explorer, fact viewer)
- [ ] Stripe billing integration

### Phase 4: Scale + Launch (2 weeks)
- [ ] Self-hosted LLM option (vLLM)
- [ ] Benchmark page (LoCoMo results)
- [ ] Documentation site
- [ ] HuggingFace model card + open source release
- [ ] Product Hunt launch

---

## Open Source Strategy

**Core (MIT License):**
- Storage layer, retrieval engine, ingestion pipeline
- CLI tool for local use
- Python SDK

**Cloud-only (proprietary):**
- Multi-tenant infrastructure
- Dashboard + analytics
- Managed GPU inference
- SLA + support

This follows the Hindsight/Mem0 model — open source drives adoption, cloud drives revenue.
