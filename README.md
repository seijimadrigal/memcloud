# MemChip

**The memory layer your AI agents are missing.**

MemChip is a self-hosted Memory-as-a-Service API that gives any AI agent persistent, accurate long-term memory. Store conversations, decisions, and facts once — recall them instantly across sessions, agents, and frameworks.

## Why MemChip?

AI agents forget everything between sessions. Context windows overflow. RAG retrieves documents, not memories. MemChip solves this with a purpose-built memory system:

- **Hybrid search** — BM25 full-text + pgvector semantic + knowledge graph + CrossEncoder reranking. Four retrieval signals fused via Reciprocal Rank Fusion, then reranked for deep relevance.
- **One-call context injection** — `POST /v1/recall` returns pre-assembled, token-budgeted context ready to inject into any agent's system prompt. No more gluing together search + profile + recent history yourself.
- **Automatic extraction** — Drop in raw text, get back structured triples, summaries, temporal events, and user profiles. Five parallel LLM calls extract everything meaningful.
- **Multi-agent collaboration** — Shared memory pools with ACL-based access control. Agent A stores a deployment note, Agent B finds it instantly. Projects and tasks scope memories automatically.
- **Conflict-aware** — When "user's favorite color is blue" meets "user now prefers green", MemChip detects the conflict, supersedes the old memory, and keeps full history.
- **Framework-agnostic** — REST API + Python SDK + TypeScript SDK + LangChain adapter + MCP server + OpenClaw plugin. Plug into anything.

## Architecture

```
Agent → POST /v1/recall → Hybrid Search → Reranking → Context Assembly → Structured Response
         ↓                   ↓                ↓
     Auto-extract       BM25 + pgvector   CrossEncoder
     (5 LLM calls)      + Graph Walk      (ms-marco)
         ↓                   ↓
     PostgreSQL 16       HNSW Index
     + pgvector          (768-dim)
```

**Search pipeline:**
1. **BM25** — PostgreSQL full-text search with GIN index
2. **pgvector** — Semantic similarity via `nomic-embed-text-v1.5` (768-dim, local, free)
3. **Knowledge graph** — 2-hop BFS walk across entity relations
4. **RRF fusion** — Merge all signals with type weighting + decay + importance
5. **CrossEncoder reranking** — `ms-marco-MiniLM-L-6-v2` rescores top-30 candidates

**Extraction pipeline** (on every `POST /v1/memories/`):
- Importance classification (0-5 score, noise filtered)
- Semantic triples (subject-predicate-object facts)
- Conversation summaries
- Temporal events (with absolute date resolution)
- User profile attributes (preferences, habits, skills)

## Quick Start

### 1. Deploy with Docker Compose

```bash
git clone https://github.com/seijimadrigal/memchip.git
cd memchip/cloud

# Create .env with your LLM API key
cat > .env << EOF
OPENROUTER_API_KEY=your_openrouter_key
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
EOF

docker compose up -d
```

This starts PostgreSQL 16 (pgvector), Redis, the API, and nginx. The embedding model downloads on first start (~530MB).

### 2. Seed the database

```bash
docker compose exec api python -m app.seed
```

This creates a default organization and API key.

### 3. Store a memory

```bash
curl -X POST http://localhost/v1/memories/ \
  -H "Authorization: Bearer mc_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "User prefers dark mode and vim keybindings. Uses Neovim with Lazy plugin manager.",
    "user_id": "seiji",
    "agent_id": "assistant"
  }'
```

The extraction pipeline runs automatically — this single call produces triples, a summary, profile attributes, and raw memory.

### 4. Search memories

```bash
curl -X POST http://localhost/v1/memories/search/ \
  -H "Authorization: Bearer mc_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What editor does the user prefer?",
    "user_id": "seiji",
    "top_k": 5
  }'
```

### 5. Get context for your agent (the key endpoint)

```bash
curl -X POST http://localhost/v1/recall \
  -H "Authorization: Bearer mc_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "seiji",
    "agent_id": "assistant",
    "query": "help me set up my dev environment",
    "token_budget": 4000,
    "format": "xml"
  }'
```

Returns pre-assembled context:

```xml
<memchip-recall>
<user-profile>
- seiji: editor_preference = Neovim with Lazy
- seiji: theme = dark mode
- seiji: keybindings = vim
</user-profile>
<relevant-memories>
- [triple] User prefers dark mode and vim keybindings
- [profile] seiji: plugin_manager = Lazy
</relevant-memories>
<recent-context>
- [summary] Discussed setting up Neovim config with LSP support
</recent-context>
</memchip-recall>
```

Inject this directly into your agent's system prompt. Supports `xml`, `markdown`, and `text` formats.

## SDK Usage

### Python

```bash
pip install httpx  # memchip SDK dependency
```

```python
from memchip.client import MemChipClient

mc = MemChipClient(
    api_key="mc_your_key",
    api_url="http://localhost/v1",
    user_id="seiji",
    agent_id="assistant"
)

# Store
mc.add("User decided to use PostgreSQL for the project")

# Search
results = mc.search("database choice")

# Get agent context (the key method)
context = mc.recall(query="what stack are we using?", token_budget=3000)
print(context["context"])  # ready to inject

# Answer from memory
answer = mc.answer("What database did we choose?")
```

### TypeScript

```typescript
import { MemChipClient } from 'memchip';

const mc = new MemChipClient({
  apiKey: 'mc_your_key',
  apiUrl: 'http://localhost/v1',
  userId: 'seiji',
  agentId: 'assistant'
});

await mc.add('User prefers TypeScript over JavaScript');
const context = await mc.recall({ query: 'language preferences', format: 'markdown' });
const results = await mc.search('preferences', { topK: 10 });
```

### LangChain

```python
from memchip.langchain import MemChipMemory
from langchain.chains import ConversationChain

memory = MemChipMemory(
    api_key="mc_your_key",
    api_url="http://localhost/v1",
    user_id="seiji",
    agent_id="assistant",
    auto_capture=True  # automatically stores each turn
)

chain = ConversationChain(memory=memory, llm=llm)
chain.run("What did we discuss last time?")
# MemChip recalls relevant context automatically
```

### MCP (Claude Code / Cursor)

Add to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "memchip": {
      "command": "python3",
      "args": [
        "path/to/mcp_server.py",
        "--api-url", "http://localhost/v1",
        "--api-key", "mc_your_key",
        "--user-id", "seiji"
      ]
    }
  }
}
```

Tools: `memory_store`, `memory_search`, `memory_answer`, `memory_list`, `memory_delete`

## API Reference

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/memories/` | Store a memory (triggers extraction) |
| `POST` | `/v1/memories/search/` | Hybrid search with reranking |
| `POST` | `/v1/memories/answer/` | Search + LLM answer synthesis |
| `POST` | `/v1/recall` | Pre-assembled context injection |
| `GET` | `/v1/memories/` | List memories with filters |
| `PUT` | `/v1/memories/{id}` | Update a memory |
| `DELETE` | `/v1/memories/{id}` | Soft delete (archive) |

### Memory Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/agents/{id}/profile/` | Dynamic memory profile (cached) |
| `GET` | `/v1/memories/{id}/history/` | Full changelog |
| `GET` | `/v1/memories/{id}/conflicts/` | Conflict chain |
| `POST` | `/v1/memories/{id}/resolve/` | Resolve a conflict |
| `POST` | `/v1/memories/bulk/import/` | Bulk import |
| `POST` | `/v1/memories/bulk/export/` | Bulk export |

### Multi-Agent

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/projects/` | Create project with shared pool |
| `POST` | `/v1/pools/access/` | Grant pool access to agent |
| `PUT` | `/v1/agents/{id}/context/` | Set agent's active project/task |
| `GET` | `/v1/graph/` | Knowledge graph visualization |

### Admin & Analytics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/health` | Health check (Postgres, Redis, model) |
| `GET` | `/v1/stats/` | Memory counts by type/agent/pool |
| `GET` | `/v1/analytics/` | Growth, activity, decay distribution |
| `GET` | `/v1/audit/` | Full audit trail |
| `POST` | `/v1/admin/keys/` | Generate per-agent API keys |
| `WS` | `/v1/ws` | Real-time memory events via WebSocket |

## Key Concepts

### Memory Types

| Type | Description | Example |
|------|-------------|---------|
| `triple` | Atomic subject-predicate-object fact | "seiji prefers dark mode" |
| `summary` | Concise session overview | "Discussed PostgreSQL migration strategy" |
| `profile` | Key-value attribute about a person | "seiji: role = software engineer" |
| `temporal` | Time-anchored event | "Deployed v2.0 on 2026-04-05" |
| `raw` | Verbatim text preserved for context | Full conversation snippet |

### Memory Scopes

| Scope | Visibility |
|-------|-----------|
| `private` | Only the creating agent |
| `task` | Agents on the same task |
| `project` | All agents on a project |
| `team` | All agents of a user |
| `global` | All agents system-wide |

### Conflict Detection

When a new memory contradicts an existing one:
- **Supersede** (similarity >= 0.70) — new replaces old, history preserved
- **Chain** (similarity 0.60-0.70) — both kept, linked as evolution
- **New** (similarity < 0.60) — stored independently

### Importance Scoring

Every memory is auto-scored 0-5. Low-importance content (score < 1) is filtered at write time:

| Score | Level | Example |
|-------|-------|---------|
| 0 | Noise | "ok", "thanks" |
| 1 | Operational | "running npm install" |
| 2 | Routine | "checked the logs" |
| 3 | Significant | "user prefers dark mode" |
| 4 | Critical | "decided to use PostgreSQL" |
| 5 | Foundational | "project architecture chosen" |

## Infrastructure

### Requirements
- Docker + Docker Compose
- 4+ CPU cores, 4GB+ RAM (8GB+ recommended)
- An LLM API key (OpenRouter, OpenAI, or Anthropic) for extraction

### Stack
- **PostgreSQL 16** with pgvector for hybrid search
- **Redis 7** for rate limiting, caching, WebSocket pub/sub
- **FastAPI** with uvicorn (2 workers)
- **nginx** reverse proxy with rate limiting
- **nomic-embed-text-v1.5** local embeddings (768-dim, free, no API key needed)
- **CrossEncoder ms-marco-MiniLM-L-6-v2** local reranking (22M params, CPU-friendly)

### Resource Usage
- API container: ~400MB RAM (model + app)
- PostgreSQL: ~200MB RAM
- Redis: ~5MB RAM
- Total: ~700MB RAM for the full stack

## License

MIT

## Contributing

Issues and PRs welcome at [github.com/seijimadrigal/memchip](https://github.com/seijimadrigal/memchip).
