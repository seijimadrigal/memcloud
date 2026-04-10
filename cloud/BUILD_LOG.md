# Memcloud — Build Log

## Sprint 1: REST API + WebSocket

### 2026-04-02 — Build & Deploy Complete ✅

**API Key:** `mc_d798cc892328f4e598803eac5f675cb1ad301fc16a78fd6e`
**API URL:** `https://api.memcloud.dev/v1/`
**Docs:** `https://api.memcloud.dev/docs`

### Test Results

| Endpoint | Status | Notes |
|----------|--------|-------|
| `GET /v1/health` | ✅ | postgres=true, redis=true, embedding_model=all-MiniLM-L6-v2 |
| `POST /v1/memories/` | ✅ | Extracted 10 memories from test text (4 triples, 1 summary, 4 profiles, 1 raw) |
| `GET /v1/memories/` | ✅ | Returns structured JSON with all memory types |
| `POST /v1/memories/search/` | ✅ | Hybrid search (BM25 + vector + graph), RRF fusion working |
| `PUT /v1/memories/{id}` | ✅ | Updates content + re-embeds |
| `DELETE /v1/memories/{id}` | ✅ | Deletes memory |
| `POST /v1/memories/answer/` | ✅ | Multi-hop CoT answering working |
| `WS /v1/ws` | ✅ | WebSocket endpoint with Redis pub/sub |
| Auth (Bearer token) | ✅ | API key validation + rate limiting via Redis |
| OpenAPI docs | ✅ | Swagger UI at /docs |

### Architecture
- **FastAPI** async API wrapping Memcloud extraction pipeline (5 parallel LLM calls)
- **PostgreSQL 16** for multi-tenant storage with FTS indexes
- **Redis 7** for rate limiting + WebSocket pub/sub broadcast
- **all-MiniLM-L6-v2** for vector embeddings (local, no API cost)
- **Hybrid search:** PostgreSQL FTS + vector cosine similarity + graph walk, RRF fusion
- **nginx** reverse proxy with WebSocket support

### Docker Containers
```
memcloud-api       — FastAPI (2 workers)
memcloud-postgres  — PostgreSQL 16 Alpine
memcloud-redis     — Redis 7 Alpine
memcloud-nginx     — nginx Alpine (reverse proxy)
```

### Files Created
- `app/main.py` — FastAPI app with all endpoints
- `app/config.py` — Environment config
- `app/database.py` — Async SQLAlchemy + PostgreSQL
- `app/models.py` — Multi-tenant models (Organization, ApiKey, Memory, Relation)
- `app/auth.py` — Bearer token auth + Redis rate limiting
- `app/schemas.py` — Pydantic request/response schemas
- `app/engine.py` — Core engine (extraction, hybrid search, RRF fusion, answer)
- `app/websocket.py` — WebSocket shared memory with Redis pub/sub
- `app/seed.py` — Database seeding script
- `docker-compose.yml` — API + PostgreSQL + Redis + nginx
- `Dockerfile` — Python 3.12 + sentence-transformers
- `nginx/nginx.conf` — Reverse proxy + WebSocket support
- `deploy.sh` — One-command deploy to VPS
- `requirements.txt` — Python dependencies

### Memory Scoping Model
```
Organization (org_id)
├── Users (user_id) → personal memories
├── Agents (agent_id) → agent-specific memories
└── Shared Pools (pool_id) → cross-agent memories
```

### Next Steps (Sprint 2)
- [ ] SSL/TLS with Let's Encrypt
- [ ] MCP Server for Claude Code / Cursor
- [ ] OpenClaw plugin (drop-in Cognee/Mem0 replacement)
- [ ] Python SDK (`pip install memcloud`)
- [ ] Domain setup (api.memcloud.dev)
