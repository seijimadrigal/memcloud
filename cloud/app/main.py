"""Memcloud API — FastAPI application. v1.0.0"""
import asyncio
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sa_text, func as sa_func, select, delete, or_

from app.database import get_db, init_db
from app.auth import authenticate, AuthContext
from app.schemas import (
    MemoryAddRequest, MemoryAddResponse, MemorySearchRequest, MemorySearchResponse,
    MemoryUpdateRequest, MemoryAnswerRequest, MemoryAnswerResponse,
    MemoryResponse, HealthResponse,
    SessionCreateRequest, SessionResponse,
    PoolAccessCreateRequest, PoolAccessResponse,
    WebhookCreateRequest, WebhookResponse,
    SchemaCreateRequest, SchemaResponse,
    InstructionCreateRequest, InstructionResponse,
    BulkImportRequest, BulkImportResponse, BulkExportRequest, BulkExportResponse,
    BulkDeleteRequest, BulkDeleteResponse,
    DecayCleanupRequest, DecayCleanupResponse, DecayPreviewResponse, DecayPreviewItem,
    AuditEntry, AuditListResponse,
    AnalyticsResponse,
    # v0.3.0
    MemoryEventResponse, EventListResponse,
    ConflictResolveRequest,
    SubscriptionCreateRequest, SubscriptionResponse,
    # v0.4.0
    ProjectCreateRequest, ProjectUpdateRequest, ProjectResponse, ProjectDetailResponse,
    AgentContextUpdateRequest, AgentContextResponse,
    # Assistant
    AssistantChatRequest, AssistantChatResponse, AssistantSource,
)
from app.engine import (
    add_memory, search_memories, answer_question, list_memories,
    get_memory, update_memory, delete_memory,
    decay_cleanup, decay_preview, get_chain, log_audit,
    # v0.3.0
    get_memory_history, list_events, get_memory_conflicts, resolve_conflict,
    create_subscription, list_subscriptions, delete_subscription,
    # v0.4.0
    create_project, get_project, list_projects, update_project, archive_project,
    get_memory_count_for_pool, get_recent_memories_for_pool,
    get_agent_context, set_agent_context, clear_agent_context,
    # v1.0.0
    recall_context,
)
from app.models import (
    Memory, MemorySession, PoolAccess, Webhook,
    MemorySchema, MemoryInstruction, MemoryAudit,
    MemoryEvent, MemorySubscription,
    Project, AgentContext,
)
from app.websocket import websocket_endpoint, manager
from app.config import EMBEDDING_MODEL, REDIS_URL
import redis.asyncio as aioredis


# --- In-memory request log (last 10K entries) ---
request_log: deque = deque(maxlen=10000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    # Start WebSocket Redis subscriber in background
    task = asyncio.create_task(manager.start_subscriber())
    yield
    # Shutdown
    task.cancel()


app = FastAPI(
    title="Memcloud API",
    description="Memory-as-a-service for AI agents. Hybrid search, multi-hop reasoning, real-time shared memory.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request logging middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = round((time.time() - start) * 1000, 2)
    
    path = request.url.path
    if path.startswith("/v1/memories"):
        req_type = "UNKNOWN"
        if request.method == "POST" and "search" in path:
            req_type = "SEARCH"
        elif request.method == "POST" and "answer" in path:
            req_type = "ANSWER"
        elif request.method == "POST":
            req_type = "ADD"
        elif request.method == "GET":
            req_type = "LIST"
        elif request.method == "PUT":
            req_type = "UPDATE"
        elif request.method == "DELETE":
            req_type = "DELETE"
        
        request_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": req_type,
            "path": path,
            "method": request.method,
            "status": response.status_code,
            "latency_ms": elapsed_ms,
            "user_agent": request.headers.get("user-agent", "")[:100],
        })
    
    return response


# --- Health ---

@app.get("/v1/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)):
    pg_ok = False
    redis_ok = False
    try:
        await db.execute(sa_text("SELECT 1"))
        pg_ok = True
    except Exception:
        pass
    try:
        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        await r.ping()
        redis_ok = True
        await r.aclose()
    except Exception:
        pass
    return HealthResponse(
        status="ok" if pg_ok and redis_ok else "degraded",
        version="1.0.0",
        postgres=pg_ok,
        redis=redis_ok,
        pgvector=True,
        embedding_model=EMBEDDING_MODEL,
    )


# --- Memory CRUD ---

@app.post("/v1/memories/", response_model=MemoryAddResponse)
async def api_add_memory(
    req: MemoryAddRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    result = await add_memory(
        db, org_id=auth.org_id, text=req.text, user_id=req.user_id,
        agent_id=req.agent_id, pool_id=req.pool_id,
        session_id=req.session_id, metadata=req.metadata,
        schema_id=req.schema_id, session_id_fk=req.session_id_fk,
        scope=req.scope or "private",
        source_type=req.source_type,
        source_ref=req.source_ref,
    )
    # Broadcast via WebSocket
    pool = req.pool_id or f"user:{req.user_id}"
    await manager.publish_event(pool, "memory.added", {
        "user_id": req.user_id, "agent_id": req.agent_id,
        "memories_created": result["memories_created"],
    })
    return MemoryAddResponse(**result)


@app.post("/v1/memories/search/", response_model=MemorySearchResponse)
async def api_search_memories(
    req: MemorySearchRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    return await search_memories(
        db, org_id=auth.org_id, query=req.query, user_id=req.user_id,
        agent_id=req.agent_id, pool_id=req.pool_id,
        search_scope=req.search_scope, top_k=req.top_k, agentic=req.agentic,
    )


@app.get("/v1/memories/")
async def api_list_memories(
    user_id: str = Query(...),
    agent_id: str = Query(None),
    pool_id: str = Query(None),
    memory_type: str = Query(None),
    category: str = Query(None, description="Deprecated — categories removed in v0.4.1"),
    scope: str = Query(None, description="Filter by scope (private, task, project, team, global)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    memories = await list_memories(
        db, org_id=auth.org_id, user_id=user_id, agent_id=agent_id,
        pool_id=pool_id, memory_type=memory_type, category=category,
        scope=scope, limit=limit, offset=offset,
    )
    return [
        {
            "id": m.id, "memory_type": m.memory_type, "content": m.content,
            "structured_data": m.structured_data, "user_id": m.user_id,
            "agent_id": m.agent_id, "pool_id": m.pool_id,
            "confidence": m.confidence, "decay_score": m.decay_score,
            "access_count": m.access_count, "chain_id": m.chain_id,
            "created_at": str(m.created_at), "updated_at": str(m.updated_at),
            # v0.3.0
            "status": m.status, "scope": m.scope,
            "conflict_status": m.conflict_status, "version": m.version,
            "source_type": m.source_type, "source_ref": m.source_ref,
            "derived_from": m.derived_from, "supersedes_id": m.supersedes_id,
            "importance": m.importance,
        }
        for m in memories
    ]


@app.get("/v1/memories/categories/")
async def api_list_categories(
    user_id: str = Query(None),
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """List all unique categories with counts."""
    q = sa_text("""
        SELECT json_array_elements_text(categories) as cat, count(*) as cnt
        FROM memories WHERE org_id = :org AND status IN ('active', 'archived')
        GROUP BY cat ORDER BY cnt DESC
    """)
    rows = (await db.execute(q, {"org": auth.org_id})).fetchall()
    return [{"category": r[0], "count": r[1]} for r in rows]


@app.get("/v1/memories/agents/")
async def api_list_agents(
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """List all unique agent_ids with counts."""
    q = sa_text("""
        SELECT COALESCE(agent_id, '') as agent, count(*) as cnt
        FROM memories WHERE org_id = :org AND status IN ('active', 'archived')
        GROUP BY agent_id ORDER BY cnt DESC
    """)
    rows = (await db.execute(q, {"org": auth.org_id})).fetchall()
    return [{"agent_id": r[0] or None, "count": r[1]} for r in rows]


@app.put("/v1/memories/{memory_id}")
async def api_update_memory(
    memory_id: str,
    req: MemoryUpdateRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    mem = await update_memory(db, memory_id, auth.org_id, content=req.content, metadata=req.metadata)
    if not mem:
        raise HTTPException(404, "Memory not found")
    await manager.publish_event(mem.pool_id or f"user:{mem.user_id}", "memory.updated", {"id": memory_id})
    return {"status": "ok", "id": mem.id}


@app.delete("/v1/memories/{memory_id}")
async def api_delete_memory(
    memory_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    ok = await delete_memory(db, memory_id, auth.org_id)
    if not ok:
        raise HTTPException(404, "Memory not found")
    return {"status": "ok", "deleted": memory_id}


@app.post("/v1/memories/answer/", response_model=MemoryAnswerResponse)
async def api_answer(
    req: MemoryAnswerRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    return await answer_question(
        db, org_id=auth.org_id, question=req.question,
        user_id=req.user_id, agent_id=req.agent_id, agentic=req.agentic,
    )


# ========== v0.3.0: Event Log Endpoints ==========

@app.get("/v1/memories/{memory_id}/history/", response_model=EventListResponse)
async def api_memory_history(
    memory_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """Get full changelog for a memory."""
    events = await get_memory_history(db, memory_id, auth.org_id)
    return EventListResponse(
        events=[
            MemoryEventResponse(
                id=e.id, org_id=e.org_id, memory_id=e.memory_id,
                event_type=e.event_type, actor_id=e.actor_id,
                actor_type=e.actor_type, source=e.source,
                previous_content=e.previous_content,
                new_content=e.new_content,
                metadata=e.metadata_,
                created_at=e.created_at,
            )
            for e in events
        ],
        total=len(events),
    )


@app.get("/v1/events/", response_model=EventListResponse)
async def api_list_events(
    event_type: str = Query(None),
    actor_id: str = Query(None),
    memory_id: str = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """List recent events across all memories."""
    events, total = await list_events(
        db, auth.org_id,
        event_type=event_type, actor_id=actor_id,
        memory_id=memory_id, limit=limit, offset=offset,
    )
    return EventListResponse(
        events=[
            MemoryEventResponse(
                id=e.id, org_id=e.org_id, memory_id=e.memory_id,
                event_type=e.event_type, actor_id=e.actor_id,
                actor_type=e.actor_type, source=e.source,
                previous_content=e.previous_content,
                new_content=e.new_content,
                metadata=e.metadata_,
                created_at=e.created_at,
            )
            for e in events
        ],
        total=total,
    )


# ========== v0.3.0: Conflict Endpoints ==========

@app.get("/v1/memories/{memory_id}/conflicts/")
async def api_memory_conflicts(
    memory_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """Get memories that supersede or are superseded by this memory."""
    conflicts = await get_memory_conflicts(db, memory_id, auth.org_id)
    return [
        {
            "id": m.id, "memory_type": m.memory_type, "content": m.content,
            "conflict_status": m.conflict_status, "version": m.version,
            "supersedes_id": m.supersedes_id,
            "created_at": str(m.created_at),
        }
        for m in conflicts
    ]


@app.post("/v1/memories/{memory_id}/resolve/")
async def api_resolve_conflict(
    memory_id: str,
    req: ConflictResolveRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """Manually resolve a memory conflict."""
    mem = await resolve_conflict(
        db, memory_id, auth.org_id,
        resolution=req.resolution.value,
        merged_content=req.merged_content,
    )
    if not mem:
        raise HTTPException(404, "Memory not found")
    return {"status": "ok", "id": mem.id, "conflict_status": mem.conflict_status}


# ========== v0.3.0: Subscription Endpoints ==========

@app.post("/v1/subscriptions/", response_model=SubscriptionResponse)
async def api_create_subscription(
    req: SubscriptionCreateRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    sub = await create_subscription(
        db, auth.org_id, req.agent_id,
        scope_filter=req.scope_filter,
        pool_filter=req.pool_filter,
        category_filter=req.category_filter,
        event_types=req.event_types,
    )
    return sub


@app.get("/v1/subscriptions/")
async def api_list_subscriptions(
    agent_id: str = Query(...),
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    subs = await list_subscriptions(db, auth.org_id, agent_id)
    return [
        {
            "id": s.id, "org_id": s.org_id, "agent_id": s.agent_id,
            "scope_filter": s.scope_filter, "pool_filter": s.pool_filter,
            "category_filter": s.category_filter, "event_types": s.event_types,
            "is_active": s.is_active, "created_at": str(s.created_at),
        }
        for s in subs
    ]


@app.delete("/v1/subscriptions/{subscription_id}")
async def api_delete_subscription(
    subscription_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    ok = await delete_subscription(db, subscription_id, auth.org_id)
    if not ok:
        raise HTTPException(404, "Subscription not found")
    return {"status": "ok", "deleted": subscription_id}


# ========== v0.4.0: Project Endpoints ==========

@app.post("/v1/projects/", response_model=ProjectResponse)
async def api_create_project(
    req: ProjectCreateRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    slug = req.slug or re.sub(r'[^a-z0-9]+', '-', req.name.lower()).strip('-')
    project = await create_project(
        db, org_id=auth.org_id,
        name=req.name, slug=slug,
        description=req.description,
        agents=req.agents,
        metadata=req.metadata,
    )
    memory_count = await get_memory_count_for_pool(db, auth.org_id, project.pool_id)
    return ProjectResponse(
        id=project.id, org_id=project.org_id,
        name=project.name, slug=project.slug,
        pool_id=project.pool_id, description=project.description,
        agents=project.agents or [], status=project.status,
        metadata=project.metadata_,
        memory_count=memory_count,
        created_at=project.created_at, updated_at=project.updated_at,
    )


@app.get("/v1/projects/", response_model=list[ProjectResponse])
async def api_list_projects(
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    projects = await list_projects(db, auth.org_id)
    results = []
    for p in projects:
        memory_count = await get_memory_count_for_pool(db, auth.org_id, p.pool_id)
        results.append(ProjectResponse(
            id=p.id, org_id=p.org_id,
            name=p.name, slug=p.slug,
            pool_id=p.pool_id, description=p.description,
            agents=p.agents or [], status=p.status,
            metadata=p.metadata_,
            memory_count=memory_count,
            created_at=p.created_at, updated_at=p.updated_at,
        ))
    return results


@app.get("/v1/projects/{project_id}", response_model=ProjectDetailResponse)
async def api_get_project(
    project_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    project = await get_project(db, project_id)
    if not project or project.org_id != auth.org_id:
        raise HTTPException(404, "Project not found")
    memory_count = await get_memory_count_for_pool(db, auth.org_id, project.pool_id)
    recent = await get_recent_memories_for_pool(db, auth.org_id, project.pool_id, limit=10)
    recent_memories = [
        {
            "id": m.id, "content": m.content, "memory_type": m.memory_type,
            "agent_id": m.agent_id, "scope": m.scope,
            "created_at": str(m.created_at),
        }
        for m in recent
    ]
    return ProjectDetailResponse(
        id=project.id, org_id=project.org_id,
        name=project.name, slug=project.slug,
        pool_id=project.pool_id, description=project.description,
        agents=project.agents or [], status=project.status,
        metadata=project.metadata_,
        memory_count=memory_count,
        recent_memories=recent_memories,
        created_at=project.created_at, updated_at=project.updated_at,
    )


@app.put("/v1/projects/{project_id}", response_model=ProjectResponse)
async def api_update_project(
    project_id: str,
    req: ProjectUpdateRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    project = await update_project(
        db, project_id, auth.org_id,
        name=req.name, description=req.description,
        agents=req.agents, status=req.status,
        metadata=req.metadata,
    )
    if not project:
        raise HTTPException(404, "Project not found")
    memory_count = await get_memory_count_for_pool(db, auth.org_id, project.pool_id)
    return ProjectResponse(
        id=project.id, org_id=project.org_id,
        name=project.name, slug=project.slug,
        pool_id=project.pool_id, description=project.description,
        agents=project.agents or [], status=project.status,
        metadata=project.metadata_,
        memory_count=memory_count,
        created_at=project.created_at, updated_at=project.updated_at,
    )


@app.delete("/v1/projects/{project_id}")
async def api_delete_project(
    project_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    project = await archive_project(db, project_id, auth.org_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return {"status": "ok", "id": project.id, "new_status": "archived"}


# ========== v0.4.0: Agent Context Endpoints ==========

@app.get("/v1/agents/{agent_id}/context/", response_model=AgentContextResponse)
async def api_get_agent_context(
    agent_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    ctx = await get_agent_context(db, auth.org_id, agent_id)
    if not ctx:
        # Return default context
        return AgentContextResponse(
            id="", org_id=auth.org_id, agent_id=agent_id,
            active_project_id=None, active_task_id=None,
            default_scope="team", default_pool_id=None,
            updated_at=datetime.utcnow(),
        )
    return AgentContextResponse(
        id=ctx.id, org_id=ctx.org_id, agent_id=ctx.agent_id,
        active_project_id=ctx.active_project_id,
        active_task_id=ctx.active_task_id,
        default_scope=ctx.default_scope,
        default_pool_id=ctx.default_pool_id,
        updated_at=ctx.updated_at,
    )


@app.put("/v1/agents/{agent_id}/context/", response_model=AgentContextResponse)
async def api_set_agent_context(
    agent_id: str,
    req: AgentContextUpdateRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    ctx = await set_agent_context(
        db, auth.org_id, agent_id,
        active_project_id=req.active_project_id,
        active_task_id=req.active_task_id,
        default_scope=req.default_scope,
        default_pool_id=req.default_pool_id,
    )
    return AgentContextResponse(
        id=ctx.id, org_id=ctx.org_id, agent_id=ctx.agent_id,
        active_project_id=ctx.active_project_id,
        active_task_id=ctx.active_task_id,
        default_scope=ctx.default_scope,
        default_pool_id=ctx.default_pool_id,
        updated_at=ctx.updated_at,
    )


@app.delete("/v1/agents/{agent_id}/context/")
async def api_clear_agent_context(
    agent_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    ok = await clear_agent_context(db, auth.org_id, agent_id)
    return {"status": "ok", "cleared": ok}


# ========== Assistant Chat ==========

MEMCLOUD_SYSTEM_PROMPT = """You are Memcloud Assistant — an AI helper for the Memcloud memory-as-a-service platform.

## What You Can Do
- Answer questions about the user's stored memories based on retrieved context below
- Explain how to use Memcloud (API, concepts, configuration)
- Show retrieval transparency — explain why certain memories were found

## Memcloud Knowledge Base
Memcloud is a memory service for AI agents with these core concepts:
- **Memories**: Extracted knowledge units (triples, summaries, profiles, temporal events)
- **Scopes**: private (single agent), task (task-scoped), project (project-scoped), team (org-wide), global (cross-org)
- **Pools**: Shared memory namespaces (e.g. "shared:team", "project:my-project"). Agents get pool access via ACL.
- **Extraction Types**: raw, triple (subject-predicate-object), summary, temporal, profile
- **Auto-capture**: The OpenClaw plugin intercepts conversations and auto-stores memories via POST /v1/memories/
- **Search**: Hybrid retrieval — BM25 full-text + vector similarity + knowledge graph walk, fused via RRF
- **Decay**: Memories decay over time based on access frequency; low-decay memories get archived
- **Projects**: Organize memories into projects (with auto-generated pools)
- **Agent Context**: Agents can set active project for auto-routing of new memories

## API Quick Reference
- POST /v1/memories/ — Add memory (extracts triples, summaries, etc.)
- POST /v1/memories/search/ — Hybrid search
- POST /v1/memories/answer/ — Search + LLM answer
- GET /v1/memories/ — List with filters
- POST /v1/projects/ — Create project (auto-generates pool)
- PUT /v1/agents/{id}/context/ — Set agent's active project

## Plugin Configuration (OpenClaw)
```json
{
  "name": "@memcloud/openclaw-memcloud",
  "config": {
    "apiUrl": "http://your-server/v1",
    "apiKey": "mc_...",
    "userId": "your-user-id",
    "agentId": "your-agent-id",
    "autoCapture": true,
    "autoRecall": true,
    "recallCount": 5
  }
}
```

## Retrieved Memory Context
{context}

## Instructions
- Answer based on the retrieved memories when the question is about stored knowledge
- Answer based on Memcloud documentation when the question is about how to use the platform
- If memories are relevant, reference them naturally
- If no relevant memories are found, say so honestly
- Be concise and helpful
"""


@app.post("/v1/assistant/chat/", response_model=AssistantChatResponse)
async def api_assistant_chat(
    req: AssistantChatRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """AI assistant with smart dual-retrieval: recent memories + semantic search."""
    import httpx
    import re
    from app.config import OPENROUTER_API_KEY, LLM_MODEL
    from app.models import Memory

    start_time = time.time()
    now_utc = datetime.now(timezone.utc)
    msg_lower = req.message.lower()

    # ── Step 1: Detect query type ──
    temporal_match = re.search(
        r'(past|last|recent|today|yesterday|this|tonight|earlier)\s*(hour|hours|minute|minutes|day|days|week|weeks|month|months)?',
        msg_lower
    )
    is_temporal = temporal_match is not None or any(w in msg_lower for w in ['today', 'tonight', 'just now', 'earlier', 'so far'])
    is_summary = any(w in msg_lower for w in ['summarize', 'summary', 'summarise', 'overview', 'recap', 'what happened', 'what did', 'what has'])

    # ── Step 2: Compute time window for temporal queries ──
    time_cutoff = None
    if is_temporal:
        if temporal_match:
            unit = temporal_match.group(2) or 'day'
            if 'hour' in unit:
                time_cutoff = now_utc - timedelta(hours=1)
            elif 'minute' in unit:
                time_cutoff = now_utc - timedelta(minutes=30)
            elif 'week' in unit:
                time_cutoff = now_utc - timedelta(weeks=1)
            elif 'month' in unit:
                time_cutoff = now_utc - timedelta(days=30)
            else:
                time_cutoff = now_utc - timedelta(hours=24)
        if 'today' in msg_lower or 'tonight' in msg_lower:
            time_cutoff = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        elif 'yesterday' in msg_lower:
            yesterday = now_utc - timedelta(days=1)
            time_cutoff = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        if time_cutoff is None:
            time_cutoff = now_utc - timedelta(hours=24)

    # ── Step 3: Dual retrieval ──
    all_memories = {}
    total_searched = 0

    # Strategy A: Direct DB fetch for recent memories (time-filtered)
    if is_temporal and time_cutoff:
        type_weights = {"summary": 1.0, "triple": 0.8, "temporal": 0.7, "profile": 0.6, "raw": 0.3}
        stmt = (
            select(Memory)
            .where(
                Memory.org_id == auth.org_id,
                Memory.user_id == req.user_id,
                Memory.status == "active",
                Memory.created_at >= time_cutoff.replace(tzinfo=None),
            )
            .order_by(Memory.created_at.desc())
            .limit(50)
        )
        if req.agent_id:
            stmt = stmt.where(
                or_(Memory.agent_id == req.agent_id, Memory.agent_id.is_(None))
            )
        result = await db.execute(stmt)
        for row in result.scalars().all():
            all_memories[row.id] = {
                "id": row.id, "content": row.content, "type": row.memory_type,
                "rrf_score": type_weights.get(row.memory_type, 0.5),
                "created_at": str(row.created_at) if row.created_at else "",
                "source": "recent_db",
            }
        total_searched += len(all_memories)

        # Also shared:team
        stmt_shared = (
            select(Memory)
            .where(
                Memory.org_id == auth.org_id,
                Memory.pool_id == "shared:team",
                Memory.status == "active",
                Memory.created_at >= time_cutoff.replace(tzinfo=None),
            )
            .order_by(Memory.created_at.desc())
            .limit(20)
        )
        result_shared = await db.execute(stmt_shared)
        for row in result_shared.scalars().all():
            if row.id not in all_memories:
                all_memories[row.id] = {
                    "id": row.id, "content": row.content, "type": row.memory_type,
                    "rrf_score": type_weights.get(row.memory_type, 0.5) * 0.9,
                    "created_at": str(row.created_at) if row.created_at else "",
                    "source": "recent_shared",
                }

    # Strategy B: Semantic search (always, fewer if we have recent)
    semantic_top_k = 5 if (is_temporal and len(all_memories) > 10) else 15
    try:
        search_result = await search_memories(
            db, org_id=auth.org_id, query=req.message, user_id=req.user_id,
            agent_id=req.agent_id, top_k=semantic_top_k, agentic=False,
        )
        for m in search_result.get("memories", []):
            if m["id"] not in all_memories:
                m["source"] = "semantic"
                all_memories[m["id"]] = m
        total_searched += search_result.get("num_candidates", 0)
    except Exception:
        pass

    try:
        shared_result = await search_memories(
            db, org_id=auth.org_id, query=req.message, user_id=req.user_id,
            agent_id=req.agent_id, pool_id="shared:team", top_k=5, agentic=False,
        )
        for m in shared_result.get("memories", []):
            if m["id"] not in all_memories:
                m["source"] = "semantic_shared"
                all_memories[m["id"]] = m
        total_searched += shared_result.get("num_candidates", 0)
    except Exception:
        pass

    # ── Step 4: Sort and select ──
    if is_temporal:
        summaries = sorted([m for m in all_memories.values() if m.get("type") == "summary"], key=lambda x: x.get("created_at", ""), reverse=True)
        triples = sorted([m for m in all_memories.values() if m.get("type") == "triple"], key=lambda x: x.get("created_at", ""), reverse=True)
        temporals = sorted([m for m in all_memories.values() if m.get("type") == "temporal"], key=lambda x: x.get("created_at", ""), reverse=True)
        others = sorted([m for m in all_memories.values() if m.get("type") not in ("summary", "triple", "temporal")], key=lambda x: x.get("created_at", ""), reverse=True)
        sorted_memories = (summaries[:15] + temporals[:10] + triples[:15] + others[:5])[:40]
    elif is_summary:
        sorted_memories = sorted(all_memories.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)[:30]
    else:
        sorted_memories = sorted(all_memories.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)[:15]

    search_time_ms = round((time.time() - start_time) * 1000, 2)

    # ── Step 5: Build context ──
    context_lines = []
    sources = []
    for m in sorted_memories:
        ts = m.get('created_at', '')[:19] if m.get('created_at') else ''
        mtype = m.get('type', 'unknown').upper()
        score = m.get('rrf_score', 0)
        src = m.get('source', 'unknown')
        context_lines.append(f"[{mtype}] (score: {score:.3f}, created: {ts}, via: {src}) {m['content']}")
        sources.append(AssistantSource(
            id=m["id"], content=m["content"][:500],
            memory_type=m.get("type", "unknown"),
            score=round(score, 4), created_at=m.get("created_at"),
        ))

    context = "\n".join(context_lines) if context_lines else "No relevant memories found."

    # ── Step 6: Build LLM prompt ──
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    system_prompt = MEMCLOUD_SYSTEM_PROMPT.replace("{context}", context)
    time_context = f"Current time: {now_str}"
    if time_cutoff:
        time_context += f"\nTime filter applied: showing memories since {time_cutoff.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    system_prompt = f"{time_context}\nTotal memories in context: {len(sorted_memories)}\n\n{system_prompt}"
    messages = [{"role": "system", "content": system_prompt}]

    if req.history:
        for msg in req.history[-10:]:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": req.message})

    # ── Step 7: Call LLM ──
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"Sorry, I couldn't generate a response. Error: {str(e)}"

    return AssistantChatResponse(
        answer=answer, sources=sources, query_used=req.message,
        search_time_ms=search_time_ms, total_memories_searched=total_searched,
    )


# ========== v1.0.0: Recall & Integrations ==========

@app.post("/v1/recall")
async def api_recall(request: Request, auth: AuthContext = Depends(authenticate), db: AsyncSession = Depends(get_db)):
    """Pre-assembled context injection for agents."""
    body = await request.json()

    user_id = body.get("user_id", "default")
    agent_id = body.get("agent_id")
    query = body.get("query")
    token_budget = body.get("token_budget", 4000)
    fmt = body.get("format", "markdown")
    include_profile = body.get("include_profile", True)
    include_recent = body.get("include_recent", True)
    top_k = body.get("top_k", 15)

    result = await recall_context(
        db, org_id=auth.org_id, user_id=user_id, agent_id=agent_id,
        query=query, token_budget=token_budget, format=fmt,
        include_profile=include_profile, include_recent=include_recent,
        top_k=top_k
    )
    return result


@app.get("/v1/integrations/langchain/config")
async def langchain_config(request: Request, auth: AuthContext = Depends(authenticate), db: AsyncSession = Depends(get_db)):
    """Return LangChain-compatible configuration."""
    return {
        "api_url": str(request.base_url).rstrip("/") + "/v1",
        "recall_endpoint": "/v1/recall",
        "add_endpoint": "/v1/memories/",
        "search_endpoint": "/v1/memories/search/",
        "supported_formats": ["xml", "markdown", "text"],
        "default_token_budget": 4000,
        "version": "1.0.0",
    }


# ========== Sessions ==========

@app.post("/v1/sessions/", response_model=SessionResponse)
async def api_create_session(
    req: SessionCreateRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    expires_at = datetime.utcnow() + timedelta(minutes=req.expires_in_minutes) if req.expires_in_minutes else None
    session = MemorySession(
        org_id=auth.org_id,
        user_id=req.user_id,
        agent_id=req.agent_id,
        name=req.name,
        expires_at=expires_at,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


@app.get("/v1/sessions/")
async def api_list_sessions(
    user_id: str = Query(...),
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(MemorySession).where(
        MemorySession.org_id == auth.org_id,
        MemorySession.user_id == user_id,
    ).order_by(MemorySession.created_at.desc())
    result = await db.execute(stmt)
    sessions = result.scalars().all()
    return [
        {
            "id": s.id, "org_id": s.org_id, "user_id": s.user_id,
            "agent_id": s.agent_id, "name": s.name,
            "expires_at": str(s.expires_at) if s.expires_at else None,
            "created_at": str(s.created_at),
        }
        for s in sessions
    ]


@app.delete("/v1/sessions/{session_id}")
async def api_delete_session(
    session_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(MemorySession).where(
        MemorySession.id == session_id, MemorySession.org_id == auth.org_id,
    ))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found")
    # Archive associated memories instead of deleting
    await db.execute(
        select(Memory).where(Memory.session_id_fk == session_id)
    )
    # Soft delete memories in this session
    from sqlalchemy import update as sa_update
    await db.execute(
        sa_update(Memory).where(Memory.session_id_fk == session_id).values(status="archived")
    )
    await db.delete(session)
    await db.commit()
    return {"status": "ok", "deleted": session_id}


# ========== Pool Access ==========

@app.post("/v1/pools/access/", response_model=PoolAccessResponse)
async def api_grant_pool_access(
    req: PoolAccessCreateRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    access = PoolAccess(
        org_id=auth.org_id,
        pool_id=req.pool_id,
        agent_id=req.agent_id,
        permissions=req.permissions,
    )
    db.add(access)
    await db.commit()
    await db.refresh(access)
    return access


@app.get("/v1/pools/{pool_id}/access/")
async def api_list_pool_access(
    pool_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(PoolAccess).where(
        PoolAccess.org_id == auth.org_id,
        PoolAccess.pool_id == pool_id,
    )
    result = await db.execute(stmt)
    accesses = result.scalars().all()
    return [
        {
            "id": a.id, "org_id": a.org_id, "pool_id": a.pool_id,
            "agent_id": a.agent_id, "permissions": a.permissions,
            "created_at": str(a.created_at),
        }
        for a in accesses
    ]


@app.delete("/v1/pools/access/{access_id}")
async def api_revoke_pool_access(
    access_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(PoolAccess).where(
        PoolAccess.id == access_id, PoolAccess.org_id == auth.org_id,
    ))
    access = result.scalar_one_or_none()
    if not access:
        raise HTTPException(404, "Pool access not found")
    await db.delete(access)
    await db.commit()
    return {"status": "ok", "deleted": access_id}


# ========== Webhooks ==========

@app.post("/v1/webhooks/", response_model=WebhookResponse)
async def api_create_webhook(
    req: WebhookCreateRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    hook = Webhook(
        org_id=auth.org_id,
        url=req.url,
        events=req.events,
        secret=req.secret,
    )
    db.add(hook)
    await db.commit()
    await db.refresh(hook)
    return hook


@app.get("/v1/webhooks/")
async def api_list_webhooks(
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Webhook).where(Webhook.org_id == auth.org_id)
    result = await db.execute(stmt)
    hooks = result.scalars().all()
    return [
        {
            "id": h.id, "org_id": h.org_id, "url": h.url,
            "events": h.events, "is_active": h.is_active,
            "created_at": str(h.created_at),
        }
        for h in hooks
    ]


@app.delete("/v1/webhooks/{webhook_id}")
async def api_delete_webhook(
    webhook_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Webhook).where(
        Webhook.id == webhook_id, Webhook.org_id == auth.org_id,
    ))
    hook = result.scalar_one_or_none()
    if not hook:
        raise HTTPException(404, "Webhook not found")
    await db.delete(hook)
    await db.commit()
    return {"status": "ok", "deleted": webhook_id}


# ========== Schemas ==========

@app.post("/v1/schemas/", response_model=SchemaResponse)
async def api_create_schema(
    req: SchemaCreateRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    schema = MemorySchema(
        org_id=auth.org_id,
        name=req.name,
        fields=[f.model_dump() for f in req.fields],
        description=req.description,
    )
    db.add(schema)
    await db.commit()
    await db.refresh(schema)
    return schema


@app.get("/v1/schemas/")
async def api_list_schemas(
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(MemorySchema).where(MemorySchema.org_id == auth.org_id)
    result = await db.execute(stmt)
    schemas = result.scalars().all()
    return [
        {
            "id": s.id, "org_id": s.org_id, "name": s.name,
            "fields": s.fields, "description": s.description,
            "created_at": str(s.created_at),
        }
        for s in schemas
    ]


@app.delete("/v1/schemas/{schema_id}")
async def api_delete_schema(
    schema_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(MemorySchema).where(
        MemorySchema.id == schema_id, MemorySchema.org_id == auth.org_id,
    ))
    schema = result.scalar_one_or_none()
    if not schema:
        raise HTTPException(404, "Schema not found")
    await db.delete(schema)
    await db.commit()
    return {"status": "ok", "deleted": schema_id}


# ========== Instructions ==========

@app.post("/v1/instructions/", response_model=InstructionResponse)
async def api_create_instruction(
    req: InstructionCreateRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    inst = MemoryInstruction(
        org_id=auth.org_id,
        user_id=req.user_id,
        instruction=req.instruction,
    )
    db.add(inst)
    await db.commit()
    await db.refresh(inst)
    return inst


@app.get("/v1/instructions/")
async def api_list_instructions(
    user_id: str = Query(...),
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(MemoryInstruction).where(
        MemoryInstruction.org_id == auth.org_id,
        MemoryInstruction.user_id == user_id,
    )
    result = await db.execute(stmt)
    instructions = result.scalars().all()
    return [
        {
            "id": i.id, "org_id": i.org_id, "user_id": i.user_id,
            "instruction": i.instruction, "is_active": i.is_active,
            "created_at": str(i.created_at),
        }
        for i in instructions
    ]


@app.delete("/v1/instructions/{instruction_id}")
async def api_delete_instruction(
    instruction_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(MemoryInstruction).where(
        MemoryInstruction.id == instruction_id, MemoryInstruction.org_id == auth.org_id,
    ))
    inst = result.scalar_one_or_none()
    if not inst:
        raise HTTPException(404, "Instruction not found")
    await db.delete(inst)
    await db.commit()
    return {"status": "ok", "deleted": instruction_id}


# ========== Bulk ==========

@app.post("/v1/memories/bulk/import/", response_model=BulkImportResponse)
async def api_bulk_import(
    req: BulkImportRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    results = []
    for item in req.memories:
        try:
            result = await add_memory(
                db, org_id=auth.org_id, text=item.text, user_id=item.user_id,
                agent_id=item.agent_id, pool_id=item.pool_id,
                session_id=item.session_id, metadata=item.metadata,
            )
            results.append({"status": "ok", "memories_created": result["memories_created"]})
        except Exception as e:
            results.append({"status": "error", "error": str(e)})
    return BulkImportResponse(total=len(req.memories), results=results)


@app.post("/v1/memories/bulk/export/", response_model=BulkExportResponse)
async def api_bulk_export(
    req: BulkExportRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    memories = await list_memories(
        db, org_id=auth.org_id, user_id=req.user_id,
        agent_id=req.agent_id, pool_id=req.pool_id,
        memory_type=req.memory_type, limit=500, offset=0,
    )
    exported = [
        {
            "id": m.id, "memory_type": m.memory_type, "content": m.content,
            "structured_data": m.structured_data, "user_id": m.user_id,
            "agent_id": m.agent_id, "pool_id": m.pool_id,
            "confidence": m.confidence, "decay_score": m.decay_score,
            "access_count": m.access_count, "chain_id": m.chain_id,
            "metadata": m.metadata_,
            "created_at": str(m.created_at), "updated_at": str(m.updated_at),
        }
        for m in memories
    ]
    return BulkExportResponse(total=len(exported), memories=exported)


@app.post("/v1/memories/bulk/delete/", response_model=BulkDeleteResponse)
async def api_bulk_delete(
    req: BulkDeleteRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """Bulk soft-delete memories by ID list (max 100 per request)."""
    if len(req.memory_ids) > 100:
        raise HTTPException(400, "Maximum 100 memory IDs per request")
    deleted = 0
    errors = []
    for mid in req.memory_ids:
        try:
            ok = await delete_memory(db, mid, auth.org_id)
            if ok:
                deleted += 1
            else:
                errors.append({"id": mid, "error": "Not found"})
        except Exception as e:
            errors.append({"id": mid, "error": str(e)})
    return BulkDeleteResponse(total=len(req.memory_ids), deleted=deleted, errors=errors)


# ========== Analytics ==========

@app.get("/v1/analytics/", response_model=AnalyticsResponse)
async def api_analytics(
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    org_id = auth.org_id

    growth_q = sa_text("""
        SELECT DATE(created_at) as day, count(*) as count
        FROM memories WHERE org_id = :org AND status IN ('active', 'archived')
        AND created_at >= NOW() - INTERVAL '30 days'
        GROUP BY DATE(created_at) ORDER BY day
    """)
    growth_rows = (await db.execute(growth_q, {"org": org_id})).fetchall()
    growth_by_day = [{"day": str(r[0]), "count": r[1]} for r in growth_rows]

    agent_q = sa_text("SELECT COALESCE(agent_id, 'unknown'), count(*) FROM memories WHERE org_id = :org AND status IN ('active', 'archived') GROUP BY agent_id")
    agent_rows = (await db.execute(agent_q, {"org": org_id})).fetchall()
    agent_activity = {r[0]: r[1] for r in agent_rows}

    type_q = sa_text("SELECT memory_type, count(*) FROM memories WHERE org_id = :org AND status IN ('active', 'archived') GROUP BY memory_type")
    type_rows = (await db.execute(type_q, {"org": org_id})).fetchall()
    type_distribution = {r[0]: r[1] for r in type_rows}

    decay_q = sa_text("""
        SELECT
            SUM(CASE WHEN decay_score >= 0.7 THEN 1 ELSE 0 END) as healthy,
            SUM(CASE WHEN decay_score >= 0.3 AND decay_score < 0.7 THEN 1 ELSE 0 END) as aging,
            SUM(CASE WHEN decay_score < 0.3 THEN 1 ELSE 0 END) as critical
        FROM memories WHERE org_id = :org AND status IN ('active', 'archived')
    """)
    decay_row = (await db.execute(decay_q, {"org": org_id})).fetchone()
    decay_distribution = {
        "healthy": int(decay_row[0] or 0),
        "aging": int(decay_row[1] or 0),
        "critical": int(decay_row[2] or 0),
    }

    return AnalyticsResponse(
        growth_by_day=growth_by_day,
        agent_activity=agent_activity,
        type_distribution=type_distribution,
        decay_distribution=decay_distribution,
    )


# ========== Decay ==========

@app.post("/v1/decay/cleanup/", response_model=DecayCleanupResponse)
async def api_decay_cleanup(
    req: DecayCleanupRequest,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    count = await decay_cleanup(db, auth.org_id, threshold=req.threshold)
    return DecayCleanupResponse(deleted_count=count)


@app.get("/v1/decay/preview/", response_model=DecayPreviewResponse)
async def api_decay_preview(
    limit: int = Query(100, ge=1, le=500),
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    memories = await decay_preview(db, auth.org_id, limit=limit)
    items = [
        DecayPreviewItem(
            id=m.id, content=m.content, memory_type=m.memory_type,
            decay_score=m.decay_score or 0.0, confidence=m.confidence or 1.0,
            access_count=m.access_count or 0,
            last_accessed_at=m.last_accessed_at,
            created_at=m.created_at,
        )
        for m in memories
    ]
    return DecayPreviewResponse(memories=items, total=len(items))


# ========== Audit ==========

@app.get("/v1/audit/", response_model=AuditListResponse)
async def api_audit(
    memory_id: str = Query(None),
    action: str = Query(None),
    actor_id: str = Query(None),
    start_date: str = Query(None),
    end_date: str = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(MemoryAudit).where(MemoryAudit.org_id == auth.org_id)
    if memory_id:
        stmt = stmt.where(MemoryAudit.memory_id == memory_id)
    if action:
        stmt = stmt.where(MemoryAudit.action == action)
    if actor_id:
        stmt = stmt.where(MemoryAudit.actor_id == actor_id)
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date)
            stmt = stmt.where(MemoryAudit.created_at >= start_dt)
        except ValueError:
            raise HTTPException(400, "Invalid start_date format. Use ISO 8601.")
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date)
            stmt = stmt.where(MemoryAudit.created_at <= end_dt)
        except ValueError:
            raise HTTPException(400, "Invalid end_date format. Use ISO 8601.")
    stmt = stmt.order_by(MemoryAudit.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    entries = result.scalars().all()

    count_stmt = select(sa_func.count()).select_from(MemoryAudit).where(MemoryAudit.org_id == auth.org_id)
    total = (await db.execute(count_stmt)).scalar() or 0

    return AuditListResponse(
        entries=[
            AuditEntry(
                id=e.id, org_id=e.org_id, memory_id=e.memory_id,
                action=e.action, actor_id=e.actor_id, actor_type=e.actor_type,
                details=e.details, created_at=e.created_at,
            )
            for e in entries
        ],
        total=total,
    )


# --- Stats ---

@app.get("/v1/stats/")
async def api_stats(
    user_id: str = Query(None),
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """Dashboard summary stats."""
    base = sa_text("SELECT count(*) FROM memories WHERE org_id = :org AND status IN ('active', 'archived')")
    total = (await db.execute(base, {"org": auth.org_id})).scalar() or 0
    
    type_q = sa_text("SELECT memory_type, count(*) FROM memories WHERE org_id = :org AND status IN ('active', 'archived') GROUP BY memory_type")
    type_rows = (await db.execute(type_q, {"org": auth.org_id})).fetchall()
    by_type = {r[0]: r[1] for r in type_rows}
    
    agent_q = sa_text("SELECT COALESCE(agent_id, 'unknown'), count(*) FROM memories WHERE org_id = :org AND status IN ('active', 'archived') GROUP BY agent_id")
    agent_rows = (await db.execute(agent_q, {"org": auth.org_id})).fetchall()
    by_agent = {r[0]: r[1] for r in agent_rows}
    
    pool_q = sa_text("SELECT COALESCE(pool_id, 'private'), count(*) FROM memories WHERE org_id = :org AND status IN ('active', 'archived') GROUP BY pool_id")
    pool_rows = (await db.execute(pool_q, {"org": auth.org_id})).fetchall()
    by_pool = {r[0]: r[1] for r in pool_rows}
    
    add_count = sum(1 for r in request_log if r["type"] == "ADD")
    search_count = sum(1 for r in request_log if r["type"] == "SEARCH")
    total_requests = len(request_log)
    
    return {
        "total_memories": total,
        "by_type": by_type,
        "by_agent": by_agent,
        "by_pool": by_pool,
        "total_requests": total_requests,
        "add_events": add_count,
        "search_events": search_count,
    }


# --- Activity Log ---

@app.get("/v1/activity/")
async def api_activity(
    type: str = Query(None, description="Filter by type: ADD, SEARCH, LIST, DELETE"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    auth: AuthContext = Depends(authenticate),
):
    """Return recent API request activity."""
    logs = list(request_log)
    logs.reverse()
    
    if type:
        logs = [l for l in logs if l["type"] == type.upper()]
    
    return {
        "total": len(logs),
        "entries": logs[offset:offset + limit],
    }


# --- WebSocket ---

app.websocket("/v1/ws")(websocket_endpoint)


# --- Graph (Knowledge Graph) ---

@app.get("/v1/graph/")
async def api_graph(
    user_id: str = Query(None),
    limit: int = Query(200, ge=1, le=5000),
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """Return knowledge graph nodes and edges from the relations table."""
    params = {"org": auth.org_id, "limit": limit}
    where = "WHERE org_id = :org"
    if user_id:
        where += " AND user_id = :uid"
        params["uid"] = user_id

    q = sa_text(f"SELECT source_entity, relation, target_entity FROM relations {where} LIMIT :limit")
    rows = (await db.execute(q, params)).fetchall()

    count_q = sa_text(f"SELECT count(*) FROM relations {where}")
    total_relations = (await db.execute(count_q, {k: v for k, v in params.items() if k != "limit"})).scalar() or 0

    node_connections: dict[str, int] = {}
    edges = []
    for src, rel, tgt in rows:
        node_connections[src] = node_connections.get(src, 0) + 1
        node_connections[tgt] = node_connections.get(tgt, 0) + 1
        edges.append({"source": src, "target": tgt, "relation": rel})

    nodes = [
        {"id": name, "label": name, "type": name.split(":")[0] if ":" in name else "entity", "connections": count}
        for name, count in sorted(node_connections.items(), key=lambda x: -x[1])
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": total_relations,
    }


# ========== Admin: Category Migration ==========

CATEGORY_MIGRATION_MAP = {
    "tasks": "project_context",
    "professional": "project_context",
    "events": "project_context",
    "technology": "technical",
    "education": "technical",
    "personal_details": "user_profile",
    "preferences": "user_profile",
    "health": "user_profile",
    "finance": "trading",
    "entertainment": "store_management",
    "travel": "store_management",
    "relationships": "relationship",
    "infrastructure": "infrastructure",
    "communication": "communication",
    "misc": "agent_ops",
}

NEW_CATEGORIES = [
    "project_context", "infrastructure", "trading", "user_profile",
    "agent_ops", "store_management", "credentials", "technical",
    "communication", "relationship",
]


@app.post("/v1/admin/migrate-categories/")
async def api_migrate_categories(
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """Migrate old Mem0-style categories to new Memcloud categories via SQL mapping."""
    results = {}
    for old_cat, new_cat in CATEGORY_MIGRATION_MAP.items():
        q = sa_text("""
            UPDATE memories
            SET categories = array_replace(categories, :old, :new)
            WHERE org_id = :org AND :old = ANY(categories)
        """)
        r = await db.execute(q, {"old": old_cat, "new": new_cat, "org": auth.org_id})
        results[f"{old_cat} -> {new_cat}"] = r.rowcount

    await db.commit()

    q = sa_text("""
        SELECT json_array_elements_text(categories) as cat, count(*) as cnt
        FROM memories WHERE org_id = :org
        GROUP BY cat ORDER BY cnt DESC
    """)
    rows = (await db.execute(q, {"org": auth.org_id})).fetchall()
    distribution = {r[0]: r[1] for r in rows}

    return {
        "status": "ok",
        "migrations": results,
        "new_distribution": distribution,
        "valid_categories": NEW_CATEGORIES,
    }


# ========== Admin: API Key Management ==========

@app.get("/v1/admin/keys/")
async def api_list_keys(
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """List all API keys for the org."""
    from app.models import ApiKey as ApiKeyModel
    stmt = select(ApiKeyModel).where(ApiKeyModel.org_id == auth.org_id)
    result = await db.execute(stmt)
    keys = result.scalars().all()
    return [
        {
            "id": k.id,
            "key_prefix": k.key_prefix,
            "name": k.name,
            "permissions": k.permissions,
            "rate_limit_per_minute": k.rate_limit_per_minute,
            "rate_limit_per_day": k.rate_limit_per_day,
            "is_active": k.is_active,
            "created_at": str(k.created_at),
        }
        for k in keys
    ]


@app.post("/v1/admin/keys/")
async def api_create_key(
    request: Request,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """Generate a new API key for an agent or service."""
    import hashlib
    import secrets
    from app.models import ApiKey as ApiKeyModel

    body = await request.json()
    name = body.get("name", "New Key")
    agent_id = body.get("agent_id")
    rate_limit_minute = body.get("rate_limit_per_minute", 120)
    rate_limit_day = body.get("rate_limit_per_day", 20000)

    # Generate key
    if agent_id:
        raw_key = f"mc_{agent_id}_{secrets.token_hex(16)}"
    else:
        raw_key = f"mc_{secrets.token_hex(24)}"

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = raw_key[:10]

    api_key = ApiKeyModel(
        org_id=auth.org_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=name,
        permissions={"read": ["*"], "write": ["*"]},
        rate_limit_per_minute=rate_limit_minute,
        rate_limit_per_day=rate_limit_day,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return {
        "id": api_key.id,
        "name": api_key.name,
        "key": raw_key,  # Only shown once!
        "key_prefix": key_prefix,
        "agent_id": agent_id,
        "rate_limit_per_minute": rate_limit_minute,
        "rate_limit_per_day": rate_limit_day,
        "message": "Save this key — it won't be shown again.",
    }


@app.get("/v1/agents/{agent_id}/profile/")
async def api_agent_profile(
    agent_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """Generate a structured memory profile for an agent — like a dynamic MEMORY.md.
    Returns the top memories organized by type, pool, and importance."""
    from app.engine import search_memories, list_memories

    org_id = auth.org_id

    # Get memory counts
    count_q = sa_text("""
        SELECT memory_type, count(*) FROM memories 
        WHERE org_id = :org AND agent_id = :agent AND status IN ('active', 'archived')
        GROUP BY memory_type ORDER BY count DESC
    """)
    type_counts = dict((await db.execute(count_q, {"org": org_id, "agent": agent_id})).fetchall())

    # Get pool distribution
    pool_q = sa_text("""
        SELECT COALESCE(pool_id, 'private') as pool, count(*) FROM memories
        WHERE org_id = :org AND agent_id = :agent AND status IN ('active', 'archived')
        GROUP BY pool_id ORDER BY count DESC
    """)
    pool_counts = dict((await db.execute(pool_q, {"org": org_id, "agent": agent_id})).fetchall())

    # Get top summaries (most useful for context)
    summaries = await list_memories(
        db, org_id, user_id="seiji", agent_id=agent_id,
        memory_type="summary", limit=10,
    )

    # Get top profiles (user/agent info)
    profiles = await list_memories(
        db, org_id, user_id="seiji", agent_id=agent_id,
        memory_type="profile", limit=10,
    )

    # Get high-importance memories (importance 4-5)
    critical_q = sa_text("""
        SELECT content, memory_type, importance, pool_id FROM memories
        WHERE org_id = :org AND agent_id = :agent AND status IN ('active', 'archived')
        AND importance >= 4
        ORDER BY importance DESC, created_at DESC LIMIT 15
    """)
    critical = (await db.execute(critical_q, {"org": org_id, "agent": agent_id})).fetchall()

    # Build structured profile
    profile_sections = []

    if critical:
        profile_sections.append({
            "section": "Critical Knowledge",
            "memories": [{"content": r[0][:300], "type": r[1], "importance": r[2]} for r in critical[:10]],
        })

    if profiles:
        profile_sections.append({
            "section": "User & Agent Info",
            "memories": [{"content": m.content[:300], "type": m.memory_type} for m in profiles[:10]],
        })

    if summaries:
        profile_sections.append({
            "section": "Key Context",
            "memories": [{"content": m.content[:300], "type": m.memory_type} for m in summaries[:10]],
        })

    # Build text version (like MEMORY.md)
    text_parts = [f"# Memory Profile: {agent_id}\n"]
    for section in profile_sections:
        text_parts.append(f"\n## {section['section']}")
        for m in section["memories"]:
            text_parts.append(f"- {m['content']}")

    return {
        "agent_id": agent_id,
        "total_memories": sum(type_counts.values()),
        "by_type": type_counts,
        "by_pool": pool_counts,
        "sections": profile_sections,
        "text": "\n".join(text_parts),
    }


@app.post("/v1/admin/reclassify/")
async def api_reclassify(
    request: Request,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """Reclassify memories that have no categories or default importance.
    Processes in batches to avoid timeout. Call repeatedly until done."""
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    batch_size = min(body.get("batch_size", 50), 100)

    from app.engine import get_extractor

    # Find unclassified memories
    stmt = (
        select(Memory)
        .where(
            Memory.org_id == auth.org_id,
            Memory.status == "active",
            or_(
                # No categories at all
                Memory.categories == None,
                sa_text("categories::text = 'null'"),
                sa_text("categories::text = '[]'"),
                # Has categories but default importance (never scored by LLM)
                sa_text("importance = 5 AND categories IS NOT NULL AND categories::text != 'null' AND categories::text != '[]'"),
            ),
        )
        .order_by(Memory.created_at.desc())
        .limit(batch_size)
    )
    result = await db.execute(stmt)
    memories = result.scalars().all()

    if not memories:
        return {"status": "done", "message": "All memories classified", "processed": 0, "remaining": 0}

    extractor = get_extractor()
    processed = 0
    results_log = []

    for mem in memories:
        try:
            classification = extractor.classify(mem.content[:1500])
            mem.importance = classification["importance"]
            mem.categories = classification["categories"]
            processed += 1
            results_log.append({
                "id": mem.id[:8],
                "importance": classification["importance"],
                "categories": classification["categories"],
                "content": mem.content[:60],
            })
        except Exception as e:
            results_log.append({"id": mem.id[:8], "error": str(e)})

    await db.commit()

    # Count remaining
    count_stmt = select(sa_func.count()).select_from(Memory).where(
        Memory.org_id == auth.org_id,
        Memory.status == "active",
        or_(
            Memory.categories == None,
            sa_text("categories::text = 'null'"),
            sa_text("categories::text = '[]'"),
            Memory.importance == 5,
        ),
    )
    remaining = (await db.execute(count_stmt)).scalar() or 0

    return {
        "status": "processing" if remaining > 0 else "done",
        "processed": processed,
        "remaining": remaining,
        "sample": results_log[:10],
    }


@app.delete("/v1/admin/keys/{key_id}")
async def api_revoke_key(
    key_id: str,
    auth: AuthContext = Depends(authenticate),
    db: AsyncSession = Depends(get_db),
):
    """Revoke an API key."""
    from app.models import ApiKey as ApiKeyModel
    result = await db.execute(select(ApiKeyModel).where(
        ApiKeyModel.id == key_id, ApiKeyModel.org_id == auth.org_id,
    ))
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(404, "Key not found")
    key.is_active = False
    await db.commit()
    return {"status": "ok", "revoked": key_id}
