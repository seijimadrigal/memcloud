"""Pydantic schemas for API request/response. v0.4.0"""
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


# ========== Enums ==========

class ScopeEnum(str, Enum):
    private = "private"
    task = "task"
    project = "project"
    team = "team"
    global_ = "global"


class ConflictResolution(str, Enum):
    accept = "accept"
    reject = "reject"
    merge = "merge"


# --- Requests ---

class MemoryAddRequest(BaseModel):
    text: str = Field(..., description="Text or conversation to memorize")
    user_id: str = Field(..., description="User ID")
    agent_id: Optional[str] = None
    pool_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    schema_id: Optional[str] = None
    session_id_fk: Optional[str] = None
    # v0.3.0
    scope: Optional[str] = Field(default="private", description="Memory scope: private, task, project, team, global")
    source_type: Optional[str] = Field(default=None, description="Source type: conversation, api_direct, extraction, merge, agent_reasoning")
    source_ref: Optional[str] = Field(default=None, description="Reference to source (session_id, url, etc)")


class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    user_id: str = Field(..., description="User ID")
    agent_id: Optional[str] = None
    pool_id: Optional[str] = None
    search_scope: Optional[List[str]] = None  # ["agent:luna", "shared:team"]
    top_k: int = Field(default=10, ge=1, le=100)
    agentic: bool = True


class MemoryUpdateRequest(BaseModel):
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryAnswerRequest(BaseModel):
    question: str
    user_id: str
    agent_id: Optional[str] = None
    agentic: bool = True


class MemoryListParams(BaseModel):
    user_id: str
    agent_id: Optional[str] = None
    pool_id: Optional[str] = None
    memory_type: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


# --- Session Schemas ---

class SessionCreateRequest(BaseModel):
    name: str
    user_id: str
    agent_id: Optional[str] = None
    expires_in_minutes: Optional[int] = Field(default=60, ge=1, le=43200)


class SessionResponse(BaseModel):
    id: str
    org_id: str
    user_id: str
    agent_id: Optional[str] = None
    name: str
    expires_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- Pool Access Schemas ---

class PoolAccessCreateRequest(BaseModel):
    pool_id: str
    agent_id: str
    permissions: Dict[str, bool] = Field(default={"read": True, "write": False, "admin": False})


class PoolAccessResponse(BaseModel):
    id: str
    org_id: str
    pool_id: str
    agent_id: str
    permissions: Dict[str, bool]
    created_at: datetime

    class Config:
        from_attributes = True


# --- Webhook Schemas ---

class WebhookCreateRequest(BaseModel):
    url: str
    events: List[str] = Field(default=["memory.added", "memory.updated", "memory.deleted"])
    secret: Optional[str] = None


class WebhookResponse(BaseModel):
    id: str
    org_id: str
    url: str
    events: List[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# --- Schema Schemas ---

class SchemaFieldDef(BaseModel):
    name: str
    type: str
    required: bool = False


class SchemaCreateRequest(BaseModel):
    name: str
    fields: List[SchemaFieldDef]
    description: Optional[str] = None


class SchemaResponse(BaseModel):
    id: str
    org_id: str
    name: str
    fields: List[Dict[str, Any]]
    description: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- Instruction Schemas ---

class InstructionCreateRequest(BaseModel):
    user_id: str
    instruction: str


class InstructionResponse(BaseModel):
    id: str
    org_id: str
    user_id: str
    instruction: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# --- Bulk Schemas ---

class BulkMemoryItem(BaseModel):
    text: str
    user_id: str
    agent_id: Optional[str] = None
    pool_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BulkImportRequest(BaseModel):
    memories: List[BulkMemoryItem]


class BulkImportResponse(BaseModel):
    status: str = "ok"
    total: int
    results: List[Dict[str, Any]]


class BulkExportRequest(BaseModel):
    user_id: str
    agent_id: Optional[str] = None
    pool_id: Optional[str] = None
    memory_type: Optional[str] = None


class BulkExportResponse(BaseModel):
    status: str = "ok"
    total: int
    memories: List[Dict[str, Any]]


class BulkDeleteRequest(BaseModel):
    memory_ids: List[str]

class BulkDeleteResponse(BaseModel):
    status: str = "ok"
    total: int
    deleted: int
    errors: List[Dict[str, Any]] = []


# --- Decay Schemas ---

class DecayCleanupRequest(BaseModel):
    threshold: float = Field(default=0.1, ge=0.0, le=1.0)


class DecayCleanupResponse(BaseModel):
    status: str = "ok"
    deleted_count: int


class DecayPreviewItem(BaseModel):
    id: str
    content: str
    memory_type: str
    decay_score: float
    confidence: float
    access_count: int
    last_accessed_at: Optional[datetime] = None
    created_at: datetime


class DecayPreviewResponse(BaseModel):
    memories: List[DecayPreviewItem]
    total: int


# --- Audit Schemas ---

class AuditEntry(BaseModel):
    id: str
    org_id: str
    memory_id: str
    action: str
    actor_id: Optional[str] = None
    actor_type: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AuditListResponse(BaseModel):
    entries: List[AuditEntry]
    total: int


# --- Analytics Schemas ---

class AnalyticsResponse(BaseModel):
    growth_by_day: List[Dict[str, Any]]
    agent_activity: Dict[str, int]
    type_distribution: Dict[str, int]
    decay_distribution: Dict[str, int]  # buckets: healthy/aging/critical


# ========== v0.3.0: Event Log Schemas ==========

class MemoryEventResponse(BaseModel):
    id: str
    org_id: str
    memory_id: str
    event_type: str
    actor_id: Optional[str] = None
    actor_type: Optional[str] = None
    source: Optional[str] = None
    previous_content: Optional[str] = None
    new_content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class EventListResponse(BaseModel):
    events: List[MemoryEventResponse]
    total: int


# ========== v0.3.0: Conflict Resolution ==========

class ConflictResolveRequest(BaseModel):
    resolution: ConflictResolution = Field(..., description="accept, reject, or merge")
    merged_content: Optional[str] = Field(default=None, description="Required when resolution is 'merge'")


# ========== v0.3.0: Provenance ==========

class ProvenanceInfo(BaseModel):
    source_type: Optional[str] = None
    source_ref: Optional[str] = None
    derived_from: Optional[List[str]] = None


# ========== v0.3.0: Subscription Schemas ==========

class SubscriptionCreateRequest(BaseModel):
    agent_id: str
    scope_filter: Optional[str] = None
    pool_filter: Optional[str] = None
    category_filter: Optional[str] = None
    event_types: List[str] = Field(default=["created", "updated", "superseded"])


class SubscriptionResponse(BaseModel):
    id: str
    org_id: str
    agent_id: str
    scope_filter: Optional[str] = None
    pool_filter: Optional[str] = None
    category_filter: Optional[str] = None
    event_types: List[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# --- Responses ---

class MemoryResponse(BaseModel):
    id: str
    memory_type: str
    content: str
    structured_data: Optional[Dict[str, Any]] = None
    user_id: str
    agent_id: Optional[str] = None
    pool_id: Optional[str] = None
    confidence: Optional[float] = None
    decay_score: Optional[float] = None
    access_count: Optional[int] = None
    chain_id: Optional[str] = None
    categories: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime
    # v0.3.0
    status: Optional[str] = "active"
    scope: Optional[str] = "private"
    conflict_status: Optional[str] = "active"
    version: Optional[int] = 1
    source_type: Optional[str] = None
    source_ref: Optional[str] = None
    derived_from: Optional[List[str]] = None
    supersedes_id: Optional[str] = None

    class Config:
        from_attributes = True


class MemoryAddResponse(BaseModel):
    status: str = "ok"
    memories_created: int
    counts: Dict[str, int]
    memory_ids: List[str]
    dedup_stats: Optional[Dict[str, int]] = None


class MemorySearchResponse(BaseModel):
    memories: List[Dict[str, Any]]
    context: str
    num_candidates: int
    num_returned: int


class MemoryAnswerResponse(BaseModel):
    answer: str
    memories_used: int
    context: str


class HealthResponse(BaseModel):
    status: str
    version: str
    postgres: bool
    redis: bool
    embedding_model: str


# ========== v0.4.0: Project Registry Schemas ==========

class ProjectCreateRequest(BaseModel):
    name: str
    slug: Optional[str] = None  # Auto-generated from name if not provided
    description: Optional[str] = None
    agents: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class ProjectUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    agents: Optional[List[str]] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ProjectResponse(BaseModel):
    id: str
    org_id: str
    name: str
    slug: str
    pool_id: str
    description: Optional[str] = None
    agents: List[str]
    status: str
    metadata: Optional[Dict[str, Any]] = None
    memory_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectDetailResponse(ProjectResponse):
    recent_memories: List[Dict[str, Any]] = Field(default_factory=list)


# ========== v0.4.0: Agent Context Schemas ==========

class AgentContextUpdateRequest(BaseModel):
    active_project_id: Optional[str] = None
    active_task_id: Optional[str] = None
    default_scope: Optional[str] = None
    default_pool_id: Optional[str] = None


# ========== Assistant Chat Schemas ==========

class AssistantChatRequest(BaseModel):
    message: str
    user_id: str
    agent_id: Optional[str] = None
    history: Optional[List[dict]] = None  # [{role: "user"/"assistant", content: "..."}]


class AssistantSource(BaseModel):
    id: str
    content: str
    memory_type: str
    score: float
    created_at: Optional[str] = None


class AssistantChatResponse(BaseModel):
    answer: str
    sources: List[AssistantSource]
    query_used: str
    search_time_ms: float
    total_memories_searched: int


class AgentContextResponse(BaseModel):
    id: str
    org_id: str
    agent_id: str
    active_project_id: Optional[str] = None
    active_task_id: Optional[str] = None
    default_scope: Optional[str] = "team"
    default_pool_id: Optional[str] = None
    updated_at: datetime

    class Config:
        from_attributes = True


# ========== v0.5.0: Recall Schemas ==========

class RecallRequest(BaseModel):
    user_id: str
    agent_id: Optional[str] = None
    query: Optional[str] = None
    token_budget: int = 4000
    format: str = "markdown"  # xml, markdown, text
    include_profile: bool = True
    include_recent: bool = True
    top_k: int = 15


class RecallResponse(BaseModel):
    context: str
    format: str
    token_count: int
    sections: dict
    latency_ms: float
