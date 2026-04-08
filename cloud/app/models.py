"""SQLAlchemy models for multi-tenant memory storage. v0.4.0"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Float, Boolean, DateTime, Integer, JSON,
    ForeignKey, Index, LargeBinary, func
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from pgvector.sqlalchemy import Vector
from app.database import Base


def new_uuid():
    return str(uuid.uuid4())


class Organization(Base):
    __tablename__ = "organizations"
    id = Column(String, primary_key=True, default=new_uuid)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = Column(String, primary_key=True, default=new_uuid)
    key_hash = Column(String(128), unique=True, nullable=False, index=True)
    key_prefix = Column(String(10), nullable=False)  # mc_xxxx for display
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    name = Column(String(255))
    permissions = Column(JSON, default=dict)  # {"read": [...], "write": [...]}
    rate_limit_per_minute = Column(Integer, default=60)
    rate_limit_per_day = Column(Integer, default=10000)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class MemorySession(Base):
    """Conversation-scoped short-term sessions."""
    __tablename__ = "memory_sessions"
    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    agent_id = Column(String(255), index=True)
    name = Column(String(500))
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class MemorySchema(Base):
    """Custom memory type definitions."""
    __tablename__ = "memory_schemas"
    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    fields = Column(JSON, default=list)  # [{name, type, required}]
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class Memory(Base):
    __tablename__ = "memories"
    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    agent_id = Column(String(255), index=True)
    pool_id = Column(String(255), index=True)  # for shared pools
    memory_type = Column(String(50), nullable=False)  # triple, summary, entity, temporal, profile, raw
    content = Column(Text, nullable=False)
    structured_data = Column(JSON)  # type-specific structured data
    embedding = Column(LargeBinary)  # numpy float32 bytes
    embedding_vec = Column(Vector(768))  # pgvector native column
    session_label = Column(String(255))  # renamed from session_id to avoid FK conflict
    confidence = Column(Float, default=1.0)
    superseded_by = Column(String, ForeignKey("memories.id"))
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    categories = Column(JSON)  # array of category strings e.g. ["finance", "technology"]
    importance = Column(Integer, default=5)  # 0-10 LLM-rated importance score

    # v0.2.0 fields
    decay_score = Column(Float, default=1.0)
    last_accessed_at = Column(DateTime)
    access_count = Column(Integer, default=0)
    chain_id = Column(String, nullable=True)
    schema_id = Column(String, ForeignKey("memory_schemas.id"), nullable=True)
    session_id_fk = Column(String, ForeignKey("memory_sessions.id"), nullable=True)

    # v0.3.0 fields — Collaborative Memory Infrastructure
    status = Column(String(50), default="active")  # active, archived
    scope = Column(String(50), default="private")  # private, task, project, team, global
    conflict_status = Column(String(50), default="active")  # active, superseded, disputed, stale
    supersedes_id = Column(String, ForeignKey("memories.id"), nullable=True)  # what memory this replaces
    version = Column(Integer, default=1)
    source_type = Column(String(100))  # conversation, api_direct, extraction, merge, agent_reasoning
    source_ref = Column(String(500))  # reference to source (session_id, url, memory_id chain)
    derived_from = Column(JSON)  # list of memory_ids this was derived from

    __table_args__ = (
        Index("idx_memories_org_user", "org_id", "user_id"),
        Index("idx_memories_org_agent", "org_id", "agent_id"),
        Index("idx_memories_org_pool", "org_id", "pool_id"),
        Index("idx_memories_type", "memory_type"),
        Index("idx_memories_chain", "chain_id"),
        Index("idx_memories_decay", "decay_score"),
        Index("idx_memories_status", "status"),
        Index("idx_memories_scope", "scope"),
        Index("idx_memories_conflict", "conflict_status"),
    )


class Relation(Base):
    """Knowledge graph edges."""
    __tablename__ = "relations"
    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    user_id = Column(String(255), nullable=False)
    source_entity = Column(String(500), nullable=False, index=True)
    relation = Column(String(500), nullable=False)
    target_entity = Column(String(500), nullable=False, index=True)
    memory_id = Column(String, ForeignKey("memories.id"))
    created_at = Column(DateTime, default=datetime.utcnow)


class PoolAccess(Base):
    """ACL for shared memory pools."""
    __tablename__ = "pool_access"
    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    pool_id = Column(String(255), nullable=False, index=True)
    agent_id = Column(String(255), nullable=False, index=True)
    permissions = Column(JSON, default=dict)  # {read: bool, write: bool, admin: bool}
    created_at = Column(DateTime, default=datetime.utcnow)


class Webhook(Base):
    """Event hook configurations."""
    __tablename__ = "webhooks"
    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    url = Column(String(2048), nullable=False)
    events = Column(JSON, default=list)  # ["memory.added", "memory.updated", "memory.deleted"]
    secret = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class MemoryAudit(Base):
    """Provenance tracking for memory operations."""
    __tablename__ = "memory_audits"
    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    memory_id = Column(String, nullable=False, index=True)
    action = Column(String(50), nullable=False)  # create, update, delete
    actor_id = Column(String(255), index=True)
    actor_type = Column(String(50))  # agent, user, system
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


class MemoryInstruction(Base):
    """Rules for what to store/ignore."""
    __tablename__ = "memory_instructions"
    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    instruction = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ========== v0.3.0: Event Log ==========

class MemoryEvent(Base):
    """Append-only event log — immutable ground truth for all memory mutations."""
    __tablename__ = "memory_events"
    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    memory_id = Column(String, ForeignKey("memories.id"), nullable=False, index=True)
    event_type = Column(String(50), nullable=False)  # created, updated, superseded, disputed, archived, accessed
    actor_id = Column(String(255), index=True)  # who did this
    actor_type = Column(String(50))  # agent, user, system
    source = Column(String(500))  # where did this come from (conversation, api, extraction, merge)
    previous_content = Column(Text)  # content before change (for updates)
    new_content = Column(Text)  # content after change
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_events_org_memory", "org_id", "memory_id"),
        Index("idx_events_type", "event_type"),
        Index("idx_events_actor", "actor_id"),
        Index("idx_events_created", "created_at"),
    )


# ========== v0.3.0: Subscriptions ==========

class MemorySubscription(Base):
    """Event-driven subscription for real-time memory notifications."""
    __tablename__ = "memory_subscriptions"
    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    agent_id = Column(String(255), nullable=False, index=True)
    scope_filter = Column(String(50))  # subscribe to specific scope
    pool_filter = Column(String(255))  # subscribe to specific pool
    category_filter = Column(String(255))  # subscribe to specific category
    event_types = Column(JSON, default=list)  # which events: ["created", "updated", "superseded"]
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ========== v0.4.0: Project Registry ==========

class Project(Base):
    __tablename__ = "projects"
    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False, index=True)
    pool_id = Column(String(255), nullable=False, unique=True)  # auto-generated: "project:<slug>"
    description = Column(Text)
    agents = Column(JSON, default=list)  # ["lyn", "luna"]
    status = Column(String(50), default="active")  # active, archived, completed
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Task(Base):
    __tablename__ = "tasks"
    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=True, index=True)
    name = Column(String(500), nullable=False)
    pool_id = Column(String(255), nullable=False, unique=True)  # auto-generated: "task:<slug>"
    agents = Column(JSON, default=list)  # ["lyn"]
    status = Column(String(50), default="active")  # active, completed, archived
    expires_at = Column(DateTime, nullable=True)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ========== v0.4.0: Agent Auto-Routing ==========

class AgentContext(Base):
    __tablename__ = "agent_contexts"
    id = Column(String, primary_key=True, default=new_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    agent_id = Column(String(255), nullable=False, unique=True, index=True)
    active_project_id = Column(String, ForeignKey("projects.id"), nullable=True)
    active_task_id = Column(String, ForeignKey("tasks.id"), nullable=True)
    default_scope = Column(String(50), default="team")
    default_pool_id = Column(String(255), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AgentProfile(Base):
    __tablename__ = "agent_profiles"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    org_id = Column(String, ForeignKey("organizations.id"), index=True)
    agent_id = Column(String, index=True)
    user_id = Column(String, index=True)
    profile_text = Column(Text)
    profile_data = Column(JSON)
    memory_count_at_update = Column(Integer, default=0)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    __table_args__ = (Index('ix_agent_profiles_org_agent_user', 'org_id', 'agent_id', 'user_id', unique=True),)


class OrgUsage(Base):
    __tablename__ = "org_usage"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    org_id = Column(String, ForeignKey("organizations.id"), index=True)
    month = Column(String, index=True)  # YYYY-MM format
    memory_count = Column(Integer, default=0)
    search_count = Column(Integer, default=0)
    add_count = Column(Integer, default=0)
    api_calls = Column(Integer, default=0)
    __table_args__ = (Index('ix_org_usage_org_month', 'org_id', 'month', unique=True),)
