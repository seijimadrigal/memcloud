"""
MemChip Python Client

Usage:
    from memchip import MemChipClient
    
    mc = MemChipClient(api_key="mc_xxx", user_id="seiji")
    
    # Store a memory
    mc.add("User prefers dark theme and Inter font")
    
    # Search memories
    results = mc.search("What theme does the user prefer?")
    
    # Get an answer
    answer = mc.answer("What font does the user like?")
    
    # List memories
    memories = mc.list(memory_type="triple", limit=10)
    
    # Delete
    mc.delete(memory_id="abc-123")
    
    # Bulk import
    mc.bulk_import([
        {"content": "fact 1", "memory_type": "raw"},
        {"content": "fact 2", "memory_type": "triple"},
    ])
    
    # Bulk export
    data = mc.bulk_export(memory_type="triple")
"""
import httpx
from typing import Optional, List, Dict, Any


class MemChipClient:
    """MemChip API client."""
    
    def __init__(
        self,
        api_key: str,
        api_url: str = "http://76.13.23.55/v1",
        user_id: str = "default",
        agent_id: Optional[str] = None,
        pool_id: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_url = api_url.rstrip("/")
        self.user_id = user_id
        self.agent_id = agent_id
        self.pool_id = pool_id
        self._client = httpx.Client(
            base_url=self.api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
    
    def add(
        self,
        text: str,
        agent_id: Optional[str] = None,
        pool_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Add a memory (triggers extraction pipeline)."""
        resp = self._client.post("/memories/", json={
            "text": text,
            "user_id": self.user_id,
            "agent_id": agent_id or self.agent_id,
            "pool_id": pool_id or self.pool_id,
            "session_id": session_id,
            "metadata": metadata,
        })
        resp.raise_for_status()
        return resp.json()
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        pool_id: Optional[str] = None,
        search_scope: Optional[List[str]] = None,
        agentic: bool = True,
    ) -> dict:
        """Search memories by semantic query."""
        resp = self._client.post("/memories/search/", json={
            "query": query,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "pool_id": pool_id or self.pool_id,
            "search_scope": search_scope,
            "top_k": top_k,
            "agentic": agentic,
        })
        resp.raise_for_status()
        return resp.json()
    
    def answer(self, question: str, agentic: bool = True) -> str:
        """Ask a question and get an answer from memory."""
        resp = self._client.post("/memories/answer/", json={
            "question": question,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "agentic": agentic,
        })
        resp.raise_for_status()
        return resp.json().get("answer", "")

    def recall(
        self,
        query: Optional[str] = None,
        agent_id: Optional[str] = None,
        token_budget: int = 4000,
        format: str = "markdown",
        include_profile: bool = True,
        include_recent: bool = True,
        top_k: int = 15,
    ) -> dict:
        """Get pre-assembled context for agent injection."""
        payload = {
            "user_id": self.user_id,
            "agent_id": agent_id or self.agent_id,
            "query": query,
            "token_budget": token_budget,
            "format": format,
            "include_profile": include_profile,
            "include_recent": include_recent,
            "top_k": top_k,
        }
        resp = self._client.post("/recall", json=payload)
        resp.raise_for_status()
        return resp.json()

    def list(
        self,
        memory_type: Optional[str] = None,
        pool_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[dict]:
        """List memories."""
        params = {"user_id": self.user_id, "limit": str(limit), "offset": str(offset)}
        if memory_type:
            params["memory_type"] = memory_type
        if pool_id or self.pool_id:
            params["pool_id"] = pool_id or self.pool_id
        resp = self._client.get("/memories/", params=params)
        resp.raise_for_status()
        return resp.json()
    
    def get(self, memory_id: str) -> dict:
        """Get a single memory by ID."""
        resp = self._client.get(f"/memories/{memory_id}")
        resp.raise_for_status()
        return resp.json()
    
    def update(self, memory_id: str, content: Optional[str] = None, metadata: Optional[dict] = None) -> dict:
        """Update a memory."""
        body = {}
        if content:
            body["content"] = content
        if metadata:
            body["metadata"] = metadata
        resp = self._client.put(f"/memories/{memory_id}", json=body)
        resp.raise_for_status()
        return resp.json()
    
    def delete(self, memory_id: str) -> dict:
        """Delete a memory."""
        resp = self._client.delete(f"/memories/{memory_id}")
        resp.raise_for_status()
        return resp.json()
    
    def bulk_import(self, memories: List[dict]) -> dict:
        """Import multiple memories at once."""
        resp = self._client.post("/memories/bulk/import/", json={
            "user_id": self.user_id,
            "memories": memories,
        })
        resp.raise_for_status()
        return resp.json()
    
    def bulk_export(
        self,
        memory_type: Optional[str] = None,
        pool_id: Optional[str] = None,
        limit: int = 1000,
    ) -> dict:
        """Export memories as JSON."""
        resp = self._client.post("/memories/bulk/export/", json={
            "user_id": self.user_id,
            "memory_type": memory_type,
            "pool_id": pool_id or self.pool_id,
            "limit": limit,
        })
        resp.raise_for_status()
        return resp.json()
    
    def stats(self) -> dict:
        """Get dashboard stats."""
        resp = self._client.get("/stats/")
        resp.raise_for_status()
        return resp.json()
    
    def analytics(self, days: int = 30) -> dict:
        """Get memory analytics."""
        resp = self._client.get("/analytics/", params={"days": str(days)})
        resp.raise_for_status()
        return resp.json()
    
    # --- Sessions ---
    def create_session(self, name: str = None, expires_in_minutes: int = None) -> dict:
        resp = self._client.post("/sessions/", json={
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "name": name,
            "expires_in_minutes": expires_in_minutes,
        })
        resp.raise_for_status()
        return resp.json()
    
    # --- Webhooks ---
    def create_webhook(self, url: str, events: List[str], pool_filter: str = None) -> dict:
        resp = self._client.post("/webhooks/", json={
            "url": url,
            "events": events,
            "pool_filter": pool_filter,
        })
        resp.raise_for_status()
        return resp.json()
    
    def list_webhooks(self) -> List[dict]:
        resp = self._client.get("/webhooks/")
        resp.raise_for_status()
        return resp.json()
