"""
Memcloud LangChain Memory Integration

Usage:
    from memchip.langchain import MemChipMemory
    from langchain.chains import ConversationChain

    memory = MemChipMemory(
        api_key="mc_xxx",
        api_url="https://api.memcloud.net/v1",
        user_id="seiji",
        agent_id="lyn"
    )

    chain = ConversationChain(memory=memory, llm=llm)
"""
from typing import Any, Dict, List, Optional


class MemChipMemory:
    """LangChain-compatible memory backed by Memcloud API.

    Implements the interface expected by LangChain's memory classes:
    - load_memory_variables(): returns context from Memcloud recall
    - save_context(): stores conversation turns to Memcloud
    - clear(): clears memory (no-op for Memcloud)
    """

    memory_key: str = "history"
    return_messages: bool = False

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.memcloud.net/v1",
        user_id: str = "default",
        agent_id: Optional[str] = None,
        token_budget: int = 4000,
        format: str = "text",
        auto_capture: bool = True,
    ):
        from memchip.client import MemChipClient
        self.client = MemChipClient(
            api_key=api_key, api_url=api_url,
            user_id=user_id, agent_id=agent_id
        )
        self.token_budget = token_budget
        self.format = format
        self.auto_capture = auto_capture
        self._last_query = None

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any] = None) -> Dict[str, str]:
        """Load context from Memcloud recall endpoint."""
        query = None
        if inputs:
            query = inputs.get("input") or inputs.get("question") or inputs.get("query")
        self._last_query = query

        try:
            result = self.client.recall(
                query=query,
                token_budget=self.token_budget,
                format=self.format,
            )
            return {self.memory_key: result.get("context", "")}
        except Exception:
            return {self.memory_key: ""}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Store conversation turn to Memcloud."""
        if not self.auto_capture:
            return

        input_text = inputs.get("input") or inputs.get("question") or str(inputs)
        output_text = outputs.get("output") or outputs.get("response") or str(outputs)

        text = f"User: {input_text}\nAssistant: {output_text}"
        try:
            self.client.add(text)
        except Exception:
            pass

    def clear(self) -> None:
        """No-op — Memcloud manages memory lifecycle via decay."""
        pass
