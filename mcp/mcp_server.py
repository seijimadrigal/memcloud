#!/usr/bin/env python3
"""
MemChip MCP Server — Model Context Protocol server for Claude Code, Cursor, Windsurf, etc.
Runs as stdio transport. Exposes memory tools via MCP.

Usage:
    python mcp_server.py --api-url http://localhost:8000 --api-key mc_xxx --user-id seiji

Config for Claude Code (~/.claude/mcp.json):
{
  "mcpServers": {
    "memchip": {
      "command": "python3",
      "args": ["/path/to/mcp_server.py", "--api-url", "http://76.13.23.55/v1", "--api-key", "mc_xxx", "--user-id", "seiji"]
    }
  }
}
"""
import argparse
import json
import sys
import httpx

# MCP protocol over stdio
def send_response(response: dict):
    msg = json.dumps(response)
    sys.stdout.write(f"Content-Length: {len(msg)}\r\n\r\n{msg}")
    sys.stdout.flush()


def read_message() -> dict:
    headers = {}
    while True:
        line = sys.stdin.readline()
        if line == "\r\n" or line == "\n":
            break
        if ":" in line:
            key, val = line.split(":", 1)
            headers[key.strip()] = val.strip()
    
    length = int(headers.get("Content-Length", 0))
    if length:
        body = sys.stdin.read(length)
        return json.loads(body)
    return {}


class MemChipMCP:
    def __init__(self, api_url: str, api_key: str, user_id: str, agent_id: str = "mcp"):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.user_id = user_id
        self.agent_id = agent_id
        self.client = httpx.Client(
            base_url=self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=30.0,
        )
    
    def get_tools(self) -> list:
        return [
            {
                "name": "memory_store",
                "description": "Store information in long-term memory. Use for facts, preferences, decisions, project context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to memorize"},
                        "pool_id": {"type": "string", "description": "Memory pool (e.g., 'shared:team', 'private')"},
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "memory_search",
                "description": "Search memories by semantic query. Returns relevant stored memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                        "pool_id": {"type": "string", "description": "Search specific pool"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_answer",
                "description": "Ask a question and get an answer based on stored memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Question to answer from memory"},
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "memory_list",
                "description": "List recent memories, optionally filtered by type or pool.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_type": {"type": "string", "description": "Filter by type: triple, summary, profile, temporal, raw"},
                        "pool_id": {"type": "string", "description": "Filter by pool"},
                        "limit": {"type": "integer", "description": "Max results (default 20)", "default": 20},
                    },
                },
            },
            {
                "name": "memory_delete",
                "description": "Delete a memory by ID.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory ID to delete"},
                    },
                    "required": ["memory_id"],
                },
            },
        ]
    
    def call_tool(self, name: str, args: dict) -> dict:
        try:
            if name == "memory_store":
                resp = self.client.post("/v1/memories/", json={
                    "text": args["text"],
                    "user_id": self.user_id,
                    "agent_id": self.agent_id,
                    "pool_id": args.get("pool_id"),
                })
                resp.raise_for_status()
                data = resp.json()
                return {"content": [{"type": "text", "text": f"Stored {data.get('memories_created', 0)} memories."}]}
            
            elif name == "memory_search":
                resp = self.client.post("/v1/memories/search/", json={
                    "query": args["query"],
                    "user_id": self.user_id,
                    "agent_id": self.agent_id,
                    "top_k": args.get("top_k", 5),
                    "pool_id": args.get("pool_id"),
                })
                resp.raise_for_status()
                data = resp.json()
                memories = data.get("memories", [])
                if not memories:
                    return {"content": [{"type": "text", "text": "No memories found."}]}
                lines = []
                for m in memories[:10]:
                    lines.append(f"- [{m.get('type', 'unknown')}] {m.get('content', '')}")
                return {"content": [{"type": "text", "text": "\n".join(lines)}]}
            
            elif name == "memory_answer":
                resp = self.client.post("/v1/memories/answer/", json={
                    "question": args["question"],
                    "user_id": self.user_id,
                    "agent_id": self.agent_id,
                })
                resp.raise_for_status()
                data = resp.json()
                return {"content": [{"type": "text", "text": data.get("answer", "No answer found.")}]}
            
            elif name == "memory_list":
                params = {"user_id": self.user_id, "limit": str(args.get("limit", 20))}
                if args.get("memory_type"):
                    params["memory_type"] = args["memory_type"]
                if args.get("pool_id"):
                    params["pool_id"] = args["pool_id"]
                resp = self.client.get("/v1/memories/", params=params)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    return {"content": [{"type": "text", "text": "No memories found."}]}
                lines = []
                for m in data[:20]:
                    lines.append(f"- [{m.get('memory_type', '?')}] {m.get('content', '')[:100]}")
                return {"content": [{"type": "text", "text": "\n".join(lines)}]}
            
            elif name == "memory_delete":
                resp = self.client.delete(f"/v1/memories/{args['memory_id']}")
                resp.raise_for_status()
                return {"content": [{"type": "text", "text": f"Deleted memory {args['memory_id']}"}]}
            
            else:
                return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}
        
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True}
    
    def run(self):
        """Main MCP server loop over stdio."""
        while True:
            try:
                msg = read_message()
                if not msg:
                    continue
                
                method = msg.get("method", "")
                msg_id = msg.get("id")
                params = msg.get("params", {})
                
                if method == "initialize":
                    send_response({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "memchip", "version": "0.2.0"},
                        },
                    })
                
                elif method == "notifications/initialized":
                    pass  # No response needed
                
                elif method == "tools/list":
                    send_response({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {"tools": self.get_tools()},
                    })
                
                elif method == "tools/call":
                    tool_name = params.get("name", "")
                    tool_args = params.get("arguments", {})
                    result = self.call_tool(tool_name, tool_args)
                    send_response({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": result,
                    })
                
                else:
                    send_response({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32601, "message": f"Method not found: {method}"},
                    })
            
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                sys.stderr.write(f"MCP Error: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemChip MCP Server")
    parser.add_argument("--api-url", required=True, help="MemChip API URL")
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--user-id", required=True, help="User ID")
    parser.add_argument("--agent-id", default="mcp", help="Agent ID")
    args = parser.parse_args()
    
    server = MemChipMCP(args.api_url, args.api_key, args.user_id, args.agent_id)
    server.run()
