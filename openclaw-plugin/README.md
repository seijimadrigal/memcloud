# openclaw-memcloud

Memcloud memory plugin for OpenClaw. Replaces Cognee/Mem0 as the memory backend.

## Install

```bash
cd /path/to/openclaw
npm install /path/to/openclaw-memcloud
# or link it:
npm link /path/to/openclaw-memcloud
```

## Configure

In your OpenClaw config (e.g. `~/.openclaw/config.yaml`):

```yaml
plugins:
  slots:
    memory: openclaw-memcloud
  entries:
    openclaw-memcloud:
      config:
        apiUrl: "https://api.memcloud.dev/v1"
        apiKey: "mc_d798cc892328f4e598803eac5f675cb1ad301fc16a78fd6e"
        userId: "seiji"
        agentId: "lyn"
        autoRecall: true
        autoCapture: true
        topK: 5
        compactionFlush: true
```

## Tools Provided

| Tool | Description |
|------|-------------|
| `memory_search` | Search memories by query |
| `memory_store` | Store new memory |
| `memory_forget` | Delete memories matching query |
| `memory_list` | List all memories |

## Hooks

- **before_agent_start** — Auto-recalls relevant memories and prepends to context
- **agent_end** — Auto-captures conversation to Memcloud
- **before_compaction** — Flushes conversation to Memcloud before context compression
