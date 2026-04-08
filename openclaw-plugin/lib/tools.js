/**
 * Register memory tools with OpenClaw — v0.3.0 (collaborative memory).
 * NOTE: OpenClaw calls execute(callId, params, context, extra)
 *       params is the 2nd argument, NOT the 1st!
 */
export function registerTools(api, client) {
  api.registerTool({
    name: "memory_search",
    description: "Search memories by semantic query. Returns relevant stored memories. Use pool_id='shared:team' to search shared memories. Use scope to filter by visibility level.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        limit: { type: "number", description: "Max results (default 5)" },
        pool_id: { type: "string", description: "Search specific pool (e.g. 'shared:team')" },
        scope: { type: "string", enum: ["private", "task", "project", "team", "global"], description: "Filter by memory scope" },
      },
      required: ["query"],
    },
    async execute(_callId, params) {
      const { query, limit, pool_id, scope } = params || {};
      const results = await client.search(query, limit || 5, pool_id, scope);
      const memories = Array.isArray(results) ? results : results?.results || results?.memories || [];
      if (!memories.length) return "No memories found.";
      return memories
        .map((m, i) => {
          const content = m.memory || m.text || m.content || JSON.stringify(m);
          const meta = [];
          if (m.scope) meta.push(`scope:${m.scope}`);
          if (m.source_type) meta.push(`source:${m.source_type}`);
          if (m.conflict_status && m.conflict_status !== "active") meta.push(`conflict:${m.conflict_status}`);
          const suffix = meta.length ? ` [${meta.join(", ")}]` : "";
          return `${i+1}. ${content}${suffix}`;
        })
        .join("\n");
    },
  });

  api.registerTool({
    name: "memory_store",
    description: "Store important information in long-term memory. Use pool_id='shared:team' for shared memories visible to all agents. Use scope to set visibility: private (only you), task (same task), project (same pool), team (all agents), global (everyone).",
    parameters: {
      type: "object",
      properties: {
        text: { type: "string", description: "Information to remember" },
        pool_id: { type: "string", description: "Memory pool (e.g. 'shared:team' for shared, omit for private)" },
        scope: { type: "string", enum: ["private", "task", "project", "team", "global"], description: "Visibility scope (default: private, or team if pool_id is shared)" },
        source_type: { type: "string", description: "Source of this memory (e.g. 'agent_reasoning', 'conversation', 'research')" },
      },
      required: ["text"],
    },
    async execute(_callId, params) {
      const { text, pool_id, scope, source_type } = params || {};
      if (!text) return "Error: no text provided";
      const meta = {};
      if (pool_id) meta.pool_id = pool_id;
      if (scope) meta.scope = scope;
      if (source_type) meta.source_type = source_type;
      // Default scope to team if storing in shared pool
      if (pool_id && pool_id.startsWith("shared:") && !scope) meta.scope = "team";
      await client.store(text, meta);
      const parts = [];
      if (pool_id) parts.push(`pool: ${pool_id}`);
      if (meta.scope) parts.push(`scope: ${meta.scope}`);
      return parts.length ? `Memory stored (${parts.join(", ")})` : "Memory stored.";
    },
  });

  api.registerTool({
    name: "memory_forget",
    description: "Archive memories matching a query (soft delete — can be recovered).",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search to find memories to archive" },
      },
    },
    async execute(_callId, params) {
      const { query } = params || {};
      if (!query) return "Please provide a query.";
      const { deleted } = await client.forget(query);
      return "Archived " + deleted + " memory/memories.";
    },
  });

  api.registerTool({
    name: "memory_list",
    description: "List all stored memories.",
    parameters: { type: "object", properties: {} },
    async execute() {
      const results = await client.list();
      const memories = Array.isArray(results) ? results : results?.results || results?.memories || [];
      if (!memories.length) return "No memories stored.";
      return memories
        .map((m, i) => {
          const content = m.memory || m.text || m.content || JSON.stringify(m);
          const meta = [];
          if (m.pool_id) meta.push(m.pool_id);
          if (m.scope) meta.push(m.scope);
          const suffix = meta.length ? ` [${meta.join(", ")}]` : "";
          return `${i+1}. ${content}${suffix}`;
        })
        .join("\n");
    },
  });
}
