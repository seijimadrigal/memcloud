/**
 * Hook handlers for auto-recall, auto-capture, and compaction safety.
 * v0.5.0: Hybrid memory orchestrator — MemChip as primary memory.
 * - Enriched auto-recall (summaries + raw, topK=15, structured context)
 * - Async auto-capture (fire-and-forget, non-blocking)
 * - Smarter compaction (extracts decisions, preferences, key facts)
 */

const NOISE_PATTERNS = [
  /^HEARTBEAT_OK$/i,
  /^NO_REPLY$/i,
  /Read HEARTBEAT\.md/i,
  /nothing needs attention/i,
  /no pending tasks/i,
  /Pre-compaction memory flush/i,
  /gateway (connected|disconnected|restart)/i,
  /WhatsApp gateway (connected|disconnected)/i,
  /systemd service/i,
  /tool schema.*not.*pool_id/i,
  /tool surface cannot/i,
  /Exec completed.*code 0/i,
  /Exec failed.*signal SIGKILL/i,
  /^System: \[/i,
];

function isObviousNoise(text) {
  return NOISE_PATTERNS.some(p => p.test(text));
}

export function registerHooks(api, client, config) {

  // ========== ENRICHED AUTO-RECALL ==========
  // Pulls 15 memories, prioritizes summaries + raw over triples,
  // builds structured context block for the agent.
  if (config.autoRecall) {
    api.on("before_agent_start", async (event) => {
      try {
        const lastMsg = event.messages?.filter(m => m.role === "user").pop();
        const query = lastMsg?.content;
        if (!query || typeof query !== "string") return {};

        // Search with higher limit, prefer summaries and raw text
        const results = await client.search(query, 15);
        const allMemories = Array.isArray(results) ? results : results?.results || results?.memories || [];
        if (!allMemories.length) return {};

        // Separate by type — prioritize summaries and raw over triples
        const summaries = [];
        const profiles = [];
        const facts = [];

        for (const m of allMemories) {
          const content = m.memory || m.text || m.content || "";
          const type = m.type || m.memory_type || "";
          if (type === "summary" || type === "raw") {
            summaries.push(content);
          } else if (type === "profile") {
            profiles.push(content);
          } else {
            facts.push(content);
          }
        }

        // Build structured context — summaries first (most context), then profiles, then facts
        const sections = [];

        if (summaries.length) {
          sections.push("Key Context:\n" + summaries.slice(0, 5).map(s => `- ${s}`).join("\n"));
        }
        if (profiles.length) {
          sections.push("User/Agent Info:\n" + profiles.slice(0, 5).map(s => `- ${s}`).join("\n"));
        }
        if (facts.length) {
          sections.push("Related Facts:\n" + facts.slice(0, 5).map(s => `- ${s}`).join("\n"));
        }

        if (!sections.length) return {};

        const context = sections.join("\n\n");

        return {
          prependContext: `<memchip-recall count="${allMemories.length}">\n${context}\n</memchip-recall>`,
        };
      } catch (err) {
        console.error("[memchip] auto-recall error:", err.message);
        return {};
      }
    });
  }

  // ========== ASYNC AUTO-CAPTURE ==========
  // Fire-and-forget — doesn't block the agent's response cycle.
  // Stores important conversation content after each turn.
  if (config.autoCapture) {
    api.on("agent_end", async (event) => {
      // Fire and forget — don't await, don't block
      _captureAsync(client, event).catch(err => {
        console.error("[memchip] auto-capture error:", err.message);
      });
    });
  }

  // ========== SMARTER COMPACTION ==========
  // Before context is trimmed, extract key decisions, preferences, and facts.
  // This is the last chance to save context before it's lost.
  if (config.compactionFlush) {
    api.on("before_compaction", async (event) => {
      try {
        const msgs = event.messages;
        if (!msgs?.length) return;

        // Extract substantive messages
        const substantive = msgs
          .filter(m => m.role === "user" || m.role === "assistant")
          .filter(m => {
            const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
            return content.length > 80 && !isObviousNoise(content);
          });

        if (substantive.length < 2) return;

        // Take a larger window — this is our last chance
        const window = substantive.slice(-20);
        const text = window
          .map(m => {
            const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
            return `${m.role}: ${content.slice(0, 500)}`;
          })
          .join("\n");

        if (text.length > 100) {
          // Store with higher priority — this is at-risk context
          await client.store(text, { source: "pre-compaction" });
        }
      } catch (err) {
        console.error("[memchip] compaction flush error:", err.message);
      }
    });
  }
}

// Async capture helper — runs in background
async function _captureAsync(client, event) {
  const msgs = event.messages;
  if (!msgs?.length) return;

  const relevant = msgs
    .filter(m => m.role === "user" || m.role === "assistant")
    .slice(-4);

  const text = relevant.map(m => {
    const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
    return `${m.role}: ${content}`;
  }).join("\n");

  if (text.length < 30) return;
  if (isObviousNoise(text)) return;

  await client.store(text, { source: "auto-capture" });
}
