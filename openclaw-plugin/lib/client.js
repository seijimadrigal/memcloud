/**
 * MemChip Cloud API client — zero dependencies, just fetch.
 */
export class MemChipClient {
  constructor({ apiUrl, apiKey, userId, agentId, orgId }) {
    this.baseUrl = apiUrl.replace(/\/+$/, "");
    this.apiKey = apiKey;
    this.userId = userId;
    this.agentId = agentId;
    this.orgId = orgId;
  }

  async _req(method, path, body) {
    const url = `${this.baseUrl}${path}`;
    const opts = {
      method,
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
    };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(url, opts);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`MemChip ${method} ${path} → ${res.status}: ${text}`);
    }
    const ct = res.headers.get("content-type") || "";
    return ct.includes("json") ? res.json() : res.text();
  }

  _meta(extra) {
    return {
      user_id: this.userId,
      agent_id: this.agentId,
      org_id: this.orgId,
      ...extra,
    };
  }

  async search(query, limit = 5, poolId, scope) {
    const meta = this._meta();
    if (poolId) meta.pool_id = poolId;
    if (scope) meta.scope = scope;
    return this._req("POST", "/memories/search/", {
      query,
      top_k: limit,
      ...meta,
    });
  }

  async store(text, metadata) {
    return this._req("POST", "/memories/", {
      text,
      ...this._meta(metadata),
    });
  }

  async list() {
    const params = new URLSearchParams({ user_id: this.userId });
    return this._req("GET", `/memories/?${params}`);
  }

  async remove(memoryId) {
    return this._req("DELETE", `/memories/${memoryId}/`);
  }

  async profile() {
    return this._req("GET", `/agents/${this.agentId}/profile/`);
  }

  async forget(query) {
    const results = await this.search(query, 20);
    const memories = Array.isArray(results) ? results : results?.results || results?.memories || [];
    let deleted = 0;
    for (const m of memories) {
      const id = m.id || m.memory_id;
      if (id) {
        await this.remove(id).catch(() => {});
        deleted++;
      }
    }
    return { deleted };
  }
}
